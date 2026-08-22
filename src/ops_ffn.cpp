#include "ops_ffn.h"

#include "ops_basic.h"

#include <ggml.h>

#include <cassert>

namespace game_ggml::internal::ops {

// ---------------------------------------------------------------------------
// GLUFFN
// ---------------------------------------------------------------------------

ggml_tensor * glu_ffn(ggml_context * ctx,
                      ggml_tensor * x,
                      ggml_tensor * w_ln1, ggml_tensor * b_ln1,
                      ggml_tensor * w_ln2, ggml_tensor * b_ln2) {
    // [D, T, B] -> [2L, T, B]
    ggml_tensor * h = linear(ctx, x, w_ln1, b_ln1);
    assert(h->ne[0] % 2 == 0);
    const int64_t L = h->ne[0] / 2;

    // Chunk along innermost dim via strided views.  Both halves keep the same
    // outer strides so downstream elementwise ops work without a copy.
    const size_t esize = ggml_element_size(h);
    ggml_tensor * x1 = ggml_view_4d(ctx, h,
        L, h->ne[1], h->ne[2], h->ne[3],
        h->nb[1], h->nb[2], h->nb[3],
        /*offset=*/0);
    ggml_tensor * x2 = ggml_view_4d(ctx, h,
        L, h->ne[1], h->ne[2], h->ne[3],
        h->nb[1], h->nb[2], h->nb[3],
        /*offset=*/L * esize);

    // gelu on x1 only, then elementwise mul with x2.  (ggml_gelu expects a
    // contiguous tensor; cont-copy the strided view first.)
    ggml_tensor * x1c = ggml_cont(ctx, x1);
    ggml_tensor * x2c = ggml_cont(ctx, x2);
    ggml_tensor * y   = ggml_mul(ctx, ggml_gelu(ctx, x1c), x2c);

    // [L, T, B] -> [D, T, B]
    return linear(ctx, y, w_ln2, b_ln2);
}

// Split-halves GLU: the ln1 weight/bias were split into two contiguous
// [in, L] / [L] pieces at load time, so the two mul_mats output contiguous
// [L, T, B] activations and the strided-view + 2x cont of the monolithic
// path disappear.
ggml_tensor * glu_ffn_split(ggml_context * ctx,
                            ggml_tensor * x,
                            ggml_tensor * w_ln1_a, ggml_tensor * b_ln1_a,
                            ggml_tensor * w_ln1_b, ggml_tensor * b_ln1_b,
                            ggml_tensor * w_ln2, ggml_tensor * b_ln2) {
    ggml_tensor * x1 = linear(ctx, x, w_ln1_a, b_ln1_a);   // [L, T, B] contiguous
    ggml_tensor * x2 = linear(ctx, x, w_ln1_b, b_ln1_b);   // [L, T, B] contiguous
    ggml_tensor * y  = ggml_mul(ctx, ggml_gelu(ctx, x1), x2);
    return linear(ctx, y, w_ln2, b_ln2);
}

// ---------------------------------------------------------------------------
// CgMLP
// ---------------------------------------------------------------------------

namespace {
// Add a 1D per-channel bias to a [T, C, B] conv output.  The bias has ne0=C
// in PyTorch but we want it to broadcast over ne[0]=T and ne[2]=B.  Reshape
// via a zero-copy view with ne = (1, C, 1, 1).
ggml_tensor * add_conv_bias(ggml_context * ctx, ggml_tensor * y, ggml_tensor * b) {
    if (!b) return y;
    const size_t esize = ggml_element_size(b);
    ggml_tensor * b4 = ggml_view_4d(ctx, b,
        /*ne0*/1, /*ne1*/b->ne[0], /*ne2*/1, /*ne3*/1,
        /*nb1*/1 * esize,
        /*nb2*/b->ne[0] * esize,
        /*nb3*/b->ne[0] * esize,
        /*offset*/0);
    return ggml_add(ctx, y, b4);
}
}

ggml_tensor * cgmlp(ggml_context * ctx,
                    ggml_tensor * x,
                    ggml_tensor * w_pw1, ggml_tensor * b_pw1,
                    ggml_tensor * w_norm,
                    ggml_tensor * w_dw,  ggml_tensor * b_dw,
                    ggml_tensor * w_pw2, ggml_tensor * b_pw2,
                    int kernel_size,
                    bool use_dw_act,
                    float norm_eps) {
    // 1x1 Conv1D weights are just Linear weights: reshape (1, in, out) -> (in, out).
    assert(w_pw1->ne[0] == 1);
    assert(w_pw2->ne[0] == 1);
    ggml_tensor * w_pw1_2d = ggml_reshape_2d(ctx, w_pw1, w_pw1->ne[1], w_pw1->ne[2]);
    ggml_tensor * w_pw2_2d = ggml_reshape_2d(ctx, w_pw2, w_pw2->ne[1], w_pw2->ne[2]);

    // Input x: ne = (D, T, B).
    // ----- pw1 -----------------------------------------------------------------
    ggml_tensor * h = linear(ctx, x, w_pw1_2d, b_pw1);  // (2L, T, B)
    h = ggml_gelu(ctx, h);

    const int64_t L = h->ne[0] / 2;
    const size_t  esize = ggml_element_size(h);

    // ----- chunk ---------------------------------------------------------------
    ggml_tensor * x1 = ggml_view_4d(ctx, h,
        L, h->ne[1], h->ne[2], h->ne[3],
        h->nb[1], h->nb[2], h->nb[3],
        /*offset=*/0);
    ggml_tensor * x2 = ggml_view_4d(ctx, h,
        L, h->ne[1], h->ne[2], h->ne[3],
        h->nb[1], h->nb[2], h->nb[3],
        /*offset=*/L * esize);

    // RMSNorm along ne0=L (PyTorch applies it on the channel dim after transpose
    // to (B, T, L) and transposes back; effect is identical).
    // `ggml_rms_norm` can take strided input; `ggml_mul` with w_norm broadcasts.
    ggml_tensor * x2_norm = rms_norm(ctx, ggml_cont(ctx, x2), w_norm, norm_eps);

    // ----- Depthwise 1D conv ---------------------------------------------------
    // ggml_conv_1d_dw expects input b shape (T, C, B) with ne=(T, C, B, 1)
    // and kernel a shape (K, 1, C).  Our x2_norm is (L, T, B); permute to
    // (T, L, B) and make contiguous before the call.
    ggml_tensor * x2_tr = ggml_cont(ctx, ggml_permute(ctx, x2_norm, 1, 0, 2, 3));
    const int stride = 1;
    const int pad    = (kernel_size - 1) / 2;
    const int dil    = 1;
    // CPU: dedicated per-channel kernel (no im2col); Vulkan: conv_1d_dw.
    ggml_tensor * conv = dwconv_1d(ctx, w_dw, x2_tr, kernel_size, pad);  // (T, L, B)
    conv = add_conv_bias(ctx, conv, b_dw);
    if (use_dw_act) conv = ggml_gelu(ctx, conv);

    // Permute back to (L, T, B) so we can multiply with x1.
    ggml_tensor * x2_back = ggml_cont(ctx, ggml_permute(ctx, conv, 1, 0, 2, 3));
    ggml_tensor * merged  = ggml_mul(ctx, ggml_cont(ctx, x1), x2_back);

    // ----- pw2 -----------------------------------------------------------------
    return linear(ctx, merged, w_pw2_2d, b_pw2);
}

}  // namespace game_ggml::internal::ops
