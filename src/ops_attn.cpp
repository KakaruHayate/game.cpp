#include "ops_attn.h"

#include "ops_basic.h"
#include "ops_ffn.h"
#include "ops_rope.h"

#include <ggml.h>

#include <cassert>
#include <cmath>

namespace game_ggml::internal::ops {

// ---------------------------------------------------------------------------
// Attention with RoPE
// ---------------------------------------------------------------------------

namespace {

// Split a (3*D, T, B) tensor along ne[0] into q/k/v views (fused QKV).
struct QKVTriple { ggml_tensor * q; ggml_tensor * k; ggml_tensor * v; };

QKVTriple chunk_three(ggml_context * ctx, ggml_tensor * qkv) {
    const int64_t D = qkv->ne[0] / 3;
    const size_t  esize = ggml_element_size(qkv);
    QKVTriple r;
    r.q = ggml_view_4d(ctx, qkv, D, qkv->ne[1], qkv->ne[2], qkv->ne[3],
        qkv->nb[1], qkv->nb[2], qkv->nb[3], /*offset=*/0);
    r.k = ggml_view_4d(ctx, qkv, D, qkv->ne[1], qkv->ne[2], qkv->ne[3],
        qkv->nb[1], qkv->nb[2], qkv->nb[3], /*offset=*/D * esize);
    r.v = ggml_view_4d(ctx, qkv, D, qkv->ne[1], qkv->ne[2], qkv->ne[3],
        qkv->nb[1], qkv->nb[2], qkv->nb[3], /*offset=*/2 * D * esize);
    return r;
}

}  // namespace

ggml_tensor * attention_with_rope(
    ggml_context * ctx,
    ggml_tensor * x,
    const AttentionWeights & W,
    ggml_tensor * positions,
    int num_heads,
    int head_dim,
    float theta) {
    const int64_t attn_dim = num_heads * head_dim;
    const int64_t T = x->ne[1];
    const int64_t B = x->ne[2];

    // Fused QKV projection (single [3*H*D, D] linear, then split).
    ggml_tensor * qkv = linear(ctx, x, W.w_qkv, W.b_qkv);   // (3*H*D, T, B)
    auto [q_flat, k_flat, v_flat] = chunk_three(ctx, qkv);

    // Reshape to (D, H, T, B) so RoPE can be applied (ne[2] == T).
    ggml_tensor * qr = ggml_reshape_4d(ctx, ggml_cont(ctx, q_flat), head_dim, num_heads, T, B);
    ggml_tensor * kr = ggml_reshape_4d(ctx, ggml_cont(ctx, k_flat), head_dim, num_heads, T, B);
    ggml_tensor * vr = ggml_reshape_4d(ctx, ggml_cont(ctx, v_flat), head_dim, num_heads, T, B);

    // RoPE.
    qr = apply_rope(ctx, qr, positions, head_dim, theta);   // (D, H, T, B)
    kr = apply_rope(ctx, kr, positions, head_dim, theta);

    // Permute to (D, T, H, B) for flash_attn_ext.
    qr = ggml_cont(ctx, ggml_permute(ctx, qr, 0, 2, 1, 3));
    kr = ggml_cont(ctx, ggml_permute(ctx, kr, 0, 2, 1, 3));
    vr = ggml_cont(ctx, ggml_permute(ctx, vr, 0, 2, 1, 3));

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    ggml_tensor * o = ggml_flash_attn_ext(ctx, qr, kr, vr,
        /*mask=*/nullptr, scale, /*max_bias=*/0.0f, /*logit_softcap=*/0.0f);
    // Output shape per ggml_flash_attn_ext: (D, H, T, B)
    // — ne[1] = q->ne[2] = H; ne[2] = q->ne[1] = T.

    // Flatten heads: (D, H, T, B) → (H*D, T, B).  This works because H*D is
    // contiguous in memory (d varies innermost, then h).
    o = ggml_cont(ctx, o);
    o = ggml_reshape_3d(ctx, o, attn_dim, T, B);

    // Output projection.
    return linear(ctx, o, W.w_out, W.b_out);
}

// ---------------------------------------------------------------------------
// PAC (parallel attention + CgMLP, merge path)
// ---------------------------------------------------------------------------

ggml_tensor * pac(
    ggml_context * ctx,
    ggml_tensor * x,
    const PACWeights & W,
    ggml_tensor * positions,
    int num_heads,
    int head_dim,
    float theta) {
    // Attention branch
    ggml_tensor * ax = rms_norm(ctx, x, W.w_a_norm);
    ggml_tensor * a  = attention_with_rope(ctx, ax, W.attn, positions,
                                           num_heads, head_dim, theta);

    // CgMLP branch
    ggml_tensor * cx = rms_norm(ctx, x, W.w_c_norm);
    ggml_tensor * c  = cgmlp(ctx, cx,
        W.w_cg_pw1, W.b_cg_pw1,
        W.w_cg_norm,
        W.w_cg_dw,  W.b_cg_dw,
        W.w_cg_pw2, W.b_cg_pw2,
        W.cg_kernel_size);

    // Concatenate along ne[0] (the channel/D axis).
    ggml_tensor * m = ggml_concat(ctx, a, c, /*dim=*/0);   // (2D, T, B)

    // Optional depthwise conv on the concatenated channels.
    if (W.merge_kernel_size != 0 && W.w_merge_dw) {
        // Permute (2D, T, B) → (T, 2D, B) for the depthwise conv layout.
        ggml_tensor * m_tr = ggml_cont(ctx, ggml_permute(ctx, m, 1, 0, 2, 3));
        const int stride = 1;
        const int pad    = (W.merge_kernel_size - 1) / 2;
        const int dil    = 1;
        ggml_tensor * cv = dwconv_1d(ctx, W.w_merge_dw, m_tr, W.merge_kernel_size, pad);
        // Per-channel bias: (2D,) broadcast over (T, 2D, B) — view with ne=(1, 2D, 1, 1).
        if (W.b_merge_dw) {
            const size_t esize = ggml_element_size(W.b_merge_dw);
            const int64_t C = W.b_merge_dw->ne[0];
            ggml_tensor * b4 = ggml_view_4d(ctx, W.b_merge_dw,
                1, C, 1, 1,
                esize, C * esize, C * esize,
                0);
            cv = ggml_add(ctx, cv, b4);
        }
        // Permute back (T, 2D, B) → (2D, T, B)
        ggml_tensor * cv_back = ggml_cont(ctx, ggml_permute(ctx, cv, 1, 0, 2, 3));
        m = ggml_add(ctx, cv_back, m);
    }

    return linear(ctx, m, W.w_merge_linear, W.b_merge_linear);
}

// ---------------------------------------------------------------------------
// EBF block
// ---------------------------------------------------------------------------
//
// F-2: the branch residuals are  x + 0.5·lay_scale(branch).  Both factors are
// diagonal and are folded into the producing linear (ffn*.ln2 / merge_linear)
// at load time (tensor_utils.cpp), so the graph no longer emits the lay_scale
// mul and the 0.5 scale node per block.  w_lay_scale* are still bound (the
// GGUF keeps the tensors) but are intentionally not referenced here.

ggml_tensor * ebf_block(
    ggml_context * ctx,
    ggml_tensor * x,
    const EBFBlockWeights & W,
    ggml_tensor * positions,
    int num_heads,
    int head_dim,
    float theta) {
    // FFN 1 (pre-attention)
    if (W.has_ffn1) {
        ggml_tensor * h = rms_norm(ctx, x, W.w_norm1);
        if (W.w_ffn1_ln1_a) {
            h = glu_ffn_split(ctx, h,
                W.w_ffn1_ln1_a, W.b_ffn1_ln1_a, W.w_ffn1_ln1_b, W.b_ffn1_ln1_b,
                W.w_ffn1_ln2, W.b_ffn1_ln2);
        } else {
            h = glu_ffn(ctx, h, W.w_ffn1_ln1, W.b_ffn1_ln1, W.w_ffn1_ln2, W.b_ffn1_ln2);
        }
        x = ggml_add(ctx, x, h);
    }

    // PAC
    ggml_tensor * p = pac(ctx, x, W.pac_w, positions, num_heads, head_dim, theta);
    x = ggml_add(ctx, x, p);

    // FFN 2 (post-attention)
    if (W.has_ffn2) {
        ggml_tensor * h = rms_norm(ctx, x, W.w_norm2);
        if (W.w_ffn2_ln1_a) {
            h = glu_ffn_split(ctx, h,
                W.w_ffn2_ln1_a, W.b_ffn2_ln1_a, W.w_ffn2_ln1_b, W.b_ffn2_ln1_b,
                W.w_ffn2_ln2, W.b_ffn2_ln2);
        } else {
            h = glu_ffn(ctx, h, W.w_ffn2_ln1, W.b_ffn2_ln1, W.w_ffn2_ln2, W.b_ffn2_ln2);
        }
        x = ggml_add(ctx, x, h);
    }

    return x;
}

}  // namespace game_ggml::internal::ops
