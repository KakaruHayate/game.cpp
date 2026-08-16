#include "ops_basic.h"

#include <ggml.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace game_ggml::internal::ops {

ggml_tensor * rms_norm(ggml_context * ctx,
                       ggml_tensor * x,
                       ggml_tensor * weight,
                       float eps) {
    ggml_tensor * n = ggml_rms_norm(ctx, x, eps);
    // Broadcast-multiply by weight (ne0 matches; higher dims broadcast).
    return ggml_mul(ctx, n, weight);
}

ggml_tensor * linear(ggml_context * ctx,
                     ggml_tensor * x,
                     ggml_tensor * weight,
                     ggml_tensor * bias) {
    // ggml_mul_mat(A, B):  A(ne0=K, ne1=M)  B(ne0=K, ne1=N, ...) -> (ne0=M, ne1=N, ...)
    // For Linear with PyTorch weight shape (out, in) stored row-major, we
    // have ggml ne0=in, ne1=out.  Input x has ne0=in, so the result has
    // ne0=out — exactly what Linear produces.
    ggml_tensor * y = ggml_mul_mat(ctx, weight, x);
    if (bias) {
        // `ggml_add` broadcasts `b` (ne0=out, higher dims=1) across higher
        // dims of `y`; this works on both CPU and Metal backends.
        y = ggml_add(ctx, y, bias);
    }
    return y;
}

ggml_tensor * layer_scale(ggml_context * ctx,
                          ggml_tensor * x,
                          ggml_tensor * scale) {
    return ggml_mul(ctx, x, scale);
}

ggml_tensor * embedding(ggml_context * ctx,
                        ggml_tensor * weight,
                        ggml_tensor * indices) {
    return ggml_get_rows(ctx, weight, indices);
}

// ---------------------------------------------------------------------------
// Depthwise 1D conv (same padding)
// ---------------------------------------------------------------------------

namespace {
// Per-thread backend capability: set only by the CPU backend (the dedicated
// GGML_OP_CONV_2D_DW kernel exists there; GPU backends use im2col).  Stored
// thread-locally so graph building on one thread is never affected by a model
// load on another thread.  Assumes a single active model per thread, which is
// the norm for this CLI; cross-model coexistence should use separate threads.
thread_local bool g_direct_dwconv = false;
}

void set_direct_dwconv(bool enable) { g_direct_dwconv = enable; }
bool direct_dwconv() { return g_direct_dwconv; }

ggml_tensor * dwconv_1d(ggml_context * ctx,
                        ggml_tensor * w_dw,
                        ggml_tensor * x,
                        int kernel_size,
                        int pad) {
    // Runtime override for debugging / A-B comparison:
    //   GAME_GGML_DWCONV=legacy  forces the im2col path on any backend.
    //   GAME_GGML_DWCONV=direct requests the dedicated kernel, but only if
    //   the backend actually supports it (g_direct_dwconv was set by the
    //   backend capability check at model load) — this env value never
    //   enables a path the backend cannot run.  Unknown values are ignored.
    bool direct = g_direct_dwconv;
    if (const char * env = std::getenv("GAME_GGML_DWCONV"); env && *env) {
        if (std::strcmp(env, "legacy") == 0) {
            direct = false;   // exact "legacy" forces im2col everywhere
        } else if (std::strcmp(env, "direct") != 0) {
            std::fprintf(stderr,
                "[DWCONV] ignoring unknown GAME_GGML_DWCONV value '%s' "
                "(expected 'direct' | 'legacy')\n", env);
        }
        // "direct" is a no-op: capability already encoded in g_direct_dwconv.
    }

    if (direct) {
        // Express the 1D depthwise conv as a 2D depthwise conv with H=1 so it
        // runs through ggml's dedicated per-channel kernel (GGML_OP_CONV_2D_DW,
        // whcn path) instead of the im2col + F16 mul_mat path of
        // ggml_conv_1d_dw.  Layout mapping:
        //   w_dw ne=(K, 1, C)          -> kernel ne=(K, 1, 1, C)   (OC=C, IC=1)
        //   x    ne=(T, C, B)          -> input  ne=(T, 1, C, B)   (W=T, H=1)
        //   out  ne=(T, 1, C, B)       -> ne=(T, C, B)
        // Memory order stays [k][c] / [t][c][b] which is exactly what the
        // whcn kernel reads (channel-major kernel rows, channel-major input).
        const int64_t K = w_dw->ne[0];
        const int64_t C = w_dw->ne[2];
        const int64_t T = x->ne[0];
        const int64_t B = x->ne[2];
        // The whcn kernel reads the kernel through a float* — only F32 is
        // supported; the depthwise weights may be stored F16 in the GGUF.
        ggml_tensor * w = (w_dw->type == GGML_TYPE_F32)
            ? w_dw : ggml_cast(ctx, w_dw, GGML_TYPE_F32);
        ggml_tensor * w4 = ggml_reshape_4d(ctx, w, K, 1, 1, C);
        ggml_tensor * x4 = ggml_reshape_4d(ctx, x, T, 1, C, B);
        if (std::getenv("GAME_GGML_DUMP_DWCONV")) {
            std::fprintf(stderr, "[DWCONV] direct K=%lld C=%lld T=%lld B=%lld w=%s x=%s\n",
                (long long) K, (long long) C, (long long) T, (long long) B,
                ggml_type_name(w_dw->type), ggml_type_name(x->type));
        }
        ggml_tensor * y4 = ggml_conv_2d_dw_direct(ctx, w4, x4,
            /*s0=*/1, /*s1=*/1, /*p0=*/pad, /*p1=*/0, /*d0=*/1, /*d1=*/1);
        return ggml_reshape_3d(ctx, y4, T, C, B);
    }

    // Legacy path (Vulkan/Metal): ggml_conv_1d_dw -> im2col_f16 requires the
    // kernel to be F16; cast the static weight when it is stored F32.
    ggml_tensor * w_f16 = (w_dw->type == GGML_TYPE_F16)
        ? w_dw : ggml_cast(ctx, w_dw, GGML_TYPE_F16);
    return ggml_conv_1d_dw(ctx, w_f16, x, /*s0=*/1, /*p0=*/pad, /*d0=*/1);
}

}  // namespace game_ggml::internal::ops
