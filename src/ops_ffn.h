#pragma once

// Feed-forward variants used inside EBF / JEBF blocks.
//
// Conventions (ggml shape order, innermost dim first):
//   Activations:  ne = (D, T, B)  — D is channel/embedding dim.
//   Weights:      same row-major layout as the source PyTorch tensor.

struct ggml_context;
struct ggml_tensor;

namespace game_ggml::internal::ops {

// --------------------------------------------------------------------------
// GLUFFN  (modules.backbones.layers.GLUFFN)
//
//   x1, x2 = chunk(ln1(x))         # each [..., L]
//   y      = gelu(x1) * x2
//   y      = ln2(y)                # back to [..., D]
//
// Weights (PyTorch → GGUF → ggml):
//   w_ln1  (2L, D)    ne = (D,  2L)
//   b_ln1  (2L,)      ne = (2L,)
//   w_ln2  (D,  L)    ne = (L,  D)
//   b_ln2  (D,)       ne = (D,)
// --------------------------------------------------------------------------
ggml_tensor * glu_ffn(ggml_context * ctx,
                      ggml_tensor * x,
                      ggml_tensor * w_ln1, ggml_tensor * b_ln1,
                      ggml_tensor * w_ln2, ggml_tensor * b_ln2);

// Same GLU FFN but with the ln1 projection pre-split into two [in, L]
// halves (load-time, see tensor_utils.cpp).  Two mul_mats produce contiguous
// [L, T, B] outputs directly, skipping the strided-view + 2x cont of the
// monolithic path.  w_ln2/b_ln2 are unchanged (L -> D).
ggml_tensor * glu_ffn_split(ggml_context * ctx,
                            ggml_tensor * x,
                            ggml_tensor * w_ln1_a, ggml_tensor * b_ln1_a,
                            ggml_tensor * w_ln1_b, ggml_tensor * b_ln1_b,
                            ggml_tensor * w_ln2, ggml_tensor * b_ln2);

// --------------------------------------------------------------------------
// CgMLP (modules.backbones.layers.CgMLP)
//
// NOTE: ggml v0.11.0's `ggml_conv_1d_dw` asserts that the input's ne[3] is 1
// (it internally reshapes to (T, 1, C, B) and im2col requires the 4th dim
// to equal 1).  GAME inference runs single clips so this matches our
// contract; training-time batched use would need a manual loop.
//
//   x   = transpose(x, 1, 2)          # [B, D, T]
//   x   = pw1(x)                      # [B, 2L, T]
//   x   = gelu(x)
//   x1, x2 = chunk(x, dim=1)          # each [B, L, T]
//   x2  = RMSNorm_along_channels(x2)
//   x2  = dw(x2)                      # depthwise 1D conv kernel=K
//   x2  = gelu(x2)                    # when use_dw_act (default)
//   x   = x1 * x2
//   x   = pw2(x)                      # [B, D, T]
//   return transpose(x, 1, 2)         # [B, T, D]
//
// Weights (PyTorch shapes → ggml ne[]):
//   w_pw1   (2L, D, 1)       ne = (1, D,  2L)    — we view as (D, 2L) below
//   b_pw1   (2L,)
//   w_norm  (L,)
//   w_dw    (L,  1, K)       ne = (K, 1,  L)
//   b_dw    (L,)             optional
//   w_pw2   (D,  L, 1)       ne = (1, L,  D)     — viewed as (L, D)
//   b_pw2   (D,)
// --------------------------------------------------------------------------
ggml_tensor * cgmlp(ggml_context * ctx,
                    ggml_tensor * x,
                    ggml_tensor * w_pw1, ggml_tensor * b_pw1,
                    ggml_tensor * w_norm,
                    ggml_tensor * w_dw,  ggml_tensor * b_dw,
                    ggml_tensor * w_pw2, ggml_tensor * b_pw2,
                    int kernel_size,
                    bool use_dw_act = true,
                    float norm_eps  = 1e-6f);

}  // namespace game_ggml::internal::ops
