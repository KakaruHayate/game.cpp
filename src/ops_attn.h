#pragma once

// Attention → PAC → EBF block → EBFBackbone.
//
// These are the building blocks of the encoder and segmenter backbones.
// Weight bundles are grouped as POD structs so callers (model loaders) can
// populate them once from GGUF and reuse the same builder for every layer.

struct ggml_context;
struct ggml_tensor;

namespace game_ggml::internal::ops {

// --------------------------------------------------------------------------
// AttentionWithRoPE (modules.backbones.ebf_with_joint_attention.AttentionWithRoPE)
//
//   q, k, v = qkv_linear(x).chunk(3, -1)   # each  [B, T, H*D]
//   → reshape to (B, H, T, D) → RoPE → scaled dot-product attention → merge
//   out    = out_linear(out)              # [B, T, D_embed]
// q_linear/kv_linear are fused by the converter into a single qkv tensor
// (llama-style); the weights are stored row-aligned so quantization is
// bit-identical to the un-fused form.
// --------------------------------------------------------------------------
struct AttentionWeights {
    ggml_tensor * w_qkv   = nullptr;
    ggml_tensor * b_qkv   = nullptr;
    ggml_tensor * w_out   = nullptr;
    ggml_tensor * b_out   = nullptr;
};

// Build the attention graph.  Expects x with ne = (D_embed, T, B).  The
// positions tensor is shared across heads / batch (ne = (T,), dtype I32).
ggml_tensor * attention_with_rope(
    ggml_context * ctx,
    ggml_tensor * x,
    const AttentionWeights & W,
    ggml_tensor * positions,
    int num_heads,
    int head_dim,
    float theta = 10000.0f);

// --------------------------------------------------------------------------
// PAC (Parallel Attention + CgMLP) — modules.backbones.ebf_with_joint_attention.PAC
//
//   a_o  = attn(a_norm(x))
//   c_o  = cgmlp(c_norm(x))
//   m_o  = [a_o; c_o]
//   if merge_dw_conv: m_o = dw(m_o) + m_o
//   m_o  = merge_linear(m_o)
// --------------------------------------------------------------------------
struct PACWeights {
    // Pre-norms
    ggml_tensor * w_a_norm       = nullptr;
    ggml_tensor * w_c_norm       = nullptr;
    // Attention
    AttentionWeights attn{};
    // CgMLP
    ggml_tensor * w_cg_pw1       = nullptr;
    ggml_tensor * b_cg_pw1       = nullptr;
    ggml_tensor * w_cg_norm      = nullptr;
    ggml_tensor * w_cg_dw        = nullptr;
    ggml_tensor * b_cg_dw        = nullptr;
    ggml_tensor * w_cg_pw2       = nullptr;
    ggml_tensor * b_cg_pw2       = nullptr;
    int           cg_kernel_size = 31;
    // Merge layer (catted [a;c] -> D)
    ggml_tensor * w_merge_linear = nullptr;
    ggml_tensor * b_merge_linear = nullptr;
    ggml_tensor * w_merge_dw     = nullptr;   // optional
    ggml_tensor * b_merge_dw     = nullptr;   // optional
    int           merge_kernel_size = 31;     // 0 disables the merge conv
};

ggml_tensor * pac(
    ggml_context * ctx,
    ggml_tensor * x,
    const PACWeights & W,
    ggml_tensor * positions,
    int num_heads,
    int head_dim,
    float theta = 10000.0f);

// --------------------------------------------------------------------------
// EBF block (modules.backbones.EBF.EBF)
//
//   if !skip_first_ffn:
//       x = lay_scale1(ffn1(norm1(x))) * 0.5 + x
//   x = lay_scale2(pac(x)) + x
//   if !skip_out_ffn:
//       x = lay_scale3(ffn2(norm2(x))) * 0.5 + x
//
// Mask handling is deferred (task 7 / attention mask path) — inference with
// B=1 doesn't need it for the encoder.
// --------------------------------------------------------------------------
struct EBFBlockWeights {
    // FFN 1
    bool            has_ffn1        = true;
    ggml_tensor *   w_norm1         = nullptr;
    ggml_tensor *   w_ffn1_ln1      = nullptr;   // ln1.weight  (D -> 2L)
    ggml_tensor *   b_ffn1_ln1      = nullptr;
    // Split ln1 halves (load-time; null => use monolithic ln1 above).
    ggml_tensor *   w_ffn1_ln1_a    = nullptr;   // .ln1.weight.a  (D -> L)
    ggml_tensor *   b_ffn1_ln1_a    = nullptr;
    ggml_tensor *   w_ffn1_ln1_b    = nullptr;   // .ln1.weight.b
    ggml_tensor *   b_ffn1_ln1_b    = nullptr;
    ggml_tensor *   w_ffn1_ln2      = nullptr;   // ln2.weight  (L ->  D)
    ggml_tensor *   b_ffn1_ln2      = nullptr;
    ggml_tensor *   w_lay_scale1    = nullptr;   // optional

    // PAC
    PACWeights      pac_w{};
    ggml_tensor *   w_lay_scale2    = nullptr;   // optional (always present for use_ls=true)

    // FFN 2
    bool            has_ffn2        = true;
    ggml_tensor *   w_norm2         = nullptr;
    ggml_tensor *   w_ffn2_ln1      = nullptr;
    ggml_tensor *   b_ffn2_ln1      = nullptr;
    ggml_tensor *   w_ffn2_ln1_a    = nullptr;   // .a/.b split halves
    ggml_tensor *   b_ffn2_ln1_a    = nullptr;
    ggml_tensor *   w_ffn2_ln1_b    = nullptr;
    ggml_tensor *   b_ffn2_ln1_b    = nullptr;
    ggml_tensor *   w_ffn2_ln2      = nullptr;
    ggml_tensor *   b_ffn2_ln2      = nullptr;
    ggml_tensor *   w_lay_scale3    = nullptr;   // optional
};

ggml_tensor * ebf_block(
    ggml_context * ctx,
    ggml_tensor * x,
    const EBFBlockWeights & W,
    ggml_tensor * positions,
    int num_heads,
    int head_dim,
    float theta = 10000.0f);

}  // namespace game_ggml::internal::ops
