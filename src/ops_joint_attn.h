#pragma once

// Joint attention (pool stream ⋈ x stream) + PJAC + JEBF block.
//
// This is the estimator-side counterpart of ops_attn.h.  Only the
// `attn_type='joint'` + `rope_mode='mixed'` + `qk_norm=true` +
// `region_token_num=1` + `use_region_bias=false` + `pool_merge_mode='mean'`
// code path is implemented — the only one used by the shipped 1.0-small
// checkpoint.  Other branches throw NotImplemented at model load.

#include "ops_attn.h"

#include <cstdint>
#include <vector>

struct ggml_backend;
struct ggml_context;
struct ggml_tensor;
typedef struct ggml_backend * ggml_backend_t;

namespace game_ggml::internal::ops {

// Build a bool (fp16) attention-mask tensor for JointAttention.
//
//   regions     : length T         (int32, 0 = padding, 1..N = valid)
//   n_regions   : total N
//   t_valid_all : assume all frames valid (B=1 inference case)
//
// Returns a flat host buffer of (P+T) * (P+T) fp16 values (0.0 = allowed,
// -10000.0 = blocked).  Caller uploads as (P+T, P+T, 1, 1) fp16 tensor.
std::vector<std::uint16_t>  // ggml_fp16_t is uint16_t bit pattern
build_joint_attn_mask_fp16(
    const std::int32_t * regions,
    int T,
    int n_regions);

// ----------------------------------------------------------------------------
// Weight bundles
// ----------------------------------------------------------------------------
struct JointAttentionWeights {
    // Pre-QKV RMSNorms (per-stream, dim=D_embed).
    ggml_tensor * w_pool_norm    = nullptr;
    ggml_tensor * w_x_norm       = nullptr;
    // QKV projections (D_embed -> 3*H*D_head).
    ggml_tensor * w_pool_qkv     = nullptr;
    ggml_tensor * b_pool_qkv     = nullptr;
    ggml_tensor * w_x_qkv        = nullptr;
    ggml_tensor * b_x_qkv        = nullptr;
    // Q/K RMSNorms per head (dim=D_head).
    ggml_tensor * w_pool_q_norm  = nullptr;
    ggml_tensor * w_pool_k_norm  = nullptr;
    ggml_tensor * w_x_q_norm     = nullptr;
    ggml_tensor * w_x_k_norm     = nullptr;
    // Output projections (H*D_head -> D_embed).
    ggml_tensor * w_pool_out     = nullptr;
    ggml_tensor * b_pool_out     = nullptr;
    ggml_tensor * w_x_out        = nullptr;
    ggml_tensor * b_x_out        = nullptr;
};

struct PJACWeights {
    JointAttentionWeights jattn{};
    // CgMLP per stream
    ggml_tensor * w_c_norm_x      = nullptr;
    ggml_tensor * w_c_norm_pool   = nullptr;
    ggml_tensor * w_cg_x_pw1      = nullptr;
    ggml_tensor * b_cg_x_pw1      = nullptr;
    ggml_tensor * w_cg_x_norm     = nullptr;
    ggml_tensor * w_cg_x_dw       = nullptr;
    ggml_tensor * b_cg_x_dw       = nullptr;
    ggml_tensor * w_cg_x_pw2      = nullptr;
    ggml_tensor * b_cg_x_pw2      = nullptr;
    int           cg_x_kernel     = 31;
    ggml_tensor * w_cg_pool_pw1   = nullptr;
    ggml_tensor * b_cg_pool_pw1   = nullptr;
    ggml_tensor * w_cg_pool_norm  = nullptr;
    ggml_tensor * w_cg_pool_dw    = nullptr;
    ggml_tensor * b_cg_pool_dw    = nullptr;
    ggml_tensor * w_cg_pool_pw2   = nullptr;
    ggml_tensor * b_cg_pool_pw2   = nullptr;
    int           cg_pool_kernel  = 7;
    // Merge per stream
    ggml_tensor * w_merge_x       = nullptr;
    ggml_tensor * b_merge_x       = nullptr;
    ggml_tensor * w_merge_dw_x    = nullptr;    // optional
    ggml_tensor * b_merge_dw_x    = nullptr;
    int           merge_x_kernel  = 31;
    ggml_tensor * w_merge_pool    = nullptr;
    ggml_tensor * b_merge_pool    = nullptr;
    ggml_tensor * w_merge_dw_pool = nullptr;    // optional
    ggml_tensor * b_merge_dw_pool = nullptr;
    int           merge_pool_kernel = 5;
};

struct JEBFBlockWeights {
    // Per-stream FFN1
    bool has_ffn1 = true;
    ggml_tensor * w_norm_ffn1_x    = nullptr;
    ggml_tensor * w_norm_ffn1_pool = nullptr;
    ggml_tensor * w_ffn1_x_ln1     = nullptr;
    ggml_tensor * w_ffn1_x_ln1_a  = nullptr;    // F-1 split halves (.a/.b)
    ggml_tensor * b_ffn1_x_ln1_a  = nullptr;
    ggml_tensor * w_ffn1_x_ln1_b  = nullptr;
    ggml_tensor * b_ffn1_x_ln1_b  = nullptr;
    ggml_tensor * b_ffn1_x_ln1     = nullptr;
    ggml_tensor * w_ffn1_x_ln2     = nullptr;
    ggml_tensor * b_ffn1_x_ln2     = nullptr;
    ggml_tensor * w_ffn1_pool_ln1  = nullptr;
    ggml_tensor * w_ffn1_pool_ln1_a = nullptr;    // F-1 split halves
    ggml_tensor * b_ffn1_pool_ln1_a = nullptr;
    ggml_tensor * w_ffn1_pool_ln1_b = nullptr;
    ggml_tensor * b_ffn1_pool_ln1_b = nullptr;
    ggml_tensor * b_ffn1_pool_ln1  = nullptr;
    ggml_tensor * w_ffn1_pool_ln2  = nullptr;
    ggml_tensor * b_ffn1_pool_ln2  = nullptr;
    ggml_tensor * w_lay_scale_ffn1_x    = nullptr;    // optional
    ggml_tensor * w_lay_scale_ffn1_pool = nullptr;    // optional

    PJACWeights pjac{};
    ggml_tensor * w_lay_scale_jpac_x    = nullptr;    // optional
    ggml_tensor * w_lay_scale_jpac_pool = nullptr;    // optional

    // Per-stream FFN2
    bool has_ffn2 = true;
    ggml_tensor * w_norm_ffn2_x    = nullptr;
    ggml_tensor * w_norm_ffn2_pool = nullptr;
    ggml_tensor * w_ffn2_x_ln1     = nullptr;
    ggml_tensor * w_ffn2_x_ln1_a  = nullptr;    // F-1 split halves
    ggml_tensor * b_ffn2_x_ln1_a  = nullptr;
    ggml_tensor * w_ffn2_x_ln1_b  = nullptr;
    ggml_tensor * b_ffn2_x_ln1_b  = nullptr;
    ggml_tensor * b_ffn2_x_ln1     = nullptr;
    ggml_tensor * w_ffn2_x_ln2     = nullptr;
    ggml_tensor * b_ffn2_x_ln2     = nullptr;
    ggml_tensor * w_ffn2_pool_ln1  = nullptr;
    ggml_tensor * w_ffn2_pool_ln1_a = nullptr;    // F-1 split halves
    ggml_tensor * b_ffn2_pool_ln1_a = nullptr;
    ggml_tensor * w_ffn2_pool_ln1_b = nullptr;
    ggml_tensor * b_ffn2_pool_ln1_b = nullptr;
    ggml_tensor * b_ffn2_pool_ln1  = nullptr;
    ggml_tensor * w_ffn2_pool_ln2  = nullptr;
    ggml_tensor * b_ffn2_pool_ln2  = nullptr;
    ggml_tensor * w_lay_scale_ffn2_x    = nullptr;
    ggml_tensor * w_lay_scale_ffn2_pool = nullptr;
};

// ----------------------------------------------------------------------------
// Graph builders
// ----------------------------------------------------------------------------

// Run joint attention across pool + x streams.  Returns (pool_out, x_out).
// Note: PyTorch performs FFN *before* PJAC; we match that ordering in
// jebf_block.
struct JoinResult { ggml_tensor * pool; ggml_tensor * x; };

JoinResult joint_attention(
    ggml_context * ctx,
    ggml_tensor * pool,                    // (D, N, 1)
    ggml_tensor * x,                       // (D, T, 1)
    const JointAttentionWeights & W,
    ggml_tensor * global_positions,        // int32 (P+T,)
    ggml_tensor * region_indices,          // int32 (P+T,)
    ggml_tensor * attn_mask_fp16,          // (P+T, P+T, 1, 1) fp16
    int num_heads,
    int head_dim,
    float theta = 10000.0f);

// PJAC: parallel joint-attn + CgMLP, per-stream merges.
JoinResult pjac(
    ggml_context * ctx,
    ggml_tensor * pool,
    ggml_tensor * x,
    const PJACWeights & W,
    ggml_tensor * global_positions,
    ggml_tensor * region_indices,
    ggml_tensor * attn_mask_fp16,
    int num_heads,
    int head_dim,
    float theta = 10000.0f);

// Full JEBF block (FFN1 + PJAC + FFN2 with residual + LayerScale on each).
JoinResult jebf_block(
    ggml_context * ctx,
    ggml_tensor * pool,
    ggml_tensor * x,
    const JEBFBlockWeights & W,
    ggml_tensor * global_positions,
    ggml_tensor * region_indices,
    ggml_tensor * attn_mask_fp16,
    int num_heads,
    int head_dim,
    float theta = 10000.0f);

}  // namespace game_ggml::internal::ops
