#include "model_estimator.h"

#include "game_ggml/errors.h"
#include "ops_basic.h"
#include "ops_ffn.h"
#include "ops_joint_attn.h"
#include "tensor_utils.h"

#include <ggml.h>

#include <string>

namespace game_ggml::internal {

static ops::JEBFBlockWeights bind_jebf_layer(
    const LoadedWeights & W, int i, const BackboneConfig & cfg)
{
    ops::JEBFBlockWeights B{};
    const std::string p = "estimator.layers." + std::to_string(i) + ".";

    B.has_ffn1 = !cfg.skip_first_ffn;
    B.has_ffn2 = !cfg.skip_out_ffn;
    if (B.has_ffn1) {
        B.w_norm_ffn1_x    = W.get(p + "norm_ffn1_x.weight");
        B.w_norm_ffn1_pool = W.get(p + "norm_ffn1_pool.weight");
        B.w_ffn1_x_ln1 = W.get(p + "ffn1_x.ln1.weight");
        B.b_ffn1_x_ln1 = W.get(p + "ffn1_x.ln1.bias");
        B.w_ffn1_x_ln1_a = W.try_get(p + "ffn1_x.ln1.weight.a");   // .a/.b split halves
        B.b_ffn1_x_ln1_a = W.try_get(p + "ffn1_x.ln1.bias.a");
        B.w_ffn1_x_ln1_b = W.try_get(p + "ffn1_x.ln1.weight.b");
        B.b_ffn1_x_ln1_b = W.try_get(p + "ffn1_x.ln1.bias.b");
        B.w_ffn1_x_ln2 = W.get(p + "ffn1_x.ln2.weight");
        B.b_ffn1_x_ln2 = W.get(p + "ffn1_x.ln2.bias");
        B.w_ffn1_pool_ln1 = W.get(p + "ffn1_pool.ln1.weight");
        B.b_ffn1_pool_ln1 = W.get(p + "ffn1_pool.ln1.bias");
        B.w_ffn1_pool_ln1_a = W.try_get(p + "ffn1_pool.ln1.weight.a");   // .a/.b split halves
        B.b_ffn1_pool_ln1_a = W.try_get(p + "ffn1_pool.ln1.bias.a");
        B.w_ffn1_pool_ln1_b = W.try_get(p + "ffn1_pool.ln1.weight.b");
        B.b_ffn1_pool_ln1_b = W.try_get(p + "ffn1_pool.ln1.bias.b");
        B.w_ffn1_pool_ln2 = W.get(p + "ffn1_pool.ln2.weight");
        B.b_ffn1_pool_ln2 = W.get(p + "ffn1_pool.ln2.bias");
        if (cfg.use_ls) {
            B.w_lay_scale_ffn1_x    = W.try_get(p + "lay_scale_ffn1_x.scale");
            B.w_lay_scale_ffn1_pool = W.try_get(p + "lay_scale_ffn1_pool.scale");
        }
    }
    if (B.has_ffn2) {
        B.w_norm_ffn2_x    = W.get(p + "norm_ffn2_x.weight");
        B.w_norm_ffn2_pool = W.get(p + "norm_ffn2_pool.weight");
        B.w_ffn2_x_ln1 = W.get(p + "ffn2_x.ln1.weight");
        B.b_ffn2_x_ln1 = W.get(p + "ffn2_x.ln1.bias");
        B.w_ffn2_x_ln1_a = W.try_get(p + "ffn2_x.ln1.weight.a");   // .a/.b split halves
        B.b_ffn2_x_ln1_a = W.try_get(p + "ffn2_x.ln1.bias.a");
        B.w_ffn2_x_ln1_b = W.try_get(p + "ffn2_x.ln1.weight.b");
        B.b_ffn2_x_ln1_b = W.try_get(p + "ffn2_x.ln1.bias.b");
        B.w_ffn2_x_ln2 = W.get(p + "ffn2_x.ln2.weight");
        B.b_ffn2_x_ln2 = W.get(p + "ffn2_x.ln2.bias");
        B.w_ffn2_pool_ln1 = W.get(p + "ffn2_pool.ln1.weight");
        B.b_ffn2_pool_ln1 = W.get(p + "ffn2_pool.ln1.bias");
        B.w_ffn2_pool_ln1_a = W.try_get(p + "ffn2_pool.ln1.weight.a");   // .a/.b split halves
        B.b_ffn2_pool_ln1_a = W.try_get(p + "ffn2_pool.ln1.bias.a");
        B.w_ffn2_pool_ln1_b = W.try_get(p + "ffn2_pool.ln1.weight.b");
        B.b_ffn2_pool_ln1_b = W.try_get(p + "ffn2_pool.ln1.bias.b");
        B.w_ffn2_pool_ln2 = W.get(p + "ffn2_pool.ln2.weight");
        B.b_ffn2_pool_ln2 = W.get(p + "ffn2_pool.ln2.bias");
        if (cfg.use_ls) {
            B.w_lay_scale_ffn2_x    = W.try_get(p + "lay_scale_ffn2_x.scale");
            B.w_lay_scale_ffn2_pool = W.try_get(p + "lay_scale_ffn2_pool.scale");
        }
    }
    if (cfg.use_ls) {
        B.w_lay_scale_jpac_x    = W.try_get(p + "lay_scale_jpac_x.scale");
        B.w_lay_scale_jpac_pool = W.try_get(p + "lay_scale_jpac_pool.scale");
    }

    // PJAC
    auto & P = B.pjac;
    P.jattn.w_pool_norm   = W.get(p + "attn.jattn.pool_norm.weight");
    P.jattn.w_x_norm      = W.get(p + "attn.jattn.x_norm.weight");
    P.jattn.w_pool_qkv    = W.get(p + "attn.jattn.pool_qkv.weight");
    P.jattn.b_pool_qkv    = W.get(p + "attn.jattn.pool_qkv.bias");
    P.jattn.w_x_qkv       = W.get(p + "attn.jattn.x_qkv.weight");
    P.jattn.b_x_qkv       = W.get(p + "attn.jattn.x_qkv.bias");
    if (cfg.qk_norm) {
        P.jattn.w_pool_q_norm = W.get(p + "attn.jattn.pool_q_norm.weight");
        P.jattn.w_pool_k_norm = W.get(p + "attn.jattn.pool_k_norm.weight");
        P.jattn.w_x_q_norm    = W.get(p + "attn.jattn.x_q_norm.weight");
        P.jattn.w_x_k_norm    = W.get(p + "attn.jattn.x_k_norm.weight");
    }
    P.jattn.w_pool_out = W.get(p + "attn.jattn.pool_out.weight");
    P.jattn.b_pool_out = W.get(p + "attn.jattn.pool_out.bias");
    P.jattn.w_x_out    = W.get(p + "attn.jattn.x_out.weight");
    P.jattn.b_x_out    = W.get(p + "attn.jattn.x_out.bias");

    // CgMLP (x + pool)
    P.w_c_norm_x     = W.get(p + "attn.c_norm_x.weight");
    P.w_c_norm_pool  = W.get(p + "attn.c_norm_pool.weight");
    P.w_cg_x_pw1   = W.get(p + "attn.c_x.pw1.weight");
    P.b_cg_x_pw1   = W.get(p + "attn.c_x.pw1.bias");
    P.w_cg_x_norm  = W.get(p + "attn.c_x.norm.weight");
    P.w_cg_x_dw    = W.get(p + "attn.c_x.dw.weight");
    P.b_cg_x_dw    = W.try_get(p + "attn.c_x.dw.bias");
    P.w_cg_x_pw2   = W.get(p + "attn.c_x.pw2.weight");
    P.b_cg_x_pw2   = W.get(p + "attn.c_x.pw2.bias");
    P.cg_x_kernel  = cfg.c_kernel_size_x;

    P.w_cg_pool_pw1  = W.get(p + "attn.c_pool.pw1.weight");
    P.b_cg_pool_pw1  = W.get(p + "attn.c_pool.pw1.bias");
    P.w_cg_pool_norm = W.get(p + "attn.c_pool.norm.weight");
    P.w_cg_pool_dw   = W.get(p + "attn.c_pool.dw.weight");
    P.b_cg_pool_dw   = W.try_get(p + "attn.c_pool.dw.bias");
    P.w_cg_pool_pw2  = W.get(p + "attn.c_pool.pw2.weight");
    P.b_cg_pool_pw2  = W.get(p + "attn.c_pool.pw2.bias");
    P.cg_pool_kernel = cfg.c_kernel_size_pool;

    // Merge
    P.w_merge_x  = W.get(p + "attn.merge_linear_x.weight");
    P.b_merge_x  = W.get(p + "attn.merge_linear_x.bias");
    P.merge_x_kernel = cfg.m_kernel_size_x;
    if (cfg.m_kernel_size_x != 0) {
        P.w_merge_dw_x = W.get(p + "attn.merge_dw_conv_x.weight");
        P.b_merge_dw_x = W.try_get(p + "attn.merge_dw_conv_x.bias");
    }
    P.w_merge_pool = W.get(p + "attn.merge_linear_pool.weight");
    P.b_merge_pool = W.get(p + "attn.merge_linear_pool.bias");
    P.merge_pool_kernel = cfg.m_kernel_size_pool;
    if (cfg.m_kernel_size_pool != 0) {
        P.w_merge_dw_pool = W.get(p + "attn.merge_dw_conv_pool.weight");
        P.b_merge_dw_pool = W.try_get(p + "attn.merge_dw_conv_pool.bias");
    }
    return B;
}

EstimatorWeights bind_estimator_weights(
    const LoadedWeights & W, const GameModelConfig & cfg)
{
    // Only the config branch used by GAME-pt-1.0-small is supported.
    if (cfg.estimator.ffn_type != "glu")
        throw NotImplemented("estimator ffn_type '" + cfg.estimator.ffn_type + "' not supported");
    if (cfg.estimator.attn_type != "joint")
        throw NotImplemented("estimator attn_type='" + cfg.estimator.attn_type + "' not supported (only 'joint')");
    if (cfg.estimator.rope_mode != "mixed")
        throw NotImplemented("estimator rope_mode='" + cfg.estimator.rope_mode + "' not supported (only 'mixed')");
    if (cfg.estimator.region_token_num != 1)
        throw NotImplemented("estimator region_token_num > 1 not supported");
    if (!cfg.estimator.qk_norm)
        throw NotImplemented("estimator without qk_norm not supported");
    if (cfg.estimator.use_region_bias)
        throw NotImplemented("estimator use_region_bias=true not supported");
    if (cfg.estimator.pool_merge_mode != "mean")
        throw NotImplemented("estimator pool_merge_mode='" + cfg.estimator.pool_merge_mode + "' not supported (only 'mean')");

    EstimatorWeights E{};
    E.w_input_proj     = W.get("estimator.input_proj.weight");
    E.b_input_proj     = W.get("estimator.input_proj.bias");
    E.w_pool_token_gen = W.get("estimator.pool_token_gen.emb");
    E.w_region_embedding = W.get("region_embedding.embedding.weight");

    E.layers.reserve(cfg.estimator.num_layers);
    for (int i = 0; i < cfg.estimator.num_layers; ++i) {
        E.layers.emplace_back(bind_jebf_layer(W, i, cfg.estimator));
    }

    if (cfg.estimator.use_out_norm) {
        E.w_output_norm_x    = W.get("estimator.output_norm_x.weight");
        E.w_output_norm_pool = W.get("estimator.output_norm_pool.weight");
    }
    E.w_output_proj_x    = W.get("estimator.output_proj_x.weight");
    E.b_output_proj_x    = W.get("estimator.output_proj_x.bias");
    E.w_output_proj_pool = W.get("estimator.output_proj_pool.weight");
    E.b_output_proj_pool = W.get("estimator.output_proj_pool.bias");
    return E;
}

EstimatorOutputs build_estimator_graph(
    ggml_context * ctx,
    ggml_tensor * x_est,
    ggml_tensor * regions_mod3,
    ggml_tensor * positions,
    ggml_tensor * region_indices,
    ggml_tensor * attn_mask_fp16,
    int N,
    const EstimatorWeights & W,
    const GameModelConfig & cfg)
{
    // x = x + region_embedding(regions % cycle)
    ggml_tensor * reg_emb = ops::embedding(ctx, W.w_region_embedding, regions_mod3);
    ggml_tensor * x = ggml_add(ctx, x_est, reg_emb);

    // input_proj
    x = ops::linear(ctx, x, W.w_input_proj, W.b_input_proj);

    // LearnablePoolTokens — for R=1 and B=1 inference with all regions valid,
    // this is simply `pool[n, :] = emb[0, :]` for n=0..N-1.
    // Weight shape (R=1, D) → ggml ne=(D, 1).  We replicate to (D, N, 1).
    const int64_t D = cfg.embedding_dim;
    ggml_tensor * pool_template = W.w_pool_token_gen;   // ne=(D, 1)
    // `ggml_repeat(a, b)` repeats a to match b's shape (broadcast).  We need
    // a (D, N, 1) shape.  Build a dummy tensor to represent the target shape.
    ggml_tensor * pool_shape_ref = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, N, 1);
    ggml_tensor * pool = ggml_repeat(ctx, pool_template, pool_shape_ref);

    // Run JEBF layers.  The last layer's x stream has no consumer (only
    // pool_logits is read), so run it pool-only: attention restricted to the
    // N pool query rows and no x-side FFN/CgMLP — numerically identical,
    // since the x rows were discarded anyway.
    for (int i = 0; i < cfg.estimator.num_layers; ++i) {
        const bool pool_only = (i == cfg.estimator.num_layers - 1);
        auto out = ops::jebf_block(ctx, pool, x, W.layers[i],
            positions, region_indices, attn_mask_fp16,
            cfg.estimator.num_heads, cfg.estimator.head_dim,
            cfg.estimator.theta, pool_only);
        pool = out.pool;
        x    = out.x;
    }

    // Output norm + proj for pool stream.
    if (W.w_output_norm_pool) pool = ops::rms_norm(ctx, pool, W.w_output_norm_pool);
    ggml_tensor * pool_logits = ops::linear(ctx, pool, W.w_output_proj_pool, W.b_output_proj_pool);

    EstimatorOutputs r{};
    r.pool_logits = ggml_cont(ctx, pool_logits);  // (bins, N, 1)
    return r;
}

}  // namespace game_ggml::internal
