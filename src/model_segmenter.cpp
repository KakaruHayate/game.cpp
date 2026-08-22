#include "model_segmenter.h"

#include "game_ggml/errors.h"
#include "ops_basic.h"
#include "ops_attn.h"
#include "ops_ffn.h"
#include "tensor_utils.h"

#include <ggml.h>

#include <cstring>
#include <string>

namespace game_ggml::internal {

// ---------------------------------------------------------------------------
// Bind weights
// ---------------------------------------------------------------------------

static ops::EBFBlockWeights bind_seg_layer(
    const LoadedWeights & W, int idx, const BackboneConfig & cfg)
{
    ops::EBFBlockWeights B{};
    const std::string p = "segmenter.layers." + std::to_string(idx) + ".";

    B.has_ffn1 = !cfg.skip_first_ffn;
    B.has_ffn2 = !cfg.skip_out_ffn;
    if (B.has_ffn1) {
        B.w_norm1    = W.get(p + "norm1.weight");
        B.w_ffn1_ln1 = W.get(p + "ffn1.ln1.weight");
        B.b_ffn1_ln1 = W.get(p + "ffn1.ln1.bias");
        B.w_ffn1_ln1_a = W.try_get(p + "ffn1.ln1.weight.a");   // .a/.b split halves
        B.b_ffn1_ln1_a = W.try_get(p + "ffn1.ln1.bias.a");
        B.w_ffn1_ln1_b = W.try_get(p + "ffn1.ln1.weight.b");
        B.b_ffn1_ln1_b = W.try_get(p + "ffn1.ln1.bias.b");
        B.w_ffn1_ln2 = W.get(p + "ffn1.ln2.weight");
        B.b_ffn1_ln2 = W.get(p + "ffn1.ln2.bias");
        if (cfg.use_ls) B.w_lay_scale1 = W.get(p + "lay_scale1.scale");
    }
    if (B.has_ffn2) {
        B.w_norm2    = W.get(p + "norm2.weight");
        B.w_ffn2_ln1 = W.get(p + "ffn2.ln1.weight");
        B.b_ffn2_ln1 = W.get(p + "ffn2.ln1.bias");
        B.w_ffn2_ln1_a = W.try_get(p + "ffn2.ln1.weight.a");   // .a/.b split halves
        B.b_ffn2_ln1_a = W.try_get(p + "ffn2.ln1.bias.a");
        B.w_ffn2_ln1_b = W.try_get(p + "ffn2.ln1.weight.b");
        B.b_ffn2_ln1_b = W.try_get(p + "ffn2.ln1.bias.b");
        B.w_ffn2_ln2 = W.get(p + "ffn2.ln2.weight");
        B.b_ffn2_ln2 = W.get(p + "ffn2.ln2.bias");
        if (cfg.use_ls) B.w_lay_scale3 = W.get(p + "lay_scale3.scale");
    }
    if (cfg.use_ls) B.w_lay_scale2 = W.get(p + "lay_scale2.scale");

    auto & P = B.pac_w;
    P.w_a_norm = W.get(p + "attn.a_norm.weight");
    P.w_c_norm = W.get(p + "attn.c_norm.weight");
    P.attn.w_qkv = W.get(p + "attn.attn.qkv.weight");
    P.attn.b_qkv = W.get(p + "attn.attn.qkv.bias");
    P.attn.w_out = W.get(p + "attn.attn.out_linear.weight");
    P.attn.b_out = W.get(p + "attn.attn.out_linear.bias");
    P.w_cg_pw1  = W.get(p + "attn.c.pw1.weight");
    P.b_cg_pw1  = W.get(p + "attn.c.pw1.bias");
    P.w_cg_norm = W.get(p + "attn.c.norm.weight");
    P.w_cg_dw   = W.get(p + "attn.c.dw.weight");
    P.b_cg_dw   = W.try_get(p + "attn.c.dw.bias");
    P.w_cg_pw2  = W.get(p + "attn.c.pw2.weight");
    P.b_cg_pw2  = W.get(p + "attn.c.pw2.bias");
    P.cg_kernel_size = cfg.c_kernel_size;
    P.w_merge_linear = W.get(p + "attn.merge_linear.weight");
    P.b_merge_linear = W.get(p + "attn.merge_linear.bias");
    P.merge_kernel_size = cfg.m_kernel_size;
    if (cfg.m_kernel_size != 0) {
        P.w_merge_dw = W.get(p + "attn.merge_dw_conv.weight");
        P.b_merge_dw = W.try_get(p + "attn.merge_dw_conv.bias");
    }
    return B;
}

SegmenterWeights bind_segmenter_weights(
    const LoadedWeights & W, const GameModelConfig & cfg)
{
    if (cfg.segmenter.ffn_type != "glu") {
        throw NotImplemented("segmenter ffn_type '" + cfg.segmenter.ffn_type + "' not supported");
    }

    SegmenterWeights S{};
    S.w_noise_embedding = W.get("noise_embedding.embedding.weight");
    if (cfg.use_languages) {
        S.w_lang_embedding = W.get("language_embedding.weight");
    }
    if (cfg.mode == "d3pm") {
        S.w_time_0 = W.get("time_embedding.0.weight");
        S.b_time_0 = W.get("time_embedding.0.bias");
        S.w_time_2 = W.get("time_embedding.2.weight");
        S.b_time_2 = W.get("time_embedding.2.bias");
    }

    S.w_input_proj = W.get("segmenter.input_proj.weight");
    S.b_input_proj = W.get("segmenter.input_proj.bias");
    S.layers.reserve(cfg.segmenter.num_layers);
    for (int i = 0; i < cfg.segmenter.num_layers; ++i) {
        S.layers.emplace_back(bind_seg_layer(W, i, cfg.segmenter));
    }
    if (cfg.segmenter.return_latent) {
        S.w_latent_norm = W.try_get("segmenter.latent_norm.weight");
        S.w_latent_proj = W.get("segmenter.latent_proj.weight");
        S.b_latent_proj = W.get("segmenter.latent_proj.bias");
    }
    if (cfg.segmenter.use_out_norm) {
        S.w_output_norm = W.get("segmenter.output_norm.weight");
    }
    S.w_output_proj = W.get("segmenter.output_proj.weight");
    S.b_output_proj = W.get("segmenter.output_proj.bias");
    return S;
}

// ---------------------------------------------------------------------------
// Build graph
// ---------------------------------------------------------------------------

// Embeddings + input_proj + front blocks[0..fn-1] -> x_front (D, T, 1).
ggml_tensor * build_segmenter_front_graph(
    ggml_context * ctx,
    ggml_tensor * x_seg,
    ggml_tensor * noise_mod3,
    ggml_tensor * t_scalar,
    ggml_tensor * lang_scalar,
    ggml_tensor * positions,
    const SegmenterWeights & W,
    const GameModelConfig & cfg,
    int fn_blocks)
{
    // ----- embedding injections (all [D, T, 1] additions) -----
    ggml_tensor * noise_emb = ops::embedding(ctx, W.w_noise_embedding, noise_mod3);
    ggml_tensor * x = ggml_add(ctx, x_seg, noise_emb);

    if (W.w_time_0 && t_scalar) {
        ggml_tensor * h = ops::linear(ctx, t_scalar, W.w_time_0, W.b_time_0);
        h = ggml_gelu(ctx, h);
        h = ops::linear(ctx, h, W.w_time_2, W.b_time_2);
        x = ggml_add(ctx, x, h);
    }

    if (W.w_lang_embedding && lang_scalar) {
        ggml_tensor * le = ops::embedding(ctx, W.w_lang_embedding, lang_scalar);
        x = ggml_add(ctx, x, le);
    }

    x = ops::linear(ctx, x, W.w_input_proj, W.b_input_proj);
    for (int i = 0; i < fn_blocks; ++i) {
        x = ops::ebf_block(ctx, x, W.layers[i], positions,
                           cfg.segmenter.num_heads, cfg.segmenter.head_dim);
    }
    return x;  // x_front
}

// Tail blocks[fn_blocks..N-1] (+ latent tap) -> (x_run, latent).
SegmenterTailOutputs build_segmenter_tail_graph(
    ggml_context * ctx,
    ggml_tensor * x_front,
    ggml_tensor * positions,
    const SegmenterWeights & W,
    const GameModelConfig & cfg,
    int fn_blocks,
    int end_blocks)
{
    if (end_blocks <= 0 || end_blocks > cfg.segmenter.num_layers) {
        end_blocks = cfg.segmenter.num_layers;
    }
    ggml_tensor * x = x_front;
    ggml_tensor * latent_tap = nullptr;
    // NB: no latent-tap guard here — sub-slices (DBCache middle/back ranges)
    // legitimately omit the tap, and the CLI path never consumes latent.
    // The fused build_segmenter_graph checks the contract instead.
    for (int i = fn_blocks; i < end_blocks; ++i) {
        x = ops::ebf_block(ctx, x, W.layers[i], positions,
                           cfg.segmenter.num_heads, cfg.segmenter.head_dim);
        if (cfg.segmenter.return_latent && i == cfg.segmenter.latent_layer_idx - 1) {
            latent_tap = x;
        }
    }

    SegmenterTailOutputs out{};
    out.x_run = x;
    if (cfg.segmenter.return_latent && latent_tap) {
        ggml_tensor * lt = latent_tap;
        if (W.w_latent_norm) lt = ops::rms_norm(ctx, lt, W.w_latent_norm);
        out.latent = ggml_cont(ctx, ops::linear(ctx, lt, W.w_latent_proj, W.b_latent_proj));
    }
    return out;
}

// output_norm + output_proj -> logits (T,).
ggml_tensor * build_segmenter_head_graph(
    ggml_context * ctx,
    ggml_tensor * x_out,
    const SegmenterWeights & W,
    const GameModelConfig & cfg)
{
    if (W.w_output_norm) x_out = ops::rms_norm(ctx, x_out, W.w_output_norm);
    ggml_tensor * logits = ops::linear(ctx, x_out, W.w_output_proj, W.b_output_proj);
    logits = ggml_cont(ctx, logits);
    return ggml_reshape_1d(ctx, logits, logits->ne[1]);
}

SegmenterOutputs build_segmenter_graph(
    ggml_context * ctx,
    ggml_tensor * x_seg,
    ggml_tensor * noise_mod3,
    ggml_tensor * t_scalar,
    ggml_tensor * lang_scalar,
    ggml_tensor * positions,
    const SegmenterWeights & W,
    const GameModelConfig & cfg)
{
    ggml_tensor * x_front = build_segmenter_front_graph(
        ctx, x_seg, noise_mod3, t_scalar, lang_scalar, positions, W, cfg,
        /*fn_blocks=*/0);

    SegmenterOutputs out{};
    SegmenterTailOutputs tail = build_segmenter_tail_graph(
        ctx, x_front, positions, W, cfg, /*fn_blocks=*/0);
    if (cfg.segmenter.return_latent && !tail.latent) {
        // latent_layer_idx is 1-based; must be a valid block index in the
        // fused full-stack pass.  Reject loudly instead of exposing null.
        throw Error("segmenter: return_latent enabled but latent_layer_idx "
                    "is outside [1, num_layers]");
    }
    out.latent = tail.latent;
    out.logits = build_segmenter_head_graph(ctx, tail.x_run, W, cfg);
    return out;
}

}  // namespace game_ggml::internal
