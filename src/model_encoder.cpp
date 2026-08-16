#include "model_encoder.h"

#include "ops_basic.h"
#include "ops_attn.h"
#include "tensor_utils.h"
#include "game_ggml/errors.h"

#include <ggml.h>

#include <sstream>

namespace game_ggml::internal {

// ---------------------------------------------------------------------------
// Bind encoder weights
// ---------------------------------------------------------------------------

static ops::EBFBlockWeights bind_ebf_layer(
    const LoadedWeights & W,
    const std::string & prefix,
    int layer_idx,
    const BackboneConfig & cfg)
{
    ops::EBFBlockWeights B{};
    const std::string p = prefix + ".layers." + std::to_string(layer_idx) + ".";

    B.has_ffn1 = !cfg.skip_first_ffn;
    B.has_ffn2 = !cfg.skip_out_ffn;

    if (B.has_ffn1) {
        B.w_norm1    = W.get(p + "norm1.weight");
        B.w_ffn1_ln1 = W.get(p + "ffn1.ln1.weight");
        B.b_ffn1_ln1 = W.get(p + "ffn1.ln1.bias");
        B.w_ffn1_ln2 = W.get(p + "ffn1.ln2.weight");
        B.b_ffn1_ln2 = W.get(p + "ffn1.ln2.bias");
        if (cfg.use_ls) B.w_lay_scale1 = W.get(p + "lay_scale1.scale");
    }
    if (B.has_ffn2) {
        B.w_norm2    = W.get(p + "norm2.weight");
        B.w_ffn2_ln1 = W.get(p + "ffn2.ln1.weight");
        B.b_ffn2_ln1 = W.get(p + "ffn2.ln1.bias");
        B.w_ffn2_ln2 = W.get(p + "ffn2.ln2.weight");
        B.b_ffn2_ln2 = W.get(p + "ffn2.ln2.bias");
        if (cfg.use_ls) B.w_lay_scale3 = W.get(p + "lay_scale3.scale");
    }
    if (cfg.use_ls) B.w_lay_scale2 = W.get(p + "lay_scale2.scale");

    // PAC
    auto & P = B.pac_w;
    P.w_a_norm = W.get(p + "attn.a_norm.weight");
    P.w_c_norm = W.get(p + "attn.c_norm.weight");

    // Attention sub-weights (q/kv fused into one qkv tensor by the converter)
    P.attn.w_qkv = W.get(p + "attn.attn.qkv.weight");
    P.attn.b_qkv = W.get(p + "attn.attn.qkv.bias");
    P.attn.w_out = W.get(p + "attn.attn.out_linear.weight");
    P.attn.b_out = W.get(p + "attn.attn.out_linear.bias");

    // CgMLP sub-weights
    P.w_cg_pw1  = W.get(p + "attn.c.pw1.weight");
    P.b_cg_pw1  = W.get(p + "attn.c.pw1.bias");
    P.w_cg_norm = W.get(p + "attn.c.norm.weight");
    P.w_cg_dw   = W.get(p + "attn.c.dw.weight");
    P.b_cg_dw   = W.try_get(p + "attn.c.dw.bias");
    P.w_cg_pw2  = W.get(p + "attn.c.pw2.weight");
    P.b_cg_pw2  = W.get(p + "attn.c.pw2.bias");
    P.cg_kernel_size = cfg.c_kernel_size;

    // Merge
    P.w_merge_linear = W.get(p + "attn.merge_linear.weight");
    P.b_merge_linear = W.get(p + "attn.merge_linear.bias");
    P.merge_kernel_size = cfg.m_kernel_size;
    if (cfg.m_kernel_size != 0) {
        P.w_merge_dw = W.get(p + "attn.merge_dw_conv.weight");
        P.b_merge_dw = W.try_get(p + "attn.merge_dw_conv.bias");
    }

    return B;
}

EncoderWeights bind_encoder_weights(
    const LoadedWeights & W,
    const BackboneConfig & cfg,
    const std::string & prefix)
{
    if (cfg.ffn_type != "glu") {
        throw NotImplemented("encoder ffn_type='" + cfg.ffn_type + "' not supported (only 'glu' for v1)");
    }

    EncoderWeights E{};
    E.w_input_proj  = W.get(prefix + ".input_proj.weight");
    E.b_input_proj  = W.get(prefix + ".input_proj.bias");

    E.layers.reserve(cfg.num_layers);
    for (int i = 0; i < cfg.num_layers; ++i) {
        E.layers.emplace_back(bind_ebf_layer(W, prefix, i, cfg));
    }

    if (cfg.use_out_norm) {
        E.w_output_norm = W.get(prefix + ".output_norm.weight");
    }
    E.w_output_proj = W.get(prefix + ".output_proj.weight");
    E.b_output_proj = W.get(prefix + ".output_proj.bias");
    return E;
}

// ---------------------------------------------------------------------------
// Build encoder graph
// ---------------------------------------------------------------------------

EncoderOutputs build_encoder_graph(
    ggml_context * ctx,
    ggml_tensor * x_in,
    const EncoderWeights & W,
    ggml_tensor * positions,
    const BackboneConfig & cfg,
    std::vector<ggml_tensor *> * intermediates)
{
    // Input projection.  x_in has ne=(D_embed, T, 1).
    ggml_tensor * x = ops::linear(ctx, x_in, W.w_input_proj, W.b_input_proj);
    if (intermediates) intermediates->push_back(x);

    for (int i = 0; i < cfg.num_layers; ++i) {
        x = ops::ebf_block(ctx, x, W.layers[i], positions,
                           cfg.num_heads, cfg.head_dim);
        if (intermediates) intermediates->push_back(x);
    }

    if (W.w_output_norm) {
        x = ops::rms_norm(ctx, x, W.w_output_norm);
        if (intermediates) intermediates->push_back(x);
    }
    // Output proj maps D_embed → 2*D_embed.
    ggml_tensor * full = ops::linear(ctx, x, W.w_output_proj, W.b_output_proj);
    if (intermediates) intermediates->push_back(full);

    // Split along ne[0] into two contiguous halves.
    const int64_t D_embed = full->ne[0] / 2;
    const int64_t T  = full->ne[1];
    const int64_t B  = full->ne[2];
    const size_t esize = ggml_element_size(full);
    ggml_tensor * full_c = ggml_cont(ctx, full);

    EncoderOutputs out;
    ggml_tensor * seg_view = ggml_view_3d(ctx, full_c,
        D_embed, T, B,
        full_c->nb[1], full_c->nb[2], /*offset=*/0);
    ggml_tensor * est_view = ggml_view_3d(ctx, full_c,
        D_embed, T, B,
        full_c->nb[1], full_c->nb[2], /*offset=*/D_embed * esize);
    // Contiguate so consumers can read via `ggml_backend_tensor_get` without
    // spilling across the strided view boundary.
    out.x_seg = ggml_cont(ctx, seg_view);
    out.x_est = ggml_cont(ctx, est_view);
    return out;
}

}  // namespace game_ggml::internal
