#pragma once

// EBFBackbone (encoder branch) — wraps the input projection, stack of EBF
// blocks, optional output norm, and output projection.  Outputs a 2D-split
// (x_seg, x_est) pair.
//
// Internal use only (not exposed via include/game_ggml/).

#include "game_ggml/config.h"
#include "ops_attn.h"

#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace game_ggml::internal {

class LoadedWeights;

// All encoder weight pointers, indexed from `LoadedWeights` at load time.
struct EncoderWeights {
    ggml_tensor * w_input_proj  = nullptr;
    ggml_tensor * b_input_proj  = nullptr;
    std::vector<ops::EBFBlockWeights> layers;
    ggml_tensor * w_output_norm = nullptr;     // optional (use_out_norm)
    ggml_tensor * w_output_proj = nullptr;
    ggml_tensor * b_output_proj = nullptr;
};

// Pull encoder tensors from a `LoadedWeights` bundle, validate shapes, and
// return a populated weights struct.  Throws GgufError or NotImplemented if
// the config / tensor set isn't supported.
EncoderWeights bind_encoder_weights(
    const LoadedWeights & W,
    const BackboneConfig & cfg,
    const std::string & prefix = "encoder");

// Build the encoder graph: input mel (ne=(D_mel, T, 1)) after the
// top-level spectrogram_projection is fed in; the encoder does its own
// dim-matching projection and produces (2*D_embed, T, 1), which we then
// split into x_seg and x_est each (D_embed, T, 1).
struct EncoderOutputs {
    ggml_tensor * x_seg;   // (D_embed, T, 1)
    ggml_tensor * x_est;   // (D_embed, T, 1)
};

EncoderOutputs build_encoder_graph(
    ggml_context * ctx,
    ggml_tensor * x_in,                 // (D_embed, T, 1) after spectrogram_projection
    const EncoderWeights & W,
    ggml_tensor * positions,            // int32 (T,)
    const BackboneConfig & cfg,
    ggml_tensor * mask = nullptr,       // optional (1, T, B) f32; 0 = padding frame
    std::vector<ggml_tensor *> * intermediates = nullptr);

}  // namespace game_ggml::internal
