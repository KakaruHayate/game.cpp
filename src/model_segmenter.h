#pragma once

// Segmenter sub-module — an EBFBackbone(8 layers, return_latent) with
// noise / time / language embeddings injected on the input side.
//
// Top-level (from GAME's midi_extraction.py):
//   x = x + noise_embedding(noise)
//   if mode == 'd3pm': x = x + time_embedding(t[..., None, None])
//   if use_languages:  x = x + language_embedding(language.unsqueeze(-1))
//   x, latent = self.segmenter(x, mask=mask)
//   x = x.squeeze(-1)   # -> (B, T) boundary logits

#include "game_ggml/config.h"
#include "ops_attn.h"

#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace game_ggml::internal {

class LoadedWeights;

struct SegmenterWeights {
    // Top-level embedding weights (outside the "segmenter." prefix).
    ggml_tensor * w_noise_embedding    = nullptr;   // (cycle_len, D)
    ggml_tensor * w_lang_embedding     = nullptr;   // (num_languages+1, D)  — nullable when use_languages=false
    ggml_tensor * w_time_0             = nullptr;   // Linear(1, 4D)
    ggml_tensor * b_time_0             = nullptr;
    ggml_tensor * w_time_2             = nullptr;   // Linear(4D, D)
    ggml_tensor * b_time_2             = nullptr;

    // Backbone (same shape as EncoderWeights + latent branch).
    ggml_tensor * w_input_proj         = nullptr;
    ggml_tensor * b_input_proj         = nullptr;
    std::vector<ops::EBFBlockWeights> layers;
    ggml_tensor * w_latent_norm        = nullptr;   // optional
    ggml_tensor * w_latent_proj        = nullptr;
    ggml_tensor * b_latent_proj        = nullptr;
    ggml_tensor * w_output_norm        = nullptr;   // optional
    ggml_tensor * w_output_proj        = nullptr;   // D -> 1
    ggml_tensor * b_output_proj        = nullptr;
};

SegmenterWeights bind_segmenter_weights(
    const LoadedWeights & W,
    const GameModelConfig & cfg);

// Graph outputs.
struct SegmenterOutputs {
    ggml_tensor * logits;   // (T,)  float32 — pre-sigmoid boundary score
    ggml_tensor * latent;   // (latent_out_dim, T, 1) — nullable if return_latent=false
};

// Tail-stack outputs (DBCache).
struct SegmenterTailOutputs {
    ggml_tensor * x_run;    // (D, T, 1) — full block-stack output
    ggml_tensor * latent;   // nullable
};

// DBCache-friendly graph split: the segmenter block stack is split at
// `fn_blocks` into a front part (always executed) and a tail part (skipped on
// cache hits).  These three builders mirror the PyTorch DBCacheSegmenter
// (input_head / run_front / run_tail / output_head).
//
// front: embeddings + input_proj + blocks[0..fn-1] -> x_front (D, T, 1)
ggml_tensor * build_segmenter_front_graph(
    ggml_context * ctx,
    ggml_tensor * x_seg,
    ggml_tensor * noise_mod3,
    ggml_tensor * t_scalar,
    ggml_tensor * lang_scalar,
    ggml_tensor * positions,
    const SegmenterWeights & W,
    const GameModelConfig & cfg,
    int fn_blocks);

// tail: blocks[fn_blocks..N-1] (+ latent tap) -> (x_run, latent)
SegmenterTailOutputs build_segmenter_tail_graph(
    ggml_context * ctx,
    ggml_tensor * x_front,
    ggml_tensor * positions,
    const SegmenterWeights & W,
    const GameModelConfig & cfg,
    int fn_blocks,
    int end_blocks = -1);  // exclusive end (default: num_layers); enables a
                           // "back" slice for DBCache bn-blocks

// head: output_norm + output_proj -> logits (T,)
ggml_tensor * build_segmenter_head_graph(
    ggml_context * ctx,
    ggml_tensor * x_out,
    const SegmenterWeights & W,
    const GameModelConfig & cfg);

// Build the segmenter graph.
//   x_seg        : (D, T, 1)          — the `x_seg` split of the encoder output
//   noise_mod3   : int32 (T,)         — `noise_regions % region_cycle_len`
//   t_scalar     : (1, 1, 1)          — float32 scalar wrapped as a 3D tensor so
//                                       the time_embedding Linear works.  Set to nullptr
//                                       to skip the time-embedding addition.
//   lang_scalar  : int32 (1,)         — language id.  Set to nullptr to skip.
//   positions    : int32 (T,)         — global frame positions for RoPE.
SegmenterOutputs build_segmenter_graph(
    ggml_context * ctx,
    ggml_tensor * x_seg,
    ggml_tensor * noise_mod3,
    ggml_tensor * t_scalar,
    ggml_tensor * lang_scalar,
    ggml_tensor * positions,
    const SegmenterWeights & W,
    const GameModelConfig & cfg);

}  // namespace game_ggml::internal
