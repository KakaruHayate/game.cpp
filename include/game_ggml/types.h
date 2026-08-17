#pragma once

// Public PODs used by the Model::infer entry point.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace game_ggml {

// One transcribed note.
struct Note {
    float offset_seconds;       // start time (seconds from waveform origin)
    float duration_seconds;
    float pitch_midi;           // fractional MIDI number, meaningful iff voiced
    bool  voiced;               // false → unvoiced / rest
};

// Inference-time knobs; all fields have sensible defaults.
struct InferParams {
    // Language id from the model's GGUF lang_map.  0 = unknown / universal.
    int language = 0;

    // Custom D3PM time schedule.  Empty vector → default schedule:
    //   ts = [t0, t0 + dt, t0 + 2*dt, ..., 1 - dt]   with  dt = (1 - t0) / nsteps.
    std::vector<float> d3pm_ts;
    float              d3pm_t0     = 0.0f;
    int                d3pm_nsteps = 1;   // single-shot denoising; 8 for higher quality

    // Boundary decoding.
    float boundary_threshold = 0.2f;
    int   boundary_radius    = 2;      // frames

    // Note presence gate (on sigmoid'd estimator logits).
    float note_threshold = 0.2f;

    // Seed for the internal Mersenne Twister that drives the D3PM random
    // boundary removal.  Set to a fixed value for reproducible runs.
    std::uint64_t seed = 0;

    // DBCache (cache-dit style) for the segmenter across D3PM steps.
    // When enabled, the tail block stack is skipped on steps whose
    // front-block residual delta falls below `db_cache_threshold`.
    //   -1 = auto: CPU 0.25, GPU backends (vulkan/metal/cuda) 0 (off) — the
    //        GPU host round-trips of the split path regress quantized
    //        weights, so default off there
    //    0 = disabled;  >0 = explicit threshold (0.25 recommended)
    float db_cache_threshold  = -1.0f;
    int   db_cache_fn_blocks  = 1;      // front blocks always executed
    int   db_cache_warmup     = 1;      // full passes before hits are allowed
    // Robustness knobs (edge-dit.cpp / CacheDiT borrowings), default = current
    // behavior unless set:
    //   reuse window: only cache steps whose fraction in [start, end] of the
    //     loop; first/last steps stay full-compute (safest).
    float db_cache_window_start = 0.0f;
    float db_cache_window_end   = 1.0f;
    //   UCache-style accumulated-error decay (>0 enables): skipped deltas are
    //     accumulated (scaled by `decay` each step) and a step is computed in
    //     full once the accumulated error exceeds `db_cache_err_limit`.
    float db_cache_err_decay   = 0.0f;   // 0 = disabled (pure per-step threshold)
    float db_cache_err_limit   = 0.5f;   // only used when err_decay > 0
    //   maximum consecutive cached steps (0 = unlimited).
    int   db_cache_max_cont    = 0;
    //   back blocks always executed: on a hit, recompute the last N blocks on
    //     top of the reconstructed x (more accurate near the output).
    int   db_cache_bn_blocks   = 0;
};

// Full-clip inference result.
struct InferResult {
    std::vector<Note> notes;

    // Number of mel frames processed (for diagnostics).
    int num_frames = 0;

    // DBCache hit/miss counters for this inference (segments summed).
    // With the cache disabled (nsteps==1 or threshold==0) both stay 0.
    int db_cache_hits   = 0;
    int db_cache_misses = 0;
};

// One audio clip queued into a batched inference (mirrors infer.py's
// SlicedAudioFileIterableDataset collate input).
struct BatchItem {
    const float * waveform = nullptr;   // 44100 Hz mono, [-1, 1]
    std::size_t   n_samples = 0;
    int           language = 0;
};

// Batched inference outcome — one result per input item, in order.
struct BatchResult {
    std::vector<InferResult> items;
};

}  // namespace game_ggml
