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
    int   db_cache_fn_blocks  = 1;      // front (warmup) blocks always executed
    int   db_cache_warmup     = 1;      // full passes before hits are allowed
};

// Full-clip inference result.
struct InferResult {
    std::vector<Note> notes;

    // Number of mel frames processed (for diagnostics).
    int num_frames = 0;
};

}  // namespace game_ggml
