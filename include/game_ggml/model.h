#pragma once

// High-level Model API.  PIMPL-backed so that consumers never include any
// ggml headers transitively — the library can be linked into a larger
// project without leaking its internal tensor engine.

#include "game_ggml/config.h"
#include "game_ggml/types.h"

#include <cstddef>
#include <memory>
#include <string>

namespace game_ggml {

class Model {
public:
    // Load a GAME GGUF file produced by `scripts/convert_pt_to_gguf.py`.
    // Initialises the best available ggml backend (Metal → CUDA → Vulkan → CPU).
    // Throws `GgufError` / `BackendError` / `NotImplemented` on failure.
    static Model load(const std::string & gguf_path);

    ~Model();
    Model(Model &&) noexcept;
    Model & operator=(Model &&) noexcept;
    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;

    // Accessors for inspection.
    const GameModelConfig & config() const noexcept;

    // Run full end-to-end inference on a 44100 Hz mono waveform in [-1, 1].
    // Throws `InvalidArgument` if the sample count is below one frame worth.
    //
    // A single Model instance is NOT thread-safe (it reuses internal buffers);
    // hold multiple instances for parallel streams.
    InferResult infer(const float * waveform, std::size_t n_samples,
                      const InferParams & params);

    // Batched inference — currently a thin wrapper that processes each item
    // sequentially with the given params (same RNG seed policy as infer, one
    // rng per item so results equal the per-item infer calls).  Intended to be
    // replaced by a real [B,T] fused path (see docs/batch-inference-design.md).
    BatchResult infer_batch(const std::vector<BatchItem> & items,
                            const InferParams & params);

    // Forward declaration of the internal state — kept opaque to consumers
    // so the public ABI doesn't depend on ggml's layout.
    struct Impl;

    // Escape hatch for the test suite: access the internal implementation.
    // Not part of the stable API — safe for use by code in this repo only.
    Impl & internals() noexcept;

private:
    Model();
    std::unique_ptr<Impl> impl_;
};

}  // namespace game_ggml
