#pragma once

// Full definition of Model::Impl, only visible to code in this repo.
// Tests include this header directly so they can inject a deterministic
// RNG into the pipeline.

#include "game_ggml/model.h"
#include "game_ggml/mel.h"

#include "gguf_io.h"
#include "model_encoder.h"
#include "model_segmenter.h"
#include "model_estimator.h"
#include "tensor_utils.h"

#include <memory>

struct ggml_backend;
struct ggml_tensor;
typedef struct ggml_backend * ggml_backend_t;

namespace game_ggml::internal { class IRandomSource; }

namespace game_ggml {

// Cross-step cache state for the segmenter (DBCache).  Lives on the Impl so
// it persists across the nsteps D3PM iterations of one audio segment; reset
// at the start of every segment.
struct SegmenterCacheState {
    bool enabled = false;
    float threshold = 0.25f;
    int   fn_blocks = 1;
    int   warmup    = 1;

    int   step  = 0;
    int   hits  = 0;
    int   misses = 0;

    std::vector<float> prev_front;   // x_front of the previous step
    std::vector<float> tail_delta;   // x_out - x_front of the previous full pass
    bool valid = false;              // tail_delta / prev_front available

    void reset() {
        step = 0;
        hits = 0;
        misses = 0;
        prev_front.clear();
        tail_delta.clear();
        valid = false;
    }
};

struct Model::Impl {
    GameModelConfig cfg;
    ggml_backend_t  backend = nullptr;

    // Weight storage (owns the ggml context + backend buffer).
    std::unique_ptr<internal::GgufFile>      gguf;
    std::unique_ptr<internal::LoadedWeights> weights;

    // Front-end.
    std::unique_ptr<MelExtractor> mel_extractor;

    // DBCache state for the segmenter (cross-D3PM-step).
    SegmenterCacheState seg_cache;

    // Top-level weights outside the three sub-models.
    ggml_tensor * w_spec_proj = nullptr;
    ggml_tensor * b_spec_proj = nullptr;

    internal::EncoderWeights   encoder_w;
    internal::SegmenterWeights segmenter_w;
    internal::EstimatorWeights estimator_w;

    ~Impl();

    // Load + bind.
    static std::unique_ptr<Impl> load(const std::string & path);

    // Main entry — used both by Model::infer (wraps its own MT19937Rng) and
    // by the test suite (passes an InjectedRng).
    InferResult infer_with_rng(
        const float * waveform, std::size_t n_samples,
        const InferParams & params,
        internal::IRandomSource & rng);

private:
    // Pipeline stages.
    void run_encoder(const float * mel, int T,
                     std::vector<float> & x_seg_out,
                     std::vector<float> & x_est_out);

    void run_segmenter_step(
        const float * x_seg_host, int T,
        const std::int32_t * noise_mod3, float t_scalar, int language,
        std::vector<float> & logits_out);

    void run_estimator(
        const float * x_est_host, int T,
        const std::int32_t * regions, int N,
        std::vector<float> & pool_logits_out);
};

}  // namespace game_ggml
