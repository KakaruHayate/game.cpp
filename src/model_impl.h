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
struct ggml_cgraph;
struct ggml_gallocr;
typedef struct ggml_backend * ggml_backend_t;
typedef struct ggml_gallocr * ggml_gallocr_t;

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

    // Robustness knobs (edge-dit.cpp / CacheDiT borrowings).
    int   total_steps      = 0;      // D3PM loop length (for the reuse window)
    float window_start     = 0.0f;   // cache only when step/total in [..,]
    float window_end       = 1.0f;
    float err_decay        = 0.0f;   // >0: UCache-style accumulated-error gate
    float err_limit        = 0.5f;
    int   max_cont         = 0;      // 0 = unlimited consecutive hits
    int   bn_blocks        = 0;      // always recompute this many tail blocks on hit

    int   step  = 0;
    int   hits  = 0;
    int   misses = 0;
    float acc_err  = 0.0f;           // accumulated skipped delta error
    int   cont_cnt = 0;              // consecutive hits

    std::vector<float> prev_front;   // x_front of the previous step
    std::vector<float> tail_delta;   // x_out - x_front of the previous full pass
    bool valid = false;              // tail_delta / prev_front available

    void reset() {
        step = 0;
        hits = 0;
        misses = 0;
        acc_err = 0.0f;
        cont_cnt = 0;
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

    // Reusable batched-segmenter graph (avoids rebuilding a 512 MB StageCtx
    // every D3PM step — the expensive part of the batch path).  Built once for
    // a given (T,B); freed in ~Impl.  Not thread-safe (single Model + single
    // batch loop), matching the rest of Impl.
    struct BatchSegGraph {
        ggml_context *  ctx   = nullptr;
        ggml_cgraph  *  graph = nullptr;
        ggml_gallocr_t  alloc = nullptr;
        int T = 0, B = 0;
        ggml_tensor * xseg      = nullptr;
        ggml_tensor * noise     = nullptr;
        ggml_tensor * t_tensor  = nullptr;
        ggml_tensor * lang      = nullptr;
        ggml_tensor * positions = nullptr;
        ggml_tensor * logits    = nullptr;
    } batch_seg;
    void release_batch_seg();
    void dump_batch_seg_ops();

    ~Impl();

    // Load + bind.
    static std::unique_ptr<Impl> load(const std::string & path);

    // Expose the mel front-end frame count for a given waveform length
    // (used by the CLI to group equal-length slices before batching).
    int frames_for(std::size_t n_samples) const {
        return mel_extractor->num_frames(n_samples);
    }

    // Main entry — used both by Model::infer (wraps its own MT19937Rng) and
    // by the test suite (passes an InjectedRng).
    InferResult infer_with_rng(
        const float * waveform, std::size_t n_samples,
        const InferParams & params,
        internal::IRandomSource & rng);

    // Batched entry.  When all items share the same mel frame count it runs a
    // fused multi-sample encoder (`[B,T]` graph) and then per-sample
    // segmenter/estimator on the shared encoding; otherwise it falls back to
    // per-item infer_with_rng (the current sequential baseline).
    std::vector<InferResult> infer_batch_impl(
        const std::vector<BatchItem> & items, const InferParams & params,
        std::vector<std::uint64_t> given_seeds /* empty => derive */);

private:
    // Pipeline stages (single-sample, existing path).
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

    // D3PM loop + estimator + decode given pre-computed encoder latents for a
    // single sample (shared by infer_with_rng and the batched path).
    InferResult infer_from_latent(
        const float * x_seg, const float * x_est, int T,
        const InferParams & params, internal::IRandomSource & rng);

    // Pipeline stages (batched).  `B` tensors processed together in one graph.
    //   mel      : (D_mel, T, B) — pre-padded, uploaded by the caller
    //   mask     : (T, B) uint8  — per-frame validity (0=pad).  Drains through
    //              the whole encoder as `masked_fill` semantics.
    void run_encoder_batch(const float * mel, int T, int B, const std::uint8_t * mask,
                           std::vector<float> & x_seg_out,   // (D, T*B)
                           std::vector<float> & x_est_out);  // (D, T*B)

    // Segmenter front+head for every sample in the batch.
    void run_segmenter_batch(const float * x_seg_host, int T, int B,
                             const std::int32_t * noise_mod3,  // (T, B)
                             float t_scalar, const int * language,  // (B)
                             std::vector<float> & logits_out);  // (T, B)

    void run_estimator_batch(const float * x_est_host, int T, int B,
                             const std::int32_t * regions,  // (T, B)
                             const int * Ns,                // (B) valid note count
                             int N_max,
                             std::vector<float> & pool_logits_out);  // flat (bins * N_max * B)
};

}  // namespace game_ggml
