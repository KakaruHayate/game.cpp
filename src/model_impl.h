#pragma once

// Full definition of Model::Impl, only visible to code in this repo.
// Tests include this header directly so they can inject a deterministic
// RNG into the pipeline.

#include "game_ggml/model.h"
#include "game_ggml/mel.h"

#include <ggml-alloc.h>
#include <ggml.h>

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

    // Device-side cache state (see run_segmenter_step): the decision metric
    // and the delta reconstruction live on the backend, so GPU EPs do not pay
    // D×T host round-trips per D3PM step.  `valid` mirrors "prev_front is
    // meaningful" — set after the first full pass.
    bool valid = false;              // tail_delta / prev_front available

    void reset() {
        step = 0;
        hits = 0;
        misses = 0;
        acc_err = 0.0f;
        cont_cnt = 0;
        valid = false;
    }
};

// Persistent (stage, T) compute context.  The graph + gallocr are built once
// per frame count T and reused across the nsteps D3PM iterations of one
// audio segment (and across chunks with identical T), eliminating per-step
// graph construction, gallocr allocation and — on GPU backends — the
// device-buffer alloc/free that goes with it.  Rebuilding happens only when
// T (or the caller's stage split parameters) change.
struct PersistentStage {
    ggml_context * ctx   = nullptr;
    ggml_cgraph  * graph = nullptr;
    ggml_gallocr_t alloc = nullptr;
    int T = 0;                        // frame count this graph was built for
    int64_t key = 0;                  // stage-specific discriminator (e.g. block range)

    // Tensors created in ctx that the caller refreshes via tensor_set before
    // each compute (input tensors).  The first entry of `outs` is the stage's
    // primary output (logits etc.).
    std::vector<ggml_tensor *> inputs;
    std::vector<ggml_tensor *> outs;

    bool matches(int t, int64_t k = 0) const { return graph != nullptr && T == t && key == k; }

    void reset() {
        if (alloc) ggml_gallocr_free(alloc);
        if (ctx)   ggml_free(ctx);
        alloc = nullptr; ctx = nullptr; graph = nullptr;
        inputs.clear(); outs.clear();
        T = 0; key = 0;
    }
    ~PersistentStage() { reset(); }
    PersistentStage() = default;
    PersistentStage(const PersistentStage &) = delete;
    PersistentStage & operator=(const PersistentStage &) = delete;
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

    // Persistent stage graphs (rebuilt lazily when T changes).  The encoder
    // stage also owns the device-resident x_seg / x_est output tensors that
    // the segmenter / estimator stages reference directly (no host round-trip
    // between stages on GPU backends).
    PersistentStage enc_stage;        // encoder: mel -> x_seg, x_est (device-resident)
    PersistentStage seg_stage;        // segmenter fused path (DBCache disabled)
    PersistentStage seg_front_stage;  // segmenter DBCache: front blocks (+ fd metric)
    PersistentStage seg_add_stage;    // segmenter DBCache: x_mid = x_front + tail_delta (hit)
    PersistentStage seg_mid_stage;    // segmenter DBCache: middle tail range (miss)
    PersistentStage seg_update_stage; // segmenter DBCache: refresh tail_delta/prev_front
    PersistentStage seg_back_stage;   // segmenter DBCache: back tail range
    PersistentStage seg_head_stage;   // segmenter DBCache: output norm + proj
    PersistentStage est_stage;        // estimator

    // Device-resident cross-stage activations — persistent NONE tensors in
    // `dbctx`/`dbbuf`, written by the encoder (x_seg/x_est) or the front
    // graph (x_front) via ggml_cpy.  Being NONE (not op nodes), downstream
    // graphs reference them as pure leaves: ggml never pulls the producer
    // chain into a downstream graph, which keeps each stage graph at its own
    // size (a CONT-view leaf would drag the entire encoder/front chain into
    // every consumer graph and re-compute it).
    ggml_tensor * x_seg_dev = nullptr;
    ggml_tensor * x_est_dev = nullptr;

    // Device-resident DBCache tensors (D×T each, rebuilt when T changes):
    //   x_front_dev    — front-graph output (metric reference + reuse base)
    //   prev_front_dev — x_front of the previous full pass (metric reference)
    //   tail_delta_dev — x_mid - x_front of the previous full pass (hit reuse)
    //   x_mid_dev      — reconstructed/forwarded middle (back-graph input)
    //   x_out_dev      — tail output (head-graph input)
    ggml_context * dbctx = nullptr;
    ggml_backend_buffer_t dbbuf = nullptr;
    ggml_tensor * x_front_dev = nullptr;
    ggml_tensor * prev_front_dev = nullptr;
    ggml_tensor * tail_delta_dev = nullptr;
    ggml_tensor * x_mid_dev = nullptr;
    ggml_tensor * x_out_dev = nullptr;
    int db_T = 0;

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
    void run_encoder(const float * mel, int T);

    void run_segmenter_step(
        int T,
        const std::int32_t * noise_mod3, float t_scalar, int language,
        std::vector<float> & logits_out);

    void run_estimator(
        int T, const std::int32_t * regions, int N,
        std::vector<float> & pool_logits_out);

    // (Re)allocate the device-resident DBCache tensors for frame count T.
    void ensure_db_tensors(int T);
};

}  // namespace game_ggml
