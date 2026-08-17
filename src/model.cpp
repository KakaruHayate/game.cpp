#include "model_impl.h"

#include "backend.h"
#include "d3pm.h"
#include "game_ggml/decode.h"
#include "game_ggml/errors.h"
#include "ops_basic.h"
#include "ops_joint_attn.h"
#include "rng.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace game_ggml {

// ============================================================================
// Model — public facade
// ============================================================================

Model::Model() = default;
Model::~Model() = default;
Model::Model(Model &&) noexcept = default;
Model & Model::operator=(Model &&) noexcept = default;

Model Model::load(const std::string & gguf_path) {
    Model m;
    m.impl_ = Impl::load(gguf_path);
    return m;
}

const GameModelConfig & Model::config() const noexcept { return impl_->cfg; }
Model::Impl & Model::internals() noexcept { return *impl_; }

InferResult Model::infer(const float * waveform, std::size_t n_samples,
                         const InferParams & params) {
    std::uint64_t seed = params.seed;
    if (seed == 0) seed = std::random_device{}();
    internal::MT19937Rng rng(seed);
    return impl_->infer_with_rng(waveform, n_samples, params, rng);
}

BatchResult Model::infer_batch(const std::vector<BatchItem> & items,
                               const InferParams & params) {
    std::uint64_t base = params.seed;
    if (base == 0) base = std::random_device{}();
    return BatchResult{ impl_->infer_batch_impl(items, params, {}) };
}

// ============================================================================
// Model::Impl — loading
// ============================================================================

Model::Impl::~Impl() {
    release_batch_seg();
    if (backend) internal::free_backend(backend);
}

void Model::Impl::release_batch_seg() {
    if (batch_seg.alloc) ggml_gallocr_free(batch_seg.alloc);
    if (batch_seg.ctx)   ggml_free(batch_seg.ctx);
    batch_seg = {};
}

std::unique_ptr<Model::Impl> Model::Impl::load(const std::string & path) {
    auto impl = std::make_unique<Impl>();
    impl->gguf    = std::make_unique<internal::GgufFile>(internal::GgufFile::open(path));
    impl->cfg     = internal::load_config(*impl->gguf);
    impl->backend = internal::init_best_backend();

    // Depthwise convs: use the dedicated per-channel GGML_OP_CONV_2D_DW kernel
    // (no im2col) on every backend that implements it — CPU, Vulkan, Metal and
    // CUDA all support it with an F32 kernel (our direct path casts the GGUF
    // F16/F32 depthwise weight to F32).  Other/unknown backends fall back to
    // ggml_conv_1d_dw (im2col + F16 kernel).
    {
        std::string bn = internal::backend_name(impl->backend);
        std::transform(bn.begin(), bn.end(), bn.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        const bool direct_ok = bn.find("cpu")    != std::string::npos ||
                               bn.find("vulkan") != std::string::npos ||
                               bn.find("cuda")   != std::string::npos ||
                               bn.find("metal")  != std::string::npos;
        internal::ops::set_direct_dwconv(direct_ok);
    }

    impl->weights = std::make_unique<internal::LoadedWeights>(
        internal::LoadedWeights::load_all(*impl->gguf, impl->backend));

    // Top-level weights.
    impl->w_spec_proj = impl->weights->get("spectrogram_projection.weight");
    impl->b_spec_proj = impl->weights->get("spectrogram_projection.bias");

    // Sub-models.
    impl->encoder_w   = internal::bind_encoder_weights(*impl->weights, impl->cfg.encoder, "encoder");
    impl->segmenter_w = internal::bind_segmenter_weights(*impl->weights, impl->cfg);
    impl->estimator_w = internal::bind_estimator_weights(*impl->weights, impl->cfg);

    // Front-end.
    MelConfig mc;
    mc.sample_rate = impl->cfg.inference.audio_sample_rate;
    mc.n_fft       = impl->cfg.inference.fft_size;
    mc.win_length  = impl->cfg.inference.win_size;
    mc.hop_length  = impl->cfg.inference.hop_size;
    mc.n_mels      = impl->cfg.inference.n_mels;
    mc.fmin        = impl->cfg.inference.fmin;
    mc.fmax        = impl->cfg.inference.fmax;
    impl->mel_extractor = std::make_unique<MelExtractor>(mc);

    return impl;
}

// ============================================================================
// Stage helpers — each owns its own ggml_context + graph
// ============================================================================

namespace {

// Thin guard making a temporary ggml_context + gallocr used by a single stage.
struct StageCtx {
    ggml_context * ctx     = nullptr;
    ggml_cgraph  * graph   = nullptr;
    ggml_gallocr_t alloc   = nullptr;
    ggml_backend_t backend = nullptr;

    StageCtx(ggml_backend_t b, std::size_t mem_bytes, int graph_nodes) {
        backend = b;
        ggml_init_params ip{};
        ip.mem_size = mem_bytes;
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        graph = ggml_new_graph_custom(ctx, graph_nodes, /*grads=*/false);
    }

    void dump_backend_support(const char * stage) {
        const char * env = std::getenv("GAME_GGML_DUMP_OPS");
        if (!env || !*env || env[0] == '0') return;

        std::map<std::string, int> unsupported;
        const int n_nodes = ggml_graph_n_nodes(graph);
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (!ggml_backend_supports_op(backend, node)) {
                unsupported[ggml_op_name(node->op)] += 1;
            }
        }

        std::fprintf(stderr,
            "[GAME_GGML_OPS] stage=%s backend=%s nodes=%d unsupported=%d\n",
            stage, game_ggml::internal::backend_name(backend),
            n_nodes,
            std::accumulate(unsupported.begin(), unsupported.end(), 0,
                [](int acc, const auto & kv) { return acc + kv.second; }));
        for (const auto & kv : unsupported) {
            std::fprintf(stderr, "[GAME_GGML_OPS]   unsupported %-24s %d\n",
                kv.first.c_str(), kv.second);
        }
    }

    void finalize(ggml_tensor * out, const char * stage = "stage") {
        ggml_set_output(out);
        ggml_build_forward_expand(graph, out);
        dump_backend_support(stage);
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(alloc, graph)) {
            throw Error("ggml_gallocr_alloc_graph failed");
        }
    }

    void compute() {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw Error("graph compute failed");
        }
    }

    ~StageCtx() {
        if (alloc) ggml_gallocr_free(alloc);
        if (ctx)   ggml_free(ctx);
    }
};

}  // namespace

// ============================================================================
// Stage 1 — encoder (mel → x_seg, x_est)
// ============================================================================

void Model::Impl::run_encoder(const float * mel, int T,
                              std::vector<float> & x_seg_out,
                              std::vector<float> & x_est_out)
{
    const int D_mel = cfg.in_dim;
    const int D_emb = cfg.embedding_dim;

    StageCtx s(backend, 256 * 1024 * 1024, 8192);

    ggml_tensor * mel_in = ggml_new_tensor_3d(s.ctx, GGML_TYPE_F32, D_mel, T, 1);
    ggml_set_input(mel_in);
    ggml_tensor * pos = ggml_new_tensor_1d(s.ctx, GGML_TYPE_I32, T);
    ggml_set_input(pos);

    // spectrogram_projection (mel → D_embed)
    ggml_tensor * x_proj = internal::ops::linear(s.ctx, mel_in, w_spec_proj, b_spec_proj);
    auto outs = internal::build_encoder_graph(s.ctx, x_proj, encoder_w, pos, cfg.encoder);

    ggml_set_output(outs.x_seg);
    ggml_set_output(outs.x_est);
    ggml_build_forward_expand(s.graph, outs.x_seg);
    ggml_build_forward_expand(s.graph, outs.x_est);
    s.dump_backend_support("encoder");
    s.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(s.alloc, s.graph)) throw Error("alloc failed (encoder)");

    ggml_backend_tensor_set(mel_in, mel, 0, ggml_nbytes(mel_in));
    std::vector<std::int32_t> pos_i(T);
    for (int i = 0; i < T; ++i) pos_i[i] = i;
    ggml_backend_tensor_set(pos, pos_i.data(), 0, pos_i.size() * sizeof(std::int32_t));

    s.compute();

    x_seg_out.resize(ggml_nelements(outs.x_seg));
    x_est_out.resize(ggml_nelements(outs.x_est));
    ggml_backend_tensor_get(outs.x_seg, x_seg_out.data(), 0, x_seg_out.size() * sizeof(float));
    ggml_backend_tensor_get(outs.x_est, x_est_out.data(), 0, x_est_out.size() * sizeof(float));
}

// Batched encoder: B samples share one (D_mel, T, B) input and a shared
// positions vector 0..T-1.  With a per-frame mask, samples may have unequal
// lengths (mels are left-aligned, T = max frames, padding frames masked 0).
void Model::Impl::run_encoder_batch(const float * mel, int T, int B,
                                    const std::uint8_t * mask,
                                    std::vector<float> & x_seg_out,
                                    std::vector<float> & x_est_out) {
    const int D_mel = cfg.in_dim;
    const int D_emb = cfg.embedding_dim;

    StageCtx s(backend, 256 * 1024 * 1024, 8192);

    ggml_tensor * mel_in = ggml_new_tensor_3d(s.ctx, GGML_TYPE_F32, D_mel, T, B);
    ggml_set_input(mel_in);
    ggml_tensor * pos = ggml_new_tensor_1d(s.ctx, GGML_TYPE_I32, T);
    ggml_set_input(pos);

    ggml_tensor * mask_f32 = nullptr;
    if (mask) {
        mask_f32 = ggml_new_tensor_3d(s.ctx, GGML_TYPE_F32, 1, T, B);  // (1,T,B)
        ggml_set_input(mask_f32);
    }

    ggml_tensor * x_proj = internal::ops::linear(s.ctx, mel_in, w_spec_proj, b_spec_proj);
    auto outs = internal::build_encoder_graph(s.ctx, x_proj, encoder_w, pos, cfg.encoder,
                                              mask_f32);

    ggml_set_output(outs.x_seg);
    ggml_set_output(outs.x_est);
    ggml_build_forward_expand(s.graph, outs.x_seg);
    ggml_build_forward_expand(s.graph, outs.x_est);
    s.dump_backend_support("encoder/batch");
    s.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(s.alloc, s.graph)) throw Error("alloc failed (encoder/batch)");

    ggml_backend_tensor_set(mel_in, mel, 0, ggml_nbytes(mel_in));
    std::vector<std::int32_t> pos_i(T);
    for (int i = 0; i < T; ++i) pos_i[i] = i;
    ggml_backend_tensor_set(pos, pos_i.data(), 0, pos_i.size() * sizeof(std::int32_t));
    if (mask) {
        std::vector<float> mf(T * B);
        for (int i = 0; i < T * B; ++i) mf[i] = mask[i] ? 1.0f : 0.0f;
        ggml_backend_tensor_set(mask_f32, mf.data(), 0, mf.size() * sizeof(float));
    }

    s.compute();

    x_seg_out.resize(ggml_nelements(outs.x_seg));
    x_est_out.resize(ggml_nelements(outs.x_est));
    ggml_backend_tensor_get(outs.x_seg, x_seg_out.data(), 0, x_seg_out.size() * sizeof(float));
    ggml_backend_tensor_get(outs.x_est, x_est_out.data(), 0, x_est_out.size() * sizeof(float));
}

// ============================================================================
// Stage 2 — segmenter (one D3PM step, DBCache-aware)
// ============================================================================

namespace {

// Normalized L1 residual between the current and previous front output
// (mirrors PyTorch DBCacheSegmenter: mean|x-x_prev| / (mean|x_prev| + eps)).
inline float front_delta(const float * cur, const float * prev, int n) {
    float num = 0.0f, den = 0.0f;
    for (int i = 0; i < n; ++i) {
        num += std::fabs(cur[i] - prev[i]);
        den += std::fabs(prev[i]);
    }
    return num / (den + 1e-8f);
}

}  // namespace

void Model::Impl::run_segmenter_step(
    const float * x_seg_host, int T,
    const std::int32_t * noise_mod3, float t_scalar, int language,
    std::vector<float> & logits_out)
{
    const int D = cfg.embedding_dim;
    SegmenterCacheState & cache = seg_cache;

    // ---------- Fused fast path (DBCache disabled, the default) ----------
    // One combined graph + a single backend submit is cheaper than the
    // 3-stage host-copy split below: fewer gallocr allocations, no extra
    // host round-trips, and it matches the pre-DBCache behavior exactly.
    if (!cache.enabled) {
        StageCtx s(backend, 512 * 1024 * 1024, 16384);

        ggml_tensor * xseg        = ggml_new_tensor_3d(s.ctx, GGML_TYPE_F32, D, T, 1);
        ggml_tensor * noise       = ggml_new_tensor_1d(s.ctx, GGML_TYPE_I32, T);
        ggml_tensor * t_tensor    = ggml_new_tensor_3d(s.ctx, GGML_TYPE_F32, 1, 1, 1);
        ggml_tensor * lang_tensor = ggml_new_tensor_1d(s.ctx, GGML_TYPE_I32, 1);
        ggml_tensor * positions   = ggml_new_tensor_1d(s.ctx, GGML_TYPE_I32, T);
        for (auto * t : {xseg, noise, t_tensor, lang_tensor, positions}) ggml_set_input(t);

        auto outs = internal::build_segmenter_graph(
            s.ctx, xseg, noise, t_tensor, lang_tensor, positions, segmenter_w, cfg);

        ggml_set_output(outs.logits);
        ggml_build_forward_expand(s.graph, outs.logits);
        if (outs.latent) { ggml_set_output(outs.latent); ggml_build_forward_expand(s.graph, outs.latent); }
        s.dump_backend_support("segmenter");
        s.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(s.alloc, s.graph)) throw Error("alloc failed (segmenter/fused)");

        ggml_backend_tensor_set(xseg,        x_seg_host, 0, ggml_nbytes(xseg));
        ggml_backend_tensor_set(noise,       noise_mod3, 0, T * sizeof(std::int32_t));
        ggml_backend_tensor_set(t_tensor,    &t_scalar,  0, sizeof(float));
        const std::int32_t l = static_cast<std::int32_t>(language);
        ggml_backend_tensor_set(lang_tensor, &l, 0, sizeof(std::int32_t));
        std::vector<std::int32_t> pos(T);
        for (int i = 0; i < T; ++i) pos[i] = i;
        ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(std::int32_t));

        s.compute();

        logits_out.resize(T);
        ggml_backend_tensor_get(outs.logits, logits_out.data(), 0, logits_out.size() * sizeof(float));
        return;
    }

    const int nf = std::min(std::max(cache.fn_blocks, 0), cfg.segmenter.num_layers);

    // ---------- Stage A: front blocks (always executed) ----------
    std::vector<float> x_front(D * T);
    {
        StageCtx s(backend, 256 * 1024 * 1024, 8192);

        ggml_tensor * xseg        = ggml_new_tensor_3d(s.ctx, GGML_TYPE_F32, D, T, 1);
        ggml_tensor * noise       = ggml_new_tensor_1d(s.ctx, GGML_TYPE_I32, T);
        ggml_tensor * t_tensor    = ggml_new_tensor_3d(s.ctx, GGML_TYPE_F32, 1, 1, 1);
        ggml_tensor * lang_tensor = ggml_new_tensor_1d(s.ctx, GGML_TYPE_I32, 1);
        ggml_tensor * positions   = ggml_new_tensor_1d(s.ctx, GGML_TYPE_I32, T);
        for (auto * t : {xseg, noise, t_tensor, lang_tensor, positions}) ggml_set_input(t);

        ggml_tensor * out_front = internal::build_segmenter_front_graph(
            s.ctx, xseg, noise, t_tensor, lang_tensor, positions, segmenter_w, cfg, nf);

        s.finalize(out_front, "segmenter/front");

        ggml_backend_tensor_set(xseg,        x_seg_host, 0, ggml_nbytes(xseg));
        ggml_backend_tensor_set(noise,       noise_mod3, 0, T * sizeof(std::int32_t));
        ggml_backend_tensor_set(t_tensor,    &t_scalar,  0, sizeof(float));
        const std::int32_t l = static_cast<std::int32_t>(language);
        ggml_backend_tensor_set(lang_tensor, &l, 0, sizeof(std::int32_t));
        std::vector<std::int32_t> pos(T);
        for (int i = 0; i < T; ++i) pos[i] = i;
        ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(std::int32_t));

        s.compute();
        ggml_backend_tensor_get(out_front, x_front.data(), 0, x_front.size() * sizeof(float));
    }

    // ---------- Decide cache hit ----------
    bool use_cache = false;
    const int N_seg = cfg.segmenter.num_layers;
    if (cache.enabled && cache.valid &&
        cache.step >= cache.warmup &&
        cache.prev_front.size() == x_front.size()) {
        const float fd = front_delta(x_front.data(), cache.prev_front.data(), D * T);

        // Reuse window: cache only mid-schedule steps (first/last computed
        // fully unless the window covers them).
        bool in_window = true;
        if (cache.total_steps > 1) {
            const float frac = static_cast<float>(cache.step + 1) / static_cast<float>(cache.total_steps);
            in_window = frac >= cache.window_start && frac <= cache.window_end;
        }

        // UCache-style accumulated error gate (opt-in via err_decay > 0).
        bool err_ok = true;
        if (cache.err_decay > 0.0f) {
            cache.acc_err = (cache.acc_err + fd) * cache.err_decay;
            err_ok = cache.acc_err <= cache.err_limit;
        }

        // Consecutive-hit guard.
        const bool cont_ok = (cache.max_cont == 0) || (cache.cont_cnt < cache.max_cont);

        use_cache = (fd < cache.threshold) && in_window && err_ok && cont_ok;
    }

    // ---------- Stage B: tail blocks (middle reused on hit, back always done) ----------
    const int nb = std::min(std::max(cache.bn_blocks, 0), N_seg - nf);
    const int middle_end = N_seg - nb;             // middle = [nf, middle_end)
    std::vector<float> x_mid(D * T);
    std::vector<float> x_out(D * T);

    auto run_tail_range = [&](const float * in_host, int start, int end,
                              std::vector<float> & out_host) {
        StageCtx s(backend, 512 * 1024 * 1024, 16384);
        ggml_tensor * x_in       = ggml_new_tensor_3d(s.ctx, GGML_TYPE_F32, D, T, 1);
        ggml_tensor * positions  = ggml_new_tensor_1d(s.ctx, GGML_TYPE_I32, T);
        for (auto * t : {x_in, positions}) ggml_set_input(t);

        auto outs = internal::build_segmenter_tail_graph(
            s.ctx, x_in, positions, segmenter_w, cfg, start, end);

        // Both outputs must be registered BEFORE gallocr allocation — expanding
        // after alloc leaves latent unallocated = UB on compute.
        ggml_set_output(outs.x_run);
        ggml_build_forward_expand(s.graph, outs.x_run);
        if (outs.latent) { ggml_set_output(outs.latent); ggml_build_forward_expand(s.graph, outs.latent); }
        s.dump_backend_support("segmenter/tail");
        s.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(s.alloc, s.graph)) throw Error("alloc failed (segmenter/tail)");

        ggml_backend_tensor_set(x_in, in_host, 0, x_front.size() * sizeof(float));
        std::vector<std::int32_t> pos(T);
        for (int i = 0; i < T; ++i) pos[i] = i;
        ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(std::int32_t));

        s.compute();
        ggml_backend_tensor_get(outs.x_run, out_host.data(), 0, out_host.size() * sizeof(float));
    };

    if (use_cache) {
        // Reconstruct the middle from the cached delta, then always run the
        // back slice (if any) so the output-end blocks see accurate input.
        for (int i = 0; i < D * T; ++i) x_mid[i] = x_front[i] + cache.tail_delta[i];
        if (nb > 0) run_tail_range(x_mid.data(), middle_end, N_seg, x_out);
        else        for (int i = 0; i < D * T; ++i) x_out[i] = x_mid[i];
        ++cache.hits;
        ++cache.cont_cnt;
    } else {
        run_tail_range(x_front.data(), nf, middle_end, x_mid);
        if (nb > 0) run_tail_range(x_mid.data(), middle_end, N_seg, x_out);
        else        for (int i = 0; i < D * T; ++i) x_out[i] = x_mid[i];
        cache.tail_delta.resize(D * T);
        for (int i = 0; i < D * T; ++i) cache.tail_delta[i] = x_mid[i] - x_front[i];
        cache.prev_front = x_front;
        cache.valid = true;
        cache.acc_err = 0.0f;
        cache.cont_cnt = 0;
        ++cache.misses;
    }

    // ---------- Stage C: head (output norm + proj) ----------
    {
        StageCtx s(backend, 16 * 1024 * 1024, 256);

        ggml_tensor * x_in = ggml_new_tensor_3d(s.ctx, GGML_TYPE_F32, D, T, 1);
        ggml_set_input(x_in);

        ggml_tensor * logits = internal::build_segmenter_head_graph(s.ctx, x_in, segmenter_w, cfg);

        s.finalize(logits, "segmenter/head");

        ggml_backend_tensor_set(x_in, x_out.data(), 0, x_out.size() * sizeof(float));
        s.compute();

        logits_out.resize(T);
        ggml_backend_tensor_get(logits, logits_out.data(), 0, logits_out.size() * sizeof(float));
    }

    ++cache.step;
    if (cache.enabled && std::getenv("GAME_GGML_DUMP_DBCACHE")) {
        std::fprintf(stderr, "[DBCACHE] step=%d %s (hits=%d misses=%d)\n",
            cache.step, use_cache ? "HIT" : "MISS", cache.hits, cache.misses);
    }
}

// ============================================================================
// Stage 2b — segmenter, batched (one D3PM step over a whole [B,T] batch)
// ============================================================================

void Model::Impl::run_segmenter_batch(
    const float * x_seg_host, int T, int B,
    const std::int32_t * noise_mod3,       // (T, B) flattened b-major: idx = b*T + t
    float t_scalar, const int * language,  // language array length B
    std::vector<float> & logits_out)       // (T, B) flattened b-major
{
    const int D = cfg.embedding_dim;

    // Fast path: the same (T,B) batch segmenter graph is reused across the
    // D3PM steps of one inference (built once on the first step).
    if (batch_seg.T == T && batch_seg.B == B && batch_seg.ctx) {
        ggml_backend_tensor_set(batch_seg.xseg,     x_seg_host, 0, ggml_nbytes(batch_seg.xseg));
        ggml_backend_tensor_set(batch_seg.noise,    noise_mod3, 0, T * B * sizeof(std::int32_t));
        ggml_backend_tensor_set(batch_seg.t_tensor, &t_scalar,  0, sizeof(float));
        ggml_backend_tensor_set(batch_seg.lang,     language,   0, B * sizeof(std::int32_t));
        std::vector<std::int32_t> pos(T);
        for (int i = 0; i < T; ++i) pos[i] = i;
        ggml_backend_tensor_set(batch_seg.positions, pos.data(), 0, pos.size() * sizeof(std::int32_t));
        if (ggml_backend_graph_compute(backend, batch_seg.graph) != GGML_STATUS_SUCCESS)
            throw Error("graph compute failed (segmenter/batch)");
        logits_out.resize(ggml_nelements(batch_seg.logits));
        ggml_backend_tensor_get(batch_seg.logits, logits_out.data(), 0, logits_out.size() * sizeof(float));
        return;
    }

    // Build path (first step for this (T,B)): construct into batch_seg.
    release_batch_seg();
    batch_seg.T = T;
    batch_seg.B = B;
    ggml_init_params ip{};
    ip.mem_size = 512 * 1024 * 1024;
    ip.no_alloc = true;
    batch_seg.ctx = ggml_init(ip);
    batch_seg.graph = ggml_new_graph_custom(batch_seg.ctx, 16384, /*grads=*/false);
    auto & c = batch_seg.ctx;
    auto * xseg        = batch_seg.xseg      = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, T, B);
    auto * noise       = batch_seg.noise     = ggml_new_tensor_1d(c, GGML_TYPE_I32, T * B);
    auto * t_tensor    = batch_seg.t_tensor  = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, 1, 1);
    auto * lang        = batch_seg.lang      = ggml_new_tensor_1d(c, GGML_TYPE_I32, B);
    auto * positions   = batch_seg.positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
    for (auto * t : {xseg, noise, t_tensor, lang, positions}) ggml_set_input(t);

    // ---- custom front (batch-safe embedding injections) ----
    ggml_tensor * noise_emb = internal::ops::embedding(c, segmenter_w.w_noise_embedding, noise);
    // embedding(weight(D,V), idx(T*B)) -> (D, T*B); reshape to (D,T,B).
    noise_emb = ggml_reshape_3d(c, ggml_cont(c, noise_emb), D, T, B);
    ggml_tensor * x = ggml_add(c, xseg, noise_emb);

    if (segmenter_w.w_time_0 && t_tensor) {
        ggml_tensor * h = internal::ops::linear(c, t_tensor, segmenter_w.w_time_0, segmenter_w.b_time_0);
        h = ggml_gelu(c, h);
        h = internal::ops::linear(c, h, segmenter_w.w_time_2, segmenter_w.b_time_2);
        // (D,1,1) -> broadcast add over (D,T,B)
        x = ggml_add(c, x, h);
    }

    if (segmenter_w.w_lang_embedding && lang) {
        ggml_tensor * le = internal::ops::embedding(c, segmenter_w.w_lang_embedding, lang);
        // (D,B) flat layout d + b*D already matches (D,1,B) ordering — reshape
        // (not view) so T becomes a no-op broadcast dim.
        le = ggml_reshape_3d(c, ggml_cont(c, le), D, 1, B);
        x = ggml_add(c, x, le);
    }

    x = internal::ops::linear(c, x, segmenter_w.w_input_proj, segmenter_w.b_input_proj);

    // ---- full block stack (+ latent tap if configured) ----
    ggml_tensor * latent_tap = nullptr;
    for (int i = 0; i < cfg.segmenter.num_layers; ++i) {
        x = internal::ops::ebf_block(c, x, segmenter_w.layers[i], positions,
                                     cfg.segmenter.num_heads, cfg.segmenter.head_dim);
        if (cfg.segmenter.return_latent && i == cfg.segmenter.latent_layer_idx - 1) {
            latent_tap = x;
        }
    }

    ggml_tensor * latent = nullptr;
    if (cfg.segmenter.return_latent && latent_tap) {
        ggml_tensor * lt = latent_tap;
        if (segmenter_w.w_latent_norm) lt = internal::ops::rms_norm(c, lt, segmenter_w.w_latent_norm);
        latent = ggml_cont(c, internal::ops::linear(c, lt, segmenter_w.w_latent_proj, segmenter_w.b_latent_proj));
        ggml_set_output(latent);
        ggml_build_forward_expand(batch_seg.graph, latent);
    }

    // ---- head ----
    if (segmenter_w.w_output_norm) x = internal::ops::rms_norm(c, x, segmenter_w.w_output_norm);
    ggml_tensor * logitsT = internal::ops::linear(c, x, segmenter_w.w_output_proj, segmenter_w.b_output_proj);
    batch_seg.logits = ggml_cont(c, logitsT);                // (1, T, B)
    ggml_set_output(batch_seg.logits);
    ggml_build_forward_expand(batch_seg.graph, batch_seg.logits);
    dump_batch_seg_ops();

    batch_seg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(batch_seg.alloc, batch_seg.graph))
        throw Error("alloc failed (segmenter/batch)");

    ggml_backend_tensor_set(xseg,     x_seg_host, 0, ggml_nbytes(xseg));
    ggml_backend_tensor_set(noise,    noise_mod3, 0, T * B * sizeof(std::int32_t));
    ggml_backend_tensor_set(t_tensor, &t_scalar,  0, sizeof(float));
    ggml_backend_tensor_set(lang,     language,   0, B * sizeof(std::int32_t));
    std::vector<std::int32_t> pos(T);
    for (int i = 0; i < T; ++i) pos[i] = i;
    ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(std::int32_t));

    if (ggml_backend_graph_compute(backend, batch_seg.graph) != GGML_STATUS_SUCCESS)
        throw Error("graph compute failed (segmenter/batch)");

    logits_out.resize(ggml_nelements(batch_seg.logits));
    ggml_backend_tensor_get(batch_seg.logits, logits_out.data(), 0, logits_out.size() * sizeof(float));
}

void Model::Impl::dump_batch_seg_ops() {
    const char * env = std::getenv("GAME_GGML_DUMP_OPS");
    if (!env || !*env || env[0] == '0') return;
    std::map<std::string,int> unsupported;
    const int n = ggml_graph_n_nodes(batch_seg.graph);
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * node = ggml_graph_node(batch_seg.graph, i);
        if (!ggml_backend_supports_op(backend, node)) unsupported[ggml_op_name(node->op)] += 1;
    }
    std::fprintf(stderr, "[GAME_GGML_OPS] stage=segmenter/batch backend=%s nodes=%d unsupported=%d\n",
        internal::backend_name(backend), n,
        std::accumulate(unsupported.begin(), unsupported.end(), 0,
            [](int a, const auto & kv){ return a + kv.second; }));
    for (const auto & kv : unsupported)
        std::fprintf(stderr, "[GAME_GGML_OPS]   unsupported %-24s %d\n", kv.first.c_str(), kv.second);
}

// ============================================================================
// Stage 3 — estimator (regions → pool_logits)
// ============================================================================

void Model::Impl::run_estimator(
    const float * x_est_host, int T,
    const std::int32_t * regions, int N,
    std::vector<float> & pool_logits_out)
{
    const int D = cfg.embedding_dim;
    const int S = N + T;

    StageCtx s(backend, 512 * 1024 * 1024, 16384);

    ggml_tensor * xest        = ggml_new_tensor_3d(s.ctx, GGML_TYPE_F32, D, T, 1);
    ggml_tensor * regions_mod = ggml_new_tensor_1d(s.ctx, GGML_TYPE_I32, T);
    ggml_tensor * positions   = ggml_new_tensor_1d(s.ctx, GGML_TYPE_I32, S);
    ggml_tensor * region_ids  = ggml_new_tensor_1d(s.ctx, GGML_TYPE_I32, S);
    ggml_tensor * mask_fp16   = ggml_new_tensor_4d(s.ctx, GGML_TYPE_F16, S, S, 1, 1);
    for (auto * t : {xest, regions_mod, positions, region_ids, mask_fp16}) ggml_set_input(t);

    auto outs = internal::build_estimator_graph(
        s.ctx, xest, regions_mod, positions, region_ids, mask_fp16, N, estimator_w, cfg);

    ggml_set_output(outs.pool_logits);
    ggml_build_forward_expand(s.graph, outs.pool_logits);
    s.dump_backend_support("estimator");
    s.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(s.alloc, s.graph)) throw Error("alloc failed (estimator)");

    // Host-side prep.
    ggml_backend_tensor_set(xest, x_est_host, 0, ggml_nbytes(xest));

    std::vector<std::int32_t> rmod(T);
    for (int i = 0; i < T; ++i) rmod[i] = regions[i] % cfg.region_cycle_len;
    ggml_backend_tensor_set(regions_mod, rmod.data(), 0, rmod.size() * sizeof(std::int32_t));

    // Global positions: pool 0..N-1, x 0..T-1
    std::vector<std::int32_t> gpos(S);
    for (int i = 0; i < N; ++i) gpos[i] = i;
    for (int i = 0; i < T; ++i) gpos[N + i] = i;
    ggml_backend_tensor_set(positions, gpos.data(), 0, gpos.size() * sizeof(std::int32_t));

    // Region RoPE indices: pool = 0 (R=1, use_pool_offset=false);
    // x = local_position_within_region + R (= +1).
    std::vector<std::int32_t> ridx(S, 0);
    {
        int cur_region = 0;
        int cur_local  = 0;
        for (int i = 0; i < T; ++i) {
            const int r = regions[i];
            if (r != cur_region) { cur_region = r; cur_local = 0; }
            ridx[N + i] = (r > 0) ? (cur_local + 1) : 0;
            ++cur_local;
        }
    }
    ggml_backend_tensor_set(region_ids, ridx.data(), 0, ridx.size() * sizeof(std::int32_t));

    auto mask = internal::ops::build_joint_attn_mask_fp16(regions, T, N);
    ggml_backend_tensor_set(mask_fp16, mask.data(), 0, mask.size() * sizeof(std::uint16_t));

    s.compute();

    pool_logits_out.resize(ggml_nelements(outs.pool_logits));
    ggml_backend_tensor_get(outs.pool_logits, pool_logits_out.data(),
                            0, pool_logits_out.size() * sizeof(float));
}

// ============================================================================
// Main entry: D3PM-orchestrated end-to-end inference
// ============================================================================

namespace {

std::vector<float> default_d3pm_schedule(float t0, int n_steps) {
    std::vector<float> ts;
    ts.reserve(n_steps);
    const float step = (1.0f - t0) / static_cast<float>(n_steps);
    for (int i = 0; i < n_steps; ++i) ts.push_back(t0 + i * step);
    return ts;
}

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Simple profiler, gated by the GAME_GGML_PROFILE env var.  Accumulates
// wall-time per stage across chunks; emits a summary to stderr on destruction.
// Stage boundaries follow the ONNX export contract (deployment/exporter.py):
//   * "encoder" = waveform → (x_seg, x_est, maskT); absorbs mel + spec_proj
//   * "segmenter" = x_seg + embeddings → boundary_logits, repeated per D3PM step
//   * "estimator" = x_est + regions → note logits
struct StageProfiler {
    bool enabled;
    using clock = std::chrono::steady_clock;
    using dur   = std::chrono::duration<double>;

    double encoder_s = 0.0, segmenter_s = 0.0, estimator_s = 0.0, decode_s = 0.0;
    int    n_segmenter_steps = 0;

    StageProfiler() {
        const char * env = std::getenv("GAME_GGML_PROFILE");
        enabled = (env && *env && env[0] != '0');
    }

    struct Scope {
        StageProfiler * p;
        double        * acc;
        clock::time_point t0;
        Scope(StageProfiler * p_, double * acc_) : p(p_), acc(acc_) {
            if (p && p->enabled) t0 = clock::now();
        }
        ~Scope() {
            if (p && p->enabled) *acc += dur(clock::now() - t0).count();
        }
    };

    Scope scope_encoder()   { return Scope(this, &encoder_s); }
    Scope scope_segmenter() { if (enabled) ++n_segmenter_steps; return Scope(this, &segmenter_s); }
    Scope scope_estimator() { return Scope(this, &estimator_s); }
    Scope scope_decode()    { return Scope(this, &decode_s); }

    ~StageProfiler() {
        if (!enabled) return;
        const double total = encoder_s + segmenter_s + estimator_s + decode_s;
        std::fprintf(stderr,
            "\n[GAME_GGML_PROFILE] per-chunk stage timings (ONNX-aligned)\n"
            "    encoder     %7.3f s  (%5.1f %%)   mel + spec_proj + 4× EBF\n"
            "    segmenter   %7.3f s  (%5.1f %%)   over %d D3PM steps\n"
            "    estimator   %7.3f s  (%5.1f %%)\n"
            "    decode/cpu  %7.3f s  (%5.1f %%)\n"
            "    ------------------------\n"
            "    total       %7.3f s\n",
            encoder_s,   100.0 * encoder_s    / total,
            segmenter_s, 100.0 * segmenter_s  / total, n_segmenter_steps,
            estimator_s, 100.0 * estimator_s  / total,
            decode_s,    100.0 * decode_s     / total,
            total);
    }
};

}  // namespace

InferResult Model::Impl::infer_with_rng(
    const float * waveform, std::size_t n_samples,
    const InferParams & params,
    internal::IRandomSource & rng)
{
    using namespace internal;
    StageProfiler prof;

    const int D = cfg.embedding_dim;
    const int T = mel_extractor->num_frames(n_samples);
    if (T <= 0) throw InvalidArgument("waveform too short for one mel frame");

    // --- 1) encoder (waveform → x_seg, x_est)  [ONNX: encoder.onnx]
    //        covers: mel extraction, spectrogram_projection, 4× EBF blocks,
    //        output split.  The mel sub-stage runs on CPU (pocketfft STFT
    //        + mel filterbank mul + log); everything after is on the backend.
    std::vector<float> x_seg_host, x_est_host;
    {
        auto _ = prof.scope_encoder();
        auto mel = mel_extractor->forward(waveform, n_samples);          // [T, 80]
        run_encoder(mel.data(), T, x_seg_host, x_est_host);
    }

    // --- 3) D3PM loop + estimator + decode (shared with the batched path)
    return infer_from_latent(x_seg_host.data(), x_est_host.data(), T, params, rng);
}

InferResult Model::Impl::infer_from_latent(
    const float * x_seg, const float * x_est, int T,
    const InferParams & params,
    internal::IRandomSource & rng) {
    using namespace internal;
    StageProfiler prof;

    const int D = cfg.embedding_dim;

    std::vector<float> ts = params.d3pm_ts.empty()
        ? default_d3pm_schedule(params.d3pm_t0, params.d3pm_nsteps)
        : params.d3pm_ts;

    // DBCache: configured per infer (reset per segment).  Mirrors PyTorch
    // reset on forward_segmenter_main.  With a single step there is nothing to
    // cache, so the fused single-graph path stays active.
    float thr = params.db_cache_threshold;
    if (thr < 0.0f) {
        const bool gpu = [this] {
            const char * bn = internal::backend_name(backend);
            if (!bn) return false;
            std::string s(bn);
            std::transform(s.begin(), s.end(), s.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return s.find("vulkan") != std::string::npos ||
                   s.find("cuda")   != std::string::npos ||
                   s.find("metal")  != std::string::npos;
        }();
        thr = gpu ? 0.0f : 0.25f;
    }
    seg_cache.enabled   = thr > 0.0f && ts.size() > 1;
    seg_cache.threshold = thr;
    seg_cache.fn_blocks = params.db_cache_fn_blocks;
    seg_cache.warmup    = params.db_cache_warmup;
    seg_cache.reset();
    seg_cache.total_steps = static_cast<int>(ts.size());
    seg_cache.window_start = params.db_cache_window_start;
    seg_cache.window_end   = params.db_cache_window_end;
    seg_cache.err_decay    = params.db_cache_err_decay;
    seg_cache.err_limit    = params.db_cache_err_limit;
    seg_cache.max_cont     = params.db_cache_max_cont;
    seg_cache.bn_blocks    = params.db_cache_bn_blocks;
    if (seg_cache.enabled) {
        std::fprintf(stderr,
            "DBCache: threshold=%.3f fn_blocks=%d warmup=%d win=[%.2f,%.2f] "
            "decay=%.2f max_cont=%d bn=%d\n",
            seg_cache.threshold, seg_cache.fn_blocks, seg_cache.warmup,
            seg_cache.window_start, seg_cache.window_end,
            seg_cache.err_decay, seg_cache.max_cont, seg_cache.bn_blocks);
    }

    std::vector<std::uint8_t> known(T, 0);
    std::vector<std::uint8_t> mask(T, 1);
    std::vector<std::uint8_t> boundaries(known);

    std::vector<std::int32_t> noise_mod(T);
    std::vector<float> probs(T);
    std::vector<float> logits;

    for (float ti : ts) {
        {
            auto _ = prof.scope_decode();
            const float p = d3pm_time_schedule(ti);
            std::vector<std::uint8_t> next(T);
            remove_mutable_boundaries(boundaries.data(), known.data(), T, p, rng, next.data());
            boundaries = std::move(next);

            auto regions = game_ggml::boundaries_to_regions(
                boundaries.data(), mask.data(), T);
            for (int i = 0; i < T; ++i) noise_mod[i] = regions[i] % cfg.region_cycle_len;
        }

        {
            auto _ = prof.scope_segmenter();
            run_segmenter_step(x_seg, T,
                noise_mod.data(), ti, params.language, logits);
        }

        {
            auto _ = prof.scope_decode();
            for (int i = 0; i < T; ++i) probs[i] = sigmoid(logits[i]);
            boundaries = game_ggml::decode_soft_boundaries(
                probs.data(), T, known.data(), mask.data(),
                params.boundary_threshold, params.boundary_radius);
        }
    }

    // --- 4) regions + N
    std::vector<std::int32_t> regions;
    int N = 0;
    {
        auto _ = prof.scope_decode();
        regions = game_ggml::boundaries_to_regions(
            boundaries.data(), mask.data(), T);
        for (int i = 0; i < T; ++i) N = std::max<int>(N, regions[i]);
    }

    InferResult result;
    result.num_frames = T;
    result.db_cache_hits   = seg_cache.hits;
    result.db_cache_misses = seg_cache.misses;
    if (N == 0) return result;

    // --- 5) estimator
    std::vector<float> pool_logits;
    { auto _ = prof.scope_estimator();
      run_estimator(x_est, T, regions.data(), N, pool_logits); }

    // --- 6) pitch decode
    {
        auto _ = prof.scope_decode();
        const int bins = cfg.estimator_out_dim;
        std::vector<float> pool_probs(pool_logits.size());
        for (std::size_t i = 0; i < pool_logits.size(); ++i) {
            pool_probs[i] = sigmoid(pool_logits[i]);
        }
        auto dec = game_ggml::decode_gaussian_blurred_probs(
            pool_probs.data(), static_cast<std::size_t>(N), static_cast<std::size_t>(bins),
            cfg.inference.midi_min, cfg.inference.midi_max,
            cfg.inference.midi_std * 3.0f,
            params.note_threshold);

        // --- 7) collect notes
        const float timestep = cfg.inference.timestep();
        std::vector<int> dur_frames(N + 1, 0);
        for (int i = 0; i < T; ++i) {
            if (regions[i] > 0 && regions[i] <= N) ++dur_frames[regions[i]];
        }
        float offset = 0.0f;
        for (int n_idx = 0; n_idx < N; ++n_idx) {
            Note nt;
            nt.offset_seconds   = offset;
            nt.duration_seconds = dur_frames[n_idx + 1] * timestep;
            nt.pitch_midi       = dec.values[n_idx];
            nt.voiced           = dec.presence[n_idx] != 0;
            result.notes.push_back(nt);
            offset += nt.duration_seconds;
        }
    }
    return result;
}

std::vector<InferResult> Model::Impl::infer_batch_impl(
    const std::vector<BatchItem> & items, const InferParams & params,
    std::vector<std::uint64_t> given_seeds) {
    using namespace internal;
    if (items.empty()) return {};

    // Derive per-item seeds (deterministic from base when given; else random).
    std::uint64_t base = params.seed;
    bool random_base = (base == 0);
    if (random_base) base = std::random_device{}();
    if (given_seeds.empty()) {
        given_seeds.resize(items.size());
        for (std::size_t i = 0; i < items.size(); ++i)
            given_seeds[i] = random_base ? std::random_device{}() : base + i;
    }

    std::vector<InferResult> out;
    out.reserve(items.size());

    // -------- fused path: mel padding + per-frame mask (infer.py semantics) -
    // mel front-end is per-item CPU; each item's mel is left-aligned into
    // (D_mel, Tmax, B) with zeros beyond its own frame count; a per-frame mask
    // (1, Tmax, B) zeroes padding frames through every EBF block.  The D3PM
    // loop + estimator + decode stay per-sample on the shared encoding.
    {
        std::vector<std::vector<float>> mel_rows;   // per item: [Ti, D_mel]
        std::vector<int> item_T;
        mel_rows.reserve(items.size());
        item_T.reserve(items.size());
        int Tmax = 0;
        for (const auto & it : items) {
            auto m = mel_extractor->forward(it.waveform, it.n_samples);
            const int Ti = mel_extractor->num_frames(it.n_samples);
            Tmax = std::max(Tmax, Ti);
            mel_rows.push_back(std::move(m));
            item_T.push_back(Ti);
        }
        // Fused path applies only when all items share the same frame count:
        // mask-padding would only approximate per-item results (PyTorch has
        // the same property), and the objective demands bit-identical output
        // vs the single-sample runs.  Unequal lengths → fallback (bit-ideal).
        const int T0 = item_T.empty() ? 0 : item_T[0];
        const bool same_t = std::all_of(item_T.begin(), item_T.end(),
            [T0](int t) { return t == T0; });
        if (!items.empty() && Tmax > 0 && same_t) {
            const int D_mel = cfg.in_dim;
            const std::size_t B = items.size();
            const int D = cfg.embedding_dim;
            const int T = Tmax;
            std::vector<float> mel_batch(static_cast<std::size_t>(D_mel) * T * B, 0.0f);
            std::vector<std::uint8_t> mask(T * B, 1);   // all frames valid (same T)
            for (std::size_t b = 0; b < B; ++b) {
                const int Ti = item_T[b];
                const auto & src = mel_rows[b];     // [Ti, D_mel] row-major
                float * dst = mel_batch.data() + b * (D_mel * T);
                for (int t = 0; t < Ti; ++t) {
                    for (int d = 0; d < D_mel; ++d)
                        dst[d + t * D_mel] = src[t * D_mel + d];
                    mask[b * T + t] = 1;            // valid frames
                }
            }
            std::vector<float> x_seg_batch, x_est_batch;
            {
                StageProfiler prof; auto _ = prof.scope_encoder();
                run_encoder_batch(mel_batch.data(), T, static_cast<int>(B),
                                  mask.data(), x_seg_batch, x_est_batch);
            }

                        std::vector<float> ts = params.d3pm_ts.empty()
                ? default_d3pm_schedule(params.d3pm_t0, params.d3pm_nsteps)
                : params.d3pm_ts;

            // Per-sample D3PM state, batched as (T,B) host arrays (b-major).
            std::vector<std::uint8_t> known(T * B, 0);
            std::vector<std::uint8_t> boundaries(T * B, 0);
            std::vector<std::int32_t> noise_mod(T * B);
            std::vector<float> probs(T * B);
            std::vector<float> logits;
            std::vector<int> languages(static_cast<std::size_t>(B), params.language);
            for (std::size_t b = 0; b < B; ++b) languages[b] = items[b].language;

            std::vector<std::unique_ptr<internal::MT19937Rng>> rngs;
            rngs.reserve(B);
            for (std::size_t b = 0; b < B; ++b)
                rngs.push_back(std::make_unique<internal::MT19937Rng>(given_seeds[b]));

            for (float ti : ts) {
                const float pv = d3pm_time_schedule(ti);
                std::vector<std::uint8_t> next(T * B);
                for (std::size_t b = 0; b < B; ++b) {
                    std::uint8_t * bd = boundaries.data() + b * T;
                    std::uint8_t * kn = known.data() + b * T;
                    std::uint8_t * nx = next.data() + b * T;
                    remove_mutable_boundaries(bd, kn, T, pv, *rngs[b], nx);
                }
                boundaries = std::move(next);
                for (std::size_t b = 0; b < B; ++b) {
                    const std::uint8_t * bd = boundaries.data() + b * T;
                    const std::uint8_t * ms = mask.data() + b * T;
                    auto regions = game_ggml::boundaries_to_regions(bd, ms, T);
                    for (int t = 0; t < T; ++t)
                        noise_mod[b * T + t] = regions[t] % cfg.region_cycle_len;
                }
                // one batched segmenter call for the WHOLE batch
                {
                    StageProfiler prof; auto _ = prof.scope_segmenter();
                    run_segmenter_batch(x_seg_batch.data(), T, static_cast<int>(B),
                                        noise_mod.data(), ti, languages.data(), logits);
                }
                // sigmoid + decode_soft_boundaries per sample
                for (std::size_t b = 0; b < B; ++b) {
                    const float * lg = logits.data() + b * T;
                    float * pr = probs.data() + b * T;
                    for (int t = 0; t < T; ++t) pr[t] = sigmoid(lg[t]);
                    std::uint8_t * bd = boundaries.data() + b * T;
                    std::uint8_t * kn = known.data() + b * T;
                    std::uint8_t * ms = mask.data() + b * T;
                    auto nb = game_ggml::decode_soft_boundaries(pr, T, kn, ms,
                        params.boundary_threshold, params.boundary_radius);
                    std::memcpy(bd, nb.data(), T);
                }
            }

            // ---- per-sample final decode + estimator (not yet fused) ----
            for (std::size_t b = 0; b < B; ++b) {
                const std::uint8_t * bd = boundaries.data() + b * T;
                const std::uint8_t * ms = mask.data() + b * T;
                auto regions = game_ggml::boundaries_to_regions(bd, ms, T);
                int N = 0;
                for (int t = 0; t < T; ++t) N = std::max<int>(N, regions[t]);

                InferResult res;
                res.num_frames = item_T[b];      // real length, not Tmax
                res.db_cache_hits   = 0;
                res.db_cache_misses = 0;
                if (N > 0) {
                    std::vector<float> pool_logits;
                    const float * xe = x_est_batch.data() + b * (D * T);
                    run_estimator(xe, T, regions.data(), N, pool_logits);
                    const int bins = cfg.estimator_out_dim;
                    std::vector<float> pool_probs(pool_logits.size());
                    for (std::size_t i = 0; i < pool_logits.size(); ++i)
                        pool_probs[i] = sigmoid(pool_logits[i]);
                    auto dec = game_ggml::decode_gaussian_blurred_probs(
                        pool_probs.data(), static_cast<std::size_t>(N), static_cast<std::size_t>(bins),
                        cfg.inference.midi_min, cfg.inference.midi_max,
                        cfg.inference.midi_std * 3.0f,
                        params.note_threshold);
                    const float timestep = cfg.inference.timestep();
                    std::vector<int> dur_frames(N + 1, 0);
                    for (int t = 0; t < T; ++t)
                        if (regions[t] > 0 && regions[t] <= N) ++dur_frames[regions[t]];
                    float offset = 0.0f;
                    for (int n_idx = 0; n_idx < N; ++n_idx) {
                        Note nt;
                        nt.offset_seconds   = offset;
                        nt.duration_seconds = dur_frames[n_idx + 1] * timestep;
                        nt.pitch_midi       = dec.values[n_idx];
                        nt.voiced           = dec.presence[n_idx] != 0;
                        res.notes.push_back(nt);
                        offset += nt.duration_seconds;
                    }
                }
                out.push_back(std::move(res));
            }
            return out;
        }
    }

    // -------- fallback: sequential per-item path --------
    for (std::size_t i = 0; i < items.size(); ++i) {
        const auto & it = items[i];
        internal::MT19937Rng rng(given_seeds[i]);
        InferParams p = params;
        p.seed = 0;
        out.push_back(infer_with_rng(it.waveform, it.n_samples, p, rng));
    }
    return out;
}

}  // namespace game_ggml
