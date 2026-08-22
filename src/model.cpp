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
#include <ggml-cpu.h>
#include <ggml.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
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

// ============================================================================
// Model::Impl — loading
// ============================================================================

Model::Impl::~Impl() {
    // Order matters: every backend-owned resource must be released before the
    // backend itself.  The member destructors would otherwise run AFTER
    // ggml_backend_free(backend) and free device buffers that belong to an
    // already-destroyed backend (CUDA/Vulkan): PersistentStage ~reset() frees
    // the gallocr (whose buffer type came from `backend`) and LoadedWeights
    // frees its backend buffers.  `dbctx`/`dbbuf` have no owner, so release
    // them here explicitly too (leak otherwise, one D×T set per Model).
    enc_stage.reset();
    seg_stage.reset();
    seg_front_stage.reset();
    seg_add_stage.reset();
    seg_mid_stage.reset();
    seg_update_stage.reset();
    seg_back_stage.reset();
    seg_head_stage.reset();
    est_stage.reset();
    if (dbctx) {
        ggml_backend_buffer_free(dbbuf);
        ggml_free(dbctx);
        dbctx = nullptr; dbbuf = nullptr;
    }
    weights.reset();
    if (backend) internal::free_backend(backend);
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

// Debug aid gated by GAME_GGML_DUMP_OPS: list ops the backend cannot run.
// (ggml v0.19 graph introspection takes a non-const cgraph.)
void dump_graph_support(const char * stage, ggml_backend_t backend, ggml_cgraph * graph) {
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

}  // namespace

// ============================================================================
// Stage 1 — encoder (mel → x_seg, x_est)
//
// The graph is built once per frame count T (PersistentStage) and its two
// outputs — x_seg / x_est — live in the encoder's backend buffer, so they
// stay resident on the device and are referenced directly by the segmenter
// and estimator stages (no host round-trip between stages).
// ============================================================================

void Model::Impl::run_encoder(const float * mel, int T)
{
    const int D_mel = cfg.in_dim;

    ensure_db_tensors(T);

    if (!enc_stage.matches(T)) {
        enc_stage.reset();

        ggml_init_params ip{};
        ip.mem_size = 32 * 1024 * 1024;
        ip.no_alloc = true;
        enc_stage.ctx = ggml_init(ip);
        enc_stage.graph = ggml_new_graph_custom(enc_stage.ctx, 8192, /*grads=*/false);

        ggml_tensor * mel_in = ggml_new_tensor_3d(enc_stage.ctx, GGML_TYPE_F32, D_mel, T, 1);
        ggml_set_input(mel_in);
        ggml_tensor * pos = ggml_new_tensor_1d(enc_stage.ctx, GGML_TYPE_I32, T);
        ggml_set_input(pos);

        // spectrogram_projection (mel → D_embed)
        ggml_tensor * x_proj = internal::ops::linear(enc_stage.ctx, mel_in, w_spec_proj, b_spec_proj);
        auto outs = internal::build_encoder_graph(enc_stage.ctx, x_proj, encoder_w, pos, cfg.encoder);

        // Copy the two outputs into the persistent NONE tensors so downstream
        // graphs reference pure leaves (no producer-chain pull-in / recompute).
        ggml_tensor * seg_out = ggml_cpy(enc_stage.ctx, outs.x_seg, x_seg_dev);
        ggml_tensor * est_out = ggml_cpy(enc_stage.ctx, outs.x_est, x_est_dev);
        ggml_set_output(seg_out);
        ggml_set_output(est_out);
        ggml_build_forward_expand(enc_stage.graph, seg_out);
        ggml_build_forward_expand(enc_stage.graph, est_out);
        dump_graph_support("encoder", backend, enc_stage.graph);
        enc_stage.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(enc_stage.alloc, enc_stage.graph)) throw Error("alloc failed (encoder)");

        // Input tensors to refresh on each compute.
        enc_stage.inputs = {mel_in, pos};
        enc_stage.T = T;

        // positions are the fixed iota 0..T-1 — set once at build time.
        std::vector<std::int32_t> pos_i(T);
        for (int i = 0; i < T; ++i) pos_i[i] = i;
        ggml_backend_tensor_set(pos, pos_i.data(), 0, pos_i.size() * sizeof(std::int32_t));
    }

    ggml_backend_tensor_set(enc_stage.inputs[0], mel, 0, ggml_nbytes(enc_stage.inputs[0]));

    if (ggml_backend_graph_compute(backend, enc_stage.graph) != GGML_STATUS_SUCCESS) {
        throw Error("graph compute failed");
    }
}

// ============================================================================
// Stage 2 — segmenter (one D3PM step, DBCache-aware)
// ============================================================================

void Model::Impl::run_segmenter_step(
    int T,
    const std::int32_t * noise_mod3, float t_scalar, int language,
    std::vector<float> & logits_out)
{
    SegmenterCacheState & cache = seg_cache;

    // ---------- Fused fast path (DBCache disabled, the default) ----------
    // One combined graph + a single backend submit is cheaper than the
    // 3-stage host-copy split below: fewer gallocr allocations, no extra
    // host round-trips, and it matches the pre-DBCache behavior exactly.
    // The graph is built once per T and reused across all D3PM steps of the
    // segment; x_seg is read straight from the encoder's device-resident
    // output tensor.
    if (!cache.enabled) {
        if (!seg_stage.matches(T)) {
            seg_stage.reset();

            ggml_init_params ip{};
            ip.mem_size = 32 * 1024 * 1024;
            ip.no_alloc = true;
            seg_stage.ctx = ggml_init(ip);
            seg_stage.graph = ggml_new_graph_custom(seg_stage.ctx, 16384, /*grads=*/false);

            ggml_tensor * noise       = ggml_new_tensor_1d(seg_stage.ctx, GGML_TYPE_I32, T);
            ggml_tensor * t_tensor    = ggml_new_tensor_3d(seg_stage.ctx, GGML_TYPE_F32, 1, 1, 1);
            ggml_tensor * lang_tensor = ggml_new_tensor_1d(seg_stage.ctx, GGML_TYPE_I32, 1);
            ggml_tensor * positions   = ggml_new_tensor_1d(seg_stage.ctx, GGML_TYPE_I32, T);
            for (auto * t : {noise, t_tensor, lang_tensor, positions}) ggml_set_input(t);

            auto outs = internal::build_segmenter_graph(
                seg_stage.ctx, x_seg_dev, noise, t_tensor, lang_tensor, positions, segmenter_w, cfg);

            ggml_set_output(outs.logits);
            ggml_build_forward_expand(seg_stage.graph, outs.logits);
            if (outs.latent) { ggml_set_output(outs.latent); ggml_build_forward_expand(seg_stage.graph, outs.latent); }
            dump_graph_support("segmenter", backend, seg_stage.graph);
            seg_stage.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(seg_stage.alloc, seg_stage.graph)) throw Error("alloc failed (segmenter/fused)");

            seg_stage.inputs = {noise, t_tensor, lang_tensor, positions};
            seg_stage.outs   = {outs.logits};
            std::vector<std::int32_t> pos(T);
            for (int i = 0; i < T; ++i) pos[i] = i;
            ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(std::int32_t));
            seg_stage.T = T;
        }

        ggml_backend_tensor_set(seg_stage.inputs[0], noise_mod3, 0, T * sizeof(std::int32_t));
        ggml_backend_tensor_set(seg_stage.inputs[1], &t_scalar,  0, sizeof(float));
        const std::int32_t l = static_cast<std::int32_t>(language);
        ggml_backend_tensor_set(seg_stage.inputs[2], &l, 0, sizeof(std::int32_t));

        if (ggml_backend_graph_compute(backend, seg_stage.graph) != GGML_STATUS_SUCCESS) {
            throw Error("graph compute failed (segmenter/fused)");
        }

        logits_out.resize(T);
        ggml_backend_tensor_get(seg_stage.outs[0], logits_out.data(), 0, logits_out.size() * sizeof(float));
        if (std::getenv("GAME_GGML_DUMP_LOGITS")) {
            float mn = logits_out[0], mx = logits_out[0], sum = 0.0f;
            int nneg = 0;
            for (int i = 0; i < T; ++i) {
                mn = std::min(mn, logits_out[i]);
                mx = std::max(mx, logits_out[i]);
                sum += logits_out[i];
                if (logits_out[i] < 0.0f) ++nneg;
            }
            std::fprintf(stderr, "[LOGITS] fused step t=%.4f n=%d min=%.4f max=%.4f mean=%.4f nneg=%d\n",
                         t_scalar, T, mn, mx, sum / T, nneg);
        }
        return;
    }

    const int nf = std::min(std::max(cache.fn_blocks, 0), cfg.segmenter.num_layers);
    const int N_seg = cfg.segmenter.num_layers;
    const int nb = std::min(std::max(cache.bn_blocks, 0), N_seg - nf);
    const int middle_end = N_seg - nb;             // middle = [nf, middle_end)

    // ---------- Stage A: front blocks + cache metric ----------
    // x_front stays on the device; only the 1-float L1 metric is read back.
    float fd = std::numeric_limits<float>::infinity();
    {
        if (!seg_front_stage.matches(T, nf)) {
            seg_front_stage.reset();

            ggml_init_params ip{};
            ip.mem_size = 32 * 1024 * 1024;
            ip.no_alloc = true;
            seg_front_stage.ctx = ggml_init(ip);
            seg_front_stage.graph = ggml_new_graph_custom(seg_front_stage.ctx, 8192, /*grads=*/false);

            ggml_tensor * noise       = ggml_new_tensor_1d(seg_front_stage.ctx, GGML_TYPE_I32, T);
            ggml_tensor * t_tensor    = ggml_new_tensor_3d(seg_front_stage.ctx, GGML_TYPE_F32, 1, 1, 1);
            ggml_tensor * lang_tensor = ggml_new_tensor_1d(seg_front_stage.ctx, GGML_TYPE_I32, 1);
            ggml_tensor * positions   = ggml_new_tensor_1d(seg_front_stage.ctx, GGML_TYPE_I32, T);
            ggml_tensor * eps_t       = ggml_new_tensor_1d(seg_front_stage.ctx, GGML_TYPE_F32, 1);
            for (auto * t : {noise, t_tensor, lang_tensor, positions, eps_t}) ggml_set_input(t);

            ggml_tensor * out_front = internal::build_segmenter_front_graph(
                seg_front_stage.ctx, x_seg_dev, noise, t_tensor, lang_tensor,
                positions, segmenter_w, cfg, nf);

            // Copy into the persistent NONE tensor so the metric chain and
            // the mid/add/update graphs reference a pure leaf (no producer
            // chain pull-in).
            ggml_tensor * xf = ggml_cpy(seg_front_stage.ctx, out_front, x_front_dev);

            // Device-side L1 metric — mirrors host front_delta() up to float
            // summation order: fd = sum|x - prev| / (sum|prev| + eps).
            ggml_tensor * diff  = ggml_abs(seg_front_stage.ctx,
                ggml_sub(seg_front_stage.ctx, x_front_dev, prev_front_dev));
            ggml_tensor * num   = ggml_sum(seg_front_stage.ctx, diff);
            ggml_tensor * den   = ggml_sum(seg_front_stage.ctx,
                ggml_abs(seg_front_stage.ctx, prev_front_dev));
            ggml_tensor * fd_t  = ggml_div(seg_front_stage.ctx, num,
                ggml_add(seg_front_stage.ctx, den, eps_t));

            ggml_set_output(xf);
            ggml_build_forward_expand(seg_front_stage.graph, xf);
            ggml_set_output(fd_t);
            ggml_build_forward_expand(seg_front_stage.graph, fd_t);
            dump_graph_support("segmenter/front", backend, seg_front_stage.graph);
            seg_front_stage.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(seg_front_stage.alloc, seg_front_stage.graph))
                throw Error("alloc failed (segmenter/front)");

            seg_front_stage.inputs = {noise, t_tensor, lang_tensor, positions, eps_t};
            seg_front_stage.outs   = {xf, fd_t};
            std::vector<std::int32_t> pos(T);
            for (int i = 0; i < T; ++i) pos[i] = i;
            ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(std::int32_t));
            const float eps = 1e-8f;
            ggml_backend_tensor_set(eps_t, &eps, 0, sizeof(float));
            seg_front_stage.T = T;
            seg_front_stage.key = nf;
        }

        ggml_backend_tensor_set(seg_front_stage.inputs[0], noise_mod3, 0, T * sizeof(std::int32_t));
        ggml_backend_tensor_set(seg_front_stage.inputs[1], &t_scalar,  0, sizeof(float));
        const std::int32_t l0 = static_cast<std::int32_t>(language);
        ggml_backend_tensor_set(seg_front_stage.inputs[2], &l0, 0, sizeof(std::int32_t));

        if (ggml_backend_graph_compute(backend, seg_front_stage.graph) != GGML_STATUS_SUCCESS) {
            throw Error("graph compute failed (segmenter/front)");
        }
        if (cache.valid) {
            ggml_backend_tensor_get(seg_front_stage.outs[1], &fd, 0, sizeof(float));
        }
    }

    // ---------- Decide cache hit ----------
    bool use_cache = false;
    if (cache.enabled && cache.valid && cache.step >= cache.warmup) {
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

    // ---------- Stage B: tail blocks (all device-resident) ----------
    //  * add graph   (hit):   x_mid_dev = x_front + tail_delta
    //  * mid graph   (miss):  x_mid_dev = tail(middle)(x_front)
    //  * update graph(miss):  tail_delta_dev = x_mid - x_front; prev_front_dev = x_front
    //  * back graph:          x_out_dev = tail(back)(x_mid_dev)
    // All reference device-resident tensors as leaves — no D×T host transfer
    // on the cache path.

    if (!seg_add_stage.matches(T, nb ? 1 : 0)) {
        seg_add_stage.reset();
        ggml_init_params ip{};
        ip.mem_size = 8 * 1024 * 1024;
        ip.no_alloc = true;
        seg_add_stage.ctx = ggml_init(ip);
        seg_add_stage.graph = ggml_new_graph_custom(seg_add_stage.ctx, 2048, /*grads=*/false); // visited-set must hold the referenced encoder/front chains
        ggml_tensor * mid = ggml_add(seg_add_stage.ctx, x_front_dev, tail_delta_dev);
        ggml_tensor * md  = ggml_cpy(seg_add_stage.ctx, mid, x_mid_dev);
        ggml_set_output(md);
        ggml_build_forward_expand(seg_add_stage.graph, md);
        if (nb == 0) {   // no back slice: head reads x_out_dev straight away
            ggml_tensor * od = ggml_cpy(seg_add_stage.ctx, x_mid_dev, x_out_dev);
            ggml_set_output(od);
            ggml_build_forward_expand(seg_add_stage.graph, od);
        }
        dump_graph_support("segmenter/add", backend, seg_add_stage.graph);
        seg_add_stage.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(seg_add_stage.alloc, seg_add_stage.graph))
            throw Error("alloc failed (segmenter/add)");
        seg_add_stage.outs = {md};
        seg_add_stage.T = T;
        seg_add_stage.key = nb ? 1 : 0;
    }

    if (!seg_mid_stage.matches(T, (static_cast<int64_t>(nf) << 20) | middle_end)) {
        seg_mid_stage.reset();
        ggml_init_params ip{};
        ip.mem_size = 32 * 1024 * 1024;
        ip.no_alloc = true;
        seg_mid_stage.ctx = ggml_init(ip);
        seg_mid_stage.graph = ggml_new_graph_custom(seg_mid_stage.ctx, 16384, /*grads=*/false);
        ggml_tensor * positions = ggml_new_tensor_1d(seg_mid_stage.ctx, GGML_TYPE_I32, T);
        ggml_set_input(positions);
        auto outs = internal::build_segmenter_tail_graph(
            seg_mid_stage.ctx, x_front_dev, positions, segmenter_w, cfg, nf, middle_end);
        ggml_tensor * md = ggml_cpy(seg_mid_stage.ctx, outs.x_run, x_mid_dev);
        ggml_set_output(md);
        ggml_build_forward_expand(seg_mid_stage.graph, md);
        dump_graph_support("segmenter/mid", backend, seg_mid_stage.graph);
        seg_mid_stage.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(seg_mid_stage.alloc, seg_mid_stage.graph))
            throw Error("alloc failed (segmenter/mid)");
        seg_mid_stage.inputs = {positions};
                seg_mid_stage.outs   = {md};
        if (middle_end > nf) {   // empty range => x_run = x_front passthrough, positions unused
            std::vector<std::int32_t> pos(T);
            for (int i = 0; i < T; ++i) pos[i] = i;
            ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(std::int32_t));
        }
        seg_mid_stage.T = T;
        seg_mid_stage.key = (static_cast<int64_t>(nf) << 20) | middle_end;
    }

    if (!seg_update_stage.matches(T, nb ? 1 : 0)) {
        seg_update_stage.reset();
        ggml_init_params ip{};
        ip.mem_size = 8 * 1024 * 1024;
        ip.no_alloc = true;
        seg_update_stage.ctx = ggml_init(ip);
        seg_update_stage.graph = ggml_new_graph_custom(seg_update_stage.ctx, 2048, /*grads=*/false);
        ggml_tensor * sub = ggml_sub(seg_update_stage.ctx, x_mid_dev, x_front_dev);
        ggml_tensor * td  = ggml_cpy(seg_update_stage.ctx, sub, tail_delta_dev);
        ggml_tensor * pf  = ggml_cpy(seg_update_stage.ctx, x_front_dev, prev_front_dev);
        ggml_set_output(td);
        ggml_build_forward_expand(seg_update_stage.graph, td);
        ggml_set_output(pf);
        ggml_build_forward_expand(seg_update_stage.graph, pf);
        if (nb == 0) {
            ggml_tensor * od = ggml_cpy(seg_update_stage.ctx, x_mid_dev, x_out_dev);
            ggml_set_output(od);
            ggml_build_forward_expand(seg_update_stage.graph, od);
        }
        dump_graph_support("segmenter/update", backend, seg_update_stage.graph);
        seg_update_stage.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(seg_update_stage.alloc, seg_update_stage.graph))
            throw Error("alloc failed (segmenter/update)");
        seg_update_stage.T = T;
        seg_update_stage.key = nb ? 1 : 0;
    }

    if (nb > 0 && !seg_back_stage.matches(T, (static_cast<int64_t>(middle_end) << 20) | N_seg)) {
        seg_back_stage.reset();
        ggml_init_params ip{};
        ip.mem_size = 32 * 1024 * 1024;
        ip.no_alloc = true;
        seg_back_stage.ctx = ggml_init(ip);
        seg_back_stage.graph = ggml_new_graph_custom(seg_back_stage.ctx, 16384, /*grads=*/false);
        ggml_tensor * positions = ggml_new_tensor_1d(seg_back_stage.ctx, GGML_TYPE_I32, T);
        ggml_set_input(positions);
        auto outs = internal::build_segmenter_tail_graph(
            seg_back_stage.ctx, x_mid_dev, positions, segmenter_w, cfg, middle_end, N_seg);
        ggml_tensor * od = ggml_cpy(seg_back_stage.ctx, outs.x_run, x_out_dev);
        ggml_set_output(od);
        ggml_build_forward_expand(seg_back_stage.graph, od);
        if (outs.latent) { ggml_set_output(outs.latent); ggml_build_forward_expand(seg_back_stage.graph, outs.latent); }
        dump_graph_support("segmenter/back", backend, seg_back_stage.graph);
        seg_back_stage.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(seg_back_stage.alloc, seg_back_stage.graph))
            throw Error("alloc failed (segmenter/back)");
        seg_back_stage.inputs = {positions};
                seg_back_stage.outs   = {od};
        std::vector<std::int32_t> pos(T);
        for (int i = 0; i < T; ++i) pos[i] = i;
        ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(std::int32_t));
        seg_back_stage.T = T;
        seg_back_stage.key = (static_cast<int64_t>(middle_end) << 20) | N_seg;
    }

    if (use_cache) {
        if (ggml_backend_graph_compute(backend, seg_add_stage.graph) != GGML_STATUS_SUCCESS) {
            throw Error("graph compute failed (segmenter/add)");
        }
        if (nb > 0) {
            if (ggml_backend_graph_compute(backend, seg_back_stage.graph) != GGML_STATUS_SUCCESS) {
                throw Error("graph compute failed (segmenter/back)");
            }
        }
        ++cache.hits;
        ++cache.cont_cnt;
    } else {
        if (ggml_backend_graph_compute(backend, seg_mid_stage.graph) != GGML_STATUS_SUCCESS) {
            throw Error("graph compute failed (segmenter/mid)");
        }
        if (ggml_backend_graph_compute(backend, seg_update_stage.graph) != GGML_STATUS_SUCCESS) {
            throw Error("graph compute failed (segmenter/update)");
        }
        if (nb > 0) {
            if (ggml_backend_graph_compute(backend, seg_back_stage.graph) != GGML_STATUS_SUCCESS) {
                throw Error("graph compute failed (segmenter/back)");
            }
        }
        cache.valid = true;
        cache.acc_err = 0.0f;
        cache.cont_cnt = 0;
        ++cache.misses;
    }

    // ---------- Stage C: head (output norm + proj), input = device x_out_dev ----------
    {
        if (!seg_head_stage.matches(T)) {
            seg_head_stage.reset();

            ggml_init_params ip{};
            ip.mem_size = 16 * 1024 * 1024;
            ip.no_alloc = true;
            seg_head_stage.ctx = ggml_init(ip);
            seg_head_stage.graph = ggml_new_graph_custom(seg_head_stage.ctx, 256, /*grads=*/false);

            ggml_tensor * logits = internal::build_segmenter_head_graph(
                seg_head_stage.ctx, x_out_dev, segmenter_w, cfg);

            ggml_set_output(logits);
            ggml_build_forward_expand(seg_head_stage.graph, logits);
            dump_graph_support("segmenter/head", backend, seg_head_stage.graph);
            seg_head_stage.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(seg_head_stage.alloc, seg_head_stage.graph))
                throw Error("alloc failed (segmenter/head)");

            seg_head_stage.outs = {logits};
            seg_head_stage.T = T;
        }

        if (ggml_backend_graph_compute(backend, seg_head_stage.graph) != GGML_STATUS_SUCCESS) {
            throw Error("graph compute failed (segmenter/head)");
        }

        logits_out.resize(T);
        ggml_backend_tensor_get(seg_head_stage.outs[0], logits_out.data(),
                                0, logits_out.size() * sizeof(float));
    }

    ++cache.step;
    if (cache.enabled && std::getenv("GAME_GGML_DUMP_DBCACHE")) {
        std::fprintf(stderr, "[DBCACHE] step=%d %s (hits=%d misses=%d)\n",
            cache.step, use_cache ? "HIT" : "MISS", cache.hits, cache.misses);
    }
}

// ============================================================================
// DBCache device-resident tensors
// ============================================================================

void Model::Impl::ensure_db_tensors(int T) {
    const int D = cfg.embedding_dim;
    if (db_T == T && x_seg_dev) return;

    if (dbctx) {
        ggml_backend_buffer_free(dbbuf);
        ggml_free(dbctx);
        dbctx = nullptr; dbbuf = nullptr;
        x_seg_dev = x_est_dev = x_front_dev = nullptr;
        prev_front_dev = tail_delta_dev = x_mid_dev = x_out_dev = nullptr;
    }
    ggml_init_params ip{};
    ip.mem_size = 128 * 1024;
    ip.no_alloc = true;
    dbctx = ggml_init(ip);
    x_seg_dev      = ggml_new_tensor_3d(dbctx, GGML_TYPE_F32, D, T, 1);
    x_est_dev      = ggml_new_tensor_3d(dbctx, GGML_TYPE_F32, D, T, 1);
    x_front_dev    = ggml_new_tensor_3d(dbctx, GGML_TYPE_F32, D, T, 1);
    prev_front_dev = ggml_new_tensor_3d(dbctx, GGML_TYPE_F32, D, T, 1);
    tail_delta_dev = ggml_new_tensor_3d(dbctx, GGML_TYPE_F32, D, T, 1);
    x_mid_dev      = ggml_new_tensor_3d(dbctx, GGML_TYPE_F32, D, T, 1);
    x_out_dev      = ggml_new_tensor_3d(dbctx, GGML_TYPE_F32, D, T, 1);
    dbbuf = ggml_backend_alloc_ctx_tensors(dbctx, backend);
    if (!dbbuf) throw Error("failed to allocate device-resident tensors");
    // Zero prev_front so the (unread until cache.valid) metric node never
    // consumes uninitialised device memory.
    std::vector<float> zeros(static_cast<std::size_t>(D) * T, 0.0f);
    ggml_backend_tensor_set(prev_front_dev, zeros.data(), 0, zeros.size() * sizeof(float));
    db_T = T;
}

// ============================================================================
// Stage 3 — estimator (regions → pool_logits)
// ============================================================================

void Model::Impl::run_estimator(
    int T, const std::int32_t * regions, int N,
    std::vector<float> & pool_logits_out)
{
    const int S = N + T;

    if (!est_stage.matches(T, N)) {
        est_stage.reset();

        ggml_init_params ip{};
        ip.mem_size = 32 * 1024 * 1024;
        ip.no_alloc = true;
        est_stage.ctx = ggml_init(ip);
        est_stage.graph = ggml_new_graph_custom(est_stage.ctx, 16384, /*grads=*/false);

        ggml_tensor * regions_mod = ggml_new_tensor_1d(est_stage.ctx, GGML_TYPE_I32, T);
        ggml_tensor * positions   = ggml_new_tensor_1d(est_stage.ctx, GGML_TYPE_I32, S);
        ggml_tensor * region_ids  = ggml_new_tensor_1d(est_stage.ctx, GGML_TYPE_I32, S);
        ggml_tensor * mask_fp16   = ggml_new_tensor_4d(est_stage.ctx, GGML_TYPE_F16, S, S, 1, 1);
        for (auto * t : {regions_mod, positions, region_ids, mask_fp16}) ggml_set_input(t);

        // x_est input is the device-resident encoder output — referenced
        // cross-context, never copied.
        auto outs = internal::build_estimator_graph(
            est_stage.ctx, x_est_dev, regions_mod, positions, region_ids, mask_fp16,
            N, estimator_w, cfg);

        ggml_set_output(outs.pool_logits);
        ggml_build_forward_expand(est_stage.graph, outs.pool_logits);
        dump_graph_support("estimator", backend, est_stage.graph);
        est_stage.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(est_stage.alloc, est_stage.graph)) throw Error("alloc failed (estimator)");

        est_stage.inputs = {regions_mod, positions, region_ids, mask_fp16};
        est_stage.outs   = {outs.pool_logits};
        est_stage.T = T;
        est_stage.key = N;
    }

    // Host-side prep.
    std::vector<std::int32_t> rmod(T);
    for (int i = 0; i < T; ++i) rmod[i] = regions[i] % cfg.region_cycle_len;
    ggml_backend_tensor_set(est_stage.inputs[0], rmod.data(), 0, rmod.size() * sizeof(std::int32_t));

    // Global positions: pool 0..N-1, x 0..T-1
    std::vector<std::int32_t> gpos(S);
    for (int i = 0; i < N; ++i) gpos[i] = i;
    for (int i = 0; i < T; ++i) gpos[N + i] = i;
    ggml_backend_tensor_set(est_stage.inputs[1], gpos.data(), 0, gpos.size() * sizeof(std::int32_t));

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
    ggml_backend_tensor_set(est_stage.inputs[2], ridx.data(), 0, ridx.size() * sizeof(std::int32_t));

    auto mask = internal::ops::build_joint_attn_mask_fp16(regions, T, N);
    ggml_backend_tensor_set(est_stage.inputs[3], mask.data(), 0, mask.size() * sizeof(std::uint16_t));

    if (ggml_backend_graph_compute(backend, est_stage.graph) != GGML_STATUS_SUCCESS) {
        throw Error("graph compute failed (estimator)");
    }

    pool_logits_out.resize(ggml_nelements(est_stage.outs[0]));
    ggml_backend_tensor_get(est_stage.outs[0], pool_logits_out.data(),
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
    //        x_seg / x_est stay resident in the encoder's backend buffer.
    {
        auto _ = prof.scope_encoder();
        auto mel = mel_extractor->forward(waveform, n_samples);          // [T, 80]
        run_encoder(mel.data(), T);
    }

    // --- 3) D3PM loop (segmenter)
    std::vector<float> ts = params.d3pm_ts.empty()
        ? default_d3pm_schedule(params.d3pm_t0, params.d3pm_nsteps)
        : params.d3pm_ts;

    // DBCache: configure + reset per segment (mirrors PyTorch reset on
    // forward_segmenter_main).  Only meaningful for multi-step D3PM loops:
    // with a single step there is nothing to cache, so the fused single-graph
    // path stays active (cache.enabled==false) even when a threshold is set
    // — that avoids paying the 3-stage split cost for --nsteps 1.
    // Default 0.25 on every backend: the DBCache decision metric and the
    // middle reconstruction now run fully on-device (1-float readback), so
    // GPU backends no longer pay the D×T host round-trip that regressed
    // quantized weights (+20% measured on Vulkan+Q8 with the old host-side
    // path).  Verified on Vulkan/RTX 2070: device-side split path matches
    // CPU note output exactly.
    float thr = params.db_cache_threshold;
    if (thr < 0.0f) thr = 0.25f;
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
            run_segmenter_step(T,
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
    // DBCache hit/miss for this inference (paths before the early return also
    // carry the counters already reflected above).
    result.db_cache_hits   = seg_cache.hits;
    result.db_cache_misses = seg_cache.misses;
    if (N == 0) return result;

    // --- 5) estimator
    std::vector<float> pool_logits;
    { auto _ = prof.scope_estimator();
      run_estimator(T, regions.data(), N, pool_logits); }

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

}  // namespace game_ggml
