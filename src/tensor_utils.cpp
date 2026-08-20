#include "tensor_utils.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>
#include <gguf.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace game_ggml::internal {

namespace {

// Depthwise-conv weights are stored F16 in the GGUF (converter constraint)
// but the dedicated conv_2d_dw kernels want an F32 kernel.  Casting inside
// the graph would re-run a conversion node on every stage compute (once per
// D3PM step, per layer).  Instead we materialise a persistent F32 copy at
// load time and point the weight table at it; the graph then sees F32 and
// builds zero cast nodes.
bool is_dwconv_weight(const char * name) {
    return (std::strstr(name, ".dw.") != nullptr || std::strstr(name, "dw_conv") != nullptr) &&
           std::strstr(name, ".weight") != nullptr;
}

// F-1: GLU FFN first linear ("ln1").  The [in, 2L] weight and [2L] bias are
// split at load time into two [in, L] / [L] halves so the graph can run two
// mul_mats that produce contiguous outputs, eliminating the two cont copies
// per GLU FFN (see glu_ffn_split in ops_ffn.cpp).
bool is_glu_ln1_weight(const char * name) {
    return std::strstr(name, ".ln1.weight") != nullptr && !is_dwconv_weight(name);
}
bool is_glu_ln1_bias(const char * name) {
    return std::strstr(name, ".ln1.bias") != nullptr;
}

// ---------------------------------------------------------------------------
// F-2: EBF layer-scale folding (load-time, schema-preserving).
//
// The EBF residual is  x + 0.5 * lay_scale(branch)  where lay_scale is a
// per-channel multiply.  Since 0.5 and lay_scale are both diagonal, they can
// be folded into the *producing* linear of the branch at load time:
//
//     out' = 0.5 * s ⊙ (W·h + b)  ==  (0.5·s·W)·h + (0.5·s⊙b)
//
// so the graph no longer emits a lay_scale mul + a 0.5 scale node per EBF
// block (two elementwise kernels per FFN, one per PAC branch).  The GGUF
// keeps its lay_scale tensors (bind code still finds them) — they are simply
// no longer referenced by the graph.  This is idempotent: the file is never
// modified, every load folds the same way.
//
// The fold is lossless for Q8_0 (only the per-block d scalars change) and
// exact for F32/F16 up to float rounding; the graph arithmetic order changes
// (scale applied before the matmul instead of after), so outputs are
// expected to match the unfolded graph to ~1e-7, not bit-exactly.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// F-2: layer-scale folding (load-time, schema-preserving) — covers both the
// single-stream EBF blocks (encoder/segmenter) and the joint EBF (estimator):
//
//   * single-stream EBF residual  x + 0.5·lay_scale(branch)     → mult 0.5
//   * joint EBF residual          x + lay_scale(branch)         → mult 1.0
//
// lay_scale is a per-channel multiply, so it folds into the branch's
// producing linear at load time (see F-2 note above).  `parse_lay_scale_name`
// maps a GGUF lay_scale tensor name to its producing linear's weight/bias
// tensor names (relative to the block base, with leading '.') and the
// extra multiplier.  Returns false for non-lay_scale names.
// ---------------------------------------------------------------------------
struct FoldTarget {
    std::string w_suffix;   // e.g. ".ffn1.ln2.weight" (base + suffix = full name)
    std::string b_suffix;
    float mult;             // extra factor applied to the lay_scale values
};

bool parse_lay_scale_name(const std::string & name, std::string & base_out,
                          FoldTarget & t_out) {
    // ---- Single-stream EBF: {base}.lay_scale{1,2,3}.scale ----
    const std::string ebf = ".lay_scale";
    const std::size_t ep = name.rfind(ebf);
    if (ep != std::string::npos) {
        const std::string tail = name.substr(ep + ebf.size());  // "N.scale"
        if (tail.size() == 7 && tail[1] == '.' &&
            tail.compare(1, std::string::npos, ".scale") == 0) {
            const int which = tail[0] - '0';
            if (which >= 1 && which <= 3) {
                base_out = name.substr(0, ep);
                if (which == 1) {
                    t_out.w_suffix = ".ffn1.ln2.weight";  t_out.b_suffix = ".ffn1.ln2.bias";
                } else if (which == 2) {
                    t_out.w_suffix = ".attn.merge_linear.weight"; t_out.b_suffix = ".attn.merge_linear.bias";
                } else {
                    t_out.w_suffix = ".ffn2.ln2.weight";  t_out.b_suffix = ".ffn2.ln2.bias";
                }
                t_out.mult = (which == 2) ? 1.0f : 0.5f;
                return true;
            }
        }
    }

    // ---- Joint EBF (estimator): {base}.lay_scale_{kind}.scale ----
    //   kind ∈ {ffn1_x, ffn1_pool, ffn2_x, ffn2_pool, jpac_x, jpac_pool}
    const std::string jp = ".lay_scale_";
    const std::size_t jpos = name.rfind(jp);
    if (jpos != std::string::npos) {
        const std::string kind = name.substr(jpos + jp.size());  // e.g. "ffn1_x.scale"
        if (kind.size() >= 7 && kind.compare(kind.size() - 6, 6, ".scale") == 0) {
            const std::string k = kind.substr(0, kind.size() - 6);  // "ffn1_x"
            base_out = name.substr(0, jpos);
            if (k.size() >= 6 && k.compare(0, 3, "ffn") == 0 &&
                (k[3] == '1' || k[3] == '2') && k[4] == '_') {
                // k = "ffn{1,2}_{x,pool}"
                const std::string stream = k.substr(5);
                if (stream == "x" || stream == "pool") {
                    t_out.w_suffix = ".ffn" + k.substr(3, 1) + "_" + stream + ".ln2.weight";
                    t_out.b_suffix = ".ffn" + k.substr(3, 1) + "_" + stream + ".ln2.bias";
                    t_out.mult = 1.0f;
                    return true;
                }
            }
            if (k.size() >= 6 && k.compare(0, 4, "jpac") == 0 && k[4] == '_') {
                const std::string stream = k.substr(5);
                if (stream == "x" || stream == "pool") {
                    t_out.w_suffix = ".attn.merge_linear_" + stream + ".weight";
                    t_out.b_suffix = ".attn.merge_linear_" + stream + ".bias";
                    t_out.mult = 1.0f;
                    return true;
                }
            }
        }
    }
    return false;
}

// Multiply the rows of a weight tensor (ne[0] = in dim, contiguous) by a
// per-output-channel factor vector.  In-place on the raw payload bytes.
void fold_linear_weight(ggml_tensor * t, std::vector<std::uint8_t> & data,
                        const std::vector<float> & factor) {
    const int64_t D_in  = t->ne[0];
    const int64_t D_out = t->ne[1];
    if (D_out != static_cast<int64_t>(factor.size())) {
        throw GgufError(std::string("fold: row count mismatch on '") + t->name +
                        "' (" + std::to_string(D_out) + " vs " +
                        std::to_string(factor.size()) + ")");
    }
    switch (t->type) {
        case GGML_TYPE_F32: {
            float * p = reinterpret_cast<float *>(data.data());
            for (int64_t o = 0; o < D_out; ++o) {
                const float f = factor[static_cast<std::size_t>(o)];
                float * row = p + o * D_in;
                for (int64_t i = 0; i < D_in; ++i) row[i] *= f;
            }
        } break;
        case GGML_TYPE_F16: {
            ggml_fp16_t * p = reinterpret_cast<ggml_fp16_t *>(data.data());
            for (int64_t o = 0; o < D_out; ++o) {
                const float f = factor[static_cast<std::size_t>(o)];
                ggml_fp16_t * row = p + o * D_in;
                for (int64_t i = 0; i < D_in; ++i)
                    row[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(row[i]) * f);
            }
        } break;
        case GGML_TYPE_Q8_0: {
            // block_q8_0: { fp16 d; int8 qs[QK8_0] } — d is the row scale of
            // the block, so a per-row multiplier becomes a per-block d
            // change.  Lossless.
            constexpr int64_t QK = 32;
            if (D_in % QK != 0)
                throw GgufError("fold: Q8_0 weight requires ne[0] % 32 == 0");
            const int64_t bpr = D_in / QK;
            struct BQ8 { ggml_fp16_t d; int8_t qs[QK]; };
            BQ8 * p = reinterpret_cast<BQ8 *>(data.data());
            for (int64_t o = 0; o < D_out; ++o) {
                const float f = factor[static_cast<std::size_t>(o)];
                BQ8 * row = p + o * bpr;
                for (int64_t b = 0; b < bpr; ++b)
                    row[b].d = ggml_fp32_to_fp16(ggml_fp16_to_fp32(row[b].d) * f);
            }
        } break;
        default:
            throw GgufError(std::string("fold: unsupported weight type '") +
                            ggml_type_name(t->type) + "' on '" + t->name + "'");
    }
}

void fold_linear_bias(ggml_tensor * t, std::vector<std::uint8_t> & data,
                      const std::vector<float> & factor) {
    if (t->ne[0] != static_cast<int64_t>(factor.size())) {
        throw GgufError(std::string("fold: bias dim mismatch on '") + t->name + "'");
    }
    switch (t->type) {
        case GGML_TYPE_F32: {
            float * p = reinterpret_cast<float *>(data.data());
            for (std::size_t i = 0; i < factor.size(); ++i) p[i] *= factor[i];
        } break;
        case GGML_TYPE_F16: {
            ggml_fp16_t * p = reinterpret_cast<ggml_fp16_t *>(data.data());
            for (std::size_t i = 0; i < factor.size(); ++i)
                p[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(p[i]) * factor[i]);
        } break;
        default:
            throw GgufError(std::string("fold: unsupported bias type '") +
                            ggml_type_name(t->type) + "' on '" + t->name + "'");
    }
}

}  // namespace

LoadedWeights LoadedWeights::load_all(const GgufFile & gguf, ggml_backend_t backend) {
    // --- 1. Build a metadata-only ggml_context that mirrors the file's
    //        tensor table.  gguf_init_from_file with params.ctx != null will
    //        create ggml_tensor entries in that context but leave them
    //        un-backed (no_alloc=true).  It sizes the context exactly for the
    //        tensor table, so no extra tensors may be added to it afterwards.
    ggml_context * ctx = nullptr;
    gguf_init_params gp{};
    gp.no_alloc = true;
    gp.ctx = &ctx;

    gguf_context * gctx = gguf_init_from_file(gguf.path().c_str(), gp);
    if (!gctx) {
        throw GgufError("failed to re-open GGUF for tensor loading: " + gguf.path());
    }

    // --- 1b. Create persistent F32 copies of F16 depthwise-conv weights
    //          (see is_dwconv_weight above), plus F-1 GLU ln1 split halves
    //          (a/b).  The gguf-init context is sized exactly for its tensor
    //          table, so these copies get their own context + backend buffer.
    std::map<std::string, ggml_tensor *> dw_f32;   // original name -> F32 copy
    std::map<std::string, std::pair<ggml_tensor *, ggml_tensor *>> glu_ab;       // ln1 weight -> {a, b}
    std::map<std::string, std::pair<ggml_tensor *, ggml_tensor *>> glu_ab_bias;  // ln1 bias  -> {a, b}
    ggml_context * ctx2 = nullptr;
    ggml_backend_buffer_t buf2 = nullptr;
    {
        ggml_init_params ip{};
        ip.mem_size   = 512 * 1024;   // metadata for dwconv copies + GLU splits
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
        ctx2 = ggml_init(ip);
        if (!ctx2) throw GgufError("failed to create weight-copy context");

        const int64_t n_tensors = gguf_get_n_tensors(gctx);
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gctx, i);
            ggml_tensor * t = ggml_get_tensor(ctx, name);
            if (!t) continue;
            if (t->type == GGML_TYPE_F16 && is_dwconv_weight(name)) {
                ggml_tensor * w32 = ggml_new_tensor_3d(ctx2, GGML_TYPE_F32,
                    t->ne[0], t->ne[1], t->ne[2]);
                dw_f32.emplace(name, w32);
            } else if (is_glu_ln1_weight(name) &&
                       (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16)) {
                // w_ln1 ne = [in, 2L] (column-major, out dim contiguous) ->
                // a = first L out-columns, b = last L out-columns.
                const int64_t L = t->ne[1] / 2;
                ggml_tensor * a = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, t->ne[0], L);
                ggml_tensor * b = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, t->ne[0], L);
                glu_ab.emplace(name, std::make_pair(a, b));
            } else if (is_glu_ln1_bias(name) &&
                       (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16)) {
                const int64_t L = t->ne[0] / 2;
                ggml_tensor * a = ggml_new_tensor_1d(ctx2, GGML_TYPE_F32, L);
                ggml_tensor * b = ggml_new_tensor_1d(ctx2, GGML_TYPE_F32, L);
                glu_ab_bias.emplace(name, std::make_pair(a, b));
            }
        }
        if (!dw_f32.empty() || !glu_ab.empty() || !glu_ab_bias.empty()) {
            buf2 = ggml_backend_alloc_ctx_tensors(ctx2, backend);
            if (!buf2) {
                ggml_free(ctx2);
                gguf_free(gctx);
                ggml_free(ctx);
                throw GgufError("failed to allocate weight-copy buffer");
            }
        }
    }

    // --- 2. Allocate backend buffer covering all tensors in ctx.
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        if (buf2) ggml_backend_buffer_free(buf2);
        ggml_free(ctx2);
        gguf_free(gctx);
        ggml_free(ctx);
        throw GgufError("ggml_backend_alloc_ctx_tensors failed; out of memory?");
    }

    // --- 3. Upload tensor payloads directly from the file.
    FILE * f = std::fopen(gguf.path().c_str(), "rb");
    if (!f) {
        ggml_backend_buffer_free(buf);
        if (buf2) ggml_backend_buffer_free(buf2);
        ggml_free(ctx2);
        gguf_free(gctx);
        ggml_free(ctx);
        throw GgufError("failed to open GGUF payload file: " + gguf.path());
    }

    const size_t data_offset = gguf_get_data_offset(gctx);
    std::vector<std::uint8_t> scratch;

    // --- 3a. F-2: EBF layer-scale fold plan.  Scan the GGUF tensor table
    //           for encoder/segmenter EBF lay_scale tensors, read their
    //           payloads once, and record which producing linear weights and
    //           biases to fold.  The upload loop below applies the fold and
    //           reuses the cached lay_scale payload (so the file is read
    //           once per tensor).  fold_weights/fold_biases are keyed by
    //           tensor name; the value is the per-channel factor vector.
    std::map<std::string, std::vector<float>>         fold_weights;
    std::map<std::string, std::vector<float>>         fold_biases;
    std::map<std::string, std::vector<std::uint8_t>>  ls_cache;
    {
        std::vector<std::uint8_t> ls_scratch;
        const int64_t n_tensors = gguf_get_n_tensors(gctx);
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gctx, i);
            std::string base;
            FoldTarget tgt;
            if (!parse_lay_scale_name(name, base, tgt)) continue;
            ggml_tensor * t = ggml_get_tensor(ctx, name);
            if (!t) continue;

            const size_t bytes  = ggml_nbytes(t);
            const size_t offset = data_offset + gguf_get_tensor_offset(gctx, i);
            ls_scratch.resize(bytes);
            if (std::fseek(f, static_cast<long>(offset), SEEK_SET) != 0 ||
                std::fread(ls_scratch.data(), 1, bytes, f) != bytes) {
                throw GgufError(std::string("short read for lay_scale '") + name + "'");
            }

            // Per-channel scale as f32.
            const int64_t D = t->ne[0];
            std::vector<float> s(static_cast<std::size_t>(D));
            if (t->type == GGML_TYPE_F32) {
                std::memcpy(s.data(), ls_scratch.data(), bytes);
            } else if (t->type == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(ls_scratch.data()),
                                      s.data(), D);
            } else {
                throw GgufError(std::string("fold: unsupported lay_scale type '") +
                                ggml_type_name(t->type) + "' on '" + name + "'");
            }

            // factor = mult · s  (0.5 for single-stream FFN branches, 1.0 for
            // PAC/PJAC and joint-FFN branches).
            std::vector<float> factor = s;
            if (tgt.mult != 1.0f) {
                for (auto & v : factor) v *= tgt.mult;
            }

            const std::string w_name = base + tgt.w_suffix;
            if (!ggml_get_tensor(ctx, w_name.c_str())) {
                throw GgufError("fold: missing target '" + w_name + "' for '" + name + "'");
            }
            fold_weights.emplace(w_name, factor);
            const std::string b_name = base + tgt.b_suffix;
            if (ggml_get_tensor(ctx, b_name.c_str())) {
                fold_biases.emplace(b_name, factor);
            }
            ls_cache.emplace(name, ls_scratch);
        }
        if (!fold_weights.empty()) {
            std::fprintf(stderr, "[FOLD] folded %zu lay_scale(s) into producing linears\n",
                         fold_weights.size());
        }
    }

    LoadedWeights out;
    out.ctx_     = ctx;
    out.buffer_  = buf;
    out.ctx2_    = ctx2;
    out.buffer2_ = buf2;

    const int64_t n_tensors = gguf_get_n_tensors(gctx);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gctx, i);
        ggml_tensor * t = ggml_get_tensor(ctx, name);
        if (!t) {
            std::fclose(f);
            ggml_backend_buffer_free(buf);
            if (buf2) ggml_backend_buffer_free(buf2);
            if (ctx2) ggml_free(ctx2);
            gguf_free(gctx);
            ggml_free(ctx);
            throw GgufError(std::string("tensor '") + name + "' missing from ggml context");
        }
        const size_t bytes  = ggml_nbytes(t);
        const size_t offset = data_offset + gguf_get_tensor_offset(gctx, i);
        scratch.resize(bytes);
        if (std::fseek(f, static_cast<long>(offset), SEEK_SET) != 0 ||
            std::fread(scratch.data(), 1, bytes, f) != bytes) {
            std::fclose(f);
            ggml_backend_buffer_free(buf);
            if (buf2) ggml_backend_buffer_free(buf2);
            if (ctx2) ggml_free(ctx2);
            gguf_free(gctx);
            ggml_free(ctx);
            throw GgufError(std::string("short read for tensor '") + name + "'");
        }

        // F-2: fold EBF layer-scale into producing linear weights/biases.
        {
            const auto fw = fold_weights.find(name);
            if (fw != fold_weights.end()) fold_linear_weight(t, scratch, fw->second);
            const auto fb = fold_biases.find(name);
            if (fb != fold_biases.end()) fold_linear_bias(t, scratch, fb->second);
            const auto lsc = ls_cache.find(name);
            if (lsc != ls_cache.end()) scratch = lsc->second;  // reuse cached payload
        }

        // F-1: split GLU ln1 weights/biases into a/b F32 halves.  The
        // original tensor is still uploaded (bind code references it), but
        // the graph uses the contiguous .a/.b halves (two mul_mats, no cont).
        {
            const auto gw = glu_ab.find(name);
            if (gw != glu_ab.end()) {
                const int64_t D_in = t->ne[0];
                const int64_t L    = t->ne[1] / 2;
                std::vector<float> half(static_cast<std::size_t>(D_in) * L);
                auto fill_half = [&](ggml_tensor * dst, int64_t col0) {
                    if (t->type == GGML_TYPE_F32) {
                        const float * src = reinterpret_cast<const float *>(scratch.data());
                        for (int64_t o = 0; o < L; ++o)
                            std::memcpy(half.data() + o * D_in, src + (col0 + o) * D_in,
                                        static_cast<std::size_t>(D_in) * sizeof(float));
                    } else {  // F16
                        const ggml_fp16_t * src = reinterpret_cast<const ggml_fp16_t *>(scratch.data());
                        for (int64_t o = 0; o < L; ++o)
                            ggml_fp16_to_fp32_row(src + (col0 + o) * D_in,
                                                  half.data() + o * D_in, D_in);
                    }
                    ggml_backend_tensor_set(dst, half.data(), 0, half.size() * sizeof(float));
                };
                fill_half(gw->second.first,  0);
                fill_half(gw->second.second, L);
                out.tensors_.emplace(std::string(name) + ".a", gw->second.first);
                out.tensors_.emplace(std::string(name) + ".b", gw->second.second);
            }
            const auto gb = glu_ab_bias.find(name);
            if (gb != glu_ab_bias.end()) {
                const int64_t L = t->ne[0] / 2;
                std::vector<float> half(static_cast<std::size_t>(L));
                auto fill_half_b = [&](ggml_tensor * dst, int64_t i0) {
                    if (t->type == GGML_TYPE_F32) {
                        const float * src = reinterpret_cast<const float *>(scratch.data());
                        for (int64_t i = 0; i < L; ++i) half[static_cast<std::size_t>(i)] = src[i0 + i];
                    } else {  // F16
                        const ggml_fp16_t * src = reinterpret_cast<const ggml_fp16_t *>(scratch.data());
                        for (int64_t i = 0; i < L; ++i)
                            half[static_cast<std::size_t>(i)] = ggml_fp16_to_fp32(src[i0 + i]);
                    }
                    ggml_backend_tensor_set(dst, half.data(), 0, half.size() * sizeof(float));
                };
                fill_half_b(gb->second.first,  0);
                fill_half_b(gb->second.second, L);
                out.tensors_.emplace(std::string(name) + ".a", gb->second.first);
                out.tensors_.emplace(std::string(name) + ".b", gb->second.second);
            }
        }

        // F16 depthwise-conv weights: store the persistent F32 copy instead.
        const auto dup = dw_f32.find(name);
        if (dup != dw_f32.end()) {
            ggml_tensor * w32 = dup->second;
            const int64_t n = ggml_nelements(t);
            std::vector<float> f32(static_cast<std::size_t>(n));
            ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(scratch.data()),
                                  f32.data(), n);
            ggml_backend_tensor_set(w32, f32.data(), 0, f32.size() * sizeof(float));
            out.tensors_.emplace(name, w32);
        } else {
            ggml_backend_tensor_set(t, scratch.data(), 0, bytes);
            out.tensors_.emplace(name, t);
        }
    }

    std::fclose(f);
    gguf_free(gctx);

    return out;
}

LoadedWeights::~LoadedWeights() {
    if (buffer_)  ggml_backend_buffer_free(buffer_);
    if (ctx_)     ggml_free(ctx_);
    if (buffer2_) ggml_backend_buffer_free(buffer2_);
    if (ctx2_)    ggml_free(ctx2_);
}

LoadedWeights::LoadedWeights(LoadedWeights && other) noexcept
    : ctx_(other.ctx_), buffer_(other.buffer_),
      ctx2_(other.ctx2_), buffer2_(other.buffer2_),
      tensors_(std::move(other.tensors_)) {
    other.ctx_ = nullptr;
    other.buffer_ = nullptr;
    other.ctx2_ = nullptr;
    other.buffer2_ = nullptr;
}

LoadedWeights & LoadedWeights::operator=(LoadedWeights && other) noexcept {
    if (this != &other) {
        if (buffer_)  ggml_backend_buffer_free(buffer_);
        if (ctx_)     ggml_free(ctx_);
        if (buffer2_) ggml_backend_buffer_free(buffer2_);
        if (ctx2_)    ggml_free(ctx2_);
        ctx_     = other.ctx_;
        buffer_  = other.buffer_;
        ctx2_    = other.ctx2_;
        buffer2_ = other.buffer2_;
        tensors_ = std::move(other.tensors_);
        other.ctx_ = nullptr;
        other.buffer_ = nullptr;
        other.ctx2_ = nullptr;
        other.buffer2_ = nullptr;
    }
    return *this;
}

ggml_tensor * LoadedWeights::get(const std::string & name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw GgufError("required tensor missing: " + name);
    }
    return it->second;
}

ggml_tensor * LoadedWeights::try_get(const std::string & name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : it->second;
}

}  // namespace game_ggml::internal
