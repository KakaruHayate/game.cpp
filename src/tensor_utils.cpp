#include "tensor_utils.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>
#include <gguf.h>

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
    //          (see is_dwconv_weight above).  The gguf-init context is sized
    //          exactly for its tensor table, so the copies get their own
    //          small context + backend buffer.
    std::map<std::string, ggml_tensor *> dw_f32;   // original name -> F32 copy
    ggml_context * ctx2 = nullptr;
    ggml_backend_buffer_t buf2 = nullptr;
    {
        ggml_init_params ip{};
        ip.mem_size   = 64 * 1024;
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
        ctx2 = ggml_init(ip);
        if (!ctx2) throw GgufError("failed to create dwconv F32 copy context");

        const int64_t n_tensors = gguf_get_n_tensors(gctx);
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gctx, i);
            ggml_tensor * t = ggml_get_tensor(ctx, name);
            if (!t || t->type != GGML_TYPE_F16 || !is_dwconv_weight(name)) continue;
            ggml_tensor * w32 = ggml_new_tensor_3d(ctx2, GGML_TYPE_F32,
                t->ne[0], t->ne[1], t->ne[2]);
            dw_f32.emplace(name, w32);
        }
        if (!dw_f32.empty()) {
            buf2 = ggml_backend_alloc_ctx_tensors(ctx2, backend);
            if (!buf2) {
                ggml_free(ctx2);
                gguf_free(gctx);
                ggml_free(ctx);
                throw GgufError("failed to allocate dwconv F32 copy buffer");
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
