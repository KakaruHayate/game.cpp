#pragma once

// Helpers for loading GGUF tensors into a ggml context + backend buffer.
// Internal use only.

#include "gguf_io.h"

#include <memory>
#include <string>
#include <unordered_map>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
struct ggml_backend_buffer;
typedef struct ggml_backend *        ggml_backend_t;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;

namespace game_ggml::internal {

// Open a GGUF file and allocate all its tensors into a freshly-created
// `ggml_context` + backend buffer.  The returned bundle owns all of it.
//
//   Resulting `context()` holds an empty ggml_context whose only inhabitants
//   are the weight tensors.  Look them up via `get(name)`.
//
//   The buffer must stay alive as long as any graph references these tensors.
class LoadedWeights {
public:
    // Construct by loading all tensors from `gguf` into `backend`.
    static LoadedWeights load_all(const GgufFile & gguf, ggml_backend_t backend);

    ~LoadedWeights();
    LoadedWeights(LoadedWeights &&) noexcept;
    LoadedWeights & operator=(LoadedWeights &&) noexcept;
    LoadedWeights(const LoadedWeights &) = delete;
    LoadedWeights & operator=(const LoadedWeights &) = delete;

    // Retrieve a tensor by name.  Throws GgufError if missing.
    ggml_tensor * get(const std::string & name) const;

    // Retrieve or return null.
    ggml_tensor * try_get(const std::string & name) const;

    // Number of tensors loaded.
    size_t size() const { return tensors_.size(); }

    // Underlying ggml context (for adding graph nodes that read these tensors).
    ggml_context * context() const { return ctx_; }

private:
    LoadedWeights() = default;

    ggml_context *         ctx_     = nullptr;
    ggml_backend_buffer_t  buffer_  = nullptr;
    // Second context/buffer pair holding the load-time F32 copies of the F16
    // depthwise-conv weights (the gguf-init context is sized exactly for its
    // tensor table, so the copies cannot live there).
    ggml_context *         ctx2_    = nullptr;
    ggml_backend_buffer_t  buffer2_ = nullptr;
    std::unordered_map<std::string, ggml_tensor *> tensors_;
};

}  // namespace game_ggml::internal
