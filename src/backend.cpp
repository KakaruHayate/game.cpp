#include "backend.h"

#include "game_ggml/errors.h"
#include "game_ggml/version.h"

#include <ggml-backend.h>
#include <ggml.h>

#if defined(GAME_GGML_HAS_METAL)
    #include <ggml-metal.h>
#endif
#if defined(GAME_GGML_HAS_CUDA)
    #include <ggml-cuda.h>
#endif
#if defined(GAME_GGML_HAS_VULKAN)
    #include <ggml-vulkan.h>
#endif
#include <ggml-cpu.h>

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <thread>

// -----------------------------------------------------------------------------
// Public version helpers (declared in version.h)
// -----------------------------------------------------------------------------
namespace game_ggml {

namespace {
    // Synthesized once at static-init time so callers get a stable pointer.
    const std::string & g_version() {
        static const std::string v = [] {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%d.%d.%d",
                GAME_GGML_VERSION_MAJOR, GAME_GGML_VERSION_MINOR, GAME_GGML_VERSION_PATCH);
            return std::string(buf);
        }();
        return v;
    }
}

const char * version_string() noexcept {
    return g_version().c_str();
}

const char * ggml_version_string() noexcept {
    // ggml v0.11.0 does not export a runtime-queryable version.  We log the
    // compile-time pin used by this project so the CLI can say "ggml 0.11.0".
    return "0.11.0";
}

// -----------------------------------------------------------------------------
// Backend enumeration (declared in game_ggml.h)
// -----------------------------------------------------------------------------
namespace {
    std::array<const char *, 4> g_backend_names = {nullptr, nullptr, nullptr, nullptr};
    int g_backend_count = 0;

    void populate_backend_names() {
        if (g_backend_count != 0) return;
        // Preference order mirrors init_best_backend().
#if defined(GAME_GGML_HAS_METAL)
        g_backend_names[g_backend_count++] = "metal";
#endif
#if defined(GAME_GGML_HAS_CUDA)
        g_backend_names[g_backend_count++] = "cuda";
#endif
#if defined(GAME_GGML_HAS_VULKAN)
        g_backend_names[g_backend_count++] = "vulkan";
#endif
        g_backend_names[g_backend_count++] = "cpu";
    }
}

const char * const * available_backends() noexcept {
    populate_backend_names();
    return g_backend_names.data();
}

int available_backends_count() noexcept {
    populate_backend_names();
    return g_backend_count;
}

}  // namespace game_ggml

// -----------------------------------------------------------------------------
// Internal: backend init helpers
// -----------------------------------------------------------------------------
namespace game_ggml::internal {

ggml_backend_t init_backend(Backend which) {
    switch (which) {
        case Backend::Metal:
#if defined(GAME_GGML_HAS_METAL)
            return ggml_backend_metal_init();
#else
            return nullptr;
#endif
        case Backend::CUDA:
#if defined(GAME_GGML_HAS_CUDA)
            return ggml_backend_cuda_init(0);
#else
            return nullptr;
#endif
        case Backend::Vulkan:
#if defined(GAME_GGML_HAS_VULKAN)
            return ggml_backend_vk_init(0);
#else
            return nullptr;
#endif
        case Backend::CPU: {
            ggml_backend_t b = ggml_backend_cpu_init();
            if (b) {
                // Default: use all hardware threads so CPU inference is not
                // accidentally single-threaded.  Override with
                // GAME_GGML_THREADS=<n> (1..INT_MAX, full string must parse).
                unsigned n = std::thread::hardware_concurrency();
                if (n == 0) n = 1;
                if (const char * env = std::getenv("GAME_GGML_THREADS"); env && *env) {
                    char * end = nullptr;
                    errno = 0;
                    const long v = std::strtol(env, &end, 10);
                    if (v >= 1 && v <= std::numeric_limits<int>::max() &&
                        end && *end == '\0') {
                        n = static_cast<unsigned>(v);
                    } else {
                        std::fprintf(stderr,
                            "warning: ignoring invalid GAME_GGML_THREADS '%s' "
                            "(expected 1..%d)\n", env,
                            std::numeric_limits<int>::max());
                    }
                }
                ggml_backend_cpu_set_n_threads(b, static_cast<int>(n));
            }
            return b;
        }
    }
    return nullptr;
}

ggml_backend_t init_best_backend() {
#if defined(GAME_GGML_HAS_METAL)
    if (auto * b = init_backend(Backend::Metal)) return b;
#endif
#if defined(GAME_GGML_HAS_CUDA)
    if (auto * b = init_backend(Backend::CUDA)) return b;
#endif
#if defined(GAME_GGML_HAS_VULKAN)
    if (auto * b = init_backend(Backend::Vulkan)) return b;
#endif
    if (auto * b = init_backend(Backend::CPU)) return b;
    throw BackendError("failed to initialize any ggml backend (including CPU)");
}

void free_backend(ggml_backend_t backend) {
    if (backend == nullptr) return;
    ggml_backend_free(backend);
}

const char * backend_name(ggml_backend_t backend) {
    if (!backend) return "<null>";
    return ggml_backend_name(backend);
}

}  // namespace game_ggml::internal
