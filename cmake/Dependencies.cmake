# ----------------------------------------------------------------------------
# External dependencies — all via FetchContent, nothing vendored.
# ----------------------------------------------------------------------------
include(FetchContent)

# Don't let child projects surprise us with their own versions.
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "" FORCE)

# Propagate backend toggles as ggml's own option names *before* add_subdirectory.
set(GGML_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(GGML_METAL           ${GAME_GGML_METAL}  CACHE BOOL "ggml: enable Metal"  FORCE)
set(GGML_CUDA            ${GAME_GGML_CUDA}   CACHE BOOL "ggml: enable CUDA"   FORCE)
set(GGML_VULKAN          ${GAME_GGML_VULKAN} CACHE BOOL "ggml: enable Vulkan" FORCE)

# Apple cold-start optimisation: pre-compile a default.metallib instead of
# embedding source; paired with the binary-archive patch below this cuts ~7 s
# off first-run launch on macOS.
if(APPLE AND GAME_GGML_METAL)
    set(GGML_METAL_EMBED_LIBRARY OFF CACHE BOOL "ggml: embed Metal library" FORCE)
endif()

# ---------------------------------------------------------------------------
# Helper: apply a git patch once; idempotent across re-configures.
# ---------------------------------------------------------------------------
function(game_ggml_apply_patch source_dir patch_file patch_name)
    find_package(Git QUIET)
    if(NOT Git_FOUND)
        message(FATAL_ERROR "Git is required to apply ${patch_name}")
    endif()

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${source_dir}" apply --check "${patch_file}"
        RESULT_VARIABLE _check
        OUTPUT_QUIET ERROR_QUIET
    )
    if(_check EQUAL 0)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${source_dir}" apply "${patch_file}"
            RESULT_VARIABLE _apply
            ERROR_VARIABLE _err
        )
        if(NOT _apply EQUAL 0)
            message(FATAL_ERROR "Failed to apply ${patch_name}: ${_err}")
        endif()
        message(STATUS "Applied ${patch_name}")
        return()
    endif()

    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${source_dir}" apply --reverse --check "${patch_file}"
        RESULT_VARIABLE _rcheck
        OUTPUT_QUIET ERROR_QUIET
    )
    if(_rcheck EQUAL 0)
        message(STATUS "${patch_name} already applied")
    else()
        message(FATAL_ERROR "Could not apply ${patch_name}; upstream source layout changed")
    endif()
endfunction()

# ---------------------------------------------------------------------------
# SPIRV-Headers shim (Windows Vulkan only)
#
# ggml v0.20.x's Vulkan backend hard-requires find_package(SPIRV-Headers CONFIG).
# Windows Vulkan SDKs older than ~1.4.35x ship the headers but not that CMake
# config file, so the windows-x64-vulkan CI job fails at configure time.
# Generate a minimal config pointing at the SDK headers when the SDK doesn't
# already provide one; headers live under <SDK>/Include/spirv-headers.
# ---------------------------------------------------------------------------
if(GAME_GGML_VULKAN AND WIN32 AND NOT EXISTS "$ENV{VULKAN_SDK}/Lib/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake")
    string(REPLACE "\\" "/" _spirv_inc "$ENV{VULKAN_SDK}/Include")
    set(_spirv_cfg_dir "${CMAKE_BINARY_DIR}/SPIRV-Headers")
    file(MAKE_DIRECTORY "${_spirv_cfg_dir}")
    file(WRITE "${_spirv_cfg_dir}/SPIRV-HeadersConfig.cmake"
        "set(SPIRV-Headers_FOUND TRUE)\n"
        "if(NOT TARGET SPIRV-Headers::SPIRV-Headers)\n"
        "    add_library(SPIRV-Headers::SPIRV-Headers INTERFACE IMPORTED)\n"
        "    set_target_properties(SPIRV-Headers::SPIRV-Headers PROPERTIES INTERFACE_INCLUDE_DIRECTORIES \"${_spirv_inc}\")\n"
        "endif()\n")
    list(APPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}")
endif()

# ---------------------------------------------------------------------------
# ggml (MIT) — tensor engine.
# Fetched + Metal binary-archive patch applied on Apple for fast cold start.
# ---------------------------------------------------------------------------
FetchContent_Declare(
    ggml
    # URL archive instead of git: with FETCHCONTENT_UPDATES_DISCONNECTED=ON
    # the populate gitupdate step must resolve the pinned ref locally, and a
    # fresh clone cannot — a shallow clone only carries the default-branch
    # HEAD, and the v0.20.2 tag/commit is not on it — so populate aborts with
    # "requested git ref ... not present locally".  A URL archive has no git
    # ref semantics: populate is a plain download+extract (network needed on
    # first populate / cache miss; the CI _deps cache then makes later runs
    # offline).  Patches are still applied with `git apply`, which needs no
    # .git directory.  (No DOWNLOAD_EXTRACT_TIMESTAMP: requires CMake 3.24+,
    # project minimum is 3.18.)
    URL      https://github.com/ggerganov/ggml/archive/refs/tags/v0.20.2.tar.gz
    URL_HASH SHA256=55dfd1ea4e6b6b3e25d9411f9525eb4df1c796c03a244e2321388b30f189cd3d
)

FetchContent_GetProperties(ggml)
if(NOT ggml_POPULATED)
    FetchContent_Populate(ggml)

    if(APPLE AND GAME_GGML_METAL)
        game_ggml_apply_patch(
            "${ggml_SOURCE_DIR}"
            "${CMAKE_CURRENT_LIST_DIR}/patches/ggml-metal-binary-archive.patch"
            "ggml Metal MTLBinaryArchive PSO cache"
        )
    endif()

    # Vulkan cold-start fix: persist VkPipelineCache to disk.  Apply on every
    # build (the patch only touches ggml-vulkan.cpp, compiled only when the
    # Vulkan backend is enabled) so CPU-only and GPU builds share one source.
    game_ggml_apply_patch(
        "${ggml_SOURCE_DIR}"
        "${CMAKE_CURRENT_LIST_DIR}/patches/ggml-vulkan-pipeline-cache.patch"
        "ggml Vulkan disk-backed VkPipelineCache"
    )

    add_subdirectory("${ggml_SOURCE_DIR}" "${ggml_BINARY_DIR}")
endif()

# ---------------------------------------------------------------------------
# pocketfft (BSD-3-Clause) — header-only STFT / r2c FFT used by mel.cpp.
# Pinned to a known-good commit on the `cpp` branch.  The repo carries
# only a single usable header so we expose it via an INTERFACE target
# rather than pull in any build system.
# ---------------------------------------------------------------------------
FetchContent_Declare(
    pocketfft
    GIT_REPOSITORY https://github.com/mreineck/pocketfft.git
    GIT_TAG        32424d2067c2e8043dc646a4e49754b2b40cc549   # cpp @ 2025-10
)
FetchContent_MakeAvailable(pocketfft)
# Idempotent: when aggregated by a parent (e.g. diffsinger.cpp), the parent
# may already define the same helper target.
if(NOT TARGET pocketfft)
    add_library(pocketfft INTERFACE)
    target_include_directories(pocketfft SYSTEM INTERFACE "${pocketfft_SOURCE_DIR}")
endif()

# ---------------------------------------------------------------------------
# dr_libs (Public Domain / MIT-0 dual) — single-header WAV reader used
# by the CLI layer.  The repo bundles many libs; we only include dr_wav.h.
# ---------------------------------------------------------------------------
FetchContent_Declare(
    dr_libs
    GIT_REPOSITORY https://github.com/mackron/dr_libs.git
    GIT_TAG        243e26ffa08a24dc8ae2e7a8c57123d9e504690c   # master @ 2025-10
)
FetchContent_MakeAvailable(dr_libs)
if(NOT TARGET dr_wav)
    add_library(dr_wav INTERFACE)
    target_include_directories(dr_wav SYSTEM INTERFACE "${dr_libs_SOURCE_DIR}")
endif()

# Emit a NOTICE line so users know exactly which third-party libs got fetched.
message(STATUS "Third-party fetched:")
message(STATUS "  ggml       ${ggml_SOURCE_DIR}")
message(STATUS "  pocketfft  ${pocketfft_SOURCE_DIR}")
message(STATUS "  dr_libs    ${dr_libs_SOURCE_DIR}")