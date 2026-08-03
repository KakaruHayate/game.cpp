cmake_minimum_required(VERSION 3.18)
project(game_ggml
    VERSION 0.1.0
    DESCRIPTION "GAME (Generative Adaptive MIDI Extractor) native ggml inference backend"
    LANGUAGES C CXX)

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(GAME_GGML_CUDA        "Enable ggml CUDA backend"     OFF)
option(GAME_GGML_VULKAN      "Enable ggml Vulkan backend"   OFF)
option(GAME_GGML_BUILD_CLI   "Build game_ggml_cli executable"   ON)
option(GAME_GGML_BUILD_TESTS "Build unit tests (requires GoogleTest via FetchContent)" OFF)

# On Apple, default to Metal backend on. Use GAME_GGML_METAL to override.
if(APPLE)
    option(GAME_GGML_METAL "Enable ggml Metal backend" ON)
else()
    option(GAME_GGML_METAL "Enable ggml Metal backend" OFF)
endif()

# -----------------------------------------------------------------------------
# Global settings
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
if(MSVC)
    foreach(_cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
        string(TOUPPER "${_cfg}" _CFG)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_CFG} "${CMAKE_BINARY_DIR}/bin")
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${_CFG} "${CMAKE_BINARY_DIR}/bin")
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${_CFG} "${CMAKE_BINARY_DIR}/lib")
    endforeach()
    # NVCC treats a bare /utf-8 as an input path. Keep this MSVC option on
    # native C/C++ sources; CUDA sources already contain ASCII identifiers.
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
endif()

if(MSVC)
    set(GAME_GGML_WARN_FLAGS /W3 /wd4244 /wd4267 /wd4996)
else()
    set(GAME_GGML_WARN_FLAGS -Wall -Wextra -Wno-unused-parameter)
endif()

# -----------------------------------------------------------------------------
# Dependencies (ggml via FetchContent, plus optional Metal patch)
# -----------------------------------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Dependencies.cmake)

# -----------------------------------------------------------------------------
# Core library: game_ggml
# -----------------------------------------------------------------------------
# NOTE: Only backend.cpp is present in Task 1; further source files are added
# in subsequent tasks.  Header-only parts of the public API live in include/.
set(GAME_GGML_CORE_SOURCES
    src/backend.cpp
    src/gguf_io.cpp
    src/tensor_utils.cpp
    src/ops_basic.cpp
    src/ops_ffn.cpp
    src/ops_rope.cpp
    src/ops_attn.cpp
    src/ops_joint_attn.cpp
    src/model_encoder.cpp
    src/model_segmenter.cpp
    src/model_estimator.cpp
    src/model.cpp
    src/mel.cpp
    src/decode.cpp
    src/rng.cpp
    src/d3pm.cpp
)

add_library(game_ggml STATIC ${GAME_GGML_CORE_SOURCES})
add_library(game_ggml::game_ggml ALIAS game_ggml)

target_include_directories(game_ggml
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ggml targets exported by the FetchContent'd ggml source tree.
# v0.11.0 provides `ggml` as the umbrella interface target.
target_link_libraries(game_ggml
    PUBLIC  ggml
    PRIVATE pocketfft
)

target_compile_options(game_ggml PRIVATE ${GAME_GGML_WARN_FLAGS})
target_compile_definitions(game_ggml PUBLIC
    GAME_GGML_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    GAME_GGML_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    GAME_GGML_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

# Propagate optional-backend feature macros so consumers can branch on them.
if(GAME_GGML_METAL)
    target_compile_definitions(game_ggml PUBLIC GAME_GGML_HAS_METAL=1)
endif()
if(GAME_GGML_CUDA)
    target_compile_definitions(game_ggml PUBLIC GAME_GGML_HAS_CUDA=1)
endif()
if(GAME_GGML_VULKAN)
    target_compile_definitions(game_ggml PUBLIC GAME_GGML_HAS_VULKAN=1)
endif()

# -----------------------------------------------------------------------------
# CLI executable
# -----------------------------------------------------------------------------
if(GAME_GGML_BUILD_CLI)
    add_executable(game_ggml_cli
        src/cli/main.cpp
        src/cli/wav_io.cpp
        src/cli/slicer.cpp
        src/cli/midi_writer.cpp
        src/cli/text_writer.cpp
    )
    target_link_libraries(game_ggml_cli PRIVATE
        game_ggml::game_ggml
        dr_wav
    )
    target_compile_options(game_ggml_cli PRIVATE ${GAME_GGML_WARN_FLAGS})
endif()

# -----------------------------------------------------------------------------
# Tests (optional)
# -----------------------------------------------------------------------------
if(GAME_GGML_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------------
# Install (minimal; mainly so consumers can embed via add_subdirectory)
# -----------------------------------------------------------------------------
install(DIRECTORY include/game_ggml
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")

install(TARGETS game_ggml
        EXPORT game_ggml-targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

message(STATUS "")
message(STATUS "game_ggml ${PROJECT_VERSION} configuration:")
message(STATUS "  Metal .............. ${GAME_GGML_METAL}")
message(STATUS "  CUDA ............... ${GAME_GGML_CUDA}")
message(STATUS "  Vulkan ............. ${GAME_GGML_VULKAN}")
message(STATUS "  Build CLI .......... ${GAME_GGML_BUILD_CLI}")
message(STATUS "  Build Tests ........ ${GAME_GGML_BUILD_TESTS}")
message(STATUS "")
