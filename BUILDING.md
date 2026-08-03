# Building from source

This document covers how to build `game_ggml_cli` from source on Linux, macOS and
Windows.  The project is self-contained — all dependencies are pulled via CMake
FetchContent at configure time.

## Prerequisites

### Linux

```bash
# Build tools
sudo apt install build-essential cmake git

# Optional: Vulkan backend (recommended for NVIDIA/AMD GPUs)
sudo apt install libvulkan-dev vulkan-tools
# glslc comes with the Vulkan SDK.  Download from LunarG:
# https://vulkan.lunarg.com/sdk/home
# Or install via package manager where available:
#   sudo apt install glslc-tools   # not available on all distros

# Optional: CUDA backend (NVIDIA GPUs)
# Install a CUDA Toolkit supported by your compiler and driver. CI uses CUDA 12.6.
# https://developer.nvidia.com/cuda-downloads

# Optional: ccache for faster rebuilds
sudo apt install ccache
```

### macOS

```bash
# Xcode Command Line Tools
xcode-select --install

# Homebrew
brew install cmake ccache

# Metal backend is built-in — no extra SDK needed.
# For Intel Mac, cross-compile from Apple Silicon is supported:
#   cmake -DCMAKE_OSX_ARCHITECTURES=x86_64 ...
```

### Windows

```bash
# Visual Studio 2022+ with "Desktop development with C++" workload
# Or Build Tools for Visual Studio (smaller install):
#   https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio

# CMake
#   https://cmake.org/download/ (or install via Visual Studio Installer)

# Vulkan SDK (optional, for Vulkan backend)
#   https://vulkan.lunarg.com/sdk/home
```

## Quick start

```bash
# Clone
git clone https://github.com/KakaruHayate/game_ggml_cli.git
cd game_ggml_cli

# Configure with CPU backend
cmake -B build -DCMAKE_BUILD_TYPE=Release \
               -DGAME_GGML_BUILD_CLI=ON \
               -DGAME_GGML_BUILD_TESTS=OFF

# Build
cmake --build build -j

# Verify
build/bin/game_ggml_cli --version
```

## Backend selection

Pass one (or more) of these flags to the **Configure** step:

| Flag | Backend | Default |
|------|---------|---------|
| `-DGAME_GGML_METAL=ON`  | Apple Metal (macOS only) | ON on Apple |
| `-DGAME_GGML_VULKAN=ON` | Vulkan (Linux/Windows)   | OFF |
| `-DGAME_GGML_CUDA=ON`   | CUDA (NVIDIA GPU)        | OFF |

If no GPU backend is enabled, the CPU backend is used as fallback.  The best
available backend is selected automatically at runtime.

### Examples

```bash
# Linux + Vulkan
cmake -B build -DCMAKE_BUILD_TYPE=Release \
               -DGAME_GGML_BUILD_CLI=ON \
               -DGAME_GGML_VULKAN=ON

# Linux + CUDA (requires the CUDA Toolkit and nvcc)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
               -DGAME_GGML_BUILD_CLI=ON \
               -DGAME_GGML_CUDA=ON \
               -DGGML_NATIVE=OFF \
               -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89"

# macOS Apple Silicon + Metal
cmake -B build -DCMAKE_BUILD_TYPE=Release \
               -DGAME_GGML_BUILD_CLI=ON \
               -DGAME_GGML_METAL=ON

# macOS Intel Mac (cross-compile from Apple Silicon runner)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
               -DGAME_GGML_BUILD_CLI=ON \
               -DGAME_GGML_METAL=ON \
               -DCMAKE_OSX_ARCHITECTURES=x86_64

# Windows + Vulkan (from Visual Studio Developer Command Prompt / PowerShell)
cmake -B build -DCMAKE_BUILD_TYPE=Release `
               -DGAME_GGML_BUILD_CLI=ON `
               -DGAME_GGML_VULKAN=ON
```

## Converting a PyTorch checkpoint to GGUF

The medium model checkpoint can be downloaded from
[OpenVPI releases](https://github.com/openvpi/GAME/releases/download/v1.0.0/GAME-1.0-medium.zip).

```bash
# Install Python dependencies
pip install torch numpy gguf pyyaml

# Convert
python scripts/convert_pt_to_gguf.py \
    --model-dir GAME-1.0-medium \
    -o game_medium.gguf

# Inspect
build/bin/game_ggml_cli inspect game_medium.gguf
```

Expected output for the medium model:
```
  architecture : game-me
  embedding_dim: 256
  encoder      : 4 layers, 8 heads
  segmenter    : 8 layers, latent@6
  estimator    : 4 layers, joint attn, R=1
  tensors      : 671
```

## Running inference

```bash
# Single file
build/bin/game_ggml_cli extract input.wav \
    -m game_medium.gguf \
    --output-formats mid \
    --output-dir out/ \
    --nsteps 8 \
    --seed 42

# Serve mode (for OpenUtau integration)
build/bin/game_ggml_cli serve game_medium.gguf
# Then write binary request frames to stdin (see src/cli/main.cpp for protocol)
```

## CI CUDA scope

The hosted CI builds and packages the Linux x64 CUDA backend with CUDA Toolkit
12.6. It verifies Toolkit discovery, CUDA compilation, linking, and that the CLI
starts with `--version`. GitHub-hosted runners do not provide an NVIDIA GPU, so
actual CUDA inference must still be smoke-tested on an NVIDIA system. The
packaged CUDA backend uses the CUDA runtime and cuBLAS libraries supplied by the
installed NVIDIA CUDA runtime/toolkit.

## Troubleshooting

### CUDA Toolkit not found

Ensure `nvcc --version` succeeds and that `CUDA_PATH` (or the platform-specific
Toolkit environment) points to the intended installation. Delete `build/` before
reconfiguring after changing CUDA Toolkit versions.

### glslc not found (Linux/Windows Vulkan)

The Vulkan SDK's `glslc` compiler is required by the ggml-vulkan backend.  If
`FindVulkan` reports `glslc` as missing, ensure the SDK is installed and its
`bin/` directory is on `PATH`:

```bash
# Linux — after installing the SDK
export VULKAN_SDK=/path/to/vulkan-sdk
export PATH="$VULKAN_SDK/bin:$PATH"

# Windows — set environment variables or pass via CMake
set VULKAN_SDK=C:\VulkanSDK\1.4.304.1
set PATH=%VULKAN_SDK%\Bin;%PATH%
```

### `unknown target CPU 'apple-m1'` (macOS x86_64 cross-compile)

When cross-compiling for Intel Mac on an Apple Silicon runner, the CPU backend
mistakenly detects the host as `apple-m1`.  Disable native CPU detection:

```bash
cmake -B build -DGGML_NATIVE=OFF ...
```

### `FetchContent` download failures

Dependencies are fetched via Git at configure time.  If you are behind a proxy:

```bash
git config --global http.proxy http://proxy:port
git config --global https.proxy http://proxy:port
```

Then delete `build/` and reconfigure.