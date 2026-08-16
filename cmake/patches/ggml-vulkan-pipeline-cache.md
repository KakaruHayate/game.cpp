# ggml-vulkan-pipeline-cache（game.cpp 自有 patch）

## Purpose

Vulkan 冷启动 Currently every process recompiles the whole compute-pipeline
set from SPIR-V on first inference（本地实测首次冷启：Vulkan+Q8 达 42s、
F32 约 7.5s；NVIDIA 驱动级 shader 缓存能缓解，但 AMD/Intel/其它驱动不能
保证，且无法随包分发）。

This patch persists `VkPipelineCache` to disk — the Vulkan analog of the
Metal binary-archive PSO cache (`cmake/patches/ggml-metal-binary-archive.patch`).
After the first run, subsequent launches (or any process on a compatible
driver) load precompiled PSO bytes instead of recompiling every shader.

## Change

`src/ggml-vulkan/ggml-vulkan.cpp`:

- `vk_device_struct` gains `pipeline_cache` / `pipeline_cache_init` /
  `pipeline_cache_path` / `pipeline_cache_dirty`.
- At device init (`ggml_vk_init_pipeline_cache`) load the on-disk cache bytes
  into `VkPipelineCacheCreateInfo::pInitialData` (stale/invalid bytes are safe
  — the driver returns `VK_INCOMPLETE` and builds a fresh cache), and create
  the handle.
- `ggml_vk_create_pipeline_func` passes the cache handle to
  `createComputePipeline` and sets `pipeline_cache_dirty = true`.
- On graph-compute completion (and at device teardown as a fallback),
  `ggml_vk_save_pipeline_cache_device` flushes `vkGetPipelineCacheData` to
  disk, throttled by the dirty flag.

## Env / control

- `GGML_VK_PIPELINE_CACHE_PATH` — cache file path.
- default: `%LOCALAPPDATA%\\game_ggml_vk_pipeline.cache` (Windows) or
  `$HOME/.cache/game_ggml_vk_pipeline.cache` (POSIX).
- `GGML_VK_DISABLE_PIPELINE_CACHE` — opt out.
- `GGML_VK_PIPELINE_CACHE_DEBUG` — print load/save diagnostics.

## Baseline & re-apply

- Applies to **ggml v0.19.0** (`ggml-vulkan.cpp`). Pinned by game.cpp
  FetchContent; re-apply per ggml upgrade via `cmake/Dependencies.cmake`
  `game_ggml_apply_patch` (idempotent: skips if already applied).

## Verified (local, RTX 2070, Vulkan)

- Run1 (no cache): builds from scratch, saves ~1.14 MB.
- Run2 (with cache): logs "loaded 1,140,086 bytes", same note output (33/33),
  warm total 0.42s.

## Notes

- The dispatch-table C functions (`vkCreatePipelineCache` /
  `vkGetPipelineCacheData` / `vkDestroyPipelineCache`) are used because this
  SDK's generated `vulkan.hpp` does not expose the corresponding C++ class
  methods.
- Devices are cached for the process lifetime in ggml-vulkan, so the save is
  driven by graph-compute completion rather than the device destructor.
