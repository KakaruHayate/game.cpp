# ggml-metal-binary-archive（本仓库所有权）

> 本 patch 原属 KakaruHayate/ggml-patch 的「生态」扩张内容，现随生态解散
> 收编回 game.cpp（即本仓库），由本仓库全权维护、随 ggml 升级重打。

## Purpose

Cache compiled Metal pipeline state objects (PSOs) to disk via
`MTLBinaryArchive` so subsequent launches skip the expensive
`newComputePipelineState` compilation.  On a fresh launch metal rebuilds
thousands of kernels, adding seconds-per-model to load time; this reduces it
to a single archive load.

## Change

`src/ggml-metal/ggml-metal-device.m` — add `MTLBinaryArchive` PSO cache:

- `ggml_metal_archive_url()` reads `GGML_METAL_ARCHIVE_PATH` (falls back to
  ~/.cache/ggml-metal archive path).
- At pipeline creation, try `newBinaryArchiveWithDescriptor` / completion
  handler; on first run nothing is cached so it compiles normally.
- After compiling, capture the pipeline state into the archive via
  `storeRenderPipelineState` and `commit` it to the archive file.

Controlled by the same `GGML_METAL_ARCHIVE_PATH` env var; disable with
`GGML_METAL_DISABLE_ARCHIVE`.

## Baseline

Applied against **ggml v0.19.0** (`ggml-metal-device.m`).  Verify with the
same command used in `cmake/Dependencies.cmake`:

```
git -C <ggml> apply --check cmake/patches/ggml-metal-binary-archive.patch
```

Test:

```
GGML_METAL_ARCHIVE_PATH=/tmp/game-mtl-archive game_ggml_cli extract ...
# second run should show near-zero PSO compile; compare GAME_GGML_PROFILE
```

## Origin

Internal Shenzhen branch derived from Metal submission overhead experiments.
Kept out-of-line from ggml main; re-apply per ggml upgrade (see
`cmake/Dependencies.cmake` `game_ggml_apply_patch`).

## Ownership (2026-08)

`ggml-patch` 已弃用；本 patch 收编于 game.cpp `cmake/patches/`（本目录）。
改动本 patch 只改本目录，不改 ggml-patch 历史形态。
