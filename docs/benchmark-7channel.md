# 7-channel inference benchmark annex

Scope: one 60 s.44100 Hz mono WAV (`w44k_60.wav`), **no slicing**, `--nsteps 8`,
compared at frame level against the PyTorch GPU fp32 no-cache baseline.
GGML rows use the `ml_qkv` QKV-fused converter output (schema v2): `game_medium.gguf`
(F32) and `game_medium_q8.gguf` (Q8). Torch runs use
`GAME-main/experiments/GAME-1.0-medium/model.pt`; ONNX rows use
`GAME-1.0.3-medium-onnx/`. Local dual-GPU rig: RTX 2070 (Vulkan) + a secondary
NVIDIA GPU (CUDA). Full methodology in `dbcache_ablation/` scripts.

> **2026-08 refresh:** the GGML rows below were re-measured on the current
> stack (ggml **v0.19.0**, CPU `GGML_NATIVE=ON`, `GGML_LLAMAFILE=ON` defaults,
> CUDA 12.6). Torch/ONNX rows are from the original run and were **not**
> re-measured (kept for reference only). Runner: `bench/run_ggml_bench.py`.

## Metric definition (frame level, not note count)

Rasterize each note list onto a 100 Hz frame grid (frame = 0.01 s) with presence
+ MIDI score, then compare to the reference's grid using the formulas from GAME
`modules/metrics/pitch.py`:

- presence `Precision` / `Recall` / `F1` — per-frame note-presence classification
- `RawPitchRMSE(midi)` — over reference-voiced frames only
- `RawPitchAccuracy` (`RPA@0.5`) — |Δ| ≤ 0.5 semitone over voiced frames
- `OverallAcc` — both presence and pitch correct / both absent.

Note count is reported only as a hint; it is not the "deterministic" evidence.

## Results

| channel | weights | cache | wall(s) | VRAMΔ(MiB) | hi/ms | notes |
|---|---:|---:|---:|---:|---:|---:|
| torch_cuda (baseline) | fp32 | off | 34.2 | +500 | – | 147 |
| torch_cuda | fp32 | on | 30.4 | +531 | – | 144 |
| torch_cpu | fp32 | off | 45.9 | – | – | 149 |
| onnx_cpu | fp32 | off | 57.7 | – | – | 154 |
| onnx_dml | fp32 | off | 100.9 | +4415 | – | 155 |
| ggml_cpu | F32 | off | 39.9 | – | 0/0 | 157 |
| ggml_cpu | F32 | on | **20.8** | – | 5/3 | 157 |
| ggml_cpu | Q8 | off | 47.0 | – | 0/0 | 154 |
| ggml_cpu | Q8 | on | 24.9 | – | 5/3 | 158 |
| ggml_vk (warm) | F32 | off | 4.04 | +673 | 0/0 | 152 |
| ggml_vk (warm) | F32 | on | 3.51 | +819 | 4/4 | 155 |
| ggml_vk (warm) | Q8 | off | 3.60 | +534 | 0/0 | 156 |
| ggml_vk (warm) | Q8 | on | 2.99 | +667 | 4/4 | 158 |
| ggml_cuda | F32 | off | 2.47 | +732 | 0/0 | 154 |
| ggml_cuda | F32 | on | 1.99 | +682 | 5/3 | 156 |
| ggml_cuda | Q8 | off | 2.24 | +457 | 0/0 | 152 |
| ggml_cuda | Q8 | on | 1.94 | +553 | 5/3 | 158 |

Frame-level quality columns (presF1 / RMSE / RPA / OA) are unchanged from the
original report for the torch/onnx rows; the refreshed GGML rows were checked
for **internal consistency** (same-input note counts across backend within
152–158, stable across cache on/off) rather than re-scored frame-level vs the
torch baseline, since the torch reference itself was not re-run.

`hi/ms` = DBCache hit/miss counters (`--cache-threshold 0.25`, `--cache-fn-blocks 1`,
`--cache-warmup 1`).

## Conclusions

1. **Speed (current stack)**: ggml_cuda ~1.9–2.5 s ≈ ggml_vk ~3–4 s ≪
   ggml_cpu(cache) ~21–25 s < torch 30–46 s < onnx 58–101 s.  CUDA is now the
   fastest backend on this rig (was ~6–8 s under the old stack; ~3× faster).
2. **Cache**: DBCache hits in the 4–5 range on all EPS. CPU wall **−48 %**
   (F32 39.9→20.8 s, Q8 47.0→24.9 s) with stable note counts (cache on/off
   drift ≤ 4 notes). GPU cache also cuts wall (CUDA F32 2.47→1.99 s,
   Vulkan F32 4.04→3.51 s) — smaller absolute gain but hit/miss consistent.
3. **Weights**: F32 ≅ Q8 on all backends (near-lossless); n1 reference produces
   162 notes on all three engines, n8 lands in 152–158 for every engine —
   no backend-specific note divergence.
4. **VRAM**: ggml_vk +534…+819, ggml_cuda +457…+732, torch +500…+531,
   **onnx DML +4415 MiB** (DML staging worst).
5. **Vulkan cold-start**: the first launch compiles shaders (30 s here); the
   persistent `VkPipelineCache` (PR9, `GGML_VK_PIPELINE_CACHE_PATH`) absorbs
   it, warm runs are ~3–4 s.

Data/scripts: `bench/run_ggml_bench.py` (refresh runner) and the original
`dbcache_ablation/{bench_engine.py,ggml_n8.py,onnx_run.py,normalize.py,game_metric.py,report.py}`.

## Recommended EP configuration by platform

| platform      | GPU           | weights | EP offered by oudep pkg | rationale (measured) |
|---------------|---------------|---------|--------------------------|----------------------|
| Windows       | NVIDIA        | F32     | CUDA (fallback Vulkan)   | cuda ~2–2.5 s, +0.46–0.73 GiB |
| Windows       | Intel/AMD/etc | F32     | Vulkan                   | ~3–4 s, +0.53–0.82 GiB |
| Windows       | integrated    | F32/Q8  | CPU (+ DBCache on)       | 39.9→20.8 s (−48 %), quality-neutral |
| Linux         | NVIDIA        | F32     | CUDA (fallback Vulkan)   | same as Windows CUDA |
| Linux         | Nouveau/AMD   | F32     | Vulkan                   | same as Windows Vulkan |
| macOS         | Apple Silicon | F32     | Metal (the only EP)      | not measured here; CI-build path |
| macOS         | Intel         | F32     | Metal (cross-compiled)   | not measured here; CI-build path |

Rules of thumb:
- GPU present → **F32** weights; CPU-only → **CPU EP with DBCache** (default on).
- On GPU, CUDA = fastest wall (~2 s), Vulkan = portable alternative (~3–4 s).
- Q8 (~3.4× smaller weights, near-lossless) but may flip a boundary note on
  Vulkan → choose the `-full` package when bit-consistent output matters;
  otherwise either package is valid.
- No user escalation needed: the CLI picks the backend from the GGUF/EP and
  resolves cache by EP.
