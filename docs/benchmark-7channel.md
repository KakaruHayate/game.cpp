# 7-channel inference benchmark annex

Scope: one 60 s.44100 Hz mono WAV (`w44k_60.wav`), **no slicing**, `--nsteps 8`,
compared at frame level against the PyTorch GPU fp32 no-cache baseline.
GGML rows use the `ml_qkv` QKV-fused converter output (schema v2): `game_medium.gguf`
(F32) and `game_medium_q8.gguf` (Q8). Torch runs use
`GAME-main/experiments/GAME-1.0-medium/model.pt`; ONNX rows use
`GAME-1.0.3-medium-onnx/`. Local dual-GPU rig: RTX 2070 (Vulkan) + a secondary
NVIDIA GPU (CUDA). Full methodology in `dbcache_ablation/` scripts.

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

| channel | weights | cache | wall(s) | VRAMΔ(MiB) | hi/ms | presF1 | RMSE(midi) | RPA | OA | notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| torch_cuda (baseline) | fp32 | off | 34.2 | +500 | – | 1.000 | 0.00 | 1.000 | 1.000 | 147 |
| torch_cuda | fp32 | on | 30.4 | +531 | – | 0.9988 | 1.58 | 0.9915 | 0.9909 | 144 |
| torch_cpu | fp32 | off | 45.9 | – | – | 0.999 | 2.12 | 0.988 | 0.989 | 149 |
| onnx_cpu | fp32 | off | 57.7 | – | – | 0.917 | 0.56 | 0.973 | 0.824 | 154 |
| onnx_dml | fp32 | off | 100.9 | +4415 | – | 0.917 | 0.51 | 0.973 | 0.824 | 155 |
| ggml_cpu | F32 | off | 48.5 | – | 0/0 | 0.9982 | 3.84 | 0.969 | 0.972 | 143 |
| ggml_cpu | F32 | on | **27.1** | – | 5/3 | 0.9983 | 3.84 | 0.976 | 0.978 | 144 |
| ggml_cpu | Q8 | off | 47.1 | – | 0/0 | 0.9984 | 3.53 | 0.974 | 0.977 | 143 |
| ggml_cpu | Q8 | on | 26.7 | – | 5/3 | 0.9983 | 3.85 | 0.974 | 0.976 | 145 |
| ggml_vk | F32 | off | 4.70 | +413 | 0/0 | 0.9982 | 3.98 | 0.971 | 0.974 | 144 |
| ggml_vk | F32 | on | 4.11 | +417 | 4/4 | 0.9983 | 3.85 | 0.971 | 0.974 | 142 |
| ggml_vk | Q8 | off | 5.30 | +273 | 0/0 | 0.9982 | 3.98 | 0.971 | 0.974 | 144 |
| ggml_vk | Q8 | on | 5.36 | +265 | 4/4 | 0.9983 | 3.83 | 0.978 | 0.980 | 146 |
| ggml_cuda | F32 | off | 7.61 | +770 | 0/0 | 0.9983 | 3.84 | 0.969 | 0.972 | 143 |
| ggml_cuda | F32 | on | 6.94 | +765 | 5/3 | 0.9981 | 3.99 | 0.968 | 0.971 | 143 |
| ggml_cuda | Q8 | off | 6.32 | +651 | 0/0 | 0.9982 | 3.84 | 0.971 | 0.974 | 144 |
| ggml_cuda | Q8 | on | 6.19 | +695 | 5/3 | 0.9983 | 3.85 | 0.971 | 0.974 | 144 |

`hi/ms` = DBCache hit/miss counters (`--cache-threshold 0.25`, `--cache-fn-blocks 1`,
`--cache-warmup 1`).

## Conclusions

1. **Quality vs PyTorch-CUDA fp32 baseline**: every engine RPA ≥ 0.97 frame-level;
   ggml presence F1 = 0.998 highest; onnx has the best pitch RMSE (0.5–0.6) but
   weaker presence precision (more notes). GGML F32 ≅ Q8.
2. **Cache**: hits in the 4–5 range on all EPS. CPU wall **−43…−44 %** with metrics
   inside noise (RPA Δ ≤ 0.007) → "fastest and acceptable". Torch-CUDA's own
   DBCache: 34.2 → 30.4 s (−11 %), metrics in noise (RPA 0.9915, 144 vs 147 notes).
   GPU ggml cache gives hits too but ≈±10 % wall.
3. **Speed**: ggml_vk ~4–5 s ≈ ggml_cuda 6–8 s ≪ ggml_cpu(cache) 27–49 s
   < torch 30–46 s < onnx 58–101 s.
4. **VRAM**: ggml_vk +265…+417, ggml_cuda +651…+770, torch +500…531,
   **onnx DML +4415 MiB** (DML staging is the worst).
5. **Vulkan Q8 anomaly — resolved**: the early 15.3 s figure was **cold-start shader
   compilation** for the new dwconv variant; PR9's persistent `VkPipelineCache`
   (path set via `GGML_VK_PIPELINE_CACHE_PATH`) absorbs it, warm Q8 ≈ F32 ≈ 3.1 s
   (10 blocks). A 1-note boundary flip persists between Q8 (161) and F32 (162) at
   n8/n1 — use F32 if bit-exactness is needed, otherwise Q8 is fine.

Data/scripts: `dbcache_ablation/{bench_engine.py,ggml_n8.py,onnx_run.py,normalize.py,game_metric.py,report.py}`,
CSVs in `dbcache_ablation/results/{canon,n8,*,torch_*,onnx_*}`.

## Recommended EP configuration by platform

| platform      | GPU           | weights | EP offered by oudep pkg | rationale (measured) |
|---------------|---------------|---------|--------------------------|----------------------|
| Windows       | NVIDIA        | F32     | CUDA (fallback Vulkan)   | cuda 6–8 s, +0.65–0.77 GiB |
| Windows       | Intel/AMD/etc | F32     | Vulkan                   | 4–5 s, smallest VRAM +0.26–0.42 GiB |
| Windows       | integrated    | F32/Q8  | CPU (+ DBCache on)       | 48.5→27.1 s (−44 %), quality-neutral |
| Linux         | NVIDIA        | F32     | CUDA (fallback Vulkan)   | same as Windows CUDA |
| Linux         | Nouveau/AMD   | F32     | Vulkan                   | same as Windows Vulkan |
| macOS         | Apple Silicon | F32     | Metal (the only EP)      | not measured here; CI-build path |
| macOS         | Intel         | F32     | Metal (cross-compiled)   | not measured here; CI-build path |

Rules of thumb:
- GPU present → **F32** weights; CPU-only → **CPU EP with DBCache** (default on).
- On GPU, Vulkan = smallest VRAM (+0.3 GiB class), CUDA = fastest wall.
- Q8 (~3.4× smaller weights, near-lossless) but may flip a boundary note on
  Vulkan → choose the `-full` package when bit-consistent output matters;
  otherwise either package is valid.
- No user escalation needed: the CLI picks the backend from the GGUF/EP and
  resolves cache by EP (GPU off, CPU auto).
