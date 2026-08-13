# GAME ggml backend

> **Languages:** [English](README.md) | [中文](README_CN.md)

Native C++ inference for the [GAME](https://github.com/openvpi/GAME) singing-voice-to-MIDI model, built on
[ggml](https://github.com/ggerganov/ggml).  Runs on CPU, Metal (default on
Apple Silicon), CUDA, or Vulkan.  Drop-in replacement for
`python infer.py extract` with no Python dependency at runtime.

## Highlights

- **End-to-end CLI** — WAV in, MIDI/TXT/CSV out (mirrors Python `extract`).
- **Small footprint** — ~50 MB GGUF for the 1.0-medium checkpoint, ~50 M params.
- **Fast startup** — Metal binary-archive patch keeps first-run latency under a
  second on Apple Silicon.
- **Third-party integration** — clean PIMPL C++ API; `add_subdirectory` and link
  `game_ggml::game_ggml`.
- **Parity-tested** — full pipeline output matches the PyTorch reference bit-for-bit
  when the same RNG numbers are injected.

## Architecture

```
waveform (44100 Hz mono)
      │
      ▼
   MelExtractor (pocketfft STFT + librosa-compatible mel)
      │ mel [T, 80]
      ▼
   Encoder (EBFBackbone, 4 layers, dim=128)
      │ x_seg, x_est  each [T, 128]
      ▼
   D3PM loop (1 step default; --nsteps 8 for higher quality)
     ├─ remove_mutable_boundaries (stochastic)
     ├─ Segmenter (EBFBackbone, 8 layers; noise/time/lang embeddings)
     └─ decode_soft_boundaries (local-max)
      │ regions [T]  +  N
      ▼
   Estimator (JEBFBackbone, 4 layers; joint attention, mixed RoPE)
      │ pool_logits [N, 257]
      ▼
   Gaussian-blurred pitch decode → notes (offset, duration, pitch, voiced)
```

## Build

```bash
cmake -S ggml_backend -B ggml_backend/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DGAME_GGML_BUILD_TESTS=ON
cmake --build ggml_backend/build -j
```

Options:

| Option | Default | Meaning |
|---|---|---|
| `GAME_GGML_METAL`       | `ON` (Apple only) | Enable Metal backend |
| `GAME_GGML_CUDA`        | `OFF`             | Enable CUDA backend |
| `GAME_GGML_VULKAN`      | `OFF`             | Enable Vulkan backend |
| `GAME_GGML_BUILD_CLI`   | `ON`              | Build `game_ggml_cli` |
| `GAME_GGML_BUILD_TESTS` | `OFF`             | Build GoogleTest suite |

## Convert a PyTorch checkpoint

```bash
pip install -r ggml_backend/scripts/requirements.txt
python ggml_backend/scripts/convert_pt_to_gguf.py \
    --model-dir GAME-pt-1.0-small \
    -o ggml_backend/assets/game_small.gguf
```

The script reads `model.pt` + `config.yaml` + `lang_map.json` from the given
directory and writes a single GGUF file containing all 671 tensors (FP32) and
74 metadata KV pairs.

Inspect the result:

```bash
./ggml_backend/build/bin/game_ggml_cli inspect ggml_backend/assets/game_small.gguf
```

## Run inference

```bash
./ggml_backend/build/bin/game_ggml_cli extract input.wav \
    -m ggml_backend/assets/game_small.gguf \
    --output-formats mid,txt,csv \
    --output-dir out/ \
    --tempo 120 \
    --seed 42
```

### CLI → `infer.py extract` option mapping

| CLI flag | Python equivalent |
|---|---|
| `-m / --model`        | `-m` |
| `-l / --language`     | `-l` (takes numeric id — use `inspect` to see the mapping) |
| `--output-formats`    | `--output-formats` |
| `--output-dir`        | `--output-dir` |
| `--tempo`             | `--tempo` |
| `--seg-threshold`     | `--seg-threshold` |
| `--seg-radius`        | `--seg-radius` (in frames) |
| `--est-threshold`     | `--est-threshold` |
| `--t0` / `--nsteps`   | `--t0` / `--nsteps` (ggml defaults to `--nsteps 1`; Python defaults to `8`) |
| `--seed`              | *(new)* — 0 pulls a random seed from the OS |
| `--pitch-format`      | `--pitch-format` |
| `--round-pitch`       | `--round-pitch` |
| `--rng-replay <path>` | *(new)* — replay D3PM random numbers from a file for bit-exact parity with PyTorch |

## Performance

Measured on Apple M4 (macOS, 16-core Apple Silicon), 3 runs per side under
`/usr/bin/time -l`, fresh subprocess each run (so peak RSS is clean).  Input:
`28.wav` — 216.0 s mono, resampled to 44.1 kHz.

Both sides consume the exact same random-number stream via `--rng-replay`
(see `scripts/align_demo.py`), so the note lists are bit-exact aligned.

### Default settings (`--nsteps 1`)

```
                          min      mean       max
PyTorch wall (s)         5.99      6.27      6.71   (MPS via Lightning)
ggml    wall (s)         3.05      3.06      3.06   (Metal, default binary)
Speedup                                    2.05 ×

PyTorch peak RSS      980.8 MB  981.2 MB  981.7 MB
ggml    peak RSS      334.1 MB  334.4 MB  334.8 MB
Memory ratio                                2.93 ×

PyTorch notes: 471  ┐
ggml    notes: 471  ├── matched 1-to-1, max |Δpitch| = 0.000 semitone
                     ┘
```

Real-time factor: **70.6×** (ggml) vs 34.4× (PyTorch).

### Higher quality (`--nsteps 8`)

```
                          min      mean       max
PyTorch wall (s)        17.73     18.05     18.65
ggml    wall (s)         9.11      9.28      9.62
Speedup                                    1.94 ×
```

Real-time factor drops to 23.3× (ggml) / 11.97× (PyTorch), but segmentation
quality is marginally higher (+9 notes recovered in this clip).

### Per-stage breakdown (ONNX-aligned)

Run with `GAME_GGML_PROFILE=1` to print a per-chunk breakdown.  Numbers below
use `--nsteps 8` so the segmenter share is visible:

```
encoder     ~0.17 s  (~16%)   waveform → x_seg/x_est  (mel + spec_proj + 4× EBF)
segmenter   ~0.79 s  (~78%)   x_seg → boundaries       (8× D3PM sampling steps)
estimator   ~0.06 s  (~ 6%)   x_est + regions → notes  (4× JEBF + joint attn)
```

Segmenter dominates because D3PM loops it `--nsteps` times.  Default `1`
keeps it cheap; bump to `4` or `8` for higher quality at linear cost.

## Reproducing the benchmark

```bash
# 1. Resample to 44.1 kHz / mono (if not already)
python3 -c "
import librosa, soundfile as sf
y, _ = librosa.load('28.wav', sr=44100, mono=True)
sf.write('/tmp/28_44100.wav', y, 44100, subtype='PCM_16')"

# 2. Capture PyTorch's D3PM RNG stream (also produces a reference MIDI)
python3 ggml_backend/scripts/align_demo.py /tmp/28_44100.wav \
    -m GAME-pt-1.0-small/model.pt \
    -g ggml_backend/assets/game_small.gguf \
    --cli ggml_backend/build/bin/game_ggml_cli \
    -l zh -o /tmp/align_out

# 3. Run the 3-per-side subprocess-isolated benchmark
python3 ggml_backend/scripts/benchmark_align.py /tmp/28_44100.wav \
    -m GAME-pt-1.0-small/model.pt \
    -g ggml_backend/assets/game_small.gguf \
    --cli ggml_backend/build/bin/game_ggml_cli \
    --rng /tmp/align_out/align_rng.bin \
    -l zh -o /tmp/bench_out --runs 3
```

## Using as a third-party library

```cmake
add_subdirectory(path/to/GAME/ggml_backend)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE game_ggml::game_ggml)
```

```cpp
// main.cpp
#include <game_ggml/model.h>
#include <vector>

int main() {
    auto model = game_ggml::Model::load("game_small.gguf");
    std::vector<float> waveform = /* ... load 44100 Hz mono ... */;

    game_ggml::InferParams params;
    params.language = 4;   // from lang_map: { "zh": 4 }
    params.seed     = 42;

    auto result = model.infer(waveform.data(), waveform.size(), params);
    for (const auto & n : result.notes) {
        if (!n.voiced) continue;
        printf("  %.2fs + %.2fs : %.2f\n",
               n.offset_seconds, n.duration_seconds, n.pitch_midi);
    }
}
```

The public header `<game_ggml/model.h>` uses PIMPL; consumers never transitively
include any ggml header.

See [`examples/external_consumer/`](examples/external_consumer/) for a minimal
standalone CMake project that builds against the library.

## Tests

```bash
ctest --test-dir ggml_backend/build --output-on-failure
```

The suite has 37 tests covering:

- Backend initialisation
- GGUF I/O round-trip
- Every op (RMSNorm, Linear, LayerScale, Embedding, GLU-FFN, CgMLP, RoPE in all
  three modes, Attention, PAC, EBF block)
- Encoder / Segmenter / Estimator end-to-end vs PyTorch reference dumps
- D3PM 8-step loop bit-exact with injected RNG (tolerates ≤ 2/100 boundary
  flips from Metal FP32 drift)
- Mel spectrogram vs `lib.feature.mel.StretchableMelSpectrogram`
- Slicer (short-clip + split-on-silence)
- MIDI writer (SMF type-0 structure)
- Text writers (TXT + CSV, note-name formatting)
- Full pipeline bit-exact E2E

Reference dumps are generated by `python scripts/dump_reference.py --category all`.
Dumps are gitignored — regenerate them as part of CI.

## Known limitations (v1)

- **44100 Hz mono WAV only** — other sample rates raise `InvalidWav`.  Resampling
  is deliberately out-of-scope to keep the footprint small.
- **FP32 weights only** — the converter emits FP32 GGUF.  Quantization is a
  later deliverable.
- **Only the shipped `1.0-small` config branch is supported**.  The estimator
  rejects `split` attention, learned pool merger, `region_token_num > 1`, and
  `use_region_bias=true` at load time with a clear `NotImplemented` message.
- **Batch size 1** per inference call — matches `infer.py extract`.  For
  parallel streams hold multiple `Model` instances.
- **Metal FP32 precision** — expected ~1e-3 per matmul; at boundary decoding
  this can flip one frame out of every few hundred vs the CPU reference.

## Dependencies

Everything is fetched at configure time by CMake; nothing is vendored.  Source
trees live under `build/_deps/<name>-src/` after the first configure.

| Dependency | Version pin | License | SPDX identifier |
|---|---|---|---|
| [ggml](https://github.com/ggerganov/ggml) | `v0.11.0` tag | MIT | MIT |
| [pocketfft](https://gitlab.mpcdf.mpg.de/mtr/pocketfft) | commit `32424d20` on `cpp` branch | BSD-3-Clause | BSD-3-Clause |
| [dr_libs](https://github.com/mackron/dr_libs) | commit `243e26ff` on `master` | Public Domain / MIT-0 (dual) | `Unlicense OR MIT-0` |
| [GoogleTest](https://github.com/google/googletest) | `v1.14.0` tag (tests only) | BSD-3-Clause | BSD-3-Clause |

Each upstream LICENSE file is preserved under `build/_deps/<name>-src/LICENSE*`
after download.  To update a dependency, change its `GIT_TAG` in
`cmake/Dependencies.cmake` and reconfigure.

## License

MIT — same as the parent [GAME project](https://github.com/openvpi/GAME).  Redistributions should
also carry the upstream license notices listed in the table above.
