# `scripts/`

Python helpers that complement the C++ binaries.  All scripts assume they are
run from the repo root and have `torch`, `numpy`, `gguf`, `pyyaml`,
`librosa`, `soundfile`, `mido` available (see `requirements.txt`).

| Script | Purpose |
|---|---|
| `convert_pt_to_gguf.py` | Convert `GAME-pt-1.0-small/model.pt` + `config.yaml` + `lang_map.json` → one FP32 GGUF file (all 671 tensors, 74 KV pairs). |
| `inspect_gguf.py`       | Pure-Python GGUF inspector; prints metadata + first N tensors. |
| `dump_reference.py`     | Produce `.bin` reference tensors used by the GoogleTest suite.  Category-based registry (`basic`, `ffn`, `rope`, `mel`, `encoder`, `segmenter`, `estimator`, `pipeline`). |
| `align_demo.py`         | Run PyTorch + ggml on the same waveform with bit-exact RNG alignment.  Captures `torch.rand_like` into `align_rng.bin`, feeds it to the C++ CLI via `--rng-replay`, and reports timestamp/pitch alignment. |
| `benchmark_align.py`    | Follow-up to `align_demo.py`: runs both backends N times each in subprocess isolation under `/usr/bin/time -l`, reports wall-time + peak-RSS statistics. |

## Typical workflow

```bash
pip install -r scripts/requirements.txt

# one-time: produce the GGUF for the C++ backend
python scripts/convert_pt_to_gguf.py \
    --model-dir GAME-pt-1.0-small \
    -o game_small.gguf

# each time you touch a C++ op: regenerate the reference dumps so tests pass
python scripts/dump_reference.py --category all --out tests/data

# one-off: verify bit-exact alignment on a real clip
python scripts/align_demo.py /path/to/your.wav \
    -m GAME-pt-1.0-small/model.pt \
    -g game_small.gguf \
    --cli build/bin/game_ggml_cli \
    -l zh -o /tmp/align_out

# one-off: speed + memory benchmark (reuses align_rng.bin from align_demo)
python scripts/benchmark_align.py /path/to/your.wav \
    -m GAME-pt-1.0-small/model.pt \
    -g game_small.gguf \
    --cli build/bin/game_ggml_cli \
    --rng /tmp/align_out/align_rng.bin \
    -l zh -o /tmp/bench_out --runs 3
```

## Notes

- `dump_reference.py` categories that depend on the real checkpoint (`mel`,
  `encoder`, `segmenter`, `estimator`, `pipeline`) skip cleanly if
  `GAME-pt-1.0-small/` isn't present; the `basic`, `ffn`, `rope` categories
  are fully synthetic and always work.
- `align_demo.py` and `benchmark_align.py` intentionally **do not** use
  Lightning Trainer — they loop chunks sequentially to keep the RNG
  consumption order identical between both sides.
- All outputs go to user-specified `-o` directories; nothing is written back
  into the repo.
