#!/usr/bin/env python3
"""Multi-backend verification for game_ggml_cli.

Runs the same waveform through the CPU / Vulkan / CUDA CLI builds
(nsteps=1 fused path and nsteps=8 DBCache path) and compares outputs:

  * CSV note lists (structure: note count, per-note pitch/offset/duration)
  * DBCache hit/miss pattern (via GAME_GGML_DUMP_DBCACHE=1 stderr)

Usage:
    python verify_backends.py --cli-cpu build/bin/game_ggml_cli.exe \
        --cli-vk build-vk/bin/game_ggml_cli.exe \
        --cli-cuda build-cuda/bin/game_ggml_cli.exe \
        --wav test10s.wav -m model.gguf -o outdir

Exit code 0 = all backend outputs match CPU reference within tolerance.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import subprocess
import sys

# Note-list tolerance: pitch is quantised (midi cents), offset/duration in
# seconds with 2-decimal CSV precision.  GPUs may shift a boundary by one
# frame (11.61 ms) at a note edge.
PITCH_EPS = 5    # cents
TIME_EPS = 0.05  # s


def run_cli(cli: pathlib.Path, wav: pathlib.Path, model: pathlib.Path,
            out_dir: pathlib.Path, seed: int, nsteps: int,
            env_extra: dict | None = None) -> tuple[int, str, str]:
    cmd = [str(cli), "extract", str(wav), "-m", str(model),
           "--output-dir", str(out_dir), "--seed", str(seed),
           "--output-formats", "mid,csv,txt"]
    if nsteps:
        cmd += ["--nsteps", str(nsteps)]
    env = dict(subprocess.os.environ)
    env["GAME_GGML_DUMP_DBCACHE"] = "1"
    if env_extra:
        env.update(env_extra)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    return r.returncode, r.stdout, r.stderr


def read_notes(csv_path: pathlib.Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def parse_pitch(p: str) -> int | None:
    """'A3-37' -> midi-ish cents value; 'rest' -> None.

    Raises ValueError for any string that is neither 'rest' nor a parsable
    pitch — a malformed pitch must not silently compare equal to a rest.
    """
    if p == "rest":
        return None
    m = re.match(r"([A-G])(#?)(\d+)([+-]\d+)?", p)
    if not m:
        raise ValueError(f"unparsable pitch: {p!r}")
    letter, sharp, octave, cents = m.groups()
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter]
    semi = base + (1 if sharp else 0) + (int(octave) + 1) * 12
    return semi * 100 + int(cents or 0)


def notes_close(a: list[dict], b: list[dict]) -> tuple[bool, str]:
    if len(a) != len(b):
        return False, f"note count {len(a)} != {len(b)}"
    for i, (ra, rb) in enumerate(zip(a, b, strict=True)):
        pa, pb = parse_pitch(ra["pitch"]), parse_pitch(rb["pitch"])
        if (pa is None) != (pb is None):
            return False, f"note[{i}] pitch rest-mismatch {ra['pitch']} vs {rb['pitch']}"
        if pa is not None and abs(pa - pb) > PITCH_EPS:
            return False, f"note[{i}] pitch {ra['pitch']} vs {rb['pitch']}"
        for k in ("offset", "duration"):
            da, db = float(ra[k]), float(rb[k])
            if abs(da - db) > TIME_EPS:
                return False, f"note[{i}] {k} {da} vs {db}"
    return True, ""


def db_pattern(stderr: str) -> str:
    hits = len(re.findall(r"DB.?cache hit|hit", stderr, re.I))
    misses = len(re.findall(r"miss", stderr, re.I))
    m = re.search(r"(?:hit|miss).*?(\d+).*?(\d+)", stderr)
    return f"hits~{hits}/misses~{misses}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli-cpu", required=True, type=pathlib.Path)
    ap.add_argument("--cli-vk", type=pathlib.Path, default=None)
    ap.add_argument("--cli-cuda", type=pathlib.Path, default=None)
    ap.add_argument("--wav", required=True, type=pathlib.Path)
    ap.add_argument("-m", "--model", required=True, type=pathlib.Path)
    ap.add_argument("-o", "--out-root", required=True, type=pathlib.Path)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    backends = [("cpu", args.cli_cpu)]
    if args.cli_vk:
        backends.append(("vk", args.cli_vk))
    if args.cli_cuda:
        backends.append(("cuda", args.cli_cuda))

    args.out_root.mkdir(parents=True, exist_ok=True)
    rc_total = 0

    for nsteps in (1, 8):
        print(f"\n===== nsteps={nsteps} =====")
        results = {}
        for name, cli in backends:
            out = args.out_root / f"{name}_n{nsteps}"
            out.mkdir(exist_ok=True)
            code, so, se = run_cli(cli, args.wav, args.model, out, args.seed,
                                   nsteps if nsteps > 1 else 0)
            csv_path = out / f"{args.wav.stem}.csv"
            notes = read_notes(csv_path)
            ok = code == 0
            results[name] = (notes, code, db_pattern(se), out)
            print(f"[{name}] rc={code} notes={len(notes)} dbc={db_pattern(se)}")

        ref, ref_code = results["cpu"][0], results["cpu"][1]
        if ref_code != 0:
            print("  !! cpu reference non-zero exit; comparison is meaningless")
            rc_total = 1
            continue
        if not ref:
            print("  !! cpu reference produced no notes; comparison is meaningless")
            rc_total = 1
            continue
        if len(backends) == 1:
            print("  !! no GPU backend supplied; nothing to compare against")
            rc_total = 1
            continue
        for name in list(results)[1:]:
            notes, code, _, out = results[name]
            if code != 0:
                print(f"  !! {name} non-zero exit")
                rc_total = 1
                continue
            close, why = notes_close(ref, notes)
            status = "MATCH" if close else "DIFF"
            print(f"  {name}: {status} vs cpu" + (f" ({why})" if why else ""))
            if not close:
                rc_total = 1

    print(f"\n{'ALL BACKENDS MATCH' if rc_total == 0 else 'MISMATCH DETECTED'}")
    return rc_total


if __name__ == "__main__":
    sys.exit(main())
