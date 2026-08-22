#!/usr/bin/env python3
"""Re-run the GGML rows of the 7-channel benchmark under the original conditions.

Mirrors dbcache_ablation/{bench_ggml.ps1, ggml_n8.py}: 60 s wav, no slice,
tempo 120, seed 42, nsteps ∈ {1,8}, DBCache = {threshold 0.25, fn_blocks 1,
warmup 1} on / threshold 0 off.  Prints wall + per-stage profile + notes +
DBCache hit/miss per run.

Usage:
    run_ggml_bench.py --exe <game_ggml_cli.exe> --tag <name> [--wav <w>] \
        [--model-dir <dir-with-game_medium[.gguf|_q8.gguf]>] [--gpu]
"""
from __future__ import annotations
import argparse, os, re, subprocess, sys, time, threading, glob

WAV_DEFAULT = r"J:\GGML-GAME\bench_dml_compare\input\w44k_60.wav"
MODEL_DIR_DEFAULT = r"J:\GGML-GAME\game_ggml_cli\build-local-cpu\ml_qkv"
VK_CACHE = r"J:\GGML-GAME\_vkpc\pipeline.cache"

def vram_mib():
    try:
        t = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2)
        return float(t.stdout.strip().splitlines()[0].strip()) if t.stdout and t.returncode == 0 else 0.0
    except Exception:
        return 0.0

def run(exe, model, wav, outdir, nsteps, cache, gpu):
    cmd = [exe, "extract", wav, "-m", model,
           "--output-formats", "csv", "--output-dir", outdir,
           "--tempo", "120", "--seed", "42", "--no-slice",
           "--nsteps", "8" if nsteps else "1"]
    if nsteps:
        if cache:
            cmd += ["--cache-threshold", "0.25", "--cache-fn-blocks", "1", "--cache-warmup", "1"]
        else:
            cmd += ["--cache-threshold", "0"]
    env = dict(os.environ)
    env["GAME_GGML_PROFILE"] = "1"
    env["GAME_GGML_THREADS"] = "16"
    env["GAME_GGML_DUMP_DBCACHE"] = "1"
    env["GGML_VK_PIPELINE_CACHE_PATH"] = VK_CACHE
    base = vram_mib() if gpu else 0.0
    peak = {"v": base}
    stop = threading.Event()
    def sampler():
        while not stop.is_set():
            if gpu:
                v = vram_mib()
                if v > peak["v"]: peak["v"] = v
            time.sleep(0.04)
    th = threading.Thread(target=sampler, daemon=True); th.start()
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         encoding="utf-8", errors="replace")
    log = "".join(l for l in p.stdout)
    p.wait()
    wall = time.perf_counter() - t0
    stop.set(); th.join(1.0)
    total = re.search(r"total\s+([\d.]+)\s*s", log)
    notes = re.search(r"total notes:\s*(\d+)", log)
    hits = len(re.findall(r"\bHIT\b", log)); misses = len(re.findall(r"\bMISS\b", log))
    stage = {}
    for m in re.finditer(r"\s+(\S+)\s+([\d.]+)\s*s\s+\(\s*([\d.]+)%\s*\)", log):
        stage[m.group(1)] = (m.group(2), m.group(3))
    dv = (peak["v"] - base) if gpu else None
    return dict(wall=wall, total=total.group(1) if total else "?", notes=notes.group(1) if notes else "?",
                hits=hits, misses=misses, vram=("N/A" if dv is None else f"+{dv:.0f}MiB"), stage=stage, rc=p.returncode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--wav", default=WAV_DEFAULT)
    ap.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--only-n8", action="store_true")
    a = ap.parse_args()
    models = {"F32": os.path.join(a.model_dir, "game_medium.gguf"),
              "Q8": os.path.join(a.model_dir, "game_medium_q8.gguf")}
    for mname, mod in models.items():
        if not os.path.isfile(mod):
            print(f"!! missing {mod}"); continue
        for nsteps in (0, 8) if not a.only_n8 else (8,):
            for cache in ((False, True) if nsteps else (False,)):
                tag = f"{a.tag} {mname} n{'8' if nsteps else '1'}{' cache' if cache else ''}"
                out = os.path.join("bench_out", f"{a.tag}-{mname}-n{nsteps if nsteps else 1}-{1 if cache else 0}")
                os.makedirs(out, exist_ok=True)
                r = run(a.exe, mod, a.wav, out, nsteps, cache, a.gpu)
                st = " | ".join(f"{k}={v}s({p}%)" for k, (v, p) in r["stage"].items())
                print(f"GGML {tag}: wall={r['wall']:.3f}s total={r['total']}s notes={r['notes']} "
                      f"dbc={r['hits']}/{r['misses']} vram={r['vram']} rc={r['rc']}")
                if st: print(f"      {st}")

if __name__ == "__main__":
    main()
