#!/usr/bin/env python3
"""Recompute frame-level metrics (presF1/RMSE/RPA/OA) for the refreshed GGML
rows against the stored torch-CUDA fp32 no-cache baseline (report.py formulas).

Runner for docs/benchmark-7channel.md refresh — fills back the quality columns
that the previous doc had, using the ORIGINAL torch reference CSV (not a re-run).
"""
from __future__ import annotations
import csv, json, math, os, re, sys

TIME = 0.01
BASE = r"J:\GGML-GAME\dbcache_ablation\results"
REF  = os.path.join(BASE, "canon", "torch_cuda.csv")   # normalized torch fp32 n8 no-cache
OUT_BASE = r"J:\GGML-GAME\_review_game_cpp\bench_out"

LETTER = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

def pitch_float(p):
    """'E5+37' -> midi semitones float (76.37); 'rest'/'0' -> None."""
    p = p.strip()
    if p in ("rest", "0", ""):
        return None
    try:
        return float(p)          # already numeric (canon)
    except ValueError:
        pass
    m = re.fullmatch(r"([A-G])(#?)(\d+)([+-]\d+)?", p)
    if not m:
        raise ValueError(f"unparsable pitch {p!r}")
    letter, sharp, octave, cents = m.groups()
    semi = LETTER[letter] + (1 if sharp else 0) + (int(octave) + 1) * 12
    return semi + int(cents or 0) / 100.0


# tag -> outdir name (warm vk used for vk rows in the doc)
ROWS = {
    "ggml_cpu F32 off":  "cpu-F32-n8-0",
    "ggml_cpu F32 on":   "cpu-F32-n8-1",
    "ggml_cpu Q8 off":   "cpu-Q8-n8-0",
    "ggml_cpu Q8 on":    "cpu-Q8-n8-1",
    "ggml_vk F32 off":   "vk_warm-F32-n8-0",
    "ggml_vk F32 on":    "vk_warm-F32-n8-1",
    "ggml_vk Q8 off":    "vk_warm-Q8-n8-0",
    "ggml_vk Q8 on":     "vk_warm-Q8-n8-1",
    "ggml_cuda F32 off": "cuda12-F32-n8-0",
    "ggml_cuda F32 on":  "cuda12-F32-n8-1",
    "ggml_cuda Q8 off":  "cuda12-Q8-n8-0",
    "ggml_cuda Q8 on":   "cuda12-Q8-n8-1",
}

def load(path):
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            p = pitch_float(r.get("pitch", ""))
            out.append((float(r["offset"]), float(r["duration"]), p))
    return out

def raster(notes, T):
    pres = [False] * T; sc = [0.0] * T
    for o, d, p in notes:
        if p is None:
            continue
        a = max(0, min(T, int(round(o / TIME)))); b = max(a, min(T, int(round((o + d) / TIME))))
        for fr in range(a, b):
            pres[fr] = True; sc[fr] = p
    return pres, sc

def metrics(pp, ps, tp, ts):
    N = len(pp)
    tp_ = sum(p and t for p, t in zip(pp, tp)); fp_ = sum(p and not t for p, t in zip(pp, tp))
    fn_ = sum(not p and t for p, t in zip(pp, tp))
    prec = tp_ / (tp_ + fp_ + 1e-6); rec = tp_ / (tp_ + fn_ + 1e-6); f1 = 2 * prec * rec / (prec + rec + 1e-6)
    tv = sum(t for t in tp)
    se = sum((s - t) ** 2 for s, t in zip(ps, ts) if t); rmse = math.sqrt(se / (tv + 1e-6))
    rpa = sum(1 for s, t in zip(ps, ts) if t and abs(s - t) <= 0.5) / (tv + 1e-6)
    oa = sum(1 for p, s, t, u in zip(pp, ps, tp, ts)
             if (p and t and abs(s - u) <= 0.5) or ((not t) and (not p))) / N
    return dict(rmse=rmse, rpa=rpa, presf1=f1, prec=prec, rec=rec, oa=oa)

def main():
    ref = load(REF)
    T = int(round(max([0.0] + [o + d for o, d, _ in ref]) / TIME)) + 1
    tp, ts = raster(ref, T)
    res = {}
    for tag, sub in ROWS.items():
        p = os.path.join(OUT_BASE, sub, "w44k_60.csv")
        if not os.path.isfile(p):
            print(f"!! missing {p}"); continue
        pp, ps = raster(load(p), T)
        m = metrics(pp, ps, tp, ts)
        res[tag] = m
        print(f"{tag:<18} presF1={m['presf1']:.4f} RMSE={m['rmse']:.3f} "
              f"RPA={m['rpa']:.3f} OA={m['oa']:.3f}")
    with open(os.path.join(OUT_BASE, "frame_metrics.json"), "w") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()
