"""Codebook state compression sweep (SM90): size and encode-cost table of the ladder recipes (CPU, seconds).

Per recipe (scripts/state_vq.py RECIPES; the released small model: card = 2 WKV layers x 4 heads + 3 token-shift
vectors, note = 1 layer x 4 heads + 2 vectors, 16 x 16 heads, 64-float vectors):
  - bits per card / note; card/note state MB for the fork's example collection (13,973 cards, 13,198 notes; codes
    only, without the fork's 19-byte record header, which adds 0.5 MB);
  - codebook MB: the stages the recipe reads (catalogs are shared by card and note: per WKV component the larger stage
    count of the two streams, per shift role / sub-vector position the larger stage count), float32 and float16;
  - encode cost per review in the fork's CPU runtime: one card and one note update per review; per updated head the
    Gram matrix (16^3 MAC) + a 16 x 16 symmetric eigendecomposition + per residual stage a nearest-codeword search over
    2^10 codewords of 32 floats (2^10 x 32 MAC); per stored shift vector and stage a search over 2^12 codewords per
    sub-vector (2^12 x 64 MAC over the m sub-vectors). The ms estimate times these kernels here, single-threaded, with
    numpy (BLAS matrix-vector product, LAPACK eigh) at batch 1 as the Rust runtime would run them per review.
Reference points: int8 / int4+int8 state (the fork's format: 4,554 / 3,530 B per card + note) cost no search.

usage: python -X utf8 artifacts/small-model/sm90/cost_sm90.py vq5902x3028 vqB2a ... [--json out.json]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from scripts import state_vq as sv  # noqa: E402

CARDS, NOTES = 13_973, 13_198
LAYOUT = {"card": (2, 4, 3), "note": (1, 4, 2)}   # (WKV layers, heads, shift vectors)
FORK_BYTES = {"int8": (2_987, 1_567), "int4 cards + int8 notes": (1_963, 1_567)}   # (card, note) B, INTEGRATION.md 11.1


def bench(fn, reps=2000):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def kernels():
    rng = np.random.default_rng(0)
    cb10 = rng.standard_normal((1024, 32)).astype(np.float32)
    cb12 = {d: rng.standard_normal((4096, d)).astype(np.float32) for d in (8, 16, 32)}
    x32 = rng.standard_normal(32).astype(np.float32)
    A = rng.standard_normal((16, 16))
    G = A @ A.T
    n10 = (cb10 * cb10).sum(1)
    t = {"wkv_stage": bench(lambda: int(np.argmin(n10 - 2 * (cb10 @ x32)))),
         "eigh16": bench(lambda: np.linalg.eigh(G)),
         "gram16": bench(lambda: A @ A.T)}
    for d, cb in cb12.items():
        n = (cb * cb).sum(1)
        x = rng.standard_normal(d).astype(np.float32)
        t[f"shift_stage_d{d}"] = bench(lambda cb=cb, n=n, x=x: int(np.argmin(n - 2 * (cb @ x))), reps=1000)
    return t


def cost(name, t):
    recs = dict(zip(("card", "note"), sv.recipe(name)))
    out = {"recipe": name}
    wkv_stages, shift_stages = {}, {}
    macs = 0
    ms = 0.0
    for s, rec in recs.items():
        layers, heads, vectors = LAYOUT[s]
        info = {"wkv_layers": layers, "heads": heads, "shift_vectors": [None] * vectors}
        if isinstance(rec, sv.IntStream):   # a stream at the fork default, sized in the fork's format
            out[f"{s}_bits"] = 8 * {("card", 4): 1_963, ("note", 8): 1_567, ("card", 8): 2_987}[(s, rec.bits)]
            out[f"{s}_format"] = f"int{rec.bits} (fork format)"
            out[f"{s}_encode_macs"] = 0
            continue
        bits = sv.entity_bits(rec, info)
        out[f"{s}_bits"] = bits
        nh = layers * heads
        st = sum(rec.stages) if rec.rank else 0
        for j in range(rec.rank):
            wkv_stages[j] = max(wkv_stages.get(j, 0), rec.stages[j])
        if rec.shift_m:
            shift_stages[rec.shift_m] = max(shift_stages.get(rec.shift_m, 0), rec.shift_stages)
        m_wkv = nh * ((16 ** 3 if rec.rank else 0) + st * 1024 * 32)
        m_shift = vectors * rec.shift_stages * 4096 * 64 if rec.shift_m else 0
        macs += m_wkv + m_shift
        ms += nh * ((t["eigh16"] + t["gram16"] if rec.rank else 0) + st * t["wkv_stage"])
        if rec.shift_m:
            ms += vectors * rec.shift_stages * rec.shift_m * t[f"shift_stage_d{64 // rec.shift_m}"]
        out[f"{s}_encode_macs"] = m_wkv + m_shift
    cb_floats = sum(wkv_stages.values()) * 1024 * 32 + 2 * sum(st * 4096 * 64 for st in shift_stages.values())
    out.update({
        "state_mb": (CARDS * out["card_bits"] + NOTES * out["note_bits"]) / 8 / 1e6,
        "bytes_per_card_note": (out["card_bits"] + out["note_bits"]) / 8,
        "codebook_mb_f32": cb_floats * 4 / 1e6, "codebook_mb_f16": cb_floats * 2 / 1e6,
        "wkv_stages_read": [wkv_stages[j] for j in sorted(wkv_stages)],
        "shift_stages_read": {f"m{m}": v for m, v in shift_stages.items()},
        "encode_macs_per_review": macs, "encode_ms_per_review_est": ms * 1e3})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recipes", nargs="+")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    t = kernels()
    print("[cost] kernel times (us, numpy single thread, batch 1): " + ", ".join(f"{k} {v * 1e6:.1f}" for k, v in t.items()))
    rows = [cost(r, t) for r in args.recipes]
    print("| recipe | card bits | note bits | B per card + note | state MB | codebook MB f32 / f16 | encode MMAC / review | encode ms / review (est.) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, (c, n) in FORK_BYTES.items():
        print(f"| {name} (fork format) | {8 * c:,} | {8 * n:,} | {c + n:,} | {(c * CARDS + n * NOTES) / 1e6:.1f} | 0 | 0 | 0 |")
    for r in rows:
        print(f"| {r['recipe']} | {r['card_bits']:,} | {r['note_bits']:,} | {r['bytes_per_card_note']:,.0f} | {r['state_mb']:.1f} | "
              f"{r['codebook_mb_f32']:.1f} / {r['codebook_mb_f16']:.1f} | {r['encode_macs_per_review'] / 1e6:.1f} | "
              f"{r['encode_ms_per_review_est']:.2f} |")
    if args.json:
        Path(args.json).write_text(json.dumps({"kernel_seconds": t, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
