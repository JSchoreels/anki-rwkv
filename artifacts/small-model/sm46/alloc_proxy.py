"""SM46: model-free bit-allocation proxy for the large-budget state codec (CPU, minutes).

On the HELD-OUT corpus users of artifacts/small-model/sm46/books (state_vq_fit.py fit --holdout-every k: every k-th
corpus user is not used by the k-means), for the SM36 student's stored card/note states:
  - WKV rank-r truncation: share of the head-matrix energy kept, r = 1..8;
  - WKV codec: relative squared reconstruction error sum ||A - A_hat||^2 / sum ||A||^2 per (rank, stages, norm bits);
  - shift codec: relative squared error per role for (m, stages, norm bits).
The codec is scripts/state_vq.py StateCodec (eigh factorization, float64), so the numbers are those of the recipes.

usage: python -X utf8 artifacts/small-model/sm46/alloc_proxy.py [--books codebooks-sm46.npz] [--holdout-every 10]
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts import state_vq as sv  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
D = Path(__file__).resolve().parent / "books"
# per-head WKV stage allocations to compare at ~550-650 bits per head (10-bit norms)
CANDIDATES = [(20, 16, 12, 8), (18, 15, 11, 8, 4), (16, 14, 11, 8, 5, 3), (15, 13, 10, 8, 6, 4),
              (14, 12, 10, 8, 6, 4), (24, 18, 14), (12, 10, 8, 6, 4, 3)]


def held_out(every, max_heads):
    files = sorted((D / "corpus").glob("*.npz"))
    files = [f for i, f in enumerate(files) if i % every == every - 1]
    wkv, shift = [], {"t": [], "c": []}
    for f in files:
        z = np.load(f)
        for key in z.files:
            parts = key.split("/")
            if parts[0] == "wkv":
                wkv.append(z[key].reshape(-1, sv.K, sv.K))
            else:
                shift[parts[3]].append(z[key])
    g = torch.Generator().manual_seed(7)
    A = torch.from_numpy(np.concatenate(wkv)).float()
    A = A[torch.randperm(A.shape[0], generator=g)[:max_heads]]
    S = {r: torch.from_numpy(np.concatenate(v)).float() for r, v in shift.items()}
    return [f.stem for f in files], A, S


def wkv_err(A, rec, books):
    codec = sv.StateCodec(rec, books)
    out = codec.wkv(A[:, None])[:, 0].double()
    return float(((A.double() - out) ** 2).sum() / (A.double() ** 2).sum())


def shift_err(X, role, rec, books):
    codec = sv.StateCodec(rec, books)
    out = codec.shift(X, role).double()
    return float(((X.double() - out) ** 2).sum() / (X.double() ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default="codebooks-sm46.npz")
    ap.add_argument("--holdout-every", type=int, default=10)
    ap.add_argument("--max-heads", type=int, default=20000)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "alloc-proxy.json"))
    args = ap.parse_args()
    users, A, S = held_out(args.holdout_every, args.max_heads)
    books = sv.Codebooks.load(D / args.books)
    R = len(books.wkv)
    print(f"[proxy] held-out users {users}; WKV heads {A.shape[0]:,}; shift t {S['t'].shape[0]:,} c {S['c'].shape[0]:,}; "
          f"books rank {R}, stages {[len(s) for s in books.wkv]}, shift { {k: len(v[0]) for k, v in books.shift.items()} }",
          flush=True)
    rep = {"held_out_users": users, "wkv_heads": int(A.shape[0])}
    _, _, sigma, valid = sv.factorize(A, sv.K)
    energy = float((A.double() ** 2).sum())
    kept = {r: float((sigma[:, :r] ** 2 * valid[:, :r]).sum() / energy) for r in range(1, 9)}
    rep["rank_energy_kept"] = kept
    print("[proxy] rank-r energy kept: " + ", ".join(f"r{r} {v:.4%}" for r, v in kept.items()), flush=True)
    rows = []
    norm_bits = sorted({b for (j, b) in books.wkv_norm if j == 0})
    # exact-direction references (truncation + norm quantization only)
    for r in range(1, R + 1):
        for nb, zero in [(None, False)] + [(b, z) for b in norm_bits if b >= 6 for z in (False,)]:
            rec = sv.StreamRecipe(rank=r, stages=(0,) * r, norm_bits=nb, shift_m=0)
            # stages 0 = exact direction; norm quantized per nb
            e = wkv_err(A, rec, books)
            rows.append({"kind": "wkv", "rank": r, "stages": [0] * r, "norm_bits": nb, "norm_zero": zero,
                         "bits_head": None, "rel_err": e})
            print(f"[proxy] wkv rank {r} exact directions, norm bits {nb} zero {zero}: rel err {e:.3e}", flush=True)
    # candidate allocations (coded directions, 10-bit norms with the zero code)
    for stages in CANDIDATES:
        for zero in (False,):
            rec = sv.StreamRecipe(rank=len(stages), stages=tuple(stages), norm_bits=10, shift_m=0)
            e = wkv_err(A, rec, books)
            rows.append({"kind": "wkv-alloc", "stages": list(stages), "norm_bits": 10, "norm_zero": zero,
                         "bits_head": rec.wkv_bits(), "rel_err": e})
            print(f"[proxy] wkv alloc {stages} n10 zero {zero}: {rec.wkv_bits()} bits/head, rel err {e:.3e}", flush=True)
    # stage grids: per component, the error as a function of its own stage count (other components exact)
    for j in range(R):
        for s in range(1, len(books.wkv[j]) + 1, 2 if len(books.wkv[j]) > 12 else 1):
            stages = tuple(s if i == j else 0 for i in range(j + 1))
            rec = sv.StreamRecipe(rank=j + 1, stages=stages, norm_bits=None, shift_m=0)
            e = wkv_err(A, rec, books)
            rows.append({"kind": "wkv-comp", "component": j, "stages": s, "rel_err": e})
        print(f"[proxy] comp {j}: " + ", ".join(f"{r['stages']}:{r['rel_err']:.2e}" for r in rows
                                                 if r["kind"] == "wkv-comp" and r["component"] == j), flush=True)
    for role in ("t", "c"):
        for (rl, m), pos in sorted(books.shift.items()):
            if rl != role:
                continue
            for st in range(1, len(pos[0]) + 1):
                for nb in (None, 8, 10):
                    if nb is not None and (role, nb) not in books.shift_norm:
                        continue
                    rec = sv.StreamRecipe(rank=0, stages=(), shift_m=m, shift_stages=st, shift_norm_bits=nb)
                    e = shift_err(S[role], role, rec, books)
                    rows.append({"kind": "shift", "role": role, "m": m, "stages": st, "norm_bits": nb,
                                 "bits": m * 12 * st + (nb or 0), "rel_err": e})
            print(f"[proxy] shift {role} m{m}: " + ", ".join(
                f"x{r['stages']}/n{r['norm_bits']}:{r['rel_err']:.2e}" for r in rows
                if r["kind"] == "shift" and r["role"] == role and r["m"] == m), flush=True)
    rep["rows"] = rows
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"[proxy] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
