"""Codebook state compression sweep (SM90): model-free proxy for the budget ladder (CPU, minutes).

On the HELD-OUT corpus users of artifacts/small-model/sm90/books (every 10th corpus user, never seen by the k-means),
for the released small model's stored states, separately for card heads (stream 0, 2 layers) and note heads (stream 2):
  - anchors: relative squared error sum ||A - A_hat||^2 / sum ||A||^2 of the int8 and int4 per-row rounding
    (scripts/state_vq.int_row_quant, the scorer's +s8 / +s4 emulation) on the same heads and shift vectors;
  - WKV: rank-r truncation energy; per-component error curves (component j coded with s stages, others exact);
  - norm quantization error at 6 / 8 / 10 bits (exact directions);
  - shift: m4 / m8 x stages x norm bits per role (t = time mixer, c = channel mixer);
  - allocations: for each per-head bit budget, ranks 3..6 and norm bits 8 / 10, a greedy stage allocation (add the
    stage with the largest error drop per bit, from the per-component curves), then the full codec's error.
The codec is scripts/state_vq.StateCodec (eigh float64), so every number is that of the recipe it names.

usage: python -X utf8 artifacts/small-model/sm90/proxy_sm90.py [--books codebooks-sm90.npz] [--heads-budgets 590 295 ...]
writes artifacts/small-model/sm90/proxy-sm90.json
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
HERE = Path(__file__).resolve().parent
D = HERE / "books"
STREAMS = {"card": "0", "note": "2"}


def held_out(every, max_heads):
    files = sorted((D / "corpus").glob("*.npz"))
    files = [f for i, f in enumerate(files) if i % every == every - 1]
    wkv = {s: [] for s in STREAMS}
    shift = {(s, r): [] for s in STREAMS for r in ("t", "c")}
    for f in files:
        z = np.load(f)
        for key in z.files:
            parts = key.split("/")
            stream = {v: k for k, v in STREAMS.items()}[parts[1]]
            if parts[0] == "wkv":
                wkv[stream].append(z[key].reshape(-1, sv.K, sv.K))
            else:
                shift[(stream, parts[3])].append(z[key])
    g = torch.Generator().manual_seed(7)
    A = {}
    for s, v in wkv.items():
        a = torch.from_numpy(np.concatenate(v)).float()
        A[s] = a[torch.randperm(a.shape[0], generator=g)[:max_heads]]
    S = {k: torch.from_numpy(np.concatenate(v)).float() for k, v in shift.items()}
    return [f.stem for f in files], A, S


def rel(x, y):
    x, y = x.double(), y.double()
    return float(((x - y) ** 2).sum() / (x ** 2).sum())


def wkv_err(A, rec, books):
    return rel(A, sv.StateCodec(rec, books).wkv(A[:, None])[:, 0])


def shift_err(X, role, rec, books):
    return rel(X, sv.StateCodec(rec, books).shift(X, role))


def greedy(curves, rank, stage_bits, budget_stage_bits, cap):
    """Stages per component for `rank` components: start at 1 stage each, add the stage with the largest drop of
    (sum of component errors) per bit until the budget is spent. curves[j][s] = error with s stages (s >= 1)."""
    st = [1] * rank
    left = budget_stage_bits - rank * stage_bits
    if left < 0:
        return None
    while left >= stage_bits:
        gains = [(curves[j][st[j]] - curves[j][st[j] + 1]) if st[j] + 1 <= cap[j] else -1 for j in range(rank)]
        j = int(np.argmax(gains))
        if gains[j] <= 0:
            break
        st[j] += 1
        left -= stage_bits
    return tuple(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default="codebooks-sm90.npz")
    ap.add_argument("--holdout-every", type=int, default=10)
    ap.add_argument("--max-heads", type=int, default=8000)
    ap.add_argument("--head-budgets", type=int, nargs="+", default=[590, 560, 300, 280, 190, 175, 145, 135, 100, 95])
    ap.add_argument("--out", default=str(HERE / "proxy-sm90.json"))
    ap.add_argument("--parts", default="wkv,shift", help="wkv and/or shift sections (anchors always)")
    args = ap.parse_args()
    users, A, S = held_out(args.holdout_every, args.max_heads)
    books = sv.Codebooks.load(D / args.books)
    R = len(books.wkv)
    caps = [len(b) for b in books.wkv]
    print(f"[proxy] held-out users {users}; heads { {s: a.shape[0] for s, a in A.items()} }; shift "
          f"{ {f'{k[0]}/{k[1]}': v.shape[0] for k, v in S.items()} }; books stages {caps}, shift "
          f"{ {f'{k[0]}/m{k[1]}': len(v[0]) for k, v in books.shift.items()} }", flush=True)
    rep = {"held_out_users": users, "books": args.books, "books_sha256": sv.file_sha256(D / args.books), "rows": []}
    rows = rep["rows"]
    # anchors: int8 / int4 row rounding (as +s8 / +s4: per row of the 16 x 16 head; per 64-float shift vector)
    for bits in (8, 4):
        q = sv.int_row_quant(bits)
        for s, a in A.items():
            rows.append({"kind": "anchor-wkv", "stream": s, "bits": bits, "rel_err": rel(a, q(a))})
        for (s, role), x in S.items():
            rows.append({"kind": "anchor-shift", "stream": s, "role": role, "bits": bits, "rel_err": rel(x, q(x))})
    for r in rows:
        print(f"[proxy] anchor {r}", flush=True)
    for s, a in (A.items() if "wkv" in args.parts else ()):
        _, _, sigma, valid = sv.factorize(a, sv.K)
        energy = float((a.double() ** 2).sum())
        kept = {r: float((sigma[:, :r] ** 2 * valid[:, :r]).sum() / energy) for r in range(1, 9)}
        rows.append({"kind": "rank-energy", "stream": s, "kept": kept})
        print(f"[proxy] {s} rank-r energy kept: " + ", ".join(f"r{r} {v:.4%}" for r, v in kept.items()), flush=True)
    # per-component curves (component j with s stages, components < j exact, norms exact); card + note heads pooled
    pooled = torch.cat([A["card"], A["note"]])
    curves = []
    for j in (range(R) if "wkv" in args.parts else ()):
        cj = {}
        for s in range(1, caps[j] + 1):
            stages = tuple(s if i == j else 0 for i in range(j + 1))
            cj[s] = wkv_err(pooled, sv.StreamRecipe(rank=j + 1, stages=stages, norm_bits=None, shift_m=0), books)
        base = wkv_err(pooled, sv.StreamRecipe(rank=j + 1, stages=(0,) * (j + 1), norm_bits=None, shift_m=0), books)
        curves.append({s: e - base for s, e in cj.items()})   # extra error of coding component j
        rows.append({"kind": "wkv-comp", "component": j, "exact_rank_err": base, "extra": curves[-1]})
        print(f"[proxy] comp {j} (rank-{j + 1} exact {base:.2e}): " + ", ".join(f"{s}:{e:.2e}" for s, e in curves[-1].items()),
              flush=True)
    for r in (range(2, R + 1) if "wkv" in args.parts else ()):
        for nb in (6, 8, 10):
            e = {s: wkv_err(a, sv.StreamRecipe(rank=r, stages=(0,) * r, norm_bits=nb, shift_m=0), books) for s, a in A.items()}
            rows.append({"kind": "wkv-norm", "rank": r, "norm_bits": nb, "rel_err": e})
        print(f"[proxy] rank {r} exact directions, norms 6/8/10 bits: " + "; ".join(
            f"{r_['norm_bits']}: card {r_['rel_err']['card']:.2e} note {r_['rel_err']['note']:.2e}"
            for r_ in rows if r_["kind"] == "wkv-norm" and r_["rank"] == r), flush=True)
    for hb in (args.head_budgets if "wkv" in args.parts else ()):
        for r in range(2, R + 1):
            for nb in (8, 10):
                st = greedy(curves, r, 10, hb - r * nb, caps)
                if st is None:
                    continue
                rec = sv.StreamRecipe(rank=r, stages=st, norm_bits=nb, shift_m=0)
                e = {s: wkv_err(a, rec, books) for s, a in A.items()}
                rows.append({"kind": "wkv-alloc", "head_budget": hb, "rank": r, "norm_bits": nb, "stages": list(st),
                             "bits_head": rec.wkv_bits(), "rel_err": e})
                print(f"[proxy] head budget {hb}: rank {r} n{nb} stages {st} = {rec.wkv_bits()} bits: card "
                      f"{e['card']:.3e} note {e['note']:.3e}", flush=True)
    for (s, role), x in (S.items() if "shift" in args.parts else ()):
        for (rl, m), pos in sorted(books.shift.items()):
            if rl != role:
                continue
            for st in range(1, len(pos[0]) + 1):
                for nb in (8, 10):
                    if (role, nb) not in books.shift_norm:
                        continue
                    rec = sv.StreamRecipe(rank=0, stages=(), shift_m=m, shift_stages=st, shift_norm_bits=nb)
                    rows.append({"kind": "shift", "stream": s, "role": role, "m": m, "stages": st, "norm_bits": nb,
                                 "bits": rec.shift_vector_bits(), "rel_err": shift_err(x, role, rec, books)})
            print(f"[proxy] shift {s}/{role} m{m}: " + ", ".join(
                f"x{r_['stages']}/n{r_['norm_bits']}={r_['bits']}b:{r_['rel_err']:.2e}" for r_ in rows
                if r_["kind"] == "shift" and r_.get("stream") == s and r_["role"] == role and r_["m"] == m), flush=True)
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(f"[proxy] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
