"""Codebook state compression sweep (SM90) CPU proof (b): the codec on the held-out corpus users (CPU, ~1 min).

  1. encoding is deterministic: StateCodec.wkv / .shift twice on the same heads -> bitwise equal outputs;
  2. decode(encode(x)) = the explicit decode of the picks: factorize, residual picks per component (rvq_picks), norms
     on the grid, sum of q(n)^2 u_hat v_hat^T -> equals StateCodec.wkv bitwise (same float64 arithmetic order);
  3. the error is as expected: component 0's held-out mean squared residual per stage vs the k-means in-sample residual
     printed by the fit (logs/fit-c0.log); every extra stage lowers the held-out error (per-component curves).
writes artifacts/small-model/sm90/proof/proof-codec.json; exit 1 on a failure.
"""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import torch  # noqa: E402

from proxy_sm90 import held_out, D, HERE  # noqa: E402
from scripts import state_vq as sv  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))


def main():
    books = sv.Codebooks.load(D / "codebooks-sm90.npz")
    _, A, S = held_out(10, 4000)
    rep, fails = {"codebooks_sha256": sv.file_sha256(D / "codebooks-sm90.npz")}, []
    # 1. determinism
    for name in ("vqC5902", "vqN3964", "vqC1470"):
        rec = [r for r in sv.recipe(name) if not isinstance(r, sv.IntStream)][0]
        stream = "card" if name.startswith("vqC") else "note"
        c = sv.StateCodec(rec, books)
        a1, a2 = c.wkv(A[stream][:, None]), c.wkv(A[stream][:, None])
        x = S[(stream, "t")]
        s1, s2 = c.shift(x, "t"), c.shift(x, "t")
        same = bool(torch.equal(a1, a2) and torch.equal(s1, s2))
        rep[f"deterministic_{name}"] = same
        if not same:
            fails.append(f"{name} not deterministic")
    # 2. explicit decode of the picks
    rec = sv.recipe("vqN3964")[1]
    c = sv.StateCodec(rec, books)
    X = A["note"]
    out = c.wkv(X[:, None])[:, 0]
    u, v, sigma, valid = sv.factorize(X, rec.rank, torch.float64)
    dec = torch.zeros(X.shape[0], sv.K, sv.K, dtype=torch.float64)
    for j in range(rec.rank):
        q = torch.cat([u[:, j], v[:, j]], dim=-1).float()
        acc, picks = sv.rvq_picks(q, c.wkv_books[j], c.wkv_n2[j], c.wkv_chunk)
        rebuilt = sum(c.wkv_books[j][s][picks[:, s]] for s in range(picks.shape[1]))   # decode = sum of codewords
        if not torch.equal(rebuilt, acc):
            fails.append(f"component {j}: sum of picked codewords != rvq accumulation")
        acc = acc.to(torch.float64)
        n = torch.sqrt(torch.where(valid[:, j], sigma[:, j], torch.ones_like(sigma[:, j])))
        nq = c.wkv_nq[j](n.float()).to(torch.float64)
        term = (nq * nq)[:, None, None] * acc[:, :sv.K, None] * acc[:, None, sv.K:]
        dec = dec + torch.where(valid[:, j, None, None], term, torch.zeros_like(term))
    d = float((dec.float() - out).abs().max())
    rep["explicit_decode_max_abs_diff"] = d
    if d != 0.0:
        fails.append(f"explicit decode differs from StateCodec.wkv by {d}")
    # 3. held-out vs in-sample residual of component 0 (unit (u, v): |q|^2 = 2)
    log = (HERE / "logs" / "fit-c0.log").read_text()
    insample = [float(m) for m in re.findall(r"wkv comp 0 stage \d+/\d+: mean squared residual ([0-9.]+)", log)]
    pooled = torch.cat([A["card"], A["note"]])
    u, v, sigma, valid = sv.factorize(pooled, 1, torch.float64)
    q = torch.cat([u[valid[:, 0], 0], v[valid[:, 0], 0]], dim=-1).float()
    held, R = [], q.clone()
    for cb in books.wkv[0]:
        R = R - cb[sv.nearest(R, cb)]
        held.append(float((R ** 2).sum(-1).mean()))
    rep["comp0_residual"] = {"in_sample": insample, "held_out": held,
                             "ratio": [h / i if i else None for h, i in zip(held, insample)]}
    if any(b >= a for a, b in zip(held, held[1:])):
        fails.append("held-out residual does not fall with every stage")
    if any(r is not None and not 0.8 <= r <= 1.6 for r in rep["comp0_residual"]["ratio"]):
        fails.append("held-out residual outside 0.8-1.6 x the in-sample residual")
    rep["fails"] = fails
    rep["passed"] = not fails
    out_path = HERE / "proof" / "proof-codec.json"
    out_path.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "comp0_residual"}, indent=1))
    print("comp0 held-out / in-sample:", [round(r, 3) for r in rep["comp0_residual"]["ratio"]])
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
