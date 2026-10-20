"""Compare two pick logs (vqdiag.py torch mode): first divergent call/row, input difference, codeword gaps.
usage: python vqcmp.py WT CASE A.npz B.npz"""
import sys
import json
WT, CASE, FA, FB = sys.argv[1:5]
sys.path.insert(0, WT)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scripts import check_state_vq_score_gpu as chk  # noqa: E402
from scripts import state_vq  # noqa: E402

import os
if os.environ.get("CASES_JSON"):
    chk.CASES = [tuple(c) for c in json.load(open(os.environ["CASES_JSON"]))]
case = [c for c in chk.CASES if c[0] == CASE][0]
books = state_vq.Codebooks.load(case[2])
codecs = state_vq.score_codecs(case[3], books, "cpu", torch.float64, kernel=False)
cands = []
for s, c in codecs.items():
    for j, st in enumerate(c.wkv_books):
        if st:
            cands.append((f"wkv{s}/comp{j}", st))
    for role, pos in c.shift_books.items():
        for p, st in enumerate(pos):
            cands.append((f"shift{s}/{role}/pos{p}", st))


def load(f):
    z = np.load(f)
    calls = {}
    for key in z.files:
        i, kind, *tag, what = key.split("/")
        calls.setdefault(int(i), {"tag": "/".join(tag)})[what] = z[key]
    return [calls[i] for i in sorted(calls)]


def gaps(x, picks, st):
    """float64 per-stage (best, second, gap) of squared distance for one row x [D], following its own picks."""
    acc = np.zeros_like(x, dtype=np.float64)
    out = []
    for s, cb in enumerate(st):
        cb = cb.double().numpy()
        d = ((x[None, :].astype(np.float64) - acc[None, :] - cb) ** 2).sum(-1)
        o = np.argsort(d)
        out.append({"stage": s, "pick": int(picks[s]), "best": int(o[0]), "second": int(o[1]),
                    "d_best": float(d[o[0]]), "gap": float(d[o[1]] - d[o[0]]),
                    "rel_gap": float((d[o[1]] - d[o[0]]) / max(d[o[0]], 1e-30))})
        acc = acc + cb[picks[s]]
    return out


def which_book(x, picks):
    xt = torch.from_numpy(x)
    for name, st in cands:
        if st[0].shape[1] == x.shape[1] and len(st) == picks.shape[1]:
            _, p = state_vq.rvq_picks(xt, st)
            if np.array_equal(p.numpy(), picks):
                return name, st
    return None, None


A, B = load(FA), load(FB)
assert len(A) == len(B), (len(A), len(B))
report = {"case": CASE, "calls": len(A), "first": None}
n_rows = n_diff_rows = 0
first_by_tag = {}
for i, (a, b) in enumerate(zip(A, B)):
    assert a["picks"].shape == b["picks"].shape and a["tag"] == b["tag"]
    n_rows += a["picks"].shape[0]
    rows = np.nonzero((a["picks"] != b["picks"]).any(-1))[0]
    n_diff_rows += len(rows)
    if len(rows) and a["tag"] not in first_by_tag:
        first_by_tag[a["tag"]] = i
    if len(rows) and report["first"] is None:
        r = int(rows[0])
        name, st = which_book(a["x"], a["picks"])
        xa, xb = a["x"][r], b["x"][r]
        report["first"] = {"call": i, "tag": a["tag"], "codebook": name, "rows_in_call": int(a["picks"].shape[0]),
                           "diff_rows_in_call": int(len(rows)), "row": r,
                           "picks_A": a["picks"][r].tolist(), "picks_B": b["picks"][r].tolist(),
                           "max_abs_dx": float(np.abs(xa - xb).max()), "norm_x": float(np.linalg.norm(xa)),
                           "gaps_A": gaps(xa, a["picks"][r], st), "gaps_B": gaps(xb, b["picks"][r], st)}
        # all rows that differ before any divergence propagates: earlier calls agree, so every diff row here is first
        g = []
        for rr in rows:
            ga = gaps(a["x"][rr], a["picks"][rr], st)
            s0 = int(np.nonzero(a["picks"][rr] != b["picks"][rr])[0][0])
            g.append((s0, ga[s0]["gap"], ga[s0]["rel_gap"], float(np.abs(a["x"][rr] - b["x"][rr]).max())))
        report["first"]["all_rows_first_call"] = g
report["rows"] = n_rows
report["diff_rows"] = n_diff_rows
report["first_call_by_tag"] = first_by_tag
# near-tie census on A's inputs before the first divergence: how often is the stage gap tiny?
census = {"rows": 0}
thr = (1e-7, 1e-6, 1e-5, 1e-4)
for t in thr:
    census[f"rel_gap<{t:g}"] = 0
stop = report["first"]["call"] if report["first"] else len(A)
book_of = {}
for i in range(min(stop + 1, len(A))):
    a = A[i]
    key = (a["tag"], a["x"].shape[1], a["picks"].shape[1])
    if key not in book_of:
        book_of[key] = which_book(a["x"][:64], a["picks"][:64])[1]
    st = book_of[key]
    x = torch.from_numpy(a["x"]).double()
    acc = torch.zeros_like(x)
    minrel = torch.full((x.shape[0],), float("inf"), dtype=torch.float64)
    for s, cb in enumerate(st):
        cb = cb.double()
        d = ((x - acc)[:, None, :] - cb[None]).pow(2).sum(-1)
        top = d.topk(2, largest=False).values
        minrel = torch.minimum(minrel, (top[:, 1] - top[:, 0]) / top[:, 0].clamp_min(1e-30))
        acc = acc + cb[torch.from_numpy(a["picks"][:, s]).long()]
    census["rows"] += x.shape[0]
    for t in thr:
        census[f"rel_gap<{t:g}"] += int((minrel < t).sum())
report["near_tie_census_until_first_divergence"] = census
print(json.dumps(report, indent=1, default=str))
