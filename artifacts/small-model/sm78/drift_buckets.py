#!/usr/bin/env python
"""int4 drift check (SM78): LogLoss of the leak-free small model (SM72s) with int8 card/note state (w8+s8) vs int4 card
+ int8 note state (w8+s4c8n), bucketed by the card's review index, from per-review CPU predictions (--dump).
usage: drift_buckets.py DUMP_DIR [DUMP_DIR...] --out JSON   (CPU only, read only on the worktree/dataset)"""
import json, sys, glob, re
from pathlib import Path
import numpy as np, pandas as pd

WT = Path("C:/Users/jonat/workspace/wt/sm73"); sys.path.insert(0, str(WT))
from scripts import compress_common as cc  # noqa: E402
DATA = cc.data_path()
EDGES = [1, 2, 4, 8, 16, 32, 64, 10**9]
NAMES = ["1", "2-3", "4-7", "8-15", "16-31", "32-63", "64+"]
A, B = "w8+s8", "w8+s4c8n"

def ll(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

def main():
    args = sys.argv[1:]; out = args[args.index("--out") + 1]; dirs = [a for a in args if a != "--out" and a != out]
    users = {}
    for d in dirs:
        for f in glob.glob(f"{d}/*_{A}.npz"):
            users[int(Path(f).name.split("_")[0])] = d
    res = {}
    check = []
    for u, d in sorted(users.items()):
        try:
            za, zb = (np.load(f"{d}/{u}_s_{v}.npz") for v in (A, B))
        except OSError:
            continue
        cid = pd.read_parquet(DATA / "revlogs" / f"user_id={u}", columns=["card_id"])["card_id"]
        idx = cid.groupby(cid).cumcount().to_numpy() + 1
        ths = za["review_th"]; assert np.array_equal(ths, zb["review_th"])
        bk = np.digitize(idx[ths - 1], EDGES) - 1
        y = za["y"]
        row = {}
        for head, key in (("curve", "p_ahead"), ("instant", "p_instant")):
            la, lb = ll(y, za[key]), ll(y, zb[key])
            row[head] = {"n": np.bincount(bk, minlength=7), "a": np.bincount(bk, la, 7), "b": np.bincount(bk, lb, 7)}
        res[u] = row
        check.append((u, row["curve"]["a"].sum() / row["curve"]["n"].sum()))
    U = sorted(res); rng = np.random.default_rng(0); boot = rng.integers(0, len(U), size=(2000, len(U)))
    summary = {"users": len(U), "buckets": NAMES, "heads": {}}
    for head in ("curve", "instant"):
        n = np.array([res[u][head]["n"] for u in U], float); a = np.array([res[u][head]["a"] for u in U]); b = np.array([res[u][head]["b"] for u in U])
        rows = []
        for k, name in enumerate(NAMES + ["all"]):
            sl = slice(None) if name == "all" else k
            nn, aa, bb = (x[:, sl] if name != "all" else x.sum(1) for x in (n, a, b))
            if name != "all": nn, aa, bb = n[:, k], a[:, k], b[:, k]
            N = nn.sum()
            if N == 0: continue
            gap = (bb.sum() - aa.sum()) / N
            bg = (bb[boot].sum(1) - aa[boot].sum(1)) / np.maximum(nn[boot].sum(1), 1)
            lo, hi = np.percentile(bg, [2.5, 97.5])
            rows.append({"bucket": name, "reviews": int(N), "users": int((nn > 0).sum()), "int8": aa.sum() / N, "int4c8n": bb.sum() / N,
                         "gap": gap, "ci": [lo, hi], "gap_pct": gap / (aa.sum() / N) * 100})
        # slope-type check: gap in 16+ vs 1-15 reviews, paired bootstrap of the difference
        lo_m, hi_m = slice(0, 4), slice(4, 7)
        def g(m, idx_):
            N = n[idx_][:, m].sum(1).sum(); return (b[idx_][:, m].sum() - a[idx_][:, m].sum()) / max(N, 1)
        g_lo = g(lo_m, slice(None)); g_hi = g(hi_m, slice(None))
        diffs = []
        for bi in boot:
            nn, aa, bb = n[bi], a[bi], b[bi]
            gl = (bb[:, lo_m].sum() - aa[:, lo_m].sum()) / max(nn[:, lo_m].sum(), 1)
            gh = (bb[:, hi_m].sum() - aa[:, hi_m].sum()) / max(nn[:, hi_m].sum(), 1)
            diffs.append(gh - gl)
        summary["heads"][head] = {"rows": rows, "gap_16plus_minus_1to15": g_hi - g_lo, "gap_16plus_minus_1to15_ci": list(np.percentile(diffs, [2.5, 97.5]))}
    Path(out).write_text(json.dumps(summary, indent=1))
    print(json.dumps({h: [(r["bucket"], r["reviews"], round(r["gap"] * 1e5, 2), [round(x * 1e5, 2) for x in r["ci"]]) for r in v["rows"]] for h, v in summary["heads"].items()}, indent=0))
    print({h: (v["gap_16plus_minus_1to15"], v["gap_16plus_minus_1to15_ci"]) for h, v in summary["heads"].items()})
    print("check mean curve int8 loss (first users):", check[:3])
main()
