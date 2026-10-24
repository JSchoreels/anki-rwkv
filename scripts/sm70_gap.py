"""SM70: where does the small model (SM56) lose to the big model? The per-review LogLoss gap SM56 - big model, broken
down by quintiles of every candidate feature (curve head: the "@c" versions). A feature whose bins concentrate the gap
(gap share far from review share) points at situations the small model handles worse.

usage: sm70_gap.py --features artifacts/small-model/sm70/features.parquet --out artifacts/small-model/sm70/gap.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sm70_features import FEATURES  # noqa: E402
from scripts.sm70_probe import logit, nll  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.read_parquet(a.features)
    df = df[df["f_a"].notna() & df["f_i"].notna()].reset_index(drop=True)
    y = df["y"].to_numpy(np.float64)
    res = {"users": int(df["user_id"].nunique()), "reviews": int(len(df)), "heads": {}}
    for head, pc, fc in (("curve", "p_a", "f_a"), ("instant", "p_i", "f_i")):
        gap = nll(y, logit(df[pc].to_numpy(np.float64))) - nll(y, logit(df[fc].to_numpy(np.float64)))
        total = gap.sum()
        H = {"gap_mean": float(gap.mean()), "features": {}}
        for name in [n for n in FEATURES if n in df.columns]:
            col = name + "@c" if head == "curve" and (name + "@c") in df.columns else name
            x = df[col].to_numpy(np.float64)
            ok = ~np.isnan(x)
            if ok.sum() == 0:
                continue
            edges = np.unique(np.quantile(x[ok], [0.2, 0.4, 0.6, 0.8]))
            b = np.searchsorted(edges, x, side="right")
            b[~ok] = len(edges) + 1
            rows = []
            for j in np.unique(b):
                m = b == j
                rows.append({"bin": int(j), "lo": float(np.nanmin(x[m])) if ok[m].any() else None,
                             "hi": float(np.nanmax(x[m])) if ok[m].any() else None,
                             "review_share": float(m.mean()), "gap_share": float(gap[m].sum() / total),
                             "gap_per_review": float(gap[m].mean())})
            conc = max(abs(r["gap_share"] - r["review_share"]) for r in rows)
            H["features"][name] = {"concentration": conc, "bins": rows}
        res["heads"][head] = H
        top = sorted(H["features"].items(), key=lambda kv: -kv[1]["concentration"])[:10]
        print(f"[sm70 gap] {head}: mean gap {H['gap_mean']:+.5f} (SM56 - big model)")
        for name, v in top:
            worst = max(v["bins"], key=lambda r: r["gap_share"] - r["review_share"])
            print(f"  {name:14s} conc {v['concentration']:.3f}; worst bin [{worst['lo']}, {worst['hi']}] "
                  f"reviews {worst['review_share']:.2f} gap {worst['gap_share']:.2f}")
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
