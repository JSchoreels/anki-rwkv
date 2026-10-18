"""Readout for the evaluation pools: review-weighted benchmark LogLoss per head (ahead = curve, instant) over the pool
users that every label has, delta vs the first label, delta % vs fork/w8+s8 (if present), and a paired user-bootstrap
95% CI (2000 resamples, seed 0) of each label minus the first.

usage: eval_pools_report.py --pool 732 --files heldout.jsonl heldout-pools.jsonl ... --labels A/w8+s8 B/w8+s8 ...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HEADS = ("ahead", "instant")
FORK = "fork/w8+s8"


def load_rows(files, labels):
    rows = {label: {} for label in labels}
    for f in files:
        for line in open(f, encoding="utf-8"):
            if not any(label in line for label in labels):
                continue
            r = json.loads(line)
            b = r.get("benchmark")
            if r.get("label") in rows and b and all(h in b and b[h].get("size") for h in HEADS):
                rows[r["label"]][r["user_id"]] = {h: (b[h]["LogLoss"], b[h]["size"]) for h in HEADS}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, required=True, choices=(464, 732, 2225))
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--resamples", type=int, default=2000)
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    from scripts.eval_pools import pool_users
    pool = pool_users(args.pool)
    rows = load_rows(args.files, args.labels)
    common = [u for u in pool if all(u in rows[label] for label in args.labels)]
    if not common:
        raise SystemExit("no pool user has rows for every label")
    print(f"pool {args.pool}: {len(pool)} users; scored by every label: {len(common)}")
    for label in args.labels:
        print(f"  {label}: missing {sum(u not in rows[label] for u in pool)} pool users")
    ref = args.labels[0]
    stats = {}
    for label in args.labels:
        for h in HEADS:
            ll = np.array([rows[label][u][h][0] for u in common])
            w = np.array([rows[label][u][h][1] for u in common], dtype=float)
            stats[label, h] = (ll, w)
    reviews = {h: int(stats[ref, h][1].sum()) for h in HEADS}
    print(f"review-weighted over the common users: ahead {reviews['ahead']:,} reviews, instant {reviews['instant']:,}\n")
    wm = {k: float((ll * w).sum() / w.sum()) for k, (ll, w) in stats.items()}
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(common), size=(args.resamples, len(common)))
    head = f"{'label':44s} {'head':8s} {'LogLoss':>9s} {'d vs first':>11s} {'d% vs fork':>10s}   paired 95% CI of (label - first)"
    print(head)
    for label in args.labels:
        for h in HEADS:
            ll, w = stats[label, h]
            line = f"{label:44s} {h:8s} {wm[label, h]:9.5f} "
            if label == ref:
                line += f"{'-':>11s} "
            else:
                line += f"{wm[label, h] - wm[ref, h]:+11.5f} "
            if FORK in args.labels:
                line += f"{100 * (wm[label, h] / wm[FORK, h] - 1):+9.2f}% "
            else:
                line += f"{'n/a':>10s} "
            if label != ref:
                rll, rw = stats[ref, h]
                boot = ((ll * w)[idx].sum(1) / w[idx].sum(1)) - ((rll * rw)[idx].sum(1) / rw[idx].sum(1))
                lo, hi = np.percentile(boot, [2.5, 97.5])
                line += f"  [{lo:+.5f}, {hi:+.5f}]"
            print(line)


if __name__ == "__main__":
    main()
