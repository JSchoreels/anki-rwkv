#!/usr/bin/env python
"""Small-model calibration diagnostic (docs/SMALL_MODEL.md section 5.3), CPU only, from per-review dumps.

Input: a folder written by `small_model_heldout.py --dump DIR` (one npz per user and label: review_th, bin, y,
p_ahead, p_instant on the benchmark test reviews), for a reference and a candidate label, optionally also scored
with `--window 128` (labels with the suffix @w128). Users present for every requested label are used.

Per head it prints, review-weighted over users as the benchmark does:
  - RMSE(bins) and LogLoss as scored;
  - mean bias (mean p - mean y), overall and by position in the user's history (review_th buckets);
  - reliability by predicted-probability quantile (pooled);
  - RMSE(bins) after a global Platt layer (a * logit + b) fitted on half the users and applied to the other half
    (2-fold by user), and after an oracle per-user mean shift (upper bound of any per-user constant correction).

  .venv/Scripts/python.exe scripts/small_model_calibration.py --dump DIR [--reference fork/w8+s8]
      [--candidate scale-w64kd-dec/w8+s8] [--window 128] [--out artifacts/small-model/calibration.json]
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EDGES = (0, 1000, 5000, 20000, None)


def rmse_bins(p, y, b):
    _, inv, cnt = np.unique(b, return_inverse=True, return_counts=True)
    return float(np.sqrt((((np.bincount(inv, p) - np.bincount(inv, y)) / cnt) ** 2 * cnt).sum() / cnt.sum()))


def log_loss(p, y):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def platt(z, y):
    a, b = 1.0, 0.0
    for _ in range(50):  # Newton steps on the logistic loss of a * z + b
        p = 1 / (1 + np.exp(-(a * z + b)))
        w = p * (1 - p)
        g = np.array([((p - y) * z).sum(), (p - y).sum()])
        h = np.array([[(w * z * z).sum(), (w * z).sum()], [(w * z).sum(), w.sum()]]) + 1e-9 * np.eye(2)
        a, b = np.array([a, b]) - np.linalg.solve(h, g)
    return float(a), float(b)


def weighted(per_user, data, users, key, head, transform=None):
    rb, ll, n = [], [], []
    for u in users:
        d = data[u][key]
        p = transform(u, d) if transform else d[head]
        rb.append(rmse_bins(p, d["y"], d["bin"]))
        ll.append(log_loss(p, d["y"]))
        n.append(len(d["y"]))
    n = np.array(n, dtype=float)
    return {"RMSE(bins)": float((np.array(rb) * n).sum() / n.sum()), "LogLoss": float((np.array(ll) * n).sum() / n.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--reference", default="fork/w8+s8")
    parser.add_argument("--candidate", default="scale-w64kd-dec/w8+s8")
    parser.add_argument("--window", type=int, default=0, help="also compare the @wW (training-protocol) dumps")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    keys = {"reference": args.reference, "candidate": args.candidate}
    if args.window:
        keys.update({f"reference@w{args.window}": f"{args.reference}@w{args.window}",
                     f"candidate@w{args.window}": f"{args.candidate}@w{args.window}"})
    files = {k: v.replace("/", "_") for k, v in keys.items()}
    users = sorted({int(os.path.basename(p).split("_")[0]) for p in glob.glob(f"{args.dump}/*.npz")})
    users = [u for u in users if all(os.path.exists(f"{args.dump}/{u}_{f}.npz") for f in files.values())]
    data = {u: {k: dict(np.load(f"{args.dump}/{u}_{f}.npz")) for k, f in files.items()} for u in users}
    total = sum(len(data[u]["reference"]["y"]) for u in users)
    report = {"labels": keys, "users": users, "benchmark_reviews": total, "heads": {}}
    print(f"{len(users)} users, {total:,} benchmark reviews; labels {keys}")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(users))
    fold = {u: int(i in set(perm[1::2])) for i, u in enumerate(users)}
    for head in ("p_ahead", "p_instant"):
        out = report["heads"][head] = {}
        print(f"== {head}")
        for key in keys:
            row = out[key] = weighted(None, data, users, key, head)
            p = np.concatenate([data[u][key][head] for u in users])
            y = np.concatenate([data[u][key]["y"] for u in users])
            row["bias"] = float(p.mean() - y.mean())
            row["bias_by_position"], row["log_loss_by_position"] = {}, {}
            for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                pick = [(data[u][key]["review_th"] >= lo) & (data[u][key]["review_th"] < (hi or 1 << 62)) for u in users]
                pp = np.concatenate([data[u][key][head][m] for u, m in zip(users, pick)])
                yy = np.concatenate([data[u][key]["y"][m] for u, m in zip(users, pick)])
                name = f"[{lo},{hi or 'inf'})"
                row["bias_by_position"][name] = float(pp.mean() - yy.mean()) if len(yy) else None
                row["log_loss_by_position"][name] = log_loss(pp, yy) if len(yy) else None
                row.setdefault("reviews_by_position", {})[name] = int(len(yy))
            q = np.quantile(p, [0, .1, .3, .5, .7, .9, 1.0])
            idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, 5)
            row["reliability"] = [[float(p[idx == k].mean()), float(y[idx == k].mean())] for k in range(6)]
            if "@" not in key:
                params = {}
                for f in (0, 1):
                    train = [u for u in users if fold[u] != f]
                    params[f] = platt(np.concatenate([logit(data[u][key][head]) for u in train]),
                                      np.concatenate([data[u][key]["y"] for u in train]))
                row["platt_params"] = params
                row["platt_2fold"] = weighted(None, data, users, key, head, lambda u, d: 1 / (
                    1 + np.exp(-(params[fold[u]][0] * logit(d[head]) + params[fold[u]][1]))))
                row["oracle_user_shift"] = weighted(None, data, users, key, head, lambda u, d: np.clip(
                    d[head] - (d[head].mean() - d["y"].mean()), 1e-5, 1 - 1e-5))
            line = f"  {key:22s} RMSE(bins) {row['RMSE(bins)']:.4f} LL {row['LogLoss']:.4f} bias {row['bias']:+.4f} |"
            line += " ".join(f" {k} {v:+.4f}" for k, v in row["bias_by_position"].items() if v is not None)
            if "platt_2fold" in row:
                line += (f" | Platt RMSE(bins) {row['platt_2fold']['RMSE(bins)']:.4f}, per-user shift "
                         f"{row['oracle_user_shift']['RMSE(bins)']:.4f}")
            print(line)
    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(out)


if __name__ == "__main__":
    main()
