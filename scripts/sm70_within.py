"""SM70 within-user probe: does a candidate input carry information a model leaves on the table for *this* user?

Complement to sm70_probe.py (pooled, cross-user). For every user, the benchmark test reviews are split in time: the
first half trains a per-user logistic recalibration, the second half scores it.
  baseline : [1, z]                       per-user Platt scaling of z = logit(prediction)
  candidate: [1, z, bins(x)]              plus the candidate's bins (quintiles of the user's training half, own bin
                                          for missing values) shifting the logit
Ridge toward the identity map (intercept 0, slope 1, bin effects 0). Delta = candidate - baseline LogLoss on the second
halves, review-weighted over users, 2,000-resample user bootstrap. A feature can only win here if its relation to the
residual is stable inside a user's history, which a recurrent model with the input could also learn.
The curve head uses the "@c" (as of the card's previous review) version of every time-varying feature.

usage: sm70_within.py --features artifacts/small-model/sm70/features.parquet --model sm56 --out .../within-sm56.json
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sm70_features import FEATURES  # noqa: E402
from scripts.sm70_probe import bootstrap, encode, logit, nll  # noqa: E402

RIDGE = 20.0
NBINS = 5


def fit(X, y, w0, ridge, iters=30):
    w = w0.copy()
    reg = np.full(X.shape[1], ridge)
    reg[0] = 1.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (p - y) + reg * (w - w0)
        h = (X * (p * (1 - p))[:, None]).T @ X + np.diag(reg)
        step = np.linalg.solve(h, g)
        w -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    return w


def design(z, encoded, linear=None):
    cols = [np.ones_like(z), z]
    for b, k in encoded:
        for j in range(1, k):
            cols.append((b == j).astype(float))
    if linear is not None:
        cols.append(linear)
    return np.column_stack(cols)


def fit_offset(X, y, z, ridge, iters=30):
    """logit = z + X w (z kept as a fixed offset: no per-user recalibration of the model itself)."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(z + X @ w)))
        g = X.T @ (p - y) + ridge * w
        h = (X * (p * (1 - p))[:, None]).T @ X + np.diag(np.full(X.shape[1], ridge))
        step = np.linalg.solve(h, g)
        w -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    return w


def one_user(args):
    df, user, cands, pcols = args
    df = df.sort_values("review_th").reset_index(drop=True)
    if "f_a" in df.columns and df["f_a"].notna().all():
        df["ctl_fork"] = logit(df["f_i"].to_numpy(np.float64))
        df["ctl_fork@c"] = logit(df["f_a"].to_numpy(np.float64))
    n = len(df)
    h = n // 2
    tr, te = np.arange(h), np.arange(h, n)
    y = df["y"].to_numpy(np.float64)
    res = {"user": user, "n_eval": n - h, "heads": {}}
    for head, pc in zip(("curve", "instant"), pcols):
        if df[pc].isna().any():
            return None
        z = logit(df[pc].to_numpy(np.float64))
        raw = float(nll(y[te], z[te]).sum())
        w = fit(design(z[tr], []), y[tr], np.array([0.0, 1.0]), RIDGE)
        base = float(nll(y[te], design(z[te], []) @ w).sum())
        out = {"raw": raw, "base": base, "cands": {}, "offset": {}}
        for cname, feats in cands.items():
            if cname == "control:big-model-prediction" and "ctl_fork" not in df.columns:
                continue
            etr, ete = [], []
            lin_tr = lin_te = None
            if cname == "control:big-model-prediction":
                gap = df[("ctl_fork@c" if head == "curve" else "ctl_fork")].to_numpy(np.float64) - z
                lin_tr, lin_te = gap[tr], gap[te]
                feats = []
            for name in feats:
                col = name + "@c" if head == "curve" and (name + "@c") in df.columns else name
                x = df[col].to_numpy(np.float64)
                btr, bte, k = encode(x[tr], x[te], max_cat=6, nbins=NBINS)
                etr.append((btr, k))
                ete.append((bte, k))
            Xtr, Xte = design(z[tr], etr, lin_tr), design(z[te], ete, lin_te)
            w0 = np.zeros(Xtr.shape[1])
            w0[1] = 1.0
            w = fit(Xtr, y[tr], w0, RIDGE)
            out["cands"][cname] = float(nll(y[te], Xte @ w).sum())
            # offset form: only the feature terms are fitted on top of the unchanged model logit
            if Xtr.shape[1] > 2:
                wo = fit_offset(Xtr[:, 2:], y[tr], z[tr], RIDGE)
                out["offset"][cname] = float(nll(y[te], z[te] + Xte[:, 2:] @ wo).sum())
            else:   # the feature is constant on this user's training half: nothing to fit
                out["offset"][cname] = raw
        res["heads"][head] = out
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="sm56", choices=("sm56", "fork"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    pcols = {"sm56": ("p_a", "p_i"), "fork": ("f_a", "f_i")}[a.model]
    data = pd.read_parquet(a.features)
    groups = {}
    for name, (g, _) in FEATURES.items():
        groups.setdefault(g, []).append(name)
    cands = {name: [name] for name in FEATURES}
    cands.update({f"group:{g}": names for g, names in groups.items() if len(names) <= 8})
    if a.model == "sm56":
        cands["control:big-model-prediction"] = ["ctl_fork"]
    if a.only:
        cands = {k: v for k, v in cands.items() if k in a.only.split(",")}
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        jobs = ((g, int(u), cands, pcols) for u, g in data.groupby("user_id", sort=True))
        rows = [r for r in ex.map(one_user, jobs, chunksize=1) if r]
    n_u = np.array([r["n_eval"] for r in rows], dtype=float)
    result = {"model": a.model, "users": len(rows), "eval_reviews": int(n_u.sum()), "ridge": RIDGE, "bins": NBINS,
              "split": "per user: first half of the test reviews trains, second half scores", "heads": {}}
    for head in ("curve", "instant"):
        base = np.array([r["heads"][head]["base"] for r in rows])
        raw = np.array([r["heads"][head]["raw"] for r in rows])
        H = {"raw_logloss": float(raw.sum() / n_u.sum()), "user_platt_logloss": float(base.sum() / n_u.sum()),
             "user_platt_minus_raw": float((base - raw).sum() / n_u.sum()),
             "user_platt_minus_raw_ci": bootstrap(base - raw, n_u), "cands": {}}
        H["offset"] = {}
        for cname in cands:
            vals = [r["heads"][head]["cands"].get(cname) for r in rows]
            if any(v is None for v in vals):
                continue
            du = np.array(vals) - base
            H["cands"][cname] = {"delta": float(du.sum() / n_u.sum()), "ci": bootstrap(du, n_u),
                                 "rel_pct": 100 * float(du.sum() / n_u.sum()) / H["user_platt_logloss"],
                                 "users_better": float(np.mean(du < 0)), "features": cands[cname]}
            do = np.array([r["heads"][head]["offset"][cname] for r in rows]) - raw
            H["offset"][cname] = {"delta": float(do.sum() / n_u.sum()), "ci": bootstrap(do, n_u),
                                  "users_better": float(np.mean(do < 0))}
        result["heads"][head] = H
        print(f"[sm70 within] {a.model} {head}: raw {H['raw_logloss']:.5f} user-platt {H['user_platt_logloss']:.5f}")
        for cname, c in sorted(H["offset"].items(), key=lambda kv: kv[1]["delta"])[:12]:
            q = H["cands"][cname]
            print(f"  {cname:30s} offset {c['delta']:+.6f} [{c['ci'][0]:+.6f}, {c['ci'][1]:+.6f}]  "
                  f"with-platt {q['delta']:+.6f}")
    result["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(result, indent=1))
    print(f"[sm70 within] done -> {a.out} ({result['seconds']}s)")


if __name__ == "__main__":
    main()
