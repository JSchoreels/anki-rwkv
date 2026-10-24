"""SM70 residual probe: how much LogLoss a candidate input would remove on top of a model's predictions.

Per head (curve = p_a, instant = p_i; big model: f_a, f_i), 5-fold cross-validation grouped by user. On the training
folds a small logistic recalibration is fitted on z = logit(prediction):
  baseline : [1, z]                                  (Platt scaling)
  candidate: [1, z, bins(x), bins(x) * z] per feature x of the candidate (deciles of the training folds, own bin for
             missing values, so a feature can shift and rescale the logit in any monotone or non-monotone way)
and scored on the held-out users' reviews. Delta = LogLoss(candidate) - LogLoss(baseline), review-weighted over all
test reviews (the SM56 headline aggregation), with a 2,000-resample user bootstrap (95% percentile interval).
The curve head uses the "@c" (as of the card's previous review) version of every time-varying feature.
Also: all features at once with a gradient-boosted tree on [z, features] vs the same tree on z alone (upper bound).

usage: sm70_probe.py --features artifacts/small-model/sm70/features.parquet --out artifacts/small-model/sm70/probe.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sm70_features import FEATURES, LEAKY_GROUPS  # noqa: E402

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def nll(y, z):
    # -log sigmoid(+-z), stable
    return np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y)


def encode(x_tr, x_te, max_cat=12, nbins=10):
    """Bin indices of x (training-fold deciles, or the categories of a small discrete feature; missing = own bin),
    renumbered to 0..k-1 over the bins seen in training (bin 0 = reference, dropped from the design)."""
    ok = ~np.isnan(x_tr)
    vals = np.unique(x_tr[ok])
    if len(vals) <= max_cat:
        edges = (vals[:-1] + vals[1:]) / 2
    else:
        edges = np.unique(np.quantile(x_tr[ok], np.linspace(0, 1, nbins + 1)[1:-1]))
        # point masses (zero-inflated features such as same-day counts): an edge value carrying more than 1% of the
        # training values gets a second edge just above it, so the mass point is a bin of its own
        xs = np.sort(x_tr[ok])
        mass = (np.searchsorted(xs, edges, side="right") - np.searchsorted(xs, edges, side="left")) / len(xs)
        extra = np.array([np.nextafter(e, np.inf) for e, m in zip(edges, mass) if m > 0.01], dtype=np.float64)
        edges = np.unique(np.concatenate([edges, extra]))
    k = len(edges) + 1

    def bins(x):
        b = np.searchsorted(edges, x, side="right")
        b[np.isnan(x)] = k
        return b
    btr, bte = bins(x_tr), bins(x_te)
    used = np.unique(btr)
    remap = np.full(k + 1, 0)            # unseen test bins fall back to the reference bin
    remap[used] = np.arange(len(used))
    return remap[btr], remap[bte], len(used)


OPT = {"bins": 10, "interact": True, "ridge": 1.0, "fit_weight": "review"}


def design(z, encoded, dtype=np.float32):
    """[1, z, onehot_j(bin), onehot_j(bin) * z for every feature and non-reference bin j] (no * z with --no-interact)."""
    m_ = 2 if OPT["interact"] else 1
    width = 2 + sum(m_ * (k - 1) for _, k in encoded)
    X = np.zeros((len(z), width), dtype=dtype)
    X[:, 0] = 1
    X[:, 1] = z
    c = 2
    rows = np.arange(len(z))
    for b, k in encoded:
        m = b > 0
        X[rows[m], c + m_ * (b[m] - 1)] = 1
        if m_ == 2:
            X[rows[m], c + 2 * (b[m] - 1) + 1] = z[m]
        c += m_ * (k - 1)
    return X


def fit_logistic(X, y, sw=None, iters=30):
    """Newton / IRLS with a small ridge on every column but the intercept (X float32, sums in float64)."""
    ridge = OPT["ridge"]
    sw = np.ones(len(y)) if sw is None else sw
    w = np.zeros(X.shape[1])
    reg = np.full(X.shape[1], ridge)
    reg[0] = 0.0
    for _ in range(iters):
        eta = X @ w.astype(X.dtype)
        p = 1 / (1 + np.exp(-eta.astype(np.float64)))
        g = (X.T @ (sw * (p - y)).astype(X.dtype)).astype(np.float64) + reg * w
        h = (X.T @ (X * (sw * p * (1 - p)).astype(X.dtype)[:, None])).astype(np.float64) + np.diag(reg + 1e-9)
        step = np.linalg.solve(h, g)
        w -= step
        if np.max(np.abs(step)) < 1e-5:
            break
    return w


def oof_losses(df, folds, z, y, feats, head, max_train, rng, chunk=500_000):
    """Per-review out-of-fold loss of the recalibration with `feats` (list of column names; [] = baseline)."""
    loss = np.empty(len(df))
    for f in range(folds.max() + 1):
        te_idx = np.flatnonzero(folds == f)
        tr_idx = np.flatnonzero(folds != f)
        if len(tr_idx) > max_train:
            tr_idx = np.sort(rng.choice(tr_idx, max_train, replace=False))
        enc_tr, enc_te = [], []
        for name in feats:
            col = name + "@c" if head == "curve" and (name + "@c") in df.columns else name
            x = df[col].to_numpy(np.float64)
            btr, bte, k = encode(x[tr_idx], x[te_idx], nbins=OPT["bins"])
            enc_tr.append((btr, k))
            enc_te.append((bte, k))
        sw = None
        if OPT["fit_weight"] == "user":
            uu = df["user_id"].to_numpy()[tr_idx]
            cnt = pd.Series(uu).map(pd.Series(uu).value_counts()).to_numpy(np.float64)
            sw = len(tr_idx) / (len(np.unique(uu)) * cnt)
        w = fit_logistic(design(z[tr_idx], enc_tr), y[tr_idx], sw)
        for s in range(0, len(te_idx), chunk):
            sl = slice(s, s + chunk)
            X = design(z[te_idx][sl], [(b[sl], k) for b, k in enc_te], dtype=np.float64)
            loss[te_idx[sl]] = nll(y[te_idx][sl], X @ w)
    return loss


def oof_tree(df, folds, z, y, feats, head, max_train, rng):
    from sklearn.ensemble import HistGradientBoostingClassifier
    cols = [(n + "@c" if head == "curve" and (n + "@c") in df.columns else n) for n in feats]
    X = np.column_stack([z] + [df[c].to_numpy(np.float64) for c in cols])
    loss = np.empty(len(df))
    for f in range(folds.max() + 1):
        te = folds == f
        tr_idx = np.flatnonzero(~te)
        if len(tr_idx) > max_train:
            tr_idx = np.sort(rng.choice(tr_idx, max_train, replace=False))
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=200,
                                           l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                           n_iter_no_change=20, random_state=0)
        m.fit(X[tr_idx], y[tr_idx])
        p = m.predict_proba(X[te])[:, 1]
        loss[te] = nll(y[te], logit(p))
    return loss


def user_sums(users, loss):
    s = pd.Series(loss).groupby(users).sum()
    return s.to_numpy()


def bootstrap(delta_u, n_u, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    k = len(n_u)
    idx = rng.integers(0, k, size=(reps, k))
    d = delta_u[idx].sum(1) / n_u[idx].sum(1)
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="sm56", choices=("sm56", "fork"))
    ap.add_argument("--max-train", type=int, default=2_000_000)
    ap.add_argument("--only", default="", help="comma list of candidate names to run (default all)")
    ap.add_argument("--no-tree", action="store_true")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--no-interact", action="store_true", help="feature bins shift the logit only (no bin * z terms)")
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--fit-weight", choices=("review", "user"), default="review",
                    help="user: every training user carries the same total weight in the fit (evaluation stays "
                         "review-weighted)")
    a = ap.parse_args()
    OPT.update(bins=a.bins, interact=not a.no_interact, ridge=a.ridge, fit_weight=a.fit_weight)
    df = pd.read_parquet(a.features)
    if a.model == "sm56" and "f_a" in df.columns and df["f_a"].notna().all():
        # positive control: the big model's own prediction as a "feature" (it is better than SM56 by ~0.3%)
        df["ctl_fork"] = logit(df["f_i"].to_numpy(np.float64))
        df["ctl_fork@c"] = logit(df["f_a"].to_numpy(np.float64))
    pcols = {"sm56": ("p_a", "p_i"), "fork": ("f_a", "f_i")}[a.model]
    if a.model == "fork":
        df = df[df["f_a"].notna() & df["f_i"].notna()].reset_index(drop=True)
    users_sorted = np.sort(df["user_id"].unique())
    perm = np.random.default_rng(0).permutation(len(users_sorted))
    fold_of = dict(zip(users_sorted, perm % 5))
    folds = df["user_id"].map(fold_of).to_numpy()
    users = df["user_id"].to_numpy()
    y = df["y"].to_numpy(np.float64)
    n_u = pd.Series(np.ones(len(df))).groupby(users).sum().to_numpy()

    groups = {}
    present = [n for n in FEATURES if n in df.columns]   # features built into this parquet
    for name in present:
        groups.setdefault(FEATURES[name][0], []).append(name)
    cands = {name: [name] for name in present}
    cands.update({f"group:{g}": names for g, names in groups.items()})
    deployable = [n for n in present if FEATURES[n][0] not in LEAKY_GROUPS]
    cands["all-deployable"] = deployable
    if "ctl_fork" in df.columns:
        cands["control:big-model-prediction"] = ["ctl_fork"]
    if a.only:
        keep = a.only.split(",")
        cands = {k: v for k, v in cands.items() if k in keep}

    result = {"options": dict(OPT), "model": a.model, "users": int(len(users_sorted)), "reviews": int(len(df)), "max_train": a.max_train,
              "heads": {}}
    for head, pc in zip(("curve", "instant"), pcols):
        rng = np.random.default_rng(1)
        z = logit(df[pc].to_numpy(np.float64))
        raw = nll(y, z)
        t0 = time.time()
        base = oof_losses(df, folds, z, y, [], head, a.max_train, rng)
        bu = user_sums(users, base)
        H = {"raw_logloss": float(raw.sum() / len(df)), "platt_logloss": float(base.sum() / len(df)),
             "platt_minus_raw": float((base.sum() - raw.sum()) / len(df)),
             "platt_minus_raw_ci": bootstrap(user_sums(users, base) - user_sums(users, raw), n_u), "cands": {}}
        print(f"[sm70 probe] {a.model} {head}: raw {H['raw_logloss']:.5f} platt {H['platt_logloss']:.5f}", flush=True)
        for cname, feats in cands.items():
            t1 = time.time()
            cl = oof_losses(df, folds, z, y, feats, head, a.max_train if len(feats) < 8 else (a.max_train // 2 if len(feats) < 20 else a.max_train // 4), rng)
            du = user_sums(users, cl) - bu
            d = float(du.sum() / n_u.sum())
            ci = bootstrap(du, n_u)
            better_users = float(np.mean(du < 0))
            H["cands"][cname] = {"delta": d, "ci": ci, "rel_pct": 100 * d / H["platt_logloss"],
                                 "users_better": better_users, "features": feats}
            print(f"  {cname:22s} {d:+.6f} [{ci[0]:+.6f}, {ci[1]:+.6f}] ({time.time() - t1:.0f}s)", flush=True)
        if not a.no_tree:
            t1 = time.time()
            tb = oof_tree(df, folds, z, y, [], head, a.max_train, rng)
            tf = oof_tree(df, folds, z, y, deployable, head, a.max_train, rng)
            for key, lo in (("tree:z-only", tb), ("tree:all-deployable", tf)):
                du = user_sums(users, lo) - bu
                H["cands"][key] = {"delta": float(du.sum() / n_u.sum()), "ci": bootstrap(du, n_u),
                                   "rel_pct": 100 * float(du.sum() / n_u.sum()) / H["platt_logloss"],
                                   "users_better": float(np.mean(du < 0))}
                print(f"  {key:22s} {H['cands'][key]['delta']:+.6f} {H['cands'][key]['ci']}", flush=True)
            du = user_sums(users, tf) - user_sums(users, tb)
            H["cands"]["tree:all-minus-z"] = {"delta": float(du.sum() / n_u.sum()), "ci": bootstrap(du, n_u),
                                              "users_better": float(np.mean(du < 0))}
            print(f"  tree all - z-only      {H['cands']['tree:all-minus-z']['delta']:+.6f} "
                  f"{H['cands']['tree:all-minus-z']['ci']} ({time.time() - t1:.0f}s)", flush=True)
        H["seconds"] = round(time.time() - t0, 1)
        result["heads"][head] = H
        Path(a.out).write_text(json.dumps(result, indent=1))
    print(f"[sm70 probe] done -> {a.out}")


if __name__ == "__main__":
    main()
