"""Residual feature screen: does a candidate feature lower the log loss of a model's own predictions?

For each (model, head): base = logistic recalibration of the model (logit + 20 logit-bin one-hots). Candidate = base
+ the feature's 8 quantile bins (+ NaN bin) and their interaction with the logit. 2-fold cross-fit by user (fit on one
half of the users, score the other), review-weighted. Reported: nats/review saved and % of the model's log loss.
Also: the student-minus-fork log-loss gap per feature bin (where the student loses).
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

D = Path(sys.argv[1])
MODELS = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("fork", "st")
users = sorted(int(p.stem) for p in (D / "fx").glob("*.npz")
               if all((D / "dump" / f"{p.stem}_{m}_w8+s8.npz").exists() for m in MODELS))
rows = {"user": [], "y": []}
P = {(m, h): [] for m in MODELS for h in ("ahead", "instant")}
F = {}
for u in users:
    z = np.load(D / "fx" / f"{u}.npz")
    dd = {m: np.load(D / "dump" / f"{u}_{m}_w8+s8.npz") for m in MODELS}
    th = dd["st"]["review_th"]
    assert all((dd[m]["review_th"] == th).all() and (dd[m]["y"] == dd["st"]["y"]).all() for m in MODELS)
    idx = th - 1
    rows["user"].append(np.full(len(th), u)); rows["y"].append(dd["st"]["y"])
    for (m, h) in P:
        P[(m, h)].append(dd[m][f"p_{h}"])
    for k in z.files:
        if z[k].ndim and k != "review_th":
            F.setdefault(k, []).append(z[k][idx].astype(float))
user = np.concatenate(rows["user"]); y = np.concatenate(rows["y"])
P = {k: np.clip(np.concatenate(v), 1e-6, 1 - 1e-6) for k, v in P.items()}
F = {k: np.concatenate(v) for k, v in F.items() if len(v) == len(users)}
if len(sys.argv) > 3:
    F = {k: v for k, v in F.items() if k in sys.argv[3].split(",")}
fold = (np.array([hash((u * 2654435761) % 1000003) for u in user]) % 2)
rng = np.random.default_rng(0)
print(f"{len(users)} users, {len(y):,} benchmark reviews; feature coverage checked per feature")


def ll(p, yy):
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))


def onehot(codes, k):
    return np.eye(k)[codes]


def design(lg, lbins, fx=None, fedges=None):
    cols = [lg[:, None], onehot(np.digitize(lg, lbins), len(lbins) + 1)]
    if fx is not None:
        c = np.where(np.isnan(fx), len(fedges) + 1, np.digitize(np.nan_to_num(fx, nan=0), fedges))
        oh = onehot(c, len(fedges) + 2)
        cols += [oh, oh * lg[:, None]]
    return np.hstack(cols)


def crossfit(p, fx=None, sub=300_000):
    lg = np.log(p / (1 - p))
    out = np.empty(len(p))
    for k in (0, 1):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        lbins = np.quantile(lg[tr], np.linspace(0.05, 0.95, 19))
        fedges = None
        if fx is not None:
            v = fx[tr][~np.isnan(fx[tr])]
            fedges = np.unique(np.quantile(v, np.linspace(1 / 8, 7 / 8, 7))) if len(v) else np.array([0.0])
        s = rng.choice(tr, min(sub, len(tr)), replace=False)
        m = LogisticRegression(C=1.0, max_iter=300).fit(design(lg[s], lbins, None if fx is None else fx[s], fedges), y[s])
        out[te] = m.predict_proba(design(lg[te], lbins, None if fx is None else fx[te], fedges))[:, 1]
    return np.clip(out, 1e-6, 1 - 1e-6)


uidx = np.unique(user, return_inverse=True)[1]
NU = uidx.max() + 1
BOOT = np.random.default_rng(1).integers(0, NU, size=(1000, NU))


def boot(d):
    """95% interval of the review-weighted mean of d, resampling users."""
    s, n = np.bincount(uidx, d, NU), np.bincount(uidx, minlength=NU)
    m = np.array([s[b].sum() / n[b].sum() for b in BOOT])
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


res = {}
for h in ("ahead", "instant"):
    for m in MODELS:
        p = P[(m, h)]
        raw = ll(p, y).mean()
        lb = ll(crossfit(p), y)
        base = lb.mean()
        res[(m, h)] = {"raw": raw, "recal": raw - base, "feat": {}, "ci": {}}
        for k, fx in F.items():
            d = lb - ll(crossfit(p, fx), y)
            res[(m, h)]["feat"][k] = d.mean()
            res[(m, h)]["ci"][k] = boot(d)
        print(f"{m:4s} {h:7s} raw LL {raw:.5f}, recalibration saves {raw - base:+.5f}", flush=True)

print("\nnats/review saved over the recalibrated model (2-fold by user); % of the model's raw log loss")
print(f"{'feature':32s} " + " ".join(f"{m + '/' + h:>18s}" for h in ('ahead', 'instant') for m in MODELS))
for k in F:
    cells = []
    for h in ("ahead", "instant"):
        for m in MODELS:
            v = res[(m, h)]["feat"][k]
            lo, hi = res[(m, h)]["ci"][k]
            star = "*" if lo > 0 else (" " if hi > 0 else "-")
            cells.append(f"{100 * v / res[(m, h)]['raw']:+.2f}%{star}")
    print(f"{k:32s} " + " ".join(f"{c:>18s}" for c in cells))

print("\nstudent - fork log-loss gap by feature quartile (review-weighted, % of fork LL in that bin)")
for h in ("ahead", "instant"):
    gap = ll(P[("st", h)], y) - ll(P[("fork", h)], y)
    lf = ll(P[("fork", h)], y)
    print(f"[{h}] overall {100 * gap.sum() / lf.sum():+.2f}%")
    for k, fx in F.items():
        ok = ~np.isnan(fx)
        if ok.mean() < 0.05:
            continue
        q = np.unique(np.quantile(fx[ok], [0.25, 0.5, 0.75]))
        c = np.digitize(fx[ok], q)
        parts = [f"{100 * gap[ok][c == i].sum() / lf[ok][c == i].sum():+.2f}% (n {100 * (c == i).mean():.0f}%)" for i in range(len(q) + 1)]
        print(f"  {k:30s} " + "  ".join(parts))
json.dump({f"{m}/{h}": v for (m, h), v in res.items()}, open(D / ("screen-" + "-".join(MODELS) + ".json"), "w"), indent=1)
