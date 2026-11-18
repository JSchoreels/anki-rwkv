"""Screen of the per-user rep-number features (repfeat.py) by review-type slice. Same method as screen.py (logistic
recalibration of the model's logit, + feature quantile bins and their interaction with the logit, 2-fold by user),
gain reported per slice as % of the fork's log loss there, i.e. directly comparable with the student-vs-fork gap."""
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression

D = Path("."); MODELS = ("fork", "st")
users = sorted(int(p.stem) for p in (D / "fx").glob("*.npz") if all((D / "dump" / f"{p.stem}_{m}_w8+s8.npz").exists() for m in MODELS))
NAMES = ("rep", "phase", "preset", "deck")
F, P, Y, U, S = {}, {(m, h): [] for m in MODELS for h in ("ahead", "instant")}, [], [], []
for u in users:
    z = np.load(D / "fx" / f"{u}.npz"); dd = {m: np.load(D / "dump" / f"{u}_{m}_w8+s8.npz") for m in MODELS}
    i = dd["st"]["review_th"] - 1; n = len(i)
    for k in z.files:
        if k.startswith(("rp_", "rr_")):
            F.setdefault(k, {})[u] = z[k][i]
    for key in P:
        P[key].append(dd[key[0]][f"p_{key[1]}"])
    Y.append(dd["st"]["y"]); U.append(np.full(n, u)); S.append(z["review_type"][i])
y = np.concatenate(Y); user = np.concatenate(U); state = np.concatenate(S)
lens = [len(a) for a in Y]
F = {k: np.concatenate([v.get(u, np.full(L, np.nan)) for u, L in zip(users, lens)]) for k, v in F.items()}
P = {k: np.clip(np.concatenate(v), 1e-6, 1 - 1e-6) for k, v in P.items()}
fold = np.array([hash((uu * 2654435761) % 1000003) for uu in user]) % 2
rng = np.random.default_rng(0)
uidx = np.unique(user, return_inverse=True)[1]; NU = uidx.max() + 1
BOOT = np.random.default_rng(1).integers(0, NU, size=(500, NU))
SL = {"all": np.ones(len(y), bool), "learning": state == 1, "review": state == 2, "relearning": state == 3}


def ll(p):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def bins(fx, edges):
    c = np.where(np.isnan(fx), len(edges) + 1, np.digitize(np.nan_to_num(fx, nan=0), edges))
    return np.eye(len(edges) + 2)[c]


def crossfit(p, feats):
    lg = np.log(p / (1 - p)); out = np.empty(len(p))
    for k in (0, 1):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        lb = np.quantile(lg[tr], np.linspace(0.05, 0.95, 19))
        def X(ix):
            cols = [lg[ix, None], np.eye(len(lb) + 1)[np.digitize(lg[ix], lb)]]
            for fx, e in zip(feats, edges):
                oh = bins(fx[ix], e); cols += [oh, oh * lg[ix, None]]
            return np.hstack(cols)
        edges = []
        for fx in feats:
            v = fx[tr][~np.isnan(fx[tr])]
            edges.append(np.unique(np.quantile(v, np.linspace(1 / 10, 9 / 10, 9))) if len(v) else np.array([0.0]))
        s = rng.choice(tr, min(400_000, len(tr)), replace=False)
        out[te] = LogisticRegression(C=1.0, max_iter=400).fit(X(s), y[s]).predict_proba(X(te))[:, 1]
    return np.clip(out, 1e-6, 1 - 1e-6)


def ci(d, s):
    a, c = np.bincount(uidx[s], d[s], NU), np.bincount(uidx[s], minlength=NU)
    m = np.array([a[b].sum() / max(c[b].sum(), 1) for b in BOOT]); return np.quantile(m, [0.025, 0.975])


print(f"{len(users)} users, {len(y):,} scored reviews")
for h in ("ahead", "instant"):
    when = "_atprev" if h == "ahead" else ""
    fork_ll = ll(P[("fork", h)])
    print(f"\n[{h}] student raw vs fork: " + "  ".join(f"{s} {100 * (ll(P[('st', h)])[m].sum() / fork_ll[m].sum() - 1):+.2f}%" for s, m in SL.items()))
    print(f"[{h}] gain of each feature over the model's own recalibration, % of fork LL in the slice (* = user CI > 0)")
    for m in MODELS:
        base = ll(crossfit(P[(m, h)], []))
        cands = {f"rp_{nm}": [F[f"rp_{nm}{when}"]] for nm in NAMES} | {f"rr_{nm}": [F[f"rr_{m}_{h}_{nm}{when}"]] for nm in NAMES}
        cands["rp all + counts"] = [F[f"rp_{nm}{when}"] for nm in NAMES] + [F[f"rp_rep_n{when}"]]
        cands["rr all"] = [F[f"rr_{m}_{h}_{nm}{when}"] for nm in NAMES]
        for name, feats in cands.items():
            d = base - ll(crossfit(P[(m, h)], feats)); cells = []
            for s, mk in SL.items():
                lo, hi = ci(d, mk); cells.append(f"{s} {100 * d[mk].sum() / fork_ll[mk].sum():+.2f}%{'*' if lo > 0 else ' '}")
            print(f"  {m:4s} {name:16s} " + "  ".join(cells), flush=True)
