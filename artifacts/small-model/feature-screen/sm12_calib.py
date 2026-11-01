"""SM12: offline Platt calibration (temperature + bias, and temperature only) of fork/student heads, 2-fold CV by user parity."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "4"
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

t0 = time.time()
SM = Path(__file__).resolve().parents[1]
DUMP = SM / "sm24-dump"
ORIG = Path(r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011")
users = sorted(int(p.name.split("_")[0]) for p in DUMP.glob("*_st_w8+s8.npz") if (DUMP / p.name.replace("_st_", "_fork_")).exists())
P = {(m, h): [] for m in ("fork", "st") for h in ("ahead", "instant")}
Y, U, S = [], [], []
for u in users:
    d = {m: np.load(DUMP / f"{u}_{m}_w8+s8.npz") for m in ("fork", "st")}
    rv = pd.read_parquet(ORIG / "revlogs" / f"user_id={u}", columns=["card_id", "day_offset", "rating"])
    same = rv.groupby(["card_id", "day_offset"]).cumcount().to_numpy() > 0
    rating = rv["rating"].to_numpy()
    prev_fail = pd.Series(rating == 1).groupby(rv["card_id"].to_numpy()).shift(1).fillna(False).to_numpy().astype(bool)
    stage = np.where(same, 0, np.where(prev_fail, 1, 2))
    i = d["st"]["review_th"] - 1
    Y.append(d["st"]["y"].astype(float)); U.append(np.full(len(i), u)); S.append(stage[i])
    for (m, h), l in P.items():
        l.append(np.clip(d[m][f"p_{h}"], 1e-7, 1 - 1e-7))
Y = np.concatenate(Y); U = np.concatenate(U); S = np.concatenate(S)
P = {k: np.concatenate(v) for k, v in P.items()}
fold = U % 2
eps = 1e-7

def loss(p, y):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

def fit(z, y, bias):
    def f(w):
        a, b = w[0], (w[1] if bias else 0.0)
        s = a * z + b; p = 1 / (1 + np.exp(-s))
        g = p - y
        val = np.mean(np.logaddexp(0, s) - y * s)
        gr = [np.mean(g * z)] + ([np.mean(g)] if bias else [])
        return val, np.array(gr)
    r = minimize(f, np.array([1.0, 0.0][: 2 if bias else 1]), jac=True, method="L-BFGS-B")
    return float(r.x[0]), float(r.x[1]) if bias else 0.0

def agg(L, mask):
    n = mask.sum(); lw = L[mask].sum() / n
    pu = pd.Series(L[mask]).groupby(U[mask]).mean().mean()
    return lw, pu

slices = {"all": np.ones(len(Y), bool), "spaced": S == 2}
out = {"users": len(users), "rows": int(len(Y)), "results": [], "params": []}
lines = [f"{len(users)} users, {len(Y)} reviews"]
for h in ("ahead", "instant"):
    for mode in ("temp+bias", "temp-only"):
        Ls = {}
        for m in ("fork", "st"):
            p = P[(m, h)]; z = np.log(p / (1 - p)); pc = np.empty_like(p)
            for f_ in (0, 1):
                tr, te = fold != f_, fold == f_
                a, b = fit(z[tr], Y[tr], mode == "temp+bias")
                pc[te] = 1 / (1 + np.exp(-(a * z[te] + b)))
                out["params"].append({"head": h, "mode": mode, "model": m, "fit_on_fold": 1 - f_ if False else f_, "eval_on_fold": 1 - f_ if False else int(1 - f_), "a": a, "b": b})
                lines.append(f"  [{h}/{mode}/{m}] fit on users parity!={f_}  a={a:.4f} b={b:+.4f}")
            Ls[m] = (loss(p, Y), loss(pc, Y))
        # fix: fit on fold != f_ and evaluated on fold f_ ; param record labelled accordingly
        for sl, mask in slices.items():
            r = {"head": h, "mode": mode, "slice": sl}
            for w, nm in ((0, "review_weighted"), (1, "per_user_mean")):
                b4 = {m: agg(Ls[m][0], mask)[w] for m in Ls}; af = {m: agg(Ls[m][1], mask)[w] for m in Ls}
                r[nm] = {"fork_before": b4["fork"], "st_before": b4["st"], "fork_after": af["fork"], "st_after": af["st"],
                         "gap_before_pct": 100 * (b4["st"] / b4["fork"] - 1), "gap_after_pct": 100 * (af["st"] / af["fork"] - 1)}
            out["results"].append(r)
            for nm in ("review_weighted", "per_user_mean"):
                x = r[nm]
                lines.append(f"[{h:7s}|{mode:9s}|{sl:6s}|{nm:15s}] fork {x['fork_before']:.5f}->{x['fork_after']:.5f}  st {x['st_before']:.5f}->{x['st_after']:.5f}  gap {x['gap_before_pct']:+.2f}% -> {x['gap_after_pct']:+.2f}%")
for q in out["params"]:
    q["fit_on_users_parity_ne"] = q.pop("fit_on_fold"); q["eval_on_users_parity"] = q.pop("eval_on_fold")
lines.append("(review-weighted == gate weighting: per-user LogLoss weighted by review count)")
lines.append(f"runtime {time.time() - t0:.0f}s")
txt = "\n".join(lines); print(txt)
(Path(__file__).parent / "sm12-calib.txt").write_text(txt)
(Path(__file__).parent / "sm12-calib.json").write_text(json.dumps(out, indent=1))
