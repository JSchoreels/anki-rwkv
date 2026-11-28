#!/usr/bin/env python
"""Small + Tiny ensemble (SM184), CPU only. See design.md. Run: python sm184_ensemble.py"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
DUMP = Path("C:/Users/jonat/workspace/wt/sm160/work/sm160-dump")
USERS732 = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json")
USERS_WFIT = Path("C:/Users/jonat/workspace/wt/sm160/artifacts/small-model/sm160/users-wfit.json")
LABEL = "w8+lr2.16.4q8x8@e2s-query"
M = {"BIO": "sm99-big-inst-e2", "BTH": "sm91s2-big-showtime",
     "SIO": "sm113-inst-e35st-st-dec2ep32k-t99", "STH": "sm118-e26-dec2ep32k-t91s2",
     "TIO": "sm130-io-dec2ep32k-t99", "TTH": "sm130-th-dec2ep32k-t91s2"}
W_SMALL, W_TINY, W_BIG = 0.78, 0.69, 0.80   # SM160 fitted weights of the Instant-Only model
GRID = np.round(np.linspace(0, 1, 101), 2)
CLIP = 1e-7


def loss_vec(y, p):
    eps = np.finfo(np.float64).eps
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return -(xlogy(y, p) + xlogy(1 - y, 1 - p))


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), CLIP, 1 - CLIP)
    return np.log(p) - np.log1p(-p)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def boot(a, b, draws=2000):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    idx = np.random.default_rng(0).integers(0, len(a), size=(draws, len(a)))
    bs = (a[idx].sum(1) / b[idx].sum(1) - 1) * 100
    return {"value": float((a.sum() / b.sum() - 1) * 100),
            "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "users": int(len(a))}


def boot_mean_diff(a, b, draws=2000):
    d = np.asarray(a) - np.asarray(b)
    d = d[np.isfinite(d)]
    idx = np.random.default_rng(0).integers(0, len(d), size=(draws, len(d)))
    bs = d[idx].mean(1)
    return {"value": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def load_user(folder, u):
    out, y0, rv = {}, None, None
    for k, tag in M.items():
        f = Path(folder) / f"{u}_{tag}_{LABEL}.npz"
        if not f.exists():
            return None
        z = np.load(f)
        if y0 is None:
            y0, rv = np.asarray(z["y"], dtype=np.float64), z["review_th"]
        elif not (np.array_equal(z["review_th"], rv) and np.array_equal(z["y"], y0)):
            raise SystemExit(f"review sets differ: user {u} {k}")
        out[k] = (logit(z["p_instant"]), logit(z["p_ahead"]))
    lg = {k: v[0] for k, v in out.items()}
    la = {k: v[1] for k, v in out.items()}
    if not all(np.isfinite(v).all() for v in list(lg.values()) + list(la.values())):
        return None
    return y0, lg, la


def small_ens(lg):
    return W_SMALL * lg["SIO"] + (1 - W_SMALL) * lg["STH"]


def tiny_ens(lg):
    return W_TINY * lg["TIO"] + (1 - W_TINY) * lg["TTH"]


def big_ens(lg):
    return W_BIG * lg["BIO"] + (1 - W_BIG) * lg["BTH"]


def users_with_dumps(folder, users):
    return [u for u in users if all((Path(folder) / f"{u}_{t}_{LABEL}.npz").exists() for t in M.values())]


def fit():
    users = users_with_dumps(DUMP / "wfit", json.loads(USERS_WFIT.read_text())["users"])
    ys, cols = [], {k: [] for k in ("SIO", "STH", "TIO", "TTH", "BIO", "SE", "TE")}
    ca = {k: [] for k in ("STH", "TTH")}
    for u in users:
        r = load_user(DUMP / "wfit", u)
        if r is None:
            continue
        y, lg, la = r
        ys.append(y)
        for k in ("SIO", "STH", "TIO", "TTH", "BIO"):
            cols[k].append(lg[k])
        cols["SE"].append(small_ens(lg))
        cols["TE"].append(tiny_ens(lg))
        for k in ("STH", "TTH"):
            ca[k].append(la[k])
    y = np.concatenate(ys)
    C = {k: np.concatenate(v) for k, v in cols.items()}
    CA = {k: np.concatenate(v) for k, v in ca.items()}
    base = loss_vec(y, sigmoid(C["SIO"])).sum()

    def grid(a, b):
        L = [loss_vec(y, sigmoid(w * a + (1 - w) * b)).sum() for w in GRID]
        return float(GRID[int(np.argmin(L))]), float(min(L))

    res = {"users": len(ys), "reviews": int(len(y))}
    wa, la_ = grid(C["SIO"], C["TIO"])
    res["wa_small_vs_tiny_io"] = wa
    wc, lc = grid(C["SE"], C["TE"])
    res["wc_small_ens_vs_tiny_ens"] = wc
    wd, ld = grid(C["BIO"], C["SIO"])
    res["wd_big_vs_small_io"] = wd
    res["w_curve_small_th_vs_tiny_th"] = grid(CA["STH"], CA["TTH"])[0]
    X = np.stack([C["SIO"], C["STH"], C["TIO"], C["TTH"]], 1)

    def f(w):
        return float(loss_vec(y, sigmoid(X @ w)).sum() / len(y))

    def g(w):
        return (X * (sigmoid(X @ w) - y)[:, None]).mean(0)

    r = minimize(f, np.array([0.5, 0.0, 0.5, 0.0]), jac=g, method="SLSQP", bounds=[(0, 1)] * 4,
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(4)}])
    res["wb_4way_SIO_STH_TIO_TTH"] = [float(x) for x in r.x]
    res["train_pct_vs_small_io"] = {"a": (la_ / base - 1) * 100, "b": float((r.fun * len(y) / base - 1) * 100),
                                    "c": (lc / base - 1) * 100, "d": (ld / base - 1) * 100}
    return res


def auc(y, p):
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    wf = fit()
    print("fit", json.dumps(wf), flush=True)
    wa, wc, wd, wcv = wf["wa_small_vs_tiny_io"], wf["wc_small_ens_vs_tiny_ens"], wf["wd_big_vs_small_io"], wf["w_curve_small_th_vs_tiny_th"]
    wb = np.array(wf["wb_4way_SIO_STH_TIO_TTH"])
    inst = {
        "Small Instant-Only alone (SM113)": lambda g: g["SIO"],
        "Big Instant-Only alone (SM99)": lambda g: g["BIO"],
        "Tiny shipped ensemble (0.69 Tiny Instant-Only + 0.31 Tiny two-head)": tiny_ens,
        "Small two-model ensemble (0.78 Small Instant-Only + 0.22 Small two-head)": small_ens,
        "Big two-model ensemble (0.80, reference)": big_ens,
        "(a) Small Instant-Only + Tiny Instant-Only": lambda g: wa * g["SIO"] + (1 - wa) * g["TIO"],
        "(b) 4-way: Small IO, Small two-head, Tiny IO, Tiny two-head": lambda g: wb[0] * g["SIO"] + wb[1] * g["STH"] + wb[2] * g["TIO"] + wb[3] * g["TTH"],
        "(c) Small two-model ensemble + Tiny two-model ensemble": lambda g: wc * small_ens(g) + (1 - wc) * tiny_ens(g),
        "(d) reference: Big Instant-Only + Small Instant-Only": lambda g: wd * g["BIO"] + (1 - wd) * g["SIO"],
    }
    curve = {
        "Small two-head alone (SM118)": lambda g: g["STH"],
        "Big two-head alone (SM91s2)": lambda g: g["BTH"],
        "Tiny two-head alone (SM130)": lambda g: g["TTH"],
        "Small two-head + Tiny two-head": lambda g: wcv * g["STH"] + (1 - wcv) * g["TTH"],
    }
    users = users_with_dumps(DUMP / "e732", json.loads(USERS732.read_text())["users"])
    S = {k: [] for k in inst}
    SC = {k: [] for k in curve}
    A = {k: [] for k in inst}
    n, used = [], []
    for u in users:
        r = load_user(DUMP / "e732", u)
        if r is None:
            continue
        y, lg, la = r
        used.append(u)
        n.append(len(y))
        for k, f in inst.items():
            z = f(lg)
            S[k].append(loss_vec(y, sigmoid(z)).sum())
            A[k].append(auc(y, z))
        for k, f in curve.items():
            SC[k].append(loss_vec(y, sigmoid(f(la))).sum())
    S = {k: np.array(v) for k, v in S.items()}
    SC = {k: np.array(v) for k, v in SC.items()}
    A = {k: np.array(v) for k, v in A.items()}
    sm, bg = "Small Instant-Only alone (SM113)", "Big Instant-Only alone (SM99)"
    out = {"label": "T24", "users": len(used), "reviews": int(sum(n)), "weights_fit_on_wfit": wf, "instant": {}, "curve": {}}
    for k in inst:
        d = {"logloss": float(S[k].sum() / sum(n)), "auc_user_mean": float(np.nanmean(A[k])),
             "vs_small_io": boot(S[k], S[sm]), "vs_big_io": boot(S[k], S[bg]),
             "auc_diff_vs_small_io": boot_mean_diff(A[k], A[sm]), "auc_diff_vs_big_io": boot_mean_diff(A[k], A[bg])}
        d["worth_considering"] = bool(d["vs_small_io"]["ci"][1] < 0 and d["vs_small_io"]["value"] <= -0.05)
        out["instant"][k] = d
    for k in curve:
        out["curve"][k] = {"logloss": float(SC[k].sum() / sum(n)), "vs_small_th": boot(SC[k], SC["Small two-head alone (SM118)"]),
                           "vs_big_th": boot(SC[k], SC["Big two-head alone (SM91s2)"])}
    (HERE / "sm184-result.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, d in out["instant"].items():
        print(f"{k}\n  vs Small IO {d['vs_small_io']['value']:+.3f}% {d['vs_small_io']['ci']}  vs Big IO {d['vs_big_io']['value']:+.3f}% {d['vs_big_io']['ci']}  AUC {d['auc_user_mean']:.4f} (vs small {d['auc_diff_vs_small_io']['value']:+.4f} {d['auc_diff_vs_small_io']['ci']}) go={d['worth_considering']}")
    for k, d in out["curve"].items():
        print(f"curve {k}: vs Small TH {d['vs_small_th']['value']:+.3f}% {d['vs_small_th']['ci']} vs Big TH {d['vs_big_th']['value']:+.3f}% {d['vs_big_th']['ci']}")
    print("users", len(used))


if __name__ == "__main__":
    sys.exit(main())
