"""Slow-Good signal (SM88): among Good answers with the same predicted recall at the card's NEXT review, do slow Goods lapse more?
CPU analysis on the 150 users of the response-time prediction (SM82) stage 2 sample.

Pairs: review j is Good on a later day than the card's previous review, the card's next review k exists and is on a later day
than j. Outcome: k is Again. Predictors: j's answer time (duration capped at the user's cap, floored 0.5 s) relative to the
user's running median Good time before j (ln ratio; >= 20 earlier Goods) and relative to the card's own earlier Good times
(geometric mean). Controls: the small model's scheduling-time prediction at k (curve head made at j, evaluated at k's real
interval; scheduling-time vs review-time accuracy (SM81) dump, read only) and its review-time instant prediction at k.

  python scripts/sm88_slow_good.py prepare --procs 4
  python scripts/sm88_slow_good.py report
"""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
from multiprocessing import Pool

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NPZ = Path("C:/Users/jonat/workspace/wt/sm82s2/artifacts/small-model/sm82/stage2/work/curves/users")   # read only
DUMP = Path("C:/Users/jonat/workspace/wt/sm81/artifacts/small-model/sm81/gpu/dump")                      # read only
SMALL = "s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon_w8+s8"
OUT = ROOT / "artifacts" / "small-model" / "sm88"
WORK = OUT / "work"
MIN_GOOD = 20
FIELDS = ["again", "pc", "pi", "rel_u", "rel_c", "card_n", "el_k"]


def low():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001
        pass


def build_user(user):
    import pandas as pd
    from scripts import compress_common as cc
    low()
    cap = int(np.load(NPZ / f"{user}.npz")["cap_ms"])
    d = np.load(DUMP / f"{user}_{SMALL}.npz")
    rev = pd.read_parquet(Path(cc.data_path()) / "revlogs" / f"user_id={user}", columns=["card_id", "day_offset", "rating", "duration", "elapsed_days"]).reset_index(drop=True)
    n = len(rev)
    card, el, rating = rev["card_id"], rev["elapsed_days"].to_numpy(), rev["rating"].to_numpy()
    t = np.maximum(np.minimum(rev["duration"].to_numpy(np.float64), cap) / 1000.0, 0.5)
    lt = np.log(t)
    good = rating == 3
    gser = pd.Series(np.where(good, t, np.nan))
    run_med = gser.expanding(min_periods=MIN_GOOD).median().shift(1).to_numpy()
    lg = pd.Series(np.where(good, lt, 0.0))
    cs = (lg.groupby(card).cumsum() - lg).to_numpy()
    cn = (pd.Series(good * 1.0).groupby(card).cumsum() - good).to_numpy()
    card_gm = np.where(cn > 0, cs / np.maximum(cn, 1), np.nan)
    nxt = pd.Series(np.arange(n, dtype=np.float64)).groupby(card).shift(-1).to_numpy()
    j = np.where(good & (el >= 1) & ~np.isnan(nxt))[0]
    k = nxt[j].astype(np.int64)
    ok = el[k] >= 1
    j, k = j[ok], k[ok]
    th = d["review_th"].astype(np.int64) - 1
    pos = {r: i for i, r in enumerate(th)}
    sel = np.array([pos.get(x, -1) for x in k], dtype=np.int64)
    ok = (sel >= 0) & ~np.isnan(run_med[j])
    j, k, sel = j[ok], k[ok], sel[ok]
    again = (rating[k] == 1).astype(np.int8)
    ymatch = float(np.mean((d["y"][sel] == 1) == (again == 0))) if len(j) else 1.0
    return user, {"again": again, "pc": d["p_curve"][sel].astype(np.float64), "pi": d["p_instant"][sel].astype(np.float64),
                  "rel_u": np.log(t[j] / run_med[j]), "rel_c": np.where(np.isnan(card_gm[j]), np.nan, lt[j] - card_gm[j]),
                  "card_n": cn[j].astype(np.int32), "el_k": el[k].astype(np.int32)}, ymatch, int(((good & (el >= 1) & ~np.isnan(nxt))).sum())


def prepare(a):
    users = sorted(int(p.stem) for p in NPZ.glob("*.npz"))
    WORK.mkdir(parents=True, exist_ok=True)
    out = {}
    with Pool(a.procs) as pool:
        for u, r, ym, cand in pool.imap_unordered(build_user, users):
            out[u] = r
            print(u, len(r["again"]), "of", cand, f"ymatch {ym:.4f}", flush=True)
    np.savez_compressed(WORK / "pairs.npz", **{f"{u}_{k}": v for u, r in out.items() for k, v in r.items()}, users=np.array(sorted(out)))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def irls(X, y, groups, iters=30):
    import scipy.sparse as sp
    sparse = sp.issparse(X)
    wm = (lambda w: X.T @ sp.diags(w) @ X) if sparse else (lambda w: X.T @ (X * w[:, None]))
    dense = lambda M: M.toarray() if sparse else M
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(X @ b)))
        w = np.maximum(mu * (1 - mu), 1e-9)
        step = np.linalg.solve(dense(wm(w)) + 1e-9 * np.eye(X.shape[1]), X.T @ (y - mu))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    mu = 1 / (1 + np.exp(-(X @ b)))
    w = np.maximum(mu * (1 - mu), 1e-9)
    Hinv = np.linalg.inv(dense(wm(w)) + 1e-9 * np.eye(X.shape[1]))
    gs = np.zeros((groups.max() + 1, X.shape[1]))
    for c in range(X.shape[1]):
        col = X[:, c].toarray().ravel() if sparse else X[:, c]
        gs[:, c] = np.bincount(groups, col * (y - mu), minlength=groups.max() + 1)
    V = Hinv @ (gs.T @ gs) @ Hinv
    return b, np.sqrt(np.diag(V))


def report(a):
    z = np.load(WORK / "pairs.npz")
    users = z["users"]
    D = {k: np.concatenate([z[f"{u}_{k}"] for u in users]) for k in FIELDS}
    uid = np.concatenate([np.full(len(z[f"{u}_again"]), i) for i, u in enumerate(users)])
    nU = len(users)
    y = D["again"].astype(np.float64)
    res = {"users": int(nU), "pairs": int(len(uid)), "lapse_rate": float(y.mean()), "pred_lapse_curve": float((1 - D["pc"]).mean()),
           "pred_lapse_instant": float((1 - D["pi"]).mean()), "median_pred_lapse_curve": float(np.median(1 - D["pc"]))}
    rng = np.random.default_rng(0)
    B = 2000
    boots = rng.integers(0, nU, size=(B, nU))
    NB = 10

    def build(rel, pc, kq, rank):
        m = ~np.isnan(rel)
        r, yy, pp, uu = rel[m], y[m], 1 - pc[m], uid[m]
        if rank:
            qd = np.zeros(len(r), np.int64)
            for i in range(nU):
                s = np.where(uu == i)[0]
                if len(s):
                    rk = r[s].argsort().argsort()
                    qd[s] = np.minimum((rk * kq) // len(s), kq - 1)
        else:
            qd = np.searchsorted(np.quantile(r, np.linspace(0, 1, kq + 1)[1:-1]), r, side="right")
        lp = logit(1 - pp)
        bd = np.searchsorted(np.quantile(lp, np.linspace(0, 1, NB + 1)[1:-1]), lp, side="right")
        T = np.zeros((nU, NB, kq, 3))
        np.add.at(T, (uu, bd, qd, 0), 1.0)
        np.add.at(T, (uu, bd, qd, 1), yy)
        np.add.at(T, (uu, bd, qd, 2), pp)
        return T

    out = {}
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    for name, rel, rank in [("user", D["rel_u"], False), ("user_rank", D["rel_u"], True), ("card", D["rel_c"], False)]:
        for cname, pc in [("curve", D["pc"]), ("instant", D["pi"])]:
            for kq in (3, 5):
                T = build(rel, pc, kq, rank)
                Tp = T.sum(0)
                nb_ = Tp[:, :, 0].sum(1)
                wb = nb_ / nb_.sum()

                def stat(Tx):
                    n = Tx[:, :, 0]
                    rate = Tx[:, :, 1] / np.maximum(n, 1)
                    pred = Tx[:, :, 2] / np.maximum(n, 1)
                    return (wb[:, None] * rate).sum(0), (wb[:, None] * pred).sum(0), n.sum(0)
                obs, prd, nq = stat(Tp)
                ob = np.empty((B, kq)); pb = np.empty((B, kq))
                for i_, b in enumerate(boots):
                    ob[i_], pb[i_], _ = stat(T[b].sum(0))
                out[f"{name}|{cname}|{kq}"] = {
                    "n": nq.tolist(), "observed_lapse": obs.tolist(), "predicted_lapse": prd.tolist(),
                    "top_minus_bottom_observed": float(obs[-1] - obs[0]), "top_minus_bottom_observed_ci": ci(ob[:, -1] - ob[:, 0]),
                    "top_minus_bottom_residual": float((obs[-1] - prd[-1]) - (obs[0] - prd[0])),
                    "top_minus_bottom_residual_ci": ci((ob[:, -1] - pb[:, -1]) - (ob[:, 0] - pb[:, 0])),
                    "oe_ratio": (obs / prd).tolist(), "oe_ratio_ci": [ci(ob[:, q] / pb[:, q]) for q in range(kq)]}
                print(name, cname, kq, "done", flush=True)
    res["tables"] = out
    reg = {}
    for name, rel in [("user", D["rel_u"]), ("card", D["rel_c"])]:
        for cname, pc in [("curve", D["pc"]), ("instant", D["pi"])]:
            m = ~np.isnan(rel)
            lp, r, yy, uu = logit(pc[m]), rel[m], y[m], uid[m]
            one = np.ones(m.sum())
            b0, s0 = irls(np.column_stack([one, lp]), yy, uu)
            b1, s1 = irls(np.column_stack([one, lp, r]), yy, uu)
            import scipy.sparse as sp
            Uc = sp.csr_matrix((np.ones(m.sum()), (np.arange(m.sum()), uu)), shape=(m.sum(), nU))
            b2, s2 = irls(sp.hstack([Uc, sp.csr_matrix(np.column_stack([lp, r]))]).tocsr(), yy, uu)
            reg[f"{name}|{cname}"] = {"n": int(m.sum()), "rel_sd": float(r.std()), "base": {"slope_logit_pred": [float(b0[1]), float(s0[1])]},
                                      "with_rel": {"slope_logit_pred": [float(b1[1]), float(s1[1])], "rel_coef": [float(b1[2]), float(s1[2])]},
                                      "with_rel_user_fe": {"slope_logit_pred": [float(b2[-2]), float(s2[-2])], "rel_coef": [float(b2[-1]), float(s2[-1])]}}
            print(name, cname, "logit done", flush=True)
    res["logit"] = reg
    res["rel_u_quantiles_10_20_50_80_90"] = np.quantile(D["rel_u"], [0.1, 0.2, 0.5, 0.8, 0.9]).tolist()
    (OUT / "slow-good.json").write_text(json.dumps(res, indent=1))
    write_md(res)


def write_md(r):
    T = r["tables"]
    f = lambda x: f"{100 * x:.2f}"
    L = ["# Slow-Good signal (SM88)", "",
         f"150-user stage-2 sample of the 732 set ({r['users']} users), {r['pairs']:,} (Good review j, next review k) pairs: j is Good on a later day than the card's previous review, k is the card's next review on a later day than j. Outcome: k is Again (overall {f(r['lapse_rate'])}%; the model's mean predicted lapse {f(r['pred_lapse_curve'])}% at scheduling time, {f(r['pred_lapse_instant'])}% at review time). Model: the leak-free small model (SM72s) cool-down from epoch 25; scheduling-time prediction = its curve head made at j, evaluated at k's real interval (scheduling-time vs review-time accuracy (SM81) dump, read only); review-time = its instant head at k. Relative time of j = ln(j's time / the user's running median Good time before j) (at least 20 earlier Goods); 'card' = ln(j's time / geometric mean of the card's earlier Good times). Predicted-recall control: 10 buckets of logit(predicted recall), standardised with pooled bucket weights. Intervals: user bootstrap, 2,000 resamples, seed 0.", ""]
    for key, title in [("user|curve|5", "Quintiles of relative time (vs the user's running median), scheduling-time prediction"),
                       ("user|instant|5", "Same, review-time (instant) prediction at k"),
                       ("user_rank|curve|5", "Within-user quintiles of relative time, scheduling-time prediction"),
                       ("card|curve|5", "Quintiles of relative time vs the card's own earlier Good times, scheduling-time prediction"),
                       ("user|curve|3", "Terciles (calibration by time tercile), scheduling-time prediction"),
                       ("user|instant|3", "Terciles, review-time prediction")]:
        t = T[key]; kq = len(t["n"])
        L += [f"## {title}", "", "| Group | pairs | actual lapse % | predicted lapse % | actual / predicted [95% CI] |", "| --- | --- | --- | --- | --- |"]
        for q in range(kq):
            nm = "fastest" if q == 0 else "slowest" if q == kq - 1 else f"middle {q + 1}"
            c = t["oe_ratio_ci"][q]
            L.append(f"| {nm} | {t['n'][q]:,.0f} | {f(t['observed_lapse'][q])} | {f(t['predicted_lapse'][q])} | {t['oe_ratio'][q]:.3f} [{c[0]:.3f}, {c[1]:.3f}] |")
        a, b = t["top_minus_bottom_observed_ci"], t["top_minus_bottom_residual_ci"]
        L += ["", f"Slowest minus fastest at equal predicted recall: actual lapse {100 * t['top_minus_bottom_observed']:+.2f} points [{100 * a[0]:+.2f}, {100 * a[1]:+.2f}]; residual (actual - predicted) {100 * t['top_minus_bottom_residual']:+.2f} points [{100 * b[0]:+.2f}, {100 * b[1]:+.2f}].", ""]
    L += ["## Logistic regression: k Again on logit(predicted recall) + relative time of j", "", "Coefficient (user-clustered robust SE); relative time in ln units.", "",
          "| Reference time | Control | pairs | SD | slope on logit(pred), alone | slope with time | time coef per ln unit | time coef, user fixed effects |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    g = lambda p: f"{p[0]:.3f} ({p[1]:.3f})"
    for k, v in r["logit"].items():
        a, c = k.split("|")
        L.append(f"| {'user running median' if a == 'user' else 'card own Goods'} | {'scheduling-time' if c == 'curve' else 'review-time'} | {v['n']:,} | {v['rel_sd']:.2f} | {g(v['base']['slope_logit_pred'])} | {g(v['with_rel']['slope_logit_pred'])} | {g(v['with_rel']['rel_coef'])} | {g(v['with_rel_user_fe']['rel_coef'])} |")
    L.append("")
    (OUT / "slow-good.md").write_text("\n".join(L))


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "report"])
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    {"prepare": prepare, "report": report}[a.cmd](a)
