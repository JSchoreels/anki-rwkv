#!/usr/bin/env python
"""AUC ceiling (SM182): the read-out (design.md "Metrics" and "Rules"), CPU only, < 4 GB.

  python sm182_report.py report        -> sm182-result.json, sm182-result.md (exit 0 decided, 2 pending)
  python sm182_report.py selftest      -> checks of the AUC / RMSE(bins) / Platt code on synthetic data
  python sm182_report.py dumpcheck --rows <smoke rows> --dump <smoke dump dir>   (the smoke ticket's pipeline check)

Inputs: the scorer rows gpu/sm182-rows-732.jsonl (REF w8+s8hx8 on show-time inputs) and the per-review dump
work/dump/<user>_<tag>_w8+s8hx8@e2s-query.npz (review_th, bin, y, p_ahead, p_instant; never committed). Models: the
control (the SM171 control cool-down, rescored here), arm R (pure ranking; its fallback arm RA if R had to be
replaced, design.md) and arm M (mixed). Heads: curve ("ahead") and instant.

Statistics: paired user bootstrap, 2,000 draws, numpy default_rng(0), 95% percentile intervals, the same draws for
every statistic, on the rule users (a finite row for every model, equal benchmark sizes, a dump of that size).
- LogLoss: % = 100 x (sum over users of LogLoss x reviews of the arm / the same of the control - 1).
- within-user AUC: mean over the users with both outcomes of the per-user AUC (ties averaged); the change is the
  mean of the per-user differences, in AUC units (and in % of the control's mean).
- overall AUC: AUC of all rule users' reviews pooled; the bootstrap uses per-user histograms of logit(p) (16,384 bins
  on [-16, 16]); the point estimate is the exact one (the binned one is reported next to it).
- calibration: RMSE(bins) per user (the benchmark's bins), review-weighted mean; the change in %; plus the pooled
  ECE (20 equal-mass bins) as a reported number.
- causal per-user Platt recalibration (as rolling recalibration on the 732 set (SM125)): per user and head, the rows
  in review order are cut into 7-day blocks from the user's first scored review; each block is mapped through
  sigmoid(a logit(p) + b) fitted (Newton, warm start) on the latest 30,000 scored rows of strictly earlier days;
  the raw model until 1,000 earlier rows, or when the fit rows have fewer than 20 of a class; a in [0.05, 20],
  b in [-15, 15]. Reported for every model: LogLoss, RMSE(bins) and within-user AUC after recalibration, and for R
  the share of its LogLoss gap to the control that the recalibration recovers.
"""
import argparse
import glob
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
G = HERE / "gpu"
ROWS = G / "sm182-rows-732.jsonl"
DUMP = HERE / "work" / "dump"
CACHE = HERE / "work" / "per-user"
SM171_ROWS = Path("C:/Users/jonat/workspace/wt/sm171/artifacts/small-model/sm171/gpu/sm171-rows-732.jsonl")
RUN = Path("C:/Users/jonat/workspace/wt/sm182-run/artifacts/small-model")
CTL_DIR = Path("C:/Users/jonat/workspace/wt/sm171-run/artifacts/small-model/sm171-tiny-ctl")
DATA = Path("C:/Users/jonat/workspace/anki-rwkv/data/anki-revlogs-10k-original-20261011/revlogs")
TRACK = "E89-F65e-F77a"
SUFFIX = "/w8+s8hx8@e2s-query"
CTL = "sm171-tiny-ctl"
HEADS = ("ahead", "instant")
HEAD_NAMES = {"ahead": "curve", "instant": "instant"}
DRAWS, SEED = 2000, 0
MIN_USERS = 720
LL_MARGIN = 0.05            # M: LogLoss not worse by more than 0.05% (CI upper)
HBINS, HLO, HHI = 16384, -16.0, 16.0
ROLL_DAYS, ROLL_ROWS, N_MIN, MIN_CLASS = 7, 30000, 1000, 20
A_RANGE, B_RANGE, EPS = (0.05, 20.0), (-15.0, 15.0), 1e-6
NAMES = {CTL: "control cool-down (SM171 control)", "sm182-tiny-r": "pure ranking (R)",
         "sm182-tiny-ra": "pure ranking with a 2% LogLoss anchor (RA, fallback of R)", "sm182-tiny-m": "mixed (M)"}


# ------------------------------------------------------------------ per-user metrics
def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def auc(y, p):
    """AUC with ties averaged (Mann-Whitney); None without both classes."""
    from scipy.stats import rankdata
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(p)
    return float((r[y > 0.5].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rmse_bins(bins, y, p):
    _, inv, cnt = np.unique(bins, return_inverse=True, return_counts=True)
    my = np.bincount(inv, y) / cnt
    mp = np.bincount(inv, p) / cnt
    return float(np.sqrt(np.sum(cnt * (my - mp) ** 2) / cnt.sum()))


def logloss_sum(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)   # sklearn's log_loss clipping (float64 eps is smaller; rows round to 6 digits)
    return float(-np.sum(y * np.log(p) + (1 - y) * np.log1p(-p)))


def hist(y, p):
    x = np.clip(logit(p), HLO, HHI - 1e-9)
    k = ((x - HLO) / (HHI - HLO) * HBINS).astype(np.int64)
    return (np.bincount(k[y > 0.5], minlength=HBINS).astype(np.float32),
            np.bincount(k[y <= 0.5], minlength=HBINS).astype(np.float32))


def fit_platt(X, Y, w0, iters=50, tol=1e-12):
    """(a, b) minimising the mean LogLoss of sigmoid(a x + b): Newton with step halving (SM123 / SM125's fit_platt,
    one event)."""
    w = np.array(w0, dtype=np.float64)

    def loss(w):
        z = w[0] * X + w[1]
        return float(np.mean(np.logaddexp(0, z) - Y * z))
    cur = loss(w)
    for _ in range(iters):
        z = w[0] * X + w[1]
        q = 0.5 * (1 + np.tanh(0.5 * z))
        r = q * (1 - q)
        g = np.array([np.mean((q - Y) * X), np.mean(q - Y)])
        h11, h12, h22 = np.mean(r * X * X) + 1e-9, np.mean(r * X), np.mean(r) + 1e-9
        det = h11 * h22 - h12 * h12
        step = np.array([(h22 * g[0] - h12 * g[1]) / det, (h11 * g[1] - h12 * g[0]) / det])
        lam = 1.0
        new = w - step
        nl = loss(new)
        while nl > cur + 1e-15 and lam > 1e-4:
            lam /= 2
            new = w - lam * step
            nl = loss(new)
        if nl > cur + 1e-15:
            break
        done = abs(cur - nl) < tol
        w, cur = new, nl
        if done:
            break
    return w


def platt_causal(y, p, day):
    """Per-user causal rolling Platt (module docstring). Returns (recalibrated p, active mask, stats)."""
    assert (np.diff(day) >= 0).all(), "rows are not in time order"
    out = p.astype(np.float64).copy()
    active = np.zeros(len(p), bool)
    X = logit(p.astype(np.float64))
    w = np.array([1.0, 0.0])
    st = {"blocks": 0, "fitted": 0, "clamped": 0}
    if len(p) == 0:
        return out, active, st
    b = (day - day[0]) // ROLL_DAYS
    cuts = np.r_[0, np.flatnonzero(np.diff(b)) + 1, len(day)]
    for i0, i1 in zip(cuts[:-1], cuts[1:]):
        st["blocks"] += 1
        lo = max(0, i0 - ROLL_ROWS)
        n = i0 - lo
        if n < N_MIN:
            continue
        assert day[i0 - 1] < day[i0]
        Yf = y[lo:i0]
        npos = Yf.sum()
        if npos < MIN_CLASS or n - npos < MIN_CLASS:
            continue
        wf = fit_platt(X[lo:i0], Yf, w)
        cl = np.array([np.clip(wf[0], *A_RANGE), np.clip(wf[1], *B_RANGE)])
        st["clamped"] += int((cl != wf).any())
        w = cl
        z = w[0] * X[i0:i1] + w[1]
        out[i0:i1] = 0.5 * (1 + np.tanh(0.5 * z))
        active[i0:i1] = True
        st["fitted"] += 1
    return out, active, st


def user_record(user, tags):
    """Per-user, per-model, per-head statistics from the dump (cached in work/per-user)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{user}.json"
    key = sorted(tags)
    if cache.exists():
        rec = json.loads(cache.read_text())
        if rec.get("tags") == key:
            return rec, np.load(CACHE / f"{user}.npz")
    day_all = None
    rec = {"tags": key, "models": {}}
    arrays = {}
    for tag in tags:
        files = glob.glob(str(DUMP / f"{user}_{tag}_w8+s8hx8@e2s-query.npz"))
        if len(files) != 1:
            return None, None
        z = np.load(files[0])
        th, bins, y = z["review_th"], z["bin"], z["y"].astype(np.float64)
        order = np.argsort(th, kind="stable")
        th, bins, y = th[order], bins[order], y[order]
        if day_all is None:
            import pandas as pd
            day_all = pd.read_parquet(DATA / f"user_id={user}", columns=["day_offset"])["day_offset"].to_numpy()
            rec["n"] = int(len(th))
            rec["th_sha"] = int(np.sum(th * 2654435761 % 1000003))
            rec["y_sum"] = float(y.sum())
        elif len(th) != rec["n"] or int(np.sum(th * 2654435761 % 1000003)) != rec["th_sha"] or float(y.sum()) != rec["y_sum"]:
            rec["mismatch"] = True
            return rec, None
        day = day_all[th - 1].astype(np.int64)
        m = {}
        for h in HEADS:
            p = z[f"p_{h}"].astype(np.float64)[order]
            cal, active, st = platt_causal(y, p, day)
            hp, hn = hist(y, p)
            arrays[f"{tag}|{h}|pos"], arrays[f"{tag}|{h}|neg"] = hp, hn
            m[h] = {"ll_sum": logloss_sum(y, p), "auc": auc(y, p), "rmse_bins": rmse_bins(bins, y, p),
                    "cal_ll_sum": logloss_sum(y, cal), "cal_auc": auc(y, cal), "cal_rmse_bins": rmse_bins(bins, y, cal),
                    "active": int(active.sum()), "platt": st}
        rec["models"][tag] = m
    rec["day_sorted"] = True
    np.savez(CACHE / f"{user}.npz", **arrays)
    cache.write_text(json.dumps(rec), encoding="utf-8")
    return rec, np.load(CACHE / f"{user}.npz")


# ------------------------------------------------------------------ read-out
def last_rows(path):
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "user_id" in r and "label" in r:
            out[(r["user_id"], r["label"])] = r
    return out


def ci(v):
    lo, hi = np.percentile(v, [2.5, 97.5])
    return [float(lo), float(hi)]


def pooled_auc_hist(P, N):
    """AUC from pooled histograms P, N [..., bins] (ties within a bin count one half)."""
    below = np.cumsum(N, axis=-1) - N
    return np.sum(P * (below + 0.5 * N), axis=-1) / (P.sum(-1) * N.sum(-1))


def pooled_auc_exact(ys, ps):
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return auc(y, p)


def pooled_ece(ys, ps, bins=20):
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    order = np.argsort(p, kind="stable")
    parts = np.array_split(order, bins)
    return float(sum(len(ix) * abs(y[ix].mean() - p[ix].mean()) for ix in parts) / len(y))


def training_stats(folder):
    p = folder / "expertium" / TRACK / "result.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    return {k: r.get(k) for k in ("status", "review_presentations", "training_seconds", "reviews_per_second",
                                  "peak_allocated_mib", "peak_reserved_mib", "pair_rank")}


def report(args):
    rows = last_rows(ROWS)
    r_tag = "sm182-tiny-ra" if (RUN / "sm182-tiny-ra" / "USED").exists() else "sm182-tiny-r"
    tags = [CTL, r_tag, "sm182-tiny-m"]
    pending = []
    # integrity: the control rescored here vs SM171's rows of the same model
    old = last_rows(SM171_ROWS)
    diffs = []
    for (u, lab), r in rows.items():
        if lab == CTL + SUFFIX and "benchmark" in r and (u, lab) in old and "benchmark" in old[(u, lab)]:
            for h in HEADS:
                diffs.append(abs(r["benchmark"][h]["LogLoss"] - old[(u, lab)]["benchmark"][h]["LogLoss"]))
    integrity = {"users": len(diffs) // 2, "max_abs_logloss_diff": max(diffs) if diffs else None}
    integrity["ok"] = bool(diffs) and max(diffs) <= 1e-6 and integrity["users"] >= MIN_USERS
    if not integrity["ok"]:
        pending.append(f"integrity vs SM171's rows failed or too few users: {integrity}")
    users = sorted({u for (u, _) in rows})
    rule, nonfinite, excluded = [], 0, {}
    for u in users:
        rs = [rows.get((u, t + SUFFIX)) for t in tags]
        if any(r is None for r in rs):
            excluded[u] = "missing row"
            continue
        if any("benchmark" not in r for r in rs):
            nonfinite += any("nonfinite" in r for r in rs)
            excluded[u] = "no benchmark (non-finite or error)"
            continue
        sizes = {r["benchmark"]["size"] for r in rs}
        if len(sizes) != 1:
            excluded[u] = "benchmark sizes differ"
            continue
        rule.append(u)
    recs, hists, bad_dump = {}, {}, []
    for k, u in enumerate(rule):
        rec, arr = user_record(u, tags)
        if rec is None or arr is None or rec["n"] != rows[(u, CTL + SUFFIX)]["benchmark"]["size"]:
            bad_dump.append(u)
            continue
        recs[u] = rec
        hists[u] = {k2: arr[k2] for k2 in arr.files}
        if k % 50 == 0:
            print(f"[sm182-report] {k}/{len(rule)} users", flush=True)
    rule = [u for u in rule if u in recs]
    if len(rule) < MIN_USERS:
        pending.append(f"{len(rule)} rule users (< {MIN_USERS})")
    if nonfinite:
        pending.append(f"{nonfinite} users with a non-finite row (left out)")
    # cross-check: our per-user LogLoss / AUC / RMSE(bins) vs the scorer's rows
    xmax = {"LogLoss": 0.0, "AUC": 0.0, "RMSE(bins)": 0.0}
    for u in rule:
        n = recs[u]["n"]
        for t in tags:
            b = rows[(u, t + SUFFIX)]["benchmark"]
            for h in HEADS:
                m = recs[u]["models"][t][h]
                xmax["LogLoss"] = max(xmax["LogLoss"], abs(m["ll_sum"] / n - b[h]["LogLoss"]))
                xmax["RMSE(bins)"] = max(xmax["RMSE(bins)"], abs(m["rmse_bins"] - b[h]["RMSE(bins)"]))
                if m["auc"] is not None and b[h]["AUC"] is not None:
                    xmax["AUC"] = max(xmax["AUC"], abs(m["auc"] - b[h]["AUC"]))
    cross_ok = xmax["LogLoss"] <= 2e-6 and xmax["RMSE(bins)"] <= 2e-6 and xmax["AUC"] <= 2e-6
    if not cross_ok:
        pending.append(f"dump does not reproduce the scorer's rows: {xmax}")
    U = len(rule)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, U, size=(DRAWS, U))
    W = np.stack([np.bincount(d, minlength=U) for d in draws]).astype(np.float64)   # [D, U] user weights
    n = np.array([recs[u]["n"] for u in rule], dtype=np.float64)
    res = {"rule_users": U, "excluded": {str(k): v for k, v in excluded.items()}, "bad_dump": bad_dump,
           "integrity": integrity, "dump_vs_rows_max_abs": xmax, "models": {t: NAMES[t] for t in tags},
           "arms": {}, "absolute": {}, "training": {}}

    def arr(t, h, key):
        return np.array([recs[u]["models"][t][h][key] if recs[u]["models"][t][h][key] is not None else np.nan
                         for u in rule], dtype=np.float64)

    for t in tags:
        res["absolute"][t] = {}
        for h in HEADS:
            ll, cl = arr(t, h, "ll_sum"), arr(t, h, "cal_ll_sum")
            a, ca = arr(t, h, "auc"), arr(t, h, "cal_auc")
            rb, crb = arr(t, h, "rmse_bins"), arr(t, h, "cal_rmse_bins")
            ok = ~np.isnan(a)
            P = np.stack([hists[u][f"{t}|{h}|pos"] for u in rule]).astype(np.float64)
            N = np.stack([hists[u][f"{t}|{h}|neg"] for u in rule]).astype(np.float64)
            res["absolute"][t][h] = {"LogLoss": float(ll.sum() / n.sum()), "LogLoss_platt": float(cl.sum() / n.sum()),
                                     "within_user_AUC": float(a[ok].mean()), "within_user_AUC_platt": float(ca[ok].mean()),
                                     "AUC_users": int(ok.sum()), "overall_AUC_binned": float(pooled_auc_hist(P.sum(0), N.sum(0))),
                                     "RMSE_bins": float((rb * n).sum() / n.sum()),
                                     "RMSE_bins_platt": float((crb * n).sum() / n.sum()),
                                     "platt_active_share": float(sum(recs[u]["models"][t][h]["active"] for u in rule) / n.sum()),
                                     "platt_clamped_blocks": int(sum(recs[u]["models"][t][h]["platt"]["clamped"] for u in rule))}
    # exact pooled AUC and ECE (one model x head at a time; reloads the dump)
    for t in tags:
        for h in HEADS:
            ys, ps = [], []
            for u in rule:
                z = np.load(DUMP / f"{u}_{t}_w8+s8hx8@e2s-query.npz")
                ys.append(z["y"].astype(np.float64))
                ps.append(z[f"p_{h}"].astype(np.float64))
            res["absolute"][t][h]["overall_AUC"] = pooled_auc_exact(ys, ps)
            res["absolute"][t][h]["ECE_pooled_20"] = pooled_ece(ys, ps)
            del ys, ps
    # paired comparisons vs the control
    for t in tags[1:]:
        res["arms"][t] = {}
        for h in HEADS:
            d = {}
            ll_a, ll_c = arr(t, h, "ll_sum"), arr(CTL, h, "ll_sum")
            pct = 100 * ((W @ ll_a) / (W @ ll_c) - 1)
            d["LogLoss_pct"] = {"point": float(100 * (ll_a.sum() / ll_c.sum() - 1)), "ci": ci(pct)}
            a_a, a_c = arr(t, h, "auc"), arr(CTL, h, "auc")
            ok = ~np.isnan(a_a) & ~np.isnan(a_c)
            Wk = W[:, ok]
            diff = a_a[ok] - a_c[ok]
            bd = (Wk @ diff) / Wk.sum(1)
            bc = (Wk @ a_c[ok]) / Wk.sum(1)
            d["within_user_AUC_diff"] = {"point": float(diff.mean()), "ci": ci(bd), "users": int(ok.sum())}
            d["within_user_AUC_pct"] = {"point": float(100 * diff.mean() / a_c[ok].mean()), "ci": ci(100 * bd / bc)}
            Pa = np.stack([hists[u][f"{t}|{h}|pos"] for u in rule]).astype(np.float64)
            Na = np.stack([hists[u][f"{t}|{h}|neg"] for u in rule]).astype(np.float64)
            Pc = np.stack([hists[u][f"{CTL}|{h}|pos"] for u in rule]).astype(np.float64)
            Nc = np.stack([hists[u][f"{CTL}|{h}|neg"] for u in rule]).astype(np.float64)
            bo = []
            for s in range(0, DRAWS, 100):
                Ws = W[s:s + 100]
                bo.append(pooled_auc_hist(Ws @ Pa, Ws @ Na) - pooled_auc_hist(Ws @ Pc, Ws @ Nc))
            bo = np.concatenate(bo)
            d["overall_AUC_diff"] = {"point": res["absolute"][t][h]["overall_AUC"] - res["absolute"][CTL][h]["overall_AUC"],
                                     "point_binned": float(pooled_auc_hist(Pa.sum(0), Na.sum(0)) - pooled_auc_hist(Pc.sum(0), Nc.sum(0))),
                                     "ci": ci(bo)}
            rb_a, rb_c = arr(t, h, "rmse_bins") * n, arr(CTL, h, "rmse_bins") * n
            d["RMSE_bins_pct"] = {"point": float(100 * (rb_a.sum() / rb_c.sum() - 1)), "ci": ci(100 * ((W @ rb_a) / (W @ rb_c) - 1))}
            # Platt: the arm recalibrated vs the control raw and vs the control recalibrated; R: gap recovered
            cl_a, cl_c = arr(t, h, "cal_ll_sum"), arr(CTL, h, "cal_ll_sum")
            d["platt_LogLoss_pct_vs_control_raw"] = {"point": float(100 * (cl_a.sum() / ll_c.sum() - 1)),
                                                     "ci": ci(100 * ((W @ cl_a) / (W @ ll_c) - 1))}
            d["platt_LogLoss_pct_vs_control_platt"] = {"point": float(100 * (cl_a.sum() / cl_c.sum() - 1)),
                                                       "ci": ci(100 * ((W @ cl_a) / (W @ cl_c) - 1))}
            gap, rec_ = W @ ll_a - W @ ll_c, W @ ll_a - W @ cl_a
            d["platt_gap_recovered_share"] = {"point": float((ll_a.sum() - cl_a.sum()) / (ll_a.sum() - ll_c.sum()))
                                              if ll_a.sum() != ll_c.sum() else None,
                                              "ci": ci(rec_ / gap) if np.all(gap != 0) else None}
            ca_a, ca_c = arr(t, h, "cal_auc"), arr(CTL, h, "cal_auc")
            okc = ~np.isnan(ca_a) & ~np.isnan(ca_c)
            d["platt_within_user_AUC_diff"] = {"point": float((ca_a[okc] - ca_c[okc]).mean()),
                                               "ci": ci((W[:, okc] @ (ca_a[okc] - ca_c[okc])) / W[:, okc].sum(1))}
            res["arms"][t][h] = d
    # rule for M (pre-registered)
    m = res["arms"]["sm182-tiny-m"]
    per_head = {h: {"auc_up": m[h]["within_user_AUC_diff"]["ci"][0] > 0,
                    "logloss_ok": m[h]["LogLoss_pct"]["ci"][1] <= LL_MARGIN} for h in HEADS}
    useful = any(v["auc_up"] for v in per_head.values()) and all(v["logloss_ok"] for v in per_head.values())
    res["rule_M"] = {"per_head": per_head, "useful": useful}
    for t, f in ((CTL, CTL_DIR), (tags[1], RUN / tags[1]), ("sm182-tiny-m", RUN / "sm182-tiny-m")):
        res["training"][t] = training_stats(f)
    res["pending"] = pending
    res["verdict"] = "pending" if pending else ("USEFUL" if useful else "NOT USEFUL")
    (HERE / "sm182-result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    write_md(res, tags)
    print(f"[sm182-report] verdict {res['verdict']} ({U} rule users)")
    return 2 if pending else 0


def write_md(res, tags):
    f = lambda x, d=3: "n/a" if x is None else f"{x:+.{d}f}"   # noqa: E731
    c = lambda v, d=3: "n/a" if v is None else f"[{v[0]:+.{d}f}, {v[1]:+.{d}f}]"   # noqa: E731
    L = ["# AUC ceiling (SM182): result tables", "",
         f"Rule users: {res['rule_users']} of the 732 set (show-time inputs, int8 weights and states, REF). "
         f"Integrity vs SM171's rows: {res['integrity']}. Dump vs rows: {res['dump_vs_rows_max_abs']}.", "",
         f"**Verdict for M (mixed): {res['verdict']}** (useful = within-user AUC up with the CI above 0 on a head, "
         f"and LogLoss CI upper <= +{LL_MARGIN}% on both heads; R is a ceiling measurement, no verdict).", "",
         "## Absolute values", "",
         "| model | head | LogLoss | within-user AUC | overall AUC | RMSE(bins) | ECE (20 bins) | LogLoss after Platt | "
         "within-user AUC after Platt | RMSE(bins) after Platt |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t in tags:
        for h in HEADS:
            a = res["absolute"][t][h]
            L.append(f"| {NAMES[t]} | {HEAD_NAMES[h]} | {a['LogLoss']:.5f} | {a['within_user_AUC']:.4f} | "
                     f"{a['overall_AUC']:.4f} | {a['RMSE_bins']:.4f} | {a['ECE_pooled_20']:.4f} | {a['LogLoss_platt']:.5f} | "
                     f"{a['within_user_AUC_platt']:.4f} | {a['RMSE_bins_platt']:.4f} |")
    L += ["", "## Against the control (paired user bootstrap, 95% CI)", "",
          "| arm | head | LogLoss % | within-user AUC change | within-user AUC % | overall AUC change | RMSE(bins) % | "
          "after Platt: LogLoss % vs control raw | vs control after Platt | gap recovered by Platt |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for t in tags[1:]:
        for h in HEADS:
            d = res["arms"][t][h]
            g = d["platt_gap_recovered_share"]
            L.append(f"| {NAMES[t]} | {HEAD_NAMES[h]} | {f(d['LogLoss_pct']['point'])}% {c(d['LogLoss_pct']['ci'])} | "
                     f"{f(d['within_user_AUC_diff']['point'], 4)} {c(d['within_user_AUC_diff']['ci'], 4)} | "
                     f"{f(d['within_user_AUC_pct']['point'], 2)}% {c(d['within_user_AUC_pct']['ci'], 2)} | "
                     f"{f(d['overall_AUC_diff']['point'], 4)} {c(d['overall_AUC_diff']['ci'], 4)} | "
                     f"{f(d['RMSE_bins_pct']['point'], 1)}% {c(d['RMSE_bins_pct']['ci'], 1)} | "
                     f"{f(d['platt_LogLoss_pct_vs_control_raw']['point'])}% {c(d['platt_LogLoss_pct_vs_control_raw']['ci'])} | "
                     f"{f(d['platt_LogLoss_pct_vs_control_platt']['point'])}% {c(d['platt_LogLoss_pct_vs_control_platt']['ci'])} | "
                     f"{f(g['point'], 2)} {c(g['ci'], 2)} |")
    L += ["", "## Training", "", "| model | reviews/s | minutes | peak allocated / reserved MiB |", "|---|---|---|---|"]
    for t, s in res["training"].items():
        if s:
            L.append(f"| {NAMES[t]} | {s['reviews_per_second'] or 0:,.0f} | {(s['training_seconds'] or 0) / 60:.1f} | "
                     f"{s['peak_allocated_mib'] or 0:,.0f} / {s['peak_reserved_mib'] or 0:,.0f} |")
    if res["pending"]:
        L += ["", "Pending: " + "; ".join(res["pending"])]
    (HERE / "sm182-result.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ selftest
def selftest(args):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(182)
    out = {}
    y = (rng.random(5000) < 0.8).astype(np.float64)
    p = np.clip(1 / (1 + np.exp(-(rng.normal(0, 1, 5000) + 1.5 * y))), 1e-4, 1 - 1e-4)
    p = np.round(p, 3)   # ties
    out["auc_vs_sklearn"] = abs(auc(y, p) - roc_auc_score(y, p))
    P, N = hist(y, p)
    out["hist_auc_vs_exact"] = abs(float(pooled_auc_hist(P.astype(np.float64), N.astype(np.float64))) - auc(y, p))
    # Platt recovers a known miscalibration: truth sigmoid(1.6 logit q + 0.4), model says q
    day = np.sort(rng.integers(0, 400, 40000))
    q = 1 / (1 + np.exp(-rng.normal(1.5, 1.2, 40000)))
    yt = (rng.random(40000) < 1 / (1 + np.exp(-(1.6 * logit(q) + 0.4)))).astype(np.float64)
    cal, active, st = platt_causal(yt, q, day)
    out["platt_logloss_raw"] = logloss_sum(yt[active], q[active]) / active.sum()
    out["platt_logloss_cal"] = logloss_sum(yt[active], cal[active]) / active.sum()
    out["platt_active_share"] = float(active.mean())
    # causality: flipping labels from a day D on leaves earlier recalibrated values unchanged
    D = int(np.quantile(day, 0.6))
    y2 = yt.copy()
    y2[day >= D] = 1 - y2[day >= D]
    cal2, _, _ = platt_causal(y2, q, day)
    out["platt_causal"] = bool(np.array_equal(cal[day < D], cal2[day < D]) and not np.array_equal(cal, cal2))
    out["rmse_bins_zero_when_calibrated"] = rmse_bins(np.zeros(4), np.array([1, 0, 1, 0.0]), np.full(4, 0.5))
    out["ok"] = (out["auc_vs_sklearn"] < 1e-12 and out["hist_auc_vs_exact"] < 1e-3 and out["platt_causal"]
                 and out["platt_logloss_cal"] < out["platt_logloss_raw"] - 0.005
                 and out["rmse_bins_zero_when_calibrated"] == 0.0)
    (G / "selftest-report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out))
    return 0 if out["ok"] else 1


def dumpcheck(args):
    """The smoke ticket's dump of the control (3 smoke users): the per-user code reproduces the scorer's rows
    (LogLoss, AUC, RMSE(bins) per head), the rows equal SM171's rows, review order and days are sorted, Platt runs."""
    global DUMP, CACHE
    DUMP, CACHE = Path(args.dump), Path(args.dump) / "per-user"
    rows = last_rows(Path(args.rows))
    old = last_rows(SM171_ROWS)
    out, ok = {}, True
    for (u, lab), r in sorted(rows.items()):
        if lab != CTL + SUFFIX or "benchmark" not in r:
            continue
        rec, _ = user_record(u, [CTL])
        b, o = r["benchmark"], old[(u, lab)]["benchmark"]
        d = {"reviews": rec["n"], "size": b["size"]}
        for h in HEADS:
            m = rec["models"][CTL][h]
            d[h] = {"logloss": abs(m["ll_sum"] / rec["n"] - b[h]["LogLoss"]), "auc": abs(m["auc"] - b[h]["AUC"]),
                    "rmse_bins": abs(m["rmse_bins"] - b[h]["RMSE(bins)"]), "vs_sm171": abs(b[h]["LogLoss"] - o[h]["LogLoss"]),
                    "platt_active": m["active"], "platt_logloss": m["cal_ll_sum"] / rec["n"], "raw_logloss": b[h]["LogLoss"]}
            ok = ok and d[h]["logloss"] <= 2e-6 and d[h]["auc"] <= 2e-6 and d[h]["rmse_bins"] <= 2e-6 and d[h]["vs_sm171"] <= 1e-6
        ok = ok and rec["n"] == b["size"]
        out[u] = d
    ok = ok and len(out) == 3
    print(json.dumps({"users": out, "ok": ok}))
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["report", "selftest", "dumpcheck"])
    p.add_argument("--rows", default="")
    p.add_argument("--dump", default="")
    args = p.parse_args()
    G.mkdir(parents=True, exist_ok=True)
    return {"report": report, "selftest": selftest, "dumpcheck": dumpcheck}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
