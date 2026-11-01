#!/usr/bin/env python
"""Time to fluency (SM84): an extension of response-time prediction (SM82) stage 2 (CPU only).

Fluency = a card's first Good answered fast on a later-day review: under 3.9 s (the answer-time head's bins 0-6, ending
at 3,891 ms), also under 5.5 s (bins 0-7, 5,477 ms) and "two fast Goods in a row" (two consecutive later-day reviews).

Per user (the 150 users of stage 2), an FSRS-6 card simulation (the user's own FSRS-6 fit, stage 2's formulas) of
cards learned on day 0 and followed for 365 days, no daily limit (cards are independent then), under scheduling
policies: one desired retention 0.50-0.98, or a fixed interval-growth factor (next interval = previous x g, 1 day after
learning and after a lapse). On every simulated Good the card is fast with the user's P(fast | Good, recall):
  model 3.9 s   the head's P(under 3.9 s | Good) over the head's recall, binned over every curve-grid point (stage 2's
                curve queries, work/curves/users/*.npz of the sm82s2 worktree)
  flat 3.9 s    the same head, one constant per user (no recall dependence): a sensitivity bound
  observed 3.9 s / observed 5.5 s   the user's own recorded Good answers at the real intervals, binned by the head's
                predicted recall (the stored head outputs stop at bin 6, so 5.5 s has no model curve)
Costs: B (the head's per-grade cost curve over recall, fixed pass mix) and C (B + the grade head's pass mix over recall),
plus A (head) seconds on the B trajectory. Observed reference: cards first seen in the user's revlog with 365 days of
follow-up, counted the same way on the recorded reviews.

  run  --curves DIR --out-dir artifacts/small-model/sm84 --procs 6
"""
from argparse import ArgumentParser
import json
from multiprocessing import Pool
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sm82s2_analyze import (COST_BINS, bucket, cost_model, fsrs_w, load_user, user_fits, _low,  # noqa: E402
                                    md_table, fmt)

FAST_MS = 500.0 * 120.0 ** (6 / 14)     # 3,891 ms, bins 0-6
FAST55_MS = 500.0 * 120.0 ** (7 / 14)   # 5,477 ms, bins 0-7
DAYS = 365
N_CARDS = 2000
DRS = [round(0.50 + 0.02 * i, 2) for i in range(25)]          # 0.50 .. 0.98
GROWTH = [1.3, 1.7, 2.0, 2.5, 3.0, 4.0]
CURVES = ["model39", "flat39", "obs39", "obs55"]
CURVE_NAMES = {"model39": "model, under 3.9 s", "flat39": "model flat, under 3.9 s", "obs39": "observed, under 3.9 s",
               "obs55": "observed, under 5.5 s", "model39p": "model persistent, under 3.9 s",
               "model39_two": "two in a row (model)", "model39p_two": "two in a row (model persistent)"}
DEFS = CURVES + ["model39p", "model39_two", "model39p_two"]   # p = persistent (the head's after-a-fast-Good ratio); _two = two fast Goods in a row
OBS_NAMES = {"obs39": "under 3.9 s", "obs55": "under 5.5 s", "model39_two": "two in a row under 3.9 s"}
SHOW_DRS = [0.50, 0.60, 0.70, 0.80, 0.86, 0.90, 0.94, 0.98]


# ------------------------------------------------------------------ per-user fast curves
def fast_curves(z):
    """P(fast | Good) over recall: model (grid points, P(Good)-weighted), flat model constant, observed (Goods at
    the real interval binned by the head's predicted recall)."""
    pg = z["grid_pg"].reshape(-1, 4).astype(np.float64); pf = z["grid_pf"].reshape(-1, 4).astype(np.float64)
    r = 1 - pg[:, 0]; b = bucket(r, COST_BINS)
    xm, ym = [], []
    for k in range(len(COST_BINS) - 1):
        m = b == k
        if m.sum() >= 20:
            wg = pg[m, 2]
            xm.append(r[m].mean()); ym.append((wg * pf[m, 2]).sum() / max(wg.sum(), 1e-12))
    fpg = z["full_pg"].astype(np.float64); fpf = z["full_pf"].astype(np.float64)
    flat = float((fpg[:, 2] * fpf[:, 2]).sum() / max(fpg[:, 2].sum(), 1e-12))
    good = z["grade"] == 3
    fr = 1 - fpg[good, 0]; dur = z["dur"][good]
    fb = bucket(fr, COST_BINS)
    xo, y39, y55 = [], [], []
    for k in range(len(COST_BINS) - 1):
        m = fb == k
        if m.sum() >= 20:
            xo.append(fr[m].mean()); y39.append(((dur[m] > 0) & (dur[m] < FAST_MS)).mean()); y55.append(((dur[m] > 0) & (dur[m] < FAST55_MS)).mean())
    o39 = float(((dur > 0) & (dur < FAST_MS)).mean()) if good.any() else flat
    o55 = float(((dur > 0) & (dur < FAST55_MS)).mean()) if good.any() else flat
    if not xo:
        xo, y39, y55 = [0.9], [o39], [o55]
    if not xm:
        xm, ym = [0.9], [flat]
    return {"model39": (np.asarray(xm), np.asarray(ym)), "flat39": (np.array([0.9]), np.array([flat])),
            "obs39": (np.asarray(xo), np.asarray(y39)), "obs55": (np.asarray(xo), np.maximum(np.asarray(y55), np.asarray(y39))),
            "summary": {"model39_at": {f"{p:.2f}": float(np.interp(p, xm, ym)) for p in (0.6, 0.7, 0.8, 0.9, 0.95, 0.98)},
                        "obs39_at": {f"{p:.2f}": float(np.interp(p, xo, y39)) for p in (0.6, 0.7, 0.8, 0.9, 0.95, 0.98)},
                        "obs55_at": {f"{p:.2f}": float(np.interp(p, xo, y55)) for p in (0.6, 0.7, 0.8, 0.9, 0.95, 0.98)},
                        "flat39": flat, "obs39_overall": o39, "obs55_overall": o55}}


# ------------------------------------------------------------------ observed fluency on the recorded reviews
def observed_fluency(user, data, cap_ms, z):
    import pandas as pd
    rev = pd.read_parquet(Path(data) / "revlogs" / f"user_id={user}", columns=["card_id", "day_offset", "rating", "duration", "elapsed_days"])
    rev = rev.reset_index(drop=True)
    card = rev["card_id"].to_numpy(); day = rev["day_offset"].to_numpy(np.int64); rating = rev["rating"].to_numpy()
    dur = rev["duration"].to_numpy(np.float64); el = rev["elapsed_days"].to_numpy()
    sec = np.minimum(dur, cap_ms) / 1000.0
    later = el >= 1
    f39 = later & (rating == 3) & (dur > 0) & (dur < FAST_MS)
    f55 = later & (rating == 3) & (dur > 0) & (dur < FAST55_MS)
    df = pd.DataFrame({"card": card, "day": day, "el": el, "later": later.astype(np.int64), "f39": f39, "f55": f55, "sec": sec})
    g = df.groupby("card", sort=False)
    first_day = g["day"].transform("first").to_numpy(); first_el = g["el"].transform("first").to_numpy()
    nrev = g["later"].cumsum().to_numpy(); cum = g["sec"].cumsum().to_numpy()
    # previous later-day review's fast-Good flag (per card, later-day rows only)
    lat = df[later]
    prev39 = np.zeros(len(df), bool)
    prev39[lat.index.to_numpy()] = lat.groupby("card", sort=False)["f39"].shift(1).fillna(False).to_numpy(bool)
    two = f39 & prev39
    eligible = (first_el == -1) & (first_day <= day.max() - DAYS)
    within = (day - first_day) <= DAYS
    out = {}
    cards = np.unique(card[eligible])
    n_cards = len(cards)
    # total seconds per eligible card within the first 365 days
    m_in = eligible & within
    tot = pd.Series(sec[m_in]).groupby(card[m_in]).sum()
    for name, flag in (("obs39", f39), ("obs55", f55), ("model39_two", two)):
        hit = eligible & within & flag
        h = pd.DataFrame({"card": card[hit], "n": nrev[hit], "s": cum[hit]}).groupby("card", sort=False).head(1)
        k = len(h)
        spent = tot.copy()
        spent.loc[h["card"].to_numpy()] = h["s"].to_numpy()
        out[name] = {"cards": n_cards, "fluent": k, "share": k / n_cards if n_cards else np.nan,
                     "reviews_mean": float(h["n"].mean()) if k else np.nan, "reviews_median": float(h["n"].median()) if k else np.nan,
                     "seconds_mean": float(h["s"].mean()) if k else np.nan,
                     "seconds_per_fluent": float(spent.sum() / k) if k else np.nan}
    # persistence: observed P(fast Good) after a fast Good vs after not, and the head's prediction on test rows
    lr = later & (np.r_[False, np.ones(len(df) - 1, bool)])
    has_prev = np.zeros(len(df), bool)
    has_prev[lat.index.to_numpy()] = lat.groupby("card", sort=False).cumcount().to_numpy() > 0
    a = has_prev & prev39; b = has_prev & ~prev39
    i = z["th"].astype(np.int64) - 1
    pfg = (z["full_pg"][:, 2] * z["full_pf"][:, 2]).astype(np.float64)
    ta, tb = a[i], b[i]
    out["persistence"] = {"obs_after_fast": float(f39[a].mean()) if a.any() else np.nan, "obs_after_other": float(f39[b].mean()) if b.any() else np.nan,
                          "n_after_fast": int(a.sum()),
                          "head_after_fast": float(pfg[ta].mean()) if ta.any() else np.nan, "head_after_other": float(pfg[tb].mean()) if tb.any() else np.nan,
                          "test_obs_after_fast": float(((z["grade"] == 3) & (z["dur"] > 0) & (z["dur"] < FAST_MS))[ta].mean()) if ta.any() else np.nan,
                          "test_n_after_fast": int(ta.sum()),
                          "head_pf_after_fast": wmean(z, ta), "head_pf_after_other": wmean(z, tb), "head_pf_has_prev": wmean(z, ta | tb)}
    out["pass_rate"] = float((rating[later] > 1).mean()) if later.any() else np.nan
    del lr
    return out


def wmean(z, m):
    """The head's P(under 3.9 s | Good) averaged over test rows m, weighted by P(Good)."""
    wg = z["full_pg"][m, 2].astype(np.float64)
    return float((wg * z["full_pf"][m, 2]).sum() / wg.sum()) if m.any() and wg.sum() > 0 else np.nan


# ------------------------------------------------------------------ card simulation
def card_sim(cm, fc, w, policy, r_mix, n=N_CARDS, days=DAYS, seed=0):
    """Cards learned on day 0, followed `days` days with no daily limit (independent cards). policy = ("dr", r) or
    ("growth", g). Returns per fluency definition: share fluent by `days`, reviews / seconds to fluency, seconds per
    fluent card; plus memorized (mean recall at `days`), reviews and seconds per card."""
    decay = -w[20]; factor = 0.9 ** (1 / decay) - 1
    rng = np.random.default_rng(seed)
    init_d = lambda g: w[4] - np.exp(w[5] * (g - 1)) + 1

    def next_d(d, g):
        nd = d + (-w[6] * (g - 3)) * (10 - d) / 9
        nd = w[7] * init_d(4) + (1 - w[7]) * nd
        return np.clip(nd, 1, 10)

    kind, val = policy
    ivl_dr = lambda s: np.maximum(1, np.round(s / factor * (val ** (1 / decay) - 1)))
    g0 = rng.choice(4, n, p=cm["first_probs"]) + 1
    S = w[g0 - 1].astype(np.float64).copy(); D = np.clip(init_d(g0), 1, 10).astype(np.float64)
    last = np.zeros(n); cur = ivl_dr(S) if kind == "dr" else np.ones(n)
    nxt = last + cur
    secB = np.full(n, cm["learn"]); secA = np.full(n, cm["learn"]); nrev = np.zeros(n, np.int64)
    fl_day = {k: np.full(n, np.inf) for k in DEFS}; fl_rev = {k: np.zeros(n) for k in DEFS}; fl_sec = {k: np.zeros(n) for k in DEFS}
    fl_secA = {k: np.zeros(n) for k in DEFS}
    prevfast = np.zeros(n, bool); prevfastp = np.zeros(n, bool)
    pa, pb = fc["persist"]
    xs, cs, mr = cm["x"], cm["c"], cm["mix_r"]
    curves = {k: fc[k] for k in CURVES}
    reviews_at_R = []
    while True:
        i = np.nonzero(nxt <= days)[0]
        if not i.size:
            break
        s, d = S[i], D[i]
        R = (1 + factor * (nxt[i] - last[i]) / s) ** decay
        u = rng.random(i.size); v = rng.random(i.size); f = rng.random(i.size)
        forget = u > R
        if r_mix and len(xs):
            m = np.stack([np.interp(R, xs, mr[:, j]) for j in range(3)], -1); m /= m.sum(-1, keepdims=True)
        else:
            m = np.broadcast_to(cm["mix"], (i.size, 3))
        cum = np.cumsum(m, -1)
        g = np.where(forget, 1, 2 + (v[:, None] > cum[:, :2]).sum(-1))
        cg = np.stack([np.interp(R, xs, cs[:, j]) for j in range(4)], -1) if len(xs) else np.broadcast_to(cm["fixed_head"], (i.size, 4))
        c = cg[np.arange(i.size), g - 1] + np.where(forget, cm["relearn"], 0.0)
        ca = cm["fixed_head"][g - 1] + np.where(forget, cm["relearn"], 0.0)
        secB[i] += c; secA[i] += ca; nrev[i] += 1
        good = g == 3
        fast = {k: good & (f < np.interp(R, *curves[k])) for k in CURVES}
        p0 = np.interp(R, *curves["model39"])
        fast["model39p"] = good & (f < np.clip(p0 * np.where(prevfastp[i], pa, pb), 0, 1))
        fast["model39_two"] = fast["model39"] & prevfast[i]
        fast["model39p_two"] = fast["model39p"] & prevfastp[i]
        prevfast[i] = fast["model39"]; prevfastp[i] = fast["model39p"]
        for k in DEFS:
            new = fast[k] & np.isinf(fl_day[k][i])
            j = i[new]
            fl_day[k][j] = nxt[j]; fl_rev[k][j] = nrev[j]; fl_sec[k][j] = secB[j]; fl_secA[k][j] = secA[j]
        reviews_at_R.append(R)
        hard = np.where(g == 2, w[15], 1.0); easy = np.where(g == 4, w[16], 1.0)
        s_ok = s * (1 + np.exp(w[8]) * (11 - d) * s ** (-w[9]) * (np.exp((1 - R) * w[10]) - 1) * hard * easy)
        s_fail = np.minimum(w[11] * d ** (-w[12]) * ((s + 1) ** w[13] - 1) * np.exp((1 - R) * w[14]), s / np.exp(w[17] * w[18]))
        s_fail = s_fail * np.maximum(np.exp(w[17] * w[18]) * np.maximum(s_fail, 1e-3) ** (-w[19]), 1)
        nd = next_d(d, g); nd = np.where(forget, next_d(nd, 3), nd)
        ns = np.maximum(np.where(forget, s_fail, s_ok), 0.001)
        S[i], D[i], last[i] = ns, nd, nxt[i]
        if kind == "dr":
            cur_i = ivl_dr(ns)
        else:
            cur_i = np.where(forget, 1.0, np.maximum(cur[i] + 1, np.round(cur[i] * val)))
        cur[i] = cur_i
        nxt[i] = last[i] + cur_i
    R_end = (1 + factor * (days - last) / S) ** decay
    Rall = np.concatenate(reviews_at_R) if reviews_at_R else np.array([np.nan])
    out = {"memorized": float(R_end.mean()), "reviews": float(nrev.mean()), "seconds_B": float(secB.mean()), "seconds_A": float(secA.mean()),
           "mean_review_recall": float(Rall.mean())}
    for k in DEFS:
        ok = np.isfinite(fl_day[k])
        nk = int(ok.sum())
        spent = np.where(ok, fl_sec[k], secB).sum(); spentA = np.where(ok, fl_secA[k], secA).sum()
        out[k] = {"share": nk / n, "reviews": float(fl_rev[k][ok].mean()) if nk else np.nan, "seconds": float(fl_sec[k][ok].mean()) if nk else np.nan,
                  "days": float(fl_day[k][ok].mean()) if nk else np.nan,
                  "seconds_per_fluent": float(spent / nk) if nk else np.nan, "seconds_per_fluent_A": float(spentA / nk) if nk else np.nan}
    return out


def _user(args):
    path, data, w_fit = args
    _low()
    z = load_user(path, data)
    cm = cost_model(z)
    w = w_fit if w_fit is not None else fsrs_w()
    fc = fast_curves(z)
    obs = observed_fluency(int(z["user"]), data, float(z["cap_ms"]), z)
    pe = obs["persistence"]
    ok = pe["test_n_after_fast"] >= 20 and np.isfinite(pe["head_pf_after_fast"]) and pe["head_pf_has_prev"] > 0
    fc["persist"] = (pe["head_pf_after_fast"] / pe["head_pf_has_prev"], pe["head_pf_after_other"] / pe["head_pf_has_prev"]) if ok else (1.0, 1.0)
    fc["summary"]["persist_a"], fc["summary"]["persist_b"] = fc["persist"]
    sims = {}
    for cmodel, rmix in (("B", False), ("C", True)):
        for dr in DRS:
            sims[f"{cmodel}|dr|{dr:.2f}"] = card_sim(cm, fc, w, ("dr", dr), rmix)
        for gr in GROWTH:
            sims[f"{cmodel}|growth|{gr:g}"] = card_sim(cm, fc, w, ("growth", gr), rmix)
    pr = obs["pass_rate"]
    dr_obs = min(DRS, key=lambda x: abs(x - pr)) if np.isfinite(pr) else 0.9
    return {"user": int(z["user"]), "w_source": "user FSRS-6 fit" if w_fit is not None else "FSRS-6 default",
            "median_time_s": float(np.median(z["rec_s"])), "curves": fc["summary"], "observed": obs, "dr_at_pass_rate": dr_obs, "sims": sims}


# ------------------------------------------------------------------ aggregate + report
def med(xs):
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], np.float64)
    return float(np.median(a)) if a.size else np.nan


def aggregate(users):
    agg = {"users": len(users)}
    def rowset(key_fn, labels, cmodel):
        rows = []
        for lab, key in labels:
            k = key_fn(key)
            rec = {"policy": lab, "memorized": med([u["sims"][k]["memorized"] for u in users]), "reviews": med([u["sims"][k]["reviews"] for u in users]),
                   "seconds_B": med([u["sims"][k]["seconds_B"] for u in users]), "mean_review_recall": med([u["sims"][k]["mean_review_recall"] for u in users])}
            for d in DEFS:
                rec[f"{d}_share"] = med([u["sims"][k][d]["share"] for u in users])
                rec[f"{d}_reviews"] = med([u["sims"][k][d]["reviews"] for u in users if u["sims"][k][d]["share"] >= 0.01])
                rec[f"{d}_seconds"] = med([u["sims"][k][d]["seconds"] for u in users if u["sims"][k][d]["share"] >= 0.01])
                rec[f"{d}_spf"] = med([u["sims"][k][d]["seconds_per_fluent"] for u in users])
                rec[f"{d}_spf_A"] = med([u["sims"][k][d]["seconds_per_fluent_A"] for u in users])
            rows.append(rec)
        return rows
    for cmodel in ("B", "C"):
        agg[f"dr_{cmodel}"] = rowset(lambda key: f"{cmodel}|dr|{key:.2f}", [(f"{d:.2f}", d) for d in DRS], cmodel)
        agg[f"growth_{cmodel}"] = rowset(lambda key: f"{cmodel}|growth|{key:g}", [(f"x{g:g}", g) for g in GROWTH], cmodel)
    # per-user best policy for fluency (cost model C, model 3.9 s): DR with the fewest seconds per fluent card, most share
    best = {}
    for d in ("model39", "model39p", "obs55", "model39_two", "model39p_two"):
        spf = [DRS[int(np.nanargmin([u["sims"][f"C|dr|{x:.2f}"][d]["seconds_per_fluent"] if np.isfinite(u["sims"][f"C|dr|{x:.2f}"][d]["seconds_per_fluent"]) else np.inf for x in DRS]))] for u in users]
        best[d] = {"dr_min_seconds_per_fluent_median": float(np.median(spf)), "quartiles": [float(x) for x in np.percentile(spf, [25, 75])],
                   "share_at_lower_bound": float(np.mean(np.asarray(spf) == DRS[0])), "share_at_upper_bound": float(np.mean(np.asarray(spf) == DRS[-1]))}
    agg["best_dr"] = best
    # ratios per user: seconds per fluent card, DR 0.80 and 0.98 vs 0.90; growth vs DR 0.90
    rel = {}
    for d in DEFS:
        base = lambda u: u["sims"]["C|dr|0.90"][d]["seconds_per_fluent"]
        rel[d] = {k: med([u["sims"][f"C|{k}"][d]["seconds_per_fluent"] / base(u) for u in users if np.isfinite(base(u)) and base(u) > 0])
                  for k in ["dr|0.50", "dr|0.70", "dr|0.80", "dr|0.94", "dr|0.98"] + [f"growth|{g:g}" for g in GROWTH]}
        rel[d]["share_dr_0.98_minus_0.90"] = med([u["sims"]["C|dr|0.98"][d]["share"] - u["sims"]["C|dr|0.90"][d]["share"] for u in users])
        rel[d]["share_dr_0.70_minus_0.90"] = med([u["sims"]["C|dr|0.70"][d]["share"] - u["sims"]["C|dr|0.90"][d]["share"] for u in users])
    agg["relative_to_0_90"] = rel
    # user speed thirds at DR 0.90 (C)
    mt = np.array([u["median_time_s"] for u in users]); q = np.quantile(mt, [1 / 3, 2 / 3])
    th = []
    for k, name in enumerate(["fastest third", "middle third", "slowest third"]):
        m = (mt < q[0]) if k == 0 else ((mt >= q[0]) & (mt < q[1])) if k == 1 else (mt >= q[1])
        sel = [u for u, mm in zip(users, m) if mm]
        rec = {"group": name, "users": len(sel), "range": f"{mt[m].min():.1f}-{mt[m].max():.1f}",
               "p_fast_good_flat": med([u["curves"]["flat39"] for u in sel]), "p_obs55": med([u["curves"]["obs55_overall"] for u in sel])}
        for d in ("model39", "model39p", "obs55", "model39_two", "model39p_two"):
            rec[f"{d}_share"] = med([u["sims"]["C|dr|0.90"][d]["share"] for u in sel])
            rec[f"{d}_reviews"] = med([u["sims"]["C|dr|0.90"][d]["reviews"] for u in sel if u["sims"]["C|dr|0.90"][d]["share"] >= 0.01])
            rec[f"{d}_spf"] = med([u["sims"]["C|dr|0.90"][d]["seconds_per_fluent"] for u in sel])
        rec["obs_share_obs39"] = med([u["observed"]["obs39"]["share"] for u in sel if u["observed"]["obs39"]["cards"] >= 50])
        rec["obs_share_obs55"] = med([u["observed"]["obs55"]["share"] for u in sel if u["observed"]["obs55"]["cards"] >= 50])
        th.append(rec)
    agg["speed_thirds"] = th
    # observed vs simulated at the user's own pass rate
    ov = []
    ok_users = [u for u in users if u["observed"]["obs39"]["cards"] >= 50]
    for d, sim_d in (("obs39", "obs39"), ("obs39", "model39"), ("obs39", "model39p"), ("obs55", "obs55"), ("model39_two", "model39_two"), ("model39_two", "model39p_two")):
        k = lambda u: f"C|dr|{u['dr_at_pass_rate']:.2f}"
        ov.append({"definition": f"{OBS_NAMES[d]} vs simulated {CURVE_NAMES[sim_d]}", "users": len(ok_users),
                   "obs_share": med([u["observed"][d]["share"] for u in ok_users]), "sim_share": med([u["sims"][k(u)][sim_d]["share"] for u in ok_users]),
                   "obs_reviews": med([u["observed"][d]["reviews_mean"] for u in ok_users]), "sim_reviews": med([u["sims"][k(u)][sim_d]["reviews"] for u in ok_users]),
                   "obs_seconds": med([u["observed"][d]["seconds_mean"] for u in ok_users]), "sim_seconds": med([u["sims"][k(u)][sim_d]["seconds"] for u in ok_users]),
                   "obs_spf": med([u["observed"][d]["seconds_per_fluent"] for u in ok_users]), "sim_spf": med([u["sims"][k(u)][sim_d]["seconds_per_fluent"] for u in ok_users])})
    agg["observed_vs_sim"] = ov
    agg["observed_cards_median"] = med([u["observed"]["obs39"]["cards"] for u in users])
    agg["observed_users_50_cards"] = len(ok_users)
    agg["pass_rate_dr_median"] = med([u["dr_at_pass_rate"] for u in users])
    p = [u["observed"]["persistence"] for u in users]
    agg["persistence"] = {k: med([x[k] for x in p]) for k in p[0]}
    agg["curves"] = {key: {pt: med([u["curves"][key][pt] for u in users]) for pt in users[0]["curves"][key]} for key in ("model39_at", "obs39_at", "obs55_at")}
    agg["curves"]["persist_a"] = med([u["curves"]["persist_a"] for u in users])
    agg["curves"]["persist_b"] = med([u["curves"]["persist_b"] for u in users])
    agg["curves"]["flat39"] = med([u["curves"]["flat39"] for u in users])
    agg["curves"]["obs39_overall"] = med([u["curves"]["obs39_overall"] for u in users])
    agg["curves"]["obs55_overall"] = med([u["curves"]["obs55_overall"] for u in users])
    agg["w_sources"] = {s: sum(u["w_source"] == s for u in users) for s in ("user FSRS-6 fit", "FSRS-6 default")}
    return agg


def render(r):
    a = r["agg"]
    L = ["# Time to fluency (SM84): how many reviews and seconds until a card is answered Good fast", ""]
    L += [f"Extension of response-time prediction (SM82) stage 2 (CPU only). Users: the same {a['users']} users of the 732 set ({a['w_sources']['user FSRS-6 fit']} with their own FSRS-6 fit, {a['w_sources']['FSRS-6 default']} default). Fluency = the card's first Good answered under 3.9 s on a later-day review (the answer-time head's bins 0-6, ending at 3,891 ms); also under 5.5 s (bins 0-7) and two fast Goods (3.9 s) in a row. Simulation: {N_CARDS:,} cards per user learned on day 0, followed {DAYS} days with no daily limit, the stage 2 FSRS-6 simulator (user's own fit, one same-day relearning step after a lapse, learning = a fixed first-review cost). Run time {r['seconds']:.0f} s.", ""]
    L += ["## Takeaways", "", "TAKEAWAYS", ""]
    L += ["## What is model-based and what is observed", "",
          "- Model-based: the memory (the user's FSRS-6 fit, not the RWKV model), the per-grade costs over recall (the answer-time head on richer frozen features (SM82c) via stage 2's cost curves: B fixed pass mix, C the grade head's pass mix over recall) and the model 3.9 s fast curve: the head's P(under 3.9 s | Good) over recall, pooled over every curve-grid point of the user (stage 2's counterfactual interval queries, including stretched intervals).",
          "- Observed: the 3.9 s and 5.5 s fast curves from the user's own recorded Good answers at the real intervals, binned by the head's predicted recall (the stored head outputs stop at bin 6, so a 5.5 s model curve needs a GPU re-query that was not done); and the observed reference: cards first seen in the user's revlog with at least 365 days of follow-up, counted the same way on the recorded reviews (the user's real scheduler, not the simulated one).",
          "- In the simulation each Good is fast independently given recall: no card-level speed (some cards are always fast, some never) and no practice effect beyond what recall carries. The persistence check below measures how much this misses.", ""]
    c = a["curves"]
    L += ["## 1. The fast curves (median over users)", "", "P(fast | Good) at a given recall:", ""]
    rows = [{"r": pt, "m": c["model39_at"][pt], "o": c["obs39_at"][pt], "o5": c["obs55_at"][pt]} for pt in c["model39_at"]]
    L += md_table(rows, ["r", "m", "o", "o5"], ["recall", "model, under 3.9 s", "observed, under 3.9 s", "observed, under 5.5 s"])
    L += ["", f"One constant per user (flat model, P(under 3.9 s | Good) at the observed intervals): median {fmt(c['flat39'])}; observed share of Goods under 3.9 s {fmt(c['obs39_overall'])}, under 5.5 s {fmt(c['obs55_overall'])}.", ""]
    p = a["persistence"]
    L += [f"Persistence (every later-day review with an earlier later-day review of the card, median over users): observed P(fast Good) after a fast Good {fmt(p['obs_after_fast'])} vs after any other answer {fmt(p['obs_after_other'])}; the head on the test reviews predicts {fmt(p['head_after_fast'])} vs {fmt(p['head_after_other'])} (actual on the same test reviews after a fast Good {fmt(p['test_obs_after_fast'])}). The plain simulation has no such memory and spreads fast answers over cards evenly, which overstates the share of cards that ever become fluent and understates two in a row. The persistent variant multiplies the model curve by the head's own ratio after a fast Good (median {fmt(c['persist_a'], 2)}x) and after any other answer (median {fmt(c['persist_b'], 2)}x), per user (1 when fewer than 20 test reviews follow a fast Good): a two-state approximation of card-level speed, still not true card heterogeneity.", ""]
    for cmodel in ("C", "B"):
        L += [f"## 2. Desired retention, cost model {cmodel} (median over users)" if cmodel == "C" else "## 2b. Desired retention, cost model B (fixed pass mix; median over users)", ""]
        if cmodel == "C":
            L += ["Share = cards fluent by day 365; reviews / seconds = up to and including the fluent review (seconds include the first learning review), over the cards that got there (users with at least 1% fluent); seconds per fluent card = all seconds spent on every card until it became fluent (or the whole year if it never did) / fluent cards, the honest cost of fluency; memorized = mean recall at day 365; seconds per card = the whole year.", ""]
        rows = [x for x in a[f"dr_{cmodel}"] if float(x["policy"]) in SHOW_DRS]
        L += md_table(rows, ["policy", "mean_review_recall", "reviews", "seconds_B", "memorized", "model39_share", "model39_reviews", "model39_seconds", "model39_spf", "model39p_share", "model39p_spf", "obs55_share", "obs55_reviews", "obs55_spf", "model39_two_share", "model39p_two_share", "model39p_two_spf"],
                      ["desired retention", "recall at review", "reviews per card", "seconds per card", "memorized", "fluent 3.9 s", "reviews to 3.9 s", "seconds to 3.9 s", "seconds per fluent card 3.9 s", "fluent 3.9 s, persistent", "seconds per fluent card 3.9 s, persistent", "fluent 5.5 s (obs curve)", "reviews to 5.5 s", "seconds per fluent card 5.5 s", "two in a row", "two in a row, persistent", "seconds per fluent card two in a row, persistent"])
        L += [""]
    L += ["## 3. Interval-growth policies (cost model C, median over users)", "",
          "Next interval = previous x g after a pass (1 day after learning and after a lapse); memory still the user's FSRS-6 fit, fast answers still from recall only.", ""]
    rows = a["growth_C"] + [x for x in a["dr_C"] if x["policy"] in ("0.80", "0.90")]
    L += md_table(rows, ["policy", "mean_review_recall", "reviews", "seconds_B", "memorized", "model39_share", "model39_reviews", "model39_spf", "model39p_share", "obs55_share", "obs55_spf", "model39_two_share", "model39p_two_share", "model39p_two_spf"],
                  ["policy", "recall at review", "reviews per card", "seconds per card", "memorized", "fluent 3.9 s", "reviews to 3.9 s", "seconds per fluent card 3.9 s", "fluent 3.9 s, persistent", "fluent 5.5 s (obs curve)", "seconds per fluent card 5.5 s", "two in a row", "two in a row, persistent", "seconds per fluent card two in a row, persistent"])
    L += ["", "Seconds per fluent card relative to desired retention 0.90 (cost model C, median of the user's ratio):", ""]
    rel = a["relative_to_0_90"]
    keys = ["dr|0.50", "dr|0.70", "dr|0.80", "dr|0.94", "dr|0.98"] + [f"growth|{g:g}" for g in GROWTH]
    rows = [{"d": CURVE_NAMES[d], **{k: rel[d][k] for k in keys}, "s98": rel[d]["share_dr_0.98_minus_0.90"], "s70": rel[d]["share_dr_0.70_minus_0.90"]} for d in rel]
    L += md_table(rows, ["d"] + keys + ["s70", "s98"], ["fluency curve"] + [k.replace("dr|", "DR ").replace("growth|", "growth x") for k in keys] + ["share fluent, DR 0.70 minus 0.90", "share fluent, DR 0.98 minus 0.90"])
    b = a["best_dr"]
    L += ["", "Per user, the desired retention with the fewest seconds per fluent card (cost model C): "
          + "; ".join(f"{CURVE_NAMES[d]}: median {fmt(v['dr_min_seconds_per_fluent_median'], 2)} (quartiles {v['quartiles'][0]:.2f}-{v['quartiles'][1]:.2f}, at 0.50 {v['share_at_lower_bound']:.0%}, at 0.98 {v['share_at_upper_bound']:.0%})" for d, v in b.items()) + ".", ""]
    L += ["## 4. By user speed (desired retention 0.90, cost model C)", ""]
    L += md_table(a["speed_thirds"], ["group", "users", "range", "p_fast_good_flat", "p_obs55", "model39_share", "model39_reviews", "model39_spf", "model39p_share", "obs55_share", "model39p_two_share", "obs_share_obs39", "obs_share_obs55"],
                  ["median recorded time", "users", "range (s)", "P(under 3.9 s given Good), head", "Goods under 5.5 s, observed", "fluent 3.9 s (sim)", "reviews to 3.9 s", "seconds per fluent card 3.9 s", "fluent 3.9 s, persistent (sim)", "fluent 5.5 s (sim, obs curve)", "two in a row, persistent (sim)", "fluent 3.9 s (observed cards)", "fluent 5.5 s (observed cards)"])
    L += ["", "## 5. Observed reference vs the simulation", "",
          f"Observed: the user's cards first seen in the revlog with at least 365 days of follow-up (median {fmt(a['observed_cards_median'], 0)} cards per user; {a['observed_users_50_cards']} users with at least 50), counted on the recorded reviews within 365 days of the first one. Simulated: cost model C at the desired retention closest to the user's observed later-day pass rate (median {fmt(a['pass_rate_dr_median'], 2)}). The real scheduler is not FSRS-6 at that target and real cards can be suspended or deleted (counted as never fluent), so this is a sanity check, not a calibration.", ""]
    L += md_table(a["observed_vs_sim"], ["definition", "users", "obs_share", "sim_share", "obs_reviews", "sim_reviews", "obs_seconds", "sim_seconds", "obs_spf", "sim_spf"],
                  ["definition", "users", "fluent by day 365, observed", "simulated", "reviews to fluency, observed", "simulated", "seconds to fluency, observed", "simulated", "seconds per fluent card, observed", "simulated"])
    L += ["", "## Limits", "",
          "- Counterfactual: the model 3.9 s curve comes from stage 2's interval queries, where only the review's query row moves (session counters, other reviews and running statistics stay as observed). The recall bins pool different cards, so part of the slope is card difficulty, not the same card getting faster.",
          "- Independence: in the simulation a Good is fast with a probability that depends on the user and the recall only. Real speed is card-specific and persistent (section 1), so the simulated share of cards ever fluent is too even across cards; read the policy comparisons (relative numbers) rather than the absolute shares.",
          "- No practice effect: the head sees the card's history, but the curve over recall does not carry it; with recall-only fast answers, longer intervals can only slow fluency down (lower recall). The stage 2 link between interval growth 1.7-4x and fast Goods is descriptive (the scheduler gives big growth to easy cards) and cannot be reproduced here.",
          "- The memory model is FSRS-6 (user fits), not the RWKV model; learning is a fixed cost per new card, a lapse gets one relearning step, fluency counts later-day reviews only. 5.5 s uses observed curves only. 150 users of the 732 set.", ""]
    return L


def rd(x):
    if isinstance(x, float):
        return float(f"{x:.5g}")
    if isinstance(x, dict):
        return {k: rd(v) for k, v in x.items()}
    if isinstance(x, list):
        return [rd(v) for v in x]
    return x


def slim(u):
    """Per-user file: curves, observed reference, cost model C at the shown desired retentions and growth factors."""
    keep = [f"C|dr|{d:.2f}" for d in SHOW_DRS] + [f"C|growth|{g:g}" for g in GROWTH]
    sims = {k: {"memorized": v["memorized"], "reviews": v["reviews"], "seconds_B": v["seconds_B"],
                **{d: {kk: v[d][kk] for kk in ("share", "reviews", "seconds", "seconds_per_fluent")} for d in ("model39", "model39p", "obs55", "model39_two", "model39p_two")}}
            for k, v in u["sims"].items() if k in keep}
    return rd({"user": u["user"], "w_source": u["w_source"], "median_time_s": u["median_time_s"], "curves": u["curves"], "observed": u["observed"],
               "dr_at_pass_rate": u["dr_at_pass_rate"], "sims_C": sims})


def main():
    ap = ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--curves", default="C:/Users/jonat/workspace/wt/sm82s2/artifacts/small-model/sm82/stage2/work/curves")
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts/small-model/sm84"))
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    from scripts import compress_common as cc
    _low()
    t0 = time.time()
    data = cc.data_path()
    files = sorted((Path(args.curves) / "users").glob("*.npz"))
    if args.limit:
        files = files[:args.limit]
    fits = user_fits([int(f.stem) for f in files])
    jobs = [(str(f), str(data), fits.get(int(f.stem))) for f in files]
    with Pool(min(args.procs, 6), initializer=_low) as pool:
        users = pool.map(_user, jobs, chunksize=1)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    res = {"name": "time to fluency (SM84)", "users": len(users), "seconds": round(time.time() - t0, 1), "agg": aggregate(users),
           "settings": {"days": DAYS, "cards": N_CARDS, "drs": DRS, "growth": GROWTH, "fast_ms": FAST_MS, "fast55_ms": FAST55_MS}}
    sfx = "" if not args.limit else f"-test{args.limit}"
    (out / f"time-to-fluency{sfx}.json").write_text(json.dumps(res, indent=1, default=float))
    (out / f"time-to-fluency-users{sfx}.json").write_text(json.dumps([slim(u) for u in users], separators=(",", ":"), default=float))
    md = out / f"time-to-fluency{sfx}.md"
    text = "\n".join(render(res)) + "\n"
    if md.exists() and "TAKEAWAYS" not in md.read_text(encoding="utf-8"):
        md = out / f"time-to-fluency{sfx}.new.md"
    md.write_text(text, encoding="utf-8")
    print(f"[sm84] {len(users)} users -> {md} ({res['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
