"""Response-time prediction (SM82) stage 2: the CPU analysis of the answer-time curve queries (scripts/sm82s2_curves.py
query): curves, calibration on the observed intervals, fast-Good patterns, time-aware workload simulation ->
artifacts/small-model/sm82/stage2/stage2-result.{md,json}. Called as `scripts/sm82s2_curves.py analyze`."""
import json
import math
from multiprocessing import Pool
import os
from pathlib import Path
import time

import numpy as np

FAST_MS = 500.0 * 120.0 ** (6 / 14)   # 3,891 ms: bins 0..6, the quantity SM82 / SM82c label "under 5.5 s"
FAST55_MS = 500.0 * 120.0 ** (7 / 14)  # 5,477 ms (actual data only: the stored head output is bins 0..6)
RATIO_EDGES = [0, 0.5, 0.9, 1.1, 1.7, 2.5, 4.0, np.inf]
RATIO_NAMES = ["< 0.5", "0.5-0.9", "0.9-1.1", "1.1-1.7", "1.7-2.5", "2.5-4", "> 4"]
E_EDGES = [1, 2, 4, 8, 22, 61, 181, 10 ** 9]
E_NAMES = ["1", "2-3", "4-7", "8-21", "22-60", "61-180", "181+"]
P_EDGES = [0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.96, 0.98, 1.0001]
DRS = [round(0.50 + 0.01 * i, 2) for i in range(48)]   # no-time-limit grid
TARGETS = [0.70, 0.80, 0.85, 0.90, 0.95]
COST_BINS = [0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0001]


def grid_names():
    from scripts.sm82s2_curves import GRID
    return GRID


def load_user(path, data):
    import pandas as pd
    z = dict(np.load(path))
    user = int(Path(path).stem)
    rev = pd.read_parquet(Path(data) / "revlogs" / f"user_id={user}", columns=["card_id", "day_offset", "rating", "duration", "elapsed_days"])
    rev = rev.reset_index(drop=True)
    card, el, rating = rev["card_id"], rev["elapsed_days"], rev["rating"]
    later = el.where(el >= 1)
    prev = later.groupby(card).shift(1).groupby(card).ffill()
    day_rev = (el != 0).astype(int)
    reps = day_rev.groupby(card).cumsum() - day_rev
    lapse = ((rating == 1) & (el >= 1)).astype(int)
    lapses = lapse.groupby(card).cumsum() - lapse
    first_ivl = later.groupby(card).transform("first")
    session = rev.groupby("day_offset").cumcount()
    i = z["th"].astype(np.int64) - 1
    z["prev_e"] = prev.to_numpy(np.float64)[i]
    z["reps"] = reps.to_numpy(np.int64)[i]
    z["lapses"] = lapses.to_numpy(np.int64)[i]
    z["first_ivl"] = first_ivl.to_numpy(np.float64)[i]
    z["session"] = session.to_numpy(np.int64)[i]
    cap = float(z["cap_ms"])
    d = rev["duration"].to_numpy(np.float64)
    first = (el.to_numpy() == -1) & (d > 0)
    same = (el.to_numpy() == 0) & (d > 0)
    z["learn_cost"] = float(np.minimum(d[first], cap).mean() / 1000) if first.any() else np.nan
    z["relearn_cost"] = float(np.minimum(d[same], cap).mean() / 1000) if same.any() else np.nan
    fr = np.bincount(rating.to_numpy()[el.to_numpy() == -1], minlength=5)[1:5].astype(float)
    z["first_probs"] = fr / fr.sum() if fr.sum() else np.array([0.24, 0.094, 0.495, 0.171])
    z["user"] = user
    z["rec_s"] = np.minimum(z["dur"].astype(np.float64), cap) / 1000.0
    return z


def derived(pg, es, pf):
    """pg, es, pf [..., 4] -> recall, E time, E time | pass, E time | Again, P(fast Good), P(Good)."""
    pg = pg.astype(np.float64); es = es.astype(np.float64); pf = pf.astype(np.float64)
    rec = 1 - pg[..., 0]
    et = (pg * es).sum(-1)
    ep = (pg[..., 1:] * es[..., 1:]).sum(-1) / np.maximum(rec, 1e-9)
    return {"recall": rec, "E": et, "E_pass": ep, "E_again": es[..., 0], "P_fast_good": pg[..., 2] * pf[..., 2],
            "P_good": pg[..., 2], "P_fast_pass": (pg[..., 1:] * pf[..., 1:]).sum(-1)}


def bucket(x, edges):
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)


# ------------------------------------------------------------------ 1. curves
def curve_tables(users):
    grid = grid_names()
    per_user = []
    for z in users:
        d = derived(z["grid_pg"], z["grid_es"], z["grid_pf"])   # [G, S]
        rows = {}
        for g, (name, kind, value) in enumerate(grid):
            rows[name] = {"t_median": float(np.median(z["grid_t"][g])), **{k: float(v[g].mean()) for k, v in d.items()}}
        per_user.append(rows)
    table = []
    for g, (name, kind, value) in enumerate(grid):
        rec = {"grid": name, "kind": kind, "value": value}
        for k in per_user[0][name]:
            rec[k] = float(np.mean([u[name][k] for u in per_user]))
        table.append(rec)
    ident = next(r for r in table if r["grid"] == "x1")
    for r in table:
        r["E_vs_x1"] = r["E"] / ident["E"] - 1
        r["E_pass_vs_x1"] = r["E_pass"] / ident["E_pass"] - 1
        r["P_fast_good_vs_x1"] = r["P_fast_good"] / ident["P_fast_good"] - 1
    return table, per_user


def recall_bins(users):
    """All (review, grid point) pairs pooled: E time | pass, E time | Again, P(fast | Good) by predicted recall bin;
    user-averaged (users with >= 30 points in the bin)."""
    out = []
    acc = {k: [] for k in range(len(P_EDGES) - 1)}
    for z in users:
        d = derived(z["grid_pg"], z["grid_es"], z["grid_pf"])
        b = bucket(d["recall"].ravel(), P_EDGES)
        pf_good = z["grid_pf"][..., 2].astype(np.float64).ravel()
        for k in range(len(P_EDGES) - 1):
            m = b == k
            if m.sum() >= 30:
                acc[k].append((m.sum(), d["E_pass"].ravel()[m].mean(), d["E_again"].ravel()[m].mean(), pf_good[m].mean(),
                               d["E"].ravel()[m].mean(), d["P_fast_good"].ravel()[m].mean()))
    for k, rows in acc.items():
        if not rows:
            continue
        a = np.asarray(rows)
        out.append({"recall_bin": f"{P_EDGES[k]:.2f}-{min(P_EDGES[k + 1], 1):.2f}", "users": len(rows), "points": int(a[:, 0].sum()),
                    "E_pass": float(a[:, 1].mean()), "E_again": float(a[:, 2].mean()), "P_fast_given_good": float(a[:, 3].mean()),
                    "E": float(a[:, 4].mean()), "P_fast_good": float(a[:, 5].mean())})
    return out


def slopes(users):
    """Per review over the 8 interval-multiplier points: OLS slope of E time | pass on recall (s per -0.1 recall) and
    the elasticity d ln E / d ln t. User-averaged medians."""
    grid = grid_names()
    mi = [g for g, (_, kind, _) in enumerate(grid) if kind == "mult"]
    s_pass, s_all, el = [], [], []
    for z in users:
        d = derived(z["grid_pg"][mi], z["grid_es"][mi], z["grid_pf"][mi])
        r = d["recall"]; rc = r - r.mean(0)
        den = (rc ** 2).sum(0)
        ok = den > 1e-8
        sp = -0.1 * ((rc * (d["E_pass"] - d["E_pass"].mean(0))).sum(0)[ok] / den[ok])
        sa = -0.1 * ((rc * (d["E"] - d["E"].mean(0))).sum(0)[ok] / den[ok])
        lt = np.log(z["grid_t"][mi].astype(np.float64)); ltc = lt - lt.mean(0)
        dl = (ltc ** 2).sum(0); okt = dl > 1e-8
        le = np.log(d["E"]); e1 = (ltc * (le - le.mean(0))).sum(0)[okt] / dl[okt]
        s_pass.append(np.median(sp) if sp.size else np.nan); s_all.append(np.median(sa) if sa.size else np.nan)
        el.append(np.median(e1) if e1.size else np.nan)
    return {"E_pass_s_per_0.1_recall_drop_median": float(np.nanmedian(s_pass)),
            "E_s_per_0.1_recall_drop_median": float(np.nanmedian(s_all)),
            "elasticity_lnE_lnt_median": float(np.nanmedian(el)),
            "E_pass_slope_user_quartiles": [float(x) for x in np.nanpercentile(s_pass, [25, 50, 75])],
            "elasticity_user_quartiles": [float(x) for x in np.nanpercentile(el, [25, 50, 75])]}


def user_types(users, per_user):
    med = np.array([np.median(z["rec_s"]) for z in users])
    ret = np.array([(z["grade"] > 1).mean() for z in users])
    out = []
    for label, vals in (("median recorded time", med), ("observed recall", ret)):
        q = np.quantile(vals, [1 / 3, 2 / 3])
        for k, name in enumerate(["low third", "middle third", "high third"]):
            m = (vals < q[0]) if k == 0 else ((vals >= q[0]) & (vals < q[1])) if k == 1 else (vals >= q[1])
            sel = [per_user[i] for i in np.nonzero(m)[0]]
            rec = {"split": label, "group": name, "users": int(m.sum()), "range": [float(vals[m].min()), float(vals[m].max())]}
            for gname in ("x0.5", "x1", "x2", "x4", "d1", "d30", "d365"):
                rec[f"recall_{gname}"] = float(np.mean([u[gname]["recall"] for u in sel]))
                rec[f"E_{gname}"] = float(np.mean([u[gname]["E"] for u in sel]))
                rec[f"E_pass_{gname}"] = float(np.mean([u[gname]["E_pass"] for u in sel]))
                rec[f"P_fast_good_{gname}"] = float(np.mean([u[gname]["P_fast_good"] for u in sel]))
            out.append(rec)
    return out


# ------------------------------------------------------------------ calibration on observed intervals
def calibration(users):
    out = {"by_interval_growth": [], "by_interval_days": []}
    for key, fn, names in (("by_interval_growth", lambda z: np.where(np.isnan(z["prev_e"]), -1, bucket(z["e"] / np.where(np.isnan(z["prev_e"]), 1, z["prev_e"]), RATIO_EDGES)),
                            ["first later-day interval"] + RATIO_NAMES),
                           ("by_interval_days", lambda z: bucket(z["e"], E_EDGES), E_NAMES)):
        acc = {}
        for z in users:
            b = fn(z)
            d = derived(z["full_pg"], z["full_es"], z["full_pf"])
            g = z["grade"].astype(np.int64) - 1
            es_true = z["full_es"][np.arange(len(g)), g].astype(np.float64)
            fast_good = (z["grade"] == 3) & (z["dur"] < FAST_MS)
            fast55 = (z["grade"] == 3) & (z["dur"] < FAST55_MS)
            pf_true = z["full_pf"][np.arange(len(g)), g].astype(np.float64)
            for k in np.unique(b):
                m = b == k
                a = acc.setdefault(int(k), np.zeros(12))
                a += [m.sum(), d["recall"][m].sum(), (z["grade"][m] > 1).sum(), d["E"][m].sum(), z["rec_s"][m].sum(),
                      es_true[m].sum(), d["P_fast_good"][m].sum(), fast_good[m].sum(), 1, pf_true[m].sum(),
                      (z["dur"][m] < FAST_MS).sum(), fast55[m].sum()]
        for k in sorted(acc):
            a = acc[k]; n = a[0]
            name = names[0] if (key == "by_interval_growth" and k == -1) else names[k + (1 if key == "by_interval_growth" else 0)]
            out[key].append({"bucket": name, "reviews": int(n), "users": int(a[8]), "recall_pred": a[1] / n, "recall_actual": a[2] / n,
                             "time_pred_s": a[3] / n, "time_actual_s": a[4] / n, "time_pred_true_grade_s": a[5] / n,
                             "fast_good_pred": a[6] / n, "fast_good_actual": a[7] / n, "fast_true_grade_pred": a[9] / n,
                             "fast_actual": a[10] / n, "good_under_5_5s_actual": a[11] / n})
    return out


# ------------------------------------------------------------------ 2. patterns
def patterns(users):
    feats = {
        "card's earlier review days": (lambda z: z["reps"], [1, 2, 3, 5, 9, 17, 10 ** 9], ["1", "2", "3-4", "5-8", "9-16", "17+"]),
        "card's earlier lapses": (lambda z: z["lapses"], [0, 1, 2, 3, 6, 10 ** 9], ["0", "1", "2", "3-5", "6+"]),
        "card's first later-day interval (days)": (lambda z: z["first_ivl"], [1, 2, 4, 8, 15, 10 ** 9], ["1", "2-3", "4-7", "8-14", "15+"]),
        "interval growth (this interval / previous)": (lambda z: np.where(np.isnan(z["prev_e"]), np.nan, z["e"] / np.where(np.isnan(z["prev_e"]), 1, z["prev_e"])),
                                                        RATIO_EDGES, RATIO_NAMES),
        "this interval (days)": (lambda z: z["e"], E_EDGES, E_NAMES),
        "position in the day's reviews": (lambda z: z["session"], [0, 10, 50, 100, 200, 10 ** 9], ["1-10", "11-50", "51-100", "101-200", "201+"]),
    }
    out = {}
    for fname, (fn, edges, names) in feats.items():
        acc = {k: [] for k in range(len(names))}
        pooled = {k: np.zeros(7) for k in range(len(names))}
        for z in users:
            x = np.asarray(fn(z), np.float64)
            ok = ~np.isnan(x)
            b = bucket(np.where(ok, x, 0), edges)
            fast_good = (z["grade"] == 3) & (z["dur"] < FAST_MS)
            fast55 = (z["grade"] == 3) & (z["dur"] < FAST55_MS)
            good = z["grade"] == 3
            d = derived(z["full_pg"], z["full_es"], z["full_pf"])
            for k in range(len(names)):
                m = ok & (b == k)
                if not m.any():
                    continue
                pooled[k] += [m.sum(), fast_good[m].sum(), good[m].sum(), (z["grade"][m] > 1).sum(), z["rec_s"][m].sum(), d["P_fast_good"][m].sum(), fast55[m].sum()]
                if m.sum() >= 50:
                    acc[k].append((fast_good[m].mean(), z["rec_s"][m].mean()))
        rows = []
        for k, name in enumerate(names):
            a = pooled[k]
            if a[0] == 0:
                continue
            ua = np.asarray(acc[k]) if acc[k] else np.full((1, 2), np.nan)
            rows.append({"bucket": name, "reviews": int(a[0]), "fast_good_share": a[1] / a[0], "fast_given_good": a[1] / max(a[2], 1),
                         "recall": a[3] / a[0], "mean_time_s": a[4] / a[0], "fast_good_pred": a[5] / a[0], "good_under_5_5s": a[6] / a[0],
                         "users_50plus": len(acc[k]), "fast_good_share_user_avg": float(np.nanmean(ua[:, 0])),
                         "mean_time_s_user_avg": float(np.nanmean(ua[:, 1]))})
        out[fname] = rows
    return out


# ------------------------------------------------------------------ 3. workload
VARIANTS = ["A_fixed_actual", "A_fixed_head", "B_recall_cost", "C_recall_cost_and_mix"]
BUDGET_S = 1800.0          # the fsrs_optimizer default daily time limit (30 min)
BUDGET_DRS = [round(0.60 + 0.02 * i, 2) for i in range(20)]    # 0.60 .. 0.98


FSRS6_FITS = Path("C:/Users/jonat/workspace/srs-benchmark/result/FSRS-6-recency.jsonl")   # read only: per-user FSRS-6 fits


def fsrs_w():
    from fsrs_optimizer.fsrs_optimizer import DEFAULT_PARAMETER
    return np.asarray(DEFAULT_PARAMETER, np.float64)


def user_fits(users):
    want = {int(u) for u in users}
    out = {}
    if FSRS6_FITS.exists():
        for line in FSRS6_FITS.read_text().splitlines():
            r = json.loads(line)
            if int(r["user"]) in want:
                ps = r.get("parameters", {})
                if ps:
                    out[int(r["user"])] = np.asarray(ps[sorted(ps, key=int)[-1]], np.float64)
    return out


def cost_model(z):
    """Per user: fixed per-grade costs (head mean at the observed intervals; actual recorded mean), recall-dependent
    per-grade costs and pass-rating mix from the head over every grid point."""
    pg = z["grid_pg"].reshape(-1, 4).astype(np.float64); es = z["grid_es"].reshape(-1, 4).astype(np.float64)
    r = 1 - pg[:, 0]
    b = bucket(r, COST_BINS)
    xs, cs, mixes = [], [], []
    for k in range(len(COST_BINS) - 1):
        m = b == k
        if m.sum() < 20:
            continue
        w = pg[m]
        xs.append(r[m].mean())
        cs.append((w * es[m]).sum(0) / np.maximum(w.sum(0), 1e-9))
        mixes.append(w[:, 1:].sum(0) / w[:, 1:].sum())
    fpg, fes = z["full_pg"].astype(np.float64), z["full_es"].astype(np.float64)
    fixed_head = (fpg * fes).sum(0) / fpg.sum(0)
    actual = np.array([z["rec_s"][z["grade"] == g].mean() if (z["grade"] == g).any() else fixed_head[g - 1] for g in (1, 2, 3, 4)])
    passes = np.bincount(z["grade"][z["grade"] > 1], minlength=5)[2:5].astype(float)
    mix = passes / passes.sum() if passes.sum() else np.array([0.224, 0.631, 0.145])
    relearn = z["relearn_cost"] if np.isfinite(z["relearn_cost"]) else fixed_head[2]
    learn = z["learn_cost"] if np.isfinite(z["learn_cost"]) else actual[0]
    return {"x": np.asarray(xs), "c": np.asarray(cs), "mix_r": np.asarray(mixes), "fixed_head": fixed_head, "fixed_actual": actual,
            "mix": mix, "first_probs": np.asarray(z["first_probs"], np.float64), "learn": float(learn), "relearn": float(relearn)}


def simulate(cm, dr, r_mix, budget_variant=None, seed=0, days=365, per_day=10, w=None):
    """FSRS-6 card simulation (default parameters, fsrs_optimizer's formulas, power forgetting curve, one same-day
    relearning step rated Good after a lapse, no fuzz).
    budget_variant None: `per_day` new cards a day, no time limit; returns every cost model's total on the same
    trajectory. budget_variant = one of VARIANTS: a daily time limit of BUDGET_S seconds charged with that variant's
    costs (due reviews first in due order, the rest wait a day; then new cards while time is left, unlimited deck)."""
    w = (cm.get("w") if cm.get("w") is not None else fsrs_w()) if w is None else w
    decay = -w[20]; factor = 0.9 ** (1 / decay) - 1
    rng = np.random.default_rng(seed)
    n = days * per_day if budget_variant is None else int(days * BUDGET_S / max(cm["learn"], 1.0)) + 1
    S = np.zeros(n); D = np.zeros(n); last = np.zeros(n); due = np.full(n, np.inf)
    counts = np.zeros(4); cost_r = 0.0; relearns = 0; learned = 0; spent = 0.0
    ivl = lambda s: np.maximum(1, np.round(s / factor * (dr ** (1 / decay) - 1)))
    init_d = lambda g: w[4] - np.exp(w[5] * (g - 1)) + 1

    def next_d(d, g):
        nd = d + (-w[6] * (g - 3)) * (10 - d) / 9
        nd = w[7] * init_d(4) + (1 - w[7]) * nd
        return np.clip(nd, 1, 10)

    xs, cs, mr = cm["x"], cm["c"], cm["mix_r"]
    use_r = len(xs) > 0
    for day in range(days):
        left = BUDGET_S
        idx = np.nonzero(due[:learned] <= day)[0]
        if idx.size:
            idx = idx[np.argsort(due[idx], kind="stable")]
            s, d = S[idx], D[idx]
            R = (1 + factor * (day - last[idx]) / s) ** decay
            u = rng.random(idx.size); v = rng.random(idx.size)
            forget = u > R
            if r_mix and use_r:
                m = np.stack([np.interp(R, xs, mr[:, j]) for j in range(3)], -1)
                m /= m.sum(-1, keepdims=True)
            else:
                m = np.broadcast_to(cm["mix"], (idx.size, 3))
            cum = np.cumsum(m, -1)
            g = np.where(forget, 1, 2 + (v[:, None] > cum[:, :2]).sum(-1))
            cg = np.stack([np.interp(R, xs, cs[:, j]) for j in range(4)], -1) if use_r else np.broadcast_to(cm["fixed_head"], (idx.size, 4))
            c_r = cg[np.arange(idx.size), g - 1]
            if budget_variant is not None:
                c_v = {"A_fixed_actual": cm["fixed_actual"][g - 1], "A_fixed_head": cm["fixed_head"][g - 1]}.get(budget_variant, c_r)
                c_v = c_v + np.where(forget, cm["relearn"], 0.0)
                keep = np.cumsum(c_v) <= left
                left -= float(c_v[keep].sum()); spent += float(c_v[keep].sum())
                idx, s, d, R, forget, g, c_r = idx[keep], s[keep], d[keep], R[keep], forget[keep], g[keep], c_r[keep]
            counts += np.bincount(g, minlength=5)[1:5]
            cost_r += float(c_r.sum())
            relearns += int(forget.sum())
            hard = np.where(g == 2, w[15], 1.0); easy = np.where(g == 4, w[16], 1.0)
            s_ok = s * (1 + np.exp(w[8]) * (11 - d) * s ** (-w[9]) * (np.exp((1 - R) * w[10]) - 1) * hard * easy)
            s_fail = np.minimum(w[11] * d ** (-w[12]) * ((s + 1) ** w[13] - 1) * np.exp((1 - R) * w[14]), s / np.exp(w[17] * w[18]))
            s_fail = s_fail * np.maximum(np.exp(w[17] * w[18]) * np.maximum(s_fail, 1e-3) ** (-w[19]), 1)
            nd = next_d(d, g)
            nd = np.where(forget, next_d(nd, 3), nd)
            ns = np.maximum(np.where(forget, s_fail, s_ok), 0.001)
            S[idx], D[idx], last[idx] = ns, nd, day
            due[idx] = day + ivl(ns)
        k = per_day if budget_variant is None else min(int(max(left, 0) // cm["learn"]), n - learned)
        if k > 0:
            new = np.arange(learned, learned + k)
            g0 = rng.choice(4, k, p=cm["first_probs"]) + 1
            S[new] = w[g0 - 1]; D[new] = np.clip(init_d(g0), 1, 10); last[new] = day
            due[new] = day + ivl(S[new])
            learned += k
            spent += k * cm["learn"] if budget_variant is not None else 0.0
    R_end = (1 + factor * (days - last[:learned]) / S[:learned]) ** decay
    memorized = float(R_end.sum())
    learn = learned * cm["learn"]; relearn = relearns * cm["relearn"]
    return {"memorized": memorized, "learned": learned, "reviews": int(counts.sum()), "lapses": relearns, "learn_s": learn, "relearn_s": relearn,
            "A_fixed_head": learn + relearn + float(counts @ cm["fixed_head"]), "A_fixed_actual": learn + relearn + float(counts @ cm["fixed_actual"]),
            "B_recall_cost": learn + relearn + cost_r, "spent_s": spent}


def _user_workload(args):
    user, cm = args
    free = []
    for dr in DRS:
        a = simulate(cm, dr, False)
        c = simulate(cm, dr, True)
        free.append({"dr": dr, "memorized": a["memorized"], "reviews": a["reviews"],
                     "A_fixed_head": a["A_fixed_head"] / a["memorized"], "A_fixed_actual": a["A_fixed_actual"] / a["memorized"],
                     "B_recall_cost": a["B_recall_cost"] / a["memorized"], "C_recall_cost_and_mix": c["B_recall_cost"] / c["memorized"],
                     "C_memorized": c["memorized"]})
    budget = []
    for dr in BUDGET_DRS:
        rec = {"dr": dr}
        for v in VARIANTS:
            b = simulate(cm, dr, v == "C_recall_cost_and_mix", budget_variant=v)
            rec[v] = b["memorized"]; rec[f"{v}_learned"] = b["learned"]
        budget.append(rec)
    return user, {"free": free, "budget": budget}


def workload(users, procs):
    fits = user_fits([z["user"] for z in users])
    jobs = []
    for z in users:
        cm = cost_model(z)
        cm["w"] = fits.get(int(z["user"]))
        cm["w_source"] = "user FSRS-6 fit" if cm["w"] is not None else "FSRS-6 default"
        jobs.append((int(z["user"]), cm))
    with Pool(procs, initializer=_low) as pool:
        res = dict(pool.map(_user_workload, jobs, chunksize=1))
    per_user = []
    for user, r in res.items():
        rec = {"user_id": user, "free": r["free"], "budget": r["budget"]}
        for v in VARIANTS:
            y = np.array([x[v] for x in r["free"]]); k = int(np.argmin(y))
            rec[f"free_opt_{v}"] = DRS[k]
            yb = np.array([x[v] for x in r["budget"]]); kb = int(np.argmax(yb))
            rec[f"budget_opt_{v}"] = BUDGET_DRS[kb]; rec[f"budget_max_{v}"] = float(yb[kb])
        for v in ("B_recall_cost", "C_recall_cost_and_mix"):
            yb = np.array([x[v] for x in r["budget"]])
            rec[f"budget_loss_{v}_at_A_opt"] = float(1 - yb[BUDGET_DRS.index(rec["budget_opt_A_fixed_head"])] / yb.max())
        per_user.append(rec)
    summary = {"variants": VARIANTS, "fsrs_parameters": {"user_fit": sum(cm["w"] is not None for _, cm in jobs), "default": sum(cm["w"] is None for _, cm in jobs),
                                                        "source": str(FSRS6_FITS), "decay_median": float(np.median([(cm["w"] if cm["w"] is not None else fsrs_w())[20] for _, cm in jobs]))}}
    # no time limit: time per memorized card at fixed targets
    tg = []
    for t in TARGETS:
        i = DRS.index(t)
        rec = {"dr": t, "memorized_share": float(np.median([p["free"][i]["memorized"] / 3650 for p in per_user]))}
        for v in VARIANTS:
            rec[v] = float(np.median([p["free"][i][v] for p in per_user]))
        rec["B_over_A_head"] = float(np.median([p["free"][i]["B_recall_cost"] / p["free"][i]["A_fixed_head"] for p in per_user]))
        rec["C_over_A_head"] = float(np.median([p["free"][i]["C_recall_cost_and_mix"] / p["free"][i]["A_fixed_head"] for p in per_user]))
        rec["A_head_over_A_actual"] = float(np.median([p["free"][i]["A_fixed_head"] / p["free"][i]["A_fixed_actual"] for p in per_user]))
        tg.append(rec)
    summary["free_targets"] = tg
    summary["free_opt"] = {v: {"median": float(np.median([p[f"free_opt_{v}"] for p in per_user])),
                               "share_at_lower_bound": float(np.mean([p[f"free_opt_{v}"] == DRS[0] for p in per_user]))} for v in VARIANTS}
    summary["free_relative_to_0_90"] = [{"dr": DRS[i], **{v: float(np.median([p["free"][i][v] / p["free"][DRS.index(0.9)][v] for p in per_user])) for v in VARIANTS}}
                                        for i in range(0, len(DRS), 5)]
    # 30-minute daily limit: memorized cards at day 365
    bo = {}
    for v in VARIANTS:
        o = np.array([p[f"budget_opt_{v}"] for p in per_user])
        bo[v] = {"opt_dr_median": float(np.median(o)), "opt_dr_mean": float(o.mean()), "opt_dr_quartiles": [float(x) for x in np.percentile(o, [25, 75])],
                 "max_memorized_median": float(np.median([p[f"budget_max_{v}"] for p in per_user])),
                 "share_at_bounds": float(np.mean((o == BUDGET_DRS[0]) | (o == BUDGET_DRS[-1])))}
    summary["budget_opt"] = bo
    a = np.array([p["budget_opt_A_fixed_head"] for p in per_user])
    for v in ("A_fixed_actual", "B_recall_cost", "C_recall_cost_and_mix"):
        o = np.array([p[f"budget_opt_{v}"] for p in per_user])
        summary[f"budget_shift_{v}_minus_A_head"] = {"mean": float((o - a).mean()), "median": float(np.median(o - a)), "share_up": float((o > a).mean()),
                                                     "share_down": float((o < a).mean()), "share_same": float((o == a).mean())}
    for v in ("B_recall_cost", "C_recall_cost_and_mix"):
        x = [p[f"budget_loss_{v}_at_A_opt"] for p in per_user]
        summary[f"budget_loss_{v}_at_A_opt"] = {"mean": float(np.mean(x)), "median": float(np.median(x)), "p90": float(np.percentile(x, 90))}
    summary["budget_curve"] = [{"dr": dr, **{v: float(np.median([p["budget"][i][v] for p in per_user])) for v in VARIANTS},
                                **{f"{v}_learned": float(np.median([p["budget"][i][f"{v}_learned"] for p in per_user])) for v in ("A_fixed_head", "B_recall_cost")}}
                               for i, dr in enumerate(BUDGET_DRS)]
    cms = [cm for _, cm in jobs]
    summary["cost_models"] = {"users": len(cms),
                              "fixed_head_median_s": [float(np.median([c["fixed_head"][g] for c in cms])) for g in range(4)],
                              "fixed_actual_median_s": [float(np.median([c["fixed_actual"][g] for c in cms])) for g in range(4)],
                              "learn_median_s": float(np.median([c["learn"] for c in cms])), "relearn_median_s": float(np.median([c["relearn"] for c in cms]))}
    pts = [0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
    summary["recall_cost_curve"] = [{"recall": p, **{f"grade_{g + 1}_s": float(np.median([np.interp(p, c["x"], c["c"][:, g]) for c in cms if len(c["x"])])) for g in range(4)},
                                     **{f"mix_{nm}": float(np.median([np.interp(p, c["x"], c["mix_r"][:, j]) for c in cms if len(c["x"])])) for j, nm in enumerate(("hard", "good", "easy"))}}
                                    for p in pts]
    return summary, per_user


def _low():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------------------ report
def fmt(x, nd=3):
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    return f"{x:.{nd}f}"


def md_table(rows, cols, heads):
    out = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    for r in rows:
        out.append("| " + " | ".join(fmt(r[c][0], r[c][1]) if isinstance(r[c], tuple) else fmt(r[c]) for c in cols) + " |")
    return out


def analyze(args):
    from scripts import compress_common as cc
    _low()
    out = Path(args.out_dir); res = Path(args.result_dir); res.mkdir(parents=True, exist_ok=True)
    data = cc.data_path()
    index = [json.loads(l) for l in (out / "query.jsonl").read_text().splitlines()]
    ok = [r for r in index if "error" not in r]
    errors = [r for r in index if "error" in r]
    files = sorted((out / "users").glob("*.npz"))
    started = time.time()
    with Pool(args.procs, initializer=_low) as pool:
        users = pool.starmap(load_user, [(str(f), str(data)) for f in files])
    # identity check vs the SM82c recorded rows (every user)
    from scripts.sm82s2_curves import SM82C_ROWS
    ref = {json.loads(l)["user_id"]: json.loads(l)["model"] for l in SM82C_ROWS.read_text().splitlines() if '"head": "model"' in l and "error" not in l}
    gaps = [abs(r["identity_sums"]["ll"] - ref[r["user_id"]]["ll"]) / abs(ref[r["user_id"]]["ll"]) for r in ok if r["user_id"] in ref]
    same_n = all(r["identity_sums"]["n"] == ref[r["user_id"]]["n"] for r in ok if r["user_id"] in ref)
    curves, per_user = curve_tables(users)
    result = {"name": "response-time prediction (SM82) stage 2", "users": len(users), "errors": errors,
              "test_reviews": int(sum(len(z["th"]) for z in users)), "grid_reviews": int(sum(len(z["sub"]) for z in users)),
              "identity_check": {"users": len(gaps), "max_rel_ll_gap_vs_sm82c": float(max(gaps)) if gaps else None, "same_review_counts": same_n},
              "query_seconds": float(sum(r.get("query_seconds", 0) for r in ok)),
              "curves": curves, "recall_bins": recall_bins(users), "slopes": slopes(users), "user_types": user_types(users, per_user),
              "calibration": calibration(users), "patterns": patterns(users)}
    wl, wl_users = workload(users, args.procs)
    for p_ in wl_users:
        p_.pop("free", None)
    result["workload"] = wl
    result["analysis_seconds"] = round(time.time() - started, 1)
    (res / "stage2-result.json").write_text(json.dumps(result, indent=1, default=float))
    (res / "stage2-workload-users.json").write_text(json.dumps(wl_users, default=float))
    (res / "stage2-result.md").write_text("\n".join(render(result)) + "\n", encoding="utf-8")
    print(f"[sm82s2-analyze] {len(users)} users, {result['test_reviews']:,} test reviews -> {res / 'stage2-result.md'} "
          f"({result['analysis_seconds']} s)", flush=True)
    return 0


def render(r):
    L = ["# Response-time prediction (SM82) stage 2: answer-time curves, fast-Good patterns, time-aware workload", ""]
    L += [f"Model: the frozen released small model (the leak-free small model (SM72s) 25 epochs + the 1-epoch 32k Muon cool-down) + the answer-time head on richer frozen features (SM82c), grade bitwise the released model's. Users: {r['users']} sampled from the 732 set (minus never-fit), {r['test_reviews']:,} test reviews on a later day than the card's previous one; curve grid on up to 3,000 random test reviews per user ({r['grid_reviews']:,}). Query: only the review's instant (query) row changes (the card's interval, cumulative interval, review day encodings and day of week; the head's elapsed-days feature); every earlier review, the session counters and the user's running time statistics stay as observed. Identity point (x1) vs the recorded time rows of the answer-time head on richer frozen features (SM82c): max relative LogLoss gap {fmt(r['identity_check']['max_rel_ll_gap_vs_sm82c'], 8)} over {r['identity_check']['users']} users, same review counts {r['identity_check']['same_review_counts']}. Errors: {len(r['errors'])}.", ""]
    L += ["## Takeaways", "", "TAKEAWAYS", ""]
    L += ["## 1. Answer-time curves", "", "User-averaged over the grid reviews (each user equal weight). Recall = 1 - P(Again) from the instant head; E time = expected recorded time (grade-mixed by the grade head, bin centres clipped at the user's cap); E time | pass = mixed over Hard/Good/Easy; P(fast Good) = P(Good) x P(under 3.9 s | Good). Fast = under 3,891 ms = the head's bins 0-6: the quantity the response-time prediction (SM82) and answer-time head on richer frozen features (SM82c) tables label 'under 5.5 s' (their bin 6 ends at 3.89 s; 5.48 s is the end of bin 7).", ""]
    rows = [{**c, "t": f"{c['t_median']:g}"} for c in r["curves"]]
    L += md_table(rows, ["grid", "t", "recall", "E", "E_pass", "E_again", "P_good", "P_fast_good", "E_vs_x1", "E_pass_vs_x1", "P_fast_good_vs_x1"],
                  ["query interval", "median days", "recall", "E time (s)", "E time pass (s)", "E time Again (s)", "P(Good)", "P(fast Good)", "E time vs x1", "E time pass vs x1", "P(fast Good) vs x1"])
    s = r["slopes"]
    L += ["", f"Per review over the 8 interval multipliers, slope of the review's own curve (per user the median over its reviews, then the median over users): E time | pass rises by {fmt(s['E_pass_s_per_0.1_recall_drop_median'])} s per 0.1 drop in recall (user quartiles {', '.join(fmt(x) for x in s['E_pass_slope_user_quartiles'])}); grade-mixed E time by {fmt(s['E_s_per_0.1_recall_drop_median'])} s; elasticity d ln E / d ln t = {fmt(s['elasticity_lnE_lnt_median'])} (user quartiles {', '.join(fmt(x) for x in s['elasticity_user_quartiles'])}).", ""]
    L += ["Time vs predicted recall (every grid point of every review pooled per user, then user-averaged; users with at least 30 points in the bin):", ""]
    L += md_table(r["recall_bins"], ["recall_bin", "users", "points", "E", "E_pass", "E_again", "P_fast_given_good", "P_fast_good"],
                  ["predicted recall", "users", "points", "E time (s)", "E time pass (s)", "E time Again (s)", "P(under 3.9 s given Good)", "P(fast Good)"])
    L += ["", "User types (thirds of users by their median recorded time, and by their observed recall):", ""]
    ut = [{**u, "rng": f"{u['range'][0]:.2f}-{u['range'][1]:.2f}"} for u in r["user_types"]]
    L += md_table(ut, ["split", "group", "users", "rng", "recall_x0.5", "recall_x1", "recall_x4", "E_pass_x0.5", "E_pass_x1", "E_pass_x2", "E_pass_x4", "P_fast_good_x1", "P_fast_good_x4", "recall_d1", "recall_d365"],
                  ["split", "group", "users", "range", "recall x0.5", "recall x1", "recall x4", "E pass x0.5 (s)", "E pass x1 (s)", "E pass x2 (s)", "E pass x4 (s)", "P(fast Good) x1", "P(fast Good) x4", "recall at 1 d", "recall at 365 d"])
    L += ["", "### Calibration on the intervals the scheduler actually chose", "", "All test reviews of the sampled users at their real interval (x1). Interval growth = this interval / the card's previous later-day interval: the high-growth buckets are the observed reviews most like the long-interval extrapolation.", ""]
    for key, title in (("by_interval_growth", "interval growth"), ("by_interval_days", "interval (days)")):
        L += md_table(r["calibration"][key], ["bucket", "reviews", "users", "recall_pred", "recall_actual", "time_pred_s", "time_actual_s", "time_pred_true_grade_s", "fast_true_grade_pred", "fast_actual", "fast_good_pred", "fast_good_actual", "good_under_5_5s_actual"],
                      [title, "reviews", "users", "recall predicted", "recall actual", "time predicted (s)", "time actual (s)", "time predicted given the true grade (s)", "under 3.9 s given the true grade predicted", "under 3.9 s actual", "fast Good predicted", "fast Good actual", "Good under 5.5 s actual"])
        L += [""]
    L += ["## 2. Fast-Good patterns (descriptive)", "", "Fast Good = rated Good in under 3.9 s (the head's bins 0-6; the Good-under-5.5 s share is shown for the actual data). Pooled over the sampled users' test reviews; user avg = mean over users with at least 50 reviews in the bucket (removes part of the between-user mix). Correlations only: the scheduler chose these intervals.", ""]
    for fname, rows in r["patterns"].items():
        L += [f"**{fname}**", ""]
        L += md_table(rows, ["bucket", "reviews", "recall", "fast_good_share", "fast_good_share_user_avg", "fast_given_good", "good_under_5_5s", "mean_time_s", "mean_time_s_user_avg", "fast_good_pred"],
                      ["bucket", "reviews", "recall", "fast Good share", "fast Good share (user avg)", "fast given Good", "Good under 5.5 s share", "mean time (s)", "mean time (s, user avg)", "fast Good predicted"])
        L += [""]
    w = r["workload"]
    cm = w["cost_models"]
    L += ["## 3. Workload: fixed per-grade costs vs recall-dependent predicted costs", "",
          f"Simulator: an FSRS-6 card simulation with fsrs_optimizer's formulas and each user's own FSRS-6 parameters (the srs-benchmark FSRS-6 recency-weighted fit, last split; {w["fsrs_parameters"]["user_fit"]} users, default parameters for {w["fsrs_parameters"]["default"]}; median decay {fmt(w["fsrs_parameters"]["decay_median"])}), one same-day relearning step rated Good after each lapse, no fuzz, common random numbers across cost models; 150 users = each sampled user's own cost model. Per user: learning cost = the user's mean recorded first-review time (median {fmt(cm['learn_median_s'], 1)} s), relearning step = the user's mean same-day review time (median {fmt(cm['relearn_median_s'], 1)} s), first-rating mix from the user's first reviews, pass-rating mix from the user's test reviews. The model is not run inside the simulator: its answer-time head is turned into a per-user, per-grade cost curve over recall (the cheapest honest version).",
          "",
          f"- A (actual): fixed per-grade review costs = the user's mean recorded time per grade (median over users Again/Hard/Good/Easy {', '.join(fmt(x, 1) for x in cm['fixed_actual_median_s'])} s): the classic setup.",
          f"- A (head): fixed per-grade costs = the head's mean per grade at the observed intervals ({', '.join(fmt(x, 1) for x in cm['fixed_head_median_s'])} s).",
          "- B: per-grade cost = the head's expected time for that grade at the card's current recall (per user, binned over every grid point of every review, linear interpolation in recall).",
          "- C: B + the pass-rating mix (Hard/Good/Easy given recall) from the grade head (changes the stability updates).",
          "", "Head cost curve (median over users):", ""]
    L += md_table(w["recall_cost_curve"], ["recall", "grade_1_s", "grade_2_s", "grade_3_s", "grade_4_s", "mix_hard", "mix_good", "mix_easy"],
                  ["recall", "Again (s)", "Hard (s)", "Good (s)", "Easy (s)", "Hard share of passes", "Good share", "Easy share"])
    L += ["", "### Objective (a) with no time limit: time per memorized card at a given recall level", "",
          "10 new cards a day for 365 days (3,650 cards), every due review done. Time per memorized card = (first reviews + reviews + relearning steps) / the sum of recall over all cards at day 365. A and B share the same review trajectory (costs do not change the schedule without a time limit), so their ratio is purely the cost shape. Median over users, seconds:", ""]
    L += md_table(w["free_targets"], ["dr", "memorized_share", "A_fixed_actual", "A_fixed_head", "B_recall_cost", "C_recall_cost_and_mix", "B_over_A_head", "C_over_A_head"],
                  ["desired retention", "memorized share at day 365", "A (actual)", "A (head)", "B", "C", "B / A (head)", "C / A (head)"])
    fo = w["free_opt"]
    L += ["", "Cost relative to desired retention 0.90 (median over users of the user's ratio):", ""]
    L += md_table(w["free_relative_to_0_90"], ["dr", "A_fixed_actual", "A_fixed_head", "B_recall_cost", "C_recall_cost_and_mix"],
                  ["desired retention", "A (actual)", "A (head)", "B", "C"])
    L += ["", f"With no time limit this ratio keeps falling down to desired retention 0.50 for every cost model (users at the 0.50 bound: A (actual) {fo['A_fixed_actual']['share_at_lower_bound']:.0%}, A (head) {fo['A_fixed_head']['share_at_lower_bound']:.0%}, B {fo['B_recall_cost']['share_at_lower_bound']:.0%}, C {fo['C_recall_cost_and_mix']['share_at_lower_bound']:.0%}): with the FSRS-6 power curve (the users' own fits), cards left unreviewed keep much of their recall within a year, so this ratio has no interior optimum and the optimum cannot move. The comparison that matters here is the cost shape (B / A above).", ""]
    bo = w["budget_opt"]
    L += ["### Objective (a) as most recall for a given time: a 30-minute daily limit", "",
          "Daily limit 1,800 s (fsrs_optimizer's default), charged with each cost model's own costs: due reviews first, the rest wait a day, then new cards while time is left (unlimited deck); 365 days; desired retention 0.60-0.98 by 0.02; the optimum = the most memorized cards (sum of recall) at day 365. Each cost model runs its own simulation.", ""]
    vr = [{"variant": v, **bo[v], "q": f"{bo[v]['opt_dr_quartiles'][0]:.2f}-{bo[v]['opt_dr_quartiles'][1]:.2f}"} for v in w["variants"]]
    L += md_table(vr, ["variant", "opt_dr_median", "opt_dr_mean", "q", "max_memorized_median", "share_at_bounds"],
                  ["costs", "optimal desired retention, median", "mean", "quartiles", "memorized at day 365 at the optimum (median)", "users with the optimum at a grid bound"])
    L += [""]
    for v in ("A_fixed_actual", "B_recall_cost", "C_recall_cost_and_mix"):
        s = w[f"budget_shift_{v}_minus_A_head"]
        L += [f"- optimum {v} minus A (head): mean {s['mean']:+.3f}, median {s['median']:+.2f}; higher for {s['share_up']:.0%} of users, lower {s['share_down']:.0%}, same {s['share_same']:.0%}."]
    for v in ("B_recall_cost", "C_recall_cost_and_mix"):
        s = w[f"budget_loss_{v}_at_A_opt"]
        L += [f"- if the costs are really {v}, keeping the fixed-cost optimum memorizes {s['mean']:.2%} fewer cards on average (median {s['median']:.2%}, 90th percentile {s['p90']:.2%})."]
    L += ["", "Memorized cards at day 365 by desired retention (median over users):", ""]
    L += md_table(w["budget_curve"][::2] + [w["budget_curve"][-1]], ["dr", "A_fixed_actual", "A_fixed_head", "B_recall_cost", "C_recall_cost_and_mix", "A_fixed_head_learned", "B_recall_cost_learned"],
                  ["desired retention", "A (actual)", "A (head)", "B", "C", "cards learned, A (head)", "cards learned, B"])
    L += ["", "Secondary objective (b), time to fluency: not computed (it needs the card-level model inside the simulator; the head's P(fast Good) depends on the card's history, which a recall-only cost curve does not carry)."]
    L += ["", "## Limits", ""] + ["- Counterfactual query: only the review's instant row moves to the new interval. The session counters, the reviews and new cards between this card's reviews, and the user's running time statistics stay as observed, so a query far from the real interval is an extrapolation; the calibration tables check the observed intervals only (by interval length and by interval growth, the closest observed analogue of a stretched interval).", '- The absolute-day grid (1-365 days) mixes cards whose real interval is far from the query (a card really seen after 58 days queried at 1 day); the multiplier grid (0.25x-4x) stays closer to what was observed.', '- Fast = under 3.9 s (the head\'s bins 0-6), not 5.5 s: the response-time prediction (SM82) and answer-time head on richer frozen features (SM82c) tables label that same quantity "under 5.5 s", but their bin 6 ends at 3,891 ms (5,477 ms is the end of bin 7). The curve queries stored bins 0-6 only; a 5.5 s model curve needs a re-query that also stores bins 0-7 (about 25 min of GPU). The actual-data columns show both thresholds.', "- Workload: the RWKV model does not run inside the simulator (too heavy for today). Its answer-time head becomes a per-user, per-grade cost curve over recall, applied to the simulator's FSRS-6 recall. The recall bins pool different cards, so part of the slope is card difficulty (hard cards are both slower and less recalled), not only the same card getting slower; the within-review slopes in section 1 are the same-card part.", "- The memory model in the simulator is FSRS-6 (the users' own fits), not the RWKV model; learning steps are a constant cost per new card and each lapse gets one relearning step. Both objectives push the optimum to the bottom of the desired-retention range for most users, so whether the optimum moves can only be read at the margin; the robust result is the change in the cost shape.", '- 150 users of the 732 set (seeded sample); the curve grid uses up to 3,000 random test reviews per user; patterns are descriptive (the scheduler chose the intervals) and pooled tables mix users (the user-averaged columns remove part of that).', '- Objective (b), time to fluency, is not computed.'] + [""]
    return L
