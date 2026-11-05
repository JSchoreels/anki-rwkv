#!/usr/bin/env python
"""The real-data check of the time-aware target (SM41r), for the small model in the review simulator (SM41).

Question (user 11-05): offline, on real benchmark review histories, are slow-answer cards actually cheaper to review
late? Plan, metric and decision rule: artifacts/small-model/sm41/realcheck-result.md (fixed before the runs).

No simulation. Every review is a REAL review of the full-run users (fullrun/users.json) in the 60 days after each
split day (the part the simulator replaced). The released small model is run teacher-forced on the real schedule:
the engine's `tf` lane alone (scripts/sm41_engine.py, imported, not changed), with its per-review diagnostic log on.
Two read-only hooks in THIS process record what the engine already computes for the tf lane:
  - the instant head's grade probabilities at the show moment (the query step, `vstep(snaps=True)`);
  - the answer-time head's 16-bin distribution per grade on later-day rows (`sm41_drift.time_probs_batch`), reduced
    to E[ms | grade] and E[ln ms | grade].
The real rows come from the same loader (sm41_sim.load_user via sm41_engine.setup); the log rows are checked against
them one by one (card, rating, answer ms, kind).

  run      --user U --splits D1,D2 --out-dir DIR   -> DIR/<user>-<split>.npz (resumable, quiet.json holds)
  analyze  --logs DIR --out-dir OUT                -> OUT/realcheck-tables.md, OUT/realcheck.json, OUT/per-user.csv
"""
from argparse import ArgumentParser
import datetime as dt
import glob
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("RWKV_NO_JIT", "1")

HERE = ROOT / "artifacts" / "small-model" / "sm41"
OUT = HERE / "realcheck"


# ------------------------------------------------------------------ run (teacher-forced tf lane + hooks)
def run(args):
    from scripts import sm41_sim as S
    from scripts import sm41_drift as DR
    from scripts import sm41_engine as E
    S.low_priority()
    user = args.user
    splits = sorted(int(x) for x in args.splits.split(","))
    od = Path(args.out_dir); od.mkdir(parents=True, exist_ok=True)
    todo = [D for D in splits if not (od / f"{user}-{D}.npz").exists()]
    if not todo:
        print(f"[sm41-realcheck] user {user}: all splits done", flush=True)
        return
    DR.wait_clear(f"user {user}")
    model = S.Model(threads=1)
    cap = {"q": [], "t": []}
    vstep0, tpb0 = E.vstep, DR.time_probs_batch

    def vstep_hook(m, X, st, fresh, snaps=False, curve=False):
        out = vstep0(m, X, st, fresh, snaps=snaps, curve=curve)
        if snaps:
            p = out[0]
            cap["q"].append(p / p.sum(-1, keepdims=True))
        return out

    def tpb_hook(m, frozen, explicit):
        tp = tpb0(m, frozen, explicit)
        mid = (m.lo + m.hi) / 2
        # [k, 4 grades, 2]: E[s | grade] (seconds, despite the saved name t_ms) and E[ln s | grade]
        cap["t"].append(np.stack([(tp * np.exp(mid)).sum(-1), (tp * mid).sum(-1)], -1))
        return tp

    E.vstep, DR.time_probs_batch = vstep_hook, tpb_hook
    cache = {}
    fits = args.fits or str(HERE / "fullrun" / "fits" / f"{user}.jsonl")
    for D in todo:
        t0 = time.perf_counter(); c0 = time.process_time()
        w0 = cache[user].seconds if user in cache else 0.0
        warm, fsrs, src = E.setup(model, user, D, cache, fits)
        DR.wait_clear(f"user {user} split {D}")
        eng = E.Engine(model, warm, D, args.days, ["tf"], fsrs=fsrs, fsrs_source=src, log=True)
        cap["q"].clear(); cap["t"].clear()
        res = eng.run()
        L = eng.log_arrays()
        n = len(L["kind"])
        kind = L["kind"].astype(int)
        rev = warm.rev
        lo, hi = eng.end, eng.stop
        # check: the tf lane's log rows are the real window rows, in order
        cid = np.asarray(eng.cards, np.int64)[L["card"].astype(int)]
        ok = (hi - lo == n and np.array_equal(cid, rev["card_id"][lo:hi])
              and np.array_equal(L["rating"].astype(int), rev["rating"][lo:hi])
              and np.array_equal(L["ms"].astype(np.int64), rev["duration"][lo:hi])
              and np.array_equal(kind, np.where(rev["elapsed_days"][lo:hi] == -1, 0,
                                                np.where(rev["elapsed_days"][lo:hi] == 0, 1, 2))))
        if not ok:
            raise RuntimeError(f"user {user} split {D}: log rows do not match the real window rows")
        q = np.concatenate(cap["q"]) if cap["q"] else np.zeros((0, 4))
        t = np.concatenate(cap["t"]) if cap["t"] else np.zeros((0, 4, 2))
        if len(q) != (kind != 0).sum() or len(t) != (kind == 2).sum():
            raise RuntimeError(f"user {user} split {D}: hook rows {len(q)}/{len(t)} vs log "
                               f"{(kind != 0).sum()}/{(kind == 2).sum()}")
        pg = np.full((n, 4), np.nan); pg[kind != 0] = q
        if not np.allclose(1 - pg[kind != 0, 0], L["p_true"][kind != 0], atol=1e-9):
            raise RuntimeError(f"user {user} split {D}: hooked P(pass) != logged P(pass)")
        tE = np.full((n, 4, 2), np.nan); tE[kind == 2] = t
        # the card's last later-day answer before each window row (pre-split history included), as the rule reads it,
        # and the user's median later-day answer time over the 60 days before the split (the engine's med_ms)
        prev_ms = np.full(n, np.nan)
        last = {}
        ed_all, dur_all, card_all = rev["elapsed_days"], rev["duration"], rev["card_id"]
        for i in range(hi):
            if i >= lo:
                prev_ms[i - lo] = last.get(int(card_all[i]), np.nan)
            if ed_all[i] > 0:
                last[int(card_all[i])] = float(dur_all[i])
        lat = ed_all[:lo] > 0
        recent = lat & (rev["day_offset"][:lo] >= D - 60)
        med_ms = float(np.median(dur_all[:lo][recent if recent.sum() >= 100 else lat]))
        L.update({"pg": pg, "t_ms": tE[..., 0], "t_ln": tE[..., 1], "prev_later_ms": prev_ms, "med_ms": med_ms,
                  "es_real": rev["elapsed_seconds"][lo:hi].astype(np.float64), "user": user, "split": D,
                  "wall_seconds": time.perf_counter() - t0, "cpu_seconds": time.process_time() - c0,
                  "warmup_seconds": warm.seconds - w0, "window_reviews": n})
        np.savez_compressed(od / f"{user}-{D}.npz", **L)
        print(f"[sm41-realcheck] user {user} split {D}: {n} real reviews, engine {res['wall_seconds']} s "
              f"({n / max(res['wall_seconds'], 1e-9):.0f} reviews/s), warm-up {warm.seconds - w0:.0f} s, "
              f"cpu {time.process_time() - c0:.0f} s, {dt.datetime.now():%H:%M:%S}", flush=True)
        del eng


# ------------------------------------------------------------------ analysis helpers
def tci(x):
    """Mean and 95% t half-width over users (NaN dropped)."""
    from scipy import stats
    x = np.asarray(x, np.float64); x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan"), float("nan"), len(x)
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))), len(x)


def fmt(m, h, d=2):
    return "—" if not np.isfinite(m) else f"{m:+.{d}f} ± {h:.{d}f}"


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


RBANDS = [0.0, 0.70, 0.80, 0.85, 0.90, 0.95, 1.01]
RB_TXT = ["< 0.70", "0.70-0.80", "0.80-0.85", "0.85-0.90", "0.90-0.95", ">= 0.95"]
LNBINS = [-np.inf, -0.5, -0.17, 0.17, 0.5, np.inf]
LN_TXT = ["< -0.5", "-0.5..-0.17", "-0.17..0.17", "0.17..0.5", ">= 0.5"]
STRATA = ["fast", "mid", "slow"]


def load_reviews(logs):
    """One table of the later-day reviews of every (user, split), with each review's outcomes."""
    import pandas as pd
    parts, meta = [], []
    for path in sorted(glob.glob(str(Path(logs) / "*.npz"))):
        z = np.load(path)
        kind = z["kind"].astype(int); day = z["day"].astype(int); card = z["card"].astype(int)
        ms = np.minimum(z["ms"], 60000.0); rating = z["rating"].astype(int)
        n = len(kind)
        # same-day minutes that follow each review of a card on its day (learning / relearning steps)
        follow_s = np.zeros(n); nxt = np.full(n, -1)
        nxt_later = np.full(n, -1)
        last_any, last_later = {}, {}
        acc = {}
        for i in range(n - 1, -1, -1):
            c = int(card[i])
            key = (c, int(day[i]))
            follow_s[i] = acc.get(key, 0.0)
            if kind[i] == 1:
                acc[key] = acc.get(key, 0.0) + ms[i] / 1000.0
            else:
                acc[key] = 0.0                       # a later-day review starts the card's day
            nxt[i] = last_any.get(c, -1)
            nxt_later[i] = last_later.get(c, -1)
            last_any[c] = i
            if kind[i] == 2:
                last_later[c] = i
        sel = np.nonzero(kind == 2)[0]
        u, D = int(z["user"]), int(z["split"])
        cpn, cpa = z["card_pre_later_n"], z["card_pre_later_again"]
        j = nxt_later[sel]
        has = j >= 0
        d = {"user": u, "split": D, "day": day[sel], "card": card[sel], "rating": rating[sel],
             "passed": (rating[sel] > 1).astype(float), "ms": ms[sel], "ed": z["ed"][sel],
             "p": z["p_true"][sel], "r7": z["r_fsrs7"][sel], "rc": z["r_curve"][sel],
             "t90b": z["t90_before"][sel], "t90a": z["t90_after"][sel],
             "prev_ms": z["prev_later_ms"][sel], "med_ms": float(z["med_ms"]),
             "follow_s": follow_s[sel],
             # t_ms holds E[s | grade] in SECONDS (the head's bins are in ln s); converted to ms here
             "tE_real": 1000 * np.take_along_axis(z["t_ms"][sel], (rating[sel] - 1)[:, None], 1)[:, 0],
             "tL_real": np.take_along_axis(z["t_ln"][sel], (rating[sel] - 1)[:, None], 1)[:, 0],
             "tE_mix": 1000 * (z["pg"][sel] * z["t_ms"][sel]).sum(-1),
             "diff_n": cpn[card[sel]], "diff_a": cpa[card[sel]],
             "next_ed": np.where(has, z["ed"][np.maximum(j, 0)], np.nan),
             "next_pass": np.where(has, (z["rating"][np.maximum(j, 0)] > 1).astype(float), np.nan),
             "next_ms": np.where(has, np.minimum(z["ms"][np.maximum(j, 0)], 60000.0), np.nan),
             "next_p": np.where(has, z["p_true"][np.maximum(j, 0)], np.nan),
             "next_tE": np.where(has, 1000 * (z["pg"] * z["t_ms"]).sum(-1)[np.maximum(j, 0)], np.nan)}
        parts.append(pd.DataFrame(d))
        meta.append({"user": u, "split": D, "window_reviews": int(z["window_reviews"]),
                     "later": int(len(sel)), "wall": float(z["wall_seconds"]), "cpu": float(z["cpu_seconds"]),
                     "warm": float(z["warmup_seconds"])})
    df = pd.concat(parts, ignore_index=True)
    df["lr"] = np.log(np.maximum(df["prev_ms"], 100.0) / df["med_ms"])      # the rule's ln ratio
    df["lnms"] = np.log(np.maximum(df["ms"], 100.0) / df["med_ms"])
    df["cost_s"] = df["ms"] / 1000.0 + df["follow_s"]                       # the review + its same-day steps
    df["lapse_min"] = np.where(df["passed"] == 0, df["follow_s"] / 60.0, np.nan)
    df["dgain"] = np.log(np.maximum(df["t90a"], 1e-3)) - np.log(np.maximum(df["t90b"], 1e-3))
    df["rband"] = np.digitize(df["p"], RBANDS[1:-1])
    df["lnbin"] = np.digitize(df["lr"], LNBINS[1:-1])
    # within-user terciles of the ln ratio (cut points over the user's own later-day reviews with a previous one)
    df["strat"] = -1
    for u, g in df[np.isfinite(df["lr"])].groupby("user"):
        q1, q2 = np.quantile(g["lr"], [1 / 3, 2 / 3])
        df.loc[g.index, "strat"] = np.digitize(g["lr"], [q1, q2])
    # card difficulty from its pre-split history (later-day Again share, shrunk to the user's mean with 4 pseudo-reviews)
    um = df.groupby("user").apply(lambda g: g["diff_a"].sum() / max(g["diff_n"].sum(), 1))
    df["diff"] = (df["diff_a"] + 4 * df["user"].map(um)) / (df["diff_n"] + 4)
    return df, pd.DataFrame(meta)


def per_user(df, f, min_n=30):
    """f(group) -> float for every user with at least min_n rows; a Series indexed by user."""
    import pandas as pd
    out = {}
    for u, g in df.groupby("user"):
        if len(g) >= min_n:
            out[u] = f(g)
    return pd.Series(out, dtype=float)


def matched(g, col, strata_cols=("rband",), weight_from=None):
    """Mean of `col` per stratum, averaged with weights = the pooled row share of each stratum (direct
    standardisation), so two groups are compared at the same predicted-recall mix."""
    w = weight_from if weight_from is not None else g.groupby(list(strata_cols)).size()
    m = g.groupby(list(strata_cols))[col].mean()
    k = m.index.intersection(w.index)
    if len(k) == 0:
        return float("nan")
    return float((m[k] * w[k]).sum() / w[k].sum())


# ------------------------------------------------------------------ analyze
def analyze(args):
    import pandas as pd
    import statsmodels.api as sm
    od = Path(args.out_dir); od.mkdir(parents=True, exist_ok=True)
    df, meta = load_reviews(args.logs)
    J, L = {}, ["# The real-data check of the time-aware target (SM41r): tables", ""]
    has = df[df["strat"] >= 0].copy()
    users = sorted(df["user"].unique())
    L += [f"{len(users)} users, {len(meta)} splits, {len(df):,} real later-day reviews in the 60-day windows "
          f"({len(has):,} with a previous later-day answer); {int(meta['window_reviews'].sum()):,} window reviews in "
          "all. Strata: the rule's ln(last later-day answer ms / user's pre-split median), within-user terciles "
          "(fast / mid / slow). Recall bands: the instant head's P(pass) at the real show moment. Every number: "
          "per user first, then mean ± 95% t CI over users.", ""]
    J["n"] = {"users": len(users), "splits": len(meta), "later": len(df), "later_with_prev": len(has)}

    # ---- 1. calibration by stratum and recall band
    L += ["## 1. Calibration of the instant head by previous answer time", ""]
    L += ["Realised pass rate - predicted P(pass), points (per user, mean ± CI over users). FSRS-7 (pre-split refit) "
          "on the same reviews for reference.", "",
          "| stratum | reviews | mean ln ratio | predicted | realised | instant head: realised - predicted | FSRS-7: "
          "realised - predicted |", "|---|---|---|---|---|---|---|"]
    J["calib"] = {}
    for s, nm in enumerate(STRATA):
        g = has[has["strat"] == s]
        a = per_user(g, lambda x: 100 * (x["passed"].mean() - x["p"].mean()))
        b = per_user(g, lambda x: 100 * (x["passed"].mean() - x["r7"].mean()))
        ma, ha, _ = tci(a); mb, hb, _ = tci(b)
        J["calib"][nm] = {"ih": [ma, ha], "fsrs7": [mb, hb]}
        L.append(f"| {nm} | {len(g):,} | {g['lr'].mean():+.2f} | {g['p'].mean():.3f} | {g['passed'].mean():.3f} | "
                 f"{fmt(ma, ha, 1)} | {fmt(mb, hb, 1)} |")
    for lab, col in (("instant head", "p"), ("FSRS-7", "r7")):
        d = per_user(has, lambda x: 100 * ((x[x.strat == 2]["passed"] - x[x.strat == 2][col]).mean()
                                           - (x[x.strat == 0]["passed"] - x[x.strat == 0][col]).mean()))
        m, h, k = tci(d)
        J["calib"][f"slow_minus_fast_{col}"] = [m, h, k]
        L.append(f"\nSlow minus fast, {lab}: {fmt(m, h, 1)} points ({k} users; > 0 = the model is too pessimistic "
                 "about slow cards relative to fast ones).")
    L += ["", "By recall band (instant head, realised - predicted, points):", "",
          "| band | " + " | ".join(STRATA) + " | slow - fast |", "|---|---|---|---|---|"]
    J["calib_band"] = {}
    for b, bt in enumerate(RB_TXT):
        cells, vals = [], {}
        for s, nm in enumerate(STRATA):
            g = has[(has["strat"] == s) & (has["rband"] == b)]
            v = per_user(g, lambda x: 100 * (x["passed"].mean() - x["p"].mean()), min_n=10)
            vals[s] = v
            m, h, _ = tci(v); cells.append(f"{fmt(m, h, 1)} ({len(g):,})")
        d = (vals[2] - vals[0]).dropna()
        m, h, k = tci(d)
        J["calib_band"][bt] = [m, h, k]
        L.append(f"| {bt} | " + " | ".join(cells) + f" | {fmt(m, h, 1)} ({k} users) |")
    # recalibration slope: per user, logistic pass ~ logit(p) + delta * lr (delta in logit units per ln ratio)
    def recal(g, col):
        if g["passed"].nunique() < 2 or len(g) < 200:
            return float("nan")
        X = np.column_stack([np.ones(len(g)), logit(g[col].to_numpy()), g["lr"].to_numpy()])
        try:
            r = sm.GLM(g["passed"].to_numpy(), X, family=sm.families.Binomial()).fit()
            return float(r.params[2])
        except Exception:
            return float("nan")
    d_ih = per_user(has, lambda g: recal(g, "p"), min_n=200)
    d_7 = per_user(has, lambda g: recal(g, "r7"), min_n=200)
    m, h, k = tci(d_ih); m7, h7, k7 = tci(d_7)
    J["recal_delta"] = {"ih": [m, h, k], "fsrs7": [m7, h7, k7]}
    L += ["", f"Recalibration slope delta (per user logistic: pass ~ logit(pred) + delta x ln ratio, logit per unit "
          f"ln ratio): instant head {fmt(m, h, 3)} ({k} users), FSRS-7 {fmt(m7, h7, 3)} ({k7} users). A positive "
          "delta for the instant head = it over-penalises slow previous answers; the rule's beta then partly "
          "corrects a miscalibration (its share = delta / beta).", ""]

    # ---- 2. cost by stratum at matched predicted recall; persistence; time head
    L += ["## 2. Cost of a review by previous answer time", ""]
    W = has.groupby("rband").size()
    J["cost"] = {}
    rows = []
    for col, lab, sc in (("ms", "seconds of the review", 1 / 1000.0), ("cost_s", "seconds incl. same-day steps", 1.0),
                         ("tE_real", "answer-time head E[s | real grade]", 1 / 1000.0),
                         ("tE_mix", "answer-time head E[s] (grade-mixed)", 1 / 1000.0)):
        vals = {s: per_user(has[has["strat"] == s], lambda g: sc * matched(g, col, weight_from=W)) for s in range(3)}
        J["cost"][col] = {}
        cells = []
        for s in range(3):
            m, h, _ = tci(vals[s]); cells.append(f"{m:.2f} ± {h:.2f}"); J["cost"][col][STRATA[s]] = [m, h]
        rr = (vals[2] / vals[0] - 1).dropna() * 100
        m, h, k = tci(rr); J["cost"][col]["slow_vs_fast_pct"] = [m, h, k]
        rows.append(f"| {lab} | " + " | ".join(cells) + f" | {fmt(m, h, 1)} |")
    L += ["Per review, standardised to the pooled recall-band mix (so slow and fast are compared at the same "
          "predicted recall):", "", "| quantity | fast | mid | slow | slow vs fast, % |", "|---|---|---|---|---|"] + rows
    # lapse cost
    lap = has[has["passed"] == 0]
    vals = {s: per_user(lap[lap["strat"] == s], lambda g: g["follow_s"].mean() / 60.0, min_n=10) for s in range(3)}
    cells = []
    for s in range(3):
        m, h, _ = tci(vals[s]); cells.append(f"{m:.2f} ± {h:.2f}")
    rr = (vals[2] / vals[0] - 1).dropna() * 100
    m, h, k = tci(rr)
    J["lapse_min"] = {"by_stratum": cells, "slow_vs_fast_pct": [m, h, k]}
    L += ["", "Lapse cost (same-day steps after a later-day Again), minutes per lapse: " +
          ", ".join(f"{STRATA[s]} {cells[s]}" for s in range(3)) + f"; slow vs fast {fmt(m, h, 1)} % ({k} users).",
          ""]
    # time head calibration in strata (real ln s - predicted E[ln s | real grade])
    cells = []
    J["timehead_bias"] = {}
    for s in range(3):
        v = per_user(has[(has["strat"] == s) & np.isfinite(has["tL_real"])],
                     lambda g: float(np.mean(np.log(np.maximum(g["ms"], 100) / 1000.0) - g["tL_real"])))
        m, h, _ = tci(v); cells.append(fmt(m, h, 3)); J["timehead_bias"][STRATA[s]] = [m, h]
    L += ["Answer-time head bias, real ln s - E[ln s | real grade] (0 = calibrated): " +
          ", ".join(f"{STRATA[s]} {cells[s]}" for s in range(3)) + ".", ""]
    # persistence: elasticity of this and the next answer time on the previous ln ratio
    def slope(g, y, x="lr"):
        g = g[np.isfinite(g[y]) & np.isfinite(g[x])]
        if len(g) < 50 or g[x].std() == 0:
            return float("nan")
        return float(np.polyfit(g[x], g[y], 1)[0])
    has["ln_next"] = np.log(np.maximum(has["next_ms"], 100.0) / has["med_ms"])
    has["ln_pred"] = np.log(np.maximum(has["tE_mix"], 100.0) / has["med_ms"])
    e1 = per_user(has, lambda g: slope(g, "lnms")); e2 = per_user(has, lambda g: slope(g, "ln_next"))
    ep = per_user(has, lambda g: slope(g, "ln_pred"))
    # within recall band and difficulty tercile (state-matched elasticity)
    def slope_matched(g, y):
        g = g.copy(); g["dq"] = np.digitize(g["diff"], np.quantile(g["diff"], [1 / 3, 2 / 3]))
        num = den = 0.0
        for _, h in g.groupby(["rband", "dq"]):
            h = h[np.isfinite(h[y])]
            if len(h) >= 20 and h["lr"].std() > 0:
                xc = h["lr"] - h["lr"].mean()
                num += float((xc * (h[y] - h[y].mean())).sum()); den += float((xc ** 2).sum())
        return num / den if den > 0 else float("nan")
    e1m = per_user(has, lambda g: slope_matched(g, "lnms"))
    J["elasticity"] = {k: tci(v) for k, v in (("this", e1), ("next", e2), ("head_pred", ep), ("this_matched", e1m))}
    L += ["Elasticity of answer time on the previous ln ratio (per-user least-squares slope of ln(time / median) on "
          "the ln ratio; 1 = fully persistent, 0 = no memory of the last answer):", "",
          f"- this review, real: {fmt(*J['elasticity']['this'][:2], 3)} "
          f"(within recall band x card-difficulty tercile: {fmt(*J['elasticity']['this_matched'][:2], 3)});",
          f"- this review, the answer-time head's E[s]: {fmt(*J['elasticity']['head_pred'][:2], 3)};",
          f"- the next later-day review, real: {fmt(*J['elasticity']['next'][:2], 3)}.", ""]

    # ---- 3. is a slow card cheaper per unit of memory when reviewed late?
    L += ["## 3. Cost per unit of memory gain vs recall at review, by stratum", ""]
    has["surv_days"] = np.where(np.isfinite(has["next_pass"]), has["next_ed"] * has["next_pass"], np.nan)
    # memory currencies: model's ln t90 growth (curve head, after - before), and the real next interval survived
    tab = {}
    for s in range(3):
        for b in range(len(RB_TXT)):
            g = has[(has["strat"] == s) & (has["rband"] == b)]
            tab[(s, b)] = g
    L += ["Pooled over users (rows), per stratum and recall band: reviews, pass rate, seconds incl. same-day steps, "
          "curve-head ln t90 growth, the real next interval survived (days, next later-day review in the window; 0 "
          "if it failed), and seconds per unit ln t90 growth.", "",
          "| stratum | band | reviews | pass | s/review | d ln t90 | next interval survived, d | s per d ln t90 |",
          "|---|---|---|---|---|---|---|---|"]
    for s in range(3):
        for b in range(len(RB_TXT)):
            g = tab[(s, b)]
            if len(g) < 50:
                continue
            L.append(f"| {STRATA[s]} | {RB_TXT[b]} | {len(g):,} | {g['passed'].mean():.3f} | {g['cost_s'].mean():.1f} | "
                     f"{g['dgain'].mean():+.3f} | {np.nanmean(g['surv_days']):.1f} | "
                     f"{g['cost_s'].mean() / g['dgain'].mean() if g['dgain'].mean() > 0 else float('nan'):.1f} |")
    L.append("")

    # Within user x stratum: the effect of lower recall at review, matched on the card's state.
    # Matching cells: user x stratum x t90-before tercile (within user) x difficulty tercile (within user).
    # Contrast: "late" (P < 0.85) vs "on time" (P >= 0.90) reviews in the same cell; cell-size weights.
    has["sq"] = -1; has["dq"] = -1
    for u, g in has.groupby("user"):
        has.loc[g.index, "sq"] = np.digitize(np.log(np.maximum(g["t90b"], 1e-3)),
                                             np.quantile(np.log(np.maximum(g["t90b"], 1e-3)), [1 / 3, 2 / 3]))
        has.loc[g.index, "dq"] = np.digitize(g["diff"], np.quantile(g["diff"], [1 / 3, 2 / 3]))
    has["late"] = np.where(has["p"] < 0.85, 1, np.where(has["p"] >= 0.90, 0, -1))
    outs = [("passed", "pass rate"), ("cost_s", "seconds incl. steps"), ("ms", "answer ms"),
            ("dgain", "d ln t90 (curve head)"), ("surv_days", "next interval survived, d"),
            ("next_pass", "next review passed"), ("lapse_min", "lapse minutes (lapses only)")]

    def contrast(g, col):
        """late - on-time difference of col within (stratum, sq, dq) cells of one user and one stratum."""
        num = den = 0.0
        for _, h in g.groupby(["sq", "dq"]):
            a = h[h["late"] == 1][col].dropna(); o = h[h["late"] == 0][col].dropna()
            if len(a) >= 3 and len(o) >= 3:
                w = min(len(a), len(o))
                num += w * (a.mean() - o.mean()); den += w
        return num / den if den > 0 else float("nan")
    J["contrast"] = {}
    L += ["### Late (P < 0.85) minus on time (P >= 0.90), within user, matched on the card's state", "",
          "Cells: user x stratum x curve-head t90 tercile x card-difficulty tercile (both within user); "
          "difference of means per cell, weighted by the smaller side; then mean ± CI over users. Slow - fast = the "
          "same contrast, slow minus fast, paired per user.", "",
          "| outcome | fast | mid | slow | slow - fast |", "|---|---|---|---|---|"]
    for col, lab in outs:
        vals = {s: per_user(has[has["strat"] == s], lambda g: contrast(g, col), min_n=30) for s in range(3)}
        cells = []
        J["contrast"][col] = {}
        for s in range(3):
            m, h, k = tci(vals[s]); cells.append(f"{fmt(m, h, 3)} ({k})"); J["contrast"][col][STRATA[s]] = [m, h, k]
        d = (vals[2] - vals[0]).dropna(); m, h, k = tci(d); J["contrast"][col]["slow_minus_fast"] = [m, h, k]
        L.append(f"| {lab} | " + " | ".join(cells) + f" | {fmt(m, h, 3)} ({k}) |")
    L.append("")

    # ---- 3b. per user: late / on-time cost per unit of memory, the exchange rate, the cost-optimal band
    L += ["### Cost per unit of memory, late vs on time (primary of point 3)", ""]
    has["rate"] = has["cost_s"] / np.maximum(has["ed"], 1.0)        # seconds per card-day of the interval it closed
    has["mem"] = (1.0 + has["p"]) / 2.0                             # mean recall over that interval (linear proxy)

    def side_means(g, cols):
        """Cell-weighted means of cols on the late and on-time sides (cells with >= 3 rows on each side)."""
        acc = {c: [0.0, 0.0] for c in cols}; wsum = 0.0
        for _, h in g.groupby(["sq", "dq"]):
            a = h[h["late"] == 1]; o = h[h["late"] == 0]
            if len(a) >= 3 and len(o) >= 3:
                w = min(len(a), len(o)); wsum += w
                for c in cols:
                    acc[c][0] += w * float(a[c].mean()); acc[c][1] += w * float(o[c].mean())
        if wsum == 0:
            return None
        return {c: (v[0] / wsum, v[1] / wsum) for c, v in acc.items()}
    recs = []
    for u, g in has.groupby("user"):
        r = {"user": u}
        for s in (0, 2):
            m = side_means(g[g["strat"] == s], ["cost_s", "dgain", "rate", "mem", "passed", "t90a"])
            if m is None:
                continue
            (cl, co), (gl, go) = m["cost_s"], m["dgain"]
            r[f"E_late_{s}"] = cl / gl if gl > 0 else np.nan
            r[f"E_on_{s}"] = co / go if go > 0 else np.nan
            r[f"lnrho_{s}"] = math.log(r[f"E_late_{s}"] / r[f"E_on_{s}"]) if (gl > 0 and go > 0) else np.nan
            (rl, ro), (ml, mo) = m["rate"], m["mem"]
            r[f"X_{s}"] = (ro - rl) / (mo - ml) if mo - ml > 1e-3 else np.nan     # s/card-day saved per recall lost
            r[f"dpass_{s}"] = m["passed"][0] - m["passed"][1]
            (tl, to) = m["t90a"]
            r[f"lnrho2_{s}"] = math.log((cl / tl) / (co / to)) if (tl > 0 and to > 0) else np.nan
        recs.append(r)
    R3 = pd.DataFrame(recs)
    for c in ("lnrho_0", "lnrho_2", "lnrho2_0", "lnrho2_2", "X_0", "X_2", "E_late_0", "E_on_0", "E_late_2", "E_on_2"):
        if c not in R3:
            R3[c] = np.nan
    inter = (R3["lnrho_2"] - R3["lnrho_0"]).dropna()
    m, h, k = tci(inter)
    J["rho"] = {"lnrho_fast": tci(R3["lnrho_0"]), "lnrho_slow": tci(R3["lnrho_2"]), "interaction": [m, h, k],
                "interaction_median": float(inter.median()) if len(inter) else float("nan"),
                "users_gain_nonpositive": int(R3[["E_late_0", "E_on_0", "E_late_2", "E_on_2"]].isna().any(axis=1).sum())}
    i2 = (R3["lnrho2_2"] - R3["lnrho2_0"]).dropna()
    J["rho2"] = {"lnrho_fast": tci(R3["lnrho2_0"]), "lnrho_slow": tci(R3["lnrho2_2"]), "interaction": tci(i2),
                 "interaction_median": float(i2.median()) if len(i2) else float("nan")}
    ok = (R3["X_2"] > 0) & (R3["X_0"] > 0)
    lx = (np.log(R3["X_2"].where(ok)) - np.log(R3["X_0"].where(ok))).dropna()
    J["exchange"] = {"X_fast": tci(R3["X_0"]), "X_slow": tci(R3["X_2"]), "ln_ratio_slow_fast": tci(lx),
                     "X_fast_median": float(R3["X_0"].median()), "X_slow_median": float(R3["X_2"].median())}
    L += ["E = seconds per unit of ln t90 growth, on the late (P < 0.85) and on-time (P >= 0.90) sides of the same "
          "state cells; rho = E_late / E_on time (> 1: reviewing late costs more per unit of memory).", "",
          f"- ln rho, fast cards: {fmt(*J['rho']['lnrho_fast'][:2], 3)} ({J['rho']['lnrho_fast'][2]} users); slow "
          f"cards: {fmt(*J['rho']['lnrho_slow'][:2], 3)} ({J['rho']['lnrho_slow'][2]} users).",
          f"- **Interaction ln rho_slow - ln rho_fast: {fmt(m, h, 3)}** ({k} users, median "
          f"{J['rho']['interaction_median']:+.3f}; < 0 = late is relatively cheaper for slow cards). Users with a "
          f"non-positive mean gain on a side (left out): {J['rho']['users_gain_nonpositive']}.",
          f"- Positive currency (seconds per day of t90 after the review, no user left out): ln rho fast "
          f"{fmt(*J['rho2']['lnrho_fast'][:2], 3)}, slow {fmt(*J['rho2']['lnrho_slow'][:2], 3)}; interaction "
          f"{fmt(*J['rho2']['interaction'][:2], 3)} ({J['rho2']['interaction'][2]} users, median "
          f"{J['rho2']['interaction_median']:+.3f}).",
          "- Holding view: X = seconds per card-day saved by reviewing late, per unit of recall lost: "
          "X = (rate_on - rate_late) / (mem_on - mem_late), rate = review seconds incl. steps / the real interval it "
          "closed, mem = (1 + P) / 2. Fast: "
          f"{fmt(*J['exchange']['X_fast'][:2], 2)} (median {J['exchange']['X_fast_median']:.2f}), slow: "
          f"{fmt(*J['exchange']['X_slow'][:2], 2)} (median {J['exchange']['X_slow_median']:.2f}); ln X_slow - ln "
          f"X_fast {fmt(*J['exchange']['ln_ratio_slow_fast'][:2], 3)} ({J['exchange']['ln_ratio_slow_fast'][2]} "
          "users; > 0 = deferring a slow card saves more seconds per unit of recall lost).", ""]

    # cost-optimal band per stratum: per-user sums per (stratum, band), bootstrap over users
    U = sorted(has["user"].unique()); nb = len(RB_TXT)
    uix = {u: i for i, u in enumerate(U)}
    A = np.zeros((len(U), 3, nb, 5))
    for (u, s, b), g in has[has["strat"] >= 0].groupby(["user", "strat", "rband"]):
        A[uix[u], int(s), int(b)] = (float(g["cost_s"].sum()), float(g["dgain"].sum()), len(g), float(g["p"].sum()),
                                     float(g["lr"].sum()))

    def optimum(Ai):
        """Pooled E per (stratum, band) from summed users; argmin band's mean P and the stratum's mean x."""
        T = Ai.sum(0)
        E = np.where((T[..., 1] > 0) & (T[..., 2] >= 200), T[..., 0] / np.maximum(T[..., 1], 1e-9), np.nan)
        pm = T[..., 3] / np.maximum(T[..., 2], 1)
        xm = T[..., 4].sum(-1) / np.maximum(T[..., 2].sum(-1), 1)
        rstar = np.full(3, np.nan)
        for s in range(3):
            if np.isfinite(E[s]).any():
                rstar[s] = pm[s, int(np.nanargmin(E[s]))]
        return E, rstar, xm
    E0, rs0, xm0 = optimum(A)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(2000):
        _, rs, xm = optimum(A[rng.integers(0, len(U), len(U))])
        boot.append((logit(rs[2]) - logit(rs[0])) / (xm[2] - xm[0]) if np.isfinite(rs[[0, 2]]).all() else np.nan)
    boot = np.asarray(boot)
    sl0 = float((logit(rs0[2]) - logit(rs0[0])) / (xm0[2] - xm0[0]))
    J["optimum"] = {"E": E0.tolist(), "rstar": rs0.tolist(), "xmean": xm0.tolist(), "slope": sl0,
                    "slope_ci": [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))],
                    "beta_equiv": -sl0}
    L += ["Pooled seconds per unit ln t90 growth by stratum and recall band (sums over users; bands with >= 200 "
          "reviews and a positive mean gain):", "", "| stratum | mean x | " + " | ".join(RB_TXT) + " | optimal P |",
          "|---|---|" + "---|" * nb + "---|"]
    for s in range(3):
        L.append(f"| {STRATA[s]} | {xm0[s]:+.2f} | " + " | ".join("—" if not np.isfinite(v) else f"{v:.0f}"
                                                                for v in E0[s]) + f" | {rs0[s]:.3f} |")
    L += ["", f"Slope of the optimal recall's logit on x (slow vs fast): {sl0:+.2f} (bootstrap over users 95%: "
          f"{J['optimum']['slope_ci'][0]:+.2f} .. {J['optimum']['slope_ci'][1]:+.2f}); the rule's beta = minus this "
          f"slope, so this one-step metric implies beta = {-sl0:+.2f}.", ""]

    # ---- 4. the implied beta: beta_sim x gamma_real / gamma_head + delta
    L += ["## 4. Implied beta", ""]
    gr = per_user(has, lambda g: slope(g, "lnms")); gh = per_user(has, lambda g: slope(g, "ln_pred"))
    has["ln_pred_next"] = np.log(np.maximum(has["next_tE"], 100.0) / has["med_ms"])
    grn = per_user(has, lambda g: slope(g, "ln_next")); ghn = per_user(has, lambda g: slope(g, "ln_pred_next"))
    k_ = gr.dropna().index.intersection(gh.dropna().index).intersection(d_ih.dropna().index)
    gr, gh, dd = gr[k_], gh[k_], d_ih[k_]
    out4 = {}
    for bs in (2.0, 3.0):
        est = bs * gr.mean() / gh.mean() + dd.mean()
        bb = []
        for _ in range(2000):
            j = rng.integers(0, len(k_), len(k_))
            bb.append(bs * gr.values[j].mean() / gh.values[j].mean() + dd.values[j].mean())
        out4[bs] = [float(est), float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))]
    mgr, hgr, _ = tci(gr); mgh, hgh, _ = tci(gh); mdif, hdif, kd = tci(gh - gr)
    J["implied_beta"] = {"gamma_real": [mgr, hgr], "gamma_head": [mgh, hgh], "head_minus_real": [mdif, hdif, kd],
                         "gamma_real_next": tci(grn), "gamma_head_next": tci(ghn),
                         "delta": tci(dd), "beta_real": {str(k): v for k, v in out4.items()}, "users": len(k_)}
    L += [f"{len(k_)} users. Elasticity of this review's answer time on x: real gamma {fmt(mgr, hgr, 3)}, the "
          f"answer-time head {fmt(mgh, hgh, 3)} (head - real {fmt(mdif, hdif, 3)}). Next later-day review: real "
          f"{fmt(*J['implied_beta']['gamma_real_next'][:2], 3)}, head "
          f"{fmt(*J['implied_beta']['gamma_head_next'][:2], 3)}. Calibration delta "
          f"{fmt(*J['implied_beta']['delta'][:2], 3)}.", "",
          "beta_real = beta_sim x gamma_real / gamma_head + delta (bootstrap over users, 95%):", ""]
    for bs, v in out4.items():
        L.append(f"- beta_sim = {bs:.0f}: beta_real = {v[0]:.2f} ({v[1]:.2f} .. {v[2]:.2f})")
    L.append("")
    R3.to_csv(od / "per-user-point3.csv", index=False)
    pu = pd.DataFrame({"gamma_real": gr, "gamma_head": gh, "delta_ih": dd})
    pu.index.name = "user"; pu.to_csv(od / "per-user-point4.csv")

    Path(od / "realcheck-tables.md").write_text("\n".join(L) + "\n")
    Path(od / "realcheck.json").write_text(json.dumps(J, indent=1, default=float))
    meta.to_csv(od / "splits.csv", index=False)
    wall, cpu, nrev = meta["wall"].sum(), meta["cpu"].sum(), meta["window_reviews"].sum()
    eng_s = wall - meta["warm"].sum()
    print("\n".join(L))
    print(f"\nrun cost: {nrev:,} window reviews; {wall / 3600:.2f} process-hours wall incl. warm-ups "
          f"({meta['warm'].sum() / 3600:.2f} h), {cpu / 3600:.2f} CPU hours; engine {nrev / max(eng_s, 1):.0f} "
          f"reviews/s per process, {nrev / max(wall, 1):.0f} reviews/s incl. warm-up")


def main():
    p = ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--user", type=int, required=True)
    r.add_argument("--splits", required=True); r.add_argument("--days", type=int, default=60)
    r.add_argument("--fits", default=None); r.add_argument("--out-dir", default=str(OUT / "logs"))
    a = sub.add_parser("analyze"); a.add_argument("--logs", default=str(OUT / "logs"))
    a.add_argument("--out-dir", default=str(OUT))
    x = p.parse_args()
    {"run": run, "analyze": analyze}[x.cmd](x)


if __name__ == "__main__":
    main()
