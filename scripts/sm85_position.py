"""Session position effect on answer time (SM85): warm-up or selection? CPU analysis on the 150 users of the response-time
prediction (SM82) stage 2 sample (their test reviews: seen cards, later day than the card's previous review).

Per user: build the review table, absorb fixed effects (user / card / day / card+day) by alternating demeaning, and keep the
Gram matrix of [regressors, outcomes]; the pooled regression is the sum of the per-user Grams, and the user bootstrap
(2,000 resamples, seed 0) re-sums resampled Grams. Outcomes: ln(time) (time = recorded duration capped at the user's cap,
floored at 0.5 s) and fast Good (Good and under 3,891 ms), both by least squares.

  python scripts/sm85_position.py prepare --procs 5     per-user Grams + descriptives -> work dir
  python scripts/sm85_position.py report                regressions, bootstrap, ordering bound -> artifacts/small-model/sm85/session-position.{md,json}
"""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NPZ = Path("C:/Users/jonat/workspace/wt/sm82s2/artifacts/small-model/sm82/stage2/work/curves/users")   # read only
OUT = ROOT / "artifacts" / "small-model" / "sm85"
WORK = Path("C:/Users/jonat/workspace/wt/sm85/artifacts/small-model/sm85/work")
FAST_MS = 500.0 * 120.0 ** (6 / 14)
POS_EDGES = [0, 10, 25, 50, 100, 200, 400, 10 ** 9]
POS_NAMES = ["1-10", "11-25", "26-50", "51-100", "101-200", "201-400", "401+"]
T_EDGES = [0, 2, 5, 10, 20, 40, 10 ** 9]          # active minutes before the review
T_NAMES = ["0-2 min", "2-5 min", "5-10 min", "10-20 min", "20-40 min", "40+ min"]
DL_EDGES = [0, 30, 100, 200, 400, 10 ** 9]        # reviews in the whole day
DL_NAMES = ["under 30", "30-99", "100-199", "200-399", "400+"]
COVS = ["ln_e", "ln_reps", "lapses", "prev_ln_t", "prev_again", "prev_hard", "prev_easy", "pred_recall"]


def low():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001
        pass


def codes(a):
    return np.unique(a, return_inverse=True)[1].astype(np.int64)


def demean(Z, g, iters=1):
    cnt = np.bincount(g).astype(np.float64)
    for j in range(Z.shape[1]):
        Z[:, j] -= (np.bincount(g, Z[:, j]) / cnt)[g]


def absorb(Z, groups):
    """Alternating projections on one or two groupings (in place)."""
    if len(groups) == 1:
        demean(Z, groups[0]); return
    prev = np.inf
    for _ in range(60):
        for g in groups:
            demean(Z, g)
        s = float((Z ** 2).sum())
        if abs(prev - s) <= 1e-9 * max(s, 1e-12):
            break
        prev = s


def build_user(path):
    import pandas as pd
    from scripts import compress_common as cc
    z = dict(np.load(path))
    user = int(Path(path).stem)
    rev = pd.read_parquet(Path(cc.data_path()) / "revlogs" / f"user_id={user}", columns=["card_id", "day_offset", "rating", "duration", "elapsed_days"]).reset_index(drop=True)
    cap = int(z["cap_ms"])
    card, el, rating, day = rev["card_id"], rev["elapsed_days"], rev["rating"], rev["day_offset"]
    dur = np.minimum(rev["duration"].to_numpy(np.float64), cap)
    rec = dur / 1000.0
    pos = day.groupby(day).cumcount().to_numpy()
    day_n = day.groupby(day).transform("size").to_numpy()
    cum = pd.Series(dur).groupby(day).cumsum().to_numpy() - dur
    day_rev = (el != 0).astype(int)
    reps = (day_rev.groupby(card).cumsum() - day_rev).to_numpy()
    lapse = ((rating == 1) & (el >= 1)).astype(int)
    lapses = (lapse.groupby(card).cumsum() - lapse).to_numpy()
    prev_t = pd.Series(rec).groupby(card).shift(1).to_numpy()
    prev_r = rating.groupby(card).shift(1).to_numpy()
    i = z["th"].astype(np.int64) - 1
    ok = ~np.isnan(prev_t[i])
    i, pg = i[ok], z["full_pg"][ok].astype(np.float64)
    t = rec[i]
    lt = np.log(np.maximum(t, 0.5))
    fg = ((rating.to_numpy()[i] == 3) & (z["dur"][ok] < FAST_MS)).astype(np.float64)
    pr = prev_r[i]
    f = {"ln_e": np.log1p(el.to_numpy()[i].astype(np.float64)), "ln_reps": np.log1p(reps[i]), "lapses": np.minimum(lapses[i], 6).astype(np.float64),
         "prev_ln_t": np.log(np.maximum(prev_t[i], 0.5)), "prev_again": (pr == 1) * 1.0, "prev_hard": (pr == 2) * 1.0, "prev_easy": (pr == 4) * 1.0,
         "pred_recall": 1 - pg[:, 0]}
    P = np.searchsorted(POS_EDGES, pos[i], side="right") - 1
    T = np.searchsorted(T_EDGES, cum[i] / 60000.0, side="right") - 1
    D = np.searchsorted(DL_EDGES, day_n[i], side="right") - 1
    cols = {f"pos_{n}": (P == k) * 1.0 for k, n in enumerate(POS_NAMES) if k > 0}
    cols.update({f"t_{n}": (T == k) * 1.0 for k, n in enumerate(T_NAMES) if k > 0})
    cols.update({f"dl_{n}": (D == k) * 1.0 for k, n in enumerate(DL_NAMES) if k > 0})
    cols.update(f)
    cols["ln_time"], cols["fast_good"] = lt, fg
    names = list(cols)
    Z = np.column_stack([cols[n] for n in names])
    cg, dg, ug = codes(card.to_numpy()[i]), codes(day.to_numpy()[i]), np.zeros(len(i), np.int64)
    pos_i, daycode = pos[i], dg
    # descriptives per position bucket
    desc = []
    for k in range(len(POS_NAMES)):
        m = P == k
        desc.append([m.sum(), fg[m].sum(), t[m].sum(), f["pred_recall"][m].sum(), el.to_numpy()[i][m].sum(), reps[i][m].sum(), lapses[i][m].sum(),
                     prev_t[i][m].sum(), f["prev_again"][m].sum(), day_n[i][m].sum(), lt[m].sum(), (rating.to_numpy()[i][m] == 1).sum()])
    blocks = {}

    def gram(name, mask, fe, use):
        Zb = Z[mask].copy()
        gs = {"user": [ug[mask]], "card": [codes(cg[mask])], "day": [codes(dg[mask])], "cardday": [codes(cg[mask]), codes(dg[mask])]}[fe]
        absorb(Zb, gs)
        idx = [names.index(n) for n in use]
        Zs = Zb[:, idx]
        blocks[name] = {"cols": use, "G": Zs.T @ Zs, "n": int(mask.sum())}

    allm = np.ones(len(i), bool)
    pos_c = [c for c in names if c.startswith("pos_")]
    t_c = [c for c in names if c.startswith("t_")]
    ys = ["ln_time", "fast_good"]
    for fe in ("user", "card", "day", "cardday"):
        gram(f"full-{fe}", allm, fe, pos_c + t_c + COVS + ys)
    first20 = pos_i < 20
    dl_c = [c for c in names if c.startswith("dl_")]
    for fe in ("user", "card"):
        gram(f"first20-{fe}", first20, fe, dl_c + COVS + ys)
    ind = (pos_i >= 200).astype(np.float64)
    pm = (pos_i < 10) | (pos_i >= 200)
    names.append("ge200"); Z = np.column_stack([Z, ind])
    gram("pair-card", pm, "card", ["ge200"] + COVS + ys)
    gram("pair-cardday", pm, "cardday", ["ge200"] + COVS + ys)
    # small arrays for the ordering bound
    small = {"day": daycode.astype(np.int32), "pos": pos_i.astype(np.int32), "t": t.astype(np.float32), "card": cg.astype(np.int32), "fg": fg.astype(np.int8)}
    return user, {"blocks": blocks, "desc": np.array(desc, float), "n": len(i), "small": small}


def work(path):
    low()
    t0 = time.time()
    u, r = build_user(path)
    r["secs"] = time.time() - t0
    return u, r


def prepare(args):
    WORK.mkdir(parents=True, exist_ok=True)
    files = sorted(NPZ.glob("*.npz"))
    done = {int(p.stem) for p in WORK.glob("*.pkl")}
    todo = [str(f) for f in files if int(f.stem) not in done]
    print(f"{len(files)} users, {len(todo)} to do", flush=True)
    with Pool(args.procs, initializer=low) as pool:
        for u, r in pool.imap_unordered(work, todo):
            with open(WORK / f"{u}.pkl", "wb") as fh:
                pickle.dump(r, fh)
            print(u, r["n"], f"{r['secs']:.0f}s", flush=True)


# ------------------------------------------------------------------ report
SPECS = [  # (id, block, regressors, label)
    ("S1", "full-user", "pos", "user effects, position buckets only"),
    ("S2", "full-user", "pos+cov", "user effects + card-state controls"),
    ("S3", "full-card", "pos", "card effects, position buckets only"),
    ("S4", "full-card", "pos+cov", "card effects + controls"),
    ("S5", "full-day", "pos+cov", "day effects + controls (within one day)"),
    ("S6", "full-cardday", "pos+cov", "card and day effects + controls"),
    ("S7", "full-cardday", "t+cov", "card and day effects + controls, minutes of active time instead of position"),
    ("S8", "full-cardday", "pos+t+cov", "card and day effects + controls, position and active time together"),
    ("S9", "full-card", "pos+t+cov", "card effects + controls, position and active time together"),
]


def solve(G, cols, regs):
    idx = [cols.index(r) for r in regs]
    out = {}
    for y in ("ln_time", "fast_good"):
        j = cols.index(y)
        A = G[np.ix_(idx, idx)]; b = G[idx, j]
        out[y] = np.linalg.lstsq(A, b, rcond=None)[0]
    return out


def regs_for(kind, cols):
    r = []
    if "pos" in kind.split("+"): r += [c for c in cols if c.startswith("pos_")]
    if "t" in kind.split("+"): r += [c for c in cols if c.startswith("t_")]
    if "cov" in kind.split("+"): r += COVS
    return r


def boot(Gs, ns, cols, regs, weights, B=2000, seed=0):
    """Gs [U, K, K]; weights [U] (1 for pooled, 1/n_u for equal user weight). Returns point [2, r], boot [B, 2, r]."""
    U = len(Gs)
    Gw = Gs * weights[:, None, None]
    pt = solve(Gw.sum(0), cols, regs)
    point = np.stack([pt["ln_time"], pt["fast_good"]])
    rng = np.random.default_rng(seed)
    M = rng.multinomial(U, np.ones(U) / U, size=B)   # counts per user
    flat = Gw.reshape(U, -1)
    S = (M @ flat).reshape(B, *Gs.shape[1:])
    bs = np.empty((B,) + point.shape)
    for b in range(B):
        s = solve(S[b], cols, regs)
        bs[b] = np.stack([s["ln_time"], s["fast_good"]])
    return point, bs


def ci(bs):
    return np.percentile(bs, [2.5, 97.5], axis=0)


def pct(x):
    return np.expm1(x) * 100


def report(args):
    files = sorted(WORK.glob("*.pkl"))
    users = [pickle.load(open(f, "rb")) for f in files]
    uid = [int(f.stem) for f in files]
    U = len(users)
    res = {"users": U, "reviews": int(sum(u["n"] for u in users)), "specs": {}, "descriptive": [], "first20": {}, "pair": {}}
    # descriptives
    D = sum(u["desc"] for u in users)
    for k, n in enumerate(POS_NAMES):
        a = D[k]
        res["descriptive"].append({"bucket": n, "reviews": int(a[0]), "fast_good": a[1] / a[0], "mean_time_s": a[2] / a[0], "pred_recall": a[3] / a[0],
                                   "mean_interval_days": a[4] / a[0], "card_reviews_before": a[5] / a[0], "card_lapses_before": a[6] / a[0],
                                   "prev_time_s": a[7] / a[0], "prev_again_share": a[8] / a[0], "mean_day_reviews": a[9] / a[0], "mean_ln_time": a[10] / a[0],
                                   "again_share": a[11] / a[0]})
    for w_name in ("equal", "pooled"):
        res["specs"][w_name] = {}
        for sid, blk, kind, label in SPECS:
            cols = users[0]["blocks"][blk]["cols"]
            regs = regs_for(kind, cols)
            Gs = np.stack([u["blocks"][blk]["G"] for u in users]); ns = np.array([u["blocks"][blk]["n"] for u in users], float)
            w = 1 / ns if w_name == "equal" else np.ones(U)
            if w_name == "equal":
                w = w * ns.mean()
            pt, bs = boot(Gs, ns, cols, regs, w)
            lo, hi = ci(bs)
            res["specs"][w_name][sid] = {"label": label, "block": blk, "regs": regs, "ln_time": pt[0].tolist(), "ln_time_ci": [lo[0].tolist(), hi[0].tolist()],
                                         "fast_good": pt[1].tolist(), "fast_good_ci": [lo[1].tolist(), hi[1].tolist()]}
        # day length (first 20 reviews) and pair contrast
        for blk, key in (("first20-user", "first20_user"), ("first20-card", "first20_card"), ("pair-card", "pair_card"), ("pair-cardday", "pair_cardday")):
            cols = users[0]["blocks"][blk]["cols"]
            for cov in (False, True):
                regs = [c for c in cols if c.startswith("dl_") or c == "ge200"] + (COVS if cov else [])
                Gs = np.stack([u["blocks"][blk]["G"] for u in users]); ns = np.array([u["blocks"][blk]["n"] for u in users], float)
                w = (ns.mean() / ns) if w_name == "equal" else np.ones(U)
                pt, bs = boot(Gs, ns, cols, regs, w)
                lo, hi = ci(bs)
                res.setdefault(key, {}).setdefault(w_name, {})["cov" if cov else "nocov"] = {
                    "regs": regs, "ln_time": pt[0].tolist(), "ln_time_ci": [lo[0].tolist(), hi[0].tolist()],
                    "fast_good": pt[1].tolist(), "fast_good_ci": [lo[1].tolist(), hi[1].tolist()]}
    # ordering bound from S6 (pooled) position factors
    sp = res["specs"]["pooled"]["S6"]
    pos_names = [r for r in sp["regs"] if r.startswith("pos_")]
    wf = np.exp(np.r_[0.0, [sp["ln_time"][sp["regs"].index(r)] for r in pos_names]])   # multiplicative factor by bucket
    wsm = wf if True else None
    rng = np.random.default_rng(0)
    sav_s, tot_s, cnt = [], [], []
    for u in users:
        s = u["small"]
        b = np.searchsorted(POS_EDGES, s["pos"], side="right") - 1
        w = wf[b]
        order = np.lexsort((s["pos"], s["day"]))
        d, w, t = s["day"][order], w[order], s["t"][order].astype(np.float64)
        starts = np.r_[0, np.nonzero(np.diff(d))[0] + 1, len(d)]
        sav = 0.0; tot = 0.0
        for a, e in zip(starts[:-1], starts[1:]):
            if e - a < 2:
                tot += t[a:e].sum(); continue
            base = t[a:e] / w[a:e]
            new = (np.sort(base)[::-1] * np.sort(w[a:e])).sum()   # largest base times at the fastest-factor positions
            sav += t[a:e].sum() - new; tot += t[a:e].sum()
        nd = len(starts) - 1
        sav_s.append(sav / nd); tot_s.append(tot / nd); cnt.append(nd)
    sav_s, tot_s, cnt = map(np.array, (sav_s, tot_s, cnt))
    M = rng.multinomial(U, np.ones(U) / U, size=2000)
    bs_eq = (M @ sav_s) / U; bs_pct = (M @ sav_s) / (M @ tot_s)
    res["ordering"] = {"saving_s_per_day_user_avg": float(sav_s.mean()), "ci": np.percentile(bs_eq, [2.5, 97.5]).tolist(),
                       "tested_time_s_per_day_user_avg": float(tot_s.mean()), "saving_share": float(sav_s.sum() / tot_s.sum()),
                       "saving_share_user_avg": float((sav_s / tot_s).mean()), "saving_share_ci": np.percentile(bs_pct, [2.5, 97.5]).tolist(),
                       "days": int(cnt.sum()), "factors": dict(zip(POS_NAMES, wf.tolist())),
                       "saving_s_per_day_user_quartiles": np.percentile(sav_s, [25, 50, 75]).tolist()}
    cols = users[0]["blocks"]["full-cardday"]["cols"]; regs = regs_for("pos+cov", cols); j = regs.index("pos_101-200")
    co = []
    for u in users:
        if u["n"] >= 5000:
            try:
                co.append(solve(u["blocks"]["full-cardday"]["G"], cols, regs)["ln_time"][j])
            except Exception:  # noqa: BLE001
                pass
    co = np.array(co)
    res["het"] = {"users": len(co), "neg_share": float((co < 0).mean()), "q": np.percentile(co, [25, 50, 75]).tolist()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "session-position.json").write_text(json.dumps(res, indent=1, default=float))
    render(res)


TAKEAWAYS = [
    "## Takeaways", "",
    "- Most of the position effect survives the controls, so it is mostly within-card, not selection: the same card is answered about 26-28% faster at positions 201-400 than in the first 10 reviews of the day (card and day effects + controls, S6: -27.6% [-31.6, -23.7] pooled, -26.6% [-30.1, -22.8] users weighted equally), and the chance of a fast Good (Good under 3.9 s) is +6.2 pp higher (pooled [+4.6, +7.7]; equal-user +5.1 pp [+3.7, +6.5]). The unadjusted gap is bigger (-43.7% time, +11.5 pp), so selection (which users, which cards) explains roughly 40-45% of the raw gradient and a within-card speed-up the rest. It is monotone (positions 11-25: -6%, 51-100: -15%, 101-200: -21%) and present for 95% of users with 5,000+ reviews.",
    "- Selection is real but secondary. User mix (heavy users answer faster) is the biggest part (user effects alone: -31%); the card mix adds little (card effects: -26%); late positions hold slightly older, less often lapsed, previously faster cards (descriptive table). Day-level effects (a good or bad day, long or short days) change almost nothing between card effects (S4) and card and day effects (S6), so users who stop early on bad days are not what drives it.",
    "- Short days do not start clearly slower: for the first 20 reviews of a day, days of 100-399 reviews vs under 30 reviews are +3% to +6% slower with 95% intervals including 0 (card effects + controls), and fast Good is 1-2 pp lower; with user effects only the difference is about 0. At most a small part of the gradient is 'users who stop early answer slower from the start'.",
    "- It tracks the number of reviews already done, not elapsed active minutes: once position and active minutes are both in with card and day effects (S8), the position effects stay (-23.7% at 201-400) and the minutes drop to about 0 (-4.7% at 40+ min); without day effects minutes proxy the day's slowness and the pair is not interpretable (S9). So it behaves like getting into the groove (practice, rhythm), not like a clock effect or fatigue.",
    "- Still confounded: (1) 'warm-up' here only means 'faster later in the day for the same card': the data cannot separate genuine warm-up from rushing or a lower reading depth late in the day (the Again share rises only slightly, 0.123 at 11-25 to 0.139 at 401+, so recall does not obviously suffer, but grade and speed choices are the user's); (2) queue order within a day (learning cards and relearn steps vs reviews, deck order) is not observed beyond the position; (3) time of day and breaks are invisible (active minutes ignore breaks); (4) the card effect only removes what is constant for a card, not its state on that day (the previous grade, predicted recall, interval and review count are controlled linearly); (5) 150 users of the 732 set (the stage 2 sample), their test reviews only.",
    "- Practical bound: if the effect is a pure multiplicative position factor and a scheduler could freely re-order a day's reviews, putting the slowest (hardest) cards late and the quickest ones early would save about 55 s per day on average per user (95% [48, 63]; user quartiles 20 / 42 / 86 s), 4.2% of the 1,222 s of review time in the tested reviews. This is an upper-bound-style figure: it assumes full freedom to re-order, ignores that the learning order carries its own effects (interleaving, learning steps) and that the card difficulty cannot be known perfectly in advance. Less than 5% of daily time, so a modest gain.",
    "",
]


def fmt_t(b, lo, hi):
    return f"{pct(b):+.1f}% [{pct(lo):+.1f}, {pct(hi):+.1f}]"


def fmt_f(b, lo, hi):
    return f"{b * 100:+.1f} pp [{lo * 100:+.1f}, {hi * 100:+.1f}]"


def render(r):
    res_user_het = r["het"]
    eq, po = r["specs"]["equal"], r["specs"]["pooled"]
    L = ["# Session position effect on answer time (SM85): warm-up or selection?", ""]
    L += [f"Users: {r['users']} (the response-time prediction (SM82) stage 2 sample of the 732 set); {r['reviews']:,} test reviews of seen cards on a later day than the card's previous review. Times capped at the user's cap (60 s) and floored at 0.5 s for ln(time); fast Good = Good under 3.9 s. Position = number of earlier reviews that day in the revlog (new cards and same-day steps count), active minutes = recorded time of the earlier reviews that day (breaks are invisible). Effects are relative to the first 10 reviews of the day; intervals are user-bootstrap 95% (2,000 resamples, seed 0).", ""]
    L += ["TAKEAWAYS_PLACEHOLDER", ""]
    L += ["## 1. Who is answered where (selection, descriptive)", "",
          "| position | reviews | fast Good | mean time (s) | predicted recall | mean interval (d) | card reviews before | card lapses before | previous time of the card (s) | day length (reviews) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for d in r["descriptive"]:
        L.append(f"| {d['bucket']} | {d['reviews']:,} | {d['fast_good']:.3f} | {d['mean_time_s']:.1f} | {d['pred_recall']:.3f} | {d['mean_interval_days']:.1f} | {d['card_reviews_before']:.1f} | {d['card_lapses_before']:.2f} | {d['prev_time_s']:.1f} | {d['mean_day_reviews']:.0f} |")
    L += ["", "Pooled over reviews (heavy users dominate); the later positions only exist on long days, which is itself selection."]

    def spec_table(w):
        S = r["specs"][w]
        names = [x for x in S["S1"]["regs"] if x.startswith("pos_")]
        L2 = ["| position (vs 1-10) | " + " | ".join(f"{sid}" for sid in S) + " |", "|---|" + "---|" * len(S)]
        return S, names, L2

    for outcome, f, title, unit in (("ln_time", fmt_t, "answer time (percent change)", ""), ("fast_good", fmt_f, "share of fast Goods (percentage points)", "")):
        L += ["", f"## 2. Position effect on {title}: controlled estimates (users weighted equally)", ""]
        L += ["Specs: " + "; ".join(f"{sid} = {s['label']}" for sid, s in eq.items()) + ".", ""]
        S = eq
        names = [x for x in S["S1"]["regs"] if x.startswith("pos_")]
        L += ["| position | " + " | ".join(S) + " |", "|---|" + "---|" * len(S)]
        for n in names:
            row = []
            for sid, s in S.items():
                if n in s["regs"]:
                    j = s["regs"].index(n)
                    row.append(f(s[outcome][j], s[outcome + "_ci"][0][j], s[outcome + "_ci"][1][j]))
                else:
                    row.append("-")
            L.append(f"| {n[4:]} | " + " | ".join(row) + " |")
    L += ["", "Active-minutes coefficients (vs 0-2 min) in the specs that include them:", "", "| active time | " + " | ".join(f"{s} ln time" for s in ("S7", "S8", "S9")) + " | " + " | ".join(f"{s} fast Good" for s in ("S7", "S8", "S9")) + " |", "|---|---|---|---|---|---|---|"]
    S = eq
    for n in [x for x in S["S7"]["regs"] if x.startswith("t_")]:
        row = []
        for outcome, f in (("ln_time", fmt_t), ("fast_good", fmt_f)):
            for sid in ("S7", "S8", "S9"):
                s = S[sid]; j = s["regs"].index(n)
                row.append(f(s[outcome][j], s[outcome + "_ci"][0][j], s[outcome + "_ci"][1][j]))
        L.append(f"| {n[2:]} | " + " | ".join(row) + " |")
    # pooled headline
    d0 = r["descriptive"][0]
    L += ["", "Pooled over reviews (every review equal), unadjusted vs adjusted. Unadjusted = bucket mean vs the 1-10 mean (ln time for the percent change):", "",
          "| position | unadjusted time | S1 time | S4 time | S6 time | unadjusted fast Good | S1 fast Good | S4 fast Good | S6 fast Good | Again share (unadjusted) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, n in enumerate(names):
        d = r["descriptive"][k + 1]
        row = [f"{pct(d['mean_ln_time'] - d0['mean_ln_time']):+.1f}%"]
        for sid in ("S1", "S4", "S6"):
            s_ = po[sid]; j = s_["regs"].index(n); row.append(fmt_t(s_["ln_time"][j], s_["ln_time_ci"][0][j], s_["ln_time_ci"][1][j]))
        row.append(f"{(d['fast_good'] - d0['fast_good']) * 100:+.1f} pp")
        for sid in ("S1", "S4", "S6"):
            s_ = po[sid]; j = s_["regs"].index(n); row.append(fmt_f(s_["fast_good"][j], s_["fast_good_ci"][0][j], s_["fast_good_ci"][1][j]))
        row.append(f"{d['again_share']:.3f} (first 10: {d0['again_share']:.3f})" if k == 0 else f"{d['again_share']:.3f}")
        L.append(f"| {n[4:]} | " + " | ".join(row) + " |")
    h = res_user_het
    L += ["", f"Heterogeneity (S6 per user, users with at least 5,000 reviews, {h['users']} users): the 101-200 coefficient on ln time is negative for {h['neg_share'] * 100:.0f}% of users; quartiles {pct(h['q'][0]):+.1f}% / {pct(h['q'][1]):+.1f}% / {pct(h['q'][2]):+.1f}%."]
    L += ["", "## 3. Does a short day start slower? (first 20 reviews of the day, effect of the day's length vs under 30 reviews)", "",
          "| day length | user effects: time | user effects: fast Good | card effects + controls: time | card effects + controls: fast Good |", "|---|---|---|---|---|"]
    a, b = r["first20_user"]["equal"]["nocov"], r["first20_card"]["equal"]["cov"]
    for n in [x for x in a["regs"] if x.startswith("dl_")]:
        row = []
        for blk, outcome, f in ((a, "ln_time", fmt_t), (a, "fast_good", fmt_f), (b, "ln_time", fmt_t), (b, "fast_good", fmt_f)):
            j = blk["regs"].index(n)
            row.append(f(blk[outcome][j], blk[outcome + "_ci"][0][j], blk[outcome + "_ci"][1][j]))
        L.append(f"| {n[3:]} | " + " | ".join(row) + " |")
    L += ["", "## 4. First 10 reviews vs the 201st on (reviews at positions 1-10 or 201+ only)", "",
          "| model | time | fast Good |", "|---|---|---|"]
    for key, label in (("pair_card", "same card (card effects)"), ("pair_card+", "same card, + controls"), ("pair_cardday", "same card and day (card and day effects)"), ("pair_cardday+", "same card and day, + controls")):
        k, cov = (key.rstrip("+"), "cov" if key.endswith("+") else "nocov")
        s = r[k]["equal"][cov]; j = s["regs"].index("ge200")
        L.append(f"| {label} | {fmt_t(s['ln_time'][j], s['ln_time_ci'][0][j], s['ln_time_ci'][1][j])} | {fmt_f(s['fast_good'][j], s['fast_good_ci'][0][j], s['fast_good_ci'][1][j])} |")
    o = r["ordering"]
    L += ["", "## 5. Rough bound: what a warm-up-friendly order could save", "",
          f"Take the multiplicative position factors of S6 (pooled): " + ", ".join(f"{k} x{v:.3f}" for k, v in o["factors"].items()) + f". For every day with at least 2 tested reviews, keep the same set of reviews and the same set of positions, divide each time by the factor of its position (the card's own speed) and re-assign: the slowest cards to the positions with the lowest factor, the quickest cards to the highest. Result over {o['days']:,} user-days: **{o['saving_s_per_day_user_avg']:.0f} s per day** (user average, 95% [{o['ci'][0]:.0f}, {o['ci'][1]:.0f}]; user quartiles {o['saving_s_per_day_user_quartiles'][0]:.0f} / {o['saving_s_per_day_user_quartiles'][1]:.0f} / {o['saving_s_per_day_user_quartiles'][2]:.0f} s) out of {o['tested_time_s_per_day_user_avg']:.0f} s of tested review time per day, i.e. {o['saving_share_user_avg'] * 100:.1f}% (pooled {o['saving_share'] * 100:.1f}%, 95% [{o['saving_share_ci'][0] * 100:.1f}, {o['saving_share_ci'][1] * 100:.1f}]). Ignores that the factors are estimates (the interval is over users only).", ""]
    i = L.index("TAKEAWAYS_PLACEHOLDER"); L[i:i + 1] = TAKEAWAYS
    (OUT / "session-position.md").write_text(chr(10).join(L))


def main():
    p = ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prepare"); q.add_argument("--procs", type=int, default=5)
    sub.add_parser("report")
    a = p.parse_args()
    {"prepare": prepare, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
