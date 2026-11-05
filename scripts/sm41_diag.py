#!/usr/bin/env python
"""The small model in the review simulator (SM41): diagnostic of the full-run result (why the instant-head policy costs
more minutes per memorised card than FSRS-7 at matched realised retention). CPU only, numpy only (no torch).

Brief: artifacts/small-model/sm41/diag-brief.md. Result: artifacts/small-model/sm41/diag-result.md.

  check     the sample's lane totals (diag/rows) against the full-run rows for the same users / splits / lanes
  analyse   the per-review logs (diag/logs/*.npz, scripts/sm41_engine.py run --log-dir) + the full-run rows (H2, H4)
            -> diag/diag-tables.md + diag/diag-result.json

Hypotheses: H1 per-card spread; H2 selection (memorised by the curve head vs the instant head); H3 counterfactual
miscalibration of the instant head's candidate rows; H4 horizon; H5 (the user's): FSRS-7's errors put more reviews at
a low true recall, where a review buys more memory per minute.
"""
from argparse import ArgumentParser
import glob
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import sm41_fullrun as FR   # noqa: E402

HERE = ROOT / "artifacts" / "small-model" / "sm41"
DIAG = HERE / "diag"
FULL = HERE / "fullrun"
BURN = 30                       # policy metrics on days 30-59 (DESIGN.md 9.3)
T90_STRATA = [(0.0, 3.0), (3.0, 20.0), (20.0, 1e9)]      # days, curve-head t90 before the review
R_BINS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0001]
IVL_BANDS = [1, 2, 4, 8, 16, 32, 64, 10 ** 6]          # [lo, next lo) days
FAMS = ("ihr", "fsrs7")
FAM_TXT = {"ihr": "instant head", "fsrs7": "FSRS-7"}


def ci(v):
    return FR.ci(v)


def fmt_ci(v, scale=1.0, d=1, sign=True):
    m, h, n = ci(v)
    if not np.isfinite(m):
        return "—"
    s = f"{scale * m:+.{d}f}" if sign else f"{scale * m:.{d}f}"
    return s + (f" ± {scale * h:.{d}f}" if np.isfinite(h) else "")


def band_label(lo, hi):
    return f"{lo}" if hi == lo + 1 else (f"{lo}+" if hi > 10 ** 5 else f"{lo}-{hi - 1}")


# ------------------------------------------------------------------ check
def check(args):
    V = ("wall_seconds", "lane_reviews_per_second", "finished", "warmup_seconds", "lanes", "acc", "memorised_instant",
         "memorised_curve", "backlog", "lane_reviews", "candidate_rows", "ticks")
    sample = json.loads((DIAG / "sample.json").read_text())["users"]
    full = {(r["user"], r["split_day"]): r for r in FR.load_rows(str(FULL / "rows" / "*.jsonl"))}
    diag = {(r["user"], r["split_day"]): r for r in FR.load_rows(str(DIAG / "rows" / "*.jsonl"))}
    worst, n_ok, lines = 0.0, 0, []
    for u in sample:
        key = (u["user"], u["split_day"])
        if key not in diag:
            lines.append(f"{key}: missing in diag rows"); continue
        a, b = diag[key], full[key]
        gap, exact = 0.0, True
        for k, name in enumerate(a["lanes"]):
            j = b["lanes"].index(name)
            for n in a["acc"]:
                x, y = np.asarray(a["acc"][n])[k], np.asarray(b["acc"][n])[j]
                gap = max(gap, float((np.abs(x - y) / np.maximum(1.0, np.abs(y))).max()))
            exact &= a["backlog"][k] == b["backlog"][j]
            for m in ("memorised_instant", "memorised_curve"):
                gap = max(gap, abs(a[m][k] - b[m][j]) / max(1.0, abs(b[m][j])))
        same_meta = all(a[k] == b[k] for k in a if k not in V)
        ok = gap <= 1e-4 and exact and same_meta
        n_ok += ok
        worst = max(worst, gap)
        lines.append(f"user {key[0]} split {key[1]}: max relative gap {gap:.2e}, backlog identical {exact}, "
                     f"metadata identical {same_meta} -> {'OK' if ok else 'FAIL'}")
    lines.append(f"{n_ok} of {len(sample)} splits match the full-run rows (relative tolerance 1e-4, proof P1's); "
                 f"worst gap {worst:.2e}")
    txt = "\n".join(lines)
    (DIAG / "check.out").write_text(txt + "\n")
    print(txt)


# ------------------------------------------------------------------ helpers on one split's log
def load_log(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    lanes = [str(x) for x in d["lanes"]]
    d["fam"] = np.array([n.split("-")[0] if n != "tf" else "tf" for n in lanes])
    return d, lanes


def episodes(d, sel_lane):
    """Later-day review episodes of one lane: the later-day review + the same-day follow-ups of that card on that day.
    Returns arrays per episode: day, card, p_true, aim_prev, r_fsrs7, ed, pass, minutes, ln t90 gain (the curve
    head's t90 after the last review of the day / t90 before the later-day review), the next interval chosen."""
    m = d["lane"] == sel_lane
    idx = np.nonzero(m)[0]
    day, card, kind = d["day"][idx], d["card"][idx], d["kind"][idx]
    out = {k: [] for k in ("day", "card", "p", "aim", "r7", "ed", "ok", "min", "gain", "ivl", "rc", "t90b")}
    open_ = {}
    for j, i in enumerate(idx):
        c = int(card[j])
        if kind[j] == 2:
            open_[c] = len(out["day"])
            out["day"].append(int(day[j])); out["card"].append(c); out["p"].append(d["p_true"][i])
            out["aim"].append(d["aim_prev"][i]); out["r7"].append(d["r_fsrs7"][i]); out["ed"].append(d["ed"][i])
            out["ok"].append(d["rating"][i] > 1); out["min"].append(d["ms"][i] / 60000.0)
            out["t90b"].append(d["t90_before"][i])
            out["gain"].append(math.log(max(d["t90_after"][i], 1e-6)) - math.log(max(d["t90_before"][i], 1e-6)))
            out["ivl"].append(d["ivl"][i]); out["rc"].append(d["r_curve"][i])
        elif kind[j] == 1 and c in open_ and out["day"][open_[c]] == int(day[j]):
            k = open_[c]
            out["min"][k] += d["ms"][i] / 60000.0
            out["gain"][k] = math.log(max(d["t90_after"][i], 1e-6)) - math.log(max(out["t90b"][k], 1e-6))
            out["ivl"][k] = d["ivl"][i]
    return {k: np.asarray(v, np.float64) for k, v in out.items()}


def card_difficulty(d):
    """Per card: tercile 0/1/2 of the real later-day Again share over the whole real history (cards with >= 2 real
    later-day reviews; the tercile cut points per user), -1 for cards with fewer (mostly new cards)."""
    n, a = d["card_later_n"], d["card_later_again"]
    share = np.where(n > 0, a / np.maximum(n, 1), np.nan)
    ok = n >= 2
    t = np.full(len(n), -1)
    if ok.sum() >= 3:
        q = np.quantile(share[ok], [1 / 3, 2 / 3])
        t[ok] = np.where(share[ok] <= q[0], 0, np.where(share[ok] <= q[1], 1, 2))
    return t


DIFF_TXT = {0: "easiest third", 1: "middle third", 2: "hardest third", -1: "thin history (< 2 real later-day reviews)"}


# ------------------------------------------------------------------ analyse
def analyse(args):
    paths = sorted(glob.glob(args.logs))
    L = ["# The small model in the review simulator (SM41): diagnostic tables", "",
         f"{len(paths)} users x 1 split (diag/sample.json), 60 days, target 0.90, instant-head desired retention "
         "(ihr-0.90) and FSRS-7 desired retention (fsrs7-0.90), 4 seeds each, daily shared-stream re-anchoring. "
         "Per-review logs from `scripts/sm41_engine.py run --log-dir`. Days 30-59 unless stated. 'True recall' = the "
         "instant head's P(pass) at the actual show moment: the simulated learner draws the grade from it. Seeds are "
         "averaged per user; mean ± 95% CI over users (t); 'paired' = FSRS-7 minus instant head per user.", ""]
    res = {}
    U = []                       # per user dict of per-family metrics
    for p in paths:
        d, lanes = load_log(p)
        diff = card_difficulty(d)
        per = {"user": Path(p).stem}
        for fam in FAMS:
            ks = [k for k, n in enumerate(lanes) if d["fam"][k] == fam]
            acc = {}
            for k in ks:
                ep = episodes(d, k)
                late = ep["day"] >= BURN
                m = d["lane"] == k
                # all reviews of the lane, days 30-59 (minutes of every kind)
                mall = m & (d["day"] >= BURN)
                vals = {}
                pt = ep["p"][late]
                vals["n_later"] = late.sum()
                vals["ret"] = ep["ok"][late].mean()
                vals["p_mean"] = pt.mean(); vals["p_sd"] = pt.std()
                for q in (0.1, 0.25, 0.5, 0.75, 0.9):
                    vals[f"p_q{int(q * 100)}"] = np.quantile(pt, q)
                vals["p_lt80"] = (pt < 0.8).mean(); vals["p_lt70"] = (pt < 0.7).mean()
                # per-card spread: mean true recall per card, realised pass share per card (cards with >= 2 reviews)
                cs = ep["card"][late]
                uc, inv, cnt = np.unique(cs, return_inverse=True, return_counts=True)
                cm = np.bincount(inv, pt) / cnt
                co = np.bincount(inv, ep["ok"][late].astype(float)) / cnt
                vals["card_p_sd"] = cm.std(); vals["card_ret_sd2"] = co[cnt >= 2].std() if (cnt >= 2).sum() > 2 else np.nan
                vals["card_p_q10"] = np.quantile(cm, 0.1); vals["card_p_q90"] = np.quantile(cm, 0.9)
                # intervals chosen at days 30-59 (every card leaving the day)
                iv = d["ivl"][mall & (d["ivl"] >= 0)]
                vals["ivl_med"] = np.median(iv); vals["ivl_p90"] = np.quantile(iv, 0.9); vals["ivl_mean"] = iv.mean()
                ivl_later = ep["ivl"][late & (ep["ivl"] >= 0)]
                vals["ivl_later_med"] = np.median(ivl_later); vals["ivl_later_p90"] = np.quantile(ivl_later, 0.9)
                vals["ed_med"] = np.median(ep["ed"][late]); vals["ed_p90"] = np.quantile(ep["ed"][late], 0.9)
                # load and memorised
                vals["minutes"] = d["ms"][mall].sum() / 60000.0
                vals["min_later"] = ep["min"][late].sum()
                vals["memorised"] = d["mem_card"][k].sum()
                vals["memorised_curve"] = d["mem_card_curve"][k].sum()
                # where the reviews go: difficulty tercile and interval band (later-day reviews, days 30-59)
                dl = diff[ep["card"][late].astype(int)]
                for t in DIFF_TXT:
                    selm = mall & (diff[d["card"].astype(int)] == t)
                    vals[f"n_diff{t}"] = (dl == t).sum()
                    vals[f"min_diff{t}"] = d["ms"][selm].sum() / 60000.0
                    vals[f"mem_diff{t}"] = d["mem_card"][k][diff == t].sum()
                    vals[f"p_diff{t}"] = pt[dl == t].mean() if (dl == t).any() else np.nan
                edl = ep["ed"][late]
                for lo, hi in zip(IVL_BANDS[:-1], IVL_BANDS[1:]):
                    sb = (edl >= lo) & (edl < hi)
                    vals[f"n_band{lo}"] = sb.sum()
                    vals[f"min_band{lo}"] = ep["min"][late][sb].sum()
                    # calibration at the chosen interval: the policy's aim vs true recall vs drawn outcome
                    a = ep["aim"][late][sb]; ok_ = np.isfinite(a)
                    vals[f"aim_band{lo}"] = a[ok_].mean() if ok_.any() else np.nan
                    vals[f"ptrue_band{lo}"] = pt[sb][ok_].mean() if ok_.any() else np.nan
                    vals[f"ret_band{lo}"] = ep["ok"][late][sb][ok_].mean() if ok_.any() else np.nan
                    vals[f"r7_band{lo}"] = ep["r7"][late][sb][ok_].mean() if ok_.any() else np.nan
                    vals[f"rc_band{lo}"] = ep["rc"][late][sb][ok_].mean() if ok_.any() else np.nan
                # calibration overall
                a = ep["aim"][late]; ok_ = np.isfinite(a)
                vals["aim_mean"] = a[ok_].mean(); vals["aim_minus_p"] = (a[ok_] - pt[ok_]).mean()
                vals["aim_abs_err"] = np.abs(a[ok_] - pt[ok_]).mean()
                vals["r7_minus_p"] = np.nanmean(ep["r7"][late] - pt)
                vals["r7_abs_err"] = np.nanmean(np.abs(ep["r7"][late] - pt))
                vals["rc_minus_p"] = np.nanmean(ep["rc"][late] - pt)
                # H5: gain per minute by true recall bin (all later-day episodes, days 0-59)
                for lo, hi in zip(R_BINS[:-1], R_BINS[1:]):
                    sb = (ep["p"] >= lo) & (ep["p"] < hi)
                    vals[f"h5n_{lo}"] = sb.sum()
                    vals[f"h5g_{lo}"] = ep["gain"][sb].sum()
                    vals[f"h5m_{lo}"] = ep["min"][sb].sum()
                    vals[f"h5ok_{lo}"] = ep["ok"][sb].sum()
                    vals[f"h5late_{lo}"] = (sb & late).sum()
                # the 1-day floor: reviews scheduled at 1 day because P(pass) at 1 day was already below the target
                fl = late & (ep["ed"] == 1) & (ep["aim"] < 0.88)
                vals["floor_share"] = fl.sum() / max(late.sum(), 1)
                vals["floor_p"] = ep["p"][fl].mean() if fl.any() else np.nan
                vals["floor_min_share"] = ep["min"][fl].sum() / max(ep["min"][late].sum(), 1e-9)
                # H5 stratified by the card's stability before the review (curve-head t90 before)
                for si, (slo, shi) in enumerate(T90_STRATA):
                    st_ = (ep["t90b"] >= slo) & (ep["t90b"] < shi)
                    for lo, hi in zip(R_BINS[:-1], R_BINS[1:]):
                        sb = st_ & (ep["p"] >= lo) & (ep["p"] < hi)
                        vals[f"s{si}n_{lo}"] = sb.sum(); vals[f"s{si}g_{lo}"] = ep["gain"][sb].sum()
                        vals[f"s{si}m_{lo}"] = ep["min"][sb].sum()
                vals["_ep"] = ep
                for kk, v in vals.items():
                    acc.setdefault(kk, []).append(v)
            per[fam] = {kk: (float(np.nanmean(v)) if kk != "_ep" else v) for kk, v in acc.items()}
            # card-level H5 link (FSRS-7 lanes: its error on each card; the card's gain per minute from the instant
            # head's lanes, a policy that does not share FSRS-7's errors)
        per["diff"] = diff
        U.append(per)

    def col(fam, key):
        return [u[fam][key] for u in U]

    def paired(key):
        return [u["fsrs7"][key] - u["ihr"][key] for u in U]

    # ---- 1. matched?
    L += ["## 1. Lanes in the sample (days 30-59)", "",
          "| metric | instant head | FSRS-7 | FSRS-7 - instant head (paired) |", "|---|---|---|---|"]
    for key, txt, sc, dd in (("ret", "realised retention", 1, 3), ("p_mean", "mean true recall", 1, 3),
                             ("aim_mean", "the policy's aim (its own prediction at the chosen interval)", 1, 3),
                             ("n_later", "later-day reviews", 1, 0), ("minutes", "minutes (every review)", 1, 0),
                             ("memorised", "memorised at day 60 (instant head)", 1, 0),
                             ("memorised_curve", "memorised at day 60 (curve head)", 1, 0)):
        L.append(f"| {txt} | {fmt_ci(col('ihr', key), sc, dd, False)} | {fmt_ci(col('fsrs7', key), sc, dd, False)} | "
                 f"{fmt_ci(paired(key), sc, dd)} |")
    cost = [u["ihr"]["minutes"] / u["ihr"]["memorised"] / (u["fsrs7"]["minutes"] / u["fsrs7"]["memorised"]) - 1 for u in U]
    L += ["", f"Cost gap at target 0.90 in the sample (instant head / FSRS-7 - 1, minutes 30-59 per memorised card): "
          f"{fmt_ci(cost, 100)} %.", ""]
    res["sample_cost_gap_0.90"] = ci(cost)

    # ---- 2. H1 / H5: true-recall spread
    L += ["## 2. True recall at review time (H1, H5): later-day reviews, days 30-59", "",
          "| statistic | instant head | FSRS-7 | paired difference |", "|---|---|---|---|"]
    for key, txt in (("p_mean", "mean"), ("p_sd", "SD over reviews"), ("p_q10", "10th percentile"),
                     ("p_q25", "25th percentile"), ("p_q50", "median"), ("p_q75", "75th percentile"),
                     ("p_q90", "90th percentile"), ("p_lt80", "share below 0.80"), ("p_lt70", "share below 0.70"),
                     ("card_p_sd", "SD over cards of the card's mean true recall"),
                     ("card_p_q10", "10th percentile over cards"), ("card_p_q90", "90th percentile over cards"),
                     ("card_ret_sd2", "SD over cards of the realised pass share (cards with >= 2 reviews)")):
        L.append(f"| {txt} | {fmt_ci(col('ihr', key), 1, 3, False)} | {fmt_ci(col('fsrs7', key), 1, 3, False)} | "
                 f"{fmt_ci(paired(key), 1, 3)} |")
        res[f"truer_{key}"] = {"ihr": ci(col("ihr", key)), "fsrs7": ci(col("fsrs7", key)), "paired": ci(paired(key))}

    # ---- 3. intervals
    L += ["", "## 3. Intervals", "", "| statistic, days | instant head | FSRS-7 | paired difference |", "|---|---|---|---|"]
    for key, txt in (("ivl_med", "median interval chosen (every card leaving the day, days 30-59)"),
                     ("ivl_p90", "90th percentile of the same"), ("ivl_mean", "mean of the same"),
                     ("ivl_later_med", "median interval chosen after a later-day review"),
                     ("ivl_later_p90", "90th percentile of the same"),
                     ("ed_med", "median elapsed days at later-day reviews"), ("ed_p90", "90th percentile of the same")):
        L.append(f"| {txt} | {fmt_ci(col('ihr', key), 1, 1, False)} | {fmt_ci(col('fsrs7', key), 1, 1, False)} | "
                 f"{fmt_ci(paired(key), 1, 1)} |")
        res[f"ivl_{key}"] = {"ihr": ci(col("ihr", key)), "fsrs7": ci(col("fsrs7", key)), "paired": ci(paired(key))}

    # ---- 4. where FSRS-7 saves
    L += ["", f"The 1-day floor (reviews at 1 day whose aim was < 0.88, i.e. P(pass) at 1 day was already below the "
          f"target): share of later-day reviews instant head {fmt_ci(col('ihr', 'floor_share'), 100, 1, False)} %, "
          f"FSRS-7 {fmt_ci(col('fsrs7', 'floor_share'), 100, 1, False)} %; their mean true recall instant head "
          f"{np.nanmean(col('ihr', 'floor_p')):.3f}, FSRS-7 {np.nanmean(col('fsrs7', 'floor_p')):.3f}; share of "
          f"later-day minutes instant head {fmt_ci(col('ihr', 'floor_min_share'), 100, 1, False)} %, FSRS-7 "
          f"{fmt_ci(col('fsrs7', 'floor_min_share'), 100, 1, False)} %."]
    res["floor"] = {f: {k: ci(col(f, k)) for k in ("floor_share", "floor_p", "floor_min_share")} for f in FAMS}
    L += ["", "## 4. Where FSRS-7 saves (days 30-59)", "",
          "Card difficulty = tercile of the card's real later-day Again share over its whole real history (per user).",
          "", "| card group | later-day reviews: instant head / FSRS-7 | paired difference | minutes: paired difference "
          "| memorised: paired difference | mean true recall: instant head / FSRS-7 |", "|---|---|---|---|---|---|"]
    for t, txt in DIFF_TXT.items():
        L.append(f"| {txt} | {np.mean(col('ihr', f'n_diff{t}')):.0f} / {np.mean(col('fsrs7', f'n_diff{t}')):.0f} | "
                 f"{fmt_ci(paired(f'n_diff{t}'), 1, 0)} | {fmt_ci(paired(f'min_diff{t}'), 1, 1)} | "
                 f"{fmt_ci(paired(f'mem_diff{t}'), 1, 1)} | {np.nanmean(col('ihr', f'p_diff{t}')):.3f} / "
                 f"{np.nanmean(col('fsrs7', f'p_diff{t}')):.3f} |")
        res[f"diff{t}"] = {"n": ci(paired(f"n_diff{t}")), "min": ci(paired(f"min_diff{t}")),
                           "mem": ci(paired(f"mem_diff{t}"))}
    L += ["", "| elapsed days at the review | later-day reviews: instant head / FSRS-7 | paired difference | "
          "minutes: paired difference |", "|---|---|---|---|"]
    for lo, hi in zip(IVL_BANDS[:-1], IVL_BANDS[1:]):
        L.append(f"| {band_label(lo, hi)} | {np.mean(col('ihr', f'n_band{lo}')):.0f} / "
                 f"{np.mean(col('fsrs7', f'n_band{lo}')):.0f} | {fmt_ci(paired(f'n_band{lo}'), 1, 0)} | "
                 f"{fmt_ci(paired(f'min_band{lo}'), 1, 1)} |")
        res[f"band{lo}"] = {"n": ci(paired(f"n_band{lo}")), "min": ci(paired(f"min_band{lo}"))}

    # ---- 5. H3 calibration at the chosen interval
    L += ["", "## 5. Calibration at the chosen interval (H3), later-day reviews days 30-59", "",
          "aim = the policy's own predicted recall at the interval it chose (instant head: the counterfactual candidate "
          "row; FSRS-7: its own curve); true = the instant head at the actual show moment; realised = the drawn pass "
          "share. Means over users (users with reviews in the band).", "",
          "| elapsed days | instant head: aim / true / realised | instant head aim - true | FSRS-7: aim / true / "
          "realised | FSRS-7 aim - true |", "|---|---|---|---|---|"]
    for lo, hi in zip(IVL_BANDS[:-1], IVL_BANDS[1:]):
        def g(fam, k):
            return np.nanmean(col(fam, f"{k}_band{lo}"))
        di = [u["ihr"][f"aim_band{lo}"] - u["ihr"][f"ptrue_band{lo}"] for u in U]
        df = [u["fsrs7"][f"aim_band{lo}"] - u["fsrs7"][f"ptrue_band{lo}"] for u in U]
        L.append(f"| {band_label(lo, hi)} | {g('ihr', 'aim'):.3f} / {g('ihr', 'ptrue'):.3f} / {g('ihr', 'ret'):.3f} | "
                 f"{fmt_ci(di, 100, 1)} pts | {g('fsrs7', 'aim'):.3f} / {g('fsrs7', 'ptrue'):.3f} / "
                 f"{g('fsrs7', 'ret'):.3f} | {fmt_ci(df, 100, 1)} pts |")
        res[f"calib_band{lo}"] = {"ihr_aim_minus_true": ci(di), "fsrs7_aim_minus_true": ci(df)}
    L += ["", f"Overall: instant head aim - true {fmt_ci(col('ihr', 'aim_minus_p'), 100, 1)} points (mean |aim - true| "
          f"{fmt_ci(col('ihr', 'aim_abs_err'), 100, 1, False)}); FSRS-7 aim - true "
          f"{fmt_ci(col('fsrs7', 'aim_minus_p'), 100, 1)} points (mean |aim - true| "
          f"{fmt_ci(col('fsrs7', 'aim_abs_err'), 100, 1, False)}). Curve head at the show moment - true: instant-head "
          f"lanes {fmt_ci(col('ihr', 'rc_minus_p'), 100, 1)}, FSRS-7 lanes {fmt_ci(col('fsrs7', 'rc_minus_p'), 100, 1)} "
          "points.", ""]
    res["calib"] = {"ihr_aim_minus_true": ci(col("ihr", "aim_minus_p")), "ihr_abs": ci(col("ihr", "aim_abs_err")),
                    "fsrs7_aim_minus_true": ci(col("fsrs7", "aim_minus_p")),
                    "fsrs7_abs": ci(col("fsrs7", "aim_abs_err")),
                    "curve_minus_true_ihr": ci(col("ihr", "rc_minus_p")),
                    "curve_minus_true_fsrs7": ci(col("fsrs7", "rc_minus_p"))}

    # ---- 6. H5 efficiency vs true recall
    L += ["## 6. Memory bought per minute vs true recall at the review (H5), every later-day review, days 0-59", "",
          "Episode = a later-day review + its same-day follow-ups (relearning steps). Gain = ln(t90 after the day) - "
          "ln(t90 before the review), t90 = the curve head's time to 90% recall (the card's 'stability' in the "
          "learner). Pooled over both policies' lanes; per user ratio of sums, then mean over users (users with >= 20 "
          "episodes in the bin).", "",
          "| true recall | share of reviews: instant head / FSRS-7 | pass share | gain per review | minutes per review "
          "| gain per minute |", "|---|---|---|---|---|---|"]
    curve_pts = []
    for lo, hi in zip(R_BINS[:-1], R_BINS[1:]):
        gpr, mpr, gpm, okr = [], [], [], []
        for u in U:
            n = u["ihr"][f"h5n_{lo}"] + u["fsrs7"][f"h5n_{lo}"]
            if n * 4 < 20:          # per-lane means: x 4 seeds
                continue
            gg = u["ihr"][f"h5g_{lo}"] + u["fsrs7"][f"h5g_{lo}"]
            mm = u["ihr"][f"h5m_{lo}"] + u["fsrs7"][f"h5m_{lo}"]
            oo = u["ihr"][f"h5ok_{lo}"] + u["fsrs7"][f"h5ok_{lo}"]
            gpr.append(gg / n); mpr.append(mm / n); gpm.append(gg / mm); okr.append(oo / n)
        shi = np.mean([u["ihr"][f"h5n_{lo}"] / sum(u["ihr"][f"h5n_{b}"] for b in R_BINS[:-1]) for u in U])
        shf = np.mean([u["fsrs7"][f"h5n_{lo}"] / sum(u["fsrs7"][f"h5n_{b}"] for b in R_BINS[:-1]) for u in U])
        hi_txt = "1.00" if hi > 1 else f"{hi:.2f}"
        L.append(f"| {lo:.2f}-{hi_txt} | {100 * shi:.1f}% / {100 * shf:.1f}% | {fmt_ci(okr, 1, 3, False)} | "
                 f"{fmt_ci(gpr, 1, 3)} | {fmt_ci(mpr, 1, 3, False)} | {fmt_ci(gpm, 1, 2)} |")
        curve_pts.append({"lo": lo, "hi": min(hi, 1.0), "users": len(gpm), "gain_per_review": ci(gpr),
                          "min_per_review": ci(mpr), "gain_per_min": ci(gpm), "pass": ci(okr),
                          "share_ihr": shi, "share_fsrs7": shf})
    res["h5_bins"] = curve_pts

    # stratified by stability before the review
    L += ["", "Stratified by the card's stability before the review (the curve head's t90), gain per minute (mean over "
          "users with >= 20 episodes in the cell):", "",
          "| true recall | " + " | ".join(f"t90 {a:g}-{b:g} d" if b < 1e8 else f"t90 {a:g}+ d" for a, b in T90_STRATA)
          + " |", "|---|" + "---|" * len(T90_STRATA)]
    res["h5_strata"] = {}
    for lo, hi in zip(R_BINS[:-1], R_BINS[1:]):
        cells = []
        for si in range(len(T90_STRATA)):
            v = []
            for u in U:
                n = u["ihr"][f"s{si}n_{lo}"] + u["fsrs7"][f"s{si}n_{lo}"]
                mm = u["ihr"][f"s{si}m_{lo}"] + u["fsrs7"][f"s{si}m_{lo}"]
                if n * 4 >= 20 and mm > 0:
                    v.append((u["ihr"][f"s{si}g_{lo}"] + u["fsrs7"][f"s{si}g_{lo}"]) / mm)
            cells.append(f"{fmt_ci(v, 1, 2)} ({len(v)})")
            res["h5_strata"][f"{si}_{lo}"] = ci(v)
        hi_txt = "1.00" if hi > 1 else f"{hi:.2f}"
        L.append(f"| {lo:.2f}-{hi_txt} | " + " | ".join(cells) + " |")

    # Jensen check per user: the user's own binned gain(R) and minutes(R) (pooled lanes), evaluated on the user's
    # FSRS-7 true-recall distribution vs on a constant at the same mean.
    jg, slope = [], []
    for u in U:
        eps = u["ihr"]["_ep"] + u["fsrs7"]["_ep"]
        P = np.concatenate([e["p"] for e in eps]); G = np.concatenate([e["gain"] for e in eps])
        M = np.concatenate([e["min"] for e in eps])
        ok = np.isfinite(P) & np.isfinite(G)
        P, G, M = P[ok], G[ok], M[ok]
        if len(P) < 200:
            continue
        # local-linear fits: gain(R), minutes(R) as quadratics in R
        cg = np.polyfit(P, G, 2); cm = np.polyfit(P, M, 2)
        pf = np.concatenate([e["p"][e["day"] >= BURN] for e in u["fsrs7"]["_ep"]])
        pf = pf[np.isfinite(pf)]
        spread = np.polyval(cg, pf).mean() / np.polyval(cm, pf).mean()
        const = np.polyval(cg, pf.mean()) / np.polyval(cm, pf.mean())
        jg.append(spread / const - 1)
        slope.append(cg[0])
    L += ["", f"Jensen check (each user's quadratic fit of gain and minutes on true recall, evaluated on the user's "
          f"FSRS-7 true-recall distribution vs a constant at the same mean): gain per minute of the spread / of the "
          f"constant - 1 = {fmt_ci(jg, 100, 1)} % over {len(jg)} users; curvature of gain(R) (quadratic coefficient) "
          f"{fmt_ci(slope, 1, 2)} (positive = convex).", ""]
    res["jensen"] = {"spread_over_const": ci(jg), "gain_curvature": ci(slope)}

    # ---- 7. where FSRS-7 lands low: its error vs the card's gain per minute
    rho, rho_d, err_t = [], [], {t: [] for t in DIFF_TXT}
    for u in U:
        diff = u["diff"]
        # card gain per minute from the instant-head lanes (episodes with true recall 0.8-0.95, the common range)
        gi = {}
        for e in u["ihr"]["_ep"]:
            for c, g_, m_ in zip(e["card"], e["gain"], e["min"]):
                a = gi.setdefault(int(c), [0.0, 0.0]); a[0] += g_; a[1] += m_
        ef = {}
        for e in u["fsrs7"]["_ep"]:
            late = e["day"] >= BURN
            for c, a_, p_ in zip(e["card"][late], e["aim"][late], e["p"][late]):
                if np.isfinite(a_):
                    ef.setdefault(int(c), []).append(a_ - p_)
        cs = [c for c in ef if c in gi and gi[c][1] > 0]
        if len(cs) < 20:
            continue
        x = np.array([np.mean(ef[c]) for c in cs]); y = np.array([gi[c][0] / gi[c][1] for c in cs])
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        rho.append(float(np.corrcoef(rx, ry)[0, 1]))
        for t in DIFF_TXT:
            v = [np.mean(ef[c]) for c in ef if diff[c] == t]
            if len(v) >= 5:
                err_t[t].append(float(np.mean(v)))
        dd = np.array([diff[c] for c in cs])
        okd = dd >= 0
        if okd.sum() > 20:
            rx2 = np.argsort(np.argsort(dd[okd])); rho_d.append(float(np.corrcoef(rx2, np.argsort(np.argsort(x[okd])))[0, 1]))
    L += ["## 7. Where FSRS-7 lands low (H5)", "",
          f"Per user, over cards: Spearman correlation between FSRS-7's mean overestimate on the card (its aim - true "
          f"recall, days 30-59) and the card's gain per minute in the instant-head lanes: {fmt_ci(rho, 1, 3)} over "
          f"{len(rho)} users (positive = FSRS-7 overestimates, i.e. reviews late, the cards where a review buys more "
          f"per minute). Correlation of the overestimate with the card's difficulty tercile: {fmt_ci(rho_d, 1, 3)}.", "",
          "| card group | FSRS-7 aim - true recall, points |", "|---|---|"]
    for t, txt in DIFF_TXT.items():
        L.append(f"| {txt} | {fmt_ci(err_t[t], 100, 1)} |")
    res["h5_link"] = {"rho_err_gpm": ci(rho), "rho_err_difficulty": ci(rho_d),
                      "err_by_diff": {str(t): ci(v) for t, v in err_t.items()}}

    # ---- 8. the answer-time mechanism (found by the diagnostic): who gets reviewed, and how slowly
    T9 = {f: [] for f in FAMS}
    for p in paths:
        d, lanes = load_log(p)
        m0 = (d["lane"] == 0) & (d["kind"] == 2)
        q = np.quantile(d["ms"][m0], [1 / 3, 2 / 3])          # the user's real later-day answer-time terciles
        for fam in FAMS:
            v = []
            for k in [k for k, n in enumerate(lanes) if d["fam"][k] == fam]:
                idx = np.nonzero(d["lane"] == k)[0]
                last, prev = {}, np.full(len(idx), np.nan)
                for j, i in enumerate(idx):
                    if d["kind"][i] == 2:
                        c = int(d["card"][i]); prev[j] = last.get(c, np.nan); last[c] = d["ms"][i]
                late = (d["day"][idx] >= BURN) & (d["kind"][idx] == 2)
                ms, pt = d["ms"][idx], d["p_true"][idx]
                row = [ms[late].mean() / 1000, late.sum()]
                for lo, hi in zip(R_BINS[3:-1], R_BINS[4:]):
                    sb = late & (pt >= lo) & (pt < hi)
                    row.append(ms[sb].mean() / 1000 if sb.any() else np.nan)
                sel = late & np.isfinite(prev)
                row.append(prev[sel].mean() / 1000)
                t = np.where(prev <= q[0], 0, np.where(prev <= q[1], 1, 2))
                for tt in (0, 1, 2):
                    sb = sel & (t == tt)
                    row += [sb.sum() / max(sel.sum(), 1), ms[sb].mean() / 1000, pt[sb].mean(), np.median(d["ed"][idx][sb])]
                v.append(row)
            T9[fam].append(np.nanmean(np.array(v, np.float64), 0))
    I9, F9 = np.array(T9["ihr"]), np.array(T9["fsrs7"])
    names = (["seconds per later-day review", "later-day reviews"] +
             [f"seconds per review at true recall {lo:.2f}-{min(hi, 1):.2f}" for lo, hi in zip(R_BINS[3:-1], R_BINS[4:])] +
             ["the card's previous answer time, s"] +
             [f"previous answer {t}: {x}" for t in ("fast (user's fastest third)", "middle", "slow (slowest third)")
              for x in ("share of reviews", "seconds per review", "true recall", "median elapsed days")])
    L += ["## 8. Answer time: where the minutes go (later-day reviews, days 30-59)", "",
          "Previous answer = the card's previous later-day answer time in the lane, in the user's real later-day "
          "answer-time terciles (teacher-forcing lane).", "",
          "| quantity | instant head | FSRS-7 | FSRS-7 - instant head (paired) |", "|---|---|---|---|"]
    res["time"] = {}
    for i, nm in enumerate(names):
        a, b = I9[:, i], F9[:, i]
        dd = 0 if "reviews" in nm and "share" not in nm else 3
        L.append(f"| {nm} | {np.nanmean(a):.{dd}f} | {np.nanmean(b):.{dd}f} | {fmt_ci(b - a, 1, dd)} |")
        res["time"][nm] = {"ihr": float(np.nanmean(a)), "fsrs7": float(np.nanmean(b)), "paired": ci(b - a)}
    L.append("")

    # ---- 9. H2 and H4 from the full-run rows (all users)
    L += [""] + fullrun_h2_h4(res)
    Path(DIAG / "diag-tables.md").write_text("\n".join(L) + "\n", encoding="utf8")
    Path(DIAG / "diag-result.json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(L))


def fullrun_h2_h4(res):
    rows = FR.load_rows(str(FULL / "rows" / "*.jsonl"))
    per = {}
    for r in rows:
        per.setdefault(r["user"], []).append((r, FR.split_policies(r)))
    users = sorted(per)
    L = []
    # H2: memorised by curve vs instant head, and the matched-retention gap with the curve head's memorised
    def with_curve(r, s):
        groups = {}
        for k, name in enumerate(r["lanes"]):
            groups.setdefault(name.split("#")[0], []).append(k)
        for b, ks in groups.items():
            mc = float(np.mean([r["memorised_curve"][k] for k in ks]))
            s[b]["memorised_curve"] = mc
            s[b]["cost_curve"] = s[b]["min_30_59"] / max(mc, 1e-9)
            # decomposition of the minutes (days 30-59): later-day reviews, seconds per later-day review, the
            # minutes of same-day steps
            ld = np.asarray(r["acc"]["load"])[ks][:, BURN // r["bucket_days"]:].sum(1)      # [lanes, 6]
            s[b]["n_later"] = float(ld[:, 2].mean())
            s[b]["s_per_later"] = float((ld[:, 5] / np.maximum(ld[:, 2], 1)).mean())
            s[b]["min_same"] = float(ld[:, 4].mean() / 60)
            s[b]["min_later"] = float(ld[:, 5].mean() / 60)
        return s
    gaps = {k: [] for k in ("cost", "cost_curve", "min_30_59", "memorised", "memorised_curve", "n_later",
                            "s_per_later", "min_later", "min_same")}
    ratio = {f: [] for f in FAMS}
    for u in users:
        vals = {k: [] for k in gaps}
        for r, s in per[u]:
            s = with_curve(r, s)
            ih = [s[f"ihr-{x:.2f}"] for x in FR.LEVELS]; fs = [s[f"fsrs7-{x:.2f}"] for x in FR.LEVELS]
            ih = [x for x in ih if np.isfinite(x["ret"])]; fs = [x for x in fs if np.isfinite(x["ret"])]
            for f, pts in (("ihr", ih), ("fsrs7", fs)):
                if pts:
                    ratio[f].append(np.mean([x["memorised_curve"] / x["memorised"] for x in pts]))
            if len(ih) < 2 or len(fs) < 2:
                continue
            lo = max(min(p["ret"] for p in ih), min(p["ret"] for p in fs))
            hi = min(max(p["ret"] for p in ih), max(p["ret"] for p in fs))
            if not hi - lo >= FR.MIN_OVERLAP:
                continue
            grid = np.linspace(lo, hi, max(2, int(round((hi - lo) / 0.005)) + 1))
            for k in gaps:
                vals[k].append(float(np.mean([FR.interp_cost(ih, x, k) / FR.interp_cost(fs, x, k) - 1 for x in grid])))
        for k in gaps:
            if vals[k]:
                gaps[k].append(float(np.mean(vals[k])))
    L += ["## 9. From the full-run rows (all users, every target)", "",
          "### H2: is the instant head's memorised count the issue? (matched realised retention, common range, "
          "paired per user, the primary metric's method)", "",
          "| quantity read off the two curves | instant head / FSRS-7 - 1, % (95% CI) |", "|---|---|"]
    txt = {"cost": "cost, memorised by the instant head (the primary metric)",
           "cost_curve": "cost, memorised by the curve head",
           "min_30_59": "minutes on days 30-59", "memorised": "memorised at day 60, instant head",
           "memorised_curve": "memorised at day 60, curve head",
           "n_later": "later-day reviews on days 30-59", "s_per_later": "seconds per later-day review",
           "min_later": "minutes of later-day reviews", "min_same": "minutes of same-day steps"}
    for k in gaps:
        L.append(f"| {txt[k]} | {fmt_ci(gaps[k], 100, 1)} (median {100 * np.median(gaps[k]):+.1f}; {len(gaps[k])} users) |")
    L += ["", f"Memorised by the curve head / by the instant head (mean over splits and targets): instant-head lanes "
          f"{np.mean(ratio['ihr']):.3f}, FSRS-7 lanes {np.mean(ratio['fsrs7']):.3f}.", ""]
    res["h2"] = {k: ci(v) for k, v in gaps.items()}
    res["h2_curve_over_instant"] = {f: float(np.mean(v)) for f, v in ratio.items()}

    # H4: minutes gap by 15-day bucket at matched realised retention (days 30-59 retention as the x axis)
    nb = 4
    bk = {b: [] for b in range(nb)}
    t90 = {b: [] for b in range(nb)}
    for u in users:
        vals = {b: [] for b in range(nb)}
        v90 = {b: [] for b in range(nb)}
        for r, s in per[u]:
            groups = {}
            for k, name in enumerate(r["lanes"]):
                groups.setdefault(name.split("#")[0], []).append(k)
            for base, ks in groups.items():
                if base not in s:
                    continue
                for b in range(nb):
                    s[base][f"min_b{b}"] = float(np.mean([np.asarray(r["acc"]["load"])[k, b, 3:].sum() / 60 for k in ks]))
            ih = [s[f"ihr-{x:.2f}"] for x in FR.LEVELS]; fs = [s[f"fsrs7-{x:.2f}"] for x in FR.LEVELS]
            for b in range(nb):
                v90[b].append(s["ihr-0.90"][f"min_b{b}"] / max(s["fsrs7-0.90"][f"min_b{b}"], 1e-9) - 1)
            ih = [x for x in ih if np.isfinite(x["ret"])]; fs = [x for x in fs if np.isfinite(x["ret"])]
            if len(ih) < 2 or len(fs) < 2:
                continue
            lo = max(min(p["ret"] for p in ih), min(p["ret"] for p in fs))
            hi = min(max(p["ret"] for p in ih), max(p["ret"] for p in fs))
            if not hi - lo >= FR.MIN_OVERLAP:
                continue
            grid = np.linspace(lo, hi, max(2, int(round((hi - lo) / 0.005)) + 1))
            for b in range(nb):
                vals[b].append(float(np.mean([FR.interp_cost(ih, x, f"min_b{b}") / FR.interp_cost(fs, x, f"min_b{b}") - 1
                                              for x in grid])))
        for b in range(nb):
            if vals[b]:
                bk[b].append(float(np.mean(vals[b])))
            t90[b].append(float(np.mean(v90[b])))
    L += ["### H4: the minutes gap by 15-day bucket", "",
          "Minutes per bucket, instant head / FSRS-7 - 1: at matched realised retention (days 30-59 retention as the "
          "common axis, the primary metric's method) and at the same target 0.90.", "",
          "| days | matched realised retention, % (95% CI) | target 0.90, % (95% CI) |", "|---|---|---|"]
    for b in range(nb):
        L.append(f"| {15 * b}-{15 * b + 14} | {fmt_ci(bk[b], 100, 1)} (median {100 * np.median(bk[b]):+.1f}) | "
                 f"{fmt_ci(t90[b], 100, 1)} (median {100 * np.median(t90[b]):+.1f}) |")
        res[f"h4_bucket{b}"] = {"matched": ci(bk[b]), "target_0.90": ci(t90[b])}
    # trend: bucket 3 - bucket 2 (paired)
    d32 = [a - b for a, b in zip(bk[3], bk[2])] if len(bk[3]) == len(bk[2]) else []
    L += ["", f"Change from days 30-44 to days 45-59 (matched): {fmt_ci(d32, 100, 1)} points.", ""]
    res["h4_trend_45_59_minus_30_44"] = ci(d32)
    return L


def main():
    ap = ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    t = sub.add_parser("analyse"); t.add_argument("--logs", default=str(DIAG / "logs" / "*.npz"))
    t.add_argument("--md", default=str(DIAG / "diag-tables.md")); t.add_argument("--json", default=str(DIAG / "diag-result.json"))
    a = ap.parse_args()
    {"check": check, "analyse": analyse}[a.cmd](a)


if __name__ == "__main__":
    main()
