"""Aggregate analysis of the RWKV Memorised day-to-day swings (input: replay_memorised.py's scratch replay.npz).

Writes only aggregate outputs next to this file: daily.csv (one row per day), buckets.csv, summary.json and PNG charts.
Per-card values stay in the scratch folder.
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--scratch", required=True)
parser.add_argument("--window-start", default="2025-12-01")
args = parser.parse_args()
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
EVAL = ROOT / "artifacts/collection-eval/2026-10-14"
from rwkv.data_processing import get_rwkv_data  # noqa: E402

z = np.load(Path(args.scratch) / "replay.npz")
full, q0, qg, qs = z["full"].astype(np.float64), z["q0"].astype(np.float64), z["qg"].astype(np.float64), z["qs"].astype(np.float64)
days, day0 = z["days"], int(z["day0"])
if "completed_through" in z.files:  # partial replay: keep the finished days only
    keep = days <= int(z["completed_through"])
    days, full, q0, qg, qs = days[keep], full[keep], q0[keep], qg[keep], qs[keep]
first_date = pd.Timestamp(str(z["first_date"]))
dates = first_date + pd.to_timedelta(days - day0, unit="D")
card_ids = z["card_ids"]
nd, nc = full.shape

# ---- card status from the collection copy (suspended etc., as of the copy)
con = sqlite3.connect(f"file:{(EVAL / 'collection-copy/collection.anki2').as_posix()}?mode=ro&immutable=1", uri=True)
cq = pd.read_sql("select id, queue, did, odid from cards", con).set_index("id")
con.close()
queue = cq.queue.reindex(card_ids).to_numpy()
suspended = queue == -1

# ---- review frame: per-day stats and FSRS-6 (default parameters) replay
frame = get_rwkv_data(EVAL / "collection-dataset", 1).reset_index(drop=True)
cidx = np.searchsorted(card_ids, frame.card_id.to_numpy())
W = [0.212, 1.2931, 2.3065, 8.2956, 6.4133, 0.8334, 3.0194, 0.001, 1.8722, 0.1666, 0.796, 1.4835, 0.0614, 0.2629,
     1.6483, 0.6014, 1.8729, 0.5425, 0.0912, 0.0658, 0.1542]
DEC = W[20]
FACTOR = 0.9 ** (-1 / DEC) - 1


def fsrs_r(t, s):
    return (1 + FACTOR * t / s) ** (-DEC)


def d0(g):
    return min(10.0, max(1.0, W[4] - np.exp(W[5] * (g - 1)) + 1))


S = np.full(nc, np.nan)
Dd = np.full(nc, np.nan)
last = np.full(nc, -1)
fs_by_day = {}
f_day = frame.day_offset.to_numpy()
f_rating = frame.rating.to_numpy()
starts = np.searchsorted(f_day, np.arange(day0, days[-1] + 2))
fsrs = np.full((nd, nc), np.nan)
for d in range(day0, days[-1] + 1):
    for j in range(starts[d - day0], starts[d - day0 + 1]):
        c, g = cidx[j], int(f_rating[j])
        if np.isnan(S[c]):
            S[c], Dd[c] = W[g - 1], d0(g)
        else:
            t = d - last[c]
            if t == 0:
                inc = np.exp(W[17] * (g - 3 + W[18])) * S[c] ** (-W[19])
                S[c] = S[c] * (max(inc, 1.0) if g >= 3 else inc)
            else:
                r = fsrs_r(t, S[c])
                if g == 1:
                    S[c] = min(S[c], W[11] * Dd[c] ** (-W[12]) * ((S[c] + 1) ** W[13] - 1) * np.exp(W[14] * (1 - r)))
                else:
                    S[c] = S[c] * (1 + np.exp(W[8]) * (11 - Dd[c]) * S[c] ** (-W[9]) * (np.exp(W[10] * (1 - r)) - 1)
                                   * (W[15] if g == 2 else 1) * (W[16] if g == 4 else 1))
            dd = Dd[c] + (-W[6] * (g - 3)) * (10 - Dd[c]) / 9
            Dd[c] = min(10.0, max(1.0, W[7] * d0(4) + (1 - W[7]) * dd))
        S[c] = max(S[c], 0.001)
        last[c] = d
    if d >= days[0]:
        a = last >= 0
        fsrs[d - days[0], a] = fsrs_r(d - last[a], S[a])
        if d == days[0]:
            pass
    if d == days[0] - 1:
        pass
# stability / elapsed per card on each query day (FSRS S after the last review), for bucketing
# (recomputed cheaply: S frozen between reviews, so the S used at day k is the S at the day's end)

# ---- daily stats
fr = frame.assign(lapse=(frame.rating == 1) & (frame.state == 2), again=frame.rating == 1, cidx=cidx)
g = fr.groupby("day_offset")
stats = pd.DataFrame({"reviews": g.size(), "again": g.again.sum(), "lapses": g.lapse.sum(),
                      "new_cards": g.is_first_review.sum() if "is_first_review" in fr else 0,
                      "top_deck_share": g.deck_id.agg(lambda s: s.value_counts(normalize=True).iloc[0]),
                      "distinct_cards": g.card_id.nunique()})
stats = stats.reindex(days, fill_value=0)
stats["again_rate"] = np.where(stats.reviews > 0, stats.again / stats.reviews.clip(lower=1), np.nan)
stats["no_review"] = (stats.reviews == 0).astype(int)
stats["weekday"] = dates.dayofweek
# gap: days since the previous review day
rev_days = np.unique(f_day)
prev_rev = np.array([rev_days[np.searchsorted(rev_days, d) - 1] if np.searchsorted(rev_days, d) > 0 else d for d in days])
stats["gap_days"] = np.where(stats.reviews > 0, days - np.array([rev_days[max(0, np.searchsorted(rev_days, d) - 1)] for d in days]), 0)

# ---- per-day decomposition of the change in the total (all cards seen so far)
reviewed = np.zeros((nd, nc), bool)
for k, d in enumerate(days):
    reviewed[k, cidx[starts[d - day0]:starts[d - day0 + 1]]] = True


def decompose(mask_cards):
    rows = []
    for k in range(1, nd):
        prev = full[k - 1]
        existed = ~np.isnan(prev) & mask_cards
        new = np.isnan(prev) & ~np.isnan(full[k]) & mask_cards
        rev = existed & reviewed[k]
        nr = existed & ~reviewed[k]
        rows.append({
            "new_cards": np.nansum(full[k, new]),
            "reviewed": np.nansum(full[k, rev] - prev[rev]),
            "time": np.nansum(q0[k, nr] - prev[nr]),
            "global_state": np.nansum(np.where(np.isnan(qg[k, nr]), q0[k, nr], qg[k, nr]) - q0[k, nr]),
            "deck_preset_note_state": np.nansum(np.where(np.isnan(qs[k, nr]), q0[k, nr], qs[k, nr])
                                                - np.where(np.isnan(qg[k, nr]), q0[k, nr], qg[k, nr])),
            "tracker_features": np.nansum(full[k, nr] - np.where(np.isnan(qs[k, nr]), q0[k, nr], qs[k, nr])),
        })
    out = pd.DataFrame(rows, index=days[1:])
    return out


comp_all = decompose(np.ones(nc, bool))
comp_ns = decompose(~suspended)
total_all = np.nansum(full, axis=1)
total_ns = np.nansum(np.where(suspended[None, :], np.nan, full), axis=1)
fsrs_all = np.nansum(fsrs, axis=1)
fsrs_ns = np.nansum(np.where(suspended[None, :], np.nan, fsrs), axis=1)

daily = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "rwkv_total_all": total_all, "rwkv_total_unsuspended": total_ns,
                      "fsrs6_default_total_all": fsrs_all, "fsrs6_default_total_unsuspended": fsrs_ns,
                      "cards_seen": (~np.isnan(full)).sum(1)}, index=days)
daily = daily.join(comp_ns.add_prefix("d_")).join(stats)
daily["d_total"] = daily.rwkv_total_unsuspended.diff()
daily["d_fsrs"] = daily.fsrs6_default_total_unsuspended.diff()
daily.round(3).to_csv(HERE / "daily.csv", index_label="anki_day")

win = daily[daily.date >= args.window_start].dropna(subset=["d_total"])
comps = ["new_cards", "reviewed", "time", "global_state", "deck_preset_note_state", "tracker_features"]
dt_ = win.d_total
var = dt_.var()
shares = {c: float(np.cov(win["d_" + c], dt_)[0, 1] / var) for c in comps}
summary = {"window_start": args.window_start, "days": int(len(win)), "cards_seen_end": int(daily.cards_seen.iloc[-1]),
           "unsuspended_seen_end": int((~np.isnan(full[-1]) & ~suspended).sum()),
           "rwkv_total_end_unsuspended": float(daily.rwkv_total_unsuspended.iloc[-1]),
           "fsrs_total_end_unsuspended": float(daily.fsrs6_default_total_unsuspended.iloc[-1]),
           "sd_daily_change_rwkv": float(dt_.std()), "sd_daily_change_fsrs": float(win.d_fsrs.std()),
           "mean_abs_daily_change_rwkv": float(dt_.abs().mean()), "mean_abs_daily_change_fsrs": float(win.d_fsrs.abs().mean()),
           "variance_share_of_daily_change": shares,
           "sd_component": {c: float(win["d_" + c].std()) for c in comps},
           "mean_component": {c: float(win["d_" + c].mean()) for c in comps}}

# ---- correlations with day stats (univariate R^2 of the daily change)
for target in ("d_total",):
    r2 = {}
    for s in ("reviews", "again", "again_rate", "lapses", "new_cards", "no_review", "gap_days", "top_deck_share", "distinct_cards"):
        x = win[s].astype(float)
        ok = x.notna()
        r2[s] = float(np.corrcoef(x[ok], win[target][ok])[0, 1] ** 2) if x[ok].std() > 0 else None
        r2[s + "_corr"] = float(np.corrcoef(x[ok], win[target][ok])[0, 1]) if x[ok].std() > 0 else None
    wd = win.groupby("weekday")[target].mean()
    ss_wd = float(((win.weekday.map(wd) - win[target].mean()) ** 2).sum() / ((win[target] - win[target].mean()) ** 2).sum())
    r2["weekday_eta2"] = ss_wd
    summary["r2_vs_day_stats"] = r2
    summary["mean_change_by_weekday"] = {int(k): float(v) for k, v in wd.items()}
    summary["mean_component_by_weekday"] = {c: {int(k): float(v) for k, v in win.groupby("weekday")["d_" + c].mean().items()}
                                            for c in comps}
    summary["weekday_eta2_by_component"] = {
        c: float(((win.weekday.map(win.groupby("weekday")["d_" + c].mean()) - win["d_" + c].mean()) ** 2).sum()
                 / ((win["d_" + c] - win["d_" + c].mean()) ** 2).sum()) for c in comps}
# multivariate OLS
X = win[["reviews", "again", "lapses", "no_review", "gap_days"]].astype(float).fillna(0)
X = np.column_stack([np.ones(len(X)), X.to_numpy()])
beta, *_ = np.linalg.lstsq(X, dt_.to_numpy(), rcond=None)
resid = dt_.to_numpy() - X @ beta
summary["ols_r2_reviews_again_lapses_noreview_gap"] = float(1 - resid.var() / dt_.var())

# ---- buckets (non-reviewed, unsuspended cards): R of the previous day, elapsed, card age, deck
kwin = np.flatnonzero(np.isin(days, win.index))
lastrev = np.full(nc, -1)
first_seen = np.full(nc, -1)
ap = []
B_R = [0, 0.5, 0.8, 0.95, 1.01]
bucket_rows = []
logit_rows = []


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# card last-review day per query day
lr = np.full((nd, nc), -1)
cur = np.full(nc, -1)
fs = np.full(nc, -1)
fsd = np.full((nd, nc), -1)
for d in range(day0, days[-1] + 1):
    cur[cidx[starts[d - day0]:starts[d - day0 + 1]]] = d
    fs[(fs < 0) & (cur >= 0)] = d
    if d >= days[0]:
        lr[d - days[0]] = cur
        fsd[d - days[0]] = fs
deck_of = z["card_deck"]
for k in kwin:
    prev = full[k - 1]
    nr = ~np.isnan(prev) & ~reviewed[k] & ~suspended
    eff = full[k] - q0[k]          # whole review-day effect on non-reviewed cards (states + tracker)
    tim = q0[k] - prev
    el = days[k] - lr[k - 1]
    age = days[k] - fsd[k - 1]
    for name, val, edges in (("R_prev", prev, B_R), ("elapsed_days", el, [0, 3, 10, 30, 90, 365, 10 ** 6]),
                             ("card_age_days", age, [0, 30, 90, 365, 10 ** 6])):
        b = np.digitize(val, edges) - 1
        for i in range(len(edges) - 1):
            m = nr & (b == i)
            bucket_rows.append({"day": days[k], "kind": name, "bucket": f"[{edges[i]},{edges[i + 1]})",
                                "cards": int(m.sum()), "review_day_effect": float(eff[m].sum()), "time_effect": float(tim[m].sum())})
    for dk in np.unique(deck_of):
        m = nr & (deck_of == dk)
        bucket_rows.append({"day": days[k], "kind": "deck", "bucket": str(dk), "cards": int(m.sum()),
                            "review_day_effect": float(eff[m].sum()), "time_effect": float(tim[m].sum())})
    # uniform logit shift test
    dl = logit(full[k, nr]) - logit(q0[k, nr])
    p = q0[k, nr]
    med = np.median(dl) if len(dl) else 0
    logit_rows.append({"day": days[k], "actual": float(eff[nr].sum()), "uniform_pred": float((p * (1 - p) * med).sum()),
                       "mean_dlogit": float(dl.mean()), "sd_dlogit": float(dl.std()), "median_dlogit": float(med)})
bk = pd.DataFrame(bucket_rows)
lg = pd.DataFrame(logit_rows)
tot_eff = win.d_deck_preset_note_state + win.d_global_state + win.d_tracker_features
var_eff = tot_eff.var()
bsum = []
for (kind, bucket), grp in bk.groupby(["kind", "bucket"]):
    s = grp.set_index("day").reindex(win.index).fillna(0)
    bsum.append({"kind": kind, "bucket": bucket, "mean_cards": s.cards.mean(),
                 "share_of_review_day_effect_variance": float(np.cov(s.review_day_effect, tot_eff)[0, 1] / var_eff),
                 "share_of_total_change_variance": float(np.cov(s.review_day_effect + s.time_effect, dt_)[0, 1] / var),
                 "sd_review_day_effect": float(s.review_day_effect.std()), "mean_time_effect": float(s.time_effect.mean())})
bsum = pd.DataFrame(bsum)
bsum.round(4).to_csv(HERE / "buckets.csv", index=False)
summary["uniform_logit_shift"] = {"corr_actual_vs_uniform": float(np.corrcoef(lg.actual, lg.uniform_pred)[0, 1]),
                                  "r2_uniform": float(1 - ((lg.actual - lg.uniform_pred) ** 2).sum() / ((lg.actual - lg.actual.mean()) ** 2).sum()),
                                  "median_abs_mean_dlogit": float(lg.mean_dlogit.abs().median()),
                                  "median_sd_dlogit": float(lg.sd_dlogit.median())}
lg.round(4).to_csv(HERE / "logit_shift_daily.csv", index=False)

# ---- example days
cols = ["date", "d_total", "d_fsrs"] + ["d_" + c for c in comps] + ["reviews", "again", "lapses", "gap_days", "top_deck_share"]
ex = pd.concat([win.nlargest(3, "d_total"), win.nsmallest(3, "d_total")])[cols]
summary["examples"] = ex.round(1).to_dict(orient="records")
(HERE / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
print(json.dumps({k: summary[k] for k in summary if k != "examples"}, indent=1, default=float)[:5000])
print(ex.round(1).to_string())
print(bsum.round(3).to_string())

# ---- charts
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

wd_ = win.copy()
x = pd.to_datetime(wd_.date)
fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
ax[0].plot(x, wd_.rwkv_total_unsuspended, color="#2a6fdb", lw=1.4, label="RWKV (fork model, rebuilt)")
ax[0].plot(x, wd_.fsrs6_default_total_unsuspended, color="#d97a1e", lw=1.4, label="FSRS-6 (default parameters)")
ax[0].set_ylabel("Memorised cards (sum of R)")
ax[0].legend(frameon=False)
ax[0].set_title("Memorised, unsuspended cards: RWKV vs FSRS-6")
colors = {"time": "#9aa5b1", "reviewed": "#3aa76d", "new_cards": "#b5d99c", "global_state": "#c0392b",
          "deck_preset_note_state": "#8e44ad", "tracker_features": "#e5a50a"}
pos = np.zeros(len(wd_))
neg = np.zeros(len(wd_))
for c in comps:
    v = wd_["d_" + c].to_numpy()
    ax[1].bar(x, np.where(v > 0, v, 0), bottom=pos, color=colors[c], width=1.0, label=c.replace("_", " "))
    ax[1].bar(x, np.where(v < 0, v, 0), bottom=neg, color=colors[c], width=1.0)
    pos += np.where(v > 0, v, 0)
    neg += np.where(v < 0, v, 0)
ax[1].plot(x, wd_.d_total, color="black", lw=0.8, label="RWKV daily change")
ax[1].plot(x, wd_.d_fsrs, color="#d97a1e", lw=0.8, label="FSRS-6 daily change")
ax[1].axhline(0, color="black", lw=0.5)
ax[1].set_ylabel("Change vs previous day (cards)")
ax[1].legend(frameon=False, ncol=4, fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "daily-decomposition.png", dpi=110)

sub = bsum[bsum.kind == "R_prev"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(sub.bucket, sub.share_of_review_day_effect_variance, color="#c0392b")
ax[0].set_title("Share of the review-day shift variance\n(non-reviewed cards, by previous-day R)")
ax[1].bar(sub.bucket, sub.mean_cards, color="#2a6fdb")
ax[1].set_title("Mean number of cards per bucket")
fig.tight_layout()
fig.savefig(HERE / "swing-by-r-bucket.png", dpi=110)
print("charts written")
