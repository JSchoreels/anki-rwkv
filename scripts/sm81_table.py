"""Scheduling-time vs review-time accuracy (SM81): the result table (CPU).

  sm81_table.py --rows R.jsonl --dump DIR --fsrs DIR --label-small L --label-big L [--existing-small F --existing-big F]
                --out-md M --out-json J

Common row set, per user: the benchmark's FSRS-7 test reviews (5 time-series test folds of the published FSRS-7
protocol, so reviews whose card was last reviewed on an earlier day) that are also RWKV benchmark reviews with all
three RWKV predictions (scheduling time = curve head from the card's previous answer, review time = instant head on the
query row, start of day = instant head from the state before the review's day), for both RWKV models. The y labels
of the two sides must agree on every common review (else: exit 1). LogLoss: float64, predictions clipped to
[eps, 1-eps] (sklearn's log_loss). Review-weighted = all common reviews pooled; per user = mean of per-user LogLoss.
CIs: paired user bootstrap, 2,000 resamples, seed 0, of the difference vs FSRS-7 (per-user fit) at scheduling time.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

EPS = np.finfo(np.float64).eps
POINTS = ("sched", "sod", "review")
POINT_NAMES = {"sched": "scheduling time", "sod": "start of day", "review": "review time"}


def losses(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    y = np.asarray(y, dtype=np.float64)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def load_rows(path, label):
    out = {}
    for r in map(json.loads, Path(path).read_text().splitlines()):
        if r.get("label") == label and "benchmark" in r:
            out[r["user_id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--fsrs", required=True)
    ap.add_argument("--label-small", required=True)
    ap.add_argument("--label-big", required=True)
    ap.add_argument("--existing-small", default="")
    ap.add_argument("--existing-big", default="")
    ap.add_argument("--min-common", type=float, default=0.5, help="fail if a user's common rows < this share of its "
                    "FSRS-7 test reviews (a sign of misaligned review ids)")
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    labels = {"small": a.label_small, "big": a.label_big}
    rows = {k: load_rows(a.rows, v) for k, v in labels.items()}
    fsrs_dir = Path(a.fsrs)
    fsrs_users = {json.loads(line)["user"] for line in (fsrs_dir / "users.jsonl").open()}
    users = sorted(set(rows["small"]) & set(rows["big"]) & fsrs_users)
    missing = {k: sorted((set(rows["small"]) | set(rows["big"]) | fsrs_users) - set(v))
               for k, v in (("small", rows["small"]), ("big", rows["big"]), ("fsrs", fsrs_users))}
    fail = []
    models = ["fsrs7_default", "fsrs7_fit", "fsrs7_instant", "small", "big"]
    S = {m: {p: [] for p in POINTS} for m in models}    # per-user loss sums
    N, per_user = [], []
    split = {m: {p: {"first": [], "later": []} for p in POINTS} for m in ("fsrs7_instant", "small", "big")}
    n_split = {"first": [], "later": []}
    counts = {"fsrs_test": 0, "rwkv_labels": 0, "common": 0, "fsrs_not_in_rwkv": 0, "rwkv_not_in_fsrs": 0,
              "rwkv_not_in_fsrs_same_day": 0}
    checks = {"sod_curve_max_abs_diff": 0.0, "sod_first_of_day_max_abs_diff": 0.0, "sod_curve_logloss_max_abs_diff": 0.0,
              "existing_rows_max_abs_diff": {}}
    import pandas as pd
    data = Path("C:/Users/jonat/workspace/anki-rwkv/data/anki-revlogs-10k-original-20261011")
    for u in users:
        f = np.load(fsrs_dir / f"{u}.npz")
        d = {k: np.load(Path(a.dump) / f"{u}_{labels[k].replace('/', '_')}.npz") for k in labels}
        if not np.array_equal(d["small"]["review_th"], d["big"]["review_th"]) or not np.array_equal(d["small"]["y"], d["big"]["y"]):
            fail.append(f"user {u}: the two RWKV dumps have different reviews / labels")
            continue
        r_th = d["small"]["review_th"].astype(np.int64)
        common, fi_, ri = np.intersect1d(f["review_th"], r_th, return_indices=True)
        n_reviews = rows["small"][u]["reviews"]
        if len(r_th) and r_th.max() > n_reviews or f["review_th"].max() > n_reviews:
            fail.append(f"user {u}: review_th beyond the user's {n_reviews} reviews")
        if not np.array_equal(f["y"][fi_].astype(np.int8), d["small"]["y"][ri].astype(np.int8)):
            fail.append(f"user {u}: y differs on {int((f['y'][fi_] != d['small']['y'][ri]).sum())} common reviews")
            continue
        if len(common) < a.min_common * len(f["review_th"]):
            fail.append(f"user {u}: only {len(common)} of {len(f['review_th'])} FSRS-7 test reviews are RWKV reviews")
            continue
        raw = pd.read_parquet(data / "revlogs" / f"user_id={u}", columns=["day_offset", "elapsed_days"])
        day = raw["day_offset"].to_numpy()
        elapsed = raw["elapsed_days"].to_numpy()
        first_of_day = np.ones(len(day), dtype=bool)
        first_of_day[1:] = day[1:] != day[:-1]
        counts["fsrs_test"] += len(f["review_th"])
        counts["rwkv_labels"] += len(r_th)
        counts["common"] += len(common)
        counts["fsrs_not_in_rwkv"] += len(f["review_th"]) - len(common)
        extra = np.setdiff1d(r_th, common)
        counts["rwkv_not_in_fsrs"] += len(extra)
        counts["rwkv_not_in_fsrs_same_day"] += int((elapsed[extra - 1] == 0).sum())
        y = f["y"][fi_]
        preds = {"fsrs7_default": {p: f["p_fsrs7_default"][fi_] for p in POINTS},
                 "fsrs7_fit": {p: f["p_fsrs7_fit"][fi_] for p in POINTS},
                 "fsrs7_instant": {"sched": f["p_fi_sched"][fi_], "sod": f["p_fi_sod"][fi_], "review": f["p_fi_instant"][fi_]}}
        for k in labels:
            preds[k] = {"sched": d[k]["p_curve"][ri], "sod": d[k]["p_sod"][ri], "review": d[k]["p_instant"][ri]}
            s = rows[k][u]["start_of_day"]
            checks["sod_curve_max_abs_diff"] = max(checks["sod_curve_max_abs_diff"], s.get("curve_max_abs_diff") or 0.0)
            checks["sod_first_of_day_max_abs_diff"] = max(checks["sod_first_of_day_max_abs_diff"], s.get("first_of_day_max_abs_diff", 0.0))
            checks["sod_curve_logloss_max_abs_diff"] = max(checks["sod_curve_logloss_max_abs_diff"], abs(
                s["benchmark"]["ahead"]["LogLoss"] - rows[k][u]["benchmark"]["ahead"]["LogLoss"]))
            if not (s.get("curve_keys_equal") and s.get("instant_keys_equal")):
                fail.append(f"user {u} {k}: start-of-day pass keys differ")
        N.append(len(common))
        fod = first_of_day[common - 1]
        n_split["first"].append(int(fod.sum()))
        n_split["later"].append(int((~fod).sum()))
        for m in models:
            for p in POINTS:
                loss = losses(y, preds[m][p])
                S[m][p].append(loss.sum())
                if m in split:
                    split[m][p]["first"].append(loss[fod].sum())
                    split[m][p]["later"].append(loss[~fod].sum())
    for k, path in (("small", a.existing_small), ("big", a.existing_big)):
        if path:
            old = load_rows(path, labels[k])
            diffs = [abs(rows[k][u]["benchmark"][h]["LogLoss"] - old[u]["benchmark"][h]["LogLoss"])
                     for u in users if u in old for h in ("ahead", "instant")]
            checks["existing_rows_max_abs_diff"][k] = {"users": sum(u in old for u in users), "max": max(diffs) if diffs else None}
    if fail:
        print("[sm81-table] FAIL: " + "; ".join(fail[:20]) + (f" (+{len(fail) - 20} more)" if len(fail) > 20 else ""))
        Path(a.out_json).write_text(json.dumps({"ok": False, "failures": fail}, indent=1))
        sys.exit(1)
    N = np.asarray(N, dtype=np.float64)
    U = len(N)
    S = {m: {p: np.asarray(v) for p, v in d.items()} for m, d in S.items()}
    rng = np.random.default_rng(0)
    idx = rng.integers(0, U, size=(2000, U))
    Nb = N[idx].sum(1)
    ref = S["fsrs7_fit"]["sched"]
    ref_rw, ref_pu = ref.sum() / N.sum(), (ref / N).mean()
    result = {"models": {}, "users": U, "reviews": int(N.sum()), "counts": counts, "checks": checks,
              "missing_users": missing, "bootstrap": {"resamples": 2000, "seed": 0, "unit": "user, paired"}}
    for m in models:
        result["models"][m] = {}
        for p in POINTS:
            s = S[m][p]
            rw, pu = s.sum() / N.sum(), (s / N).mean()
            drw = (s[idx].sum(1) - ref[idx].sum(1)) / Nb
            dpu = ((s - ref) / N)[idx].mean(1)
            result["models"][m][p] = {
                "logloss_review_weighted": rw, "logloss_per_user": pu,
                "pct_vs_fsrs7_fit_sched_review_weighted": 100 * (rw / ref_rw - 1),
                "pct_vs_fsrs7_fit_sched_per_user": 100 * (pu / ref_pu - 1),
                "diff_ci95_review_weighted": [float(np.quantile(drw, 0.025)), float(np.quantile(drw, 0.975))],
                "diff_ci95_per_user": [float(np.quantile(dpu, 0.025)), float(np.quantile(dpu, 0.975))]}
        sc, so, rv = (S[m][p] for p in POINTS)
        if m in ("fsrs7_instant", "small", "big"):
            share = (sc.sum() - so.sum()) / (sc.sum() - rv.sum())
            bs = (sc[idx].sum(1) - so[idx].sum(1)) / (sc[idx].sum(1) - rv[idx].sum(1))
            share_pu = ((sc - so) / N).mean() / ((sc - rv) / N).mean()
            bs_pu = ((sc - so) / N)[idx].mean(1) / ((sc - rv) / N)[idx].mean(1)
            result["models"][m]["share_known_at_start_of_day"] = {
                "review_weighted": share, "ci95_review_weighted": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                "per_user": share_pu, "ci95_per_user": [float(np.quantile(bs_pu, 0.025)), float(np.quantile(bs_pu, 0.975))]}
            parts = {}
            for part in ("first", "later"):
                n = float(sum(n_split[part]))
                ll = {p: float(sum(split[m][p][part]) / n) if n else None for p in POINTS}
                gap = (ll["sched"] - ll["review"]) if n else None
                parts[part] = {"reviews": int(n), "logloss_review_weighted": ll,
                               "share_known_at_start_of_day": (ll["sched"] - ll["sod"]) / gap if n and gap else None}
            result["models"][m]["by_position_in_day"] = parts
    Path(a.out_json).write_text(json.dumps(result, indent=1))
    Path(a.out_md).write_text(render(result), encoding="utf-8")
    print(f"[sm81-table] OK: {U} users, {int(N.sum()):,} common reviews -> {a.out_md}")


NAMES = {"fsrs7_default": "FSRS-7, default parameters (no fit)",
         "fsrs7_fit": "FSRS-7, per-user fit (reference)",
         "fsrs7_instant": "FSRS7-Instant (shared start fit, online updates)",
         "small": "small model (the leak-free small model (SM72s), cool-down from epoch 25)",
         "big": "big model (1 leak-free epoch)"}


def render(r):
    def cell(c, kind):
        ll = c[f"logloss_{kind}"]
        pct = c[f"pct_vs_fsrs7_fit_sched_{kind}"]
        lo, hi = c[f"diff_ci95_{kind}"]
        if abs(pct) < 1e-12:
            return f"{ll:.4f}"
        return f"{ll:.4f} ({pct:+.1f}%, [{lo:+.4f}, {hi:+.4f}])"
    out = ["# Scheduling-time vs review-time accuracy (SM81)", "",
           f"{r['users']} users of the 732 set (the 723 users of the standard read-out + the 3 long ones when present), {r['reviews']:,} common test reviews (every "
           "model scored on exactly the same reviews). LogLoss, lower is better; (% vs FSRS-7 per-user fit at scheduling "
           "time, [paired user-bootstrap 95% CI of the LogLoss difference, 2,000 resamples, seed 0]).", "",
           "| Model | Scheduling time, review-weighted | Scheduling time, per user | Start of day, review-weighted | "
           "Start of day, per user | Review time, review-weighted | Review time, per user |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for m, name in NAMES.items():
        c = r["models"][m]
        out.append(f"| {name} | " + " | ".join(cell(c[p], k) for p in POINTS for k in ("review_weighted", "per_user")) + " |")
    out += ["", "FSRS-7 has no updates: its one prediction is the same at all three times.", ""]
    for m in ("fsrs7_instant", "small", "big"):
        s = r["models"][m]["share_known_at_start_of_day"]
        b = r["models"][m]["by_position_in_day"]
        out.append(f"- {NAMES[m]}: {100 * s['review_weighted']:.0f}% of the review-time gain over its own scheduling-time "
                   f"prediction is already known at the start of the day (review-weighted, CI [{100 * s['ci95_review_weighted'][0]:.0f}%, "
                   f"{100 * s['ci95_review_weighted'][1]:.0f}%]; per user {100 * s['per_user']:.0f}%); on the user's first "
                   f"review of a day {fmt_share(b['first'])}, on the later reviews of the day {fmt_share(b['later'])}.")
    c = r["counts"]
    out += ["", f"Rows: {c['fsrs_test']:,} FSRS-7 test reviews, {c['rwkv_labels']:,} RWKV benchmark reviews, {c['common']:,} in "
            f"both; {c['fsrs_not_in_rwkv']:,} FSRS-7 test reviews outside the RWKV set (its first fold starts later), "
            f"{c['rwkv_not_in_fsrs']:,} RWKV reviews outside the FSRS-7 set ({c['rwkv_not_in_fsrs_same_day']:,} of them same-day "
            "repeats, which the published FSRS-7 protocol does not test)."]
    return "\n".join(out) + "\n"


def fmt_share(part):
    return "n/a" if part["share_known_at_start_of_day"] is None else f"{100 * part['share_known_at_start_of_day']:.0f}% ({part['reviews']:,} reviews)"


if __name__ == "__main__":
    main()
