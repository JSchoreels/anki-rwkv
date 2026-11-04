#!/usr/bin/env python
"""The small model in the review simulator (SM41), stage 2 part 3: the full run's check and report (CPU, seconds).

Design and the primary metric (fixed before launch): artifacts/small-model/sm41/DESIGN.md section 10.

  check    rows / splits / logs of artifacts/small-model/sm41/fullrun (finished splits, missing splits, Tracebacks)
  report   fullrun/fullrun-tables.md + fullrun/fullrun-result.json
"""
from argparse import ArgumentParser
import glob
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1] / "artifacts" / "small-model" / "sm41"
OUT = HERE / "fullrun"
LEVELS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
MATCH = [0.75, 0.80, 0.85, 0.90]          # secondary matched realised-retention points
MIN_OVERLAP = 0.02                        # primary: common realised-retention range at least this wide
FAM = {"ihr": "instant-head desired retention", "fsrs7": "FSRS-7 desired retention"}
PRICES = [1.0, 3.0, 10.0]                 # minutes (days 30-59) per extra memorised card
T975 = {k: v for k, v in zip(range(1, 31), [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
                                             2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
                                             2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042])}


def t975(n):
    return T975.get(n - 1, 1.96 + 2.4 / max(n - 1, 1)) if n > 1 else float("nan")


def ci(v):
    v = np.asarray([x for x in v if np.isfinite(x)], np.float64)
    if len(v) < 2:
        return float(v.mean()) if len(v) else float("nan"), float("nan"), len(v)
    return float(v.mean()), float(t975(len(v)) * v.std(ddof=1) / math.sqrt(len(v))), len(v)


def load_rows(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        rows += [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    return rows


def lane_block(r, k, lo_b, hi_b):
    """Counters of lane k over buckets [lo_b, hi_b) (scripts/sm41_engine.py lane_summary, without the torch import)."""
    A = {n: np.asarray(v)[k, lo_b:hi_b].sum(0) for n, v in r["acc"].items()}
    lat = A["later_old"] + A["later_new"]
    load = A["load"]
    gl, tl = A["grades_later"], A["time_later"]
    div = lambda a, b: float(a / b) if b else float("nan")   # noqa: E731
    return {"ret": div(lat[1], lat[0]), "pred": div(lat[2], lat[0]), "aim": div(A["aim"][1], A["aim"][0]),
            "minutes": load[3:].sum() / 60, "reviews": load[:3].sum(), "easy": div(gl[3], gl[1:].sum()),
            "s_later": div(tl[1], tl[0]), "ft_pred": div(A["first_touch"][2], A["first_touch"][0])}


def split_policies(r):
    """Per split: lane base name -> seed-averaged metrics (retention, minutes and memorised averaged over seeds,
    then the ratios)."""
    nb = len(r["acc"]["load"][0])
    burn = 30 // r["bucket_days"]
    groups = {}
    for k, name in enumerate(r["lanes"]):
        groups.setdefault(name.split("#")[0], []).append(k)
    tf = lane_block(r, groups["tf"][0], burn, nb)
    out = {}
    for base, ks in groups.items():
        late = [lane_block(r, k, burn, nb) for k in ks]
        whole = [lane_block(r, k, 0, nb) for k in ks]
        m = lambda xs, key: float(np.nanmean([x[key] for x in xs]))   # noqa: E731
        mem = float(np.mean([r["memorised_instant"][k] for k in ks]))
        out[base] = {"ret": m(late, "ret"), "aim": m(late, "aim"), "pred": m(late, "pred"),
                     "min_30_59": m(late, "minutes"), "min_all": m(whole, "minutes"), "memorised": mem,
                     "cost": m(late, "minutes") / max(mem, 1e-9), "cost_all": m(whole, "minutes") / max(mem, 1e-9),
                     "easy_drift": m(late, "easy") - tf["easy"], "s_drift": m(late, "s_later") - tf["s_later"],
                     "ft_state_drift": m(late, "ft_pred") - tf["ft_pred"],
                     "backlog": float(np.mean([r["backlog"][k] for k in ks]))}
    return out


def interp_cost(pts, r_star, key="cost"):
    """Cost at realised retention r_star, linear along the policy's points sorted by realised retention; NaN when
    r_star is outside the realised range (no extrapolation)."""
    xs = np.array([p["ret"] for p in pts]); ys = np.array([p[key] for p in pts])
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    if len(xs) < 2:
        return float("nan")
    o = np.argsort(xs); xs, ys = xs[o], ys[o]
    if not xs[0] <= r_star <= xs[-1]:
        return float("nan")
    return float(np.interp(r_star, xs, ys))


def check(args):
    users = json.loads((OUT / "users.json").read_text())["users"]
    want = {(u["user"], s["split_day"]) for u in users for s in u["splits"]}
    rows = load_rows(str(OUT / "rows" / "*.jsonl"))
    got = {(r["user"], r["split_day"]) for r in rows}
    dup = len(rows) - len(got)
    tb = [p for p in glob.glob(str(OUT / "logs" / "*.log")) if "Traceback" in Path(p).read_text(errors="replace")]
    fits = load_rows(str(OUT / "fits" / "*.jsonl"))
    lanes = {len(r["lanes"]) for r in rows}
    print(f"splits planned {len(want)}, finished {len(got & want)}, missing {len(want - got)}, duplicates {dup}, "
          f"lanes per row {sorted(lanes)}, fits {len(fits)}, logs with Traceback {len(tb)}")
    if want - got:
        print("missing:", sorted(want - got)[:40])
    for p in tb:
        print("Traceback in", p)
    src = {r["fsrs7"].split(" (")[0] for r in rows}
    print("FSRS-7 sources in rows:", src)


def report(args):
    rows = load_rows(args.rows)
    if not rows:
        raise SystemExit("no rows")
    rows.sort(key=lambda r: (r["user"], r["split_day"]))
    fits = {(f["user"], f["split_day"]): f for f in load_rows(str(OUT / "fits" / "*.jsonl"))}
    per = {}            # user -> list of split dicts (base -> metrics)
    for r in rows:
        per.setdefault(r["user"], []).append(split_policies(r))
    users = sorted(per)
    L = ["# The small model in the review simulator (SM41): full-run tables", ""]
    nsplit = len(rows)
    L += [f"{len(users)} users of the 732 set, {nsplit} split points, {rows[0]['days'] if rows else 0} days, "
          f"{len(rows[0]['lanes']) if rows else 0} lanes per split. Metrics on days 30-59 (after the 30-day burn-in "
          "of the switch), 4 seeds averaged. Cost = minutes of every review on days 30-59 / memorised cards at day 60 "
          "(the instant head's recall summed over the card set). Means over users (a user's splits averaged first), "
          "95% CI over users (t). FSRS-7 = pre-split refit per (user, split).", ""]
    out = {"users": len(users), "splits": nsplit}

    # ---- policy table
    bases = ["tf", "fr-shared1"] + [f"{p}-{r:.2f}" for p in ("ihr", "fsrs7") for r in LEVELS] + \
        ["ihr-0.90-free", "fsrs7-0.90-free"]

    def label(b):
        if b == "tf":
            return "real schedule, teacher forcing (tf)"
        if b == "fr-shared1":
            return "real schedule, sampled, daily shared-stream re-anchoring (fr-shared1)"
        kind, _, rest = b.partition("-")
        name = {"ihr": "instant-head desired retention", "fsrs7": "FSRS-7 desired retention"}[kind]
        r_, _, free = rest.partition("-")
        return f"{name} {r_}{' without re-anchoring' if free else ''} ({b})"

    def user_mean(b, key):
        return [float(np.nanmean([s[b][key] for s in per[u] if b in s])) for u in users if any(b in s for s in per[u])]

    L += ["## Policies (mean over users)", "", "| policy | retention 30-59 | aim | min/day 30-59 | memorised | "
          "cost: min per memorised card | cost, 60 days | Easy drift, points | seconds drift | backlog |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    out["policies"] = {}
    for b in bases:
        if not any(b in s for u in users for s in per[u]):
            continue
        v = {k: float(np.mean(user_mean(b, k))) for k in ("ret", "aim", "min_30_59", "memorised", "cost", "cost_all",
                                                            "easy_drift", "s_drift", "backlog")}
        out["policies"][b] = v
        f = lambda x, d=3: "—" if not np.isfinite(x) else f"{x:.{d}f}"   # noqa: E731
        L.append(f"| {label(b)} | {f(v['ret'])} | {f(v['aim'])} | {f(v['min_30_59'] / 30, 1)} | "
                 f"{f(v['memorised'], 0)} | {f(v['cost'], 3)} | {f(v['cost_all'], 3)} | {f(100 * v['easy_drift'], 1)} | "
                 f"{f(v['s_drift'], 1)} | {f(v['backlog'], 0)} |")

    # ---- primary: matched realised retention, paired per user
    def curves(s):
        return ([s[f"ihr-{r:.2f}"] for r in LEVELS if f"ihr-{r:.2f}" in s],
                [s[f"fsrs7-{r:.2f}"] for r in LEVELS if f"fsrs7-{r:.2f}" in s])

    def overlap_gap(ih, fs, key):
        """Mean relative gap over the common realised-retention range of the two curves (grid step 0.005)."""
        rng = lambda pts: (min(p["ret"] for p in pts), max(p["ret"] for p in pts))   # noqa: E731
        lo = max(rng(ih)[0], rng(fs)[0]); hi = min(rng(ih)[1], rng(fs)[1])
        if not hi - lo >= MIN_OVERLAP:
            return float("nan"), 0.0
        grid = np.linspace(lo, hi, max(2, int(round((hi - lo) / 0.005)) + 1))
        return float(np.mean([interp_cost(ih, x, key) / interp_cost(fs, x, key) - 1 for x in grid])), hi - lo

    L += ["", "## Primary: instant head vs FSRS-7 at matched realised retention (paired per user)", "",
          "Relative gap = cost of instant-head desired retention / cost of FSRS-7 desired retention - 1, both read off "
          "each policy's 6-point curve (realised retention on days 30-59 -> cost, linear between the points sorted by "
          "realised retention, no extrapolation) at the same realised retention, per split. PRIMARY (fixed before "
          f"launch): the mean gap over the common realised-retention range of the two curves (grid step 0.005; splits "
          f"whose ranges overlap by less than {MIN_OVERLAP:.2f} are left out), a user's splits averaged, mean over "
          "users with a 95% CI over users. The fixed points 0.75-0.90 are secondary (users whose both curves cover the "
          "point). Negative = the instant head needs fewer minutes per memorised card. Decision rule (fixed before "
          "launch): the instant head moves on to the time-aware objective only if the primary gap is < 0 with the 95% "
          "CI excluding 0; otherwise FSRS-7 stays the scheduler baseline.", "",
          "| realised retention | users | relative gap, % (95% CI) | median, % | users where the instant head is cheaper "
          "| gap, 60-day cost, % (95% CI) |", "|---|---|---|---|---|---|"]
    out["matched"] = {}
    for rs in ["overlap"] + MATCH:
        g, g_all, width = [], [], []
        for u in users:
            vals, vals_all = [], []
            for s in per[u]:
                ih, fs = curves(s)
                ih = [x for x in ih if np.isfinite(x["ret"])]; fs = [x for x in fs if np.isfinite(x["ret"])]
                if len(ih) < 2 or len(fs) < 2:
                    continue
                if rs == "overlap":
                    a, w = overlap_gap(ih, fs, "cost"); a_all, _ = overlap_gap(ih, fs, "cost_all")
                    if np.isfinite(a):
                        vals.append(a); vals_all.append(a_all); width.append(w)
                else:
                    a, b = interp_cost(ih, rs), interp_cost(fs, rs)
                    if np.isfinite(a) and np.isfinite(b):
                        vals.append(a / b - 1)
                        vals_all.append(interp_cost(ih, rs, "cost_all") / interp_cost(fs, rs, "cost_all") - 1)
            if vals:
                g.append(float(np.mean(vals))); g_all.append(float(np.mean(vals_all)))
        m, h, n = ci(g)
        ma, ha, _ = ci(g_all)
        key = rs if rs == "overlap" else f"{rs:.2f}"
        out["matched"][key] = {"mean": m, "ci95": h, "users": n, "median": float(np.median(g)) if g else None,
                               "cheaper_share": float(np.mean(np.array(g) < 0)) if g else None,
                               "mean_60day": ma, "ci95_60day": ha}
        if rs == "overlap":
            out["matched"][key]["mean_overlap_width"] = float(np.mean(width)) if width else None
        name = (f"common range (PRIMARY; mean width {np.mean(width) if width else float('nan'):.3f})"
                if rs == "overlap" else f"{rs:.2f}")
        L.append(f"| {name} | {n} | {100 * m:+.1f} ± {100 * h:.1f} | "
                 f"{100 * np.median(g) if g else float('nan'):+.1f} | {100 * np.mean(np.array(g) < 0) if g else 0:.0f}% | "
                 f"{100 * ma:+.1f} ± {100 * ha:.1f} |")
    pm = out["matched"].get("overlap", {})
    if pm and pm["users"]:
        verdict = ("instant head cheaper (CI excludes 0)" if pm["mean"] + pm["ci95"] < 0 else
                   "FSRS-7 cheaper (CI excludes 0)" if pm["mean"] - pm["ci95"] > 0 else "no significant difference")
        out["verdict"] = verdict
        L += ["", f"Primary verdict: **{verdict}** ({100 * pm['mean']:+.1f} ± {100 * pm['ci95']:.1f}% over "
              f"{pm['users']} users)."]

    # ---- per-user best desired retention at a stated price
    L += ["", "## Each user's best desired retention at a stated price", "",
          "Value(R) = memorised cards at day 60 - minutes on days 30-59 / P, P = the price in minutes per extra "
          "memorised card; a user's best R = argmax over the 6 targets (splits averaged). 'Fixed' = the single R with "
          "the best mean value over users. Gain = the mean over users of value(own best) - value(fixed), in memorised "
          "cards and in % of memorised.", "",
          "| policy | price P | best R per user: 0.70 / 0.75 / 0.80 / 0.85 / 0.90 / 0.95 | fixed R | gain of a per-user "
          "target, cards (95% CI) | gain, % of memorised |", "|---|---|---|---|---|---|"]
    out["best_r"] = {}
    for fam in ("ihr", "fsrs7"):
        for P in PRICES:
            V, M = [], []
            for u in users:
                row, mem = [], []
                for r in LEVELS:
                    b = f"{fam}-{r:.2f}"
                    xs = [s[b] for s in per[u] if b in s]
                    row.append(float(np.mean([x["memorised"] - x["min_30_59"] / P for x in xs])) if xs else float("nan"))
                    mem.append(float(np.mean([x["memorised"] for x in xs])) if xs else float("nan"))
                V.append(row); M.append(mem)
            V, M = np.array(V), np.array(M)
            if not len(V) or np.isnan(V).all():
                continue
            best = np.nanargmax(V, 1)
            fixed = int(np.nanargmax(np.nanmean(V, 0)))
            gain = V[np.arange(len(V)), best] - V[:, fixed]
            gm, gh, _ = ci(gain)
            pct = float(np.mean(gain / M[:, fixed])) * 100
            counts = [int((best == k).sum()) for k in range(len(LEVELS))]
            out["best_r"][f"{fam}@{P:g}"] = {"counts": counts, "fixed": LEVELS[fixed], "gain_cards": gm,
                                             "gain_ci95": gh, "gain_pct": pct}
            L.append(f"| {FAM[fam]} | {P:g} | {' / '.join(map(str, counts))} | "
                     f"{LEVELS[fixed]:.2f} | {gm:+.1f} ± {gh:.1f} | {pct:+.2f}% |")

    # ---- re-anchoring
    L += ["", "## Does re-anchoring change the ranking? (target 0.90, paired per user)", "",
          "Relative cost gap instant head vs FSRS-7 at the same TARGET 0.90 (not matched retention), with and without "
          "daily shared-stream re-anchoring; difference = free - anchored.", "",
          "| lanes | users | retention: instant head / FSRS-7 | cost gap, % (95% CI) |", "|---|---|---|---|"]
    out["reanchor"] = {}
    diffs = {}
    for tag, a, b in (("anchored", "ihr-0.90", "fsrs7-0.90"), ("free", "ihr-0.90-free", "fsrs7-0.90-free")):
        g, ra, rb = [], [], []
        for u in users:
            xs = [(s[a], s[b]) for s in per[u] if a in s and b in s]
            if xs:
                g.append(float(np.mean([x["cost"] / y["cost"] - 1 for x, y in xs])))
                ra.append(float(np.mean([x["ret"] for x, _ in xs]))); rb.append(float(np.mean([y["ret"] for _, y in xs])))
        diffs[tag] = dict(zip([u for u in users if any(a in s and b in s for s in per[u])], g))
        m, h, n = ci(g)
        out["reanchor"][tag] = {"mean": m, "ci95": h, "users": n}
        if n:
            L.append(f"| {tag} | {n} | {np.mean(ra):.3f} / {np.mean(rb):.3f} | {100 * m:+.1f} ± {100 * h:.1f} |")
    common = sorted(set(diffs.get("anchored", {})) & set(diffs.get("free", {})))
    if common:
        m, h, n = ci([diffs["free"][u] - diffs["anchored"][u] for u in common])
        flips = sum(np.sign(diffs["free"][u]) != np.sign(diffs["anchored"][u]) for u in common)
        out["reanchor"]["free_minus_anchored"] = {"mean": m, "ci95": h, "users": n, "sign_flips": int(flips)}
        L.append(f"| free - anchored | {n} | | {100 * m:+.1f} ± {100 * h:.1f} (sign flips in {flips} users) |")
    for b in ("ihr-0.90-free", "fsrs7-0.90-free"):
        if b in out["policies"]:
            v = out["policies"][b]
            L.append(f"\nDrift of {label(b)} vs teacher forcing on days 30-59: Easy share {100 * v['easy_drift']:+.1f} "
                     f"points, seconds per later review {v['s_drift']:+.1f}.")

    # ---- drift check on every split
    m, h, n = ci([float(np.nanmean([s["fr-shared1"]["ft_state_drift"] for s in per[u]])) for u in users])
    out["drift_fr_shared1"] = {"mean": m, "ci95": h, "users": n}
    L += ["", f"Drift check: first-touch state drift of daily shared-stream re-anchoring (fr-shared1) on the real "
          f"schedule, days 30-59: {100 * m:+.1f} ± {100 * h:.1f} points over {n} users (drift study: -0.6 ± 1.0)."]

    # ---- FSRS-7 fits and speed
    seen = [f["benchmark_last_fold_train_max_day"] >= f["split_day"] for f in fits.values()]
    lr = sum(r["lane_reviews"] for r in rows); wall = sum(r["wall_seconds"] for r in rows)
    wu = sum(r.get("warmup_seconds", 0) for r in rows)
    out["fsrs7_fits"] = {"fits": len(fits), "benchmark_fit_saw_split_day": int(sum(seen))}
    out["speed"] = {"lane_reviews": lr, "wall_seconds": wall, "warmup_seconds": wu,
                    "lane_reviews_per_second": lr / max(wall, 1)}
    L += ["", f"FSRS-7: {len(fits)} pre-split refits; the srs-benchmark fit had seen the split day in {sum(seen)} of "
          f"them.", "",
          f"Speed: {lr:,} lane reviews in {wall / 3600:.2f} process-hours ({lr / max(wall, 1):,.0f} lane reviews/s per "
          f"process, 1 thread), warm-ups {wu / 3600:.2f} process-hours; {(wall + wu) / 3600:.2f} CPU process-hours "
          "in all.", ""]
    Path(args.md).write_text("\n".join(L) + "\n", encoding="utf8")
    Path(args.json).write_text(json.dumps(out, indent=1))
    print("\n".join(L))


def main():
    ap = ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    t = sub.add_parser("report"); t.add_argument("--rows", default=str(OUT / "rows" / "*.jsonl"))
    t.add_argument("--md", default=str(OUT / "fullrun-tables.md"))
    t.add_argument("--json", default=str(OUT / "fullrun-result.json"))
    a = ap.parse_args()
    {"check": check, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
