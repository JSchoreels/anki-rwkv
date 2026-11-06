#!/usr/bin/env python
"""The small model in the review simulator (SM41): the changing-learner simulation (SM41g) report (CPU, numpy only).

Plan, metrics and decision rule (written before the runs): artifacts/small-model/sm41/learner-drift-result.md.

  check    rows per scenario, Tracebacks, and the no-drift rerun (`none`) against the existing rows (full run +
           screen 3): every counter and the memorised sums, largest relative gap
  report   per scenario (improve, decline; none as a check): gaps vs FSRS-7 (matched realised retention, matched
           memorised cards days 30-59 and 60 days) for the instant head and the time-aware rule (beta = 2), the
           change vs no drift (paired per user), reviews/day and median later-day interval at days 30-59, the
           `-free` lanes at 0.90 -> learner-drift/tables.md + .json
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
from scripts import sm41_fullrun as FR     # noqa: E402
from scripts import sm41_policy as P       # noqa: E402
from scripts import sm41_frontier as FT    # noqa: E402

HERE = ROOT / "artifacts" / "small-model" / "sm41"
OUT = HERE / "learner-drift"
ALL = False        # --dir learner-drift-all: one fused row per split (tf + 3 scenarios as lanes), 117 users, 304 splits
GTAG = {"improve": "-g0.5_0.2", "decline": "-gm0.5_m0.2"}      # lane name suffix of a fused scenario
SCEN = ("improve", "decline")
R3 = [0.80, 0.85, 0.90]
FAMS = {"ihr": ("instant head", lambda R: f"ihr-{R:.2f}"), "s2": ("time-aware beta = 2", lambda R: f"ihr-{R:.2f}-s2")}


def rows_by_key(pattern):
    out = {}
    for r in FR.load_rows(pattern):
        out.setdefault((r["user"], r["split_day"]), r)    # screen 3 ran twice on some users: the first row
    return out


def scen_view(r, sc):
    """The rows of one scenario out of a fused row (tf + the scenario's lanes, names without the drift tag)."""
    nl = len(r["lanes"])
    ks, names = [], []
    for k, n in enumerate(r["lanes"]):
        b, _, sd = n.partition("#")
        tagged = [t for t in GTAG.values() if b.endswith(t)]
        if b == "tf" or (sc == "none" and not tagged) or (sc in GTAG and b.endswith(GTAG[sc])):
            ks.append(k)
            names.append((b[:-len(GTAG[sc])] if sc in GTAG and b != "tf" else b) + "#" + sd)
    v = dict(r)
    v["lanes"] = names
    v["acc"] = {n: np.asarray(a)[ks].tolist() for n, a in r["acc"].items()}
    for key, val in r.items():
        if isinstance(val, list) and len(val) == nl and key != "lanes":
            v[key] = [val[k] for k in ks]
    return v


def scen_rows(sc):
    """(user, split) -> row of scenario sc ('improve' / 'decline' / 'none')."""
    if not ALL:
        return rows_by_key(str(OUT / sc / "rows" / "*.jsonl"))
    return {k: scen_view(r, sc) for k, r in rows_by_key(str(OUT / "rows" / "*.jsonl")).items()}


def U(ks, v):
    """Per-user mean of per-split values (users with several splits count once; NaN-free means)."""
    g = {}
    for k, x in zip(ks, v):
        if np.isfinite(x):
            g.setdefault(k[0], []).append(x)
    return [float(np.mean(x)) for x in g.values()]


def baseline():
    """(user, split) -> merged no-drift lanes: the full run (instant head, FSRS-7, -free) + screen 3 (beta = 2)."""
    full = rows_by_key(str(HERE / "fullrun" / "rows" / "*.jsonl"))
    s3 = rows_by_key(str(HERE / "policy" / "screen3" / "rows" / "*.jsonl"))
    return full, s3


def lane_extra(r, base, lo_b=2, hi_b=4):
    """Seed-averaged reviews/day and seconds per later-day review at days 30-59, and the median elapsed days of
    later-day reviews (seeds pooled; NaN without the histogram) for lane base `base` of row r."""
    ks = [k for k, n in enumerate(r["lanes"]) if n.split("#")[0] == base]
    if not ks:
        return None
    blocks = [FR.lane_block(r, k, lo_b, hi_b) for k in ks]
    days = (hi_b - lo_b) * r["bucket_days"]
    out = {"rev_day": float(np.mean([b["reviews"] for b in blocks])) / days,
           "min_day": float(np.mean([b["minutes"] for b in blocks])) / days,
           "s_later": float(np.nanmean([b["s_later"] for b in blocks])), "ivl": float("nan")}
    if "ivl_later" in r["acc"]:
        h = np.asarray(r["acc"]["ivl_later"])[ks, lo_b:hi_b].sum((0, 1))
        out["ivl"] = hist_median(h, np.asarray(r["ivl_edges"], np.float64))
    return out


def hist_median(h, edges):
    tot = h.sum()
    if tot <= 0:
        return float("nan")
    c = np.cumsum(h)
    j = int(np.searchsorted(c, tot / 2))
    lo = edges[j]
    hi = edges[j + 1] if j + 1 < len(edges) else edges[j] * 1.1
    prev = c[j - 1] if j else 0.0
    f = (tot / 2 - prev) / max(h[j], 1e-9)
    return float(lo + f * (hi - lo)) if hi - lo <= 1 else float(math.exp(math.log(lo) + f * (math.log(hi) - math.log(lo))))


def gaps(sp_pol, sp_fs, fam):
    """(plan gap, memorised gap days 30-59, memorised gap 60 days) for family fam of split policies sp_pol vs FSRS-7
    of sp_fs (usually the same row)."""
    a = [sp_pol[FAMS[fam][1](R)] for R in R3]
    b = [sp_fs[f"fsrs7-{v:.2f}"] for v in FR.LEVELS]
    return P.overlap_gap(a, b), FT.gap(a, b, "min_30_59"), FT.gap(a, b, "min_all")


def fmt(v, d=1, scale=100):
    m, h, n = FR.ci(v)
    if not np.isfinite(m):
        return "—"
    return f"{scale * m:+.{d}f} ± {scale * h:.{d}f}"


def check(args):
    full, s3 = baseline()
    L = []
    if ALL:
        files = sorted(glob.glob(str(OUT / "rows" / "*.jsonl")))
        rows = FR.load_rows(str(OUT / "rows" / "*.jsonl"))
        tb = [p for p in glob.glob(str(OUT / "logs" / "*.log")) if "Traceback" in Path(p).read_text()]
        jf = OUT / "jobs.txt"
        jobs = [l.split() for l in jf.read_text().splitlines() if l.strip()] if jf.exists() else []
        want = sum(len(j[1].split(",")) for j in jobs)
        L.append(f"fused: {len(files)} of {len(jobs)} row files, {len(rows)} of {want} split rows, Tracebacks in "
                 f"{len(tb)} logs {tb}")
    for sc in (() if ALL else SCEN + ("none",)):
        files = sorted(glob.glob(str(OUT / sc / "rows" / "*.jsonl")))
        rows = FR.load_rows(str(OUT / sc / "rows" / "*.jsonl"))
        tb = [p for p in glob.glob(str(OUT / sc / "logs" / "*.log")) if "Traceback" in Path(p).read_text()]
        L.append(f"{sc}: {len(files)} row files, {len(rows)} rows, Tracebacks in {len(tb)} logs {tb}")
    worst, worst_at = 0.0, ""
    n = 0
    for (u, D), r in scen_rows("none").items():
        for k, name in enumerate(r["lanes"]):
            ref = full.get((u, D)) if name in full.get((u, D), {"lanes": []})["lanes"] else None
            if ref is None and (u, D) in s3 and name in s3[(u, D)]["lanes"]:
                ref = s3[(u, D)]
            if ref is None:
                continue
            j = ref["lanes"].index(name)
            n += 1
            for key in ref["acc"]:
                a, b = np.asarray(r["acc"][key])[k], np.asarray(ref["acc"][key])[j]
                g = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1.0)))
                if g > worst:
                    worst, worst_at = g, f"user {u} lane {name} {key}"
            for key in ("memorised_instant", "backlog"):
                a, b = r[key][k], ref[key][j]
                g = abs(a - b) / max(abs(b), 1.0)
                if g > worst:
                    worst, worst_at = g, f"user {u} lane {name} {key}"
    L.append(f"no drift lanes vs existing rows: {n} lanes compared, largest relative gap {worst:.2e} {worst_at}")
    txt = "\n".join(L)
    print(txt)
    (OUT / "check.out").write_text(txt + "\n", encoding="utf8")


def report(args):
    full, s3 = baseline()
    none = scen_rows("none")
    if ALL:
        keys = sorted(none)
    else:
        sample = json.loads(P.SAMPLE.read_text())["users"]
        keys = [(u["user"], u["split_day"]) for u in sample]

    def base_sp(key):
        if ALL:      # the fused run's own no-drift lanes (checked against the existing rows by `check`)
            return FR.split_policies(none[key])
        sp = FR.split_policies(full[key])
        sp.update({k: v for k, v in FR.split_policies(s3[key]).items() if k != "tf"})
        return sp

    def base_extra(key, b):
        if ALL:
            return lane_extra(none[key], b)
        r = full[key] if b in {n.split("#")[0] for n in full[key]["lanes"]} else s3[key]
        return lane_extra(r, b)

    res = {"users": len(keys), "scenarios": {}}
    L = ["# The changing-learner simulation (SM41g): tables", "",
         "Users of the run (splits of a user averaged first), 60 days, seeds 0-3. Gaps: cost of the policy / FSRS-7 - 1, paired per "
         "user, mean ± 95% CI over users (%). Change vs no drift = the scenario's gap minus the no-drift gap on the "
         "same user (difference in differences, points). No drift = the full run + screen 3 rows.", ""]
    scen = [s for s in SCEN + (() if ALL else ("none",)) if scen_rows(s)]
    sp0 = {k: base_sp(k) for k in keys}
    g0 = {fam: {k: gaps(sp0[k], sp0[k], fam) for k in keys} for fam in FAMS}
    nu = len({k[0] for k in keys})
    L += ["## Gaps vs FSRS-7", "",
          "| scenario | policy | users | matched retention (plan) | change vs no drift | matched memorised, days 30-59 | "
          "change | matched memorised, 60 days | change |", "|---|---|---|---|---|---|---|---|---|"]
    for fam, (nm, _) in FAMS.items():
        cols = [U(keys, [g0[fam][k][i] for k in keys]) for i in range(3)]
        L.append(f"| no drift | {nm} | {nu} | {fmt(cols[0])} | — | {fmt(cols[1])} | — | {fmt(cols[2])} | — |")
        res.setdefault("no_drift", {})[fam] = [FR.ci(c) for c in cols]
    for sc in scen:
        rows = scen_rows(sc)
        ks = [k for k in keys if k in rows]
        sp = {k: FR.split_policies(rows[k]) for k in ks}
        out = res["scenarios"].setdefault(sc, {"users": len(ks)})
        for fam, (nm, _) in FAMS.items():
            g = {k: gaps(sp[k], sp[k], fam) for k in ks}
            cells = []
            for i in range(3):
                v = U(ks, [g[k][i] for k in ks])
                dd = U(ks, [g[k][i] - g0[fam][k][i] for k in ks])
                cells += [fmt(v), fmt(dd)]
                out.setdefault(fam, {})[["plan", "mem_30_59", "mem_60"][i]] = {"gap": FR.ci(v), "change": FR.ci(dd)}
            L.append(f"| {sc} | {nm} | {len({k[0] for k in ks})} | " + " | ".join(cells) + " |")
    # schedules less: reviews/day and median interval vs FSRS-7 at the same target, change vs no drift
    L += ["", "## Does it schedule less? (days 30-59)", "",
          "Per user and target (0.80 / 0.85 / 0.90): ln(scenario / no drift) of the policy minus the same for FSRS-7 "
          "at the same target, averaged over the 3 targets, then mean ± 95% CI over users (in %, as 100 x the log "
          "difference). Negative = the policy's reviews/day fell more (or rose less) than FSRS-7's. Intervals: the "
          "no-drift histogram comes from the `none` rerun.", "",
          "| scenario | policy | reviews/day vs FSRS-7, change | median later-day interval vs FSRS-7, change | "
          "seconds per later-day review vs FSRS-7, change |", "|---|---|---|---|---|"]
    for sc in [s for s in scen if s != "none"]:
        rows = scen_rows(sc)
        ks = [k for k in keys if k in rows]
        for fam, (nm, f) in FAMS.items():
            vals = {"rev_day": [], "ivl": [], "s_later": []}
            vk = {q: [] for q in vals}
            for k in ks:
                per = {q: [] for q in vals}
                for R in R3:
                    a1, f1 = lane_extra(rows[k], f(R)), lane_extra(rows[k], f"fsrs7-{R:.2f}")
                    a0, f0 = base_extra(k, f(R)), base_extra(k, f"fsrs7-{R:.2f}")
                    i0 = (lane_extra(none[k], f(R)), lane_extra(none[k], f"fsrs7-{R:.2f}")) if k in none else None
                    for q in vals:
                        if q == "ivl":
                            if i0 is None:
                                continue
                            x = math.log(a1[q] / i0[0][q]) - math.log(f1[q] / i0[1][q])
                        else:
                            x = math.log(a1[q] / a0[q]) - math.log(f1[q] / f0[q])
                        if np.isfinite(x):
                            per[q].append(x)
                for q in vals:
                    if per[q]:
                        vals[q].append(float(np.mean(per[q]))); vk[q].append(k)
            vals = {q: U(vk[q], v) for q, v in vals.items()}
            L.append(f"| {sc} | {nm} | {fmt(vals['rev_day'])} | {fmt(vals['ivl'])} | {fmt(vals['s_later'])} |")
            res["scenarios"][sc].setdefault(fam, {})["schedules_less"] = {q: FR.ci(v) for q, v in vals.items()}
    # per-lane table
    L += ["", "## Per policy and target (days 30-59, mean over users)", "",
          "| scenario | lane | realised retention | reviews/day | min/day | s per later-day review | median later-day "
          "interval (days) | memorised at day 60 |", "|---|---|---|---|---|---|---|---|"]
    lanes = [f"ihr-{R:.2f}" for R in R3] + [f"ihr-{R:.2f}-s2" for R in R3] + [f"fsrs7-{v:.2f}" for v in FR.LEVELS] + \
        ([] if ALL else ["ihr-0.90-free", "fsrs7-0.90-free"])
    for sc in ["no drift"] + [s for s in scen if s != "none"]:
        rows = None if sc == "no drift" else scen_rows(sc)
        ks = keys if rows is None else [k for k in keys if k in rows]
        for b in lanes:
            ex, sp_ = [], []
            for k in ks:
                if rows is None:
                    e = base_extra(k, b)
                    if k in none:
                        e = dict(e, ivl=lane_extra(none[k], b)["ivl"])
                    ex.append(e); sp_.append(sp0[k][b])
                else:
                    ex.append(lane_extra(rows[k], b)); sp_.append(FR.split_policies(rows[k])[b])
            mm = lambda xs, q: float(np.nanmean([x[q] for x in xs]))   # noqa: E731
            L.append(f"| {sc} | {b} | {mm(sp_, 'ret'):.3f} | {mm(ex, 'rev_day'):.1f} | {mm(ex, 'min_day'):.1f} | "
                     f"{mm(ex, 's_later'):.1f} | {mm(ex, 'ivl'):.1f} | {mm(sp_, 'memorised'):.0f} |")
    # re-anchoring: cost gap at 0.90 anchored vs free (the 20-user run only: the confirmation has no -free lanes)
    if not ALL:
        L += ["", "## Re-anchoring check: cost gap instant head / FSRS-7 at target 0.90 (days 30-59 minutes per memorised "
              "card)", "", "| scenario | anchored | change vs no drift | `-free` | change vs no drift |", "|---|---|---|---|---|"]
        c0 = {k: (sp0[k]["ihr-0.90"]["cost"] / sp0[k]["fsrs7-0.90"]["cost"] - 1,
                  sp0[k]["ihr-0.90-free"]["cost"] / sp0[k]["fsrs7-0.90-free"]["cost"] - 1) for k in keys}
        L.append(f"| no drift | {fmt([c0[k][0] for k in keys])} | — | {fmt([c0[k][1] for k in keys])} | — |")
        for sc in [s for s in scen if s != "none"]:
            rows = rows_by_key(str(OUT / sc / "rows" / "*.jsonl"))
            ks = [k for k in keys if k in rows]
            sp = {k: FR.split_policies(rows[k]) for k in ks}
            c = {k: (sp[k]["ihr-0.90"]["cost"] / sp[k]["fsrs7-0.90"]["cost"] - 1,
                     sp[k]["ihr-0.90-free"]["cost"] / sp[k]["fsrs7-0.90-free"]["cost"] - 1) for k in ks}
            L.append(f"| {sc} | {fmt([c[k][0] for k in ks])} | {fmt([c[k][0] - c0[k][0] for k in ks])} | "
                     f"{fmt([c[k][1] for k in ks])} | {fmt([c[k][1] - c0[k][1] for k in ks])} |")
            res["scenarios"][sc]["anchor_check"] = {"anchored": FR.ci([c[k][0] for k in ks]),
                                                    "anchored_change": FR.ci([c[k][0] - c0[k][0] for k in ks]),
                                                    "free": FR.ci([c[k][1] for k in ks]),
                                                    "free_change": FR.ci([c[k][1] - c0[k][1] for k in ks])}
    # speed
    L += ["", "## Speed (CPU, 1 thread per process)", "", "| scenario | lane reviews | engine process-hours | "
          "lane reviews/s per process | warm-up process-hours |", "|---|---|---|---|---|"]
    for sc in (["fused (all 3 scenarios, one run)"] if ALL else scen):
        rows = list(rows_by_key(str(OUT / ("rows" if ALL else sc + "/rows") / "*.jsonl")).values())
        lr = sum(r["lane_reviews"] for r in rows); w = sum(r["wall_seconds"] for r in rows)
        wu = sum(r.get("warmup_seconds", 0) for r in rows)
        L.append(f"| {sc} | {lr:,} | {w / 3600:.2f} | {lr / max(w, 1e-9):.0f} | {wu / 3600:.2f} |")
        res["scenarios"].setdefault(sc, {})["speed"] = {"lane_reviews": lr, "engine_s": w, "warmup_s": wu}
    txt = "\n".join(L) + "\n"
    (OUT / "tables.md").write_text(txt, encoding="utf8")
    (OUT / "tables.json").write_text(json.dumps(res, indent=1), encoding="utf8")
    print(txt)


def main():
    ap = ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("check", "report"):
        q = sub.add_parser(n)
        q.add_argument("--dir", default="learner-drift", help="learner-drift (20 users, one run per scenario) or "
                       "learner-drift-all (all 117 users, fused scenarios)")
    a = ap.parse_args()
    global OUT, ALL
    OUT = HERE / a.dir
    ALL = a.dir.endswith("-all")
    {"check": check, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
