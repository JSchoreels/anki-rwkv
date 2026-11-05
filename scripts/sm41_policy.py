#!/usr/bin/env python
"""The small model in the review simulator (SM41), part B: the policy search report (CPU, numpy only).

Plan, metric and decision rule (written before the runs): artifacts/small-model/sm41/policy-result.md.

  choose  --rows GLOB --out LANES   the decision rule of the plan on the screen rows -> the confirmation lanes
  report  --rows GLOB [--exclude-sample] [--md PATH]
          every policy family in the rows (lane base without its target, e.g. ihr-*-s1) vs FSRS-7 and vs the
          instant head, both read from the full-run rows of the same users / splits / seeds; matched realised
          retention over the common range (the full run's primary metric), paired per user, 95% CI over users.
"""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import sm41_fullrun as FR   # noqa: E402

HERE = ROOT / "artifacts" / "small-model" / "sm41"
SAMPLE = HERE / "diag" / "sample.json"
SCREEN_R = [0.80, 0.85, 0.90]
TXT = {"ihr": "instant-head desired retention", "curve": "curve-head desired retention",
       "hyb": "hybrid (FSRS-7's interval corrected by the instant head)", "fsrs7": "FSRS-7 desired retention"}


def family(base):
    """'ihr-0.85-s1' -> ('ihr-*-s1', 0.85)."""
    parts = base.split("-")
    return "-".join([parts[0], "*"] + parts[2:]), float(parts[1])


def label(fam):
    parts = fam.split("-")
    s = TXT.get(parts[0], parts[0])
    for o in parts[2:]:
        s += {"n": f", target noise s = {o[1:]}", "s": f", time-aware target beta = {o[1:]}",
              "a": f", alpha = {o[1:]}"}.get(o[0], f", {o}")
    return f"{s} ({fam})"


def overlap_gap(a, b, key="cost"):
    """Mean relative gap a / b - 1 over the common realised-retention range (the full run's primary metric)."""
    a = [x for x in a if np.isfinite(x["ret"])]; b = [x for x in b if np.isfinite(x["ret"])]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    lo = max(min(p["ret"] for p in a), min(p["ret"] for p in b))
    hi = min(max(p["ret"] for p in a), max(p["ret"] for p in b))
    if not hi - lo >= FR.MIN_OVERLAP:
        return float("nan")
    grid = np.linspace(lo, hi, max(2, int(round((hi - lo) / 0.005)) + 1))
    return float(np.mean([FR.interp_cost(a, x, key) / FR.interp_cost(b, x, key) - 1 for x in grid]))


def fmt(v, scale=100, d=1):
    m, h, n = FR.ci(v)
    if not np.isfinite(m):
        return "—"
    return f"{scale * m:+.{d}f} ± {scale * h:.{d}f}"


def report(args):
    rows = FR.load_rows(args.rows)
    full = {(r["user"], r["split_day"]): r for r in FR.load_rows(str(HERE / "fullrun" / "rows" / "*.jsonl"))}
    sample = {u["user"] for u in json.loads(SAMPLE.read_text())["users"]}
    if args.exclude_sample:
        rows = [r for r in rows if r["user"] not in sample]
    per = {}                    # user -> list of (screen split policies, full-run split policies)
    for r in rows:
        per.setdefault(r["user"], []).append((FR.split_policies(r), FR.split_policies(full[(r["user"], r["split_day"])])))
    users = sorted(per)
    fams = sorted({family(b)[0] for r in rows for b in {n.split("#")[0] for n in r["lanes"]} if b != "tf"})
    L = [f"Rows: `{args.rows}`{' (screen users excluded)' if args.exclude_sample else ''}; {len(users)} users, "
         f"{len(rows)} splits.", "",
         "| policy | users | gap vs FSRS-7, % (95% CI) | median | users cheaper than FSRS-7 | gap vs instant head "
         "(0.80-0.90), % (95% CI) | gap vs FSRS-7, 60-day cost |", "|---|---|---|---|---|---|---|"]
    out = {"users": len(users), "families": {}}
    ref_ihr3 = lambda fr: [fr[f"ihr-{x:.2f}"] for x in SCREEN_R]          # noqa: E731
    ref_fs = lambda fr: [fr[f"fsrs7-{x:.2f}"] for x in FR.LEVELS]          # noqa: E731
    # the reference: the instant head on the same 3 targets, vs FSRS-7
    for fam in ["__ihr3"] + fams:
        g_fs, g_ih, g_all = [], [], []
        for u in users:
            a, b, c = [], [], []
            for sp, fr in per[u]:
                pts = ref_ihr3(fr) if fam == "__ihr3" else [sp[k] for k in sp if k != "tf" and family(k)[0] == fam]
                if len(pts) < 2:
                    continue
                a.append(overlap_gap(pts, ref_fs(fr))); c.append(overlap_gap(pts, ref_fs(fr), "cost_all"))
                b.append(overlap_gap(pts, ref_ihr3(fr)))
            a = [x for x in a if np.isfinite(x)]; b = [x for x in b if np.isfinite(x)]; c = [x for x in c if np.isfinite(x)]
            if a:
                g_fs.append(np.mean(a))
            if b:
                g_ih.append(np.mean(b))
            if c:
                g_all.append(np.mean(c))
        m, h, n = FR.ci(g_fs)
        name = "instant-head desired retention, targets 0.80 / 0.85 / 0.90 (the reference, full-run rows)" \
            if fam == "__ihr3" else label(fam)
        L.append(f"| {name} | {n} | {fmt(g_fs)} | {100 * np.median(g_fs):+.1f} | "
                 f"{100 * np.mean(np.array(g_fs) < 0):.0f}% | {'—' if fam == '__ihr3' else fmt(g_ih)} | {fmt(g_all)} |")
        out["families"][fam] = {"vs_fsrs7": FR.ci(g_fs), "median": float(np.median(g_fs)) if g_fs else None,
                                "cheaper_share": float(np.mean(np.array(g_fs) < 0)) if g_fs else None,
                                "vs_ihr": FR.ci(g_ih), "vs_fsrs7_60day": FR.ci(g_all),
                                "per_user_vs_fsrs7": dict(zip([str(u) for u in users], map(float, g_fs)))
                                if len(g_fs) == len(users) else None}
    # per-target table
    L += ["", "| policy | target | realised retention 30-59 | min/day 30-59 | memorised | cost |", "|---|---|---|---|---|---|"]
    bases = sorted({b for r in rows for b in {n.split("#")[0] for n in r["lanes"]} if b != "tf"},
                   key=lambda b: family(b))
    refs = [f"ihr-{x:.2f}" for x in SCREEN_R] + [f"fsrs7-{x:.2f}" for x in FR.LEVELS]
    for b in refs + bases:
        src = 1 if b in refs else 0
        v = [np.mean([s[src][b][k] for s in per[u]]) for u in users for k in ("ret",)]
        mm = lambda key: float(np.nanmean([np.mean([s[src][b][key] for s in per[u]]) for u in users]))   # noqa: E731
        L.append(f"| {b}{' (full run)' if src else ''} | {family(b)[1]:.2f} | {np.nanmean(v):.3f} | "
                 f"{mm('min_30_59') / 30:.1f} | {mm('memorised'):.0f} | {mm('cost'):.3f} |")
    txt = "\n".join(L) + "\n"
    if args.md:
        Path(args.md).write_text(txt, encoding="utf8")
        Path(args.md).with_suffix(".json").write_text(json.dumps(out, indent=1))
    print(txt)


def screen_gaps(rows_glob):
    """family -> per-user gaps vs FSRS-7 (the screen's primary metric)."""
    rows = FR.load_rows(rows_glob)
    full = {(r["user"], r["split_day"]): r for r in FR.load_rows(str(HERE / "fullrun" / "rows" / "*.jsonl"))}
    per = {}
    for r in rows:
        per.setdefault(r["user"], []).append((FR.split_policies(r), FR.split_policies(full[(r["user"], r["split_day"])])))
    fams = sorted({family(b)[0] for r in rows for b in {n.split("#")[0] for n in r["lanes"]} if b != "tf"})
    out = {}
    for fam in fams:
        g = []
        for u in sorted(per):
            a = [overlap_gap([sp[k] for k in sp if k != "tf" and family(k)[0] == fam],
                             [fr[f"fsrs7-{x:.2f}"] for x in FR.LEVELS]) for sp, fr in per[u]]
            a = [x for x in a if np.isfinite(x)]
            if a:
                g.append(float(np.mean(a)))
        if g:
            out[fam] = FR.ci(g)
    return out


def choose(args):
    """The plan's rule: a family passes if its screen gap vs FSRS-7 is < 0 or its CI includes 0; confirm the best
    time-aware family (option s) if it passes, plus the best other passing family; at most 2 (if no time-aware
    family passes, the best 2 passing families)."""
    g = screen_gaps(args.rows)
    passing = {f: v for f, v in g.items() if v[0] < 0 or v[0] - v[1] <= 0}
    ta = sorted([f for f in passing if "-s" in f], key=lambda f: passing[f][0])
    other = sorted([f for f in passing if "-s" not in f], key=lambda f: passing[f][0])
    pick = (ta[:1] + other[:1]) if ta else other[:2]
    lanes = ["tf"] + [fam.replace("*", f"{r:.2f}") + f"#{sd}" for fam in pick for r in SCREEN_R for sd in range(4)]
    txt = {"screen_gaps_vs_fsrs7": g, "passing": sorted(passing), "confirm": pick}
    Path(args.out).write_text(",".join(lanes) + chr(10))
    Path(args.out).with_suffix(".json").write_text(json.dumps(txt, indent=1))
    print(json.dumps(txt, indent=1))


def main():
    ap = ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("report"); t.add_argument("--rows", required=True)
    t.add_argument("--exclude-sample", action="store_true"); t.add_argument("--md", default="")
    c = sub.add_parser("choose"); c.add_argument("--rows", required=True); c.add_argument("--out", required=True)
    a = ap.parse_args()
    {"report": report, "choose": choose}[a.cmd](a)


if __name__ == "__main__":
    main()
