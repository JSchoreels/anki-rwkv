"""Render the SM66 result tables (markdown) from summary.json. usage: sm66_report.py artifacts/small-model/sm66/summary.json"""
import json
import sys
from pathlib import Path

r = json.loads(Path(sys.argv[1]).read_text())
S = r["schemes"]
NAMES = {
    "global_static": "global calibration (control)",
    "offline_B": "per-user bias, offline",
    "offline_BT": "per-user bias+temperature, offline",
    "offline_B_on_global": "per-user bias on top of global, offline",
    "offline_BT_on_global": "per-user bias+temperature on top of global, offline",
}
for blk in r["config"]["online_blocks"]:
    NAMES[f"online{blk}_B"] = f"per-user bias, online (block {blk})"
    NAMES[f"online{blk}_BT"] = f"per-user bias+temperature, online (block {blk})"
    NAMES[f"online{blk}_B_allblocks"] = f"per-user bias, online (block {blk}), all blocks after the first"
    NAMES[f"online{blk}_BT_allblocks"] = f"per-user bias+temperature, online (block {blk}), all blocks after the first"


def f(x):
    return f"{x:+.5f}"


def cell(e):
    return f"{f(e['delta'])} [{f(e['ci'][0])}, {f(e['ci'][1])}] ({e['delta_pct']:+.2f}%)"


out = []
out.append("Delta = LogLoss with the correction - without, review-weighted over users, 95% user bootstrap (2,000 resamples, seed 0). Negative = better.\n")
for h in ("curve", "instant"):
    out.append(f"### {h.capitalize()} head\n")
    out.append("| Scheme | Small model (SM56) | Big model | Gap small-big change (before -> after) |")
    out.append("|---|---|---|---|")
    b = S[h]["global_static"]
    for k, nm in NAMES.items():
        if k not in S[h]:
            continue
        row = S[h][k]
        g = row["gap"]
        out.append(f"| {nm} | {cell(row['small'])} | {cell(row['big'])} | {f(g['change'])} [{f(g['ci'][0])}, {f(g['ci'][1])}] ({g['small_minus_big_before']:.5f} -> {g['small_minus_big_after']:.5f}) |")
    out.append("")
out.append("### Per-user correction versus the global control (small model; delta of LogLoss, same evaluation reviews)\n")
out.append("| Head | Scheme | Small: per-user - global | Big: per-user - global |")
out.append("|---|---|---|---|")
for h in ("curve", "instant"):
    for k, nm in NAMES.items():
        row = S[h].get(k, {})
        if "small_vs_global_static" in row and k != "global_static":
            a, c = row["small_vs_global_static"], row["big_vs_global_static"]
            out.append(f"| {h} | {nm} | {f(a['delta'])} [{f(a['lo'])}, {f(a['hi'])}] | {f(c['delta'])} [{f(c['lo'])}, {f(c['hi'])}] |")
out.append("")
out.append("### By user size (terciles of reviews per user; small model, then big model delta)\n")
tr = r["tercile_review_ranges"]
out.append("Terciles (min-max dumped test reviews, users): " + "; ".join(f"T{i + 1} {a}-{b_} ({c})" for i, (a, b_, c) in enumerate(tr)) + "\n")
out.append("| Head | Scheme | T1 small | T2 small | T3 small | T1 big | T2 big | T3 big |")
out.append("|---|---|---|---|---|---|---|---|")
for h in ("curve", "instant"):
    for k in ("global_static", "offline_B", "offline_BT", f"online{r['config']['online_blocks'][0]}_B", f"online{r['config']['online_blocks'][0]}_BT"):
        row = S[h][k]["terciles"]
        cells = [f"{f(row[f'T{t}']['small']['delta'])} [{f(row[f'T{t}']['small']['lo'])}, {f(row[f'T{t}']['small']['hi'])}]" for t in (1, 2, 3)]
        cells += [f"{f(row[f'T{t}']['big']['delta'])} [{f(row[f'T{t}']['big']['lo'])}, {f(row[f'T{t}']['big']['hi'])}]" for t in (1, 2, 3)]
        out.append(f"| {h} | {NAMES[k]} | " + " | ".join(cells) + " |")
out.append("")
out.append("### Share of users improved and CV-chosen prior strengths (small model)\n")
out.append("| Head | Scheme | Users improved | Prior strength per CV fold |")
out.append("|---|---|---|---|")
for h in ("curve", "instant"):
    for k in ("offline_B", "offline_BT", f"online{r['config']['online_blocks'][0]}_B", f"online{r['config']['online_blocks'][0]}_BT"):
        e = S[h][k]["small"]
        pri = "; ".join(f"f{fo}: " + ",".join(f"{a}={v:g}" for a, v in p.items()) for fo, p in e["cv_prior"].items())
        out.append(f"| {h} | {NAMES[k]} | {e['frac_users_improved'] * 100:.0f}% | {pri} |")
print("\n".join(out))
