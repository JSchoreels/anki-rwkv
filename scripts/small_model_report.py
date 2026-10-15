"""Small-model track report: full-history held-out results (scripts/small_model_heldout.py) against the fork.

1. Frontier table: every label (TAG/VARIANT) on the users every reported label has, review-weighted LogLoss and
   RMSE(bins) per head (weights = benchmark test reviews per user, as compress_report_b), and the paired Δ% against
   the reference label (default fork/w8+s8 = fork int8 weights + int8 state) with a 95% user-bootstrap CI.
2. Paired k-order screen (--pair CAND=BASE, repeatable, one per order seed): per order seed the paired review-weighted
   delta (candidate - base) per head; the accuracy-loop rule (docs/ACCURACY_LOOP.md, adopted 2026-10-14) -- all k
   deltas the same sign and |mean| > 2 * sigma * sqrt(2) / sqrt(k) -- with sigma measured here as the larger SD across
   order seeds of the bases and of the candidates (full-history protocol, these users); the window-protocol sigma
   (0.00220 curve / 0.00096 instant) is printed beside it. --rmse-bins adds the same paired rule on RMSE(bins) for
   every screen (calibration screens; no window-protocol sigma exists for it).

usage: small_model_report.py [--jsonl artifacts/small-model/heldout.jsonl] [--labels a/fp32,b/fp32 ...]
       [--reference fork/w8+s8] [--pair screen-w64kd-o0/fp32=screen-w64-o0/fp32 ...] [--out artifacts/small-model/heldout-report]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WINDOW_SIGMA = {"ahead": 0.00220, "instant": 0.00096}
HEADS = {"ahead": "curve", "instant": "instant"}


def load(path):
    per = {}
    meta = {}
    for row in map(json.loads, Path(path).read_text().splitlines()):
        if "benchmark" not in row or "ahead" not in row["benchmark"] or "instant" not in row["benchmark"]:
            continue
        per.setdefault(row["label"], {})[row["user_id"]] = row["benchmark"]
        meta[row["label"]] = {k: row.get(k) for k in ("parameters", "weight_bytes", "state_bits", "arch", "kind")}
    return per, meta


def arrays(per, label, users, head, metric):
    values = np.array([per[label][u][head][metric] for u in users], dtype=float)
    sizes = np.array([per[label][u][head]["size"] for u in users], dtype=float)
    return values, sizes


def rw(values, sizes):
    return float((values * sizes).sum() / sizes.sum())


def boot_delta_pct(cand, ref, sizes, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sizes), size=(n, len(sizes)))
    c = (cand[idx] * sizes[idx]).sum(1) / sizes[idx].sum(1)
    r = (ref[idx] * sizes[idx]).sum(1) / sizes[idx].sum(1)
    return list(np.percentile((c / r - 1) * 100, [2.5, 97.5]))


def frontier(per, meta, labels, reference):
    users = sorted(set.intersection(*(set(per[label]) for label in labels + [reference])))
    rows = []
    for label in labels:
        row = {"label": label, **meta[label], "users": len(users)}
        for head in HEADS:
            for metric in ("LogLoss", "RMSE(bins)"):
                cand, sizes = arrays(per, label, users, head, metric)
                ref, _ = arrays(per, reference, users, head, metric)
                row[f"{head}.{metric}"] = rw(cand, sizes)
                row[f"{head}.{metric}.delta_pct"] = (rw(cand, sizes) / rw(ref, sizes) - 1) * 100
                row[f"{head}.{metric}.ci95"] = boot_delta_pct(cand, ref, sizes)
        rows.append(row)
    return users, rows


def screen(per, pairs, metric="LogLoss"):
    labels = sorted({x for pair in pairs for x in pair})
    users = sorted(set.intersection(*(set(per[label]) for label in labels)))
    k = len(pairs)
    out = {"users": len(users), "k": k, "metric": metric, "heads": {}}
    for head in HEADS:
        deltas, base_vals, cand_vals = [], [], []
        for cand, base in pairs:
            c, sizes = arrays(per, cand, users, head, metric)
            b, _ = arrays(per, base, users, head, metric)
            deltas.append(rw(c, sizes) - rw(b, sizes))
            cand_vals.append(rw(c, sizes))
            base_vals.append(rw(b, sizes))
        sigma = max(float(np.std(base_vals, ddof=1)), float(np.std(cand_vals, ddof=1))) if k > 1 else float("nan")
        band = 2 * sigma * math.sqrt(2) / math.sqrt(k) if k > 1 else float("nan")
        window_band = 2 * WINDOW_SIGMA[head] * math.sqrt(2) / math.sqrt(k) if metric == "LogLoss" else float("nan")
        mean = float(np.mean(deltas))
        same_sign = all(d < 0 for d in deltas) or all(d > 0 for d in deltas)
        out["heads"][HEADS[head]] = {
            "deltas": deltas, "mean_delta": mean, "base_values": base_vals, "cand_values": cand_vals,
            "sigma_full_history": sigma, "band_full_history": band, "band_window_sigma": window_band,
            "consistent_sign": same_sign,
            "better_beyond_band": same_sign and mean < 0 and abs(mean) > max(band, 0) if k > 1 else None,
            "worse_beyond_band": same_sign and mean > 0 and abs(mean) > max(band, 0) if k > 1 else None}
    return out


def fmt_ci(ci):
    return f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"


def markdown(users, rows, reference, screens):
    lines = [f"# Small-model held-out report (full history, {len(users)} fork-unseen validation users)", "",
             f"Reference for Δ%: `{reference}`. Review-weighted over benchmark test reviews; 95% user-bootstrap CI.", "",
             "| Label | Params | Weight bytes | Curve LogLoss | Δ% [CI] | Instant LogLoss | Δ% [CI] | Curve RMSE(bins) | Δ% | Instant RMSE(bins) | Δ% |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['label']} | {r['parameters']:,} | {r['weight_bytes']:,} | {r['ahead.LogLoss']:.5f} | "
                     f"{r['ahead.LogLoss.delta_pct']:+.2f}% {fmt_ci(r['ahead.LogLoss.ci95'])} | {r['instant.LogLoss']:.5f} | "
                     f"{r['instant.LogLoss.delta_pct']:+.2f}% {fmt_ci(r['instant.LogLoss.ci95'])} | {r['ahead.RMSE(bins)']:.5f} | "
                     f"{r['ahead.RMSE(bins).delta_pct']:+.1f}% | {r['instant.RMSE(bins)']:.5f} | {r['instant.RMSE(bins).delta_pct']:+.1f}% |")
    for name, s in screens.items():
        lines += ["", f"## Paired screen: {name} ({s.get('metric', 'LogLoss')}, k = {s['k']}, {s['users']} users)", "",
                  "| Head | Deltas (cand - base) | Mean | Band (full-history sigma) | Band (window sigma) | Same sign | Verdict |",
                  "|---|---|---:|---:|---:|---|---|"]
        for head, h in s["heads"].items():
            verdict = "better" if h["better_beyond_band"] else "worse" if h["worse_beyond_band"] else "inside band"
            lines.append(f"| {head} | {', '.join(f'{d:+.5f}' for d in h['deltas'])} | {h['mean_delta']:+.5f} | "
                         f"{h['band_full_history']:.5f} | {h['band_window_sigma']:.5f} | {h['consistent_sign']} | {verdict} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default="artifacts/small-model/heldout.jsonl")
    parser.add_argument("--labels", help="comma list (default: every label in the file)")
    parser.add_argument("--reference", default="fork/w8+s8")
    parser.add_argument("--pair", action="append", default=[], help="CAND=BASE; group with --screen-name")
    parser.add_argument("--screen", action="append", default=[],
                        help="NAME:CAND1=BASE1,CAND2=BASE2,... one paired k-order screen per flag")
    parser.add_argument("--rmse-bins", action="store_true", help="also apply the paired rule to RMSE(bins)")
    parser.add_argument("--out", default="artifacts/small-model/heldout-report")
    args = parser.parse_args()
    per, meta = load(ROOT / args.jsonl)
    labels = args.labels.split(",") if args.labels else sorted(per)
    users, rows = frontier(per, meta, labels, args.reference)
    screens = {}
    for spec in args.screen:
        name, _, body = spec.partition(":")
        screens[name] = screen(per, [tuple(p.split("=")) for p in body.split(",")])
        if args.rmse_bins:
            screens[f"{name} RMSE(bins)"] = screen(per, [tuple(p.split("=")) for p in body.split(",")], "RMSE(bins)")
    if args.pair:
        screens["pairs"] = screen(per, [tuple(p.split("=")) for p in args.pair])
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.with_suffix(".json")).write_text(json.dumps({"users": users, "reference": args.reference, "rows": rows,
                                                     "screens": screens}, indent=1))
    text = markdown(users, rows, args.reference, screens)
    out.with_suffix(".md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
