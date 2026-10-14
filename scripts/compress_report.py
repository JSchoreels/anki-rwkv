"""Summarise the compression experiments: collection (one user) and held-out users, curve and instant LogLoss.

Prints Markdown tables and writes artifacts/compression/summary.json. Held-out aggregates use only users scored for
every listed label (paired), review-weighted (benchmark test reviews) and user-averaged, with a 95% bootstrap interval
over users for the paired user-averaged difference against the reference label.

usage: compress_report.py [--reference fork/fp32] [--fork-int8 fork/fp32+s8]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "compression"
FORK_FP32 = {"curve": 0.4417, "instant": 0.4366}  # artifacts/collection-eval/2026-10-14 (main checkout), Model B full


def pct(value, ref):
    return 100.0 * (value / ref - 1.0)


def collection_table(rows, reference, fork_int8):
    by = {r["label"]: r for r in rows if r["protocol"] == "full"}
    if reference not in by:
        return [], {}
    ref, i8 = by[reference], by.get(fork_int8)
    compact = compact_bytes()
    lines = ["| Variant | Weight bytes | Bytes, f16 scales/other | Card/note state bytes | Curve LogLoss | Δ% vs fork fp32 | Δ% vs fork int8 | "
             "Instant LogLoss | Δ% vs fork fp32 | Δ% vs fork int8 | Curve RMSE(bins) | Instant RMSE(bins) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    summary = {}
    for label, r in by.items():
        c, i = r["curve"]["LogLoss"], r["instant"]["LogLoss"]
        entry = {"weight_bytes": r["weight_bytes"]["total"], "card_state_bytes": r["card_state_bytes"],
                 "note_state_bytes": r["note_state_bytes"], "curve": c, "instant": i,
                 "curve_rmse_bins": r["curve"]["RMSE(bins)"], "instant_rmse_bins": r["instant"]["RMSE(bins)"],
                 "curve_delta_pct_vs_fp32": pct(c, ref["curve"]["LogLoss"]),
                 "instant_delta_pct_vs_fp32": pct(i, ref["instant"]["LogLoss"])}
        if i8:
            entry["curve_delta_pct_vs_int8"] = pct(c, i8["curve"]["LogLoss"])
            entry["instant_delta_pct_vs_int8"] = pct(i, i8["instant"]["LogLoss"])
        summary[label] = entry
        f16 = compact.get(r["weights"]) if label.startswith("fork/") else None
        entry["weight_bytes_f16_scales_other"] = f16
        lines.append(f"| {label} | {entry['weight_bytes']:,} | {f'{f16:,}' if f16 else '—'} | {r['card_state_bytes']:,} / {r['note_state_bytes']:,} | "
                     f"{c:.4f} | {entry['curve_delta_pct_vs_fp32']:+.2f}% | "
                     f"{entry.get('curve_delta_pct_vs_int8', float('nan')):+.2f}% | {i:.4f} | "
                     f"{entry['instant_delta_pct_vs_fp32']:+.2f}% | {entry.get('instant_delta_pct_vs_int8', float('nan')):+.2f}% | "
                     f"{entry['curve_rmse_bins']:.4f} | {entry['instant_rmse_bins']:.4f} |")
    return lines, summary


def compact_bytes():
    sys.path.insert(0, str(ROOT))
    import torch
    from scripts import compress_quant as cq
    state = torch.load(cq.FORK_WEIGHTS, map_location="cpu", weights_only=True)
    names = cq.linear_weight_names(state)
    return {k: (cq.weight_bytes(state, names, b, g, scale_bytes=2, other_bytes=2)["total"] if b else None)
            for k, (b, g) in cq.WEIGHT_SCHEMES.items()}


def heldout_table(rows, reference, labels=None, seed=0):
    per = {}
    for r in rows:
        if "benchmark" not in r or "ahead" not in r["benchmark"]:
            continue
        per.setdefault(r["label"], {})[r["user_id"]] = r
    if reference not in per:
        return [], {}
    labels = labels or sorted(per)
    users = sorted(set.intersection(*(set(per[l]) for l in labels)))
    if not users:
        return [], {}
    rng = np.random.default_rng(seed)
    boots = rng.integers(0, len(users), size=(2000, len(users)))
    out = {"users": len(users), "labels": {}}
    lines = [f"Paired users: {len(users)}.", "",
             "| Variant | Weight bytes | Curve LL (review-weighted) | Δ% | Curve LL (user mean) | Δ% [95% CI] | "
             "Instant LL (review-weighted) | Δ% | Instant LL (user mean) | Δ% [95% CI] |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def stats(label, head):
        size = np.array([per[label][u]["benchmark"][head]["size"] for u in users], dtype=float)
        ll = np.array([per[label][u]["benchmark"][head]["LogLoss"] for u in users])
        return size, ll

    for label in labels:
        entry = {"weight_bytes": per[label][users[0]].get("weight_bytes")}
        cells = []
        for head, name in (("ahead", "curve"), ("instant", "instant")):
            size, ll = stats(label, head)
            _, ref = stats(reference, head)
            weighted, weighted_ref = float((ll * size).sum() / size.sum()), float((ref * size).sum() / size.sum())
            mean, mean_ref = float(ll.mean()), float(ref.mean())
            diff = ll - ref
            ci = np.percentile(diff[boots].mean(axis=1), [2.5, 97.5]) / mean_ref * 100
            entry[name] = {"review_weighted": weighted, "user_mean": mean,
                           "review_weighted_delta_pct": pct(weighted, weighted_ref),
                           "user_mean_delta_pct": pct(mean, mean_ref), "user_mean_delta_pct_ci95": list(map(float, ci)),
                           "users_worse": int((diff > 0).sum())}
            cells += [f"{weighted:.4f}", f"{pct(weighted, weighted_ref):+.2f}%", f"{mean:.4f}",
                      f"{pct(mean, mean_ref):+.2f}% [{ci[0]:+.2f}, {ci[1]:+.2f}]"]
        out["labels"][label] = entry
        lines.append(f"| {label} | {entry['weight_bytes'] or 0:,} | " + " | ".join(cells) + " |")
    return lines, out


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="fork/fp32")
    parser.add_argument("--fork-int8", default="fork/fp32+s8")
    parser.add_argument("--heldout", default="heldout-scores.jsonl")
    parser.add_argument("--labels", help="comma list of held-out labels (default: all)")
    args = parser.parse_args()
    summary = {}
    lines, summary["collection"] = collection_table(read_jsonl(OUT / "collection-scores.jsonl"), args.reference,
                                                    args.fork_int8)
    print("## Collection (full history, 173,690 benchmark test reviews)\n")
    print("\n".join(lines) + "\n")
    labels = args.labels.split(",") if args.labels else None
    lines, summary["heldout"] = heldout_table(read_jsonl(OUT / args.heldout), args.reference, labels)
    print("## Held-out users (full history)\n")
    print("\n".join(lines))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.exit(main())
