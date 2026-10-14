"""Report held-out evaluation B (artifacts/compression/heldout-b.jsonl): paired variants vs fp32 on identical users.

For each head (curve = ahead, instant) and metric (LogLoss, RMSE(bins), AUC): review-weighted mean (weights = the
user's benchmark test reviews) and user mean, Δ% vs fp32, and 95% bootstrap intervals over users (2,000 resamples) for
the paired Δ% of both aggregates. Also splits users at 20,000 reviews to show the effect of long histories.
Writes artifacts/compression/heldout-b-summary.json and prints Markdown.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "compression"
VARIANTS = ["fp32", "fp32+s8", "w8-pc+s8", "w4-g32+s8", "w4-g64+s8", "w4-g32+s4"]
METRICS = ["LogLoss", "RMSE(bins)", "AUC"]
HEADS = [("ahead", "curve"), ("instant", "instant")]


def load(path):
    per = {}
    errors = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if "error" in row:
            errors[row["user_id"]] = row["error"]
            continue
        if "ahead" in row["benchmark"] and "instant" in row["benchmark"]:
            per.setdefault(row["variant"], {})[row["user_id"]] = row
    return per, errors


def table(per, users, seed=0):
    rng = np.random.default_rng(seed)
    boots = rng.integers(0, len(users), size=(2000, len(users)))
    result = {}
    for head, hname in HEADS:
        size = np.array([per["fp32"][u]["benchmark"][head]["size"] for u in users], dtype=float)
        for metric in METRICS:
            ref = np.array([per["fp32"][u]["benchmark"][head][metric] for u in users], dtype=float)
            for variant in VARIANTS:
                if variant not in per:
                    continue
                val = np.array([per[variant][u]["benchmark"][head][metric] for u in users], dtype=float)
                rw, rw_ref = (val * size).sum() / size.sum(), (ref * size).sum() / size.sum()
                um, um_ref = val.mean(), ref.mean()
                b_rw = ((val[boots] * size[boots]).sum(1) / (ref[boots] * size[boots]).sum(1) - 1) * 100
                b_um = (val[boots].mean(1) / ref[boots].mean(1) - 1) * 100
                result.setdefault(variant, {}).setdefault(hname, {})[metric] = {
                    "review_weighted": rw, "review_weighted_delta_pct": (rw / rw_ref - 1) * 100,
                    "review_weighted_ci95": list(np.percentile(b_rw, [2.5, 97.5])),
                    "user_mean": um, "user_mean_delta_pct": (um / um_ref - 1) * 100,
                    "user_mean_ci95": list(np.percentile(b_um, [2.5, 97.5])),
                    "users_worse": int((val > ref).sum()) if metric != "AUC" else int((val < ref).sum())}
    return result


def markdown(result, users, title):
    lines = [f"### {title} ({len(users)} users)", "",
             "| Variant | Head | LogLoss (rw) | Δ% [95% CI] | LogLoss (user mean) | Δ% [95% CI] | RMSE(bins) (rw) | Δ% [95% CI] | AUC (rw) | Δ% [95% CI] |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        if variant not in result:
            continue
        for _, hname in HEADS:
            m = result[variant][hname]

            def cell(x, key):
                lo, hi = x[f"{key}_ci95"]
                return f"{x[f'{key}_delta_pct']:+.3f}% [{lo:+.3f}, {hi:+.3f}]"
            lines.append(f"| {variant} | {hname} | {m['LogLoss']['review_weighted']:.5f} | {cell(m['LogLoss'], 'review_weighted')} | "
                         f"{m['LogLoss']['user_mean']:.5f} | {cell(m['LogLoss'], 'user_mean')} | "
                         f"{m['RMSE(bins)']['review_weighted']:.5f} | {cell(m['RMSE(bins)'], 'review_weighted')} | "
                         f"{m['AUC']['review_weighted']:.5f} | {cell(m['AUC'], 'review_weighted')} |")
    return lines


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    run = json.loads((Path(r"C:\Users\jonat\workspace\anki-rwkv\artifacts\full-run\run-8k-20261013") / "run.json").read_text())
    counts = run["review_counts"]
    per, errors = load(OUT / "heldout-b.jsonl")
    present = [v for v in VARIANTS if v in per]
    users = sorted(set.intersection(*(set(per[v]) for v in present)))
    long_users = [u for u in users if counts[str(u)] > 20000]
    short_users = [u for u in users if counts[str(u)] <= 20000]
    summary = {"users": len(users), "variants": present, "errors": errors,
               "reviews": int(sum(counts[str(u)] for u in users)),
               "benchmark_test_reviews": int(sum(per["fp32"][u]["benchmark"]["ahead"]["size"] for u in users)),
               "all": table(per, users)}
    out = markdown(summary["all"], users, "All paired users")
    for name, group in (("histories ≤ 20,000 reviews", short_users), ("histories > 20,000 reviews", long_users)):
        if len(group) >= 20:
            summary[name] = table(per, group)
            out += [""] + markdown(summary[name], group, name)
    summary["model_seconds"] = {v: float(sum(per[v][u]["model_seconds"] for u in users)) for v in present}
    (OUT / "heldout-b-summary.json").write_text(json.dumps(summary, indent=1))
    print(f"users {len(users)}, reviews {summary['reviews']:,}, benchmark test reviews "
          f"{summary['benchmark_test_reviews']:,}, errors {len(errors)}\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
