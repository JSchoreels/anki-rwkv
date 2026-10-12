"""Replicate statistics for exact R01 runs (R01-M01; analysis only, no training).

Works on log reviews/s. With one arm (A/A calibration) it estimates run-to-run noise: the SD of
log rates and of differences between consecutive runs, and how many interleaved blocks would
detect a given gain. With two arms it pairs the k-th reference run with the k-th candidate run in
launch order (R C R C ... blocks) and reports the mean block effect with a t interval. Promotion
rule proposed for the round: lower bound above zero promotes; positive estimate with an interval
including zero is a stack candidate; otherwise reject.

  r01_replicate_stats.py --runs DIR [DIR ...] [--reference-track T] [--output JSON]
"""
from argparse import ArgumentParser
import json
import math
from pathlib import Path
import statistics
import sys

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.r01_paired_timing import compare, load_run


def run_record(case):
    case = Path(case)
    result = json.loads((case / "result.json").read_text(encoding="utf-8"))
    return {"case": case, "track": result["track"], "started": result.get("training_started_utc"),
            "rate": result["reviews_per_second"], "log_rate": math.log(result["reviews_per_second"]),
            "trained": result["trained_parameter_sha256"],
            "timed": (case / "updates.jsonl").exists() and "host_started_s" in (case / "updates.jsonl").read_text(encoding="utf-8").split("\n", 1)[0]}


def blocks_needed(sd_difference, gain_percent, alpha=0.05, power=0.80):
    """Interleaved blocks for a one-sided test of a gain, normal approximation."""
    delta = math.log(1 + gain_percent / 100)
    z = stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)
    return max(2, math.ceil((z * sd_difference / delta) ** 2))


def interval(values, level=0.95):
    n = len(values)
    mean = statistics.fmean(values)
    if n < 2:
        return mean, None, None
    half = stats.t.ppf(0.5 + level / 2, n - 1) * statistics.stdev(values) / math.sqrt(n)
    return mean, mean - half, mean + half


def percent(log_value):
    return None if log_value is None else 100 * (math.exp(log_value) - 1)


def analyse(cases, reference_track=None):
    runs = sorted((run_record(case) for case in cases), key=lambda run: run["started"] or "")
    reference_track = reference_track or runs[0]["track"]
    arms = {"reference": [r for r in runs if r["track"] == reference_track],
            "candidate": [r for r in runs if r["track"] != reference_track]}
    out = {"runs": [{"case": r["case"].as_posix(), "track": r["track"], "started": r["started"],
                     "reviews_per_second": r["rate"]} for r in runs], "arms": {}}
    for name, arm in arms.items():
        if not arm:
            continue
        logs = [r["log_rate"] for r in arm]
        entry = {"n": len(arm), "mean_reviews_per_second": math.exp(statistics.fmean(logs)),
                 "exact_trained_weights": len({r["trained"] for r in arm}) == 1}
        if len(arm) >= 2:
            diffs = [b - a for a, b in zip(logs, logs[1:])]
            entry.update(sd_log_rate_percent=100 * statistics.stdev(logs),
                         consecutive_difference_sd_percent=100 * statistics.stdev(diffs) if len(diffs) >= 2 else None,
                         median_pairwise_spread_percent=statistics.median(
                             100 * (math.exp(abs(a - b)) - 1) for i, a in enumerate(logs) for b in logs[i + 1:]))
            timed = [r for r in arm if r["timed"]]
            if len(timed) >= 2:
                pairs = [compare([load_run(a["case"])], [load_run(b["case"])], resamples=500) for a, b in zip(timed, timed[1:])]
                entry["consecutive_total_gain_percent"] = [p["wall"]["gain_percent"] for p in pairs]
                entry["consecutive_median_update_gain_percent"] = [p["wall"]["median_update_gain_percent"] for p in pairs]
            # Run-to-run SD of a difference between two runs; sqrt(2) x the single-run SD when runs are independent.
            sd_difference = math.sqrt(2) * statistics.stdev(logs)
            entry["blocks_needed_one_sided_5pct_power_80"] = {f"{g}%": blocks_needed(sd_difference, g) for g in (1, 2, 3, 5)}
        out["arms"][name] = entry
    if arms["candidate"]:
        effects = [math.log(c["rate"] / r["rate"]) for r, c in zip(arms["reference"], arms["candidate"])]
        mean, low, high = interval(effects)
        decision = ("needs at least two blocks" if low is None else "promote" if low > 0 else
                    "stack candidate" if mean > 0 else "reject")
        out["blocks"] = {"n": len(effects), "effects_percent": [percent(e) for e in effects],
                         "mean_gain_percent": percent(mean), "ci95_percent": [percent(low), percent(high)],
                         "proposed_decision": decision,
                         "exact_parity_across_arms": len({r["trained"] for r in runs}) == 1}
    return out


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="Case directories (each holding result.json)")
    parser.add_argument("--reference-track")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = analyse([ROOT / case for case in args.runs], args.reference_track)
    if args.output:
        target = (ROOT / args.output).resolve()
        if not target.is_relative_to(ROOT / "artifacts"):
            parser.error("Write analysis outputs under artifacts/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    for name, arm in out["arms"].items():
        print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in arm.items()
                                if not isinstance(v, list)}, default=str)[:600])
    if "blocks" in out:
        b = out["blocks"]
        print(f"blocks {b['n']}: mean {b['mean_gain_percent']:+.2f}% CI {b['ci95_percent']} -> {b['proposed_decision']}")


if __name__ == "__main__":
    main()
