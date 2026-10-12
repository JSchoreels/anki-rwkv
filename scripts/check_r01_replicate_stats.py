"""Synthetic checks for scripts/r01_replicate_stats.py (CPU only, no training)."""
from argparse import ArgumentParser
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance
from scripts.r01_replicate_stats import analyse, blocks_needed


def write(folder, track, rate, minute, trained="same"):
    folder.mkdir(parents=True)
    (folder / "result.json").write_text(json.dumps({"track": track, "reviews_per_second": rate, "trained_parameter_sha256": trained,
                                                    "training_started_utc": f"2026-10-12T21:{minute:02d}:00+00:00"}), encoding="utf-8")
    (folder / "updates.jsonl").write_text(json.dumps({"step": 1}) + "\n", encoding="utf-8")
    return folder


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        parser.error("Use a new workspace artifact directory")
    checks = []

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})
        if not condition:
            raise AssertionError(name)

    f = output / "fixtures"
    aa = [write(f / f"aa{i}", "A", rate, i) for i, rate in enumerate([100.0, 102.0, 99.0, 101.0])]
    calibration = analyse(aa)
    arm = calibration["arms"]["reference"]
    logs = [math.log(r) for r in (100, 102, 99, 101)]
    mean = sum(logs) / 4
    sd = math.sqrt(sum((x - mean) ** 2 for x in logs) / 3)
    check("calibration SD is the sample SD of log rates", abs(arm["sd_log_rate_percent"] - 100 * sd) < 1e-9)
    check("runs are ordered by training start, not argument order", [r["reviews_per_second"] for r in analyse(list(reversed(aa)))["runs"]] == [100, 102, 99, 101])
    check("A/A calibration has no candidate decision", "blocks" not in calibration)
    check("more blocks are needed for a smaller gain", arm["blocks_needed_one_sided_5pct_power_80"]["1%"] > arm["blocks_needed_one_sided_5pct_power_80"]["5%"])
    check("block count follows the normal approximation", blocks_needed(0.02, 2.0) == math.ceil(((1.6448536 + 0.8416212) * 0.02 / math.log(1.02)) ** 2))
    ab = []
    for i, (a, b) in enumerate([(100, 104), (98, 102.5), (101, 104.5)]):
        ab += [write(f / f"ab{i}a", "A", a, 10 + 2 * i), write(f / f"ab{i}b", "B", b, 11 + 2 * i)]
    clear = analyse(ab, "A")["blocks"]
    check("consistent gains give a positive interval and promotion", clear["ci95_percent"][0] > 0 and clear["proposed_decision"] == "promote")
    mixed = []
    for i, (a, b) in enumerate([(100, 103), (100, 99), (100, 102)]):
        mixed += [write(f / f"mx{i}a", "A", a, 20 + 2 * i), write(f / f"mx{i}b", "B", b, 21 + 2 * i)]
    noisy = analyse(mixed, "A")["blocks"]
    check("a positive mean with an interval including zero is a stack candidate", noisy["mean_gain_percent"] > 0 and noisy["ci95_percent"][0] < 0 and noisy["proposed_decision"] == "stack candidate")
    worse = []
    for i, (a, b) in enumerate([(100, 98), (100, 99), (100, 101)]):
        worse += [write(f / f"wr{i}a", "A", a, 30 + 2 * i), write(f / f"wr{i}b", "B", b, 31 + 2 * i)]
    check("a negative mean is rejected", analyse(worse, "A")["blocks"]["proposed_decision"] == "reject")
    single = analyse([write(f / "one-a", "A", 100, 40), write(f / "one-b", "B", 105, 41)], "A")["blocks"]
    check("one block is not enough for a decision", single["proposed_decision"] == "needs at least two blocks")
    differing = analyse([write(f / "d-a", "A", 100, 50), write(f / "d-b", "B", 105, 51, trained="other")], "A")["blocks"]
    check("differing trained weights are reported as non-exact", differing["exact_parity_across_arms"] is False)
    result = {"status": "passed", "checks": checks, "synthetic_only": True, "provenance": process_provenance()}
    (output / "result.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "checks": len(checks), "synthetic_only": True}))


if __name__ == "__main__":
    main()
