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
    check("consistent gains give a positive interval and promotion", clear["t_ci95_percent"][0] > 0 and clear["proposed_decision"] == "promote")
    mixed = []
    for i, (a, b) in enumerate([(100, 103), (100, 99), (100, 102)]):
        mixed += [write(f / f"mx{i}a", "A", a, 20 + 2 * i), write(f / f"mx{i}b", "B", b, 21 + 2 * i)]
    noisy = analyse(mixed, "A")["blocks"]
    check("a positive mean with an interval including zero is a stack candidate", noisy["mean_gain_percent"] > 0 and noisy["t_ci95_percent"][0] < 0 and noisy["proposed_decision"] == "stack candidate")
    worse = []
    for i, (a, b) in enumerate([(100, 98), (100, 99), (100, 101)]):
        worse += [write(f / f"wr{i}a", "A", a, 30 + 2 * i), write(f / f"wr{i}b", "B", b, 31 + 2 * i)]
    check("a negative mean is rejected", analyse(worse, "A")["blocks"]["proposed_decision"] == "reject")
    single = analyse([write(f / "one-a", "A", 100, 40), write(f / "one-b", "B", 105, 41)], "A")["blocks"]
    check("one block is not enough for a decision", single["proposed_decision"] == "needs at least two blocks")
    differing = analyse([write(f / "d-a", "A", 100, 50), write(f / "d-b", "B", 105, 51, trained="other")], "A")["blocks"]
    check("differing trained weights are reported as non-exact", differing["exact_parity_across_arms"] is False)
    # Per-update path: B is 2% faster on every update; one reference run has a start-up spike.
    import random
    rng = random.Random(7)
    sizes = [rng.choice([20000, 45000, 65536]) for _ in range(60)]
    base = [0.05 + size / 400000 for size in sizes]

    def timed(folder, track, scale, minute, spike=False):
        folder.mkdir(parents=True)
        t, lines = 0.3, []
        for k, (size, duration) in enumerate(zip(sizes, base)):
            duration *= scale * (1 + rng.gauss(0, 0.003)) * (4 if spike and k < 3 else 1)
            lines.append(json.dumps({"step": k + 1, "user_id": 1, "offset": k, "reviews": size, "seed": k, "loss": 0.5,
                                     "gradient_norm": 1.0, "host_started_s": t, "gpu_ms": duration * 900}))
            t += duration
        (folder / "updates.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (folder / "result.json").write_text(json.dumps({"track": track, "reviews_per_second": sum(sizes) / t, "training_seconds": t,
                                                        "trained_parameter_sha256": "same",
                                                        "training_started_utc": f"2026-10-13T01:{minute:02d}:00+00:00"}), encoding="utf-8")
        return folder

    blocks = []
    for i in range(2):
        blocks += [timed(f / f"tb{i}a", "A", 1.0, 2 * i, spike=(i == 0)), timed(f / f"tb{i}b", "B", 1 / 1.02, 2 * i + 1)]
    per_update = analyse(blocks, "A", calibrated_sd_percent=1.01)["blocks"]
    check("timed blocks use the trimmed per-update estimator", per_update["estimator"].startswith("trimmed"))
    check("a start-up spike in a reference run does not inflate the trimmed block effect", all(abs(e - 2) < 0.5 for e in per_update["effects_percent"]))
    check("two blocks of a 2% gain promote under the calibrated SD", per_update["proposed_decision"] == "promote" and per_update["one_sided_95_lower_percent"] > 0)
    same = []
    for i in range(2):
        same += [timed(f / f"ts{i}a", "A", 1.0, 10 + 2 * i), timed(f / f"ts{i}b", "B", 1.0, 11 + 2 * i)]
    check("no true gain is not promoted under the calibrated SD", analyse(same, "A", calibrated_sd_percent=1.01)["blocks"]["proposed_decision"] != "promote")
    result = {"status": "passed", "checks": checks, "synthetic_only": True, "provenance": process_provenance()}
    (output / "result.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "checks": len(checks), "synthetic_only": True}))


if __name__ == "__main__":
    main()
