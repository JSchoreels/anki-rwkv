"""Synthetic checks for scripts/r01_paired_timing.py (CPU only, no training)."""
from argparse import ArgumentParser
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance
from scripts.r01_paired_timing import compare, load_run


def write_case(directory, sizes, base, scale, spike=None, loss=0.5, timing=True):
    directory.mkdir(parents=True)
    t, lines = 0.3, []
    for k, (size, duration) in enumerate(zip(sizes, base)):
        duration = duration * scale * (3 if k == spike else 1)
        record = {"step": k + 1, "user_id": 1, "offset": k, "reviews": size, "seed": k, "loss": loss, "gradient_norm": 1.0}
        if timing:
            record.update(host_started_s=t, host_dispatch_s=duration / 2, gpu_started_ms=t * 1000, gpu_ms=duration * 800)
        lines.append(json.dumps(record))
        t += duration
    (directory / "updates.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (directory / "result.json").write_text(json.dumps({"training_seconds": t}), encoding="utf-8")
    return directory


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        parser.error("Use a new workspace artifact directory")
    fixtures = output / "fixtures"
    rng = random.Random(1)
    sizes = [rng.choice([20000, 45000, 65536]) for _ in range(40)]
    base = [0.05 + size / 400000 for size in sizes]
    a1 = load_run(write_case(fixtures / "a1", sizes, base, 1.0))
    a2 = load_run(write_case(fixtures / "a2", sizes, base, 1.0, spike=7))
    b1 = load_run(write_case(fixtures / "b1", sizes, base, 1 / 1.05))
    changed = load_run(write_case(fixtures / "changed", sizes, base, 1.0, loss=0.6))
    checks = []

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})
        if not condition:
            raise AssertionError(name)

    clean = compare([a1], [b1], resamples=2000)
    check("uniform 5% speed-up reads +5% total and median", abs(clean["wall"]["gain_percent"] - 5) < 1e-6 and abs(clean["wall"]["median_update_gain_percent"] - 5) < 1e-6)
    check("device spans pair the same way", abs(clean["gpu"]["gain_percent"] - 5) < 1e-6)
    check("eight equal-work segments are reported", len(clean["wall"]["segment_gain_percent"]) == 8)
    spiked = compare([a2], [b1], resamples=2000)
    check("a reference spike inflates the total but not the median", spiked["wall"]["gain_percent"] > 8 and abs(spiked["wall"]["median_update_gain_percent"] - 5) < 1e-6)
    check("the spike widens the update-resampling interval", spiked["wall"]["gain_ci95_percent"][1] - spiked["wall"]["gain_ci95_percent"][0] > 1)
    best = compare([a1, a2], [b1], resamples=2000)
    check("best-of across reference runs removes the spike", abs(best["wall"]["gain_percent"] - 5) < 1e-6)
    check("identical update values are recognised as exact", clean["exact_update_values"])
    check("changed losses are reported as non-exact", not compare([a1], [changed], resamples=200)["exact_update_values"])
    shorter = write_case(fixtures / "shorter", sizes[:-1], base[:-1], 1.0)
    try:
        compare([a1], [load_run(shorter)], resamples=10)
    except ValueError:
        check("a different update sequence is refused", True)
    else:
        check("a different update sequence is refused", False)
    untimed = write_case(fixtures / "untimed", sizes, base, 1.0, timing=False)
    try:
        load_run(untimed)
    except ValueError:
        check("runs without per-update timing are refused", True)
    else:
        check("runs without per-update timing are refused", False)
    result = {"status": "passed", "checks": checks, "synthetic_only": True, "provenance": process_provenance()}
    (output / "result.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "checks": len(checks), "synthetic_only": True}))


if __name__ == "__main__":
    main()
