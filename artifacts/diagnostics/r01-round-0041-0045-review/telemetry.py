"""Summarize recorded GPU power/utilization/VRAM telemetry for R01 full runs (analysis only, no training)."""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
R = ROOT / "artifacts/retraining/gpu-retrain-r01"
ITERATIONS = ["0027", "0030", "0031", "0033", "0034", "0035", "0036", "0037", "0038",
              "0039", "0040", "0041", "0042", "0043", "0044", "0045"]


def summarize(iteration):
    results = sorted((R / f"iteration-{iteration}" / "expertium").glob("*/result.json"))
    power = sorted((R / f"iteration-{iteration}" / "expertium").glob("*/gpu_power.jsonl"))
    if not results or not power:
        return None
    result = json.loads(results[0].read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in power[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    busy = [row for row in rows if row.get("utilization.gpu", 0) >= 20]
    used = [row["memory.used"] for row in rows]
    reserved = result.get("peak_reserved_mib")
    return {
        "iteration": iteration,
        "track": result.get("track"),
        "reviews_per_second": result["reviews_per_second"],
        "training_seconds": result["training_seconds"],
        "discarded_primer_seconds": result.get("discarded_primer_seconds"),
        "validation_seconds": result.get("validation_seconds"),
        "case_elapsed_seconds": result.get("case_elapsed_seconds"),
        "busy_samples": len(busy),
        "busy_utilization_median": statistics.median(row["utilization.gpu"] for row in busy),
        "busy_power_median_w": statistics.median(row["power.draw"] for row in busy),
        "busy_watts_per_utilization_point_median": statistics.median(row["power.draw"] / row["utilization.gpu"] for row in busy),
        "memory_used_median_mib": statistics.median(used),
        "memory_used_max_mib": max(used),
        "peak_reserved_mib": reserved,
        "memory_used_median_minus_reserved_mib": statistics.median(used) - reserved if reserved else None,
    }


if __name__ == "__main__":
    out = [item for item in (summarize(i) for i in ITERATIONS) if item]
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("telemetry.json")
    target.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for item in out:
        print(f'{item["iteration"]} {item["reviews_per_second"]:>8.0f}/s W/util {item["busy_watts_per_utilization_point_median"]:.2f} '
              f'used-reserved {item["memory_used_median_minus_reserved_mib"]:>6.0f} max used {item["memory_used_max_mib"]:>6.0f}')
