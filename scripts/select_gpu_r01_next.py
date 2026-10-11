"""Freeze one continuation candidate per architecture from a fresh profile."""

from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance
from scripts.gpu_r01_training import atomic_json


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("baseline", "expertium"), required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/retraining/gpu-retrain-r01/iteration-0003"))
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    protocol = json.loads((output / "protocol.json").read_text())
    case = json.loads((output / args.model / "F01/result.json").read_text())
    profile_path = output / "profiles" / f"{args.model}-F01/result.json"
    profile = json.loads(profile_path.read_text())
    if case["status"] != "complete" or not profile["workload"]["full_history_profile_available"]:
        raise ValueError("Selection requires a complete, sufficiently observed 100-user control")
    counters = profile["phases"]["training including user transitions"]["counters"]
    gpu = counters["nvidia_utilization.gpu"]["mean"]
    worker = counters["worker_pool_cpu_percent"]["mean"]
    wait = sum(json.loads(line)["queue_wait_seconds"] for line in (output / args.model / "F01/updates.jsonl").read_text().splitlines())
    track = "F01-F06" if args.model == "baseline" else "F01-F16"
    if args.model == "expertium" and (worker < 80 or gpu >= 80):
        raise ValueError("The frozen F16 candidate is not supported by the fresh bottleneck profile")
    reason = (f"Fresh F01 GPU activity is {gpu:.1f}% and worker-pool CPU {worker:.1f}%. " +
              ("Alex is near the GPU-activity target; remove internal NaN/count waits, consolidate metric transfers and buffer log writes, preserving finite-loss/gradient checks and loss arithmetic."
               if args.model == "baseline" else "Expertium's preparation workers are saturated while GPU activity remains below target; raise the pool from 6 to 8 workers and the bounded queue from 12 to 16 on this 8-core/16-thread host, preserving ordered batches and seeds."))
    selection_path = output / "selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    if args.model in selection:
        raise FileExistsError("Preserve the previous selection")
    selection[args.model] = {"track": track, "options": protocol["case_options"][track], "reason": reason,
                             "selected_utc": datetime.now(timezone.utc).isoformat(),
                             "profile": profile_path.relative_to(output).as_posix(),
                             "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
                             "measured_incumbent_reviews_per_second": case["reviews_per_second"],
                             "measured_future_wait_seconds_sum": wait,
                             "idealized_scenarios": {"unchanged_GPU_work_at80_percent_activity_reviews_per_second": case["reviews_per_second"] * 80 / gpu,
                                                      "remove_all_measured_future_wait_reviews_per_second": case["review_presentations"] / (case["training_seconds"] - wait),
                                                      "linear_8_vs6_worker_capacity_reviews_per_second": case["reviews_per_second"] * 8 / 6 if args.model == "expertium" else None,
                                                      "interpretation": "Conditional capacity arithmetic only; not measured candidate throughput or a forecast. CPU contention, RAM/IPC/loads and unchanged work can dominate. GPU activity is not occupancy."},
                             "minimum_available_RAM_GiB": counters["ram_available_bytes"]["min"] / 2**30,
                             "selection_provenance": process_provenance()}
    atomic_json(selection_path, selection)
    print(json.dumps({"model": args.model, "track": track, "reason": reason}))


if __name__ == "__main__":
    main()
