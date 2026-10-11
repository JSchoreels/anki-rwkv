"""Select a continuation candidate using an applicable recorded profile."""

from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance
from scripts.gpu_r01_training import atomic_json
from scripts.gpu_r01_policy import POLICY, find_profile, read_reference, require_active


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("baseline", "expertium"), required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/retraining/gpu-retrain-r01/iteration-0003"))
    parser.add_argument("--profile", type=Path, help="Optional replacement profile when the recorded profile is no longer applicable")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    protocol = json.loads((output / "protocol.json").read_text())
    require_active(args.model, protocol.get("prior_no_20_percent_gain_streak", {}).get(args.model, 0))
    control = protocol.get("control_tracks", {}).get(args.model, protocol.get("control_track", "F01"))
    amendment_path = output / "reference-reuse-amendment.json"
    amendment = json.loads(amendment_path.read_text()) if amendment_path.exists() else {}
    reused = (protocol.get("loop_policy") == POLICY and "references" in protocol) or args.model in amendment.get("reuse_models", [])
    if reused:
        metadata = protocol.get("retuning_candidates", {}).get(args.model, protocol["references"][args.model]["previous"])
        case, result_path = read_reference(metadata, ROOT)
        profile_path = find_profile(result_path)
    else:
        result_path = output / args.model / control / "result.json"
        case = json.loads(result_path.read_text())
        profile_path = output / "profiles" / f"{args.model}-{control}/result.json"
    if args.profile:
        profile_path = (ROOT / args.profile).resolve()
    if profile_path is None:
        raise ValueError("No applicable recorded profile; capture a focused profile with a stated reason")
    profile = json.loads(profile_path.read_text())
    workload = profile["workload"]
    applicable = workload["full_history_profile_available"] or (
        reused and workload.get("full_history_complete") and workload.get("complete_training_observation"))
    if case["status"] != "complete" or not applicable:
        raise ValueError("Selection requires a complete, sufficiently observed reference profile")
    if (profile["workload"]["logged_training_reviews"] != case["review_presentations"]
            or profile["workload"]["logged_updates"] != case["updates"]):
        raise ValueError("Profile exposure differs from the recorded reference")
    identities = [item for item in profile.get("inputs", []) if item.get("path") == result_path.relative_to(ROOT).as_posix()]
    if not identities or identities[0].get("sha256") != hashlib.sha256(result_path.read_bytes()).hexdigest():
        raise ValueError("Profile does not preserve the selected run's result identity")
    counters = profile["phases"]["training including user transitions"]["counters"]
    gpu = counters["nvidia_utilization.gpu"]["mean"]
    worker = counters["worker_pool_cpu_percent"]["mean"]
    wait = sum(json.loads(line)["queue_wait_seconds"] for line in result_path.with_name("updates.jsonl").read_text().splitlines())
    track = protocol.get("candidate_tracks", {}).get(args.model, "F01-F06" if args.model == "baseline" else "F01-F16")
    options = protocol.get("case_options_by_model", {}).get(args.model, protocol.get("case_options"))
    if track == "F01-F16" and (worker < 80 or gpu >= 80):
        raise ValueError("The F16 candidate is not supported by the recorded bottleneck profile")
    explanation = protocol.get("case_explanations", {}).get(args.model, {}).get(track)
    reason = (f"Recorded {case['track']} GPU activity is {gpu:.2f}% and worker-pool CPU {worker:.2f}%. " + (explanation if explanation else
              ("Alex is near the GPU-activity target; remove internal NaN/count waits, consolidate metric transfers and buffer log writes, preserving finite-loss/gradient checks and loss arithmetic."
               if args.model == "baseline" else "Expertium's preparation workers are saturated while GPU activity remains below target; raise the pool from 6 to 8 workers and the bounded queue from 12 to 16 on this 8-core/16-thread host, preserving ordered batches and seeds.")))
    selection_path = output / "selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    if args.model in selection:
        raise FileExistsError("Preserve the previous selection")
    selection[args.model] = {"track": track, "options": options[track], "reason": reason,
                             "selected_utc": datetime.now(timezone.utc).isoformat(),
                             "profile": Path(os.path.relpath(profile_path, output)).as_posix(),
                             "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
                             "reference_result": result_path.relative_to(ROOT).as_posix(),
                             "retuning_resource_saving_candidate": args.model in protocol.get("retuning_candidates", {}),
                             "profile_reused": reused and not args.profile,
                             "recorded_profile_sufficiency": workload["full_history_profile_available"],
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
