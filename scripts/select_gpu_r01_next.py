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
from scripts.gpu_r01_policy import POLICY, FEEDING_TRACKS, find_profile, read_reference, require_active, reusable_profile, resolve_case
from scripts.report_gpu_r01_next import describe


def cpu_evidence(resources_path):
    """Worker-pool percentages hide main-process load and physical-core saturation."""
    import psutil
    if not resources_path.exists():
        return None
    samples = [json.loads(line) for line in resources_path.read_text(encoding="utf-8").splitlines()]
    samples = [s for s in samples if s.get("phase") == "training including user transitions" and s.get("processes")]
    if not samples:
        return None
    main, busy = [], []
    for sample in samples:
        pids = {item["pid"] for item in sample["processes"]}
        roots = [item for item in sample["processes"] if item.get("parent_pid") not in pids]
        if len(roots) == 1:
            main.append(roots[0]["cpu_percent_one_core_100"])
        busy.append(sum(sample["cpu_per_core_percent"]) / 100)
    physical = psutil.cpu_count(logical=False)
    logical = len(samples[0]["cpu_per_core_percent"])
    busy_mean = sum(busy) / len(busy)
    return {"main_process_cpu_percent_one_core_mean": sum(main) / len(main) if main else None,
            "busy_logical_threads_mean": busy_mean, "logical_threads": logical, "physical_cores": physical,
            "near_physical_core_saturation": physical is not None and busy_mean >= 0.8 * physical,
            "samples": len(samples),
            "interpretation": "Sampled at the profiler interval; physical cores read on the selecting host, which is assumed to be the training host. SMT threads add less than a full core."}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("baseline", "expertium"), required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/retraining/gpu-retrain-r01/iteration-0003"))
    parser.add_argument("--profile", type=Path, help="Optional replacement profile when the recorded profile is no longer applicable")
    parser.add_argument("--ceiling-justification", help="Why a feeding-only candidate whose recorded wait ceiling is below the target is still worth a screen")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    protocol = json.loads((output / "protocol.json").read_text())
    require_active(args.model, protocol.get("prior_no_20_percent_gain_streak", {}).get(args.model, 0), protocol)
    control = protocol.get("control_tracks", {}).get(args.model, protocol.get("control_track", "F01"))
    amendment_path = output / "reference-reuse-amendment.json"
    amendment = json.loads(amendment_path.read_text()) if amendment_path.exists() else {}
    reused = (protocol.get("loop_policy") == POLICY and "references" in protocol) or args.model in amendment.get("reuse_models", [])
    if reused:
        metadata = protocol.get("retuning_candidates", {}).get(args.model, protocol["references"][args.model]["previous"])
        case, result_path = read_reference(metadata, ROOT)
        profile_path = reusable_profile(result_path, ROOT) or find_profile(result_path)
    else:
        case, result_path, reused_case = resolve_case(output, args.model, control, ROOT)
        profile_path = reusable_profile(result_path, ROOT) or output / "profiles" / f"{args.model}-{control}/result.json"
        reused = reused or reused_case
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
    cpu = cpu_evidence(result_path.with_name("resources.jsonl"))
    target = 100 * protocol.get("acceptance", {}).get("target_throughput_gain", 0.20)
    feeding_ceiling = 100 * ((case["review_presentations"] / (case["training_seconds"] - wait)) / case["reviews_per_second"] - 1)
    added = set(track.split("-")) - set(case["track"].split("-"))
    feeding_only = bool(added) and added <= FEEDING_TRACKS
    justification = (args.ceiling_justification or "").strip()
    if feeding_only and feeding_ceiling < target and not justification:
        raise ValueError(f"{track} only targets input feeding, but removing all recorded preparation waits would gain "
                         f"{feeding_ceiling:.2f}%, below the {target:.0f}% target; bundle more work or pass --ceiling-justification")
    explanation = protocol.get("case_explanations", {}).get(args.model, {}).get(track)
    cpu_text = ""
    if cpu:
        cpu_text = (f"Main process CPU averaged {cpu['main_process_cpu_percent_one_core_mean']:.0f}% of one core; system load averaged "
                    f"{cpu['busy_logical_threads_mean']:.1f} busy logical threads on {cpu['physical_cores']} physical cores. ")
    reason = (f"Recorded {describe(case['track'])} GPU activity is {gpu:.2f}% and worker-pool CPU {worker:.2f}%. " + cpu_text + (explanation if explanation else
              ("Alex is near the GPU-activity target; remove internal NaN/count waits, consolidate metric transfers and buffer log writes, preserving finite-loss/gradient checks and loss arithmetic."
               if args.model == "baseline" else "Expertium's preparation workers are saturated while GPU activity remains below target; raise the pool from 6 to 8 workers and the bounded queue from 12 to 16 on this 8-core/16-thread host, preserving ordered batches and seeds.")))
    selection_path = output / "selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    if args.model in selection:
        prior = selection[args.model]
        if (prior["track"] == track and prior["options"] == options[track]
                and prior["profile_sha256"] == hashlib.sha256(profile_path.read_bytes()).hexdigest()):
            print(json.dumps({"action": "reuse selection", "model": args.model, "track": track}))
            return
        raise FileExistsError("Preserve the previous selection; reconcile the changed evidence")
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
                             "cpu_evidence": cpu,
                             "target_reachability": {"target_gain_percent": target, "added_components": sorted(added),
                                                     "feeding_only_candidate": feeding_only,
                                                     "remove_all_waits_ceiling_gain_percent": feeding_ceiling,
                                                     "ceiling_reaches_target": feeding_ceiling >= target,
                                                     "justification": justification or None},
                             "selection_provenance": process_provenance()}
    atomic_json(selection_path, selection)
    print(json.dumps({"model": args.model, "track": track, "reason": reason}))


if __name__ == "__main__":
    main()
