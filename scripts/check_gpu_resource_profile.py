"""Synthetic CPU checks of irregular time weighting and workload/counter gates."""

from copy import deepcopy
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance
from scripts.profile_gpu_resources import analyze, intervals, process_delta, snapshot, summarize


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/retraining/gpu-retrain-r01/profile-checks-v1"))
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Output must remain below workspace artifacts")
    output.mkdir(parents=True, exist_ok=False)
    provenance = process_provenance()
    checks = []

    def check(name, value):
        if not value:
            raise AssertionError(name)
        checks.append(name)

    weights = [2, 8, 0]
    values = summarize([0, 100, 100], weights)
    check("irregular intervals weighted by time, not sample count", values["mean"] == 80 and values["median"] == 100)
    missing = summarize([None, None], [1, 1])
    check("missing counters remain null", missing["mean"] is None and missing["p95"] is None)
    single = summarize([100], [0])
    check("single observation does not imply duration or occupancy", single["mean"] is None and single["max"] == 100)
    boundary = [{"elapsed_seconds": time, "phase": phase} for time, phase in ((0, "startup"), (2, "startup"), (10, "training"), (12, "training"))]
    check("phase boundaries and final sample are not extrapolated", intervals(boundary) == [2, 0, 2, 0])
    updates = [{"step": user, "user_id": user, "offset": 0, "reviews": 2, "preparation_seconds": 0.2, "queue_wait_seconds": 0.1} for user in range(1, 501)]
    cohort = {"rows": [{"user_id": user, "history_decile": (user - 1) // 50 + 1, "train_range": [0, 2]} for user in range(1, 501)], "training_reviews": 1000}
    case = {"config": {"PREFETCH_WORKERS": 2}, "status": "complete", "evaluation_users": 500,
            "model": "synthetic", "track": "control", "provenance": {"synthetic": True},
            "review_presentations": 1000, "updates": 500, "progress_users": 500}
    case["training_seconds"] = 10
    samples = []
    for index, time in enumerate((0, 2, 10)):
        processes = [{"pid": 1, "parent_pid": 999, "name": "python.exe", "cpu_percent_one_core_100": 10,
                      "io_read_bytes": index * 10, "io_write_bytes": index * 20, "page_faults": index * 2},
                     {"pid": 2, "parent_pid": 1, "name": "python.exe", "cpu_percent_one_core_100": 80,
                      "io_read_bytes": index * 10, "io_write_bytes": index * 20, "page_faults": index * 2},
                     {"pid": 3, "parent_pid": 1, "name": "python.exe", "cpu_percent_one_core_100": 20 if index != 1 else None,
                      "io_read_bytes": index * 10, "io_write_bytes": index * 20, "page_faults": index * 2}]
        samples.append({"elapsed_seconds": time, "phase": "training including user transitions", "processes": processes,
                        "cpu_per_core_percent": [25, 50], "nvidia_utilization.gpu": 0 if index == 0 else 100,
                        "nvidia_utilization.memory": None, "ram_available_bytes": 4 * 1024 ** 3,
                        "process_tree_rss_bytes": 3 * 1024 ** 3, "unavailable_counters": {"paging": "unsupported"}})
    result = analyze(case, cohort, samples, updates, {"samples": 3}, minimum_seconds=5)
    phase = result["phases"]["training including user transitions"]
    check("GPU idle-time fraction uses observed intervals", phase["gpu_below20_percent_time_fraction"] == 0.2)
    check("incomplete worker CPU samples are unknown, not zero", phase["counters"]["worker_pool_cpu_percent"]["mean"] == 50 and phase["counters"]["worker_pool_cpu_percent"]["observed_seconds"] == 2)
    check("complete coverage and long-enough observation recognized", result["workload"]["full_history_profile_available"])
    small_cohort = {"rows": cohort["rows"][:80], "training_reviews": 160, "evaluation_users": 20}
    small_case = {**case, "training_users": 80, "evaluation_users": 20, "review_presentations": 160, "updates": 80, "progress_users": 80}
    check("disjoint 80-training/20-evaluation cohort recognized", analyze(small_case, small_cohort, samples, updates[:80], {"samples": 3}, 5)["workload"]["full_history_profile_available"])
    check("small-cohort partial exposure rejected", not analyze(small_case, small_cohort, samples, updates[:79], {"samples": 3}, 5)["workload"]["full_history_profile_available"])
    check("small-cohort evaluation metadata mismatch rejected", not analyze({**small_case, "evaluation_users": 80}, small_cohort, samples, updates[:80], {"samples": 3}, 5)["workload"]["full_history_profile_available"])
    packed = [{**updates[0], "reviews": 160, "parts": [[user, 0, 2] for user in range(1, 81)]}]
    packed_case = {**small_case, "updates": 1}
    packed_result = analyze(packed_case, small_cohort, samples, packed, {"samples": 3}, 5)
    check("packed batches preserve all 80 users and review exposure", packed_result["workload"]["full_history_profile_available"])
    check("packed update counted once in each participating decile", packed_result["workload"]["history_deciles"]["1"]["updates"] == 1 and packed_result["workload"]["history_deciles"]["2"]["reviews"] == 60)
    for name, mutate in (
            ("packed part total mismatch rejected", lambda row: row.update(reviews=159)),
            ("empty packed parts rejected", lambda row: row.update(parts=[])),
            ("unknown packed user rejected", lambda row: row["parts"][1].__setitem__(0, 999)),
            ("packed per-user offset gap rejected", lambda row: row["parts"][1].__setitem__(1, 1)),
            ("duplicate packed exposure rejected", lambda row: row["parts"][1].__setitem__(0, 1))):
        invalid = deepcopy(packed)
        mutate(invalid[0])
        try:
            analyze(packed_case, small_cohort, samples, invalid, {"samples": 3}, 5)
        except ValueError:
            check(name, True)
        else:
            raise AssertionError(name)
    changed = deepcopy(case)
    changed["status"] = "running"
    check("live prefixes do not establish full-history saturation", not analyze(changed, cohort, samples, updates, None, 5)["workload"]["full_history_profile_available"])
    changed["status"] = "evaluating"
    evaluating = analyze(changed, cohort, samples, updates, None, 5)
    check("finished training distinguished from unfinished evaluation", evaluating["workload"]["complete_training_observation"] and not evaluating["workload"]["full_history_profile_available"])
    changed = deepcopy(case)
    changed["smoke_only"] = True
    check("smokes do not establish full-history saturation", not analyze(changed, cohort, samples, updates, {"samples": 3}, 5)["workload"]["full_history_profile_available"])
    check("short duration fails adequacy even at full coverage", not analyze(case, cohort, samples, updates, {"samples": 3}, 120)["workload"]["full_history_profile_available"])
    changed = deepcopy(case)
    changed["updates"] = 1
    check("raw case/update metadata must agree", not analyze(changed, cohort, samples, updates, {"samples": 3}, 5)["workload"]["full_history_profile_available"])
    check("a single I/O snapshot is not a measured zero delta", process_delta(samples[:1], "io_read_bytes")["delta"] is None)
    check("actual zero I/O differences can be measured", process_delta([samples[0], samples[0]], "io_read_bytes")["delta"] == 0)
    damaged = deepcopy(updates)
    damaged[1]["offset"] = 1
    try:
        analyze(case, cohort, samples, damaged, None)
    except ValueError:
        check("coverage gaps rejected", True)
    else:
        raise AssertionError("coverage gap accepted")
    tail = output / "live-tail.jsonl"
    tail.write_bytes(b'{"sample":1}\n{"unfinished":')
    parsed, metadata = snapshot(tail, lines=True)
    check("unfinished live tail explicitly excluded", parsed == [{"sample": 1}] and metadata["ignored_unfinished_tail"])
    (output / "result.json").write_text(json.dumps({"status": "passed", "synthetic_only": True, "checks": checks, "provenance": provenance}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "synthetic_only": True, "checks": len(checks)}))


if __name__ == "__main__":
    main()
