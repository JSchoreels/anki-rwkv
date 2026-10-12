"""Offline R01 resource profiles; no training, CUDA initialization or promotion.

Profile preserved B01 telemetry without changing its live source. Time weights
describe observed within-phase sampling intervals, not continuous occupancy.
"""

from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance
from scripts.gpu_r01_policy import reusable_profile

GIB = 1024 ** 3


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def snapshot(path, lines=False, optional=False):
    if not path.exists():
        if optional:
            return None, {"path": path.relative_to(ROOT).as_posix(), "status": "unavailable"}
        raise FileNotFoundError(path)
    payload = path.read_bytes()
    metadata = {"path": path.relative_to(ROOT).as_posix(), "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(), "ignored_unfinished_tail": False}
    if not lines:
        return json.loads(payload), metadata
    if payload and not payload.endswith(b"\n"):
        # Writers flush complete JSON lines, but a live read can end mid-write.
        # Preserve the hash and explicitly exclude just that unfinished tail.
        payload = payload.rsplit(b"\n", 1)[0] if b"\n" in payload else b""
        metadata["ignored_unfinished_tail"] = True
    return [json.loads(line) for line in payload.splitlines() if line.strip()], metadata


def intervals(samples):
    weights = []
    for index, row in enumerate(samples):
        if not finite(row.get("elapsed_seconds")):
            raise ValueError("Telemetry elapsed time is missing or nonfinite")
        following = samples[index + 1] if index + 1 < len(samples) else None
        if following and following["elapsed_seconds"] <= row["elapsed_seconds"]:
            raise ValueError("Telemetry times must increase")
        weights.append(following["elapsed_seconds"] - row["elapsed_seconds"]
                       if following and following["phase"] == row["phase"] else 0.0)
    return weights


def summarize(values, weights):
    observed = [(float(value), weight) for value, weight in zip(values, weights) if finite(value)]
    weighted = sorted((value, weight) for value, weight in observed if weight > 0)
    duration = sum(weight for _, weight in weighted)
    if not observed:
        return {"status": "unavailable", "samples": 0, "observed_seconds": 0,
                "mean": None, "p10": None, "median": None, "p95": None, "min": None, "max": None}

    def quantile(fraction):
        if not duration:
            return None
        threshold, accumulated = fraction * duration, 0.0
        for value, weight in weighted:
            accumulated += weight
            if accumulated >= threshold:
                return value
        return weighted[-1][0]

    return {"status": "observed" if duration else "single sample; no interval estimate", "samples": len(observed),
            "observed_seconds": duration, "mean": sum(value * weight for value, weight in weighted) / duration if duration else None,
            "p10": quantile(0.1), "median": quantile(0.5), "p95": quantile(0.95),
            "min": min(value for value, _ in observed), "max": max(value for value, _ in observed)}


def process_delta(samples, field):
    observed, previous, resets = {}, {}, 0
    total, comparable_intervals = 0, 0
    for row in samples:
        for process in row["processes"]:
            value = process.get(field)
            if not finite(value):
                continue
            pid = process["pid"]
            observed[pid] = True
            if pid in previous:
                change = value - previous[pid]
                if change >= 0:
                    total += change
                    comparable_intervals += 1
                else:
                    resets += 1
            previous[pid] = value
    return {"delta": total if comparable_intervals else None, "observed_pids": len(observed), "counter_resets": resets,
            "comparable_intervals": comparable_intervals,
            "scope": "within observed samples/process lifetimes; first-seen counters excluded"}


def phase_profile(samples, weights, expected_workers):
    workers, known_pool = [], []
    process_values = defaultdict(list)
    all_pids = sorted({process["pid"] for row in samples for process in row["processes"]})
    for row in samples:
        root = row["processes"][0]["pid"] if row["processes"] else None
        selected = [p for p in row["processes"] if p["parent_pid"] == root and p["name"].lower() in ("python.exe", "python", "python3")]
        workers.append(len(selected))
        complete_pool = len(selected) == expected_workers and all(finite(p.get("cpu_percent_one_core_100")) for p in selected)
        known_pool.append(sum(p["cpu_percent_one_core_100"] for p in selected) / expected_workers if complete_pool and expected_workers else None)
        lookup = {p["pid"]: p for p in row["processes"]}
        for pid in all_pids:
            process_values[pid].append(lookup.get(pid, {}).get("cpu_percent_one_core_100"))
    counters = {key: summarize([row.get(key) for row in samples], weights) for key in (
        "nvidia_utilization.gpu", "nvidia_utilization.memory", "nvidia_memory.used",
        "ram_available_bytes", "ram_used_percent", "process_tree_rss_bytes",
        "torch_cuda_allocated_bytes", "torch_cuda_reserved_bytes", "torch_cuda_peak_reserved_bytes", "swap_used_bytes")}
    core_count = max((len(row["cpu_per_core_percent"]) for row in samples), default=0)
    per_core = {str(core): summarize([row["cpu_per_core_percent"][core] if core < len(row["cpu_per_core_percent"]) else None for row in samples], weights)
                for core in range(core_count)}
    counters["worker_pool_cpu_percent"] = summarize(known_pool, weights)
    counters["observed_worker_count"] = summarize(workers, weights)
    gpu_values = [(row.get("nvidia_utilization.gpu"), weight) for row, weight in zip(samples, weights) if finite(row.get("nvidia_utilization.gpu")) and weight > 0]
    gpu_seconds = sum(weight for _, weight in gpu_values)
    return {"samples": len(samples), "observed_interval_seconds": sum(weights), "counters": counters,
            "gpu_below20_percent_time_fraction": sum(weight for value, weight in gpu_values if value < 20) / gpu_seconds if gpu_seconds else None,
            "per_core_cpu_percent": per_core, "per_pid_cpu_percent_one_core100": {str(pid): summarize(values, weights) for pid, values in process_values.items()},
            "process_io_read_bytes": process_delta(samples, "io_read_bytes"), "process_io_write_bytes": process_delta(samples, "io_write_bytes"),
            "process_page_faults": process_delta(samples, "page_faults")}


def analyze(case, protocol, samples, updates, sampler_summary, minimum_seconds=120):
    expected_workers = case["config"]["PREFETCH_WORKERS"]
    weights = intervals(samples)
    phases = {}
    for name in dict.fromkeys(row["phase"] for row in samples):
        selected = [(row, weight) for row, weight in zip(samples, weights) if row["phase"] == name]
        phases[name] = phase_profile([row for row, _ in selected], [weight for _, weight in selected], expected_workers)
    cohort = {row["user_id"]: row for row in protocol["rows"]}
    coverage = defaultdict(int)
    deciles = {str(index): {"logged_users": set(), "completed_users": 0, "reviews": 0, "updates": 0} for index in range(1, 11)}
    for expected_step, update in enumerate(updates, 1):
        parts = update.get("parts", [[update["user_id"], update["offset"], update["reviews"]]])
        if (update["step"] != expected_step or not isinstance(parts, list) or not parts
                or any(not isinstance(part, (list, tuple)) or len(part) != 3
                       or any(type(value) is not int for value in part) or part[1] < 0 or part[2] <= 0 for part in parts)
                or list(parts[0][:2]) != [update["user_id"], update["offset"]]
                or sum(part[2] for part in parts) != update["reviews"]):
            raise ValueError("Saved update has invalid packed parts, update gap or review total")
        touched_deciles = set()
        for user, offset, reviews in parts:
            if user not in cohort or offset != coverage[user]:
                raise ValueError("Saved updates have an unknown user, update gap or review gap")
            coverage[user] += reviews
            decile = str(cohort[user]["history_decile"])
            group = deciles[decile]
            group["logged_users"].add(user)
            group["reviews"] += reviews
            touched_deciles.add(decile)
        for decile in touched_deciles:
            # A mixed update participates in each affected decile once.
            deciles[decile]["updates"] += 1
    for user, count in coverage.items():
        row = cohort[user]
        expected = row["train_range"][1] - row["train_range"][0]
        if count > expected:
            raise ValueError("Logged review coverage exceeds the recorded cohort")
        if count == expected:
            deciles[str(row["history_decile"])]["completed_users"] += 1
    for group in deciles.values():
        group["logged_users"] = len(group["logged_users"])
    training = phases.get("training including user transitions", {})
    window = training.get("observed_interval_seconds", 0)
    training_complete = (case["status"] in ("evaluating", "complete") and not case.get("smoke_only")
                         and finite(case.get("training_seconds")) and case["training_seconds"] > 0
                         and case["evaluation_users"] == protocol.get("evaluation_users", len(cohort))
                         and case.get("training_users", len(cohort)) == len(cohort)
                         and len(coverage) == len(cohort) and sum(coverage.values()) == protocol["training_reviews"]
                         and case.get("review_presentations") == sum(coverage.values()) and case.get("updates") == len(updates)
                         and case.get("progress_users") == len(cohort))
    complete = (training_complete and case["status"] == "complete" and sampler_summary is not None
                and sampler_summary["samples"] == len(samples))
    all_unavailable = {key: value for sample in samples for key, value in sample.get("unavailable_counters", {}).items()}
    return {"model": case["model"], "track": case["track"], "case_status": case["status"],
            "scope": ("complete-case telemetry" if complete else "complete training coverage; evaluation/case profile still pending"
                      if training_complete else "partial/short telemetry; not a full-history saturation or speed result"),
            "training_provenance": case["provenance"], "implementation_sha256": case.get("implementation_sha256"),
            "expected_preparation_workers": expected_workers, "phases": phases,
            "workload": {"planned_training_users": len(cohort), "planned_evaluation_users": protocol.get("evaluation_users", len(cohort)),
                         "logged_updates": len(updates), "logged_training_reviews": sum(coverage.values()), "logged_users": len(coverage),
                         "history_deciles": deciles, "full_history_complete": complete,
                         "complete_training_observation": training_complete,
                         "training_observed_seconds": window, "minimum_observation_seconds": minimum_seconds,
                         "observation_window_sufficient": window >= minimum_seconds,
                         "full_history_profile_available": complete and window >= minimum_seconds},
            "queue": {"preparation_seconds_sum_across_parallel_workers": sum(row["preparation_seconds"] for row in updates),
                      "preparation_seconds_per_update": summarize([row["preparation_seconds"] for row in updates], [1] * len(updates)),
                      "future_wait_seconds_sum": sum(row["queue_wait_seconds"] for row in updates),
                      "future_wait_seconds_per_update": summarize([row["queue_wait_seconds"] for row in updates], [1] * len(updates)),
                      "interpretation": "worker preparation durations can overlap; future waits may occur on background staging threads; sums are not critical-path stall time"},
            "sampler": sampler_summary,
            "unavailable": {**all_unavailable, "actual_GPU_bytes_per_second": "not sampled", "physical_disk_IO": "process I/O includes IPC/device traffic; physical disk counters not sampled",
                            "Windows_swap_in_out_bytes": "installed psutil Windows swap_memory returns literal zero placeholders; treat as unavailable, not measured zero",
                            "precise_steady_vs_user_transition": "current telemetry combines training and user transitions; exact phase timestamps not recorded",
                            "kernel_occupancy": "not sampled"},
            "interpretation": "per-core CPU is system-wide; summed process RSS can double-count shared pages; page faults include soft faults; memory activity percent is not achieved bandwidth; prefix profiles do not select or promote a candidate"}


def render(output, result):
    def number(value, digits=1, divisor=1):
        return f"{value / divisor:,.{digits}f}" if finite(value) else "—"
    lines = [f"# Resource profile: {result['model']}/{result['track']}", "", f"Case status: **{result['case_status']}**. Scope: **{result['scope']}**.", "",
             "Means/quantiles use observed within-phase sampling intervals. Phase boundaries and the final sample carry no extrapolated interval.", "",
             "| Phase | Samples | Observed s | GPU compute mean / p95 % | GPU memory activity mean % | Worker pool CPU mean % | Minimum available RAM GiB | Maximum summed RSS GiB | Peak reserved VRAM GiB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, phase in result["phases"].items():
        c = phase["counters"]
        lines.append(f"| {name} | {phase['samples']} | {number(phase['observed_interval_seconds'])} | {number(c['nvidia_utilization.gpu']['mean'])} / {number(c['nvidia_utilization.gpu']['p95'])} | {number(c['nvidia_utilization.memory']['mean'])} | {number(c['worker_pool_cpu_percent']['mean'])} | {number(c['ram_available_bytes']['min'], 2, GIB)} | {number(c['process_tree_rss_bytes']['max'], 2, GIB)} | {number(c['torch_cuda_peak_reserved_bytes']['max'], 2, GIB)} |")
    w = result["workload"]
    lines += ["", f"Logged coverage: **{w['logged_users']}/{w['planned_training_users']} training users**, **{w['logged_updates']:,} updates**, **{w['logged_training_reviews']:,} reviews**. Held-out cohort: **{w['planned_evaluation_users']} users**.",
              f"Complete training observation: **{w['complete_training_observation']}**. Resource telemetry does not establish predictive quality.",
              f"Full-history profile available: **{w['full_history_profile_available']}**; observed training window: **{number(w['training_observed_seconds'])} s**.", "",
              "| History decile | Logged users | Completed training histories | Reviews | Updates |", "|---|---:|---:|---:|---:|"]
    for decile, values in w["history_deciles"].items():
        lines.append(f"| {decile} | {values['logged_users']} | {values['completed_users']} | {values['reviews']:,} | {values['updates']} |")
    lines += ["", "Packed updates count once in each participating history decile; decile update counts need not sum to the optimizer update total."]
    q = result["queue"]
    lines += ["", f"Summed preparation across workers: {number(q['preparation_seconds_sum_across_parallel_workers'])} s; summed future waits: {number(q['future_wait_seconds_sum'])} s.",
              q["interpretation"], "", result["interpretation"], "", "Unavailable/limited counters:", "",
              *[f"- {key}: {value}." for key, value in result["unavailable"].items()], ""]
    if result["sampler"]:
        s = result["sampler"]
        lines += [f"Sampler wall: {number(s['sampling_wall_seconds'], 3)} s; sampler-thread CPU: {number(s.get('sampling_thread_cpu_seconds'), 3)} s. This does not estimate critical-path slowdown.", ""]
    else:
        lines += ["Sampler overhead: — (sampler still running or summary unavailable).", ""]
    lines += ["Source evidence:", ""]
    for item in result["inputs"]:
        link = os.path.relpath(ROOT / item["path"], output).replace("\\", "/")
        lines.append(f"- [{item['path']}]({link}): {item.get('sha256', item.get('status'))}.")
    (output / "profile.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("artifacts/benchmarks/gpu-baseline-b01/protocol.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-seconds", type=float, default=120)
    parser.add_argument("--repeat-reason", help="Why the existing profile does not answer this analysis question")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts") or args.minimum_seconds <= 0:
        parser.error("Use a new artifact directory and positive observation minimum")
    previous = reusable_profile((ROOT / args.case).resolve() / "result.json", ROOT, args.minimum_seconds)
    if previous and not (args.repeat_reason or "").strip():
        print(json.dumps({"action": "reuse profile", "profile": previous.relative_to(ROOT).as_posix()}))
        return
    if output.exists():
        parser.error("Preserve the previous profile; use a new artifact directory")
    output.mkdir(parents=True)
    started = time.perf_counter()
    report = {"status": "running", "analysis_provenance": process_provenance(), "captured_utc": datetime.now(timezone.utc).isoformat(), "inputs": [],
              "repeat_reason": args.repeat_reason}
    try:
        folder = (ROOT / args.case).resolve()
        values = []
        for path, lines, optional in ((folder / "result.json", False, False), ((ROOT / args.protocol).resolve(), False, False),
                                      (folder / "resources.jsonl", True, False), (folder / "updates.jsonl", True, False),
                                      (folder / "resources-summary.json", False, True)):
            value, metadata = snapshot(path, lines=lines, optional=optional)
            report["inputs"].append(metadata)
            values.append(value)
        report.update(analyze(*values, minimum_seconds=args.minimum_seconds))
        report["status"] = "profiled; not quality validation or promotion"
        render(output, report)
    except Exception as exc:
        report.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        report["analysis_wall_seconds"] = time.perf_counter() - started
        (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "scope": report["scope"], "workload": report["workload"]}, default=list))


if __name__ == "__main__":
    main()
