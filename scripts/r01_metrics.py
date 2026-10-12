"""Compact R01 run metrics, including the machine-state drift indicators.

Reads one full case directory (`iteration-NNNN/<model>/<track>`) and, when it
exists, its resource profile. Used by r01_iterate.py and r01_close_round.py so
agents do not extract these numbers by hand.
"""

import json
from pathlib import Path


def case_metrics(case):
    """Return training/throughput/wall times plus queue, preparation, GC and profile indicators."""
    case = Path(case)
    result = json.loads((case / "result.json").read_text(encoding="utf-8"))
    metrics = {key: result.get(key) for key in ("status", "training_seconds", "reviews_per_second", "process_wall_seconds",
                                                 "validation_seconds", "discarded_primer_seconds")}
    updates = case / "updates.jsonl"
    if updates.exists():
        rows = [json.loads(line) for line in updates.read_text(encoding="utf-8").splitlines() if line.strip()]
        metrics["queue_wait_seconds"] = sum(row.get("queue_wait_seconds") or 0.0 for row in rows)
        metrics["preparation_seconds"] = sum(row.get("preparation_seconds") or 0.0 for row in rows)
    if "training_gc" in result:
        metrics["gc_seconds"] = sum(entry["seconds"] for entry in result["training_gc"].values())
    profile = case.parents[1] / "profiles" / f"{case.parent.name}-{case.name}" / "result.json"
    if profile.exists():
        phases = json.loads(profile.read_text(encoding="utf-8"))["phases"]
        training = next((value for key, value in phases.items() if key.startswith("training")), None)
        if training:
            counters = training["counters"]
            metrics["worker_cpu_percent"] = counters.get("worker_pool_cpu_percent", {}).get("mean")
            metrics["gpu_percent"] = counters.get("nvidia_utilization.gpu", {}).get("mean")
    provenance = result.get("provenance", {})
    metrics["experiment_git_commit"] = provenance.get("git_commit")
    metrics["source_archive"] = provenance.get("source_archive")
    return metrics


def drift_line(new, reference):
    """One line comparing drift indicators of a run with its reference ('new (ref)')."""
    fields = (("queue wait", "queue_wait_seconds", "{:.2f} s"), ("summed preparation", "preparation_seconds", "{:.1f} s"),
              ("validation", "validation_seconds", "{:.1f} s"), ("primer", "discarded_primer_seconds", "{:.1f} s"),
              ("GC", "gc_seconds", "{:.2f} s"), ("worker CPU", "worker_cpu_percent", "{:.1f}%"), ("GPU", "gpu_percent", "{:.1f}%"))
    parts = []
    for label, key, form in fields:
        if new.get(key) is None:
            continue
        old = reference.get(key)
        parts.append(f"{label} {form.format(new[key])}" + (f" ({form.format(old)})" if old is not None else ""))
    return "; ".join(parts) + " [new (reference)]"
