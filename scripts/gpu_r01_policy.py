"""User-approved R01 references, measured-loss decisions and stopping rules."""

import hashlib
import json
import math
from pathlib import Path

POLICY = "reused-references-v1"
LOSS_METRICS = tuple(task + "_log_loss_" + kind for task in ("curve", "instant")
                     for kind in ("label_weighted", "user_average"))


def result_metadata(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root / "artifacts"):
        raise ValueError("References must remain under workspace artifacts")
    return {"path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_reference(metadata, root):
    path = (Path(root) / metadata["path"]).resolve()
    if result_metadata(path, root)["sha256"] != metadata["sha256"]:
        raise ValueError("Frozen historical reference changed")
    case = json.loads(path.read_text(encoding="utf-8"))
    if case.get("status") != "complete" or case.get("smoke_only"):
        raise ValueError("References require completed full-history runs")
    return case, path


def accepted_reference(output, model, root):
    """Follow retained references when a candidate was rejected or fastmode-only."""
    output = Path(output).resolve()
    report = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    comparison = report["comparisons"][model]
    if comparison.get("accepted_reference"):
        metadata = comparison["accepted_reference"]
        read_reference(metadata, root)
        return metadata
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    track = comparison.get("incumbent_after_iteration")
    if comparison.get("fastmode"):
        metadata = protocol["references"][model]["previous"]
        read_reference(metadata, root)
        return metadata
    if track is None:
        # Historical iteration 0002 recorded acceptance in its decision text.
        track = comparison["track"] if comparison["decision"].startswith("promising preliminary") else "control"
    path = output / model / track / "result.json"
    if path.exists():
        metadata = result_metadata(path, root)
        read_reference(metadata, root)
        return metadata
    metadata = protocol["references"][model]["previous"]
    read_reference(metadata, root)
    return metadata


def streak(comparison):
    return comparison.get("no_20_percent_gain_streak", comparison.get("no_20_percent_gain_streak_after_this_iteration", 0))


def require_active(model, no_gain_streak):
    if no_gain_streak >= 3:
        raise RuntimeError(f"{model}: stop after {no_gain_streak} consecutive iterations without a 20% gain")


def point_losses_pass(metrics, relative_limit):
    """The loss ceiling applies to measured deltas, not confidence bounds."""
    if not math.isfinite(relative_limit) or not 0 <= relative_limit < 1:
        raise ValueError("Finite relative loss ceiling required")
    if set(metrics) != set(LOSS_METRICS):
        raise ValueError("All four binary loss metrics are required")
    for metric in metrics.values():
        change = metric["relative_delta"]
        if not math.isfinite(change):
            raise ValueError("Finite measured loss deltas required")
        if change > relative_limit and not math.isclose(change, relative_limit, rel_tol=0, abs_tol=1e-12):
            return False
    return True


def decide(gain_percent, losses, *, relative_limit, baseline_limit, prior_streak, resources_reduced=False):
    if type(gain_percent) not in (int, float) or not math.isfinite(gain_percent):
        raise ValueError("Finite measured throughput gain required")
    metrics = losses["previous"]["metrics"]
    accurate = point_losses_pass(metrics, relative_limit)
    baseline_preferred = point_losses_pass(losses["original_baseline"]["metrics"], baseline_limit)
    worse = any(value["relative_delta"] > 0 for value in metrics.values())
    faster = gain_percent > 0
    fastmode = faster and (worse or not accurate)
    retune = not faster and accurate and not worse and resources_reduced
    accepted = faster and accurate and not fastmode
    decision = ("retain separate opt-in fastmode" if fastmode else
                "retain main improvement; preliminary only" if accepted else
                "investigate retuning freed resources" if retune else
                "retain previous accepted run; speed target missed")
    return {"faster": faster, "passes_preliminary_quality": accurate,
            "baseline_preferred_limit_met": baseline_preferred,
            "fastmode": fastmode, "accepted_main": accepted,
            "resource_retuning_pending": retune, "decision": decision,
            "meets_20_percent_target": gain_percent >= 20,
            "no_20_percent_gain_streak": 0 if gain_percent >= 20 else prior_streak + 1}


def find_profile(result_path):
    result_path = Path(result_path)
    profiles = result_path.parents[2] / "profiles"
    name = result_path.parent.parent.name + "-" + result_path.parent.name
    for folder in (name + "-v2", name):
        path = profiles / folder / "result.json"
        if path.exists():
            return path
    return None


def resource_reductions(candidate_path, reference_path, root):
    """Descriptive evidence for investigating retuning, not a speed claim."""
    paths = [find_profile(path) for path in (candidate_path, reference_path)]
    if any(path is None for path in paths):
        return {}
    profiles = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for result_path, profile in zip((candidate_path, reference_path), profiles):
        metadata = result_metadata(result_path, root)
        workload = profile["workload"]
        case = json.loads(Path(result_path).read_text(encoding="utf-8"))
        if (not workload.get("full_history_complete") or not workload.get("complete_training_observation")
                or workload.get("logged_training_reviews") != case["review_presentations"]
                or workload.get("logged_updates") != case["updates"]
                or not any(item.get("path") == metadata["path"] and item.get("sha256") == metadata["sha256"]
                           for item in profile.get("inputs", []))):
            return {}
    counters = [p["phases"]["training including user transitions"]["counters"] for p in profiles]
    reductions = {}
    for key, statistic in (("nvidia_utilization.gpu", "mean"),
                           ("torch_cuda_peak_reserved_bytes", "max"),
                           ("process_tree_rss_bytes", "max")):
        values = [c.get(key, {}).get(statistic) for c in counters]
        if all(type(v) in (int, float) and math.isfinite(v) for v in values) and values[0] < values[1]:
            reductions[key] = {"candidate": values[0], "reference": values[1]}
    # Pool percentages are normalized by worker count; compare absolute worker load.
    values = [c.get("worker_pool_cpu_percent", {}).get("mean") for c in counters]
    workers = [c.get("observed_worker_count", {}).get("mean") for c in counters]
    if all(type(v) in (int, float) and math.isfinite(v) for v in values + workers):
        absolute = [value * count for value, count in zip(values, workers)]
        if absolute[0] < absolute[1]:
            reductions["worker_cpu_load_proxy"] = {"candidate": absolute[0], "reference": absolute[1]}
    return reductions
