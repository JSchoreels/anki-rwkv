"""User-approved R01 references, measured-loss decisions and stopping rules."""

import hashlib
from itertools import combinations
import json
import math
import re
import statistics
from pathlib import Path

POLICY = "reused-references-v1"
LOSS_METRICS = tuple(task + "_log_loss_" + kind for task in ("curve", "instant")
                     for kind in ("label_weighted", "user_average"))
# Smaller drops are within sampling noise and cannot free capacity worth retuning.
MATERIAL_REDUCTIONS = {"nvidia_utilization.gpu": ("points", 5.0),
                       "torch_cuda_peak_reserved_bytes": ("relative", 0.10),
                       "process_tree_rss_bytes": ("relative", 0.10),
                       "worker_cpu_load_proxy": ("relative", 0.10)}
# Options that openly change configuration or update schedule. Review exposure, users,
# initial weights and held-out inputs must still match; only these fields may differ.
DECLARED_CHANGES = {"batch_chunks": ({"BATCH_CHUNKS"}, {"updates", "coverage"}),
                    "pack_users": (set(), {"updates", "coverage"}),
                    "training_seed": ({"SEED"}, {"initial_parameter_sha256"}),
                    "gpu_memory_fraction": ({"GPU_MEMORY_FRACTION"}, set())}
# Track components by the bottleneck they target, for ceiling checks during selection.
FEEDING_TRACKS = {"F01", "F02", "F03", "F04", "F14", "F16", "F18", "F19", "F20", "F28", "F30"}


def round_boundary(protocol):
    boundary = protocol.get("round_start_iteration")
    if boundary is not None and (not isinstance(boundary, str) or not re.fullmatch(r"\d{4}", boundary)
                                 or boundary > protocol["iteration"] or not protocol.get("round_authorization")):
        raise ValueError("Resumed round requires its recorded user authorization and a valid iteration boundary")
    return boundary


def normalized_text(data):
    return data.replace(b"\r\n", b"\n")


def effective_loop(output, root, *, current=False):
    """Read live instructions and verify the frozen definition or its amendment."""
    output, root = Path(output).resolve(), Path(root).resolve()
    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    amendment_path = output / "reference-reuse-amendment.json"
    amendment = json.loads(amendment_path.read_text(encoding="utf-8")) if amendment_path.exists() else {}
    if amendment and amendment.get("original_protocol_sha256") != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
        raise ValueError("Loop amendment does not identify the frozen protocol")
    snapshot = output / amendment.get("loop_snapshot", "loop-definition.md")
    digest = amendment.get("loop_sha256", protocol.get("loop_sha256"))
    if not snapshot.resolve().is_relative_to(output):
        raise ValueError("Loop snapshot must remain in its iteration")
    live = root / "docs/GPU_RETRAIN_LOOP.md"
    live_digest = hashlib.sha256(live.read_bytes()).hexdigest()
    if snapshot.exists():
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen loop definition changed")
    elif digest == live_digest:
        snapshot = live
    else:
        raise ValueError("Missing effective loop definition")
    # Git line-ending conversion alone is not an instruction change.
    same_text = snapshot.exists() and normalized_text(snapshot.read_bytes()) == normalized_text(live.read_bytes())
    if current and digest != live_digest and not same_text:
        raise ValueError("LOOP changed: reconcile the protocol in a recorded amendment before another launch")
    return protocol, amendment, {"path": snapshot.relative_to(root).as_posix(), "sha256": digest,
                                 "amendment": amendment_path.relative_to(root).as_posix() if amendment else None}


def recorded_cases(root):
    base = Path(root) / "artifacts/retraining/gpu-retrain-r01"
    for output in sorted(base.glob("iteration-*")):
        if not (output / "protocol.json").exists():
            continue
        protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
        protocol = {**protocol, "_file_sha256": hashlib.sha256((output / "protocol.json").read_bytes()).hexdigest()}
        for model in ("baseline", "expertium"):
            # Uncounted timing replicates (R01-M01) are recorded runs too; they feed reuse and the noise band.
            paths = [*output.glob(f"{model}/*/result.json"),
                     *output.glob(f"replicates/*/{model}/*/result.json"),
                     *output.glob(f"functional-smoke-*/{model}/*/result.json")]
            for path in paths:
                case = json.loads(path.read_text(encoding="utf-8"))
                yield path, case, protocol


def matching_case(case, recorded_protocol, protocol, model, options, *, smoke, source=True):
    """Reuse only matching training exposure, configuration and, for candidates, code."""
    context = ("cohort_sha256", "dataset_revision", "dataset_path", "seeds", "training",
               "training_reviews", "validation_reviews", "rows", "evaluation_rows", "execution_mode")
    if any(key not in protocol or key not in recorded_protocol or protocol[key] != recorded_protocol[key] for key in context):
        return False
    if case.get("model") != model or case.get("options") != options or bool(case.get("smoke_only")) != smoke:
        return False
    if case.get("protocol_sha256") != recorded_protocol.get("_file_sha256"):
        return False
    if source and case.get("implementation_sha256") != protocol.get("implementation_sha256"):
        return False
    reviews = case.get("review_presentations")
    expected = 1024 if smoke else protocol["training_reviews"]
    if type(reviews) is not int or not (0 <= reviews <= expected if case.get("status") == "failed" else reviews == expected):
        return False
    training = protocol["training"]
    chunks, workers = training[model + "_chunks_workers"]
    workers = options.get("preparation_workers", workers)
    chunks = options.get("batch_chunks", chunks)
    expected_config = {"MODEL": model, "SEED": options.get("training_seed", protocol["seeds"][0]), "BATCH_CHUNKS": chunks,
                       "CHUNK_REVIEWS": training["chunk_reviews"], "PREFETCH_WORKERS": workers,
                       "PREFETCH_BATCHES": workers * 2, "CPU_THREADS": training["cpu_threads"],
                       "EPOCHS": training["epochs"], "DATA_PATH": protocol["dataset_path"],
                       "DATASET_REVISION": protocol["dataset_revision"], "DTYPE": "bfloat16",
                       "WKV_BACKEND": "nvrtc", "GPU_MEMORY_FRACTION": options.get("gpu_memory_fraction", training["gpu_memory_fraction"]),
                       "PEAK_LR": training["peak_lr"], "WARMUP_STEPS": training["warmup_updates"]}
    if any(case.get("config", {}).get(key) != value for key, value in expected_config.items()):
        return False
    updates = (protocol["planned_updates"][model] if "batch_chunks" not in options and not options.get("pack_users")
               else planned_updates(protocol["rows"], chunks, training["chunk_reviews"], pack_users=options.get("pack_users", False)))
    if case.get("status") == "complete" and case.get("updates") != ((1 if options.get("pack_users") else 2) if smoke else updates):
        return False
    return case.get("training_users") == (2 if smoke else len(protocol["training_users"])) \
        and case.get("evaluation_users") == (2 if smoke else len(protocol["heldout_users"]))


def launch_plan(root, output, args):
    """Live controller guard, also used for protocols lacking the new policy tag."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if not output.is_relative_to(root / "artifacts"):
        raise ValueError("Output must remain under workspace artifacts")
    protocol, amendment, definition = effective_loop(output, root, current=True)
    model, track, smoke = args.model, args.track, bool(args.smoke)
    if model not in protocol["models"]:
        raise ValueError("Model is outside the active iteration")
    options = protocol.get("case_options_by_model", {}).get(model, protocol.get("case_options"))
    if options is None:
        from scripts.run_gpu_r01 import OPTIONS
        options = OPTIONS
    if track not in options:
        raise ValueError("Track is outside the frozen iteration budget")
    for metadata in protocol.get("references", {}).get(model, {}).values():
        read_reference(metadata, root)
    control = protocol.get("control_tracks", {}).get(model, protocol.get("control_track", "control"))
    repeat_reason = (getattr(args, "repeat_reason", None) or "").strip()
    check_reason = (getattr(args, "check_reason", None) or "").strip()
    records = list(recorded_cases(root))
    matches = [(path, case) for path, case, old in records if case.get("status") == "complete"
               and matching_case(case, old, protocol, model, options[track], smoke=smoke, source=smoke or track != control)]
    # A completed full run already exercises the code a proposed smoke would check.
    if smoke and not matches:
        matches = [(path, case) for path, case, old in records if case.get("status") == "complete"
                   and matching_case(case, old, protocol, model, options[track], smoke=False)]
    if matches and not repeat_reason:
        path, case = matches[0]
        return {"action": "reuse", "result": result_metadata(path, root), "effective_loop": definition,
                "scope": "full-history" if not case.get("smoke_only") else "focused smoke"}
    prior = amendment.get("prior_no_20_percent_gain_streak", protocol.get("prior_no_20_percent_gain_streak", {}))
    require_active(model, prior.get(model, 0), protocol)
    # Older protocols must not evade a stop recorded by a later completed iteration.
    require_active(model, latest_streak(root, model, prior.get(model, 0), round_boundary(protocol)), protocol)
    if track == control and model in protocol.get("references", {}) and not repeat_reason:
        raise ValueError("Reuse the recorded accepted run; repeating a reference requires --repeat-reason")
    if smoke and not check_reason:
        raise ValueError("A new smoke requires --check-reason describing the changed behaviour")
    failed = [path for path, case, old in records if case.get("status") == "failed"
              and matching_case(case, old, protocol, model, options[track], smoke=smoke, source=False)]
    if failed and not repeat_reason:
        raise ValueError("A retry requires --repeat-reason explaining what fixes the recorded failure")
    folder = output / ("functional-smoke-v1" if smoke else "") / model / track
    if (folder / "result.json").exists():
        raise FileExistsError("Preserve the previous attempt; use a new attempt directory with a repeat reason")
    # Sensitivity studies run several predeclared tracks per model; optimization iterations select one.
    if track != control and not smoke and track not in protocol.get("study_tracks", {}).get(model, []):
        selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
        if selection[model]["track"] != track:
            raise ValueError("Candidate was not selected from recorded profile evidence")
    return {"action": "run", "model": model, "track": track, "smoke": smoke,
            "repeat_reason": repeat_reason or None, "check_reason": check_reason or None,
            "effective_loop": definition, "protocol_sha256": hashlib.sha256((output / "protocol.json").read_bytes()).hexdigest()}


def check_inputs(root, paths):
    return {name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest() for name in paths}


def latest_streak(root, model, default=0, round_start=None):
    """Reused reports cannot replace the stopping state of an actual trial."""
    for path in sorted((Path(root) / "artifacts/retraining/gpu-retrain-r01").glob("iteration-*/comparison.json"), reverse=True):
        if round_start and path.parent.name.removeprefix("iteration-") < round_start:
            continue
        comparison = json.loads(path.read_text(encoding="utf-8")).get("comparisons", {}).get(model)
        if comparison is not None and not comparison.get("reused_candidate"):
            return streak(comparison)
    return default


def resolve_case(output, model, track, root):
    records_path = Path(output) / "reused-results.json"
    records = json.loads(records_path.read_text(encoding="utf-8")) if records_path.exists() else {}
    record = records.get(model, {}).get(track)
    if record is not None:
        case, path = read_reference(record["result"], root)
        return case, path, True
    path = Path(output) / model / track / "result.json"
    return json.loads(path.read_text(encoding="utf-8")), path, False


def reusable_check(root, suite, inputs):
    root = Path(root)
    paths = [*root.glob("artifacts/diagnostics/*/result.json"),
             *root.glob("artifacts/retraining/gpu-retrain-r01/iteration-*/*checks*/result.json")]
    for path in sorted(paths):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") == "passed" and report.get("check_suite") == suite and report.get("check_inputs") == inputs:
            return path
    return None


def reusable_profile(result_path, root, minimum_seconds=None):
    result_path, root = Path(result_path).resolve(), Path(root).resolve()
    metadata = result_metadata(result_path, root)
    base = root / "artifacts/retraining/gpu-retrain-r01"
    candidates = [*base.glob("iteration-*/profiles/*/result.json")]
    for path in sorted(candidates):
        profile = json.loads(path.read_text(encoding="utf-8"))
        workload = profile.get("workload", {})
        if (minimum_seconds is not None and workload.get("minimum_observation_seconds") != minimum_seconds
                and not (workload.get("full_history_profile_available") and workload.get("training_observed_seconds", 0) >= minimum_seconds)):
            continue
        if not workload.get("full_history_complete") or not workload.get("complete_training_observation"):
            continue
        inputs = profile.get("inputs", [])
        if not any(item.get("path") == metadata["path"] and item.get("sha256") == metadata["sha256"] for item in inputs):
            continue
        if all((root / item["path"]).is_file() and hashlib.sha256((root / item["path"]).read_bytes()).hexdigest() == item.get("sha256")
               and (root / item["path"]).resolve().is_relative_to(root / "artifacts") for item in inputs if "sha256" in item):
            return path
    return None


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


def fixed_round_budget(protocol):
    """An explicit human count overrides misses, but never authorizes extra trials."""
    budget = protocol.get("round_iteration_budget")
    index = protocol.get("round_iteration_index")
    if budget is None and index is None:
        return None
    if (type(budget) is not int or budget < 1 or type(index) is not int or index < 1
            or not round_boundary(protocol)):
        raise ValueError("Fixed-count round requires positive budget/index and recorded human authorization")
    return budget, index


def round_budget_fields(iteration, previous=None, *, authorization=None, budget=None):
    if budget is not None:
        if type(budget) is not int or budget < 1 or not (authorization or "").strip():
            raise ValueError("--round-iterations requires a positive count and new --round-authorization")
        return {"round_iteration_budget": budget, "round_iteration_index": 1,
                "round_start_iteration": iteration, "round_authorization": authorization.strip()}
    if authorization:
        return {}  # A newly authorized uncounted round does not inherit an old count.
    fixed = fixed_round_budget(previous or {})
    if fixed is None:
        return {}
    count, index = fixed
    return {"round_iteration_budget": count, "round_iteration_index": index + 1,
            "round_start_iteration": previous["round_start_iteration"],
            "round_authorization": previous["round_authorization"]}


def round_loss_limit(previous=None, *, new_round=False, requested=None):
    """A round freezes its relative loss ceiling at start; continuations inherit it unchanged."""
    if requested is not None:
        if not new_round:
            raise ValueError("--loss-ceiling is frozen at round start; pass it with --round-authorization")
        if type(requested) is not float or not math.isfinite(requested) or not 0 <= requested < 1:
            raise ValueError("--loss-ceiling must be a fraction in [0, 1), for example 0.001 for 0.1%")
        return requested
    if not new_round and previous and "round_relative_loss_limit" in previous:
        return previous["round_relative_loss_limit"]
    return None


def require_active(model, no_gain_streak, protocol=None):
    fixed = fixed_round_budget(protocol or {})
    if fixed is not None:
        budget, index = fixed
        if index > budget:
            raise RuntimeError(f"{model}: explicit {budget}-iteration round is exhausted; new authorization required")
        return
    if no_gain_streak >= 3:
        raise RuntimeError(f"{model}: stop after {no_gain_streak} consecutive misses of the applicable throughput target")


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


def decide(gain_percent, losses, *, relative_limit, baseline_limit, prior_streak, resources_reduced=False,
           speed_tolerance_percent=0.0, target_percent=20.0, exact_parity=False, exact_target_percent=None,
           exact_promotion_rule="fixed_target", timing_noise_available=True, loss_tolerance=False):
    """Use the frozen promotion rule; keep historical fixed-target decisions intact.

    loss_tolerance (a human-set round ceiling, 2026-10-12): measured regressions within the ceiling,
    against both the previous accepted run and the round's starting reference, may be promoted
    instead of becoming fastmode. Without it, any regression of a faster candidate stays fastmode.
    """
    if type(loss_tolerance) is not bool:
        raise ValueError("Loss tolerance must be a frozen boolean")
    if exact_promotion_rule not in ("fixed_target", "timing_noise") or type(timing_noise_available) is not bool:
        raise ValueError("Valid frozen exact-promotion rule and timing-noise availability required")
    if type(gain_percent) not in (int, float) or not math.isfinite(gain_percent):
        raise ValueError("Finite measured throughput gain required")
    if type(speed_tolerance_percent) not in (int, float) or not 0 <= speed_tolerance_percent < 100:
        raise ValueError("Finite non-negative speed tolerance required")
    metrics = losses["previous"]["metrics"]
    accurate = point_losses_pass(metrics, relative_limit)
    if loss_tolerance:
        if "round_start" not in losses:
            raise ValueError("A tolerated round ceiling needs the round's starting reference to bound cumulative drift")
        accurate = accurate and point_losses_pass(losses["round_start"]["metrics"], relative_limit)
    baseline_preferred = point_losses_pass(losses["original_baseline"]["metrics"], baseline_limit)
    worse = any(value["relative_delta"] > 0 for value in metrics.values())
    exact_noise = exact_parity and exact_promotion_rule == "timing_noise"
    exact_gate = exact_parity and exact_target_percent is not None and not exact_noise
    if (exact_gate or exact_noise) and any(value["relative_delta"] != 0 for value in metrics.values()):
        raise ValueError("Exact parity cannot accompany changed measured scores")
    if exact_gate:
        if (type(exact_target_percent) not in (int, float) or not math.isfinite(exact_target_percent)
                or not 0 < exact_target_percent < 100):
            raise ValueError("Finite positive exact-parity throughput target required")
        target_percent = exact_target_percent
    if exact_noise:
        target_percent = speed_tolerance_percent
    faster = (timing_noise_available and gain_percent > speed_tolerance_percent if exact_noise else
              gain_percent >= target_percent if exact_gate else gain_percent > speed_tolerance_percent)
    slower = gain_percent < -speed_tolerance_percent
    fastmode = faster and (not accurate if loss_tolerance else (worse or not accurate))
    missing_noise = exact_noise and not timing_noise_available
    retune = not missing_noise and not faster and not slower and accurate and not worse and resources_reduced
    accepted = faster and accurate and not fastmode
    # Human request 2026-10-12: keep positive gains below the noise band visible so several can be stacked.
    stack_candidate = (not accepted and not fastmode and not missing_noise and accurate and gain_percent > 0
                       and (exact_parity or loss_tolerance or not worse))
    decision = ("pending measured timing-noise evidence" if missing_noise else
                "retain separate opt-in fastmode" if fastmode else
                "retain main improvement; preliminary only" if accepted else
                "investigate retuning freed resources" if retune else
                "retain previous accepted run; speed target missed")
    met_target = faster if exact_noise else gain_percent >= target_percent
    return {"faster": faster, "speed_change": "timing noise unavailable" if missing_noise else "faster" if faster else "slower" if slower else "same within recorded repeat spread",
            "speed_tolerance_percent": speed_tolerance_percent, "passes_preliminary_quality": accurate,
            "baseline_preferred_limit_met": baseline_preferred,
            "fastmode": fastmode, "accepted_main": accepted,
            "stack_candidate": stack_candidate, "loss_tolerance_within_limit": loss_tolerance,
            "resource_retuning_pending": retune, "decision": decision,
            "exact_speed_gate_applied": exact_gate,
            "exact_promotion_rule": exact_promotion_rule if exact_parity else "timing_noise_with_quality_checks",
            "timing_noise_gate_applied": exact_noise,
            # Field names predate the configurable target; the protocol's target applies.
            "throughput_target_percent": target_percent,
            "meets_20_percent_target": met_target,
            "no_20_percent_gain_streak": 0 if met_target else prior_streak + 1}


def recount_streak(root, model, round_start, through, target_percent):
    """Re-derive a round's miss count when the LOOP target changes; recorded decisions stay unchanged."""
    count = 0
    for path in sorted((Path(root) / "artifacts/retraining/gpu-retrain-r01").glob("iteration-*/comparison.json")):
        name = path.parent.name.removeprefix("iteration-")
        if (round_start and name < round_start) or name > through:
            continue
        comparison = json.loads(path.read_text(encoding="utf-8")).get("comparisons", {}).get(model)
        if comparison is None or comparison.get("reused_candidate"):
            continue
        gain = comparison.get("references", {}).get("previous", {}).get("throughput_gain_percent")
        # Older comparison formats lack the gain; keep their recorded count.
        threshold = comparison.get("throughput_target_percent", target_percent) if comparison.get("exact_speed_gate_applied") else target_percent
        count = streak(comparison) if gain is None else 0 if gain >= threshold else count + 1
    return count


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
    for key, value in reductions.items():
        kind, threshold = MATERIAL_REDUCTIONS[key]
        drop = value["reference"] - value["candidate"]
        value["material"] = drop >= threshold if kind == "points" else drop >= threshold * value["reference"]
        value["material_threshold"] = f"{threshold:g} percentage points" if kind == "points" else f"{100 * threshold:g}% of reference"
    return reductions


def declared_differences(*cases):
    config, exposure = set(), set()
    for case in cases:
        for option, (keys, fields) in DECLARED_CHANGES.items():
            if option in case.get("options", {}):
                config |= keys
                exposure |= fields
    return config, exposure


def same_exposure(left, right, keys):
    """Compare exposure fields, allowing only declared update-schedule differences."""
    _, allowed = declared_differences(left, right)
    for key in keys:
        if key in allowed:
            if key == "coverage" and ({user: value["reviews"] for user, value in left["coverage"].items()}
                                      != {user: value["reviews"] for user, value in right["coverage"].items()}):
                return key
            continue
        if key not in left or key not in right or left[key] != right[key]:
            return key
    return None


def planned_updates(rows, chunks, chunk_reviews=128, *, pack_users=False):
    if pack_users:
        total_chunks = sum((row["available_reviews"] + chunk_reviews - 1) // chunk_reviews for row in rows)
        return (total_chunks + chunks - 1) // chunks
    span = chunks * chunk_reviews
    return sum((row["available_reviews"] + span - 1) // span for row in rows)


def material_reduction(reductions):
    """Only a material saving can justify the LOOP's resource-retuning step."""
    return any(value.get("material") for value in reductions.values())


def pairwise_spreads(groups):
    """Percent max/min - 1 for every pair of runs within each identical-configuration group."""
    return [100 * (max(a, b) / min(a, b) - 1) for group in groups for a, b in combinations(group, 2)]


def speed_noise_band(root, model):
    """Median throughput spread over all pairs of identical-configuration repeats.

    Human decision 2026-10-12 (option C): the median replaces the widest group's max/min, which could
    only grow and let one slow-state repeat set the band (0042 vs 0046 would have frozen 22.149%).
    """
    groups = {}
    for path, case, protocol in recorded_cases(root):
        if (case.get("model") != model or case.get("status") != "complete" or case.get("smoke_only")
                or type(case.get("reviews_per_second")) not in (int, float) or not case.get("trained_parameter_sha256")):
            continue
        key = json.dumps([protocol.get("cohort_sha256"), protocol.get("training"), protocol.get("execution_mode"),
                          case.get("options"), case.get("config"), case["trained_parameter_sha256"]], sort_keys=True)
        groups.setdefault(key, []).append((path, case["reviews_per_second"]))
    repeats = [group for group in groups.values() if len(group) >= 2]
    if not repeats:
        return {"percent": 0.0, "available": False, "sources": [],
                "rule": "no recorded repeat of an identical configuration; any positive gain counts as faster"}
    spreads = pairwise_spreads([[rate for _, rate in group] for group in repeats])
    widest = max(max(rate for _, rate in group) / min(rate for _, rate in group) for group in repeats)
    return {"percent": statistics.median(spreads), "available": True, "estimator": "median_pairwise_spread",
            "pairs": len(spreads), "widest_group_percent": 100 * (widest - 1),
            "sources": [result_metadata(path, root) for group in repeats for path, _ in group],
            "rule": "median over all pairs of complete full-history repeats with identical cohort, training settings, options, config and trained weights of max/min - 1 reviews/s (human decision 2026-10-12; replaces the widest group's max/min)"}
