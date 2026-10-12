"""Report remaining-track results without granting approval or launching work."""

from argparse import ArgumentParser
import hashlib
import json
import math
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance
from scripts.gpu_training_targets import METRICS, compare_budgets
from scripts.report_gpu_batch1_relative import value_with_percent

USED_SCRIPTS = {"train_gpu_future.py", "gpu_future_models.py", "gpu_training_targets.py", "gpu_fp8.py",
                "gpu_row_tiles.py", "experiment_provenance.py", "trial_batches.py", "gpu_batch1.py",
                "run_gpu_batch1.py", "benchmark_gpu_trial.py", "train_gpu_trial.py", "gpu_optimizations.py",
                "gpu_packed_encodings.py", "compare.py", "gpu_power.py"}


def number(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def source_fingerprint(report):
    provenance = report["provenance"]
    with ZipFile(ROOT / provenance["source_archive"]) as archive:
        encoded = archive.read("source-manifest.json")
    if hashlib.sha256(encoded).hexdigest() != provenance["source_tree_sha256"]:
        raise ValueError("Recorded source manifest fingerprint differs")
    files = json.loads(encoded)["files"]
    consumed = {name: digest for name, digest in files.items()
                if name.startswith(("rwkv/", "expertium/", "scripts/kernels/"))
                or name in ("pyproject.toml", "uv.lock")
                or (name.startswith("scripts/") and Path(name).name in USED_SCRIPTS)}
    return hashlib.sha256(json.dumps(consumed, sort_keys=True).encode()).hexdigest()


def supported_tasks(report):
    task = report["config"].get("task", "joint")
    if task not in ("joint", "curve", "instant"):
        raise ValueError("Unsupported task configuration")
    return ("curve", "instant") if task == "joint" else (task,)


def final_users(report, expected_users):
    if not report.get("history") or report["history"][-1]["update"] != report["updates"]:
        raise ValueError("Missing final measured checkpoint")
    rows = report["history"][-1]["per_user"]
    users = {row["user_id"]: row for row in rows}
    if len(users) != len(rows) or set(users) != expected_users:
        raise ValueError("Distinct evaluation users differ from the selected cohort")
    tasks = supported_tasks(report)
    for row in rows:
        if not row.get("input_sha256") or not number(row.get("raw_reviews")):
            raise ValueError("Missing held-out input/review identity")
        for task in ("curve", "instant"):
            score, labels = row[task + "_log_loss"], row[task + "_labels"]
            if task in tasks:
                if not number(score) or not isinstance(labels, int) or labels <= 0:
                    raise ValueError("Supported task needs finite binary loss and labels for every user")
            elif score is not None or labels != 0:
                raise ValueError("Unsupported task values must remain null with zero labels")
    for task in ("curve", "instant"):
        count = sum(row[task + "_labels"] for row in rows)
        expected = {task + "_labels": count, task + "_evaluated_users": len(rows) if task in tasks else 0,
                    task + "_log_loss_label_weighted": (sum(row[task + "_log_loss"] * row[task + "_labels"] for row in rows) / count if count else None),
                    task + "_log_loss_user_average": (sum(row[task + "_log_loss"] for row in rows) / len(rows) if count else None)}
        for key, value in expected.items():
            for metrics in (report["evaluation"], report["history"][-1]["metrics"]):
                recorded = metrics.get(key)
                if ((value is None and recorded is not None) or
                        (value is not None and (not number(recorded) or not math.isclose(value, recorded, rel_tol=1e-12, abs_tol=1e-12)))):
                    raise ValueError("Aggregate/final checkpoint differs from paired per-user scores")
    return users


def bootstrap_metrics(candidate, control, tasks, resamples=10000, family=64):
    """Paired user bootstrap; tests of both controls form one acceptance condition."""
    if set(candidate) != set(control) or not candidate:
        raise ValueError("Paired user sets differ or are empty")
    if resamples < 2 or family < 1:
        raise ValueError("Invalid bootstrap settings")
    users = sorted(candidate)
    # Bounded scratch allocation even for larger future cohorts.
    rng = np.random.default_rng(20251012)
    data, sampled = {}, {}
    for task in tasks:
        label_key, loss_key = task + "_labels", task + "_log_loss"
        if any(candidate[user][label_key] != control[user][label_key] for user in users):
            raise ValueError("Supported-task held-out label counts differ")
        weights = np.array([candidate[user][label_key] for user in users], dtype=float)
        delta = np.array([candidate[user][loss_key] - control[user][loss_key] for user in users])
        if not np.isfinite(delta).all() or not (weights > 0).all():
            raise ValueError("Invalid paired losses or label weights")
        data[task] = weights, delta
        sampled[task] = {"label_weighted": [], "user_average": []}
    for start in range(0, resamples, 256):
        indices = rng.integers(0, len(users), size=(min(256, resamples - start), len(users)))
        for task, (weights, delta) in data.items():
            values, masses = delta[indices], weights[indices]
            sampled[task]["user_average"].append(values.mean(axis=1))
            sampled[task]["label_weighted"].append((values * masses).sum(axis=1) / masses.sum(axis=1))
    result = {}
    for task, (weights, delta) in data.items():
        for kind, pieces in sampled[task].items():
            distribution = np.concatenate(pieces)
            result[task + "_log_loss_" + kind] = {
                "delta": float(delta.mean() if kind == "user_average" else np.average(delta, weights=weights)),
                "ci95": [float(value) for value in np.quantile(distribution, [0.025, 0.975])],
                "familywise_upper": float(np.quantile(distribution, 1 - 0.05 / family))}
    return result


def paired_inputs(candidate, control):
    for user, row in candidate.items():
        for key in ("input_sha256", "raw_reviews"):
            if row[key] != control[user][key]:
                raise ValueError(f"Held-out {key} differs for user {user}")


def total_cost(report):
    cost = {key: report[key] for key in ("training_seconds", "elapsed_seconds")}
    if any(not number(value) or value <= 0 for value in cost.values()):
        raise ValueError("Measured training and complete elapsed seconds must be positive")
    prior = report["model"].get("pretraining")
    if prior:
        manifest = prior["manifest"]
        if any(not number(manifest.get(key)) for key in cost):
            return {"status": "inconclusive: prior pretraining time not recorded", "fine_tuning": cost,
                    "prior_training_reviews": manifest["training_reviews"]}
        cost = {key: value + manifest[key] for key, value in cost.items()}
    return {"status": "measured", **cost}


def compare_case(plan, case, candidate, controls):
    if not candidate or candidate.get("status") != "complete":
        return {"status": candidate.get("status", "pending") if candidate else "pending"}
    if candidate.get("config", {}).get("synthetic_only"):
        return {"status": "synthetic only; not validation"}
    if plan["authorization"].get("status") != "approved" or not plan["authorization"].get("user_request"):
        return {"status": "not validated: broader run approval missing"}
    if plan["acceptance"].get("status") != "approved":
        return {"status": "not validated: acceptance criteria not approved"}
    cohort = plan["cohort"]
    users = {row["user_id"] for row in cohort["rows"]}
    if len(users) < 500 or len(users) != cohort["evaluation_users"] or len(users) != len(cohort["rows"]):
        raise ValueError("Requires at least 500 distinct selected evaluation users")
    if users & set(cohort["independent_final_test_users"]) or any(not 1 <= user <= 5000 for user in users):
        raise ValueError("Invalid cohort or final-test overlap")
    if any(not value or value.get("status") != "complete" for value in controls):
        return {"status": "pending matched repeated controls"}
    if any(value.get("config", {}).get("synthetic_only") for value in controls):
        return {"status": "synthetic control; not validation"}
    config = candidate["config"]
    if config.get("evaluation_users") != len(users):
        raise ValueError("Declared evaluation count differs from the cohort")
    for key, value in case["config"].items():
        if config.get(key) != value:
            raise ValueError(f"Candidate configuration differs from proposed case: {key}")
    consumed = source_fingerprint(candidate)
    expected_reviews = sum(row["train_range"][1] - row["train_range"][0] for row in cohort["rows"])
    fields = ("model", "seed", "dataset_path", "dataset_revision", "evaluation_users", "epochs", "batch_chunks",
              "preparation_workers", "checkpoint_steps", "checkpoint_every", "checkpoint_tile", "raw_pointers")
    for control in controls:
        if source_fingerprint(control) != consumed:
            raise ValueError("Consumed source differs from the matched control")
        if any(config.get(key) != control["config"].get(key) for key in fields):
            raise ValueError("Seed/data/batch/backend/checkpoint protocol differs from control")
        if control["raw_review_presentations"] != expected_reviews:
            raise ValueError("Control did not complete the selected training exposure")
    if config["dataset_path"] != cohort["dataset_path"] or config["dataset_revision"] != cohort["dataset_revision"]:
        raise ValueError("Candidate dataset differs from the approved cohort")
    capacity_change = bool(config.get("width_heads") or config.get("remove_layers") or config.get("pretrained_checkpoint"))
    if not capacity_change:
        for control in controls:
            if candidate["model"]["initial_backbone_sha256"] != control["model"]["initial_backbone_sha256"]:
                raise ValueError("Unchanged-capacity initial backbone differs from control")
            if config.get("task", "joint") == "joint" and candidate["model"]["initial_parameter_sha256"] != control["model"]["initial_parameter_sha256"]:
                raise ValueError("Unchanged-shape initial parameters differ from control")
    candidate_users = final_users(candidate, users)
    control_users = [final_users(control, users) for control in controls]
    expected_heldout = {row["user_id"]: row["validation_range"][1] - row["validation_range"][0] for row in cohort["rows"]}
    if any(row["raw_reviews"] != expected_heldout[user] for user, row in candidate_users.items()):
        raise ValueError("Held-out raw-review exposure differs from cohort")
    paired_inputs(control_users[0], control_users[1])
    for control in control_users:
        paired_inputs(candidate_users, control)
    tolerance = plan["acceptance"]["max_binary_logloss_regression"]
    if not number(tolerance):
        raise ValueError("Invalid predeclared quality tolerance")
    controls_stable = all(abs(controls[0]["evaluation"][key] - controls[1]["evaluation"][key]) <= tolerance for key in METRICS)
    tasks = supported_tasks(candidate)
    pairs = [bootstrap_metrics(candidate_users, control, tasks) for control in control_users]
    quality = all(metric["familywise_upper"] <= tolerance for pair in pairs for metric in pair.values())
    cost = total_cost(candidate)
    control_costs = [total_cost(control) for control in controls]
    speed = cost["status"] == "measured" and all(value["status"] == "measured" for value in control_costs)
    if speed:
        speed = all(cost[key] < min(value[key] for value in control_costs) for key in ("training_seconds", "elapsed_seconds"))
    full_exposure = candidate["raw_review_presentations"] == expected_reviews and not candidate.get("stop_reason")
    if not controls_stable:
        decision = "inconclusive: controls disagree"
    elif not full_exposure:
        decision = "reduced exposure; observed checkpoints/target times only"
    elif not quality:
        decision = "quality non-inferiority not established"
    elif cost["status"] != "measured":
        decision = cost["status"]
    elif not speed:
        decision = "no training and complete elapsed-time gain"
    else:
        decision = "validated for this protocol"
    return {"status": decision, "controls_stable": controls_stable, "quality_pass": quality, "speed_pass": speed,
            "full_exposure": full_exposure, "paired_metrics_against_each_control": pairs,
            "candidate_cost": cost, "control_costs": control_costs,
            "actual_updates": [candidate["updates"], *(value["updates"] for value in controls)],
            "actual_raw_presentations": [candidate["raw_review_presentations"], *(value["raw_review_presentations"] for value in controls)],
            "observed_budgets": compare_budgets(controls[0]["history"], candidate["history"]),
            "target_first_observation_seconds": candidate.get("first_target_observation_seconds"),
            "scope": "one training seed; paired-user uncertainty conditional on that seed; checkpoint budget comparisons are descriptive and retain gaps"}


def specialist_costs(results, comparisons):
    curve, instant = (results.get("expertium/F13-" + task) for task in ("curve", "instant"))
    controls = [results.get("expertium/" + name) for name in ("control", "control-repeat")]
    if any(not value or value.get("status") != "complete" for value in (curve, instant, *controls)):
        return {"status": "pending both specialists and matched repeated controls"}
    if any(value["config"].get("synthetic_only") for value in (curve, instant, *controls)):
        return {"status": "synthetic only; not validation"}
    pairs = [comparisons["expertium/F13-" + task] for task in ("curve", "instant")]
    if not all(pair.get("full_exposure") and pair.get("controls_stable") for pair in pairs):
        return {"status": "pending matched full-exposure quality comparisons"}
    costs = [total_cost(value) for value in (curve, instant, *controls)]
    if any(cost["status"] != "measured" for cost in costs):
        return {"status": "inconclusive: specialist/control complete cost missing"}
    summed = {key: costs[0][key] + costs[1][key] for key in ("training_seconds", "elapsed_seconds")}
    faster = all(summed[key] < min(cost[key] for cost in costs[2:]) for key in summed)
    quality = all(pair.get("quality_pass") for pair in pairs)
    return {"status": ("validated combined specialist costs for this protocol" if quality and faster else
                       "combined quality non-inferiority not established" if not quality else "no summed specialist cost gain"),
            "quality_pass": quality, "speed_pass": faster, "summed_cost": summed,
            "summed_raw_presentations": curve["raw_review_presentations"] + instant["raw_review_presentations"],
            "summed_reviews_per_training_second": (curve["raw_review_presentations"] + instant["raw_review_presentations"]) / summed["training_seconds"],
            "scope": "two separately trained models; supported scores come from their individual paired tests; deployment/state cost unmeasured"}


def build_report(plan, runs):
    results, comparisons = {}, {}
    for case in plan["cases"]:
        path = runs / case["name"] / "result.json"
        if path.exists():
            results[case["name"]] = json.loads(path.read_text())
    for case in plan["cases"]:
        name, config = case["name"], case["config"]
        if name.split("/")[-1] in ("control", "control-repeat", "F09-control-tile8", "F09-control-tile8-repeat"):
            continue
        prefix = config["model"] + "/"
        controls = ("F09-control-tile8", "F09-control-tile8-repeat") if config.get("row_tiles") else ("control", "control-repeat")
        try:
            comparisons[name] = compare_case(plan, case, results.get(name), [results.get(prefix + control) for control in controls])
        except (ValueError, KeyError, OSError) as exc:
            comparisons[name] = {"status": "unmatched/incomplete evidence", "error": str(exc)}
    return {"authorization": plan["authorization"], "results": results, "comparisons": comparisons,
            "specialist_bundle": specialist_costs(results, comparisons), "deferred_cases": plan.get("deferred_cases", []),
            "scope": "pending/failed tracks retained; synthetic values are never quality validation; one seed"}


def render(report, plan, runs):
    lines = ["# Remaining GPU optimization comparisons", "",
             f"Authorization: **{report['authorization']['status']}**.", "",
             "All proposed cases remain visible. Missing measurements are pending. Synthetic fixture scores/timings are omitted.",
             "Both Curve and Instant scores are binary log loss; lower is better. Times include separate validation/setup scope.",
             "Parenthesized percentages are 100 × (candidate − first matched control) / control. Losses use their own weighted/user denominator; training, complete time and throughput are separate.", "",
             "| Case | Status / decision | Curve weighted / user | Instant weighted / user | Train / complete s | Raw reviews/s |",
             "|---|---|---:|---:|---:|---:|"]
    for case in plan["cases"]:
        name = case["name"]
        value = report["results"].get(name)
        decision = report["comparisons"].get(name, {}).get("status", value.get("status", "pending") if value else "pending")
        if not value or value.get("status") != "complete" or value.get("config", {}).get("synthetic_only"):
            if value and value.get("config", {}).get("synthetic_only"):
                decision = "synthetic only; not validation"
            lines.append(f"| {name} | {decision} | — | — | — | — |")
            continue
        scores = value["evaluation"]
        control_name = "F09-control-tile8" if case["config"].get("row_tiles") else "control"
        control = report["results"].get(case["config"]["model"] + "/" + control_name)
        matched = (control and control.get("status") == "complete" and not control.get("config", {}).get("synthetic_only")
                   and decision != "unmatched/incomplete evidence")
        def losses(task):
            weighted, user = (scores.get(task + "_log_loss_" + kind) for kind in ("label_weighted", "user_average"))
            if weighted is None or user is None:
                return "— (unsupported)"
            return " / ".join(value_with_percent(metric, control["evaluation"].get(task + "_log_loss_" + kind) if matched else None)
                              for metric, kind in ((weighted, "label_weighted"), (user, "user_average")))
        times = " / ".join(value_with_percent(value[key], control[key] if matched else None, 2, 2) for key in ("training_seconds", "elapsed_seconds"))
        speed = value_with_percent(value["reviews_per_second"], control["reviews_per_second"] if matched else None, 0, 2)
        lines.append(f"| {name} | {decision} | {losses('curve')} | {losses('instant')} | {times} | {speed} |")
    lines += ["", f"Summed Curve + Instant specialist costs: **{report['specialist_bundle']['status']}**.", "",
              "Paired deltas, adjusted intervals against both controls, checkpoint budget gaps and summed costs are retained in comparison.json.",
              "Budget comparisons select observed checkpoints; there is no interpolation and no assumption of equal exposure at equal time.", "",
              "| Case | Process-start Git commit | Dirty | Preserved source |", "|---|---|---|---|"]
    for name, value in report["results"].items():
        saved = value.get("provenance", {})
        lines.append(f"| {name} | {saved.get('git_commit', '— (not recorded)')} | {saved.get('git_dirty', '—')} | {saved.get('source_archive', '— (not recorded)')} |")
    lines += ["", "Deferred cases:", ""]
    lines.extend(f"- {case['track']}: {case['requires']}" for case in report["deferred_cases"])
    return "\n".join(lines) + "\n"


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Save analysis below workspace artifacts")
    provenance = process_provenance()
    plan = json.loads(args.plan.read_text())
    report = build_report(plan, args.runs.resolve())
    report["analysis_provenance"] = provenance
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "summary.md").write_text(render(report, plan, args.runs.resolve()), encoding="utf-8")
    print(f"Report retains {len(plan['cases'])} proposed cases; {len(report['results'])} recorded results")


if __name__ == "__main__":
    main()
