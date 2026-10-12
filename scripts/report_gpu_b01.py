"""Matched B01 quality/time report with absolute and percentage deltas."""

import json
import os
from pathlib import Path

from scripts.report_gpu_batch1 import paired
from scripts.report_gpu_batch1_relative import delta, value_with_percent

ROOT = Path(__file__).resolve().parents[1]
BEGIN, END = "<!-- gpu-b01-progress:start -->", "<!-- gpu-b01-progress:end -->"


def controls_stable(controls):
    if not all(c and c["status"] == "complete" and not c.get("smoke_only") and c["evaluation_users"] == 500 for c in controls):
        return False
    keys = ("implementation_sha256", "initial_parameter_sha256", "validation_input_sha256", "updates", "review_presentations",
            "training_curve_labels", "training_instant_labels", "execution_mode", "protocol_sha256")
    return (all(controls[0].get(key) == controls[1].get(key) for key in keys)
            and all(abs(controls[0]["evaluation"][key] - controls[1]["evaluation"][key]) <= 0.001
                    for key in controls[0]["evaluation"] if "log_loss" in key))


def assess(suite):
    rows = {(row["model"], row["track"]): row for row in suite["results"]}
    comparisons = []
    for model in ("baseline", "expertium"):
        controls = [rows.get((model, track)) for track in ("control", "control-repeat")]
        complete = all(c and c["status"] == "complete" for c in controls)
        stable = controls_stable(controls)
        candidate = rows.get((model, "B01"))
        result = {"model": model, "controls_complete": complete, "controls_stable": bool(stable),
                  "status": "pending", "against_controls": [], "accepted": False}
        if not complete:
            result["status"] = "pending repeated controls"
        elif not stable:
            result["status"] = "inconclusive: controls disagree"
        elif not candidate or candidate["status"] != "complete":
            result["status"] = candidate["status"] if candidate else "pending B01"
        else:
            for control in controls:
                if candidate.get("execution_mode") != control.get("execution_mode") or candidate.get("protocol_sha256") != control.get("protocol_sha256"):
                    raise ValueError("Unmatched B01 mode/protocol")
                values = paired(candidate, control)
                values.update(control=control["track"],
                              speed_pass=candidate["cold_training_seconds"] < control["cold_training_seconds"],
                              delta_training_seconds=candidate["cold_training_seconds"] - control["cold_training_seconds"],
                              delta_training_percent=100 * (candidate["cold_training_seconds"] / control["cold_training_seconds"] - 1),
                              delta_reviews_per_second=candidate["reviews_per_second"] - control["reviews_per_second"],
                              delta_speed_percent=100 * (candidate["reviews_per_second"] / control["reviews_per_second"] - 1))
                result["against_controls"].append(values)
            result["accepted"] = all(p["quality_pass"] and p["speed_pass"] for p in result["against_controls"])
            result["status"] = "accepted for this protocol" if result["accepted"] else "quality non-inferiority not established" if not all(p["quality_pass"] for p in result["against_controls"]) else "no complete-training speed gain"
        comparisons.append(result)
    return {"models": comparisons, "B01_accepted_both_models": all(value["accepted"] for value in comparisons)}


def write_report(output, suite, update_document=True):
    assessment = assess(suite)
    rows = {(row["model"], row["track"]): row for row in suite["results"]}
    mode = suite.get("protocol", {}).get("execution_mode", {})
    mode_text = (f"Frozen execution mode: TorchScript executor optimization={mode.get('torch_graph_executor_optimize', 'unavailable')}; "
                 f"deterministic algorithms={mode.get('deterministic_algorithms', 'unavailable')}; "
                 f"CUBLAS workspace={mode.get('CUBLAS_WORKSPACE_CONFIG', 'inherited; not recorded in this protocol')}.")
    lines = ["# B01: 500-user matched execution baseline", "", f"Suite: **{suite['status']}**; complete cases: **{sum(row['status'] == 'complete' for row in suite['results'])}/6**.", "",
             "Same preserved 500 users, seed 2025, temporal budgets, model/chunk batching, optimizer and BF16/FP32 precision.",
             mode_text,
             "Training dropout remains enabled. Evaluation uses original features, FP32 master, dropout off and WKV tile32.",
             "B01: F14 packed GPU features + F01 ordered prefetch + F03 worker-local/packed IPC + F04 transfer overlap + F05 exact-shape graphs + F06 reduced synchronization/logging.",
             "One exact graph slot, second-use admission, eager fallback; Default evicts on any shape miss, Expertium retains until admission/eviction. No bucketed shapes or F02 sample cache.", "",
             "Percentages are 100 × (candidate − own-model control) / control. Positive LL is worse, negative time and positive reviews/s are faster.",
             "Complete training includes loads/workers/preparation/transfers/materialization/updates/capture/logging. Setup, primer, evaluation, saves, process wall and telemetry overhead are recorded separately.", ""]
    for model in ("baseline", "expertium"):
        lines += [f"## {model}", "", "| Case | Status | Curve LL weighted / user (%) | Instant LL weighted / user (%) | Train s (%) | Reviews/s (%) | Peak reserved MiB |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        reference = rows.get((model, "control"))
        for track in ("control", "control-repeat", "B01"):
            row = rows.get((model, track))
            if not row or row["status"] != "complete":
                status = row["status"] if row else "running" if suite.get("active_case") == {"model": model, "track": track} else "pending"
                lines.append(f"| {track} | {status} | — | — | — | — | — |")
                continue
            matched = reference and reference["status"] == "complete"
            def score(task):
                return " / ".join(value_with_percent(row["evaluation"][task + "_log_loss_" + kind],
                                                     reference["evaluation"][task + "_log_loss_" + kind] if matched and track != "control" else None)
                                  for kind in ("label_weighted", "user_average"))
            train = value_with_percent(row["cold_training_seconds"], reference["cold_training_seconds"] if matched and track != "control" else None, 2, 2)
            speed = value_with_percent(row["reviews_per_second"], reference["reviews_per_second"] if matched and track != "control" else None, 0, 2)
            lines.append(f"| [{track}]({model}/{track}/result.json) | complete | {score('curve')} | {score('instant')} | {train} | {speed} | {row['peak_reserved_mib']:.0f} |")
        result = next(row for row in assessment["models"] if row["model"] == model)
        lines += ["", f"Decision: **{result['status']}**; repeated controls stable: **{result['controls_stable']}**.", "",
                  "| B01 comparison | Δ Curve LL weighted / user (%) | Δ Instant LL weighted / user (%) | Δ train s (%) | Δ reviews/s (%) | Adjusted upper LL bounds Curve / Instant (weighted, user) |",
                  "|---|---:|---:|---:|---:|---|"]
        for comparison in result["against_controls"]:
            control = rows[(model, comparison["control"])]
            candidate = rows[(model, "B01")]
            def changes(task):
                return " / ".join(delta(candidate["evaluation"][task + "_log_loss_" + kind], control["evaluation"][task + "_log_loss_" + kind]) for kind in ("label_weighted", "user_average"))
            def bounds(task):
                return "(" + ", ".join(f"{comparison['metrics'][task + '_log_loss_' + kind]['familywise_upper']:+.6f}" for kind in ("label_weighted", "user_average")) + ")"
            lines.append(f"| vs {comparison['control']} | {changes('curve')} | {changes('instant')} | {delta(candidate['cold_training_seconds'], control['cold_training_seconds'], 2, 2)} | {delta(candidate['reviews_per_second'], control['reviews_per_second'], 0, 2)} | {bounds('curve')} / {bounds('instant')} |")
        if not result["against_controls"]:
            lines.append("| pending | — | — | — | — | — |")
    lines += ["", "Quality gate: all four paired-user one-sided bootstrap upper bounds ≤0.001 against both controls; 10,000 resamples, seed 20251012, existing Bonferroni divisor 64 retained. Repeated controls agree within 0.001; cold training faster than both. User uncertainty is conditional on one training seed.", "",
              f"B01 accepted for both models: **{assessment['B01_accepted_both_models']}**. The bundle result does not independently validate every component.", "",
              "Historical F01–F04 results remain preserved under their original optimized-executor mode; they are not the matched controls for B01.", "", "## Process-start source", "",
              "| Case | Experiment commit | Dirty | Source archive |", "|---|---|---|---|"]
    for row in suite["results"]:
        provenance = row.get("provenance", {})
        archive = provenance.get("source_archive")
        archive_link = Path(os.path.relpath(ROOT / archive, output)).as_posix() if archive else None
        lines.append(f"| {row['model']}/{row['track']} | {provenance.get('git_commit', '—')} | {provenance.get('git_dirty', '—')} | " + (f"[source.zip]({archive_link})" if archive_link else "—") + " |")
    lines += ["", "Implementation remains uncommitted. Missing values remain pending; raw failed and inconclusive attempts are retained.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "paired-comparisons.json").write_text(json.dumps(assessment, indent=2) + "\n", encoding="utf-8")
    if update_document and not any(row.get("smoke_only") for row in suite["results"]):
        path = ROOT / "docs/GPU_OPTIMIZATIONS.md"
        existing = path.read_text(encoding="utf-8")
        link = output.relative_to(ROOT).as_posix()
        block = "\n".join([BEGIN, "### B01 comparison progress", "", f"Suite: **{suite['status']}**; complete cases: **{sum(row['status'] == 'complete' for row in suite['results'])}/6**.", "",
                           *[f"- {value['model']}: {value['status']}." for value in assessment["models"]], "",
                           f"[Four held-out losses, absolute/percentage deltas against both controls, uncertainty, full-history timing and revisions](../{link}/summary.md).", END])
        if BEGIN in existing:
            begin, end = existing.index(BEGIN), existing.index(END) + len(END)
            existing = existing[:begin] + block + existing[end:]
        else:
            existing += "\n" + block + "\n"
        path.write_text(existing, encoding="utf-8")
