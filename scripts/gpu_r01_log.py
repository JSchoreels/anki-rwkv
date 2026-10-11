"""Render the R01 run ledger from recorded results without running experiments."""

from argparse import ArgumentParser
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/retraining/gpu-retrain-r01"
MISSING = "—"
LOSSES = (
    ("Instant User", "instant_log_loss_user_average"),
    ("Instant Weighted", "instant_log_loss_label_weighted"),
    ("Curve User", "curve_log_loss_user_average"),
    ("Curve Weighted", "curve_log_loss_label_weighted"),
)
DESCRIPTIONS = {
    "control": "Baseline with prefetch across users disabled.",
    "F01": "Prepare the next user while the current user trains, preserving ordered updates and review exposure.",
    "F03": "Keep preparation in workers and pack interprocess transfers.",
    "F04": "Overlap asynchronous GPU transfers with training.",
    "F06": "Defer internal loss/count synchronizations, consolidate metric transfers and buffer logs while retaining finite-loss and gradient checks.",
    "F16": "Raise Expertium preparation from six to eight workers with bounded queues.",
}
FORMAT_NOTES = """<!-- r01-log-format:start -->
One row records each training attempt, including failed attempts and functional smoke runs. Each value and each percentage delta has its own column. Losses are held-out binary logloss; User is the unweighted user mean and Weighted is the label-weighted mean. Percentages use `100 × (run / reference − 1)`, rounded to two decimal places; positive speed deltas are faster and positive loss deltas are worse.

Previous means the preceding completed comparable full-history run of the same model. Baseline means the latest preceding comparable baseline/control run (no-prefetch control in iteration 0002, F01 incumbent in iteration 0003). The reference columns link the exact runs used. Failed, unfinished and smoke runs do not become comparison references; missing values and unavailable comparisons are `—`.

The fixed nested cohorts and 20% held-out split are recorded in [GPU_OPTIMIZATIONS.md](GPU_OPTIMIZATIONS.md#current-loop-fixed-cohorts-and-iteration-0002-protocol). Training users reached are actual counts; incomplete runs retain their planned counts in Notes. Training reviews count raw review presentations. Training seconds include loads, preparation, worker startup/shutdown, transfers, updates and logs; setup/primer, checkpoint save and evaluation are separate. Incomplete rows show partial elapsed training time and no final throughput. Full-history runs remain preliminary screens; smoke runs are infrastructure checks. Process-start provenance and source snapshots are preserved in the linked raw results and iteration records.
<!-- r01-log-format:end -->"""


def number(value, spec):
    if type(value) not in (int, float) or not math.isfinite(value):
        return MISSING
    return format(value, spec)


def percent(value, reference):
    if (type(value) not in (int, float) or type(reference) not in (int, float)
            or not math.isfinite(value) or not math.isfinite(reference) or reference <= 0):
        return MISSING
    change = round(100 * (value / reference - 1), 2)
    return f"{0.0 if change == 0 else change:+.2f}%"


def link(path, label):
    return f"[{label}]({Path(os.path.relpath(path, ROOT / 'docs')).as_posix()})"


def table_row(values):
    return "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in values) + " |"


def load_runs(output):
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    sources = [path for path in output.glob("*/*/result.json") if path.parent.parent.name in protocol["models"]]
    sources += list(output.glob("functional-smoke-*/*/*/result.json"))
    return [{"path": path, "case": json.loads(path.read_text(encoding="utf-8")),
             "output": output, "control": protocol.get("control_track", "control")}
            for path in sources]


def comparable(left, right):
    """Execution tuning may change worker counts, but not exposure or seeds."""
    if left.get("model") != right.get("model"):
        return False
    keys = ("training_users", "evaluation_users", "updates", "review_presentations",
            "initial_parameter_sha256", "validation_input_sha256")
    if any(key not in left or key not in right or left[key] != right[key] for key in keys):
        return False
    configs = [{key: value for key, value in case.get("config", {}).items()
                if key not in ("PREFETCH_WORKERS", "PREFETCH_BATCHES")} for case in (left, right)]
    labels = [tuple(case.get("evaluation", {}).get(task + "_labels") for task in ("curve", "instant"))
              for case in (left, right)]
    return bool(configs[0]) and configs[0] == configs[1] and labels[0] == labels[1] and all(labels[0])


def references(run, runs):
    case = run["case"]
    if case.get("smoke_only") or case.get("status") != "complete":
        return None, None
    prior = [other for other in runs if other is not run
             and other["case"].get("training_started_utc", "") < case.get("training_started_utc", "")
             and other["case"].get("status") == "complete" and not other["case"].get("smoke_only")
             and comparable(case, other["case"])]
    prior.sort(key=lambda other: other["case"].get("training_started_utc", ""))
    controls = [other for other in prior if other["case"]["track"] == other["control"]]
    return (prior[-1] if prior else None, controls[-1] if controls else None)


def run_row(run, runs):
    case, path = run["case"], run["path"]
    complete = case.get("status") == "complete"
    smoke = case.get("smoke_only", False)
    previous, baseline = references(run, runs)
    refs = (previous, baseline)
    ids = case.get("track", "control").split("-")
    description = " ".join(DESCRIPTIONS.get(track, f"Track {track}; see the iteration record.") for track in ids)
    provenance = case.get("provenance", {})
    evaluation = case.get("evaluation") or {}
    if complete:
        reached = case.get("training_users")
    else:
        reached = case.get("progress_users")
    notes = "Infrastructure check; excluded from comparisons." if smoke else ""
    if not smoke and case.get("track") == run["control"]:
        notes += " Baseline/control for this iteration."
    if not complete:
        notes += f" Partial attempt; {case.get('training_users', MISSING)} training users planned; {case.get('evaluation_users', MISSING)} held-out users planned."
    if path.parent.name != case.get("track"):
        notes += f" Preserved attempt: {path.parent.name}."
    values = [case.get("training_started_utc", MISSING),
              f"`{provenance['git_commit']}`" if provenance.get("git_commit") else MISSING,
              {"baseline": "Alex (default)", "expertium": "Expertium (light)"}.get(case.get("model"), case.get("model", MISSING)),
              "Baseline" if ids == ["control"] else ", ".join(ids), description,
              number(reached, ",.0f"),
              number(case.get("review_presentations"), ",.0f"), number(case.get("updates"), ",.0f"),
              number(case.get("training_seconds") if complete else case.get("training_elapsed_seconds"), ".3f"),
              number(case.get("reviews_per_second") if complete else None, ",.0f")]
    values += [percent(case.get("reviews_per_second"), ref["case"].get("reviews_per_second")) if ref else MISSING for ref in refs]
    for _, metric in LOSSES:
        values.append(number(evaluation.get(metric), ".9f"))
        values += [percent(evaluation.get(metric), ref["case"].get("evaluation", {}).get(metric)) if ref else MISSING for ref in refs]
    values += [link(ref["path"], ref["output"].name + ": " + ref["case"]["track"]) if ref else MISSING for ref in refs]
    values += [link(path, "result"), notes.strip() or MISSING]
    return table_row(values)


def render_block(output, runs):
    iteration = output.name.removeprefix("iteration-")
    # Column additions require explicit user confirmation (AGENT.md).
    headers = ["Training start (UTC)", "Trial commit", "Model", "Improvement IDs", "What was tried",
               "Training users reached", "Training reviews", "Updates", "Training seconds", "Reviews/s",
               "Speed Δ vs previous", "Speed Δ vs baseline"]
    for label, _ in LOSSES:
        headers += [label, label + " Δ vs previous", label + " Δ vs baseline"]
    headers += ["Previous reference", "Baseline reference", "Raw result", "Notes"]
    record = output / "summary.md"
    if not record.exists():
        record = output / "protocol.json"
    lines = [f"<!-- r01-{iteration}:start -->", f"## R01 iteration {iteration}", "",
             link(record, "Iteration record"), "", table_row(headers), table_row(["---"] * len(headers))]
    own_runs = sorted((run for run in runs if run["output"] == output), key=lambda run: run["case"].get("training_started_utc", ""))
    lines += [run_row(run, runs) for run in own_runs]
    implementation_path = output / "implementation-commit.json"
    if implementation_path.exists():
        implementation = json.loads(implementation_path.read_text(encoding="utf-8"))
        lines += ["", f"Implementation commit after experiments: `{implementation['git_commit']}`. Tested revision remains `{implementation['experiment_git_commit']}` with per-process source snapshots in the linked raw results and iteration record; " + link(implementation_path, "commit record") + "."]
    lines += ["", f"<!-- r01-{iteration}:end -->"]
    return "\n".join(lines)


def update_run_log(output=None):
    outputs = sorted(path for path in BASE.glob("iteration-*") if path.is_dir() and (path / "protocol.json").exists())
    if output is not None:
        output = Path(output).resolve()
        if output not in outputs:
            outputs.append(output)
    runs = [run for directory in outputs for run in load_runs(directory)]
    path = ROOT / "docs/GPU_OPTIMIZATIONS_LOG.md"
    text = path.read_text(encoding="utf-8") if path.exists() else "# GPU retraining run log\n"
    format_start, format_end = "<!-- r01-log-format:start -->", "<!-- r01-log-format:end -->"
    if format_start in text:
        begin = text.index(format_start)
        end = text.index(format_end, begin) + len(format_end)
        text = text[:begin] + FORMAT_NOTES + text[end:]
    else:
        first = text.find("<!-- r01-")
        first = first if first >= 0 else len(text)
        text = text[:first].rstrip() + "\n\n" + FORMAT_NOTES + "\n\n" + text[first:]
    for directory in ([output] if output is not None else outputs):
        if not any(run["output"] == directory for run in runs):
            continue
        block = render_block(directory, runs)
        marker = block.splitlines()[0]
        end_marker = marker.replace(":start", ":end")
        if marker in text:
            begin = text.index(marker)
            end = text.index(end_marker, begin) + len(end_marker)
            text = text[:begin] + block + text[end:]
        else:
            text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Refresh only this iteration; by default refresh all recorded R01 iterations")
    args = parser.parse_args()
    print(update_run_log(args.output))
