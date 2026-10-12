"""Report conditional paired-user quality uncertainty and complete training time."""

import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRACKS = [f"F{i:02d}" for i in range(1, 9)]


def paired(candidate, control):
    for key in ("implementation_sha256", "initial_parameter_sha256", "validation_input_sha256", "updates", "review_presentations",
                "training_curve_labels", "training_instant_labels", "evaluation_users"):
        if candidate.get(key) != control.get(key):
            raise ValueError(f"Unmatched paired field: {key}")
    if candidate.get("smoke_only") or candidate["evaluation_users"] != 500:
        return {"status": "smoke only; not validation"}
    a = {row["user_id"]: row for row in candidate["per_user"]}
    b = {row["user_id"]: row for row in control["per_user"]}
    if set(a) != set(b) or len(a) != 500:
        raise ValueError("Distinct held-out user sets do not match")
    rng = np.random.default_rng(20251012)
    indices = rng.integers(0, len(a), size=(10000, len(a)))
    values = {}
    for name, count in (("curve_log_loss", "curve_labels"), ("instant_log_loss", "instant_labels")):
        users = sorted(a)
        weights = np.array([a[user][count] for user in users], dtype=float)
        if not all(a[user][count] == b[user][count] for user in users):
            raise ValueError("Paired held-out label counts differ")
        delta = np.array([a[user][name] - b[user][name] for user in users])
        samples = delta[indices]
        sampled_weights = weights[indices]
        means = {"user_average": samples.mean(axis=1),
                 "label_weighted": (samples * sampled_weights).sum(axis=1) / sampled_weights.sum(axis=1)}
        for aggregation, distribution in means.items():
            values[name + "_" + aggregation] = {
                "delta": float(delta.mean() if aggregation == "user_average" else np.average(delta, weights=weights)),
                "ci95": [float(v) for v in np.quantile(distribution, [0.025, 0.975])],
                "familywise_upper": float(np.quantile(distribution, 1 - 0.05 / 64))}
    return {"status": "paired", "metrics": values,
            "quality_pass": all(value["familywise_upper"] <= 0.001 for value in values.values())}


def write_report(output, suite):
    rows = {(value["model"], value["track"]): value for value in suite["results"]}
    lines = ["# First GPU optimization batch: 500-user temporal comparison", "",
             f"Suite status: **{suite['status']}**. Missing values are pending, not zero.", "",
             "One training seed (2025), same 500 known users and complete chronological splits per case.",
             "The bootstrap measures user uncertainty conditional on this seed. Startup, JIT primer,",
             "cache creation and validation are separately recorded in each result JSON. F02's cold",
             "training seconds include cache creation; warm reuse timing is separate. CUDA Graph",
             "capture is inside training time. Raw presentations include every training review once.", ""]
    comparisons = []
    for model in ("baseline", "expertium"):
        control = rows.get((model, "control"))
        repeated = rows.get((model, "control-repeat"))
        stable = bool(control and repeated and control["status"] == repeated["status"] == "complete")
        if stable:
            stable = all(abs(control["evaluation"][key] - repeated["evaluation"][key]) <= 0.001
                         for key in control["evaluation"] if "log_loss" in key)
            stable = stable and all(control.get(key) == repeated.get(key) for key in
                                    ("implementation_sha256", "initial_parameter_sha256", "validation_input_sha256",
                                     "updates", "review_presentations", "training_curve_labels", "training_instant_labels"))
        lines += [f"## {model}", "", "| Case | Status | Curve LL weighted / user | Instant LL weighted / user | Train s / cold s | Reviews/s | Peak reserved MiB |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for track in ["control", *TRACKS, "control-repeat"]:
            value = rows.get((model, track))
            if not value or value["status"] != "complete":
                status = value["status"] if value else "pending"
                if suite.get("active_case") == {"model": model, "track": track} and not value:
                    status = "running"
                lines.append(f"| {track} | {status} | — | — | — | — | — |")
                continue
            evaluation = value["evaluation"]
            lines.append(f"| [{track}]({model}/{track}/result.json) | complete | {evaluation['curve_log_loss_label_weighted']:.6f} / {evaluation['curve_log_loss_user_average']:.6f} | {evaluation['instant_log_loss_label_weighted']:.6f} / {evaluation['instant_log_loss_user_average']:.6f} | {value['training_seconds']:.2f} / {value['cold_training_seconds']:.2f} | {value['reviews_per_second']:,.0f} | {value['peak_reserved_mib']:.0f} |")
        lines += ["", "| Track | Δ Curve weighted / user | Δ Instant weighted / user | Adjusted upper bounds Curve / Instant (weighted, user) | Decision |",
                  "|---|---:|---:|---|---|"]
        for track in TRACKS:
            value = rows.get((model, track))
            if not (value and control and value["status"] == control["status"] == "complete"):
                lines.append(f"| {track} | — | — | — | pending matched results |")
                continue
            try:
                result = paired(value, control)
            except ValueError as exc:
                result = {"status": "unmatched", "error": str(exc)}
            if result["status"] != "paired":
                lines.append(f"| {track} | — | — | — | {result['status']}: {result.get('error', '')} |")
            else:
                metrics = result["metrics"]
                c = [metrics["curve_log_loss_" + kind] for kind in ("label_weighted", "user_average")]
                i = [metrics["instant_log_loss_" + kind] for kind in ("label_weighted", "user_average")]
                speed = stable and value["cold_training_seconds"] < min(control["training_seconds"], repeated["training_seconds"])
                if not stable:
                    decision = "pending repeated control" if not repeated else "inconclusive: controls disagree"
                elif result["quality_pass"] and speed:
                    decision = "validated for this protocol"
                elif not result["quality_pass"]:
                    decision = "quality non-inferiority not established"
                else:
                    decision = "no complete cold-training speed gain"
                result.update(decision=decision, speed_pass=speed, controls_stable=stable)
                lines.append(f"| {track} | {c[0]['delta']:+.6f} / {c[1]['delta']:+.6f} | {i[0]['delta']:+.6f} / {i[1]['delta']:+.6f} | ({c[0]['familywise_upper']:+.6f}, {c[1]['familywise_upper']:+.6f}) / ({i[0]['familywise_upper']:+.6f}, {i[1]['familywise_upper']:+.6f}) | {decision} |")
            comparisons.append({"model": model, "track": track, **result})
        lines += ["", f"Controls stable: **{stable}**.", ""]
    lines += ["## Source revisions", "", "| Case | Experiment commit | Dirty | Preserved source |", "|---|---|---|---|"]
    for value in suite["results"]:
        provenance = value.get("provenance", {})
        archive = provenance.get("source_archive")
        source = f"[source.zip]({Path(os.path.relpath(ROOT / archive, output)).as_posix()})" if archive else "—"
        lines.append(f"| {value['model']}/{value['track']} | {provenance.get('git_commit', '—')} | {provenance.get('git_dirty', '—')} | {source} |")
    lines += ["", "All eight tracks remain visible, including pending/failed/inconclusive cases.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "paired-comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n", encoding="utf-8")
    if output.name == "gpu-batch1-500" and not any(r.get("smoke_only") for r in suite["results"]):
        update_document(output, suite, comparisons)


def update_document(output, suite, comparisons):
    path = ROOT / "docs/GPU_OPTIMIZATIONS.md"
    text = path.read_text(encoding="utf-8")
    link = output.relative_to(ROOT).as_posix()
    content = ["<!-- gpu-batch1-progress:start -->", "### First-batch progress", "",
               f"Suite: **{suite['status']}**; completed cases: **{sum(r['status'] == 'complete' for r in suite['results'])}/20**.", "",
               "| Track | Default | Expertium |", "|---|---|---|"]
    for track in TRACKS:
        statuses = []
        for model in ("baseline", "expertium"):
            comparison = next((r for r in comparisons if r["model"] == model and r["track"] == track), None)
            result = next((r for r in suite["results"] if r["model"] == model and r["track"] == track), None)
            status = comparison.get("decision", comparison["status"]) if comparison else result["status"] if result else "pending"
            if suite.get("active_case") == {"model": model, "track": track} and not result:
                status = "running"
            statuses.append(status)
        content.append(f"| {track} | {statuses[0]} | {statuses[1]} |")
    content += ["", f"[Curve/Instant losses, timing, uncertainty and experiment revisions](../{link}/summary.md).",
                "Measurements and pending cases are retained in the raw suite; implementation remains uncommitted.",
                "<!-- gpu-batch1-progress:end -->"]
    block = "\n".join(content)
    begin_marker, end_marker = "<!-- gpu-batch1-progress:start -->", "<!-- gpu-batch1-progress:end -->"
    if begin_marker in text:
        begin = text.index(begin_marker)
        end = text.index(end_marker, begin) + len(end_marker)
        text = text[:begin] + block + text[end:]
    else:
        text += "\n\n" + block + "\n"
    # Preserve the stable idea rows and their outstanding scope.
    for track in TRACKS:
        rows = text.splitlines()
        for index, line in enumerate(rows):
            if line.startswith(f"| {track} |"):
                cells = line.split("|")
                if len(cells) >= 9 and "Next experiment" not in line:
                    metrics, timings, decisions = [], [], []
                    for model, label in (("baseline", "Default"), ("expertium", "Expertium")):
                        result = next((r for r in suite["results"] if r["model"] == model and r["track"] == track), None)
                        comparison = next((r for r in comparisons if r["model"] == model and r["track"] == track), None)
                        if result and result["status"] == "complete":
                            values = result["evaluation"]
                            metrics.append(f"{label} weighted {values['curve_log_loss_label_weighted']:.6f} / {values['instant_log_loss_label_weighted']:.6f}; user {values['curve_log_loss_user_average']:.6f} / {values['instant_log_loss_user_average']:.6f}")
                            timings.append(f"{label} {result['training_seconds']:.2f} s / {result['reviews_per_second']:,.0f}")
                        else:
                            metrics.append(f"{label} — ({result['status'] if result else 'pending'})")
                            timings.append(f"{label} —")
                        if comparison:
                            decisions.append(f"{label}: {comparison.get('decision', comparison['status'])}")
                    cells[4] = " " + ("; ".join(decisions) if decisions else "Implemented; approved 500-user comparison pending") + " "
                    cells[5] = " " + "; ".join(metrics) + " "
                    cells[6] = " " + "; ".join(timings) + " "
                    rows[index] = "|".join(cells)
                    break
        text = "\n".join(rows) + "\n"
    path.write_text(text, encoding="utf-8")
