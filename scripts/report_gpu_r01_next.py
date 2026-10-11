"""Report a continuation against recorded accepted and original baseline runs."""

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
from scripts.gpu_r01_statistics import METRICS, paired_relative_losses
from scripts.gpu_r01_log import update_run_log
from scripts.gpu_r01_training import atomic_json
from scripts.gpu_r01_policy import POLICY, decide, read_reference, resource_reductions, result_metadata


def delta(value, reference):
    return 100 * (value / reference - 1)


def link(path, base):
    return Path(os.path.relpath(path, base)).as_posix()


def describe(track):
    names = {"control": "baseline", "F01": "ordered user prefetch", "F03": "packed worker communication",
             "F04": "overlapping GPU transfers", "F06": "reduced synchronization", "F07": "reduced padding",
             "F08": "fused shift operations", "F14": "GPU feature encodings", "F16": "more preparation workers"}
    return ", ".join(f"{part} ({names.get(part, 'see track record')})" for part in track.split("-"))


def parity(left, right, left_path, right_path):
    keys = ("step", "user_id", "offset", "reviews", "seed", "loss", "gradient_norm", "curve_labels", "instant_labels", "lr")
    logs = [[json.loads(line) for line in (path / "updates.jsonl").read_text().splitlines()]
            for path in (left_path, right_path)]
    mismatches = [i + 1 for i, (a, b) in enumerate(zip(*logs)) if any(a[key] != b[key] for key in keys)]
    fields = ("initial_parameter_sha256", "trained_parameter_sha256", "validation_input_sha256", "updates",
              "review_presentations", "training_curve_labels", "training_instant_labels", "coverage", "evaluation")
    checks = {key: left[key] == right[key] for key in fields}
    return {"fields": checks, "equal_update_counts": len(logs[0]) == len(logs[1]),
            "update_mismatch_count": len(mismatches), "first_update_mismatch": next(iter(mismatches), None),
            "exact_recorded_parity": all(checks.values()) and len(logs[0]) == len(logs[1]) and not mismatches}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/retraining/gpu-retrain-r01/iteration-0003"))
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    protocol = json.loads((output / "protocol.json").read_text())
    iteration = protocol["iteration"]
    selection = json.loads((output / "selection.json").read_text())
    implementation_path = output / "implementation-commit.json"
    implementation = json.loads(implementation_path.read_text()) if implementation_path.exists() else None
    settings = protocol["acceptance"]
    amendment_path = output / "reference-reuse-amendment.json"
    amendment = json.loads(amendment_path.read_text()) if amendment_path.exists() else {}
    cases, comparisons, references = {}, {}, {}
    sources = sorted(path for path in output.glob("*/*/result.json") if path.parent.parent.name in protocol["models"])
    for model in protocol["models"]:
        reused = protocol.get("loop_policy") == POLICY or model in amendment.get("reuse_models", [])
        references[model] = {}
        for kind, metadata in protocol["references"][model].items():
            reference, path = read_reference(metadata, ROOT)
            references[model][kind] = (reference, path.parent)
        control_track = protocol.get("control_tracks", {}).get(model, protocol.get("control_track", "control"))
        if reused:
            control, control_path = references[model]["previous"]
        else:
            control_path = output / model / control_track
            control = json.loads((control_path / "result.json").read_text())
        track = selection[model]["track"]
        candidate_path = output / model / track
        candidate = json.loads((candidate_path / "result.json").read_text())
        if control["status"] != "complete" or candidate["status"] != "complete":
            raise ValueError("Both complete matched runs are required")
        if not reused and (control["implementation_sha256"] != candidate["implementation_sha256"]
                or control["protocol_sha256"] != candidate["protocol_sha256"]):
            raise ValueError("Matched runs have different implementations/protocols")
        if reused and (candidate["implementation_sha256"] != protocol["implementation_sha256"]
                       or candidate["protocol_sha256"] != hashlib.sha256((output / "protocol.json").read_bytes()).hexdigest()):
            raise ValueError("Candidate differs from the frozen protocol/source")
        options = protocol.get("case_options_by_model", {}).get(model, protocol.get("case_options"))
        for key in ("options",):
            if candidate[key] != options[track] or control[key] != options[control_track]:
                raise ValueError("Run options differ from frozen protocol")
        for current in (control, candidate):
            for reference, _ in references[model].values():
                config = dict(current["config"])
                expected = dict(reference["config"])
                for key in ("PREFETCH_WORKERS", "PREFETCH_BATCHES"):
                    config.pop(key)
                    expected.pop(key)
                if config != expected:
                    raise ValueError("Training behavior differs from historical reference")
                for key in ("training_users", "evaluation_users", "updates", "review_presentations", "coverage", "initial_parameter_sha256", "validation_input_sha256"):
                    if current[key] != reference[key]:
                        raise ValueError(f"Exposure/cohort/seed mismatch: {key}")
        matched = references[model] if reused else {**references[model], "matched_incumbent": (control, control_path)}
        result = {"track": track, "references": {}, "parity": parity(candidate, control, candidate_path, control_path),
                  "incumbent_parity_vs_previous": parity(control, references[model]["previous"][0], control_path, references[model]["previous"][1])}
        for kind, (reference, reference_path) in matched.items():
            losses = paired_relative_losses(candidate["per_user"], reference["per_user"],
                                            resamples=settings["resamples"], bootstrap_seed=settings["bootstrap_seed"],
                                            family_tests=settings["family_tests"], alpha=settings["alpha"],
                                            relative_limit=settings["baseline_loss_limit"] if kind == "original_baseline" else settings["relative_loss_limit"])
            result["references"][kind] = {"path": (reference_path / "result.json").relative_to(ROOT).as_posix(),
                                          "throughput_gain_percent": delta(candidate["reviews_per_second"], reference["reviews_per_second"]),
                                          "training_time_delta_percent": delta(candidate["training_seconds"], reference["training_seconds"]),
                                          "process_wall_delta_percent": delta(candidate["process_wall_seconds"], reference["process_wall_seconds"]),
                                          "losses": losses}
        gain = result["references"]["previous"]["throughput_gain_percent"]
        reductions = resource_reductions(candidate_path / "result.json", control_path / "result.json", ROOT)
        result.update(decide(gain, {kind: item["losses"] for kind, item in result["references"].items()},
                             relative_limit=settings["relative_loss_limit"], baseline_limit=settings["baseline_loss_limit"],
                             prior_streak=amendment.get("prior_no_20_percent_gain_streak", protocol["prior_no_20_percent_gain_streak"])[model], resources_reduced=bool(reductions)))
        result["resource_reductions"] = reductions
        result["incumbent_after_iteration"] = track if result["accepted_main"] else control_track
        result["accepted_reference"] = result_metadata(candidate_path / "result.json", ROOT) if result["accepted_main"] else protocol["references"][model]["previous"]
        if result["resource_retuning_pending"]:
            result["retuning_candidate"] = result_metadata(candidate_path / "result.json", ROOT)
            result["retuning_next_step"] = "Use the recorded resource reduction to investigate higher preparation parallelism or batching; preserve cohort and disclose any changed update schedule. Conditional capacity estimates are not measured gains."
        comparisons[model] = result
        cases[model] = {track: candidate} if reused else {control_track: control, track: candidate}
    report = {"iteration": iteration, "status": "complete preliminary iteration", "generated_utc": datetime.now(timezone.utc).isoformat(),
              "analysis_provenance": process_provenance(), "comparisons": comparisons, "implementation_commit": implementation,
              "loop_policy": POLICY,
              "limitations": "20 distinct held-out users; one training seed and one timing/configuration; no broader validation or timing-repeatability claim"}
    atomic_json(output / "comparison.json", report)
    lines = [f"# R01 iteration {iteration}", "", "Status: **complete preliminary screen**. One new iteration requested on 2026-10-11.", "",
             f"Same fixed **80 training / 20 unseen held-out users**, **{protocol['training_reviews']:,} training / {protocol['validation_reviews']:,} evaluation reviews**, seed 2025, one epoch, BF16 compute/FP32 master. Chunks contain 128 reviews; prefetch is bounded at twice the worker count. Exposure, update order, per-update seeds, optimizer and learning-rate schedule remain fixed. Exact worker/chunk settings for each case are preserved in its raw config.", "",
             "[Frozen protocol](protocol.json), [selection from recorded profiles](selection.json), [loop definition](loop-definition.md), [raw comparison](comparison.json). References retain their original process provenance. Timing includes complete training loads, worker startup/shutdown, preparation, transfers, updates and logs. Setup/primer, save and evaluation are separate; process wall includes them.", "",
             "Acceptance: measured relative regression ≤1% for all four binary losses against the previous accepted run; ≤2% versus the fixed original baseline is preferred. Confidence bounds are reported separately. Faster but less accurate variants remain separate opt-in fastmode. Accurate resource-saving candidates without a speed gain remain pending retuning. Target ≥20% reviews/s gain versus the previous accepted run; stop each model after three consecutive misses. This remains a preliminary screen; broader validation is pending.", "",
             "## Full-history measurements", "", "| Model / IDs | Training s | Reviews/s | Gain vs previous / original baseline | Process wall s | Weighted Curve / Instant | User-average Curve / Instant |",
             "|---|---:|---:|---|---:|---|---|"]
    for model, runs in cases.items():
        for track, case in runs.items():
            e = case["evaluation"]
            gains = [delta(case["reviews_per_second"], references[model][kind][0]["reviews_per_second"]) for kind in ("previous", "original_baseline")]
            lines.append(f"| {model}: {describe(track)} | {case['training_seconds']:.3f} | {case['reviews_per_second']:,.0f} | " + " / ".join(f"{g:+.2f}%" for g in gains) +
                         f" | {case['process_wall_seconds']:.3f} | {e[METRICS[0]]:.9f} / {e[METRICS[2]]:.9f} | {e[METRICS[1]]:.9f} / {e[METRICS[3]]:.9f} |")
    lines += ["", "## Resource evidence and decisions", ""]
    for model, result in comparisons.items():
        lines += [f"{model}: {selection[model]['reason']}", ""]
        for track in cases[model]:
            profile = output / "profiles" / f"{model}-{track}" / "result.json"
            if profile.exists():
                p = json.loads(profile.read_text())
                c = p["phases"]["training including user transitions"]["counters"]
                lines += [f"{describe(track)}: GPU activity **{c['nvidia_utilization.gpu']['mean']:.1f}%**, worker-pool CPU **{c['worker_pool_cpu_percent']['mean']:.1f}%**, peak reserved VRAM **{c['torch_cuda_peak_reserved_bytes']['max']/2**30:.2f} GiB**, minimum available system RAM **{c['ram_available_bytes']['min']/2**30:.2f} GiB**. [Resource profile]({link(profile.with_name('profile.md'), output)}). Full-history sampling-window sufficiency: `{p['workload']['full_history_profile_available']}`; GPU activity is not occupancy.", ""]
        lines += [f"Decision: **{result['decision']}**. Exact recorded parity versus previous accepted run: **{result['parity']['exact_recorded_parity']}**; logged update mismatches: **{result['parity']['update_mismatch_count']}**. Consecutive iterations without a 20% gain: **{result['no_20_percent_gain_streak']}**. Three-iteration stop threshold reached: **{result['no_20_percent_gain_streak'] >= 3}**.", "",
                  "| Binary loss | Candidate | Δ vs previous / original | Familywise upper vs previous / original |", "|---|---:|---|---|"]
        for key in METRICS:
            metrics = [result["references"][kind]["losses"]["metrics"][key] for kind in ("previous", "original_baseline")]
            lines.append(f"| {key} | {metrics[0]['candidate']:.9f} | " + " / ".join(f"{m['relative_delta_percent']:+.2f}%" for m in metrics) + " | " + " / ".join(f"{100*m['relative_familywise_upper']:+.2f}%" for m in metrics) + " |")
        lines.append("")
        if result["resource_retuning_pending"]:
            lines += [result["retuning_next_step"], ""]
    lines += ["## Evidence and provenance", "", "Each process captures its own dirty source snapshot and full commit at start; the shared training-implementation fingerprint is checked at every user boundary. Historical references retain their original revisions. [Ordered-prefetch contracts](prefetch-checks-v1/result.json); functional smoke results are separate from full-history timing.", ""]
    for path in [*sources, *sorted(output.glob("functional-smoke-v1/*/*/result.json"))]:
        case = json.loads(path.read_text())
        p = case["provenance"]
        lines.append(f"- [{case['model']}/{case['track']}{' smoke' if case['smoke_only'] else ''}]({link(path,output)}): {case['status']}; training start UTC {case.get('training_started_utc','—')}; commit `{p['git_commit']}`, dirty `{p['git_dirty']}`; [source snapshot]({link(ROOT/p['source_archive'], output)}), [updates]({link(path.with_name('updates.jsonl'),output)}), [checkpoint]({link(path.with_name('master.pth'),output)}).")
    lines += ["", f"Implementation commit: `{implementation['git_commit']}` (after experiments)." if implementation else "Implementation commit pending the completed speed/quality decision.", "", report["limitations"] + ". This iteration is complete. The latest user request authorizes five completed current-loop iterations in total, with the loop reread between runs; no broader validation is authorized.", ""]
    update = output / "authorization-update-001.json"
    if update.exists():
        lines += ["[Updated continuation authorization](authorization-update-001.json), [revised loop definition](loop-definition-update-001.md). The revised definition separates run-log columns, rounds percentages to two decimals, commits opt-in fastmode too, and asks whether freed resources can be used to increase throughput.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    index = ROOT / f"artifacts/retraining/gpu-retrain-r01/iterations/iteration-{iteration}.md"
    index.write_text(f"# R01 iteration {iteration}\n\nComplete preliminary screen. [Full report](../iteration-{iteration}/summary.md), [protocol](../iteration-{iteration}/protocol.json), [comparison](../iteration-{iteration}/comparison.json).\n", encoding="utf-8")
    update_run_log(output)
    print(json.dumps({model: {"track": c["track"], "gain_percent": c["references"]["previous"]["throughput_gain_percent"],
                              "exact": c["parity"]["exact_recorded_parity"], "decision": c["decision"]} for model,c in comparisons.items()}))


if __name__ == "__main__":
    main()
