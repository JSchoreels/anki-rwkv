"""Report a continuation screen against matched, previous and original controls."""

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


def delta(value, reference):
    return 100 * (value / reference - 1)


def link(path, base):
    return Path(os.path.relpath(path, base)).as_posix()


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
    cases, comparisons, references = {}, {}, {}
    sources = sorted(path for path in output.glob("*/*/result.json") if path.parent.parent.name in protocol["models"])
    for model in protocol["models"]:
        references[model] = {}
        for kind, metadata in protocol["references"][model].items():
            path = ROOT / metadata["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
                raise ValueError("Frozen historical reference changed")
            references[model][kind] = (json.loads(path.read_text()), path.parent)
        control_path = output / model / protocol["control_track"]
        control = json.loads((control_path / "result.json").read_text())
        track = selection[model]["track"]
        candidate_path = output / model / track
        candidate = json.loads((candidate_path / "result.json").read_text())
        if control["status"] != "complete" or candidate["status"] != "complete":
            raise ValueError("Both complete matched runs are required")
        if (control["implementation_sha256"] != candidate["implementation_sha256"]
                or control["protocol_sha256"] != candidate["protocol_sha256"]):
            raise ValueError("Matched runs have different implementations/protocols")
        for key in ("options",):
            if candidate[key] != protocol["case_options"][track] or control[key] != protocol["case_options"][protocol["control_track"]]:
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
        matched = {**references[model], "matched_incumbent": (control, control_path)}
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
        gain = result["references"]["matched_incumbent"]["throughput_gain_percent"]
        accurate = all(result["references"][kind]["losses"]["all_four_strict_upper_bounds_below_limit"] for kind in ("matched_incumbent", "previous"))
        result.update(faster=gain > 0, passes_preliminary_quality=accurate, meets_20_percent_target=gain >= 20,
                      no_20_percent_gain_streak=0 if gain >= 20 else protocol["prior_no_20_percent_gain_streak"][model] + 1,
                      decision="retain opt-in improvement; preliminary only" if gain > 0 and accurate else
                               "retain separate opt-in fastmode; quality screen failed" if gain > 0 else "retain F01 incumbent; speed screen failed")
        result["incumbent_after_iteration"] = track if gain > 0 and accurate else protocol["control_track"]
        comparisons[model] = result
        cases[model] = {protocol["control_track"]: control, track: candidate}
    report = {"iteration": iteration, "status": "complete preliminary iteration", "generated_utc": datetime.now(timezone.utc).isoformat(),
              "analysis_provenance": process_provenance(), "comparisons": comparisons, "implementation_commit": implementation,
              "limitations": "20 distinct held-out users; one training seed and one timing/configuration; no broader validation or timing-repeatability claim"}
    atomic_json(output / "comparison.json", report)
    lines = [f"# R01 iteration {iteration}", "", "Status: **complete preliminary screen**. One new iteration requested on 2026-10-11.", "",
             f"Same fixed **80 training / 20 unseen held-out users**, **{protocol['training_reviews']:,} training / {protocol['validation_reviews']:,} evaluation reviews**, seed 2025, one epoch, BF16 compute/FP32 master. Alex uses 128 chunks/4 workers; Expertium control 256 chunks/6 workers, F16 candidate 8 workers. Chunks contain 128 reviews; prefetch is bounded at twice the worker count. Exposure, update order, per-update seeds, optimizer and learning-rate schedule remain fixed.", "",
             "[Frozen protocol](protocol.json), [selection from fresh profiles](selection.json), [loop definition](loop-definition.md), [raw comparison](comparison.json). Deterministic execution and original evaluation path are unchanged. Timing includes complete training loads, worker startup/shutdown, preparation, transfers, updates and logs. Setup/primer, save and evaluation are separate; process wall includes them. Resource-monitor overhead is included equally.", "",
             "Acceptance: strict <1% relative regression for separate weighted/user-average Curve/Instant losses versus both fresh F01 incumbent and previous F01; preferably <2% versus original baseline. Paired-user bootstrap: 10,000 resamples, seed 20261012, alpha 0.05, family size 24. Target ≥20% reviews/s gain versus matched incumbent. This remains a preliminary 20-evaluation-user screen; the 500-evaluation-user validation gate is pending.", "",
             "## Full-history measurements", "", "| Model / IDs | Training s | Reviews/s | Gain vs matched / previous / original baseline | Process wall s | Weighted Curve / Instant | User-average Curve / Instant |",
             "|---|---:|---:|---|---:|---|---|"]
    for model, runs in cases.items():
        for track, case in runs.items():
            e = case["evaluation"]
            gains = [delta(case["reviews_per_second"], ref[0]["reviews_per_second"]) for ref in
                     ((runs[protocol["control_track"]], None), references[model]["previous"], references[model]["original_baseline"])]
            lines.append(f"| {model}/{track} | {case['training_seconds']:.3f} | {case['reviews_per_second']:,.0f} | " + " / ".join(f"{g:+.2f}%" for g in gains) +
                         f" | {case['process_wall_seconds']:.3f} | {e[METRICS[0]]:.9f} / {e[METRICS[2]]:.9f} | {e[METRICS[1]]:.9f} / {e[METRICS[3]]:.9f} |")
    lines += ["", "## Resource evidence and decisions", ""]
    for model, result in comparisons.items():
        lines += [f"{model}: {selection[model]['reason']}", ""]
        for track in cases[model]:
            profile = output / "profiles" / f"{model}-{track}" / "result.json"
            if profile.exists():
                p = json.loads(profile.read_text())
                c = p["phases"]["training including user transitions"]["counters"]
                lines += [f"{track}: GPU activity **{c['nvidia_utilization.gpu']['mean']:.1f}%**, worker-pool CPU **{c['worker_pool_cpu_percent']['mean']:.1f}%**, peak reserved VRAM **{c['torch_cuda_peak_reserved_bytes']['max']/2**30:.2f} GiB**, minimum available system RAM **{c['ram_available_bytes']['min']/2**30:.2f} GiB**. [Resource profile]({link(profile.with_name('profile.md'), output)}). Full-history sampling-window sufficiency: `{p['workload']['full_history_profile_available']}`; GPU activity is not occupancy.", ""]
        lines += [f"Decision: **{result['decision']}**. Exact recorded parity versus fresh incumbent: **{result['parity']['exact_recorded_parity']}**; logged update mismatches: **{result['parity']['update_mismatch_count']}**. Fresh incumbent parity versus previous iteration: **{result['incumbent_parity_vs_previous']['exact_recorded_parity']}**. Consecutive iterations without 20% matched gain: **{result['no_20_percent_gain_streak']}**. The three-iteration stop threshold is not reached.", "",
                  "| Binary loss | Candidate | Δ vs matched / previous / original | Familywise upper vs matched / previous / original |", "|---|---:|---|---|"]
        for key in METRICS:
            metrics = [result["references"][kind]["losses"]["metrics"][key] for kind in ("matched_incumbent", "previous", "original_baseline")]
            lines.append(f"| {key} | {metrics[0]['candidate']:.9f} | " + " / ".join(f"{m['relative_delta_percent']:+.2f}%" for m in metrics) + " | " + " / ".join(f"{100*m['relative_familywise_upper']:+.2f}%" for m in metrics) + " |")
        lines.append("")
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
    print(json.dumps({model: {"track": c["track"], "gain_percent": c["references"]["matched_incumbent"]["throughput_gain_percent"],
                              "exact": c["parity"]["exact_recorded_parity"], "decision": c["decision"]} for model,c in comparisons.items()}))


if __name__ == "__main__":
    main()
