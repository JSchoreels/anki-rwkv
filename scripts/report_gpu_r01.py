"""Report one R01 iteration without claiming broader validation."""

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance
from scripts.gpu_r01_statistics import METRICS, paired_relative_losses
from scripts.run_gpu_batch1 import atomic_json


def delta(value, reference):
    return 100 * (value / reference - 1)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/retraining/gpu-retrain-r01/iteration-0002"))
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    protocol = json.loads((output / "protocol.json").read_text())
    sources = sorted(path for path in output.glob("*/*/result.json") if path.parent.parent.name in protocol["models"])
    cases = { (r["model"], r["track"]): r for path in sources for r in [json.loads(path.read_text())] if path.parent.name == r["track"] }
    selection_path = output / "selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    implementation_path = output / "implementation-commit.json"
    implementation = json.loads(implementation_path.read_text()) if implementation_path.exists() else None
    comparisons = {}
    for model in protocol["models"]:
        reference = cases.get((model, "control"))
        selected = selection.get(model, {}).get("track")
        candidate = cases.get((model, selected))
        if not candidate or not reference or candidate["status"] != "complete" or reference["status"] != "complete":
            continue
        losses = paired_relative_losses(candidate["per_user"], reference["per_user"], resamples=10000,
                                        bootstrap_seed=20261012, family_tests=8, alpha=0.05, relative_limit=0.01)
        logs = [[json.loads(line) for line in (output / model / track / "updates.jsonl").read_text().splitlines()] for track in ("control", selected)]
        keys = ("step", "user_id", "offset", "reviews", "seed", "loss", "gradient_norm", "curve_labels", "instant_labels", "lr")
        mismatches = [index + 1 for index, (left, right) in enumerate(zip(*logs)) if any(left[key] != right[key] for key in keys)]
        parity = {key: candidate.get(key) == reference.get(key) for key in
                  ("implementation_sha256", "protocol_sha256", "initial_parameter_sha256", "trained_parameter_sha256", "validation_input_sha256",
                   "updates", "review_presentations", "training_curve_labels", "training_instant_labels", "coverage", "evaluation")}
        gain = delta(candidate["reviews_per_second"], reference["reviews_per_second"])
        comparisons[model] = {"track": selected, "losses": losses, "parity": parity,
                              "equal_update_counts": len(logs[0]) == len(logs[1]), "update_mismatch_count": len(mismatches), "first_update_mismatch": next(iter(mismatches), None),
                              "throughput_gain_percent": gain, "training_time_delta_percent": delta(candidate["training_seconds"], reference["training_seconds"]),
                              "process_wall_delta_percent": delta(candidate["process_wall_seconds"], reference["process_wall_seconds"]),
                              "meets_20_percent_speed_target": gain >= 20,
                              "no_20_percent_gain_streak_after_this_iteration": 0 if gain >= 20 else 1,
                              "decision": ("promising preliminary speed screen; broader validation pending" if gain >= 20 and losses["all_four_strict_upper_bounds_below_limit"] else
                                           "speed target missed" if gain < 20 else "quality screen failed; retain baseline")}
    report = {"status": "complete preliminary iteration" if len(comparisons) == 2 else "in progress",
              "iteration": "0002", "generated_utc": datetime.now(timezone.utc).isoformat(),
              "analysis_provenance": process_provenance(), "protocol": "protocol.json", "comparisons": comparisons,
              "implementation_commit": implementation,
              "limitations": "20 held-out users; one training seed and one timing measurement/configuration; no broader validation or repeatability claim"}
    atomic_json(output / "comparison.json", report)
    lines = ["# R01 iteration 0002", "", f"Status: **{report['status']}**. Current-loop authorization: one iteration on 2026-10-11.", "",
             "This round follows the current user-holdout procedure. It is separate from iteration 0001's unfinished B01 prerequisite. Historical attempts remain preserved.", "",
             f"Fixed cohort: **80 training users / 20 disjoint held-out users**; **{protocol['training_reviews']:,} full-history training reviews / {protocol['validation_reviews']:,} held-out reviews**. Evaluation concerns unseen users, using 128-review chunks and the original feature/loss path. No evaluation user receives gradient updates.", "",
             "Seed 2025; one epoch; BF16 compute/FP32 master; Default 128 chunks/4 workers, Expertium 256 chunks/6 workers; LR 7e-4, five-update warmup, clip 0.5. Optimized JIT executor disabled; deterministic Torch algorithms; CUBLAS workspace :4096:8; WKV tile32. Candidates preserve exposure, batching and update schedule within each architecture.", "",
             "Budget: two fresh architecture baselines, resource profiles, then one selected execution candidate per architecture. Strict <1% relative regression on all four binary losses against the previous run and the baseline; these are the same matched control in this first round. Target ≥20% raw reviews/s gain. Paired-user bootstrap: 10,000 resamples, seed 20261012, alpha 0.05, family size8. The 20-user screen cannot satisfy the repository's 500-evaluation-user validation gate.", "",
             "[Frozen protocol and exact run budgets](protocol.json), [persistent nested cohorts](../cohorts-user-holdout-v1.json), [raw comparisons and parity checks](comparison.json).", "",
             "## Measurements", "", "| Model / case | Training s | Raw reviews/s | Gain vs previous / baseline | Process wall s | Weighted Curve / Instant | User-average Curve / Instant |", "|---|---:|---:|---:|---:|---|---|"]
    for model in protocol["models"]:
        for track in ("control", selection.get(model, {}).get("track")):
            case = cases.get((model, track))
            if not case:
                continue
            if case["status"] != "complete":
                lines.append(f"| {model}/{track}: {case['status']} | — | — | — | — | — | — |")
                continue
            evaluation = case["evaluation"]
            gain = f"{comparisons[model]['throughput_gain_percent']:+.2f}% / same" if track != "control" and model in comparisons else "—"
            lines.append(f"| {model}/{track} | {case['training_seconds']:.3f} | {case['reviews_per_second']:,.0f} | {gain} | {case['process_wall_seconds']:.3f} | {evaluation[METRICS[0]]:.9f} / {evaluation[METRICS[2]]:.9f} | {evaluation[METRICS[1]]:.9f} / {evaluation[METRICS[3]]:.9f} |")
    lines += ["", "Training time includes complete histories, loads/workers/preparation, transfer, updates and logs. Setup/primer, checkpoint saving and evaluation are separate in each raw result. Process wall includes Python/import/setup, training, save and evaluation. Telemetry overhead is included equally in all cases; sampler collection time is not an estimate of critical-path slowdown.", "", "## Resource profiles and selection", ""]
    for model in protocol["models"]:
        profile_name = model + "-control-v2" if (output / "profiles" / (model + "-control-v2")).exists() else model + "-control"
        lines.append(f"- {model}: [baseline profile](profiles/{profile_name}/profile.md). " + (selection[model]["reason"] if model in selection else "Candidate selection pending baseline evidence."))
        if model in selection:
            scenarios = selection[model]["idealized_scenarios"]
            activity = scenarios["constant_GPU_work_per_review_at80_percent_activity"]
            wait = scenarios["remove_all_current_main_thread_future_waits_reviews_per_second"]
            lines[-1] += f" Idealized capacity scenarios: holding GPU work/review fixed at 80% activity gives {activity:,.0f} reviews/s; removing all currently measured main-thread future waits gives {wait:,.0f} reviews/s. These are conditional arithmetic scenarios, not measured candidate throughput or a forecast; CPU/IPC/load limits and unchanged work can dominate."
    if selection:
        lines += ["", "[Pre-run candidate selection, assumptions and scenario arithmetic](selection.json). The earlier `profiles/baseline-control/` result retains a hardcoded 500-user display caption from the historical reporter; its raw coverage is correct. The corrected generic-cohort rendering and independently captured analysis are in `profiles/baseline-control-v2/`."]
    for model in comparisons:
        candidate_profile = output / "profiles" / (model + "-F01") / "result.json"
        if candidate_profile.exists():
            baseline_profile_name = model + "-control-v2" if (output / "profiles" / (model + "-control-v2")).exists() else model + "-control"
            baseline_profile = json.loads((output / "profiles" / baseline_profile_name / "result.json").read_text())
            candidate = json.loads(candidate_profile.read_text())
            b = baseline_profile["phases"]["training including user transitions"]["counters"]
            c = candidate["phases"]["training including user transitions"]["counters"]
            lines += ["", f"{model} resource change: GPU activity {b['nvidia_utilization.gpu']['mean']:.1f}% → {c['nvidia_utilization.gpu']['mean']:.1f}%; worker-pool CPU {b['worker_pool_cpu_percent']['mean']:.1f}% → {c['worker_pool_cpu_percent']['mean']:.1f}%; peak reserved VRAM {c['torch_cuda_peak_reserved_bytes']['max'] / 2**30:.2f} GiB. [Complete candidate profile](profiles/{model}-F01/profile.md). Sampling-window sufficiency: {candidate['workload']['full_history_profile_available']}."]
    lines += ["", "## Quality, parity and decision", ""]
    for model, value in comparisons.items():
        exact = all(value["parity"].values()) and value["equal_update_counts"] and not value["update_mismatch_count"]
        lines.append(f"{model}/{value['track']}: **{value['decision']}**. Final parameter/input/coverage/score hashes and all logged update fields {'match exactly' if exact else 'do not all match'}; {value['update_mismatch_count']} logged update differences. Training time {value['training_time_delta_percent']:+.2f}%, process wall {value['process_wall_delta_percent']:+.2f}%. Exact parity here concerns this run's recorded tensors and metrics, not independent statistical validation.")
        lines += ["", "| Binary loss | Candidate | Delta vs previous / baseline | Relative one-sided familywise upper | <1% bound |", "|---|---:|---:|---:|---|"]
        for key, metric in value["losses"]["metrics"].items():
            lines.append(f"| {key} | {metric['candidate']:.9f} | {metric['relative_delta_percent']:+.6f}% / same | {100 * metric['relative_familywise_upper']:+.6f}% | {metric['strict_upper_below_limit']} |")
        lines.append("")
    if implementation:
        lines += [f"Successful F01/R01 code was committed after these experiments as `{implementation['git_commit']}`. The tested process-start revision remains `{implementation['experiment_git_commit']}` with dirty source snapshots; the later implementation commit is not represented as the experiment revision. [Commit record](implementation-commit.json).", ""]
    else:
        lines += ["Implementation commit: pending final comparisons and the current loop's commit/fastmode rule.", ""]
    if len(comparisons) == 2:
        lines += ["Iteration stop: the requested one-iteration budget is exhausted. Record each architecture's 20%-gain streak in the raw comparison; the three-iteration no-gain stop rule is not reached by this single round. No second iteration or larger validation run is started.", ""]
    lines += [report["limitations"] + ". No further iteration or larger cohort is launched by this report.", "", "## Run provenance and raw evidence", ""]
    lines.append("[Loop definition used for this iteration](loop-definition.md). The user's update adds committing faster/accurate code, retaining faster/less-accurate code as separate opt-in fastmode, and recording speed gains even when quality fails. A transient Windows progress-file write failure required one fresh-weight infrastructure retry; both the failed attempt and earlier protocols are preserved.")
    lines += ["", "Functional checks: [ordered-prefetch contracts](prefetch-checks-v1/result.json), [resource-profile/cohort guards](profile-checks-v2/result.json), [bounded Windows write retries](write-retry-checks-v1/result.json). The pre-amendment two-user GPU smoke artifacts remain in `functional-smoke-v1/`; their source/protocol are their own and are not claimed as the full run's source.", ""]
    for path in sources:
        if path.parent.name.endswith("failed-atomic-write-v1"):
            lines.append(f"- [Failed infrastructure attempt]({path.relative_to(output).as_posix()}): Windows transient progress-file replacement denial after 88 updates; preserved in full. Restarted from fresh weights with bounded write retries; not a quality or throughput result.")
    for (model, track), case in cases.items():
        p = case["provenance"]
        lines.append(f"- {model}/{track}: started training UTC {case.get('training_started_utc', '—')}; commit `{p['git_commit']}`, dirty `{p['git_dirty']}`; [source snapshot](../../../../{p['source_archive']}); [raw result]({model}/{track}/result.json), [updates]({model}/{track}/updates.jsonl), [resources]({model}/{track}/resources.jsonl), [checkpoint]({model}/{track}/master.pth). Implementation was uncommitted when this experiment started.")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    index = ROOT / "artifacts/retraining/gpu-retrain-r01/iterations/iteration-0002.md"
    index.write_text("# R01 iteration 0002\n\n" + f"Status: **{report['status']}**. [Open the full iteration record](../iteration-0002/summary.md), [frozen protocol](../iteration-0002/protocol.json), and [raw comparisons](../iteration-0002/comparison.json).\n", encoding="utf-8")
    log_path = ROOT / "docs/GPU_OPTIMIZATIONS_LOG.md"
    prefix = """# GPU retraining run log

Current R01 runs are recorded below. Prior B01/F01–F08 attempts remain in the [optimization history](GPU_OPTIMIZATION_HISTORY.md) and linked raw artifact reports; their missing historical revisions are not inferred.

<!-- r01-0002:start -->
## R01 iteration 0002

One iteration requested on 2026-10-11. F01 loads/prepares the next user while the current user trains, preserving ordered updates, review exposure and seeds. Baselines use the same loop with prefetch across users disabled. [Full iteration record](../artifacts/retraining/gpu-retrain-r01/iteration-0002/summary.md).

| Start of training (UTC) | Model / IDs | Train / test users | Training reviews / updates | Training s / reviews/s | Speed delta vs previous / baseline | Weighted Curve / Instant (delta vs previous / baseline) | User-average Curve / Instant | Trial commit / dirty / source |
|---|---|---:|---:|---|---|---|---|---|
"""
    for path in sorted(sources, key=lambda p: json.loads(p.read_text()).get("training_started_utc", "")):
        case = json.loads(path.read_text())
        p = case["provenance"]
        link = "../" + path.relative_to(ROOT).as_posix()
        if case["status"] != "complete":
            logged = [json.loads(line) for line in (path.parent / "updates.jsonl").read_text().splitlines()] if (path.parent / "updates.jsonl").exists() else []
            users = len({row["user_id"] for row in logged})
            partial = f"{case['training_elapsed_seconds']:.3f} partial / —" if "training_elapsed_seconds" in case else "—"
            prefix += f"| {case.get('training_started_utc', '—')} | [{case['model']}/{path.parent.name}: {case['status']}]({link}) | {users} reached / 0 evaluated ({case.get('training_users', '—')} / {case.get('evaluation_users', '—')} planned) | {case.get('review_presentations', '—')} / {case.get('updates', '—')} | {partial} | — | — | — | `{p['git_commit']}` / {p['git_dirty']} / [snapshot](../{p['source_archive']}) |\n"
            continue
        e = case["evaluation"]
        comparison = comparisons.get(case["model"]) if case["track"] != "control" else None
        gain = f"{comparison['throughput_gain_percent']:+.2f}% / same" if comparison else "—"
        losses = " / ".join(f"{e[key]:.9f}" + (f" ({comparison['losses']['metrics'][key]['relative_delta_percent']:+.6f}% / same)" if comparison else " (—)") for key in (METRICS[0], METRICS[2]))
        users = " / ".join(f"{e[key]:.9f}" + (f" ({comparison['losses']['metrics'][key]['relative_delta_percent']:+.6f}% / same)" if comparison else " (—)") for key in (METRICS[1], METRICS[3]))
        prefix += f"| {case['training_started_utc']} | [{case['model']}/{case['track']}]({link}) | {case['training_users']} / {case['evaluation_users']} | {case['review_presentations']:,} / {case['updates']} | {case['training_seconds']:.3f} / {case['reviews_per_second']:,.0f} | {gain} | {losses} | {users} | `{p['git_commit']}` / {p['git_dirty']} / [snapshot](../{p['source_archive']}) |\n"
    prefix += "\nFunctional smoke runs (two train/two disjoint test users, 1,024 training/1,024 held-out reviews each) are infrastructure checks, excluded from speed/quality comparisons:\n\n"
    for path in sorted(output.glob("functional-smoke-v1/*/*/result.json")):
        case = json.loads(path.read_text())
        p = case["provenance"]
        prefix += f"- [{case['model']}/{case['track']}: {case['status']}](../{path.relative_to(ROOT).as_posix()}), training UTC {case.get('training_started_utc', '—')}; {case.get('training_seconds', 0):.3f} s, {case.get('reviews_per_second', 0):,.0f} reviews/s; binary losses in raw result, no comparable previous/baseline; commit `{p['git_commit']}`, dirty `{p['git_dirty']}`, [snapshot](../{p['source_archive']}).\n"
    prefix += "\n<!-- r01-0002:end -->\n"
    if implementation:
        prefix = prefix.replace("\n<!-- r01-0002:end -->", f"\nImplementation commit after successful comparisons: `{implementation['git_commit']}`. Experiment revision remains `{implementation['experiment_git_commit']}` (dirty, per-run snapshots above); [commit record](../artifacts/retraining/gpu-retrain-r01/iteration-0002/implementation-commit.json).\n\n<!-- r01-0002:end -->")
    if log_path.exists():
        old = log_path.read_text(encoding="utf-8")
        marker = "<!-- r01-0002:start -->"
        if marker in old:
            begin = old.index(marker)
            end = old.index("<!-- r01-0002:end -->", begin) + len("<!-- r01-0002:end -->")
            prefix = old[:begin] + prefix[prefix.index(marker):].rstrip() + old[end:]
        else:
            prefix = old.rstrip() + "\n\n" + prefix[prefix.index(marker):]
    log_path.write_text(prefix, encoding="utf-8")
    print(json.dumps({"status": report["status"], "comparisons": {key: {name: value[name] for name in ("track", "throughput_gain_percent", "update_mismatch_count", "decision")} for key, value in comparisons.items()}}))


if __name__ == "__main__":
    main()
