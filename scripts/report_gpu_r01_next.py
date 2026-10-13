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
from scripts.gpu_r01_policy import POLICY, declared_differences, same_exposure, decide, material_reduction, read_reference, resource_reductions, result_metadata, effective_loop, resolve_case, latest_streak


def delta(value, reference):
    return 100 * (value / reference - 1)


def link(path, base):
    return Path(os.path.relpath(path, base)).as_posix()


def describe(track):
    names = {"control": "baseline", "F01": "ordered user prefetch", "F03": "packed worker communication",
             "F04": "overlapping GPU transfers", "F06": "reduced synchronization", "F07": "reduced padding",
             "F08": "fused shift operations", "F14": "GPU feature encodings", "F16": "more preparation workers",
             "F02": "preprocessing cache", "F19": "batched ID-encoding draws", "F17": "larger training batches",
             "F20": "bulk pinned transfers", "F21": "parallel held-out preparation", "F22": "ordinary-training execution mode",
             "F23": "grouped master/child copies", "F24": "fewer hierarchy splits", "F25": "single-thread main process",
             "E06": "retained iteration-0006 Expertium configuration: F01, F16, F14, F03, F04",
             "E11": "exact-parity part of the iteration-0011 Expertium candidate: F19, F23, F25, F20, F21",
             "E13": "accepted iteration-0013 Expertium configuration: E06, E11, F26, F27, F17",
             "F28": "worker pool started during setup", "F29": "cross-user batch packing", "F30": "exact NumPy sample construction",
             "F26": "sort-free exact shift backward", "F27": "sort-free exact row-selection backward",
             "E20": "accepted iteration-0020 Expertium configuration: E13, F30, F28",
             "F31": "exact row-tiled recurrence kernels", "E22": "accepted iteration-0022 Expertium configuration: E20, F31",
             "F32": "exact multi-SM GroupNorm weight/bias backward", "F33": "host reads deferred by one update",
             "E24": "accepted iteration-0024 Expertium configuration: E22, F32, F33",
             "F36": "dispatch thread on its own physical core", "F37": "loss without host synchronization", "F35": "exact row-selection backward and recurrence layout tuning", "F38": "shared shift-backward orders and stream-argument kernel launches", "F39": "six preparation workers", "F40": "five preparation workers",
             "E27": "accepted iteration-0027 Expertium configuration: E24, F39",
             "E30": "accepted iteration-0030 Expertium configuration: E27, F35, F38",
              "F42": "batch-wide column extraction for exact sample preparation",
              "F43": "four preparation workers after columnar samples",
               "F44": "batch-wide chunk-local segment transforms",
               "F45": "direct float64 sample matrix construction",
               "F46": "load-pipelined exact shift backward",
                "F48": "deterministic algorithms without uninitialized-memory fill",
                "F50": "skip zero hierarchy padding sources in exact row backward",
                "F49": "independent hierarchy split CUDA streams",
                "F53": "worker-precomputed stable scatter orders",
                "F34d": "exact bonus product glue",
                "F51a": "aligned batched split gather",
                "F52a": "fused padding index mapping",
                "F34a": "exact gate forward glue",
                "F34b": "exact gate forward/backward glue",
                "F34c": "exact channel ReLU-square glue",
                "F58": "recurrence next-step scalar prefetch",
                "F58a": "recurrence recomputation input pipeline",
                "F58b": "recurrence per-row gradient pipeline",
                "F58c": "forward recurrence scalar pipeline",
                "F58d": "forward recurrence row-value pipeline",
                "F58e": "forward recurrence scalar and row-value pipeline",
                "F58f": "recomputation row-value-only pipeline",
                "F55": "wave-aware recurrence backward layout",
                "F62": "flat optimizer buffers",
                "F54": "compact exact recurrence previous-state storage",
             "E31": "accepted iteration-0031 Expertium configuration: E30, F37",
             "E33": "accepted iteration-0033 Expertium configuration: E31, F42"}
    return ", ".join(f"{part} ({names.get(part, 'see track record')})" for part in track.split("-"))


def parity(left, right, left_path, right_path):
    keys = ("step", "user_id", "offset", "reviews", "seed", "loss", "gradient_norm", "curve_labels", "instant_labels", "lr")
    logs = [[json.loads(line) for line in (path / "updates.jsonl").read_text().splitlines()]
            for path in (left_path, right_path)]
    mismatches = [i + 1 for i, (a, b) in enumerate(zip(*logs)) if any(a[key] != b[key] for key in keys)
                  or a.get("parts", [[a["user_id"], a["offset"], a["reviews"]]])
                  != b.get("parts", [[b["user_id"], b["offset"], b["reviews"]]])]
    fields = ("initial_parameter_sha256", "trained_parameter_sha256", "validation_input_sha256", "updates",
              "review_presentations", "training_curve_labels", "training_instant_labels", "coverage", "evaluation")
    checks = {key: left[key] == right[key] for key in fields}
    return {"fields": checks, "equal_update_counts": len(logs[0]) == len(logs[1]),
            "update_mismatch_count": len(mismatches), "first_update_mismatch": next(iter(mismatches), None),
            "exact_recorded_parity": all(checks.values()) and len(logs[0]) == len(logs[1]) and not mismatches}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/retraining/gpu-retrain-r01/iteration-0003"))
    parser.add_argument("--brief", action="store_true", help="Print one decision line per model instead of JSON")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    protocol = json.loads((output / "protocol.json").read_text())
    iteration = protocol["iteration"]
    selection = json.loads((output / "selection.json").read_text())
    implementation_path = output / "implementation-commit.json"
    implementation = json.loads(implementation_path.read_text()) if implementation_path.exists() else None
    settings = protocol["acceptance"]
    target_percent = 100 * settings.get("target_throughput_gain", 0.20)
    amendment_path = output / "reference-reuse-amendment.json"
    amendment = json.loads(amendment_path.read_text()) if amendment_path.exists() else {}
    settings = {**settings, **amendment.get("acceptance_overrides", {})}
    target_percent = 100 * settings.get("target_throughput_gain", 0.20)
    _, _, definition = effective_loop(output, ROOT)
    cases, comparisons, references = {}, {}, {}
    sources = sorted(path for path in output.glob("*/*/result.json") if path.parent.parent.name in protocol["models"])
    for model in protocol["models"]:
        reused = protocol.get("loop_policy") == POLICY or model in amendment.get("reuse_models", [])
        references[model] = {}
        for kind, metadata in protocol["references"][model].items():
            reference, path = read_reference(metadata, ROOT)
            references[model][kind] = (reference, path.parent)
        control_track = protocol.get("control_tracks", {}).get(model, protocol.get("control_track", "control"))
        if not (output / model / control_track / "result.json").exists() and "previous" in protocol["references"][model]:
            reused = True
        if reused:
            control, control_path = references[model]["previous"]
        else:
            control_path = output / model / control_track
            control = json.loads((control_path / "result.json").read_text())
        track = selection[model]["track"]
        candidate, candidate_result, reused_candidate = resolve_case(output, model, track, ROOT)
        candidate_path = candidate_result.parent
        if reused_candidate and candidate_result not in sources:
            sources.append(candidate_result)
        if control["status"] != "complete" or candidate["status"] != "complete":
            raise ValueError("Both complete matched runs are required")
        if not reused and not reused_candidate and (control["implementation_sha256"] != candidate["implementation_sha256"]
                or control["protocol_sha256"] != candidate["protocol_sha256"]):
            raise ValueError("Matched runs have different implementations/protocols")
        if (reused or reused_candidate) and (candidate["implementation_sha256"] != protocol["implementation_sha256"]
                       or (not reused_candidate and candidate["protocol_sha256"] != hashlib.sha256((output / "protocol.json").read_bytes()).hexdigest())):
            raise ValueError("Candidate differs from the frozen protocol/source")
        options = protocol.get("case_options_by_model", {}).get(model, protocol.get("case_options"))
        for key in ("options",):
            if candidate[key] != options[track] or control[key] != options[control_track]:
                raise ValueError("Run options differ from frozen protocol")
        for current in (control, candidate):
            for reference, _ in references[model].values():
                config = dict(current["config"])
                expected = dict(reference["config"])
                declared, _ = declared_differences(current, reference)
                for key in ("PREFETCH_WORKERS", "PREFETCH_BATCHES", *declared):
                    config.pop(key, None)
                    expected.pop(key, None)
                if config != expected:
                    raise ValueError("Training behavior differs from historical reference")
                mismatch = same_exposure(current, reference, ("training_users", "evaluation_users", "updates", "review_presentations", "coverage", "initial_parameter_sha256", "validation_input_sha256"))
                if mismatch:
                    raise ValueError(f"Exposure/cohort/seed mismatch: {mismatch}")
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
        # Protocols frozen before the band existed keep their original zero tolerance.
        band = settings.get("speed_noise_band", {}).get(model, {"percent": 0.0, "available": False})
        block_result = None
        if settings.get("exact_promotion_rule") == "replicated_blocks":
            # Human approval 2026-10-13: interleaved reference/candidate replicates decide speed.
            from scripts.gpu_r01_policy import block_decision, replicate_blocks
            from scripts.r01_paired_timing import load_run, trimmed_gain
            params = settings["replicated_blocks"]
            pairs = replicate_blocks(output, model, control_track, track)
            effects = [trimmed_gain(load_run(r)["wall"], load_run(c)["wall"], params["mad_limit"])["gain_percent"] for r, c in pairs]
            block_result = block_decision(effects, params["calibrated_sd_percent"], tuple(params["looks"]),
                                          params["boundary_z"], params["max_blocks"])
            trained = lambda case: json.loads((case / "result.json").read_text())["trained_parameter_sha256"]
            block_result.update(pairs=[[r.relative_to(ROOT).as_posix(), c.relative_to(ROOT).as_posix()] for r, c in pairs],
                                reference_replicates_exact=all(trained(r) == control["trained_parameter_sha256"] for r, _ in pairs),
                                candidate_replicates_exact=all(trained(c) == candidate["trained_parameter_sha256"] for _, c in pairs))
        result.update(decide(gain, {kind: item["losses"] for kind, item in result["references"].items()},
                             relative_limit=settings["relative_loss_limit"], baseline_limit=settings["baseline_loss_limit"],
                             prior_streak=amendment.get("prior_no_20_percent_gain_streak", protocol["prior_no_20_percent_gain_streak"])[model],
                             resources_reduced=material_reduction(reductions) if "speed_noise_band" in settings else bool(reductions),
                             speed_tolerance_percent=band["percent"],
                             target_percent=target_percent,
                             exact_parity=result["parity"]["exact_recorded_parity"],
                             exact_target_percent=100 * settings["exact_target_throughput_gain"] if "exact_target_throughput_gain" in settings else None,
                             exact_promotion_rule=settings.get("exact_promotion_rule", "fixed_target"),
                             timing_noise_available=band.get("available", False),
                             loss_tolerance=settings.get("regressions_within_limit_promotable", False),
                             blocks=block_result))
        result["speed_noise_band"] = band
        result["resource_reductions"] = reductions
        result["reused_candidate"] = reused_candidate
        if reused_candidate:
            result["no_20_percent_gain_streak"] = latest_streak(ROOT, model, amendment.get("prior_no_20_percent_gain_streak", protocol["prior_no_20_percent_gain_streak"])[model], protocol.get("round_start_iteration"))
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
    stop_text = (f"complete the explicit {protocol['round_iteration_budget']}-iteration count despite consecutive misses"
                 if protocol.get("round_iteration_budget") else "stop each model after three consecutive misses of the applicable target")
    promotion_text = ("Bitwise-equal changes are promoted only when their gain exceeds the frozen measured timing-noise band; no fixed 2% or 5% minimum applies. Missing measured noise leaves promotion pending. Non-exact changes keep the timing-noise/quality rules and nominal throughput target."
                      if settings.get("exact_promotion_rule") == "timing_noise" else
                      f"Target ≥{target_percent:g}% reviews/s gain versus the previous accepted run" +
                      (f" for nonexact changes, or ≥{100 * settings['exact_target_throughput_gain']:g}% measured gain for exact recorded parity" if "exact_target_throughput_gain" in settings else ""))
    lines = [f"# R01 iteration {iteration}", "", f"Status: **complete preliminary screen**. Authorization: {protocol.get('round_authorization', protocol['authorization'])}.", "",
             f"Same fixed **{protocol['training_user_count']} training / {protocol['evaluation_users']} unseen held-out users**, **{protocol['training_reviews']:,} training / {protocol['validation_reviews']:,} evaluation reviews**, seed 2025, one epoch, BF16 compute/FP32 master. Chunks contain 128 reviews; prefetch is bounded at twice the worker count. Review exposure stays fixed; declared batch-size or cross-user packing changes may alter update membership, counts, batch seeds, dropout and the learning-rate trajectory. Exact worker/chunk settings and explanations are preserved in the protocol and raw results.", "",
             "[Frozen protocol](protocol.json), [selection from recorded profiles](selection.json), [loop definition](loop-definition.md), [raw comparison](comparison.json). References retain their original process provenance. Timing includes complete training loads, worker startup/shutdown, preparation, transfers, updates and logs. Setup/primer, save and evaluation are separate; process wall includes them.", "",
             f"Acceptance: measured relative regression ≤1% for all four binary losses against the previous accepted run; ≤2% versus the fixed original baseline is preferred. Confidence bounds are reported separately. Faster but less accurate variants remain separate opt-in fastmode. Accurate resource-saving candidates without a speed gain remain pending retuning. {promotion_text}" + f"; {stop_text}. This remains a preliminary screen; broader validation is pending.", "",
             "## Full-history measurements", "", "| Model / IDs | Training s | Reviews/s | Gain vs previous / original baseline | Process wall s | Weighted Curve / Instant | User-average Curve / Instant |",
             "|---|---:|---:|---|---:|---|---|"]
    effective_line = f"Effective rules: [loop definition]({link(ROOT / definition['path'], output)})."
    if "exact_target_throughput_gain" in settings:
        effective_line += f" Exact recorded parity uses the human-authorized **{100 * settings['exact_target_throughput_gain']:g}% measured throughput gate**, with timing spread reported separately; other changes retain the **{target_percent:g}% target** and original timing rule."
    if definition["amendment"]:
        effective_line += f" [Protocol amendment]({link(ROOT / definition['amendment'], output)}) records the transition; the original protocol remains preserved."
    lines[6:6] = [effective_line, ""]
    for model, runs in cases.items():
        for track, case in runs.items():
            e = case["evaluation"]
            gains = [delta(case["reviews_per_second"], references[model][kind][0]["reviews_per_second"]) for kind in ("previous", "original_baseline")]
            lines.append(f"| {model}: {describe(track)} | {case['training_seconds']:.3f} | {case['reviews_per_second']:,.0f} | " + " / ".join(f"{g:+.2f}%" for g in gains) +
                         f" | {case['process_wall_seconds']:.3f} | {e[METRICS[0]]:.9f} / {e[METRICS[2]]:.9f} | {e[METRICS[1]]:.9f} / {e[METRICS[3]]:.9f} |")
    lines += ["", "## Resource evidence and decisions", ""]
    for model, result in comparisons.items():
        lines += [f"{model}: {selection[model]['reason']}", ""]
        if result["reused_candidate"]:
            lines += ["Candidate measurements were reused from the linked original result; no new training attempt or stop-count increment.", ""]
        for track in cases[model]:
            case = cases[model][track]
            if case["options"].get("prestart_pool"):
                readiness = (f"**{case['prestarted_workers']} distinct workers** acknowledged initializer imports before training" if case.get("prestarted_worker_pids") else
                             f"**{case.get('prestarted_workers', '—')} distinct worker PIDs** returned warmup jobs; this does not establish that every configured worker finished imports")
                lines += [f"F28 starts worker imports during setup: {readiness}. Setup-to-ready elapsed **{case.get('pool_prestart_seconds', 0):.3f} s**, final ready wait **{case.get('pool_ready_wait_seconds', 0):.6f} s**. These costs are outside the training timer and inside the reported process wall; shutdown remains inside training.", ""]
            if case.get("cache_stats"):
                cache = case["cache_stats"]
                setup = case["cache_setup"]
                mode = case["options"].get("preprocessing_cache_mode", "cold_or_warm")
                lines += [f"F02 (preprocessing cache), mode `{mode}`: **{cache['hits']} hits / {cache['misses']} misses**, **{cache['not_persisted']} unpersisted batches**. Summed worker elapsed costs: build **{cache['build_seconds']:.3f} s**, read **{cache['read_seconds']:.3f} s**, write **{cache['write_seconds']:.3f} s**; reads/writes include lock waits and bookkeeping, and these concurrent costs are not wall-time savings. Content fingerprint setup **{setup['identity_seconds']:.3f} s** is outside training and inside process wall. Disk budget **{setup['max_bytes']/2**30:.2f} GiB** per model; existing bytes at start **{setup['disk_bytes_before']/2**30:.2f} GiB**. Cold construction remains included in the measured training time; warm reuse must be interpreted separately.", ""]
            if case.get('gpu_copy_milliseconds', 0):
                lines += [f"{describe(track)}: measured GPU input-stage time **{case['gpu_copy_milliseconds']/1000:.3f} s**, with **{case['gpu_copy_overlap_training_milliseconds']/1000:.3f} s** overlapping the preceding training batch. The input-stage event covers copying and GPU feature materialization; this is separate device timing, not a wall-time saving estimate.", ""]
            profile = output / "profiles" / f"{model}-{track}" / "result.json"
            if profile.exists():
                p = json.loads(profile.read_text())
                if "phases" not in p:
                    profile = next((path for path in sorted((output / "profiles").glob(f"{model}-{track}-*/result.json"), reverse=True)
                                    if "phases" in json.loads(path.read_text())), profile)
                    p = json.loads(profile.read_text())
                c = p["phases"]["training including user transitions"]["counters"]
                lines += [f"{describe(track)}: GPU activity **{c['nvidia_utilization.gpu']['mean']:.1f}%**, worker-pool CPU **{c['worker_pool_cpu_percent']['mean']:.1f}%**, peak reserved VRAM **{c['torch_cuda_peak_reserved_bytes']['max']/2**30:.2f} GiB**, minimum available system RAM **{c['ram_available_bytes']['min']/2**30:.2f} GiB**. [Resource profile]({link(profile.with_name('profile.md'), output)}). Full-history sampling-window sufficiency: `{p['workload']['full_history_profile_available']}`; GPU activity is not occupancy.", ""]
        if result["speed_noise_band"].get("available"):
            timing = (f"The recorded ±{result['speed_noise_band']['percent']:.2f}% repeat spread remains descriptive under the authorized exact-parity gate; the measured gain does not establish repeatability." if result.get("exact_speed_gate_applied") else
                      f"Gains within ±{result['speed_noise_band']['percent']:.2f}% ({'median spread over pairs of repeats of an identical configuration' if result['speed_noise_band'].get('estimator') == 'median_pairwise_spread' else 'largest recorded spread between repeats of an identical configuration'}) count as the same speed. Retuning requires a material resource reduction.")
            lines += [f"Speed change: **{result['speed_change']}** under the applicable rule. {timing}", ""]
        miss_text = (f"Consecutive iterations without exceeding the measured {result['throughput_target_percent']:g}% timing-noise band"
                     if result.get("timing_noise_gate_applied") and result["speed_noise_band"].get("available") else
                     "Measured timing noise is unavailable; promotion is pending" if result.get("timing_noise_gate_applied") else
                     f"Consecutive iterations without a {result['throughput_target_percent']:g}% gain")
        lines += [f"Decision: **{result['decision']}**. Exact recorded parity versus previous accepted run: **{result['parity']['exact_recorded_parity']}**; logged update mismatches: **{result['parity']['update_mismatch_count']}**. {miss_text}: **{result['no_20_percent_gain_streak']}**. Three-miss threshold reached: **{result['no_20_percent_gain_streak'] >= 3}**. Round stopping rule: {stop_text}.", "",
                  "| Binary loss | Candidate | Δ vs previous / original | Familywise upper vs previous / original |", "|---|---:|---|---|"]
        for key in METRICS:
            metrics = [result["references"][kind]["losses"]["metrics"][key] for kind in ("previous", "original_baseline")]
            lines.append(f"| {key} | {metrics[0]['candidate']:.9f} | " + " / ".join(f"{m['relative_delta_percent']:+.2f}%" for m in metrics) + " | " + " / ".join(f"{100*m['relative_familywise_upper']:+.2f}%" for m in metrics) + " |")
        lines.append("")
        if result.get("replicated_blocks"):
            b = result["replicated_blocks"]
            lines += [f"Replicate blocks: **{b['n']}** reference/candidate pairs; trimmed per-update gain "
                      + (f"mean **{b['mean_percent']:+.2f}%**, lower bound **{b['lower_bound_percent']:+.2f}%** at z = {b['boundary_z']:g} "
                         f"(block SD used {b['sd_used_percent']:.2f}%); " if "mean_percent" in b else "")
                      + f"block decision **{b['decision']}**. Replicates exact: reference {b.get('reference_replicates_exact')}, "
                        f"candidate {b.get('candidate_replicates_exact')}.", ""]
        if result.get("stack_candidate"):
            lines += [f"Stack candidate: **positive {result['references']['previous']['throughput_gain_percent']:+.2f}% below the promotion rule**; "
                      "keep the option for later stacking with other small gains (2026-10-12 human request).", ""]
        if result["resource_retuning_pending"]:
            lines += [result["retuning_next_step"], ""]
    checks = output / "prefetch-checks-v1/result.json"
    if not checks.exists():
        checks = ROOT / "artifacts/retraining/gpu-retrain-r01/iteration-0004/prefetch-checks-v1/result.json"
    lines += ["## Evidence and provenance", "", "Each process captures its own dirty source snapshot and full commit at start; the shared training-implementation fingerprint is checked at every user boundary. Historical references retain their original revisions. " + f"[Recorded ordered-prefetch contracts]({link(checks, output)}); functional smoke results are separate from full-history timing.", ""]
    for path in [*sources, *sorted(output.glob("functional-smoke-v1/*/*/result.json"))]:
        case = json.loads(path.read_text())
        p = case["provenance"]
        checkpoint = path.with_name('master.pth')
        saved = f"[checkpoint]({link(checkpoint, output)})" if checkpoint.exists() else "no checkpoint saved"
        log = path.with_name('updates.jsonl')
        updates = f"[updates]({link(log, output)})" if log.exists() else "no training updates"
        lines.append(f"- [{case['model']}: {describe(case['track'])}{' smoke' if case['smoke_only'] else ''}]({link(path,output)}): {case['status']}; training start UTC {case.get('training_started_utc','—')}; commit `{p['git_commit']}`, dirty `{p['git_dirty']}`; [source snapshot]({link(ROOT/p['source_archive'], output)}), {updates}, {saved}.")
    lines += ["", f"Implementation commit: `{implementation['git_commit']}` (after experiments)." if implementation else "Implementation commit pending the completed speed/quality decision.", "", report["limitations"] + ". This iteration is complete. Recorded scope: " + protocol["budget"] + "; no broader validation is authorized.", ""]
    update = output / "authorization-update-001.json"
    if update.exists():
        lines += ["[Updated continuation authorization](authorization-update-001.json), [revised loop definition](loop-definition-update-001.md). The revised definition separates run-log columns, rounds percentages to two decimals, commits opt-in fastmode too, and asks whether freed resources can be used to increase throughput.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    index = ROOT / f"artifacts/retraining/gpu-retrain-r01/iterations/iteration-{iteration}.md"
    index.write_text(f"# R01 iteration {iteration}\n\nComplete preliminary screen. [Full report](../iteration-{iteration}/summary.md), [protocol](../iteration-{iteration}/protocol.json), [comparison](../iteration-{iteration}/comparison.json).\n", encoding="utf-8")
    update_run_log(output)
    if args.brief:
        for model, c in comparisons.items():
            case = json.loads((output / model / c["track"] / "result.json").read_text())
            previous, baseline = c["references"]["previous"], c["references"]["original_baseline"]
            print(f"{iteration} {model} {c['track']}: {case.get('reviews_per_second', float('nan')):,.0f} reviews/s, "
                  f"{previous['throughput_gain_percent']:+.2f}% vs previous, {baseline['throughput_gain_percent']:+.2f}% vs baseline, "
                  f"process wall {previous['process_wall_delta_percent']:+.2f}%; exact {c['parity']['exact_recorded_parity']}; "
                  f"{c['decision']}; misses {c['no_20_percent_gain_streak']}")
        return
    print(json.dumps({model: {"track": c["track"], "gain_percent": c["references"]["previous"]["throughput_gain_percent"],
                              "exact": c["parity"]["exact_recorded_parity"], "decision": c["decision"]} for model,c in comparisons.items()}))


if __name__ == "__main__":
    main()
