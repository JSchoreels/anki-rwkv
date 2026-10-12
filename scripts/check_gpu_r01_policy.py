"""CPU-only checks of R01 workflow decisions; no training or quality validation."""

from argparse import ArgumentParser
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance
import shutil
from scripts.gpu_r01_policy import (LOSS_METRICS, POLICY, accepted_reference, decide, material_reduction,
                                    read_reference, require_active, resource_reductions, result_metadata,
                                    check_inputs, reusable_check, speed_noise_band, pairwise_spreads, same_exposure, declared_differences,
                                    round_budget_fields, round_loss_limit, block_decision, replicate_blocks)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat-reason", help="Concrete reason to repeat unchanged workflow checks")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Use a new workspace artifact directory")
    inputs = check_inputs(ROOT, ("scripts/check_gpu_r01_policy.py", "scripts/gpu_r01_policy.py",
                                "scripts/run_gpu_r01.py", "scripts/run_isolated_gpu_r01.py", "scripts/isolate_gpu_r01.py",
                                "scripts/select_gpu_r01_next.py", "scripts/report_gpu_r01_next.py", "scripts/gpu_r01_log.py",
                                "scripts/profile_gpu_resources.py", "scripts/gpu_r01_training.py",
                                "scripts/gpu_r01_packing.py", "scripts/gpu_r01_samples.py"))
    previous = reusable_check(ROOT, "R01 workflow", inputs)
    if previous and not (args.repeat_reason or "").strip():
        print(json.dumps({"action": "reuse checks", "result": previous.relative_to(ROOT).as_posix()}))
        return
    if output.exists():
        parser.error("Preserve the previous checks; use a new workspace artifact directory")
    output.mkdir(parents=True)
    result = {"synthetic_only": True, "provenance": process_provenance(), "checks": [], "status": "running",
              "check_suite": "R01 workflow", "check_inputs": inputs, "repeat_reason": args.repeat_reason}

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        result["checks"].append(name)

    def reject(name, function):
        try:
            function()
        except (ValueError, RuntimeError):
            check(name, True)
        else:
            raise AssertionError(name)

    def loss(delta, upper=0.5):
        return {"metrics": {key: {"relative_delta": delta, "relative_familywise_upper": upper} for key in LOSS_METRICS}}

    def decision(delta=0, gain=20, **kwargs):
        return decide(gain, {"previous": loss(delta), "original_baseline": loss(delta)},
                      relative_limit=0.01, baseline_limit=0.02, prior_streak=2, **kwargs)

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    try:
        check("confidence bound does not replace measured loss ceiling", decision(0.009)["passes_preliminary_quality"])
        check("one percent measured boundary is allowed", decision(0.01)["passes_preliminary_quality"])
        check("regression above one percent fails", not decision(0.0101)["passes_preliminary_quality"])
        check("faster unchanged scores are accepted", decision()["accepted_main"])
        check("faster improved scores are accepted", decision(-0.01)["accepted_main"])
        check("faster slightly worse scores stay fastmode-only", decision(0.001)["fastmode"] and not decision(0.001)["accepted_main"])
        check("faster excessive regression stays fastmode-only", decision(0.03)["fastmode"] and not decision(0.03)["accepted_main"])
        check("same speed and less resource use remains pending retuning", decision(gain=0, resources_reduced=True)["resource_retuning_pending"])
        check("resource-saving improvement without speed gain is not discarded", decision(-0.01, gain=-0.1, resources_reduced=True, speed_tolerance_percent=2)["resource_retuning_pending"])
        check("no resource evidence does not fabricate a retuning benefit", not decision(gain=0)["resource_retuning_pending"])
        within = decision(gain=3.11, speed_tolerance_percent=5.68)
        check("gain inside the recorded repeat spread is the same speed, not a promotion",
              within["speed_change"].startswith("same") and not within["accepted_main"] and not within["fastmode"])
        check("same speed with a material resource saving is sent to retuning",
              decision(gain=0.07, resources_reduced=True, speed_tolerance_percent=5.68)["resource_retuning_pending"])
        check("a slowdown beyond the repeat spread cannot be sent to retuning",
              not decision(gain=-6.48, resources_reduced=True, speed_tolerance_percent=1.97)["resource_retuning_pending"])
        check("a gain beyond the repeat spread is accepted", decision(gain=26.63, speed_tolerance_percent=5.68)["accepted_main"])
        check("the 20% target still uses the measured gain", decision(gain=19.99, speed_tolerance_percent=5.68)["no_20_percent_gain_streak"] == 3)
        reject("negative speed tolerance is rejected", lambda: decision(speed_tolerance_percent=-1))
        check("a 17% gain misses a 20% target", decision(gain=17.4)["no_20_percent_gain_streak"] == 3)
        check("the same gain meets a 5% target and resets the miss count",
              decision(gain=17.4, target_percent=5.0)["no_20_percent_gain_streak"] == 0 and decision(gain=17.4, target_percent=5.0)["throughput_target_percent"] == 5.0)
        check("a gain below 5% still counts as a miss under the 5% target", decision(gain=4.9, target_percent=5.0)["no_20_percent_gain_streak"] == 3)
        exact = decision(gain=3.69, target_percent=5, speed_tolerance_percent=5.68, exact_parity=True, exact_target_percent=2)
        check("authorized exact-parity gate accepts measured gain inside timing spread", exact["accepted_main"] and exact["no_20_percent_gain_streak"] == 0 and exact["throughput_target_percent"] == 2)
        check("exact gate includes its two-percent boundary", decision(gain=2, exact_parity=True, exact_target_percent=2)["accepted_main"])
        check("exact gate rejects a smaller gain", not decision(gain=1.99, exact_parity=True, exact_target_percent=2)["accepted_main"])
        check("score-only or nonexact changes keep the original gate", not decision(gain=3.69, target_percent=5, speed_tolerance_percent=5.68, exact_target_percent=2)["accepted_main"])
        reject("changed measured scores cannot claim exact parity", lambda: decision(0.001, gain=3.69, exact_parity=True, exact_target_percent=2))
        noise_options = {"exact_parity": True, "exact_promotion_rule": "timing_noise", "speed_tolerance_percent": 1.0,
                         "target_percent": 5, "exact_target_percent": 2}
        above_noise = decision(gain=1.5, **noise_options)
        check("new exact rule promotes below two/five percent above measured noise", above_noise["accepted_main"] and above_noise["no_20_percent_gain_streak"] == 0)
        check("new exact rule does not apply the historical fixed gate", not above_noise["exact_speed_gate_applied"] and above_noise["timing_noise_gate_applied"])
        check("new exact rule requires strictly exceeding noise", not decision(gain=1.0, **noise_options)["accepted_main"])
        within_noise = decision(gain=3.69, exact_parity=True, exact_promotion_rule="timing_noise", speed_tolerance_percent=5.68)
        check("new exact rule rejects gain inside noise despite exceeding two percent", not within_noise["accepted_main"] and within_noise["no_20_percent_gain_streak"] == 3)
        missing = decision(gain=10, exact_parity=True, exact_promotion_rule="timing_noise", timing_noise_available=False)
        check("unknown noise is not silently treated as a measured zero band", not missing["accepted_main"] and missing["decision"] == "pending measured timing-noise evidence")
        reject("new exact rule rejects changed losses", lambda: decision(0.001, gain=10, exact_parity=True, exact_promotion_rule="timing_noise"))
        reject("invalid exact promotion rule is rejected", lambda: decision(exact_promotion_rule="unknown"))
        # 2026-10-12 human decisions: a frozen round loss ceiling (0.1% to start) and recorded stack candidates.
        def tolerant(delta=0.0, start=None, gain=20, **kwargs):
            values = {"previous": loss(delta), "original_baseline": loss(delta), "round_start": loss(delta if start is None else start)}
            return decide(gain, values, relative_limit=0.001, baseline_limit=0.02, prior_streak=0,
                          loss_tolerance=True, speed_tolerance_percent=1.0, **kwargs)
        check("round ceiling promotes a faster regression within it", tolerant(0.0009)["accepted_main"] and not tolerant(0.0009)["fastmode"])
        check("round ceiling includes its boundary", tolerant(0.001)["accepted_main"])
        check("regression beyond the round ceiling stays fastmode", tolerant(0.0011)["fastmode"] and not tolerant(0.0011)["accepted_main"])
        check("cumulative drift since round start is bounded by the same ceiling",
              tolerant(0.0005, start=0.0015)["fastmode"] and not tolerant(0.0005, start=0.0015)["accepted_main"])
        reject("tolerated ceiling requires the round-start reference",
               lambda: decide(20, {"previous": loss(0), "original_baseline": loss(0)}, relative_limit=0.001,
                              baseline_limit=0.02, prior_streak=0, loss_tolerance=True))
        reject("tolerance flag must be boolean", lambda: decision(loss_tolerance=1))
        check("without a round ceiling any regression stays fastmode", decision(0.0005)["fastmode"])
        stack = decision(gain=3.0, exact_parity=True, exact_promotion_rule="timing_noise", speed_tolerance_percent=7.045)
        check("exact gain below noise is a stack candidate", stack["stack_candidate"] and not stack["accepted_main"])
        check("promoted gain is not also a stack candidate", not decision(gain=8.0, exact_parity=True, exact_promotion_rule="timing_noise", speed_tolerance_percent=7.045)["stack_candidate"])
        check("slower exact result is not a stack candidate", not decision(gain=-0.3, exact_parity=True, exact_promotion_rule="timing_noise", speed_tolerance_percent=7.045)["stack_candidate"])
        check("unknown noise is not a stack candidate", not decision(gain=3.0, exact_parity=True, exact_promotion_rule="timing_noise", timing_noise_available=False)["stack_candidate"])
        check("non-exact regression without a round ceiling is not a stack candidate", not decision(0.0005, gain=3.0, speed_tolerance_percent=7.045)["stack_candidate"])
        check("non-exact regression within a round ceiling can be a stack candidate", tolerant(0.0005, gain=0.5)["stack_candidate"])
        check("round ceiling is frozen at round start", round_loss_limit(new_round=True, requested=0.001) == 0.001)
        check("continuation inherits the frozen ceiling", round_loss_limit({"round_relative_loss_limit": 0.001}) == 0.001)
        check("a new round without a ceiling does not inherit one", round_loss_limit({"round_relative_loss_limit": 0.001}, new_round=True) is None)
        reject("ceiling cannot change mid-round", lambda: round_loss_limit({"round_relative_loss_limit": 0.001}, requested=0.002))
        reject("ceiling must be a fraction below one", lambda: round_loss_limit(new_round=True, requested=1.5))
        reject("ceiling must be finite", lambda: round_loss_limit(new_round=True, requested=float("nan")))
        # Human approval 2026-10-13: replicate blocks with a Pocock boundary at 2/4/6 blocks.
        check("fewer than two blocks asks for more", block_decision([1.0])["decision"] == "run more blocks")
        check("two clear blocks promote", block_decision([2.4, 2.1])["decision"] == "promote")
        check("a small positive mean at two blocks extends to four", block_decision([0.8, 0.6])["decision"] == "extend" and block_decision([0.8, 0.6])["next_look"] == 4)
        check("a non-positive mean is rejected at once", block_decision([0.3, -0.5])["decision"] == "reject")
        check("a positive mean still unproven at six blocks becomes a stack candidate", block_decision([0.4, 0.2, 0.5, 0.1, 0.3, 0.2])["decision"] == "stack candidate")
        check("the observed block SD widens the bound when larger than the calibration", block_decision([6.0, -1.0])["decision"] != "promote")
        check("the boundary is the frozen Pocock value", abs(block_decision([2.0, 2.0])["boundary_z"] - 1.9927) < 1e-12)
        reject("more blocks than the cap are refused", lambda: block_decision([1.0] * 9))
        def blocked(decision_blocks, delta=0.0, exact=True):
            return decide(0.0, {"previous": loss(delta), "original_baseline": loss(delta)}, relative_limit=0.01, baseline_limit=0.02,
                          prior_streak=0, exact_parity=exact, exact_promotion_rule="replicated_blocks", blocks=decision_blocks)
        promoted = blocked(block_decision([2.4, 2.1]))
        check("replicated rule promotes from blocks, not from the total gain", promoted["accepted_main"] and promoted["no_20_percent_gain_streak"] == 0)
        pending = blocked(block_decision([0.8, 0.6]))
        check("an extending block decision leaves the iteration pending", pending["decision_pending"] and not pending["accepted_main"] and not pending["stack_candidate"])
        stacked = blocked(block_decision([0.4, 0.2, 0.5, 0.1, 0.3, 0.2]))
        check("a final positive unproven block result is a stack candidate", stacked["stack_candidate"] and not stacked["accepted_main"])
        check("non-exact promotion from blocks still needs no regression without a ceiling", blocked(block_decision([2.4, 2.1]), delta=0.0005, exact=False)["fastmode"])
        reject("replicated rule without a block decision is refused", lambda: decide(1.0, {"previous": loss(0), "original_baseline": loss(0)}, relative_limit=0.01,
                                                                                      baseline_limit=0.02, prior_streak=0, exact_promotion_rule="replicated_blocks"))
        pairing = output / "pairing-fixture"
        for name, track in (("r1", "R"), ("r2", "C"), ("r3", "C"), ("r4", "R"), ("r5", "R"), ("r6", "C")):
            write(pairing / "replicates" / name / "expertium" / track / "result.json", {"track": track})
        pairs = [(a.parts[-3], b.parts[-3]) for a, b in replicate_blocks(pairing, "expertium", "R", "C")]
        check("blocks pair each candidate with the reference just before it", pairs == [("r1", "r2"), ("r5", "r6")])
        shutil.rmtree(pairing)
        # Human decision 2026-10-12 (option C): the band is the median spread over all repeat pairs.
        import statistics
        recorded = [[55810, 58982], [73577, 74619], [170718, 182745], [197499, 161687]]
        check("median pairwise band ignores one slow-state repeat group", abs(statistics.median(pairwise_spreads(recorded)) - 6.364) < 0.01)
        check("one pair gives its own spread", abs(pairwise_spreads([[120, 126]])[0] - 5) < 1e-9)
        alex = [[28521, 28899, 29084, 26985]]
        check("a four-run group contributes all six pairs", len(pairwise_spreads(alex)) == 6 and abs(statistics.median(pairwise_spreads(alex)) - 3.833) < 0.01)
        check("adding a closer repeat can lower the band", statistics.median(pairwise_spreads(recorded + [[100, 100.5], [100, 101]])) < statistics.median(pairwise_spreads(recorded)))
        accepted_case = {"options": {}, "updates": 276, "review_presentations": 960, "training_users": 80,
                         "coverage": {"1": {"reviews": 900, "updates": 2}, "2": {"reviews": 60, "updates": 1}}}
        larger = {**accepted_case, "options": {"batch_chunks": 512}, "updates": 138,
                  "coverage": {"1": {"reviews": 900, "updates": 1}, "2": {"reviews": 60, "updates": 1}}}
        exposure_keys = ("training_users", "updates", "review_presentations", "coverage")
        check("declared larger batches may change only update counts", same_exposure(larger, accepted_case, exposure_keys) is None)
        packed = {**larger, "options": {"pack_users": True}}
        check("cross-user packing independently declares changed updates", same_exposure(packed, accepted_case, exposure_keys) is None)
        from scripts.gpu_r01_policy import planned_updates
        check("packed update plan keeps partial user chunks separate", planned_updates([{"available_reviews": 129}, {"available_reviews": 127}], 2, pack_users=True) == 2)
        check("declared larger batches still require identical per-user reviews",
              same_exposure({**larger, "coverage": {"1": {"reviews": 899, "updates": 1}, "2": {"reviews": 61, "updates": 1}}},
                            accepted_case, exposure_keys) == "coverage")
        check("undeclared update-count changes are rejected", same_exposure({**accepted_case, "updates": 138}, accepted_case, exposure_keys) == "updates")
        check("declared options name only their configuration keys",
              declared_differences(larger, {"options": {"gpu_memory_fraction": 0.9}})[0] == {"BATCH_CHUNKS", "GPU_MEMORY_FRACTION"})
        check("twenty percent resets the miss streak", decision()["no_20_percent_gain_streak"] == 0)
        check("below twenty percent increments the miss streak", decision(gain=19.99)["no_20_percent_gain_streak"] == 3)
        require_active("baseline", 2)
        reject("three misses prevent another model run", lambda: require_active("baseline", 3))
        fixed = {"iteration": "0046", **round_budget_fields("0046", authorization="Human requested nine iterations", budget=9)}
        for index in range(1, 10):
            fixed["iteration"] = f"{45 + index:04d}"
            require_active("expertium", index + 3, fixed)
            check(f"explicit count permits iteration {index} despite misses", fixed["round_iteration_index"] == index)
            fields = round_budget_fields(f"{46 + index:04d}", fixed)
            fixed = {"iteration": f"{46 + index:04d}", **fields}
        reject("explicit count blocks a tenth iteration", lambda: require_active("expertium", 0, fixed))
        reject("count needs fresh authorization", lambda: round_budget_fields("0046", budget=9))
        reject("zero count is invalid", lambda: round_budget_fields("0046", authorization="Human", budget=0))
        reject("count without boundary cannot bypass stop", lambda: require_active("expertium", 3, {"round_iteration_budget": 9, "round_iteration_index": 1}))
        check("new uncounted authorization does not inherit count", round_budget_fields("0056", fixed, authorization="Human new round") == {})
        separate = {"previous": loss(0), "original_baseline": loss(0.025)}
        check("two percent original-baseline target remains preferred, not mandatory",
              decide(20, separate, relative_limit=0.01, baseline_limit=0.02, prior_streak=0)["accepted_main"])
        reject("nonfinite throughput is not accepted", lambda: decision(gain=float("nan")))

        from scripts import run_gpu_r01 as runner, select_gpu_r01_next as selector
        from scripts import report_gpu_r01_next as reporter, gpu_r01_log as ledger
        root = output / "fixtures"
        base = root / "artifacts/retraining/gpu-retrain-r01"
        old = base / "iteration-0002"
        new = base / "iteration-0003"
        (root / "docs").mkdir(parents=True)
        (root / "docs/GPU_RETRAIN_LOOP.md").write_bytes((ROOT / "docs/GPU_RETRAIN_LOOP.md").read_bytes())
        cohort = {"stages": {"100": {"users": 100, "training_users": list(range(1, 81)), "heldout_users": list(range(81, 101)),
                  "rows": [{"user_id": i, "available_reviews": 12} for i in range(1, 81)], "evaluation_rows": [],
                  "training_reviews": 960, "validation_reviews": 40}}, "independent_final_test_users": [],
                  "dataset_path": "synthetic", "dataset_revision": "synthetic", "sampling_seed": 20261011}
        cohort_path = base / "cohorts-user-holdout-v1.json"
        write(cohort_path, cohort)
        write(old / "protocol.json", {"iteration": "0002", "models": ["baseline", "expertium"],
                                      "cohort_sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest()})
        comparisons = {m: {"track": "F01", "decision": "promising preliminary speed screen", "no_20_percent_gain_streak_after_this_iteration": 0}
                       for m in ("baseline", "expertium")}
        write(old / "comparison.json", {"comparisons": comparisons})
        plan = {"control_tracks": {m: "F01" for m in comparisons}, "candidate_tracks": {m: "F01-F08" for m in comparisons},
                "case_options_by_model": {m: {"F01": {"prefetch_users": True}, "F01-F08": {"prefetch_users": True, "fused_shift": True}} for m in comparisons},
                "case_explanations": {}}
        plan_path = base / "plan.json"
        write(plan_path, plan)
        for model in comparisons:
            for track, speed in (("control", 100), ("F01", 120)):
                case = {"model": model, "track": track, "status": "complete", "smoke_only": False,
                        "training_started_utc": "2026-01-01T00:00:00+00:00" if track == "control" else "2026-01-01T01:00:00+00:00",
                        "options": plan["case_options_by_model"][model]["F01"] if track == "F01" else {},
                        "config": {"MODEL": model, "SEED": 2025, "PREFETCH_WORKERS": 4, "PREFETCH_BATCHES": 8},
                        "training_users": 80, "evaluation_users": 20, "updates": 80, "review_presentations": 960,
                        "initial_parameter_sha256": "initial", "trained_parameter_sha256": "final", "validation_input_sha256": "inputs",
                        "training_curve_labels": 640, "training_instant_labels": 640, "coverage": {},
                        "training_seconds": 960 / speed, "reviews_per_second": speed, "process_wall_seconds": 20,
                        "per_user": [{"user_id": i, "curve_labels": 2, "instant_labels": 2, "curve_log_loss": 0.5, "instant_log_loss": 0.5} for i in range(81, 101)],
                        "evaluation": {**{key: 0.5 for key in LOSS_METRICS}, "curve_labels": 40, "instant_labels": 40},
                        "implementation_sha256": "old-source", "protocol_sha256": "old-protocol",
                        "provenance": {"git_commit": "synthetic-only", "git_dirty": False, "source_archive": "artifacts/synthetic-only-source.zip"}}
                path = old / model / track / "result.json"
                write(path, case)
                updates = [{"step": i, "user_id": i, "offset": 0, "reviews": 12, "seed": 2025, "loss": 0.5,
                            "gradient_norm": 0.1, "curve_labels": 8, "instant_labels": 8, "lr": 0.001, "queue_wait_seconds": 0} for i in range(1, 81)]
                path.with_name("updates.jsonl").write_text("".join(json.dumps(row) + "\n" for row in updates))
            profile = {"workload": {"full_history_profile_available": False, "logged_training_reviews": 960, "logged_updates": 80,
                                   "full_history_complete": True, "complete_training_observation": True},
                       "phases": {"training including user transitions": {"counters": {
                           "nvidia_utilization.gpu": {"mean": 70}, "worker_pool_cpu_percent": {"mean": 90},
                           "ram_available_bytes": {"min": 8 * 2**30}, "observed_worker_count": {"mean": 4},
                           "torch_cuda_peak_reserved_bytes": {"max": 4 * 2**30}}}}}
            profile["inputs"] = [result_metadata(old / model / "F01/result.json", root)]
            write(old / "profiles" / (model + "-F01") / "result.json", profile)

        patches = [patch.object(runner, "ROOT", root), patch.object(runner, "COHORT", cohort_path),
                   patch.object(runner, "process_provenance", lambda: {"synthetic_only": True}),
                   patch.object(runner, "R01SourceWatch", lambda _: SimpleNamespace(implementation_sha256="new-source"))]
        from contextlib import ExitStack
        with ExitStack() as stack:
            for item in patches: stack.enter_context(item)
            with redirect_stdout(io.StringIO()): runner.prepare(new, "0003", old, plan_path)
            protocol = json.loads((new / "protocol.json").read_text())
            check("continuation freezes recorded references and no repeated-control budget", protocol["loop_policy"] == POLICY and "reuse recorded" in protocol["budget"])
            frozen_band = protocol["acceptance"]["speed_noise_band"]["baseline"]
            check("protocol predeclares an unavailable speed band without repeats", frozen_band["percent"] == 0.0 and frozen_band["available"] is False)
            repeat = base / "iteration-repeat"
            write(repeat / "protocol.json", json.loads((old / "protocol.json").read_text()))
            repeated = json.loads((old / "baseline/F01/result.json").read_text())
            repeated.update(reviews_per_second=126, training_seconds=960 / 126,
                            protocol_sha256=hashlib.sha256((repeat / "protocol.json").read_bytes()).hexdigest())
            write(repeat / "baseline/F01/result.json", repeated)
            band = speed_noise_band(root, "baseline")
            check("identical repeats define the speed band from their measured spread", band["available"] and abs(band["percent"] - 5) < 1e-9 and len(band["sources"]) == 2 and band["estimator"] == "median_pairwise_spread")
            repeated["trained_parameter_sha256"] = "different-weights"
            write(repeat / "baseline/F01/result.json", repeated)
            check("runs with different trained weights are not treated as repeats", not speed_noise_band(root, "baseline")["available"])
            shutil.rmtree(repeat)
            reject("accepted configuration cannot be rerun in a continuation", lambda: runner.run_case(SimpleNamespace(model="baseline", track="F01", smoke=False), new))
            for model in comparisons: comparisons[model]["no_20_percent_gain_streak_after_this_iteration"] = 3
            write(old / "comparison.json", {"comparisons": comparisons})
            stopped = base / "iteration-stopped"
            reject("both stopped models prevent preparing another iteration", lambda: runner.prepare(stopped, "0004", old, plan_path))
            check("stopped preparation writes no new protocol", not stopped.exists())
            resumed = base / "iteration-0005"
            with redirect_stdout(io.StringIO()):
                runner.prepare(resumed, "0005", old, plan_path, round_authorization="Synthetic human request for a new round")
            resumed_protocol = json.loads((resumed / "protocol.json").read_text())
            check("a quoted human request starts a new round and keeps the earlier stop counts",
                  resumed_protocol["round_start_iteration"] == "0005"
                  and resumed_protocol["round_authorization"] == "Synthetic human request for a new round"
                  and resumed_protocol["prior_no_20_percent_gain_streak"] == {"baseline": 0, "expertium": 0}
                  and resumed_protocol["historical_no_gain_streak"] == {"baseline": 3, "expertium": 3})
            shutil.rmtree(resumed)
            counted = base / "iteration-counted"
            with redirect_stdout(io.StringIO()):
                runner.prepare(counted, "0005", old, plan_path, round_authorization="Human requested nine iterations", round_iterations=9)
            counted_protocol = json.loads((counted / "protocol.json").read_text())
            blocks_frozen = counted_protocol["acceptance"].get("replicated_blocks", {})
            check("new protocols freeze the replicate-block rule (human approval 2026-10-13)",
                  counted_protocol["acceptance"]["exact_promotion_rule"] == "replicated_blocks" and "exact_target_throughput_gain" not in counted_protocol["acceptance"]
                  and blocks_frozen.get("looks") == [2, 4, 6] and blocks_frozen.get("boundary_z") == 1.9927 and blocks_frozen.get("calibrated_sd_percent") == 1.01)
            check("preparation freezes explicit human count", counted_protocol["round_iteration_budget"] == 9 and counted_protocol["round_iteration_index"] == 1)
            ceiling = base / "iteration-ceiling"
            with redirect_stdout(io.StringIO()):
                runner.prepare(ceiling, "0005", old, plan_path, round_authorization="Human: start at 0.1%", round_iterations=10, loss_ceiling=0.001)
            ceiling_protocol = json.loads((ceiling / "protocol.json").read_text())
            check("preparation freezes the round loss ceiling and its round-start reference",
                  ceiling_protocol["round_relative_loss_limit"] == 0.001 and ceiling_protocol["acceptance"]["relative_loss_limit"] == 0.001
                  and ceiling_protocol["acceptance"]["regressions_within_limit_promotable"] is True
                  and all(refs["round_start"] == refs["previous"] for refs in ceiling_protocol["references"].values()))
            shutil.rmtree(ceiling)
            counted_protocol["round_iteration_index"] = 3
            write(counted / "protocol.json", counted_protocol)
            write(counted / "comparison.json", {"comparisons": comparisons})
            continued = base / "iteration-counted-next"
            with redirect_stdout(io.StringIO()):
                runner.prepare(continued, "0006", counted, plan_path)
            continued_protocol = json.loads((continued / "protocol.json").read_text())
            check("preparation carries count past three misses", continued_protocol["round_iteration_index"] == 4 and continued_protocol["round_iteration_budget"] == 9)
            counted_protocol["round_iteration_index"] = 9
            write(counted / "protocol.json", counted_protocol)
            exhausted = base / "iteration-count-exhausted"
            reject("preparation refuses exhausted explicit count", lambda: runner.prepare(exhausted, "0006", counted, plan_path))
            check("exhausted count writes no protocol", not exhausted.exists())
            shutil.rmtree(counted)
            shutil.rmtree(continued)
            comparisons["expertium"]["no_20_percent_gain_streak_after_this_iteration"] = 0
            write(old / "comparison.json", {"comparisons": comparisons})
            remaining = base / "iteration-one-active"
            with redirect_stdout(io.StringIO()): runner.prepare(remaining, "one-active", old, plan_path)
            check("stopped model is excluded while the other may continue", json.loads((remaining / "protocol.json").read_text())["models"] == ["expertium"])
            for model in comparisons: comparisons[model]["no_20_percent_gain_streak_after_this_iteration"] = 0
            write(old / "comparison.json", {"comparisons": comparisons})

        # The remaining report fixtures exercise historical fixed-target decisions.
        # Fresh protocols and measured-noise decision boundaries are checked above.
        protocol["acceptance"]["exact_promotion_rule"] = "fixed_target"
        protocol["acceptance"]["exact_target_throughput_gain"] = 0.02
        write(new / "protocol.json", protocol)
        for model in comparisons:
            with patch.object(selector, "ROOT", root), patch.object(selector, "process_provenance", lambda: {"synthetic_only": True}), \
                 patch.object(sys, "argv", ["select", "--output", str(new), "--model", model]), redirect_stdout(io.StringIO()):
                selector.main()
            case = json.loads((old / model / "F01/result.json").read_text())
            case.update(track="F01-F08", options=plan["case_options_by_model"][model]["F01-F08"],
                        training_started_utc="2026-01-02T00:00:00+00:00", implementation_sha256="new-source",
                        protocol_sha256=hashlib.sha256((new / "protocol.json").read_bytes()).hexdigest(),
                        reviews_per_second=144, training_seconds=960 / 144)
            write(new / model / "F01-F08/result.json", case)
            (new / model / "F01-F08/updates.jsonl").write_bytes((old / model / "F01/updates.jsonl").read_bytes())
        selection = json.loads((new / "selection.json").read_text())
        check("selection reuses complete profiles without a fresh control or arbitrary duration floor", all(s["profile_reused"] for s in selection.values()))
        check("non-feeding candidate records its ceiling without needing a justification",
              all(not s["target_reachability"]["feeding_only_candidate"] and s["target_reachability"]["justification"] is None for s in selection.values()))
        ceiling_output = base / "iteration-ceiling"
        ceiling_protocol = deepcopy(protocol)
        ceiling_protocol.update(iteration="ceiling", models=["expertium"], candidate_tracks={"expertium": "F01-F19"},
                                case_options_by_model={"expertium": {"F01": {"prefetch_users": True},
                                                                     "F01-F19": {"prefetch_users": True, "batched_id_encodings": True}}})
        write(ceiling_output / "protocol.json", ceiling_protocol)

        def select_ceiling(*extra):
            with patch.object(selector, "ROOT", root), patch.object(selector, "process_provenance", lambda: {"synthetic_only": True}), \
                 patch.object(sys, "argv", ["select", "--output", str(ceiling_output), "--model", "expertium", *extra]), redirect_stdout(io.StringIO()):
                selector.main()
        reject("feeding-only candidate below the target ceiling needs a justification", select_ceiling)
        select_ceiling("--ceiling-justification", "CPU is near physical-core saturation; less worker CPU may relieve the main process")
        reachability = json.loads((ceiling_output / "selection.json").read_text())["expertium"]["target_reachability"]
        check("justified below-ceiling feeding candidate records the shortfall and its reason",
              reachability["feeding_only_candidate"] and not reachability["ceiling_reaches_target"] and reachability["justification"])
        shutil.rmtree(ceiling_output)
        (base / "iterations").mkdir()
        with patch.object(reporter, "ROOT", root), patch.object(reporter, "process_provenance", lambda: {"synthetic_only": True}), \
             patch.object(reporter, "update_run_log", lambda *_: None), patch.object(sys, "argv", ["report", "--output", str(new)]), redirect_stdout(io.StringIO()):
            reporter.main()
        report = json.loads((new / "comparison.json").read_text())
        check("report completes using only candidate runs", all(c["accepted_main"] for c in report["comparisons"].values()) and not (new / "baseline/F01").exists())
        check("report measures gains against recorded references", all(abs(c["references"]["previous"]["throughput_gain_percent"] - 20) < 1e-10 for c in report["comparisons"].values()))
        check("accepted result metadata is retained", accepted_reference(new, "baseline", root)["path"].endswith("F01-F08/result.json"))
        for kind in ("retune", "fastmode"):
            target = base / ("iteration-" + kind)
            p = deepcopy(protocol)
            p.update(iteration=kind, models=["baseline"])
            write(target / "protocol.json", p)
            write(target / "selection.json", {"baseline": selection["baseline"]})
            c = json.loads((new / "baseline/F01-F08/result.json").read_text())
            c.update(protocol_sha256=hashlib.sha256((target / "protocol.json").read_bytes()).hexdigest(),
                     reviews_per_second=120 if kind == "retune" else 150,
                     training_seconds=960 / (120 if kind == "retune" else 150))
            if kind == "fastmode":
                for row in c["per_user"]: row.update(curve_log_loss=0.5025, instant_log_loss=0.5025)
                c["evaluation"].update({key: 0.5025 for key in LOSS_METRICS})
            candidate_path = target / "baseline/F01-F08/result.json"
            write(candidate_path, c)
            candidate_path.with_name("updates.jsonl").write_bytes((new / "baseline/F01-F08/updates.jsonl").read_bytes())
            reduced = deepcopy(profile)
            reduced["inputs"] = [result_metadata(candidate_path, root)]
            counters = reduced["phases"]["training including user transitions"]["counters"]
            counters["nvidia_utilization.gpu"]["mean"] = 40
            counters["torch_cuda_peak_reserved_bytes"]["max"] = 2 * 2**30
            write(target / "profiles/baseline-F01-F08/result.json", reduced)
            check(kind + " resource reduction is read from recorded evidence", bool(resource_reductions(candidate_path, old / "baseline/F01/result.json", root)))
            check(kind + " large recorded saving is material", material_reduction(resource_reductions(candidate_path, old / "baseline/F01/result.json", root)))
            small = deepcopy(reduced)
            small["phases"]["training including user transitions"]["counters"].update(
                {"nvidia_utilization.gpu": {"mean": 69}, "torch_cuda_peak_reserved_bytes": {"max": 4 * 2**30 - 1}})
            write(target / "profiles/baseline-F01-F08/result.json", small)
            small_reductions = resource_reductions(candidate_path, old / "baseline/F01/result.json", root)
            check(kind + " marginal saving is recorded but not material", bool(small_reductions) and not material_reduction(small_reductions))
            write(target / "profiles/baseline-F01-F08/result.json", reduced)
            stale = deepcopy(reduced)
            stale["inputs"][0]["sha256"] = "different-run"
            profile_path = target / "profiles/baseline-F01-F08/result.json"
            write(profile_path, stale)
            check(kind + " mismatched profile identity cannot claim resource savings", not resource_reductions(candidate_path, old / "baseline/F01/result.json", root))
            write(profile_path, reduced)
            with patch.object(reporter, "ROOT", root), patch.object(reporter, "process_provenance", lambda: {"synthetic_only": True}), \
                 patch.object(reporter, "update_run_log", lambda *_: None), patch.object(sys, "argv", ["report", "--output", str(target)]), redirect_stdout(io.StringIO()):
                reporter.main()
            cmp = json.loads((target / "comparison.json").read_text())["comparisons"]["baseline"]
            check(kind + " report retains the main accepted reference", cmp["accepted_reference"] == p["references"]["baseline"]["previous"] and not cmp["accepted_main"])
            check(kind + " report records the correct next action", cmp["resource_retuning_pending"] if kind == "retune" else cmp["fastmode"])
            check(kind + " accepted-reference resolution never promotes the pending variant", accepted_reference(target, "baseline", root) == p["references"]["baseline"]["previous"])
        with patch.object(ledger, "ROOT", root), patch.object(ledger, "BASE", base):
            path = ledger.update_run_log()
            rows = [s for s in path.read_text().splitlines() if s.startswith("|")]
            check("LOG retains 25 columns", all(s.count("|") == 26 for s in rows))
            check("LOG uses fixed original baseline for cumulative speed", "+44.00%" in path.read_text())
        metadata = protocol["references"]["baseline"]["previous"]
        reference_path = root / metadata["path"]
        preserved = reference_path.read_bytes()
        reference_path.write_bytes(preserved + b" ")
        reject("changed historical evidence is rejected", lambda: read_reference(metadata, root))
        reference_path.write_bytes(preserved)
        # Separate synthetic workspace exercises the live controller, including legacy protocols.
        from scripts import run_isolated_gpu_r01 as launcher, isolate_gpu_r01 as isolation
        from scripts.gpu_r01_policy import launch_plan, reusable_profile
        from zipfile import ZipFile
        guard_root = output / "launch-fixtures"
        guard_base = guard_root / "artifacts/retraining/gpu-retrain-r01"
        history = guard_base / "iteration-0010"
        target = guard_base / "iteration-0011"
        (guard_root / "docs").mkdir(parents=True)
        live_loop = guard_root / "docs/GPU_RETRAIN_LOOP.md"
        live_loop.write_bytes((ROOT / "docs/GPU_RETRAIN_LOOP.md").read_bytes())
        (guard_root / "docs/GPU_OPTIMIZATIONS_LOG.md").write_text("# Synthetic test ledger\n")
        (guard_root / ".gitignore").write_text("artifacts/\ndata/\n")
        (guard_root / "AGENT.md").write_text("Synthetic test fixture only.\n")
        (guard_root / "data").mkdir()
        write(guard_base / "cohorts-user-holdout-v1.json", cohort)
        gp = deepcopy(protocol)
        gp.update(models=["baseline"], iteration="0010", loop_policy=None, references={},
                  implementation_sha256="new-source", prior_no_20_percent_gain_streak={"baseline": 2})
        # Historical protocols keep their 72% limit after the future default changes.
        gp["training"]["gpu_memory_fraction"] = 0.72
        write(history / "protocol.json", gp)
        (history / "loop-definition.md").write_bytes(live_loop.read_bytes())
        template = json.loads((new / "baseline/F01-F08/result.json").read_text())
        template["config"] = {"MODEL": "baseline", "SEED": 2025, "BATCH_CHUNKS": 128, "CHUNK_REVIEWS": 128,
                              "PREFETCH_WORKERS": 4, "PREFETCH_BATCHES": 8, "CPU_THREADS": 4, "EPOCHS": 1,
                              "DATA_PATH": "synthetic", "DATASET_REVISION": "synthetic", "DTYPE": "bfloat16",
                              "WKV_BACKEND": "nvrtc", "GPU_MEMORY_FRACTION": 0.72, "PEAK_LR": 7e-4, "WARMUP_STEPS": 5}
        template["protocol_sha256"] = hashlib.sha256((history / "protocol.json").read_bytes()).hexdigest()
        for track, speed, options in (("control", 100, {}), ("F01", 120, {"prefetch_users": True}),
                                      ("F01-F08", 144, plan["case_options_by_model"]["baseline"]["F01-F08"])):
            c = deepcopy(template)
            c.update(track=track, options=options, reviews_per_second=speed, training_seconds=960 / speed)
            write(history / "baseline" / track / "result.json", c)
            (history / "baseline" / track / "updates.jsonl").write_bytes((new / "baseline/F01-F08/updates.jsonl").read_bytes())
        gp.update(iteration="0011", references={"baseline": {
            "previous": result_metadata(history / "baseline/F01/result.json", guard_root),
            "original_baseline": result_metadata(history / "baseline/control/result.json", guard_root)}})
        write(target / "protocol.json", gp)
        (target / "loop-definition.md").write_bytes(live_loop.read_bytes())
        write(target / "selection.json", {"baseline": selection["baseline"]})
        request = SimpleNamespace(model="baseline", track="F01-F08", smoke=False, repeat_reason=None, check_reason=None)
        check("legacy protocol reuses matching completed candidate", launch_plan(guard_root, target, request)["action"] == "reuse")
        with patch.object(launcher, "subprocess", SimpleNamespace(run=lambda *a, **k: (_ for _ in ()).throw(AssertionError("GPU subprocess started")))), redirect_stdout(io.StringIO()):
            check("matching result bypasses isolation and subprocess execution", launcher.launch_case(request, target, root=guard_root) == 0)
        check("reuse record links original evidence without copying a fake run", not (target / "baseline/F01-F08/result.json").exists())
        request.smoke = True
        check("completed full-history case makes another smoke unnecessary", launch_plan(guard_root, target, request)["action"] == "reuse")
        request.smoke = False
        request.repeat_reason = "Repeat requested to investigate changed timing conditions"
        check("explicit repeat has its reason recorded", launch_plan(guard_root, target, request)["repeat_reason"] == request.repeat_reason)
        request.repeat_reason = None
        # Reusing a result must not manufacture another iteration or increment its stop count.
        (guard_base / "iterations").mkdir()
        with patch.object(reporter, "ROOT", guard_root), patch.object(reporter, "process_provenance", lambda: {"synthetic_only": True}), \
             patch.object(reporter, "update_run_log", lambda *_: None), patch.object(sys, "argv", ["report", "--output", str(target)]), redirect_stdout(io.StringIO()):
            reporter.main()
        reused_comparison = json.loads((target / "comparison.json").read_text())["comparisons"]["baseline"]
        check("reused measurements preserve the no-gain streak", reused_comparison["reused_candidate"] and reused_comparison["no_20_percent_gain_streak"] == 2)
        check("report links the effective definition", "Effective rules:" in (target / "summary.md").read_text())
        original_loop = live_loop.read_bytes()
        live_loop.write_bytes(original_loop + b"\nChanged instruction in synthetic fixture.\n")
        reject("live loop changes block legacy and frozen launches", lambda: launcher.launch_case(request, target, root=guard_root, runtime=Path("artifacts/legacy-runtime")))
        converted = original_loop.replace(b"\r\n", b"\n") if b"\r\n" in original_loop else original_loop.replace(b"\n", b"\r\n")
        live_loop.write_bytes(converted)
        check("line-ending conversion alone does not block launches", converted != original_loop and launch_plan(guard_root, target, request)["action"] == "reuse")
        live_loop.write_bytes(original_loop)
        changed = deepcopy(gp)
        changed.update(iteration="0012", candidate_tracks={"baseline": "F01-F07"}, implementation_sha256="changed-source")
        changed["case_options_by_model"]["baseline"] = {"F01": {"prefetch_users": True},
                                                       "F01-F07": {"prefetch_users": True, "padding_factor": 0.3}}
        fresh = guard_base / "iteration-0012"
        write(fresh / "protocol.json", changed)
        (fresh / "loop-definition.md").write_bytes(original_loop)
        write(fresh / "selection.json", {"baseline": {"track": "F01-F07"}})
        request.track = "F01-F07"
        request.smoke = True
        reject("new smoke requires a purpose tied to changed behaviour", lambda: launch_plan(guard_root, fresh, request))
        request.check_reason = "Verify changed padding retains complete review coverage"
        check("focused smoke records its purpose", launch_plan(guard_root, fresh, request)["check_reason"] == request.check_reason)
        request.track = "F01"
        # A differing execution mode prevents reuse and requires an explicit reference-repeat reason.
        changed["execution_mode"]["wkv_tile"] = 64
        write(fresh / "protocol.json", changed)
        reject("old protocol tag cannot bypass reference-repeat guard", lambda: launch_plan(guard_root, fresh, request))
        changed["execution_mode"]["wkv_tile"] = 32
        write(fresh / "protocol.json", changed)
        request.track, request.smoke = "F01-F07", False
        write(fresh / "comparison.json", {"comparisons": {"baseline": {"no_20_percent_gain_streak": 3}}})
        reject("later stop record blocks an older frozen runner", lambda: launcher.launch_case(request, fresh, root=guard_root, runtime=Path("artifacts/legacy-runtime")))
        reused_report = guard_base / "iteration-0013/comparison.json"
        write(reused_report, {"comparisons": {"baseline": {"reused_candidate": True, "no_20_percent_gain_streak": 0}}})
        reject("a reused report cannot reset a later recorded stop", lambda: launch_plan(guard_root, fresh, request))
        reused_report.unlink()
        write(fresh / "comparison.json", {"comparisons": {"baseline": {"no_20_percent_gain_streak": 2}}})
        failed_case = deepcopy(template)
        failed_case.update(status="failed", track="F01-F07", implementation_sha256="previous-broken-source",
                           options=changed["case_options_by_model"]["baseline"]["F01-F07"], review_presentations=100)
        write(history / "baseline/F01-F07/result.json", failed_case)
        reject("a repaired source still requires a reason to retry its failed case", lambda: launch_plan(guard_root, fresh, request))
        request.repeat_reason = "Repair the failed padding implementation while preserving review coverage"
        check("a repaired failure permits a recorded retry", launch_plan(guard_root, fresh, request)["repeat_reason"] == request.repeat_reason)
        # Check and profile caches are keyed by their actual evidence.
        check_fixture = guard_root / "artifacts/diagnostics/passed-check/result.json"
        write(check_fixture, {"status": "passed", "check_suite": "example", "check_inputs": {"source": "same"}})
        check("unchanged checks are reused", reusable_check(guard_root, "example", {"source": "same"}) == check_fixture)
        check("changed check inputs require new checks", reusable_check(guard_root, "example", {"source": "changed"}) is None)
        pr = deepcopy(profile)
        pr["inputs"] = [result_metadata(history / "baseline/F01/result.json", guard_root)]
        pr["workload"]["minimum_observation_seconds"] = 120
        profile_fixture = history / "profiles/baseline-F01/result.json"
        write(profile_fixture, pr)
        check("existing profile is found across iteration folders", reusable_profile(history / "baseline/F01/result.json", guard_root, 120) == profile_fixture)
        pr["workload"].update(minimum_observation_seconds=60, training_observed_seconds=150, full_history_profile_available=True)
        write(profile_fixture, pr)
        check("adequate complete profile survives a changed duration setting", reusable_profile(history / "baseline/F01/result.json", guard_root, 120) == profile_fixture)
        check("profile below the requested observation duration is not reused", reusable_profile(history / "baseline/F01/result.json", guard_root, 180) is None)
        ref = history / "baseline/F01/result.json"
        ref_bytes = ref.read_bytes()
        ref.write_bytes(ref_bytes + b" ")
        check("changed raw evidence invalidates profile reuse", reusable_profile(ref, guard_root, 120) is None)
        ref.write_bytes(ref_bytes)
        # Actual source extraction, Git initialization, isolated environment and export; GPU subprocess is mocked.
        archive = guard_root / "artifacts/sources/test/source.zip"
        archive.parent.mkdir(parents=True)
        frozen_script = b"# Synthetic worker, never executed on GPU.\n"
        with ZipFile(archive, "w") as z:
            z.writestr("scripts/run_gpu_r01.py", frozen_script)
            z.writestr("scripts/gpu_r01_training.py", b"# GPU_R01_SHARED_DATA_PATH supported in synthetic fixture.\n")
            z.writestr("source-manifest.json", json.dumps({"git_commit": "synthetic-only", "files": {
                "scripts/run_gpu_r01.py": hashlib.sha256(frozen_script).hexdigest(),
                "scripts/gpu_r01_training.py": hashlib.sha256(b"# GPU_R01_SHARED_DATA_PATH supported in synthetic fixture.\n").hexdigest()}}))
        changed["preparation_provenance"] = {"source_archive": archive.relative_to(guard_root).as_posix()}
        write(fresh / "protocol.json", changed)
        launched = []
        def fake_child(command, *, cwd, env):
            launched.append((command, cwd, env))
            c = deepcopy(template)
            c.update(track="F01-F07", options=changed["case_options_by_model"]["baseline"]["F01-F07"],
                     implementation_sha256="changed-source", protocol_sha256=hashlib.sha256((fresh / "protocol.json").read_bytes()).hexdigest())
            write(cwd / fresh.relative_to(guard_root) / "baseline/F01-F07/result.json", c)
            return SimpleNamespace(returncode=0)
        with patch.object(isolation, "ROOT", guard_root), patch.object(launcher, "subprocess", SimpleNamespace(run=fake_child)), redirect_stdout(io.StringIO()):
            check("fresh case isolates automatically before launching", launcher.launch_case(request, fresh, root=guard_root) == 0)
        command, runtime, environment = launched[0]
        check("isolated code retains its frozen source", (runtime / "scripts/run_gpu_r01.py").read_bytes() == frozen_script)
        check("isolated runtime has independent Git provenance", (runtime / ".git").is_dir() and json.loads((runtime / "isolation.json").read_text())["isolated_git_commit"])
        check("isolated runtime shares data without copying it or requiring a junction", environment["GPU_R01_SHARED_DATA_PATH"] == str(guard_root / changed["dataset_path"]) and not (runtime / "data").exists())
        check("temporary files stay within runtime", Path(environment["TEMP"]).is_relative_to(runtime) and environment["TEMP"] == environment["TMP"])
        check("live controller issues a matching launch ticket", json.loads(Path(environment["GPU_R01_LAUNCH_TICKET"]).read_text())["track"] == request.track)
        check("results are exported without an active leftover lock", (fresh / "baseline/F01-F07/result.json").exists() and not (guard_root / "artifacts/gpu-r01.lock").exists())
        with patch.object(runner, "ROOT", runtime), patch.dict("os.environ", {}, clear=True):
            reject("direct frozen-worker invocation requires current controller", lambda: runner.run_case(request, runtime / fresh.relative_to(guard_root)))
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="failed", error=str(exc))
        raise
    finally:
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(result["checks"]), "synthetic_only": True}))


if __name__ == "__main__":
    main()
