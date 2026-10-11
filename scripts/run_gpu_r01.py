"""One R01 screen with nested random cohorts and disjoint held-out users."""

from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance
from scripts.gpu_r01_policy import POLICY, accepted_reference, read_reference, require_active, streak
from scripts import run_gpu_batch1 as shared

BASE = ROOT / "artifacts/retraining/gpu-retrain-r01"
COHORT = BASE / "cohorts-user-holdout-v1.json"
OPTIONS = {"control": {}, "F01": {"prefetch_users": True},
           "F03": {"worker_local": True, "packed_ipc": True},
           "F04": {"async_transfer": True},
           "F01-F06": {"prefetch_users": True, "reduced_sync": True},
           "F01-F16": {"prefetch_users": True, "preparation_workers": 8}}


class R01SourceWatch(shared.SourceWatch):
    def __init__(self, provenance):
        super().__init__(provenance)
        with ZipFile(ROOT / provenance["source_archive"]) as archive:
            manifest = json.loads(archive.read("source-manifest.json"))["files"]
        for name in ("scripts/run_gpu_r01.py", "scripts/gpu_r01_training.py", "scripts/gpu_r01_prefetch.py", "scripts/gpu_r01_sync.py", "scripts/gpu_r01_staging.py", "scripts/gpu_b01.py", "scripts/gpu_resource_monitor.py", "scripts/gpu_r01_policy.py"):
            self.hashes[name] = manifest[name]
        self.implementation_sha256 = hashlib.sha256(json.dumps(self.hashes, sort_keys=True).encode()).hexdigest()
        self.check()


def prepare(output, iteration="0002", previous=None, plan=None):
    import numpy as np
    if (output / "protocol.json").exists():
        raise FileExistsError("Preserve the frozen iteration; use a new output for a new round")
    previous_comparison = None
    models = ["baseline", "expertium"]
    if previous is not None:
        previous = (ROOT / previous).resolve()
        previous_comparison = json.loads((previous / "comparison.json").read_text())
        models = [model for model in models if model in previous_comparison["comparisons"]
                  and streak(previous_comparison["comparisons"][model]) < 3]
        if not models:
            raise RuntimeError("R01 stopped: both models reached three consecutive iterations without a 20% gain")
    provenance = process_provenance()
    if not COHORT.exists():
        counts_path = ROOT / "artifacts/diagnostics/check-temp/first-5000-review-counts.json"
        counts = json.loads(counts_path.read_text())
        old = json.loads((ROOT / "artifacts/benchmarks/gpu-baseline-b01/protocol.json").read_text())
        reserved = old["independent_final_test_users"]
        data = ROOT / old["dataset_path"]
        manifest = json.loads((data / "download_manifest.json").read_text())
        if manifest["revision"] != counts["revision"]:
            raise ValueError("Saved counts and pinned dataset revisions differ")
        ordered = sorted(range(1, 5001), key=lambda user: (counts["per_user"][str(user)], user))
        deciles = {int(user): index + 1 for index, group in enumerate(np.array_split(ordered, 10)) for user in group}
        rng = np.random.default_rng(20261011)
        sampled = [int(user) for user in rng.permutation([user for user in range(1, 5001) if user not in set(reserved)])[:500]]
        train, heldout, stages = [], [], {}
        previous_count = 0
        for count in (100, 250, 500):
            added = sampled[previous_count:count]
            # Stable roles across nested cohorts; each added block is 80/20.
            train.extend(added[:len(added) * 4 // 5])
            heldout.extend(added[len(added) * 4 // 5:])
            def rows(ids):
                return [{"user_id": user, "history_decile": deciles[user],
                         "available_reviews": counts["per_user"][str(user)],
                         "train_range": [0, counts["per_user"][str(user)]],
                         "validation_range": [0, counts["per_user"][str(user)]]} for user in sorted(ids)]
            stages[str(count)] = {"users": count, "training_users": sorted(train), "heldout_users": sorted(heldout),
                                  "rows": rows(train), "evaluation_rows": rows(heldout),
                                  "training_reviews": sum(counts["per_user"][str(user)] for user in train),
                                  "validation_reviews": sum(counts["per_user"][str(user)] for user in heldout)}
            previous_count = count
        shared.atomic_json(COHORT, {"sampling_seed": 20261011, "sampling": "random without replacement from IDs 1-5000, excluding the preserved independent final-test cohort",
                                   "dataset_revision": counts["revision"], "dataset_path": old["dataset_path"],
                                   "counts_source": counts_path.relative_to(ROOT).as_posix(),
                                   "counts_sha256": hashlib.sha256(counts_path.read_bytes()).hexdigest(),
                                   "independent_final_test_users": reserved, "stages": stages,
                                   "prepared_utc": datetime.now(timezone.utc).isoformat(), "provenance": provenance})
    cohort = json.loads(COHORT.read_text())
    stage = cohort["stages"]["100"]
    for count, value in cohort["stages"].items():
        train, heldout = set(value["training_users"]), set(value["heldout_users"])
        if train & heldout or len(train) + len(heldout) != int(count) or len(heldout) * 5 != int(count):
            raise ValueError("Cohort roles/counts are invalid")
        if (train | heldout) & set(cohort["independent_final_test_users"]):
            raise ValueError("Independent final-test overlap")
    output.mkdir(parents=True, exist_ok=True)
    protocol = {**stage, "protocol_status": "frozen before experiments", "iteration": iteration,
                "authorization": "2026-10-11: user requested exactly one iteration of current docs/GPU_RETRAIN_LOOP.md",
                "cohort_file": COHORT.relative_to(ROOT).as_posix(), "cohort_sha256": hashlib.sha256(COHORT.read_bytes()).hexdigest(),
                "dataset_path": cohort["dataset_path"], "dataset_revision": cohort["dataset_revision"],
                "evaluation_users": len(stage["heldout_users"]), "training_user_count": len(stage["training_users"]),
                "evaluation": "unseen users; complete histories, chunked at 128 reviews; no gradient updates or fitting on held-out users",
                "holdout": "20% disjoint user holdout; mask labels pointing outside the retained full history",
                "seeds": [2025], "sampling_seed": cohort["sampling_seed"], "models": models,
                "loop_policy": POLICY,
                "training": {"epochs": 1, "chunk_reviews": 128, "baseline_chunks_workers": [128, 4],
                             "expertium_chunks_workers": [256, 6], "dtype": "BF16", "master_dtype": "FP32",
                             "peak_lr": 7e-4, "warmup_updates": 5, "clip": 0.5, "cpu_threads": 4,
                             "prefetch_depth": "2 times preparation workers", "gpu_memory_fraction": 0.72,
                             "initialization": "fresh seed-paired weights and AdamW; no checkpoint reuse"},
                "execution_mode": {"torch_graph_executor_optimize": False, "deterministic_algorithms": True,
                                   "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "wkv_tile": 32},
                "planned_updates": {model: sum((row["available_reviews"] + chunks * 128 - 1) // (chunks * 128) for row in stage["rows"])
                                    for model, chunks in (("baseline", 128), ("expertium", 256))},
                "budget": "two fresh baselines plus one profile-selected candidate per architecture; one training seed; no larger stages in this iteration",
                "eligible_tracks": {key: value for key, value in OPTIONS.items() if key != "control"},
                "acceptance": {"relative_loss_limit": 0.01, "baseline_loss_limit": 0.02, "target_throughput_gain": 0.20,
                               "metrics": "Curve/Instant binary logloss, label-weighted and user-averaged, each separately",
                               "parity": "compare update metadata/loss/norm, initial/final weight hashes and held-out input hashes; exact final weights and scores preferred",
                               "resamples": 10000, "bootstrap_seed": 20261012, "family_tests": 8, "alpha": 0.05,
                               "quality_gate": "measured relative loss regression <=1%; confidence intervals reported separately; original baseline <=2% preferred",
                               "status": "preliminary only; 20 distinct evaluation users do not satisfy the 500-evaluation-user validation gate"},
                "timing": "raw review presentations/s uses complete training including loads, preparation/worker startup+shutdown, transfers, updates and logs; setup/primer/save/evaluation/process wall recorded separately",
                "frozen_utc": datetime.now(timezone.utc).isoformat(), "preparation_provenance": provenance,
                "implementation_sha256": R01SourceWatch(provenance).implementation_sha256}
    if previous is not None:
        old_protocol = json.loads((previous / "protocol.json").read_text())
        if old_protocol["cohort_sha256"] != protocol["cohort_sha256"]:
            raise ValueError("Continuation must reuse the exact frozen cohort")
        references, controls = {}, {}
        for model in protocol["models"]:
            accepted = accepted_reference(previous, model, ROOT)
            incumbent, _ = read_reference(accepted, ROOT)
            controls[model] = incumbent["track"]
            baseline = old_protocol.get("references", {}).get(model, {}).get("original_baseline")
            if baseline is None:
                from scripts.gpu_r01_policy import result_metadata
                baseline = result_metadata(previous / model / "control/result.json", ROOT)
            read_reference(baseline, ROOT)
            references[model] = {"previous": accepted, "original_baseline": baseline}
        protocol.update(control_tracks=controls,
                        previous_iteration=old_protocol["iteration"], references=references,
                        authorization="2026-10-11: user requested one new iteration",
                        budget="reuse recorded original baselines and accepted runs; one selected candidate per active model; focused checks as needed; no larger cohort")
        protocol["acceptance"]["family_tests"] = len(models) * 4 * 2
        protocol["acceptance"]["comparisons"] = "measured regression vs previous accepted run <=1%; original baseline <=2% preferred; uncertainty reported separately"
        protocol["prior_no_20_percent_gain_streak"] = {model: previous_comparison["comparisons"][model].get("no_20_percent_gain_streak", previous_comparison["comparisons"][model].get("no_20_percent_gain_streak_after_this_iteration", 0)) for model in protocol["models"]}
        protocol["resource_retuning_pending"] = {model: previous_comparison["comparisons"][model].get("resource_retuning_pending", False) for model in models}
        protocol["retuning_candidates"] = {model: previous_comparison["comparisons"][model]["retuning_candidate"]
                                          for model in models if protocol["resource_retuning_pending"][model]}
        if plan is None:
            if any(protocol["resource_retuning_pending"].values()):
                raise ValueError("Investigate the recorded resource-saving candidate and provide a retuning plan")
            candidates = {model: "F01-F06" if model == "baseline" else "F01-F16" for model in models}
            proposed = {"control_tracks": controls, "candidate_tracks": candidates,
                        "case_options_by_model": {model: {controls[model]: read_reference(references[model]["previous"], ROOT)[0]["options"],
                                                          candidates[model]: OPTIONS[candidates[model]]} for model in models},
                        "case_explanations": {}}
        else:
            plan_path = (ROOT / plan).resolve()
            proposed = json.loads(plan_path.read_text())
            protocol.update(plan_path=plan_path.relative_to(ROOT).as_posix(), plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest())
        protocol["retuning_dispositions"] = proposed.get("retuning_dispositions", {})
        for model, review in protocol["retuning_dispositions"].items():
            if model in protocol["retuning_candidates"] and review.get("decision") == "continue from accepted reference":
                if not review.get("evidence"):
                    raise ValueError("Retuning disposition requires recorded resource evidence")
                protocol["retuning_candidates"].pop(model)
        for model in models:
            if proposed["control_tracks"][model] != controls[model]:
                raise ValueError("Plan must retain the previous accepted run")
            if proposed["candidate_tracks"][model] == controls[model]:
                raise ValueError("Choose an improvement candidate rather than rerunning the accepted configuration")
            if set(proposed["case_options_by_model"][model]) != {controls[model], proposed["candidate_tracks"][model]}:
                raise ValueError("Plan must describe the accepted configuration and one candidate")
            incumbent, _ = read_reference(references[model]["previous"], ROOT)
            if proposed["case_options_by_model"][model][controls[model]] != incumbent["options"]:
                raise ValueError("Accepted configuration changed")
        protocol.update(candidate_tracks={m: proposed["candidate_tracks"][m] for m in models},
                        case_options_by_model={m: proposed["case_options_by_model"][m] for m in models},
                        case_explanations=proposed.get("case_explanations", {}),
                        eligible_tracks={m: proposed["candidate_tracks"][m] for m in models},
                        check_policy="reuse applicable profiles; focused smoke checks for changed code, no mandatory paired repeats")
    protocol["loop_sha256"] = hashlib.sha256((ROOT / "docs/GPU_RETRAIN_LOOP.md").read_bytes()).hexdigest()
    shared.atomic_json(output / "protocol.json", protocol)
    print(json.dumps({"protocol": str(output / "protocol.json"), "train_users": 80, "heldout_users": 20,
                      "training_reviews": stage["training_reviews"], "heldout_reviews": stage["validation_reviews"],
                      "planned_updates": protocol["planned_updates"]}))


def run_case(args, output):
    import psutil
    protocol = json.loads((output / "protocol.json").read_text())
    require_active(args.model, protocol.get("prior_no_20_percent_gain_streak", {}).get(args.model, 0))
    if args.model not in protocol["models"]:
        raise ValueError("Model is outside the active iteration")
    control_track = protocol.get("control_tracks", {}).get(args.model, protocol.get("control_track", "control"))
    if protocol.get("loop_policy") == POLICY and "references" in protocol and args.track == control_track:
        raise ValueError("Reuse the recorded accepted run; a control rerun needs a separate justified protocol")
    os.environ["GPU_BATCH1_DATA_PATH"] = protocol["dataset_path"]
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = protocol["execution_mode"]["CUBLAS_WORKSPACE_CONFIG"]
    os.environ["CUPY_CACHE_DIR"] = str(ROOT / "artifacts/cache/cupy-cache")
    provenance = process_provenance()
    if R01SourceWatch(provenance).implementation_sha256 != protocol["implementation_sha256"]:
        raise RuntimeError("Training source changed after protocol freeze; preserve this iteration")
    options = protocol.get("case_options_by_model", {}).get(args.model, protocol.get("case_options", OPTIONS))
    if args.track not in options:
        raise ValueError("Track is outside the frozen iteration budget")
    if args.track != control_track and not args.smoke:
        selection = json.loads((output / "selection.json").read_text())
        if selection[args.model]["track"] != args.track:
            raise ValueError("Candidate was not selected from baseline resource evidence")
    folder = output / args.model / args.track
    if (folder / "result.json").exists():
        raise FileExistsError("Preserve every prior case; no implicit overwrite")
    if args.smoke:
        output = output / "functional-smoke-v1"
        folder = output / args.model / args.track
        if (folder / "result.json").exists():
            raise FileExistsError("Preserve the previous smoke")
        shared.atomic_json(output / "protocol.json", protocol)
    lock = ROOT / "artifacts/gpu-r01.lock"
    other = ROOT / "artifacts/gpu-batch1.lock"
    if other.exists() and psutil.pid_exists(json.loads(other.read_text())["pid"]):
        raise RuntimeError("Another GPU suite is active")
    with lock.open("x") as file:
        json.dump({"pid": os.getpid(), "output": output.relative_to(ROOT).as_posix()}, file)
    import torch
    from scripts import gpu_batch1, gpu_power, gpu_r01_training
    from scripts.gpu_resource_monitor import ResourceMonitor
    if torch.cuda.is_initialized():
        raise RuntimeError("R01 requires a fresh CUDA process")
    torch._C._set_graph_executor_optimize(False)
    torch.use_deterministic_algorithms(True, warn_only=False)
    gpu_batch1.TRACKS = options
    gpu_r01_training.SourceWatch = R01SourceWatch
    if options[args.track].get("gpu_packed_features"):
        from scripts.gpu_r01_staging import training_batches
        from scripts.gpu_b01 import transfer_batches
        gpu_batch1.training_batches = training_batches
        gpu_batch1.transfer_batches = transfer_batches
    elif options[args.track].get("prefetch_users"):
        from scripts.gpu_r01_prefetch import training_batches
        gpu_batch1.training_batches = training_batches
    args.output = output.relative_to(ROOT).as_posix()
    args.expected_source = protocol["implementation_sha256"]
    saves = []
    save = torch.save
    def measured_save(value, destination, *values, **kwargs):
        started = time.perf_counter()
        try:
            return save(value, destination, *values, **kwargs)
        finally:
            if Path(destination).name == "master.pth":
                saves.append({"elapsed_seconds": time.perf_counter() - started, "path": Path(destination).relative_to(ROOT).as_posix()})
    try:
        with ResourceMonitor(folder) as monitor:
            power = gpu_power.limited_gpu
            gpu_power.limited_gpu = lambda *values: monitor.training_phase(power, *values)
            torch.save = measured_save
            try:
                code = gpu_r01_training.run_case(args)
            finally:
                gpu_power.limited_gpu = power
                torch.save = save
        result = json.loads((folder / "result.json").read_text())
        result.update(iteration=protocol["iteration"], execution_mode=protocol["execution_mode"], checkpoint_saves=saves,
                      protocol_sha256=hashlib.sha256((output / "protocol.json").read_bytes()).hexdigest(),
                      process_wall_seconds=time.time() - psutil.Process().create_time(),
                      evaluation_scope=protocol["evaluation"], quality_status="preliminary screen; broader validation pending")
        shared.atomic_json(folder / "result.json", result)
        print(json.dumps({key: result.get(key) for key in ("model", "track", "status", "training_seconds", "reviews_per_second", "evaluation", "error")}))
        return code
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/retraining/gpu-retrain-r01/iteration-0002"))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--iteration", default="0002")
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--case", action="store_true")
    parser.add_argument("--model", choices=("baseline", "expertium"))
    parser.add_argument("--track", default="control")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Output must remain under workspace artifacts")
    if args.prepare:
        return prepare(output, args.iteration, args.previous, args.plan)
    if args.case and args.model:
        return run_case(args, output)
    parser.error("Choose --prepare or --case --model")


if __name__ == "__main__":
    raise SystemExit(main())
