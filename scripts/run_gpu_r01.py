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
from scripts import run_gpu_batch1 as shared

BASE = ROOT / "artifacts/retraining/gpu-retrain-r01"
COHORT = BASE / "cohorts-user-holdout-v1.json"
OPTIONS = {"control": {}, "F01": {"prefetch_users": True},
           "F03": {"worker_local": True, "packed_ipc": True},
           "F04": {"async_transfer": True}}


class R01SourceWatch(shared.SourceWatch):
    def __init__(self, provenance):
        super().__init__(provenance)
        with ZipFile(ROOT / provenance["source_archive"]) as archive:
            manifest = json.loads(archive.read("source-manifest.json"))["files"]
        for name in ("scripts/run_gpu_r01.py", "scripts/gpu_r01_training.py", "scripts/gpu_r01_prefetch.py", "scripts/gpu_resource_monitor.py"):
            self.hashes[name] = manifest[name]
        self.implementation_sha256 = hashlib.sha256(json.dumps(self.hashes, sort_keys=True).encode()).hexdigest()
        self.check()


def prepare(output):
    import numpy as np
    if (output / "protocol.json").exists():
        raise FileExistsError("Preserve the frozen iteration; use a new output for a new round")
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
        previous = 0
        for count in (100, 250, 500):
            added = sampled[previous:count]
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
            previous = count
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
    protocol = {**stage, "protocol_status": "frozen before experiments", "iteration": "0002",
                "authorization": "2026-10-11: user requested exactly one iteration of current docs/GPU_RETRAIN_LOOP.md",
                "cohort_file": COHORT.relative_to(ROOT).as_posix(), "cohort_sha256": hashlib.sha256(COHORT.read_bytes()).hexdigest(),
                "dataset_path": cohort["dataset_path"], "dataset_revision": cohort["dataset_revision"],
                "evaluation_users": len(stage["heldout_users"]), "training_user_count": len(stage["training_users"]),
                "evaluation": "unseen users; complete histories, chunked at 128 reviews; no gradient updates or fitting on held-out users",
                "holdout": "20% disjoint user holdout; mask labels pointing outside the retained full history",
                "seeds": [2025], "sampling_seed": cohort["sampling_seed"], "models": ["baseline", "expertium"],
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
                               "status": "preliminary only; 20 distinct evaluation users do not satisfy the 500-evaluation-user validation gate"},
                "timing": "raw review presentations/s uses complete training including loads, preparation/worker startup+shutdown, transfers, updates and logs; setup/primer/save/evaluation/process wall recorded separately",
                "frozen_utc": datetime.now(timezone.utc).isoformat(), "preparation_provenance": provenance,
                "implementation_sha256": R01SourceWatch(provenance).implementation_sha256}
    shared.atomic_json(output / "protocol.json", protocol)
    print(json.dumps({"protocol": str(output / "protocol.json"), "train_users": 80, "heldout_users": 20,
                      "training_reviews": stage["training_reviews"], "heldout_reviews": stage["validation_reviews"],
                      "planned_updates": protocol["planned_updates"]}))


def run_case(args, output):
    import psutil
    protocol = json.loads((output / "protocol.json").read_text())
    os.environ["GPU_BATCH1_DATA_PATH"] = protocol["dataset_path"]
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = protocol["execution_mode"]["CUBLAS_WORKSPACE_CONFIG"]
    os.environ["CUPY_CACHE_DIR"] = str(ROOT / "artifacts/cache/cupy-cache")
    provenance = process_provenance()
    if R01SourceWatch(provenance).implementation_sha256 != protocol["implementation_sha256"]:
        raise RuntimeError("Training source changed after protocol freeze; preserve this iteration")
    if args.track != "control" and not args.smoke:
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
    gpu_batch1.TRACKS = OPTIONS
    gpu_r01_training.SourceWatch = R01SourceWatch
    if args.track == "F01":
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
        result.update(iteration="0002", execution_mode=protocol["execution_mode"], checkpoint_saves=saves,
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
    parser.add_argument("--case", action="store_true")
    parser.add_argument("--model", choices=("baseline", "expertium"))
    parser.add_argument("--track", choices=tuple(OPTIONS), default="control")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Output must remain under workspace artifacts")
    if args.prepare:
        return prepare(output)
    if args.case and args.model:
        return run_case(args, output)
    parser.error("Choose --prepare or --case --model")


if __name__ == "__main__":
    raise SystemExit(main())
