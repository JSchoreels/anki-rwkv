#!/usr/bin/env python
"""Decay-only training from a full-run checkpoint (Expertium's WSD decay phase; docs/EXPERTIUM_CATCH_UP.md item 2).

Upstream trains warmup-stable (WS) to a checkpoint, then runs a separate decay process from it (TRAIN_MODE "D",
LR from peak to 0). This does the same on the R01 trainer: it starts from an epoch's master weights and AdamW state
(read-only, any path), trains on a fresh shuffle of that run's training users until `--fraction` of an epoch's
reviews, with the learning rate decaying from 7e-4 to 0 over exactly those reviews, then scores held-out users.

  prepare --source-run DIR --epoch N --out artifacts/... [--fraction 0.25 --shape one_minus_sin
          --evaluate quick|validation|validation+final_test|none --seed 1 --extra-options JSON]
  run     --out artifacts/...        (one GPU process; submit through scripts/r01_queue.py)

The source run directory is only read. Its label database is opened read-only (lock=False) for the benchmark
metrics. Compare against the source epoch's own scores on the same users (matched control).
"""
from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_full_training import atomic_json, rows_for  # noqa: E402

MODEL = "expertium"


def source_case(source_run, config, epoch):
    folder = source_run / f"epoch-{epoch:02d}" / config["model"] / config["track"]
    result = json.loads((folder / "result.json").read_text())
    if result.get("status") not in ("evaluating", "complete") or not (folder / "optimizer.pth").exists():
        raise RuntimeError(f"{folder} has no finished epoch checkpoint (status {result.get('status')})")
    return folder, result


def prepare(source_run, epoch, out, fraction, shape, evaluate, seed, extra_options, final):
    if out.exists():
        raise FileExistsError(f"{out} exists; pick a new output")
    config = json.loads((source_run / "run.json").read_text())
    folder, result = source_case(source_run, config, epoch)
    counts = config["review_counts"]
    order = config["train_users"][:]
    random.Random(seed).shuffle(order)
    budget = fraction * config["training_reviews_per_epoch"]
    users, reviews = [], 0
    for user in order:
        if reviews >= budget:
            break
        users.append(user)
        reviews += counts[str(user)]
    sets = {"quick": config["quick_validation_users"], "validation": config["validation_users"],
            "validation+final_test": sorted(config["validation_users"] + config["final_test_users"]), "none": []}
    evaluation = sets[evaluate]
    options = dict(config["options"], **(extra_options or {}))
    from scripts.expertium_options import compatible_options
    options, dropped = compatible_options(options)
    dataset = Path(config["dataset_path"])
    dataset = dataset if dataset.is_absolute() else source_run.parents[2] / dataset
    schedule = {"decay_start": 0.0, "decay_end": 1.0, "shape": shape, "final": final}
    full_run = {"epoch": epoch, "evaluate": evaluate != "none", "init_master": str(folder / "master.pth"),
                "init_optimizer": str(folder / "optimizer.pth"), "update_offset": result["update_offset_end"],
                "lr_schedule": schedule, "benchmark_db": str(source_run / "label-db"), "evaluation_set": evaluate}
    protocol = {"rows": rows_for(users, counts), "evaluation_rows": rows_for(evaluation, counts),
                "dataset_path": str(dataset), "dataset_revision": config["dataset_revision"],
                "training": {"epochs": 1, "chunk_reviews": 128, "gpu_memory_fraction": 0.95},
                "execution_mode": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}, "full_run": full_run}
    decay = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_run": str(source_run), "source_epoch": epoch,
             "source_case": str(folder), "source_update_offset_end": result["update_offset_end"], "model": MODEL,
             "track": config["track"], "options": options, "dropped_width32_options": dropped,
             "fraction_of_epoch_reviews": fraction, "training_users": len(users), "training_reviews": reviews,
             "shuffle_seed": seed, "lr": f"7e-4 -> {final * 7e-4:g}, shape {shape}, over all {reviews:,} reviews",
             "evaluation_set": evaluate, "evaluation_users": len(evaluation),
             "control": f"epoch-{epoch:02d} scores of the same users in {folder / 'result.json'}"}
    out.mkdir(parents=True)
    atomic_json(out / "decay.json", decay)
    atomic_json(out / "protocol.json", protocol)
    print(json.dumps({key: decay[key] for key in ("training_users", "training_reviews", "evaluation_users", "lr")}))
    return 0


def readonly_benchmark_info(db_path, db_size, user_id):
    """rwkv.get_result.get_benchmark_info without write access to another checkout's label database."""
    import lmdb
    from rwkv.utils import load_tensor
    env = lmdb.open(str(db_path), readonly=True, lock=False, subdir=True)
    try:
        with env.begin(write=False) as txn:
            if txn.get(f"{user_id}_review_ths".encode()) is not None:
                return (load_tensor(txn, f"{user_id}_review_ths", device="cpu").tolist(),
                        load_tensor(txn, f"{user_id}_rmse_bins", device="cpu").tolist())
    finally:
        env.close()
    return [], []


def run(out):
    decay = json.loads((out / "decay.json").read_text())
    protocol = json.loads((out / "protocol.json").read_text())
    os.environ["GPU_BATCH1_DATA_PATH"] = protocol["dataset_path"]
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = protocol["execution_mode"]["CUBLAS_WORKSPACE_CONFIG"]
    os.environ["CUPY_CACHE_DIR"] = str(ROOT / "artifacts/cache/cupy-cache")
    import torch
    from rwkv import get_result
    from scripts import gpu_batch1, gpu_r01_training
    from scripts.run_gpu_r01 import R01SourceWatch
    get_result.get_benchmark_info = readonly_benchmark_info
    case = decay["options"]
    torch._C._set_graph_executor_optimize(bool(case.get("jit_optimize", False)))
    torch.use_deterministic_algorithms(not case.get("nondeterministic", False), warn_only=False)
    torch.utils.deterministic.fill_uninitialized_memory = not case.get("skip_uninitialized_fill", False)
    gpu_batch1.TRACKS = {decay["track"]: case}
    gpu_r01_training.SourceWatch = R01SourceWatch
    if case.get("gpu_packed_features"):
        from scripts.gpu_r01_staging import training_batches
        from scripts.gpu_b01 import transfer_batches
        gpu_batch1.training_batches = training_batches
        gpu_batch1.transfer_batches = transfer_batches
    elif case.get("prefetch_users"):
        from scripts.gpu_r01_prefetch import training_batches
        gpu_batch1.training_batches = training_batches
    args = Namespace(model=MODEL, track=decay["track"], smoke=False, expected_source=None,
                     output=out.relative_to(ROOT).as_posix())
    return gpu_r01_training.run_case(args)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--shape", default="one_minus_sin", choices=("one_minus_sin", "cosine", "linear", "sqrt"))
    parser.add_argument("--final", type=float, default=0.0, help="final learning rate as a fraction of the peak")
    parser.add_argument("--evaluate", default="quick", choices=("quick", "validation", "validation+final_test", "none"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--extra-options", type=json.loads, default=None)
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    if not out.is_relative_to(ROOT / "artifacts"):
        parser.error("--out must be under this checkout's artifacts/")
    if args.command == "prepare":
        if args.source_run is None or args.epoch is None or not 0 < args.fraction <= 1:
            parser.error("prepare needs --source-run, --epoch and 0 < --fraction <= 1")
        return prepare(args.source_run.resolve(), args.epoch, out, args.fraction, args.shape, args.evaluate, args.seed,
                       args.extra_options, args.final)
    return run(out)


if __name__ == "__main__":
    raise SystemExit(main())
