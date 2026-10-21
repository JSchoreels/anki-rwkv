#!/usr/bin/env python
"""Decay-only training from a full-run checkpoint (Expertium's WSD decay phase; docs/EXPERTIUM_CATCH_UP.md item 2).

Upstream trains warmup-stable (WS) to a checkpoint, then runs a separate decay process from it (TRAIN_MODE "D",
LR from peak to 0). This does the same on the R01 trainer: it starts from an epoch's master weights and AdamW state
(read-only, any path), trains on a fresh shuffle of that run's training users until `--fraction` of an epoch's
reviews, with the learning rate decaying from 7e-4 to 0 over exactly those reviews, then scores held-out users.

  prepare --source-run DIR --epoch N --out artifacts/... [--fraction 0.25 --shape one_minus_sin
          --evaluate quick|validation|validation+final_test|none --seed 1 --extra-options JSON
          --fresh-optimizer --train-tier gate|final]   (fresh: start with an empty optimizer state instead of the source
          optimizer.pth; train-tier: train on that accuracy-loop tier's users from artifacts/accuracy/tiers.json instead of
          the source run's train_users, e.g. the 8k final tier from a gate-tier checkpoint; SM55)
  run     --out artifacts/...        (one GPU process; submit through scripts/r01_queue.py)
  auto    <prepare arguments>         (prepare unless --out exists, then run; lets a queued job wait for the epoch)
  compare --out artifacts/...        (decayed vs the source epoch on the same held-out users; summary.md/json)
  smoke   --out artifacts/...        (60-120 s GPU smoke of a prepared decay: the same options and checkpoint on the
                                      trainer's smoke rows, 2 users x 512 reviews, in a scratch copy under
                                      artifacts/small-model/smoke/ that is deleted on success; the teacher cache is
                                      not used. Put it at the start of a long ticket: `smoke ... && run ...`)

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


def tier_users(config, name):
    """Training users and reviews per epoch of accuracy-loop tier `name`, refused unless the tier comes from the same
    split file and dataset as the source run (so the review counts and the held-out pools stay those of the source)."""
    tiers = json.loads((ROOT / "artifacts/accuracy/tiers.json").read_text())
    if not str(config.get("split", "")).endswith(f" of {tiers['split']}"):
        raise RuntimeError(f"source run split {config.get('split')!r} is not a tier of {tiers['split']}")
    if Path(tiers["dataset_path"]).resolve() != Path(config["dataset_path"]).resolve()             or tiers["dataset_revision"] != config["dataset_revision"]:
        raise RuntimeError("tiers.json and the source run read different datasets")
    tier = tiers["tiers"][name]
    counts = config["review_counts"]
    if sum(counts[str(u)] for u in tier["train_users"]) != tier["train_reviews"]:
        raise RuntimeError(f"tier {name} review count does not match the source run's review_counts")
    return tier["train_users"][:], tier["train_reviews"]


def prepare(source_run, epoch, out, fraction, shape, evaluate, seed, extra_options, final, fresh_optimizer=False,
            train_tier=None):
    if out.exists():
        raise FileExistsError(f"{out} exists; pick a new output")
    config = json.loads((source_run / "run.json").read_text())
    folder, result = source_case(source_run, config, epoch)
    counts = config["review_counts"]
    if train_tier:
        order, epoch_reviews = tier_users(config, train_tier)
    else:
        order, epoch_reviews = config["train_users"][:], config["training_reviews_per_epoch"]
    random.Random(seed).shuffle(order)
    budget = fraction * epoch_reviews
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
    if fresh_optimizer:
        # The source optimizer state is of another kind (AdamW vs Muon): start the decay from a new, empty state.
        full_run.pop("init_optimizer")
        full_run["optimizer_state"] = "fresh"
    protocol = {"rows": rows_for(users, counts), "evaluation_rows": rows_for(evaluation, counts),
                "dataset_path": str(dataset), "dataset_revision": config["dataset_revision"],
                "training": {"epochs": 1, "chunk_reviews": options.get("chunk_reviews", 128), "gpu_memory_fraction": 0.95},
                "execution_mode": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}, "full_run": full_run}
    decay = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_run": str(source_run), "source_epoch": epoch,
             "source_case": str(folder), "source_update_offset_end": result["update_offset_end"], "model": MODEL,
             "track": config["track"], "options": options, "dropped_width32_options": dropped,
             "fraction_of_epoch_reviews": fraction, "training_users": len(users), "training_reviews": reviews,
             "train_users_from": f"tier {train_tier} (artifacts/accuracy/tiers.json)" if train_tier else "source run.json",
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


def smoke(out):
    """Run the prepared decay `out` in the trainer's smoke mode on a scratch copy; never touches `out`."""
    import shutil
    import time
    scratch = ROOT / "artifacts/small-model/smoke" / f"{out.name}-{os.getpid()}"
    scratch.mkdir(parents=True)
    decay = json.loads((out / "decay.json").read_text())
    distillation = decay["options"].get("distillation")
    if isinstance(distillation, dict):
        decay["options"]["distillation"] = {k: v for k, v in distillation.items() if k != "cache"}
    atomic_json(scratch / "decay.json", decay)
    shutil.copy(out / "protocol.json", scratch / "protocol.json")
    started = time.perf_counter()
    code = run(scratch, smoke=True)
    result_path = scratch / MODEL / decay["track"] / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    print(json.dumps({"smoke": out.name, "exit": code, "status": result.get("status"), "updates": result.get("updates"),
                      "reviews": result.get("review_presentations"), "error": result.get("error"),
                      "seconds": round(time.perf_counter() - started, 1)}), flush=True)
    if code == 0 and result.get("status") == "complete":
        shutil.rmtree(scratch, ignore_errors=True)
        return 0
    return code or 1


def run(out, smoke=False):
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
    args = Namespace(model=MODEL, track=decay["track"], smoke=smoke, expected_source=None,
                     output=out.relative_to(ROOT).as_posix())
    return gpu_r01_training.run_case(args)


def summarize(per_user, users):
    import numpy as np
    from scripts.full_run_metrics import aggregate
    rows = [row for row in per_user if row["user_id"] in users]
    bench = aggregate(rows)
    out = {"users": len(rows)}
    for head, key in (("ahead", "curve"), ("instant", "instant")):
        out[f"benchmark_{key}_logloss_weighted"] = bench.get(head, {}).get("LogLoss", {}).get("weighted")
        out[f"benchmark_{key}_logloss_user_average"] = bench.get(head, {}).get("LogLoss", {}).get("user_average")
        labels = np.array([row[f"{key}_labels"] for row in rows], dtype=float)
        losses = np.array([row[f"{key}_log_loss"] for row in rows], dtype=float)
        out[f"r01_{key}_logloss_label_weighted"] = float((losses * labels).sum() / labels.sum())
    return out


def compare(out):
    decay = json.loads((out / "decay.json").read_text())
    result = json.loads((out / MODEL / decay["track"] / "result.json").read_text())
    if "per_user" not in result:  # trained with --evaluate none: nothing to compare, the checkpoint is the output
        print(f"[decay] {out}: no evaluation in result.json (--evaluate none); skipping compare")
        return 0
    source = json.loads((Path(decay["source_case"]) / "result.json").read_text())
    users = {row["user_id"] for row in result["per_user"]}
    by_user = {row["user_id"]: row for row in source["per_user"]}
    missing = sorted(users - set(by_user))
    if missing:
        raise RuntimeError(f"Source epoch has no scores for {len(missing)} evaluated users")
    summary = {"decayed": summarize(result["per_user"], users), "source_epoch": summarize(source["per_user"], users),
               "reviews_per_second": result.get("reviews_per_second"), "training_seconds": result.get("training_seconds"),
               "updates": result.get("updates"), "training_reviews": result.get("review_presentations"),
               "provenance_commit": (result.get("provenance") or {}).get("commit")}
    deltas = {}
    for key in ("curve", "instant"):
        paired = [(row["benchmark"]["ahead" if key == "curve" else "instant"]["LogLoss"],
                   by_user[row["user_id"]]["benchmark"]["ahead" if key == "curve" else "instant"]["LogLoss"])
                  for row in result["per_user"]
                  if ("ahead" if key == "curve" else "instant") in row.get("benchmark", {})
                  and ("ahead" if key == "curve" else "instant") in by_user[row["user_id"]].get("benchmark", {})]
        import numpy as np
        diff = np.array([a - b for a, b in paired])
        deltas[key] = {"users": len(diff), "mean_user_delta": float(diff.mean()),
                       "standard_error": float(diff.std(ddof=1) / np.sqrt(len(diff))), "users_improved": int((diff < 0).sum())}
    summary["paired_benchmark_logloss_delta_decayed_minus_source"] = deltas
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    d, c = summary["decayed"], summary["source_epoch"]
    lines = [f"# Decay-only from epoch {decay['source_epoch']} ({decay['fraction_of_epoch_reviews']} epoch, {decay['lr']})", "",
             f"{d['users']} held-out users ({decay['evaluation_set']}); control = the source epoch's scores of the same users.", "",
             "| | Benchmark curve LogLoss (weighted / user avg) | Benchmark instant LogLoss (weighted / user avg) | R01 curve / instant (label-weighted) |",
             "|---|---|---|---|"]
    for name, row in (("source epoch", c), ("decayed", d)):
        lines.append(f"| {name} | {row['benchmark_curve_logloss_weighted']:.5f} / {row['benchmark_curve_logloss_user_average']:.5f} | "
                     f"{row['benchmark_instant_logloss_weighted']:.5f} / {row['benchmark_instant_logloss_user_average']:.5f} | "
                     f"{row['r01_curve_logloss_label_weighted']:.5f} / {row['r01_instant_logloss_label_weighted']:.5f} |")
    lines += ["", f"Paired per-user benchmark LogLoss delta (decayed - source): curve {deltas['curve']['mean_user_delta']:+.5f} "
              f"(SE {deltas['curve']['standard_error']:.5f}, {deltas['curve']['users_improved']}/{deltas['curve']['users']} improved); "
              f"instant {deltas['instant']['mean_user_delta']:+.5f} (SE {deltas['instant']['standard_error']:.5f}, "
              f"{deltas['instant']['users_improved']}/{deltas['instant']['users']} improved).",
              f"Training: {summary['training_reviews']:,} reviews in {summary['training_seconds']:.0f} s, "
              f"{summary['reviews_per_second']:,.0f} reviews/s, {summary['updates']} updates."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "auto", "compare", "smoke"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--shape", default="one_minus_sin", choices=("one_minus_sin", "cosine", "linear", "sqrt"))
    parser.add_argument("--final", type=float, default=0.0, help="final learning rate as a fraction of the peak")
    parser.add_argument("--evaluate", default="quick", choices=("quick", "validation", "validation+final_test", "none"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--extra-options", type=json.loads, default=None)
    parser.add_argument("--fresh-optimizer", action="store_true",
                        help="do not load the source epoch's optimizer.pth (needed when --extra-options changes the optimizer kind)")
    parser.add_argument("--train-tier", choices=("screen", "gate", "final"), default=None,
                        help="train on this accuracy-loop tier's users (artifacts/accuracy/tiers.json) instead of the source run's")
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    if not out.is_relative_to(ROOT / "artifacts"):
        parser.error("--out must be under this checkout's artifacts/")
    if args.command == "compare":
        return compare(out)
    if args.command == "smoke":
        return smoke(out)
    if args.command in ("prepare", "auto") and not (args.command == "auto" and out.exists()):
        if args.source_run is None or args.epoch is None or not 0 < args.fraction <= 1:
            parser.error("prepare needs --source-run, --epoch and 0 < --fraction <= 1")
        code = prepare(args.source_run.resolve(), args.epoch, out, args.fraction, args.shape, args.evaluate, args.seed,
                       args.extra_options, args.final, args.fresh_optimizer, args.train_tier)
        if args.command == "prepare" or code:
            return code
    code = run(out)
    return code or compare(out)


if __name__ == "__main__":
    raise SystemExit(main())
