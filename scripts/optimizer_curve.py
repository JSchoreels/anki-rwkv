#!/usr/bin/env python
"""F82 screen: held-out loss against training users, AdamW versus Muon.

One pass over a fixed, ordered list of training users with the accepted R01 trainer (constant learning rate
after a 5-update warmup, so the weights after k users approximate a k-user run). Master weights are saved
every `--every` users; each snapshot is scored afterwards on a fixed set of unseen test users. Every variant
uses the same users, order, seed and batches; only the optimizer differs.

  prepare  --run DIR [--train 500 --test 10 --every 10 --seed 20261013]
  train    --run DIR --variant NAME [--optimizer adamw|muon --lr-scale 1.0 --training-seed 2025]
  evaluate --run DIR --variant NAME
  report   --run DIR
"""
from argparse import ArgumentParser, Namespace
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_full_training import (DATASET, R01_COHORT, REVISION, atomic_json, review_counts, rows_for,
                                       stratified_split)

TRACK = "E89-F65e-F77a"
PLAN = ROOT / "artifacts/retraining/gpu-retrain-r01/plans/iteration-0090-claude.json"


def prepare(run, train, test, every, seed):
    run.mkdir(parents=True, exist_ok=False)
    counts = review_counts()
    reserved = set(json.loads(R01_COHORT.read_text())["independent_final_test_users"])
    others = [user for user in counts if user not in reserved]
    groups = stratified_split(others, counts, {"train": train, "test": test}, seed)
    order = groups["train"][:]
    __import__("random").Random(seed).shuffle(order)
    options = json.loads(PLAN.read_text())["case_options_by_model"]["expertium"][TRACK]
    config = {"dataset_path": DATASET, "dataset_revision": REVISION, "model": "expertium", "track": TRACK,
              "options": options, "split_seed": seed, "snapshot_every_users": every,
              "split": "user-disjoint, stratified by history-length decile, outside the preserved final-test cohort",
              "train_order": order, "test_users": groups["test"],
              "review_counts": {str(user): counts[user] for user in order + groups["test"]},
              "training_reviews": sum(counts[user] for user in order),
              "test_reviews": sum(counts[user] for user in groups["test"]),
              "lr": "peak 7e-4, 5-update linear warmup, then constant (R01 schedule)"}
    atomic_json(run / "run.json", config)
    print(json.dumps({key: config[key] for key in ("training_reviews", "test_reviews", "test_users")}))


def train(run, variant, optimizer, lr_scale, training_seed):
    config = json.loads((run / "run.json").read_text())
    folder = run / variant
    if (folder / "variant.json").exists():
        raise FileExistsError(f"{folder} already trained; pick a new variant name")
    spec = {"optimizer": optimizer, "lr_scale": lr_scale, "training_seed": training_seed}
    protocol = {"rows": rows_for(config["train_order"], config["review_counts"]), "evaluation_rows": [],
                "dataset_path": config["dataset_path"], "dataset_revision": config["dataset_revision"],
                "training": {"epochs": 1, "chunk_reviews": 128, "gpu_memory_fraction": 0.95},
                "execution_mode": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
                "full_run": {"evaluate": False, "snapshot_users": config["snapshot_every_users"],
                             "control_dir": folder.relative_to(ROOT).as_posix()}}
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / "protocol.json", protocol)
    os.environ["GPU_BATCH1_DATA_PATH"] = config["dataset_path"]
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["CUPY_CACHE_DIR"] = str(ROOT / "artifacts/cache/cupy-cache")
    import torch
    from rwkv import train_rwkv
    from scripts import gpu_batch1, gpu_r01_training
    from scripts.run_gpu_r01 import R01SourceWatch
    case = dict(config["options"], training_seed=training_seed)
    torch._C._set_graph_executor_optimize(bool(case.get("jit_optimize", False)))
    torch.use_deterministic_algorithms(not case.get("nondeterministic", False), warn_only=False)
    torch.utils.deterministic.fill_uninitialized_memory = not case.get("skip_uninitialized_fill", False)
    gpu_batch1.TRACKS = {config["track"]: case}
    gpu_r01_training.SourceWatch = R01SourceWatch
    from scripts.gpu_r01_staging import training_batches
    from scripts.gpu_b01 import transfer_batches
    gpu_batch1.training_batches = training_batches
    gpu_batch1.transfer_batches = transfer_batches
    if optimizer == "muon":
        from scripts.muon_optimizer import get_hybrid_optimizer
        # run_case imports get_optimizer from rwkv.train_rwkv when it starts.
        train_rwkv.get_optimizer = lambda namespace, model: get_hybrid_optimizer(model, namespace.PEAK_LR, lr_scale)
    elif optimizer != "adamw" or lr_scale != 1.0:
        raise ValueError("AdamW runs the accepted optimizer unchanged (lr_scale 1)")
    atomic_json(folder / "variant.json", dict(spec, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    args = Namespace(model=config["model"], track=config["track"], smoke=False, expected_source=None,
                     output=folder.relative_to(ROOT).as_posix())
    return gpu_r01_training.run_case(args)


def snapshots(run, variant):
    case = run / variant / "expertium" / TRACK
    found = []
    for meta_path in sorted((case / "snapshots").glob("users-*/snapshot.json")):
        meta = json.loads(meta_path.read_text())
        found.append((meta["progress_users"], meta["review_presentations"], meta["updates"], meta_path.parent / "master.pth"))
    result = json.loads((case / "result.json").read_text())
    if result.get("status") == "complete":
        found.append((result["progress_users"], result["review_presentations"], result["updates"], case / "master.pth"))
    return found, result


def evaluate_variant(run, variant):
    config = json.loads((run / "run.json").read_text())
    os.environ["GPU_BATCH1_DATA_PATH"] = config["dataset_path"]
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    torch.use_deterministic_algorithms(True, warn_only=False)
    from rwkv.data_processing import get_rwkv_data
    from rwkv.prepare_batch import prepare as prepare_batch
    from scripts.benchmark_gpu_trial import benchmark_models
    from scripts.compare import evaluate
    from scripts.gpu_batch1 import DATA, make_samples, split_frame
    device = torch.device("cuda")
    master, _, _ = benchmark_models({"MODEL": "expertium", "DTYPE": "bfloat16", "SEED": 2025}, device)
    rows = rows_for(config["test_users"], config["review_counts"])
    # Held-out inputs prepared once on the host (same seeds as R01 evaluation) and reused for every snapshot.
    inputs = []
    for row in rows:
        user = row["user_id"]
        section = split_frame(get_rwkv_data(DATA, user), row, "validation")
        inputs.append((user, [prepare_batch(make_samples(section.iloc[o:o + 1024], user, 128), seed=2025 + user + o)
                              for o in range(0, len(section), 1024)]))
    found, _ = snapshots(run, variant)
    out_path = run / variant / "curve.json"
    curve = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {point["users"] for point in curve}
    for users, reviews, updates, weights in found:
        if users in done:
            continue
        started = time.perf_counter()
        master.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        per_user = [{"user_id": user, **evaluate(master, ({"batch": b.to(device)} for b in batches))} for user, batches in inputs]
        point = {"users": users, "review_presentations": reviews, "updates": updates,
                 **pooled(per_user), "per_user": per_user, "seconds": round(time.perf_counter() - started, 2)}
        curve = sorted(curve + [point], key=lambda p: p["users"])
        atomic_json(out_path, curve)
        print(f"{variant} users {users:4d}: curve {point['curve_log_loss']:.5f} instant {point['instant_log_loss']:.5f}", flush=True)


def pooled(per_user):
    out = {}
    for metric, labels in (("curve_log_loss", "curve_labels"), ("instant_log_loss", "instant_labels")):
        total = sum(row[labels] for row in per_user)
        out[metric] = sum(row[metric] * row[labels] for row in per_user) / total
        out[metric + "_user_mean"] = sum(row[metric] for row in per_user) / len(per_user)
    return out


def users_to_reach(curve, metric, target):
    """First snapshot whose loss is at or below the target (None if never reached)."""
    for point in curve:
        if point[metric] <= target:
            return point["users"]
    return None


def report(run):
    variants = sorted(path.parent.name for path in run.glob("*/curve.json"))
    curves = {name: json.loads((run / name / "curve.json").read_text()) for name in variants}
    reference = curves.get("adamw")
    lines = ["# F82 optimizer curve", "", f"Run `{run.relative_to(ROOT).as_posix()}`. Label-weighted held-out losses on the test users.", ""]
    summary = {}
    for name, curve in curves.items():
        result = snapshots(run, name)[1]
        spec = json.loads((run / name / "variant.json").read_text())
        rate = result["review_presentations"] / result["training_elapsed_seconds"] if result.get("training_elapsed_seconds") else None
        entry = {**spec, "final_users": curve[-1]["users"], "final_curve": curve[-1]["curve_log_loss"],
                 "final_instant": curve[-1]["instant_log_loss"], "reviews_per_second": rate}
        if reference and name != "adamw":
            for metric in ("curve_log_loss", "instant_log_loss"):
                entry[f"users_to_reach_adamw_final_{metric}"] = users_to_reach(curve, metric, reference[-1][metric])
        summary[name] = entry
    lines += ["| Variant | Optimizer | LR scale | Users | Curve | Instant | Users to AdamW final curve | Users to AdamW final instant | Reviews/s |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, entry in summary.items():
        rate = "—" if entry["reviews_per_second"] is None else f"{entry['reviews_per_second']:,.0f}"
        lines.append(f"| {name} | {entry['optimizer']} | {entry['lr_scale']} | {entry['final_users']} | {entry['final_curve']:.5f} | "
                     f"{entry['final_instant']:.5f} | {entry.get('users_to_reach_adamw_final_curve_log_loss', '—')} | "
                     f"{entry.get('users_to_reach_adamw_final_instant_log_loss', '—')} | {rate} |")
    lines += ["", "| Users | " + " | ".join(f"{name} curve | {name} instant" for name in variants) + " |",
              "|---|" + "---|---|" * len(variants)]
    for point in curves[variants[0]]:
        cells = []
        for name in variants:
            match = next((p for p in curves[name] if p["users"] == point["users"]), None)
            cells += ["—", "—"] if match is None else [f"{match['curve_log_loss']:.5f}", f"{match['instant_log_loss']:.5f}"]
        lines.append(f"| {point['users']} | " + " | ".join(cells) + " |")
    (run / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(run / "summary.json", summary)
    print(json.dumps(summary, indent=1))


def main():
    parser = ArgumentParser()
    parser.add_argument("command", choices=("prepare", "train", "evaluate", "report"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--variant")
    parser.add_argument("--train", type=int, default=500)
    parser.add_argument("--test", type=int, default=10)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--optimizer", choices=("adamw", "muon"), default="adamw")
    parser.add_argument("--lr-scale", type=float, default=1.0)
    parser.add_argument("--training-seed", type=int, default=2025)
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else ROOT / args.run
    if args.command == "prepare":
        prepare(run, args.train, args.test, args.every, args.seed)
    elif args.command == "train":
        sys.exit(train(run, args.variant, args.optimizer, args.lr_scale, args.training_seed))
    elif args.command == "evaluate":
        evaluate_variant(run, args.variant)
    else:
        report(run)


if __name__ == "__main__":
    main()
