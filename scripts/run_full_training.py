#!/usr/bin/env python
"""Full multi-epoch training run on a fixed user split, built on the accepted R01 trainer.

One process per epoch: each epoch starts from the previous epoch's master weights and AdamW
state (checkpoints in <run>/epoch-NN/), so a crash or STOP loses at most the current epoch and
`run` resumes at the first incomplete epoch. Training users are reshuffled each epoch.
Held-out users are never trained on. After each non-final epoch a fixed quick-validation subset
is scored; after the final epoch the full validation and the sealed final-test users are scored
(final test reported separately, never used for decisions). Benchmark LogLoss / RMSE(bins) / AUC
use the upstream label database and `rwkv/get_result.get_stats` (see scripts/full_run_metrics.py).

  prepare  --run DIR [--train 8000 --validation 1500 --test 500 --quick 300 --epochs 5]
  run      --run DIR          (all remaining epochs; one GPU process each; resumes from the latest snapshot)
  epoch    --run DIR --epoch N
  save     --run DIR          (snapshot at the next user boundary; training continues)
  stop     --run DIR          (snapshot at the next user boundary, then stop; `run` resumes from it)
  latest   --run DIR          (path of the newest saved weights, for trying the model)
  pages    --run DIR [--push-every SECONDS]   (pages/data/train-8k.json; optional commit+push loop)
"""
from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATASET = "data/anki-revlogs-10k-original-20261011"
REVISION = "75299740cff05894ef42d7ad990666691efdd2da"
R01_COHORT = ROOT / "artifacts/retraining/gpu-retrain-r01/cohorts-user-holdout-v1.json"
PAGES_JSON = ROOT / "pages/data/train-8k.json"
SLICE_PERCENT = 5
SNAPSHOT_USERS = 400
# A08 (docs/GPU_TRAINING.md, promoted 2026-10-14, settled 2026-10-15: three decay order seeds all beat the epoch-10 control): new runs end with a 0.25-epoch
# cool-down, WSD decay from the last epoch's weights and AdamW state (scripts/expertium_decay.py), scored as the final result.
FINAL_DECAY_EPOCHS = 0.25
FINAL_DECAY_SHAPE = "one_minus_sin"
FINAL_DECAY_DIR = "decay-final"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.1 * (attempt + 1))
    os.replace(tmp, path)


def review_counts():
    import pyarrow.parquet as pq
    base = ROOT / DATASET / "revlogs"
    return {user: pq.ParquetFile(base / f"user_id={user}" / "data.parquet").metadata.num_rows for user in range(1, 10001)}


def stratified_split(users, counts, sizes, seed):
    """Split users into groups of the given sizes, proportionally within history-length deciles."""
    rng = random.Random(seed)
    ordered = sorted(users, key=lambda user: (counts[user], user))
    deciles = [ordered[len(ordered) * d // 10:len(ordered) * (d + 1) // 10] for d in range(10)]
    total = len(users)
    groups = {name: [] for name in sizes}
    # Largest-remainder quotas per decile so every group gets its exact size.
    quotas = {name: [size * len(decile) / total for decile in deciles] for name, size in sizes.items()}
    for name, size in sizes.items():
        floors = [int(q) for q in quotas[name]]
        order = sorted(range(10), key=lambda d: quotas[name][d] - floors[d], reverse=True)
        for d in order[:size - sum(floors)]:
            floors[d] += 1
        quotas[name] = floors
    for d, decile in enumerate(deciles):
        members = decile[:]
        rng.shuffle(members)
        start = 0
        for name in sizes:
            take = quotas[name][d]
            groups[name].extend(members[start:start + take])
            start += take
    for name, size in sizes.items():
        if len(groups[name]) != size:
            raise ValueError(f"Split group {name} has {len(groups[name])} users, expected {size}")
    return {name: sorted(group) for name, group in groups.items()}


def label_job(args):
    toml, user = args
    sys.argv = [sys.argv[0], "--config", toml]
    from rwkv import find_equalize_test_reviews as labels
    labels.process(user)
    return user


def prepare(run, train, validation, test, quick, epochs, seed, track, extra_options=None, final_lr_schedule=None,
            final_decay_epochs=FINAL_DECAY_EPOCHS):
    run.mkdir(parents=True, exist_ok=False)
    counts = review_counts()
    reserved = json.loads(R01_COHORT.read_text())["independent_final_test_users"]
    others = [user for user in counts if user not in set(reserved)]
    extra_test = test - len(reserved)
    if extra_test < 0:
        raise ValueError("Final-test group is smaller than the preserved final-test cohort")
    groups = stratified_split(others, counts, {"train": train, "validation": validation, "test_extra": extra_test}, seed)
    final_test = sorted(reserved + groups["test_extra"])
    quick_users = sorted(random.Random(seed + 1).sample(groups["validation"], quick))
    plan = json.loads((ROOT / "artifacts/retraining/gpu-retrain-r01/plans/iteration-0090-claude.json").read_text())
    options = plan["case_options_by_model"]["expertium"][track]
    dropped = []
    if extra_options or final_lr_schedule:
        # Expertium catch-up switches (docs/EXPERTIUM_CATCH_UP.md); width-32-only kernels go when the width changes.
        from scripts.expertium_options import LearningRate, compatible_options
        options, dropped = compatible_options({**options, **(extra_options or {})})
        if final_lr_schedule:
            LearningRate(final_lr_schedule, 1)
    toml = run / "labels.toml"
    toml.write_text(f'DATA_PATH = "{DATASET}"\nLABEL_FILTER_LMDB_PATH = "{(run / "label-db").relative_to(ROOT).as_posix()}"\n'
                    'LABEL_FILTER_LMDB_SIZE = 10_000_000_000\nALGO = "FSRS-6"\nSHORT = true\nSECS = true\n'
                    'USER_START = 1\nUSER_END = 10000\nPROCESSES = 8\n', encoding="utf-8")
    config = {"created_utc": datetime.now(timezone.utc).isoformat(), "dataset_path": DATASET, "dataset_revision": REVISION,
              "model": "expertium", "track": track, "options": options, "epochs": epochs, "split_seed": seed,
              "slice_percent": SLICE_PERCENT,
              "split": "user-disjoint; stratified by history-length decile over the 9,900 users outside the preserved "
                       "100-user final-test cohort; that cohort plus random others form the sealed final test",
              "train_users": groups["train"], "validation_users": groups["validation"], "quick_validation_users": quick_users,
              "final_test_users": final_test, "preserved_final_test_users": sorted(reserved),
              "review_counts": {str(user): counts[user] for user in counts},
              "training_reviews_per_epoch": sum(counts[user] for user in groups["train"]),
              "evaluation": {"quick": "after every non-final epoch on quick_validation_users",
                             "final": "after the final epoch on validation_users and final_test_users (reported separately)"},
              "lr": "AdamW peak 7e-4, 5-update linear warmup once, then constant (R01 schedule)",
              "final_decay_epochs": final_decay_epochs}
    if final_decay_epochs:
        config["lr"] += f"; then a {final_decay_epochs}-epoch {FINAL_DECAY_SHAPE} decay to 0 (A08 cool-down, {FINAL_DECAY_DIR}/)"
    if extra_options or final_lr_schedule:
        config.update(extra_options=extra_options or {}, dropped_width32_options=dropped,
                      final_lr_schedule=final_lr_schedule)
        if final_lr_schedule:
            config["lr"] += f"; final epoch: {final_lr_schedule}"
    atomic_json(run / "run.json", config)
    started = time.time()
    users = sorted(groups["validation"] + final_test)
    with multiprocessing.Pool(8) as pool:
        for done, _ in enumerate(pool.imap_unordered(label_job, [(str(toml), user) for user in users], chunksize=4), 1):
            if done % 200 == 0:
                print(f"[labels] {done}/{len(users)} users", flush=True)
    print(json.dumps({"run": run.relative_to(ROOT).as_posix(), "train": train, "validation": validation, "final_test": len(final_test),
                      "quick": quick, "training_reviews_per_epoch": config["training_reviews_per_epoch"],
                      "label_seconds": round(time.time() - started)}))


def rows_for(users, counts):
    return [{"user_id": user, "available_reviews": counts[str(user)], "train_range": [0, counts[str(user)]],
             "validation_range": [0, counts[str(user)]]} for user in users]


def epoch_dir(run, epoch):
    return run / f"epoch-{epoch:02d}"


def epoch_result(run, epoch, config):
    path = epoch_dir(run, epoch) / config["model"] / config["track"] / "result.json"
    return json.loads(path.read_text()) if path.exists() else None


def write_protocol(run, epoch, config):
    """Per-epoch frozen inputs: shuffled training order, evaluation users and checkpoint chain."""
    folder = epoch_dir(run, epoch)
    counts = config["review_counts"]
    order = config["train_users"][:]
    random.Random(config["split_seed"] * 1000 + epoch).shuffle(order)
    final = epoch == config["epochs"]
    evaluation = sorted(config["validation_users"] + config["final_test_users"]) if final else config["quick_validation_users"]
    full_run = {"epoch": epoch, "evaluate": True, "benchmark_db": (run / "label-db").relative_to(ROOT).as_posix(),
                "evaluation_set": "validation+final_test" if final else "quick_validation",
                "snapshot_users": config.get("snapshot_users", SNAPSHOT_USERS), "control_dir": run.relative_to(ROOT).as_posix()}
    if config.get("fast_evaluation", True):
        # EV01 (docs/EVAL_SPEED.md): default since 2026-10-14; set "fast_evaluation": false in run.json for the reference evaluation.
        # Models evaluate_fast refuses (EV08) fall back to the reference and record evaluation_mode.
        full_run.update(fast_evaluation=True, fast_evaluation_chunks=int(config.get("fast_evaluation_chunks", 512)))
    if final and config.get("final_lr_schedule"):
        full_run["lr_schedule"] = config["final_lr_schedule"]
    resume_file = run / f"epoch-{epoch:02d}.resume.json"
    if resume_file.exists():
        # Mid-epoch resume: skip users fully trained before the interruption; a user in flight is retrained.
        resume = json.loads(resume_file.read_text())
        done = set(resume["done_users"])
        order = [user for user in order if user not in done]
        full_run.update(init_master=resume["master"], init_optimizer=resume["optimizer"], update_offset=resume["update_offset_end"],
                        resumed={"done_users": len(done), "from": resume["folder"], "attempts": resume["attempts"]})
    elif epoch > 1:
        previous = epoch_result(run, epoch - 1, config)
        if not previous or previous.get("status") != "complete":
            raise RuntimeError(f"Epoch {epoch - 1} is not complete")
        source = epoch_dir(run, epoch - 1) / config["model"] / config["track"]
        full_run.update(init_master=(source / "master.pth").relative_to(ROOT).as_posix(),
                        init_optimizer=(source / "optimizer.pth").relative_to(ROOT).as_posix(),
                        update_offset=previous["update_offset_end"])
    protocol = {"rows": rows_for(order, counts), "evaluation_rows": rows_for(evaluation, counts),
                "dataset_path": config["dataset_path"], "dataset_revision": config["dataset_revision"],
                "training": {"epochs": 1, "chunk_reviews": 128, "gpu_memory_fraction": 0.95},
                "execution_mode": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}, "full_run": full_run}
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / "protocol.json", protocol)
    return folder


def run_epoch(run, epoch):
    """One epoch in this (fresh CUDA) process, set up like scripts/run_gpu_r01.py --case."""
    config = json.loads((run / "run.json").read_text())
    folder = write_protocol(run, epoch, config)
    protocol = json.loads((folder / "protocol.json").read_text())
    os.environ["GPU_BATCH1_DATA_PATH"] = config["dataset_path"]
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = protocol["execution_mode"]["CUBLAS_WORKSPACE_CONFIG"]
    os.environ["CUPY_CACHE_DIR"] = str(ROOT / "artifacts/cache/cupy-cache")
    import torch
    from scripts import gpu_batch1, gpu_r01_training
    from scripts.run_gpu_r01 import R01SourceWatch
    case = config["options"]
    torch._C._set_graph_executor_optimize(bool(case.get("jit_optimize", False)))
    torch.use_deterministic_algorithms(not case.get("nondeterministic", False), warn_only=False)
    torch.utils.deterministic.fill_uninitialized_memory = not case.get("skip_uninitialized_fill", False)
    gpu_batch1.TRACKS = {config["track"]: case}
    gpu_r01_training.SourceWatch = R01SourceWatch
    if case.get("gpu_packed_features"):
        from scripts.gpu_r01_staging import training_batches
        from scripts.gpu_b01 import transfer_batches
        gpu_batch1.training_batches = training_batches
        gpu_batch1.transfer_batches = transfer_batches
    elif case.get("prefetch_users"):
        from scripts.gpu_r01_prefetch import training_batches
        gpu_batch1.training_batches = training_batches
    args = Namespace(model=config["model"], track=config["track"], smoke=False, expected_source=None,
                     output=folder.relative_to(ROOT).as_posix())
    return gpu_r01_training.run_case(args)


def run_all(run):
    config = json.loads((run / "run.json").read_text())
    (run / "STOP").unlink(missing_ok=True)
    (run / "SAVE").unlink(missing_ok=True)
    for epoch in range(1, config["epochs"] + 1):
        result = epoch_result(run, epoch, config)
        if result and result.get("status") == "complete":
            continue
        folder = epoch_dir(run, epoch)
        if folder.exists():
            # Preserve a failed or interrupted attempt; resume from its latest snapshot when it has one,
            # otherwise from the previous resume point or the previous epoch's checkpoint.
            attempt = 1
            while folder.with_name(f"{folder.name}-attempt-{attempt}").exists():
                attempt += 1
            moved = folder.with_name(f"{folder.name}-attempt-{attempt}")
            shutil.move(str(folder), str(moved))
            latest = moved / config["model"] / config["track"] / "snapshots" / "latest.json"
            if latest.exists() and result and result.get("status") != "complete":
                snapshot = json.loads(latest.read_text())
                snapshot_folder = moved / config["model"] / config["track"] / "snapshots" / Path(snapshot["folder"]).name
                resume_file = run / f"epoch-{epoch:02d}.resume.json"
                previous = json.loads(resume_file.read_text()) if resume_file.exists() else {"done_users": [], "attempts": []}
                atomic_json(resume_file, {
                    "done_users": sorted(set(previous["done_users"]) | set(snapshot["done_users"])),
                    "master": (snapshot_folder / "master.pth").relative_to(ROOT).as_posix(),
                    "optimizer": (snapshot_folder / "optimizer.pth").relative_to(ROOT).as_posix(),
                    "update_offset_end": snapshot["update_offset_end"], "folder": snapshot_folder.relative_to(ROOT).as_posix(),
                    "attempts": previous["attempts"] + [moved.name], "exact": False,
                    "note": "resumed mid-epoch: users fully trained before the snapshot are skipped; users in flight are retrained from their start"})
        if (run / "STOP").exists():
            print("[run] STOP file present; not starting a new epoch", flush=True)
            return 1
        print(f"[run] epoch {epoch}/{config['epochs']} starting {datetime.now().strftime('%H:%M:%S')}", flush=True)
        code = subprocess.call([sys.executable, "-X", "utf8", str(Path(__file__)), "epoch", "--run", str(run), "--epoch", str(epoch)])
        result = epoch_result(run, epoch, config)
        if code != 0 or not result or result.get("status") != "complete":
            print(f"[run] epoch {epoch} failed (exit {code}, status {result and result.get('status')}); rerun `run` to resume", flush=True)
            return 1
        print(f"[run] epoch {epoch} complete: {result['reviews_per_second']:,.0f} reviews/s, "
              f"{result['training_seconds'] / 60:.1f} min training", flush=True)
    return decay_stage(run, config)


def decay_stage(run, config):
    """A08 cool-down appended after the last epoch; the main epochs are untouched. Off when run.json has no
    `final_decay_epochs` (runs prepared before 2026-10-14) or it is 0. Its evaluation of validation + final test is the final result;
    the last main epoch keeps its own evaluation as the matched control (`decay-final/summary.md`)."""
    fraction = config.get("final_decay_epochs") or 0
    if not fraction:
        return 0
    out = run / FINAL_DECAY_DIR
    result_path = out / config["model"] / config["track"] / "result.json"
    if result_path.exists() and json.loads(result_path.read_text()).get("status") == "complete" and (out / "summary.md").exists():
        return 0
    if out.exists():
        attempt = 1
        while out.with_name(f"{out.name}-attempt-{attempt}").exists():
            attempt += 1
        shutil.move(str(out), str(out.with_name(f"{out.name}-attempt-{attempt}")))
    if (run / "STOP").exists():
        print("[run] STOP file present; not starting the cool-down", flush=True)
        return 1
    print(f"[run] cool-down {fraction} epoch starting {datetime.now().strftime('%H:%M:%S')}", flush=True)
    code = subprocess.call([sys.executable, "-X", "utf8", str(ROOT / "scripts/expertium_decay.py"), "auto", "--source-run", str(run),
                            "--epoch", str(config["epochs"]), "--out", str(out.relative_to(ROOT)), "--fraction", str(fraction),
                            "--shape", FINAL_DECAY_SHAPE, "--evaluate", "validation+final_test"])
    if code != 0 or not result_path.exists():
        print(f"[run] cool-down failed (exit {code}); rerun `run` to repeat it", flush=True)
        return 1
    return 0


def page_state(run):
    """pages/data/train-8k.json content from the run's records."""
    from scripts.full_run_metrics import aggregate
    config = json.loads((run / "run.json").read_text())
    n = len(config["train_users"])
    epochs, status, rates, epoch_seconds = [], "waiting", [], []
    evaluation_seconds = None
    started = None
    updated = config["created_utc"]
    for epoch in range(1, config["epochs"] + 1):
        result = epoch_result(run, epoch, config)
        row = {"epoch": epoch, "users_done": 0, "finished_utc": None, "evaluation": None}
        if result:
            started = started or result.get("started_utc")
            protocol_path = epoch_dir(run, epoch) / "protocol.json"
            resumed = (json.loads(protocol_path.read_text())["full_run"].get("resumed") or {}).get("done_users", 0)                 if protocol_path.exists() else 0
            done = n if result.get("status") in ("evaluating", "complete") else resumed + result.get("progress_users", 0)
            row["users_done"] = done
            if result.get("training_seconds"):
                rates.append(result["reviews_per_second"])
                epoch_seconds.append(result["training_seconds"])
            if result["status"] in ("evaluating", "complete"):
                users = sorted(config["validation_users"] + config["final_test_users"]) if epoch == config["epochs"] \
                    else config["quick_validation_users"]
                scored = len(result.get("per_user") or []) or result.get("evaluation_progress_users", 0)
                row["evaluation"] = {"users_done": scored, "users_total": len(users), "finished_utc": None}
                if result["status"] == "complete" and result.get("per_user"):
                    validation = set(config["validation_users"])
                    main = aggregate(result["per_user"], validation if epoch == config["epochs"] else None)
                    head = main.get("ahead") or {}
                    row["evaluation"].update(
                        finished_utc=result.get("evaluated_utc") or updated, set=("validation" if epoch == config["epochs"] else "quick validation"),
                        users=main["users"], log_loss=head.get("LogLoss", {}).get("weighted"),
                        rmse_bins=head.get("RMSE(bins)", {}).get("weighted"), auc=head.get("AUC", {}).get("weighted"),
                        instant=main.get("instant"))
                    if result.get("validation_seconds") and epoch == config["epochs"]:
                        evaluation_seconds = result["validation_seconds"]
                    if epoch == config["epochs"]:
                        row["evaluation"]["final_test"] = aggregate(result["per_user"], set(config["final_test_users"]))
            if result.get("status") == "failed":
                status = "failed"
            path = epoch_dir(run, epoch) / config["model"] / config["track"] / "result.json"
            updated = max(updated, datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat())
            if status != "failed":
                status = {"running": "training", "evaluating": "evaluating", "complete": "training"}.get(result["status"], result["status"])
        epochs.append(row)
    if all(r["evaluation"] and r["evaluation"]["finished_utc"] for r in epochs[-1:]):
        status = "finished"
    reviews = config["training_reviews_per_epoch"]
    rate = rates[-1] if rates else 206_800
    full_eval_reviews = sum(config["review_counts"][str(u)] for u in config["validation_users"] + config["final_test_users"])
    return {"mock": False, "model": f"Expertium, accepted R01 configuration 0090 ({config['track']}), BF16",
            "status": status, "training_users": n, "evaluation_users": len(config["validation_users"]) + len(config["final_test_users"]),
            "validation_users": len(config["validation_users"]), "final_test_users": len(config["final_test_users"]),
            "quick_validation_users": len(config["quick_validation_users"]),
            "slice_percent": config["slice_percent"], "epochs_planned": config["epochs"],
            "evaluate_after_epochs": list(range(1, config["epochs"] + 1)),
            "training_reviews_per_epoch": reviews, "reviews_per_second": rate,
            "estimates": {"epoch_seconds": (sum(epoch_seconds) / len(epoch_seconds)) if epoch_seconds else reviews / rate,
                          "evaluation_seconds": evaluation_seconds or full_eval_reviews / 32_500,
                          "quick_evaluation_seconds": sum(config["review_counts"][str(u)] for u in config["quick_validation_users"]) / 32_500},
            "started_utc": started or config["created_utc"], "updated_utc": updated,
            "checkpoints": "after every epoch (master weights and AdamW state)",
            "metric_note": "Benchmark test reviews (FSRS benchmark selection and RMSE bins), curve head, review-weighted. "
                           "Rows 1-4: quick validation subset; last row: full validation set (final-test users reported separately).",
            "epochs": epochs}


def latest_weights(run):
    """Newest saved master weights: an epoch-end checkpoint or a mid-epoch snapshot."""
    candidates = list(run.glob("epoch-*/*/*/master.pth")) + list(run.glob("epoch-*/*/*/snapshots/users-*/master.pth"))
    if not candidates:
        print("no saved weights yet")
        return 1
    newest = max(candidates, key=lambda path: path.stat().st_mtime)
    print(newest.relative_to(ROOT).as_posix())
    return 0


def pages(run, push_every):
    while True:
        state = page_state(run)
        atomic_json(PAGES_JSON, state)
        print(f"[pages] {state['status']} {datetime.now().strftime('%H:%M:%S')}", flush=True)
        if push_every:
            message = f"Pages: 8k training progress ({state['status']})\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
            subprocess.call([sys.executable, str(ROOT / "scripts/r01_claim.py"), "run", "--claim", "git", "--owner", "full-run-pages",
                             "--purpose", "commit and push training progress page", "--", "sh", "-c",
                             "git add pages/data/train-8k.json && (git diff --cached --quiet -- pages/data/train-8k.json || "
                             "git -c user.name='Jonathan Schoreels' -c user.email=jonathan.schoreels@jisko.be "
                             f"commit -q -m \"{message}\" -- pages/data/train-8k.json) && git push -q origin main"], cwd=ROOT)
        if not push_every or state["status"] in ("finished", "failed"):
            return 0
        time.sleep(push_every)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "epoch", "pages", "save", "stop", "latest"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--train", type=int, default=8000)
    parser.add_argument("--validation", type=int, default=1500)
    parser.add_argument("--test", type=int, default=500)
    parser.add_argument("--quick", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--track", default="E89-F65e-F77a")
    parser.add_argument("--push-every", type=int, default=0)
    parser.add_argument("--extra-options", type=json.loads, default=None,
                        help="prepare: Expertium catch-up track options as JSON (docs/EXPERTIUM_CATCH_UP.md)")
    parser.add_argument("--final-lr-schedule", type=json.loads, default=None,
                        help='prepare: final-epoch decay, e.g. {"decay_start": 0.75, "shape": "one_minus_sin"}')
    parser.add_argument("--final-decay-epochs", type=float, default=FINAL_DECAY_EPOCHS,
                        help="prepare: A08 cool-down after the last epoch, in epochs of training reviews (default 0.25; 0 = off)")
    args = parser.parse_args()
    if not 0 <= args.final_decay_epochs <= 1:
        parser.error("--final-decay-epochs must be in [0, 1]")
    run = (ROOT / args.run).resolve()
    if not run.is_relative_to(ROOT / "artifacts"):
        parser.error("Run directory must be under artifacts/")
    if args.command == "prepare":
        return prepare(run, args.train, args.validation, args.test, args.quick, args.epochs, args.seed, args.track,
                       args.extra_options, args.final_lr_schedule, args.final_decay_epochs) or 0
    if args.command == "run":
        return run_all(run)
    if args.command == "epoch":
        return run_epoch(run, args.epoch)
    if args.command in ("save", "stop"):
        (run / args.command.upper()).touch()
        print(f"{args.command.upper()} requested; it takes effect at the next user boundary (seconds)")
        return 0
    if args.command == "latest":
        return latest_weights(run)
    return pages(run, args.push_every)


if __name__ == "__main__":
    raise SystemExit(main())
