#!/usr/bin/env python
"""Accuracy-exploration loop harness (docs/ACCURACY_LOOP.md): tiered, paired, seeded experiments.

Tiers are fixed user subsets of an existing user-disjoint full-run split (run.json: train_users,
validation_users, quick_validation_users, final_test_users). The sealed final-test users are never
evaluated outside the `final` tier. Training goes through scripts/run_full_training.py (one process
per epoch, accepted R01 trainer); evaluation is a separate, resumable job (scripts/accuracy_eval.py),
so every GPU job stays short enough for the shared queue. Every experiment writes one result record
(artifacts/accuracy/experiments/<name>/result.json) plus one line in artifacts/accuracy/results.jsonl.

  tiers      [--split RUN_JSON]                     build artifacts/accuracy/tiers.json (CPU, once)
  prepare    NAME --tier screen --seed S [--order-seed O] [--options JSON] [--env JSON] [--epochs N]
             [--init-master P [--init-optimizer P --update-offset N]] [--checkpoint P] [--evaluator MOD:FN]
             [--checkpoint-transform MOD:FN] [--eval-users N] [--note TEXT]
  train      NAME [--epochs-per-job K]               GPU (queue): remaining epochs (K per job; exit 3 = more to do)
  evaluate   NAME [NAME ...] [--max-minutes M]       GPU (queue): resumable held-out scoring; several
                                                     experiments with one env share one pass (paired inputs)
  collection NAME [--threads 2]                      CPU: user's own collection, full layout, vs the fork
  report     NAME                                    result.json + results.jsonl line + Markdown log row
  compare    CONTROL CANDIDATE                       paired per-user deltas with user-bootstrap intervals
  noise      NAME NAME [...]                         seed-noise band from identical configs, different seeds
  check-eval NAME REFERENCE_RESULT_JSON             per-user agreement with a recorded evaluation
  status

Hooks (per experiment, pass-through; see docs/ACCURACY_LOOP.md#hooks): "options" are merged into the R01
track options the trainer reads; "env" is set for every train/eval/collection process; "model"/"track"
name the trainer model; "init" starts epoch 1 from given weights (e.g. LR decay from a checkpoint);
"evaluator" replaces the default per-user scorer; "checkpoint_transform" maps the trained master.pth to the
weights to score (e.g. quantization); "epoch_runner" replaces the epoch entry point.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MAIN = Path(os.environ.get("ACCURACY_MAIN_CHECKOUT", r"C:\Users\jonat\workspace\anki-rwkv"))
DEFAULT_SPLIT = MAIN / "artifacts/full-run/run-8k-20261013/run.json"
DEFAULT_LABEL_DB = MAIN / "artifacts/full-run/run-8k-20261013/label-db"
BASE = ROOT / "artifacts/accuracy"
TIERS = BASE / "tiers.json"
RESULTS = BASE / "results.jsonl"
FORK_FULL = {"curve": 0.4417, "instant": 0.4366}  # fork RWKV 2.76M, full layout, 173,690 collection test reviews
FORK_SEEN_FROM = 5000  # the fork's model was trained on dataset users 5000-10000; >= this id counts as seen
TIER_SEED = 20261014
EVAL_MODES = {"fast": "fast (EV01, c6530b9)", "reference": "reference (full-run in-training path, bitwise)"}
EVAL_REVIEWS_PER_SECOND = 25_000  # full-run evaluation rate (epoch 7: 22.2M reviews in 888 s); for ETAs only


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def exp_dir(name):
    return BASE / "experiments" / name


def load_hook(spec):
    module, _, function = spec.partition(":")
    return getattr(importlib.import_module(module), function)


# ---------------------------------------------------------------- tiers

def build_tiers(split_path):
    from scripts.run_full_training import stratified_split
    split = load(split_path)
    counts = {int(k): v for k, v in split["review_counts"].items()}
    train, validation, quick = split["train_users"], split["validation_users"], split["quick_validation_users"]
    # Nested training tiers, each stratified by history-length decile: screen (2k) inside gate (5k) inside final (all).
    parts = stratified_split(train, counts, {"screen": 2000, "gate_extra": 3000, "rest": len(train) - 5000}, TIER_SEED)
    # Screen evaluation: the full run's 300 quick-validation users (per-epoch history exists for them) plus 700
    # stratified others from the validation set. Gate and final evaluate on every validation user.
    others = [u for u in validation if u not in set(quick)]
    extra = stratified_split(others, counts, {"extra": 700, "rest": len(others) - 700}, TIER_SEED + 1)["extra"]
    screen_eval = sorted(quick + extra)
    tiers = {
        "created_utc": now(), "split": str(split_path), "split_sha256": hashlib.sha256(Path(split_path).read_bytes()).hexdigest(),
        "dataset_path": str((MAIN / split["dataset_path"]).resolve()), "dataset_revision": split["dataset_revision"],
        "tier_seed": TIER_SEED, "fork_seen_from_user": FORK_SEEN_FROM,
        "rule": "training tiers nested and stratified by history-length decile within train_users; screen evaluation = "
                "quick_validation_users + 700 stratified other validation users; gate/final evaluation = all validation_users; "
                "final_test_users are sealed (final tier only)",
        "tiers": {
            "screen": {"train_users": sorted(parts["screen"]), "eval_users": screen_eval, "epochs": 1},
            "gate": {"train_users": sorted(parts["screen"] + parts["gate_extra"]), "eval_users": sorted(validation), "epochs": 1},
            "final": {"train_users": sorted(train), "eval_users": sorted(validation), "epochs": 10,
                      "final_test_users": sorted(split["final_test_users"])},
        },
    }
    for name, tier in tiers["tiers"].items():
        tier["train_reviews"] = sum(counts[u] for u in tier["train_users"])
        tier["eval_reviews"] = sum(counts[u] for u in tier["eval_users"])
        tier["eval_users_fork_unseen"] = sum(1 for u in tier["eval_users"] if u < FORK_SEEN_FROM)
    if set(tiers["tiers"]["gate"]["eval_users"]) & set(split["final_test_users"]):
        raise AssertionError("evaluation tiers must not contain sealed final-test users")
    atomic_json(TIERS, tiers)
    for name, tier in tiers["tiers"].items():
        print(f"{name}: train {len(tier['train_users'])} users / {tier['train_reviews']:,} reviews per epoch; "
              f"eval {len(tier['eval_users'])} users ({tier['eval_users_fork_unseen']} with id<{FORK_SEEN_FROM}) / "
              f"{tier['eval_reviews']:,} reviews (~{tier['eval_reviews'] / EVAL_REVIEWS_PER_SECOND / 60:.0f} min at "
              f"{EVAL_REVIEWS_PER_SECOND:,}/s)")
    return 0


# ---------------------------------------------------------------- prepare

def ensure_runtime_caches():
    """Copy the native-extension build and CuPy kernel cache from the main checkout when this worktree lacks them."""
    from rwkv.model.native_launch import build_identity
    key = build_identity()[1]
    target = ROOT / "artifacts/cache/r01-native" / key
    source = MAIN / "artifacts/cache/r01-native" / key
    if not (target / "build.json").exists() and (source / "build.json").exists() and source != target:
        shutil.copytree(source, target)
        print(f"[prepare] copied native build {key[:12]} from the main checkout")
    cupy = ROOT / "artifacts/cache/cupy-cache"
    if not cupy.exists() and (MAIN / "artifacts/cache/cupy-cache").exists() and MAIN != ROOT:
        shutil.copytree(MAIN / "artifacts/cache/cupy-cache", cupy)
        print("[prepare] copied the CuPy kernel cache from the main checkout")


def prepare(args):
    folder = exp_dir(args.name)
    if folder.exists():
        raise SystemExit(f"experiment {args.name} already exists: {folder}")
    tiers = load(TIERS)
    tier = tiers["tiers"][args.tier]
    split = load(tiers["split"])
    if split.get("review_counts") is None:
        raise SystemExit("split has no review_counts")
    options = dict(split["options"])
    overrides = json.loads(args.options) if args.options else {}
    options.update(overrides)
    if args.seed is not None:
        options["training_seed"] = args.seed
    seed = options.get("training_seed", 2025)
    eval_users = tier["eval_users"]
    if args.eval_user_ids:
        chosen = [int(u) for u in args.eval_user_ids.split(",")]
        if not set(chosen) <= set(eval_users):
            raise SystemExit("--eval-user-ids must be a subset of the tier's evaluation users")
        eval_users = sorted(chosen)
    elif args.eval_users and args.eval_users < len(eval_users):
        # Deterministic, history-stratified subset of the tier's evaluation users (documented as a deviation).
        from scripts.run_full_training import stratified_split
        counts = {int(k): v for k, v in split["review_counts"].items()}
        eval_users = sorted(stratified_split(eval_users, counts, {"keep": args.eval_users, "rest": len(eval_users) - args.eval_users},
                                             TIER_SEED + 2)["keep"])
    kind = "score" if args.checkpoint else "train"
    experiment = {
        "name": args.name, "created_utc": now(), "tier": args.tier, "kind": kind, "seed": seed, "note": args.note,
        "order_seed": args.order_seed if args.order_seed is not None else split["split_seed"],
        "model": args.model or split["model"], "track": args.track or split["track"],
        "epochs": 0 if kind == "score" else (args.epochs or tier["epochs"]),
        "options": options, "option_overrides": overrides, "env": json.loads(args.env) if args.env else {},
        "init": ({"master": str(Path(args.init_master).resolve()),
                  "optimizer": str(Path(args.init_optimizer).resolve()) if args.init_optimizer else None,
                  "update_offset": args.update_offset} if args.init_master else None),
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "evaluator": args.evaluator, "checkpoint_transform": args.checkpoint_transform, "epoch_runner": args.epoch_runner,
        "train_users": len(tier["train_users"]) if kind == "train" else 0,
        "eval_users": eval_users, "eval_users_count": len(eval_users),
        "eval_users_note": None if len(eval_users) == len(tier["eval_users"]) else
        f"stratified subset of {len(tier['eval_users'])} tier evaluation users",
        "label_db": str(DEFAULT_LABEL_DB), "split": tiers["split"], "split_sha256": tiers["split_sha256"],
        "dataset_path": tiers["dataset_path"], "dataset_revision": tiers["dataset_revision"],
    }
    if args.tier == "final":
        experiment["final_test_users"] = tier["final_test_users"]
    folder.mkdir(parents=True)
    if kind == "train":
        # A scripts/run_full_training.py run folder. final_test_users stays empty outside the final tier, and the
        # in-training evaluation is switched off (accuracy_loop evaluates separately), so sealed users are never read.
        run = {"created_utc": now(), "dataset_path": tiers["dataset_path"], "dataset_revision": tiers["dataset_revision"],
               "model": experiment["model"], "track": experiment["track"], "options": options, "epochs": experiment["epochs"],
               # The per-epoch training-user order is shuffled with split_seed * 1000 + epoch (run_full_training).
               "split_seed": args.order_seed if args.order_seed is not None else split["split_seed"],
               "slice_percent": split.get("slice_percent"),
               "split": f"accuracy-loop tier {args.tier} of {tiers['split']}",
               "train_users": tier["train_users"], "validation_users": eval_users, "quick_validation_users": eval_users,
               "final_test_users": [], "review_counts": split["review_counts"],
               "training_reviews_per_epoch": tier["train_reviews"], "snapshot_users": 400,
               "lr": "R01 schedule (AdamW peak 7e-4, 5-update warmup, constant) unless an option hook changes it"}
        atomic_json(folder / "run" / "run.json", run)
    atomic_json(folder / "experiment.json", experiment)
    ensure_runtime_caches()
    print(json.dumps({"experiment": args.name, "kind": kind, "tier": args.tier, "seed": seed, "epochs": experiment["epochs"],
                      "train_users": experiment["train_users"], "eval_users": len(eval_users),
                      "option_overrides": overrides, "env": experiment["env"]}))
    return 0


# ---------------------------------------------------------------- train

def child_env(experiment, threads=None):
    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in experiment.get("env", {}).items()})
    env["PYTHONUTF8"] = "1"
    if threads:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = str(threads)
    return env


def epoch_result(experiment, epoch):
    path = exp_dir(experiment["name"]) / "run" / f"epoch-{epoch:02d}" / experiment["model"] / experiment["track"] / "result.json"
    return load(path) if path.exists() else None


def train(args):
    experiment = load(exp_dir(args.name) / "experiment.json")
    if experiment["kind"] != "train":
        raise SystemExit("score-only experiment: nothing to train")
    run = exp_dir(args.name) / "run"
    phases = exp_dir(args.name) / "phases.jsonl"
    ran = 0
    for epoch in range(1, experiment["epochs"] + 1):
        result = epoch_result(experiment, epoch)
        if result and result.get("status") == "complete":
            continue
        if args.epochs_per_job and ran >= args.epochs_per_job:
            print(f"[train] stopping after {ran} epoch(s) this job; rerun to continue at epoch {epoch}", flush=True)
            return 3
        ran += 1
        folder = run / f"epoch-{epoch:02d}"
        if folder.exists():
            attempt = 1
            while folder.with_name(f"{folder.name}-attempt-{attempt}").exists():
                attempt += 1
            shutil.move(str(folder), str(folder.with_name(f"{folder.name}-attempt-{attempt}")))
        started, started_utc = time.perf_counter(), now()
        print(f"[train] {args.name} epoch {epoch}/{experiment['epochs']} starting {datetime.now():%H:%M:%S}", flush=True)
        code = subprocess.call([sys.executable, "-X", "utf8", str(Path(__file__)), "_epoch", args.name, "--epoch", str(epoch)],
                               cwd=ROOT, env=child_env(experiment))
        wall = time.perf_counter() - started
        result = epoch_result(experiment, epoch) or {}
        record = {"phase": "train", "epoch": epoch, "started_utc": started_utc, "wall_seconds": round(wall, 1), "exit": code,
                  "status": result.get("status"), "training_seconds": result.get("training_seconds"),
                  "reviews_per_second": result.get("reviews_per_second"), "review_presentations": result.get("review_presentations"),
                  "updates": result.get("updates"), "case_elapsed_seconds": result.get("case_elapsed_seconds")}
        with phases.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if code != 0 or result.get("status") != "complete":
            print(f"[train] epoch {epoch} failed (exit {code}, status {result.get('status')}): {result.get('error')}", flush=True)
            return 1
        print(f"[train] epoch {epoch} complete: {result['reviews_per_second']:,.0f} reviews/s, "
              f"{result['training_seconds'] / 60:.1f} min training, {wall / 60:.1f} min process wall", flush=True)
    return 0


def run_epoch(args):
    """Child process: one epoch through scripts/run_full_training.py with in-training evaluation off and hooks applied."""
    experiment = load(exp_dir(args.name) / "experiment.json")
    run = exp_dir(args.name) / "run"
    from scripts import run_full_training as rft
    original = rft.write_protocol

    def write_protocol(run_dir, epoch, config):
        folder = original(run_dir, epoch, config)
        protocol = load(folder / "protocol.json")
        full_run = protocol["full_run"]
        full_run.update(evaluate=False, evaluation_set="none (scripts/accuracy_loop.py evaluates separately)")
        protocol["evaluation_rows"] = []
        init = experiment.get("init")
        if epoch == 1 and init:
            full_run["init_master"] = init["master"]
            if init.get("optimizer"):
                full_run["init_optimizer"] = init["optimizer"]
            full_run["update_offset"] = int(init.get("update_offset") or 0)
        protocol["accuracy_loop"] = {"experiment": experiment["name"], "tier": experiment["tier"], "seed": experiment["seed"],
                                     "option_overrides": experiment["option_overrides"], "env": experiment["env"]}
        atomic_json(folder / "protocol.json", protocol)
        return folder

    rft.write_protocol = write_protocol
    if experiment.get("epoch_runner"):
        return load_hook(experiment["epoch_runner"])(run, args.epoch, experiment, rft)
    return rft.run_epoch(run, args.epoch)


# ---------------------------------------------------------------- evaluate

def checkpoint_for(experiment):
    if experiment["kind"] == "score":
        path = Path(experiment["checkpoint"])
    else:
        last = experiment["epochs"]
        result = epoch_result(experiment, last)
        if not result or result.get("status") != "complete":
            raise SystemExit(f"{experiment['name']}: epoch {last} is not complete")
        path = exp_dir(experiment["name"]) / "run" / f"epoch-{last:02d}" / experiment["model"] / experiment["track"] / "master.pth"
    if experiment.get("checkpoint_transform"):
        out = exp_dir(experiment["name"]) / "transformed"
        out.mkdir(exist_ok=True)
        path = Path(load_hook(experiment["checkpoint_transform"])(path, out, experiment))
    return path


def evaluate(args):
    experiments = [load(exp_dir(name) / "experiment.json") for name in args.names]
    first = experiments[0]
    for other in experiments[1:]:
        for key in ("eval_users", "env", "model", "evaluator", "dataset_path", "label_db"):
            if other.get(key) != first.get(key):
                raise SystemExit(f"shared evaluation needs identical {key}: {first['name']} vs {other['name']}")
    split = load(first["split"])
    counts = split["review_counts"]
    users = first["eval_users"]
    spec = {"checkpoints": {e["name"]: str(checkpoint_for(e)) for e in experiments},
            "outputs": {e["name"]: str(exp_dir(e["name"]) / "eval" / "per_user.jsonl") for e in experiments},
            "users": [{"user_id": u, "available_reviews": counts[str(u)]} for u in users],
            "dataset_path": first["dataset_path"], "label_db": first["label_db"], "model": first["model"],
            "evaluator": first.get("evaluator"), "max_minutes": args.max_minutes, "workers": args.workers,
            "mode": "reference" if first.get("evaluator") else args.mode, "batch_chunks": args.batch_chunks,
            "options": first["options"]}
    spec_path = exp_dir(first["name"]) / "eval" / f"spec-{datetime.now():%Y%m%d-%H%M%S}.json"
    atomic_json(spec_path, spec)
    remaining = sum(counts[str(u)] for u in users)
    print(f"[evaluate] {', '.join(args.names)}: {len(users)} users, {remaining:,} reviews, mode {spec['mode']}", flush=True)
    started, started_utc = time.perf_counter(), now()
    code = subprocess.call([sys.executable, "-X", "utf8", str(ROOT / "scripts/accuracy_eval.py"), str(spec_path)],
                           cwd=ROOT, env=child_env(first))
    wall = time.perf_counter() - started
    for e in experiments:
        done = len(read_jsonl(exp_dir(e["name"]) / "eval" / "per_user.jsonl"))
        with (exp_dir(e["name"]) / "phases.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"phase": "evaluate", "started_utc": started_utc, "wall_seconds": round(wall, 1), "exit": code,
                                "shared_with": [n for n in args.names if n != e["name"]], "users_done": done,
                                "eval_mode": EVAL_MODES[spec["mode"]] if not first.get("evaluator") else f"hook {first['evaluator']}",
                                "users_total": len(users), "spec": spec_path.relative_to(ROOT).as_posix()}) + "\n")
    print(f"[evaluate] exit {code} after {wall / 60:.1f} min" + (" (time budget reached; rerun to resume)" if code == 3 else ""),
          flush=True)
    return code


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["user_id"]] = row
    return list(rows.values())


# ---------------------------------------------------------------- collection

def collection(args):
    experiment = load(exp_dir(args.name) / "experiment.json")
    path = checkpoint_for(experiment)
    out = exp_dir(args.name) / "collection.json"
    if out.exists() and load(out).get("checkpoint") == str(Path(path).resolve()) and not args.force:
        print(f"[collection] {args.name}: already scored ({out.relative_to(ROOT).as_posix()}); --force to rescore")
        return 0
    started, started_utc = time.perf_counter(), now()
    code = subprocess.call([sys.executable, "-X", "utf8", str(ROOT / "scripts/accuracy_collection.py"),
                            "--checkpoint", str(path), "--out", str(out), "--threads", str(args.threads),
                            "--label", args.name], cwd=ROOT, env=child_env(experiment, threads=args.threads))
    with (exp_dir(args.name) / "phases.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"phase": "collection", "started_utc": started_utc,
                            "wall_seconds": round(time.perf_counter() - started, 1), "exit": code}) + "\n")
    return code


# ---------------------------------------------------------------- metrics and records

def summarize(per_user, subset=None):
    """Benchmark LogLoss/RMSE(bins)/AUC (review-weighted and user-averaged; scripts/full_run_metrics.aggregate)
    and the R01 all-label losses (label-weighted and user-averaged)."""
    from scripts.full_run_metrics import aggregate
    rows = [r for r in per_user if subset is None or subset(r["user_id"])]
    if not rows:
        return {"users": 0}
    out = aggregate(rows)
    for metric, count in (("curve_log_loss", "curve_labels"), ("instant_log_loss", "instant_labels")):
        weights = [r[count] for r in rows]
        out.setdefault("r01_all_labels", {})[metric] = {
            "label_weighted": sum(r[metric] * w for r, w in zip(rows, weights)) / sum(weights),
            "user_average": sum(r[metric] for r in rows) / len(rows), "labels": int(sum(weights))}
    return out


SUBSETS = {"all": None, "fork_unseen": lambda u: u < FORK_SEEN_FROM, "fork_seen": lambda u: u >= FORK_SEEN_FROM}


def model_facts(experiment):
    """Parameter count and per-card recurrent state bytes (float32 formula) of the experiment's architecture."""
    import importlib as il
    os.environ.update({str(k): str(v) for k, v in experiment.get("env", {}).items()})
    try:
        architecture = il.import_module("expertium.architecture")
        config = architecture.DEFAULT_ANKI_RWKV_CONFIG
        card = dict(config.modules)["card_id"]
        head = card.d_model // card.n_heads
        per_layer = card.n_heads * head * head + 2 * card.d_model  # WKV matrix state + time-mix and channel-mix shifts
        facts = {"card_state_bytes_float32": card.n_layers * per_layer * 4, "card_layers": card.n_layers,
                 "d_model": card.d_model, "state_formula": "card layers x (heads x (d/heads)^2 + 2d) x 4 B"}
    except Exception as exc:  # noqa: BLE001 - an architecture hook may not match this formula
        facts = {"card_state_bytes_float32": None, "state_error": f"{type(exc).__name__}: {exc}"}
    try:
        import torch
        path = checkpoint_for(experiment)
        state = torch.load(path, map_location="cpu", weights_only=True)
        facts["params"] = int(sum(v.numel() for v in state.values() if hasattr(v, "numel") and v.is_floating_point()))
        facts["checkpoint"] = str(path)
        facts["checkpoint_bytes"] = Path(path).stat().st_size
    except SystemExit:
        pass
    return facts


def report(args):
    folder = exp_dir(args.name)
    experiment = load(folder / "experiment.json")
    phases = read_phases(folder / "phases.jsonl")
    per_user = read_jsonl(folder / "eval" / "per_user.jsonl")
    expected = set(experiment["eval_users"])
    done = {r["user_id"] for r in per_user}
    record = {"name": args.name, "reported_utc": now(), "tier": experiment["tier"], "kind": experiment["kind"],
              "seed": experiment["seed"], "order_seed": experiment.get("order_seed"),
              "model": experiment["model"], "track": experiment["track"],
              "epochs": experiment["epochs"], "train_users": experiment["train_users"],
              "eval_users": len(expected), "eval_users_done": len(done & expected),
              "option_overrides": experiment["option_overrides"], "env": experiment["env"], "init": experiment["init"],
              "checkpoint_transform": experiment.get("checkpoint_transform"), "evaluator": experiment.get("evaluator"),
              "note": experiment.get("note")}
    trains = [p for p in phases if p["phase"] == "train" and p.get("status") == "complete"]
    if experiment["kind"] == "train":
        results = [epoch_result(experiment, e) or {} for e in range(1, experiment["epochs"] + 1)]
        record["training"] = {
            "epochs_complete": sum(1 for r in results if r.get("status") == "complete"),
            "review_presentations": sum(r.get("review_presentations") or 0 for r in results),
            "training_seconds": sum(r.get("training_seconds") or 0 for r in results),
            "reviews_per_second_by_epoch": [r.get("reviews_per_second") for r in results],
            "updates": sum(r.get("updates") or 0 for r in results),
            "process_wall_seconds": sum(p["wall_seconds"] for p in trains)}
        t = record["training"]
        t["reviews_per_second"] = t["review_presentations"] / t["training_seconds"] if t["training_seconds"] else None
        record["provenance"] = (results[0] or {}).get("provenance")
    evals = [p for p in phases if p["phase"] == "evaluate"]
    record["evaluation_wall_seconds"] = sum(p["wall_seconds"] for p in evals)
    record["eval_modes"] = sorted({r.get("eval_mode") or EVAL_MODES["reference"] for r in per_user})
    record["evaluation_jobs"] = len(evals)
    if per_user:
        record["metrics"] = {name: summarize([r for r in per_user if r["user_id"] in expected], subset)
                             for name, subset in SUBSETS.items()}
        record["evaluation_reviews"] = sum(r.get("reviews", 0) for r in per_user)
        record["evaluation_model_seconds"] = sum(r.get("seconds", 0) for r in per_user)
    if (folder / "collection.json").exists():
        coll = load(folder / "collection.json")
        record["collection_full"] = {head: {k: coll[head][k] for k in ("LogLoss", "RMSE(bins)", "AUC")} for head in ("curve", "instant")}
        record["collection_full"]["delta_pct_vs_fork"] = {
            head: 100 * (coll[head]["LogLoss"] / FORK_FULL[head] - 1) for head in ("curve", "instant")}
        record["collection_wall_seconds"] = sum(p["wall_seconds"] for p in phases if p["phase"] == "collection")
    record["model_facts"] = model_facts(experiment)
    record["status"] = "complete" if done >= expected else f"partial ({len(done & expected)}/{len(expected)} users evaluated)"
    atomic_json(folder / "result.json", record)
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps({k: record.get(k) for k in ("name", "reported_utc", "status", "tier", "kind", "seed", "epochs",
                                                        "train_users", "eval_users_done", "option_overrides", "env")}
                           | {"metrics": compact(record), "provenance_commit": (record.get("provenance") or {}).get("git_commit")}) + "\n")
    print(markdown_row(record))
    return 0


def read_phases(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def compact(record):
    out = {}
    for subset, values in (record.get("metrics") or {}).items():
        for head, label in (("ahead", "curve"), ("instant", "instant")):
            for metric in ("LogLoss", "RMSE(bins)", "AUC"):
                value = (values.get(head) or {}).get(metric)
                if value:
                    out[f"{subset}.{label}.{metric}"] = round(value["weighted"], 5)
                    out[f"{subset}.{label}.{metric}.user"] = round(value["user_average"], 5)
    if record.get("training"):
        out["reviews_per_second"] = round(record["training"]["reviews_per_second"] or 0)
    if record.get("collection_full"):
        out["collection.curve"] = record["collection_full"]["curve"]["LogLoss"]
        out["collection.instant"] = record["collection_full"]["instant"]["LogLoss"]
    return out


def fmt(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def markdown_row(record):
    m = (record.get("metrics") or {}).get("all") or {}
    u = (record.get("metrics") or {}).get("fork_unseen") or {}
    get = lambda block, head, metric: ((block.get(head) or {}).get(metric) or {}).get("weighted")  # noqa: E731
    t = record.get("training") or {}
    c = record.get("collection_full") or {}
    facts = record.get("model_facts") or {}
    cells = [record["reported_utc"][:16].replace("T", " "), f"`{record['name']}`", record["tier"], str(record["seed"]),
             str(record["train_users"]), f"{record['eval_users_done']}/{record['eval_users']}",
             fmt(get(m, "ahead", "LogLoss")), fmt(get(m, "instant", "LogLoss")),
             fmt(get(m, "ahead", "RMSE(bins)")), fmt(get(m, "instant", "RMSE(bins)")),
             fmt(get(m, "ahead", "AUC")), fmt(get(m, "instant", "AUC")),
             fmt(get(u, "ahead", "LogLoss")), fmt(get(u, "instant", "LogLoss")),
             f"{t['reviews_per_second']:,.0f}" if t.get("reviews_per_second") else "—",
             str(facts.get("params") or "—"), str(facts.get("card_state_bytes_float32") or "—"),
             f"{fmt((c.get('curve') or {}).get('LogLoss'))} ({c['delta_pct_vs_fork']['curve']:+.2f}%)" if c else "—",
             f"{fmt((c.get('instant') or {}).get('LogLoss'))} ({c['delta_pct_vs_fork']['instant']:+.2f}%)" if c else "—",
             f"{(t.get('process_wall_seconds') or 0) / 60:.1f}" if t else "—",
             f"{record.get('evaluation_wall_seconds', 0) / 60:.1f}"]
    return "| " + " | ".join(cells) + " |"


def paired(a_rows, b_rows, head, subset=None, resamples=10_000, seed=20261014):
    """Candidate minus control on identical users: review-weighted and user-averaged benchmark LogLoss deltas,
    with a percentile bootstrap over users (95%)."""
    a = {r["user_id"]: r for r in a_rows}
    b = {r["user_id"]: r for r in b_rows}
    users = sorted(u for u in a.keys() & b.keys() if (subset is None or subset(u))
                   and head in a[u].get("benchmark", {}) and head in b[u].get("benchmark", {}))
    if not users:
        return None
    w = [a[u]["benchmark"][head]["size"] for u in users]
    d = [b[u]["benchmark"][head]["LogLoss"] - a[u]["benchmark"][head]["LogLoss"] for u in users]
    weighted = sum(x * y for x, y in zip(d, w)) / sum(w)
    average = sum(d) / len(d)
    rng = random.Random(seed)
    n = len(users)
    boot_w, boot_u = [], []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        sw = sum(w[i] for i in idx)
        boot_w.append(sum(d[i] * w[i] for i in idx) / sw)
        boot_u.append(sum(d[i] for i in idx) / n)
    boot_w.sort()
    boot_u.sort()
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]  # noqa: E731
    sd = math.sqrt(sum((x - average) ** 2 for x in d) / (n - 1)) if n > 1 else float("nan")
    return {"users": n, "weighted": weighted, "weighted_ci95": [q(boot_w, 0.025), q(boot_w, 0.975)],
            "user_average": average, "user_average_ci95": [q(boot_u, 0.025), q(boot_u, 0.975)],
            "per_user_sd": sd, "better_users": sum(1 for x in d if x < 0)}


def compare(args):
    a = read_jsonl(exp_dir(args.control) / "eval" / "per_user.jsonl")
    b = read_jsonl(exp_dir(args.candidate) / "eval" / "per_user.jsonl")
    out = {"control": args.control, "candidate": args.candidate, "computed_utc": now(), "sign": "candidate - control; negative is better"}
    for name, subset in SUBSETS.items():
        for head, label in (("ahead", "curve"), ("instant", "instant")):
            out[f"{name}.{label}"] = paired(a, b, head, subset)
    target = BASE / "comparisons" / f"{args.candidate}--vs--{args.control}.json"
    atomic_json(target, out)
    for key, value in out.items():
        if isinstance(value, dict):
            print(f"{key:22s} users {value['users']:5d}  weighted {value['weighted']:+.5f} "
                  f"[{value['weighted_ci95'][0]:+.5f}, {value['weighted_ci95'][1]:+.5f}]  user-avg {value['user_average']:+.5f} "
                  f"[{value['user_average_ci95'][0]:+.5f}, {value['user_average_ci95'][1]:+.5f}]  better users {value['better_users']}")
    print(f"[compare] {target.relative_to(ROOT).as_posix()}")
    return 0


def noise(args):
    """Seed-noise band: identical configuration and users, different training seeds. With two seeds the band is the
    observed |difference| of each aggregate (what a paired single-seed comparison sees under the null); with three
    or more, also the across-seed SD and the SD of a two-run difference (SD x sqrt 2)."""
    runs = {name: load(exp_dir(name) / "result.json") for name in args.names}
    keys = sorted(set.intersection(*(set(compact(r)) for r in runs.values())) - {"reviews_per_second"})
    out = {"runs": list(runs), "seeds": [r["seed"] for r in runs.values()], "computed_utc": now(), "metrics": {}}
    for key in keys:
        values = [compact(r)[key] for r in runs.values()]
        entry = {"values": values, "range": max(values) - min(values)}
        if len(values) >= 3:
            mean = sum(values) / len(values)
            sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
            entry.update(sd=sd, sd_of_difference=sd * math.sqrt(2))
        out["metrics"][key] = entry
    per_user = [read_jsonl(exp_dir(name) / "eval" / "per_user.jsonl") for name in args.names]
    out["paired_first_two"] = {f"{name}.{label}": paired(per_user[0], per_user[1], head, subset)
                               for name, subset in SUBSETS.items() for head, label in (("ahead", "curve"), ("instant", "instant"))}
    target = BASE / "noise" / ("--".join(args.names) + ".json")
    atomic_json(target, out)
    for key in keys:
        if "LogLoss" in key:
            e = out["metrics"][key]
            print(f"{key:36s} {' '.join(f'{v:.5f}' for v in e['values'])}  range {e['range']:.5f}"
                  + (f"  sd {e['sd']:.5f}" if "sd" in e else ""))
    print(f"[noise] {target.relative_to(ROOT).as_posix()}")
    return 0


def check_eval(args):
    """Compare an experiment's per-user scores with a reference result.json (e.g. the full run's in-training evaluation)."""
    rows = {r["user_id"]: r for r in read_jsonl(exp_dir(args.name) / "eval" / "per_user.jsonl")}
    reference = {r["user_id"]: r for r in load(args.reference)["per_user"]}
    users = sorted(rows.keys() & reference.keys())
    worst = {}
    exact = 0
    for u in users:
        a, b = rows[u], reference[u]
        diffs = {k: abs(a[k] - b[k]) for k in ("curve_log_loss", "instant_log_loss", "curve_labels", "instant_labels")}
        for head in ("ahead", "instant"):
            for metric in ("LogLoss", "RMSE(bins)", "AUC"):
                x = ((a.get("benchmark") or {}).get(head) or {}).get(metric)
                y = ((b.get("benchmark") or {}).get(head) or {}).get(metric)
                if x is not None and y is not None:
                    diffs[f"{head}.{metric}"] = abs(x - y)
        exact += all(v == 0 for v in diffs.values())
        for k, v in diffs.items():
            worst[k] = max(worst.get(k, 0), v)
    out = {"experiment": args.name, "reference": str(args.reference), "users": len(users), "bitwise_equal_users": exact,
           "max_abs_diff": worst, "computed_utc": now()}
    atomic_json(BASE / "checks" / f"check-eval-{args.name}.json", out)
    print(json.dumps(out, indent=1))
    return 0 if users and exact == len(users) else 1


def status(_args):
    for folder in sorted((BASE / "experiments").glob("*")):
        e = load(folder / "experiment.json")
        phases = read_phases(folder / "phases.jsonl")
        done = len(read_jsonl(folder / "eval" / "per_user.jsonl"))
        trained = sum(1 for epoch in range(1, e["epochs"] + 1) if (epoch_result(e, epoch) or {}).get("status") == "complete")
        print(f"{e['name']:28s} {e['tier']:6s} seed {e['seed']}  trained {trained}/{e['epochs']}  evaluated {done}/{e['eval_users_count']}"
              f"  phases {len(phases)}  result {'yes' if (folder / 'result.json').exists() else 'no'}")
    return 0


def main():
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("tiers")
    p.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    p = sub.add_parser("prepare")
    p.add_argument("name")
    p.add_argument("--tier", choices=("screen", "gate", "final"), required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--order-seed", type=int, help="training-user order seed (default: the split's seed; A15 order noise)")
    p.add_argument("--options", help="JSON object merged into the split's accepted R01 track options")
    p.add_argument("--env", help="JSON object of environment variables for every process of this experiment")
    p.add_argument("--model")
    p.add_argument("--track")
    p.add_argument("--init-master")
    p.add_argument("--init-optimizer")
    p.add_argument("--update-offset", type=int, default=0)
    p.add_argument("--checkpoint", help="score-only experiment: evaluate this master.pth, no training")
    p.add_argument("--evaluator", help="MOD:FN replacing the default per-user scorer (scripts/accuracy_eval.py)")
    p.add_argument("--checkpoint-transform", help="MOD:FN(path, out_dir, experiment) -> weights to score")
    p.add_argument("--epoch-runner", help="MOD:FN(run_dir, epoch, experiment, run_full_training) replacing the epoch entry")
    p.add_argument("--eval-users", type=int, help="evaluate a stratified subset of the tier's evaluation users")
    p.add_argument("--eval-user-ids", help="comma-separated explicit evaluation users (checks); subset of the tier's")
    p.add_argument("--note")
    p = sub.add_parser("train")
    p.add_argument("name")
    p.add_argument("--epochs-per-job", type=int, help="stop after this many epochs (exit 3) to keep queue jobs short")
    p = sub.add_parser("_epoch")
    p.add_argument("name")
    p.add_argument("--epoch", type=int, required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("names", nargs="+")
    p.add_argument("--max-minutes", type=float, default=55.0)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--mode", choices=("fast", "reference"), default="fast",
                   help="fast = EV01 scripts/fast_evaluation.py (approved for accuracy comparisons 2026-10-14); reference = bitwise")
    p.add_argument("--batch-chunks", type=int, default=512)
    p = sub.add_parser("collection")
    p.add_argument("name")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--force", action="store_true")
    sub.add_parser("report").add_argument("name")
    p = sub.add_parser("compare")
    p.add_argument("control")
    p.add_argument("candidate")
    sub.add_parser("noise").add_argument("names", nargs="+")
    p = sub.add_parser("check-eval")
    p.add_argument("name")
    p.add_argument("reference", type=Path)
    sub.add_parser("status")
    args = parser.parse_args()
    handlers = {"tiers": lambda a: build_tiers(a.split), "prepare": prepare, "train": train, "_epoch": run_epoch,
                "evaluate": evaluate, "collection": collection, "report": report, "compare": compare,
                "noise": noise, "check-eval": check_eval, "status": status}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
