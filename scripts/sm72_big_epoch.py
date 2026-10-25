#!/usr/bin/env python
"""SM72 step 1: one leak-free epoch for the big model (the fork's 2,762,884-parameter RWKV) from its shipped weights.

The shipped .pth loads as is into the variant class with architecture "fork" (expertium.variants: the fork's
rwkv.architecture config, stream-major layer order on the canonical gather/scatter path; the scorer gives identical
rows for "fork::" and "expertium:fork:" on users 4676 and 516), which is where the trainer's carried state hooks in
(scripts/state_carry_train.py; CPU check: scripts/check_state_carry_lanes.py --architecture fork). This writes a
scripts/expertium_decay.py output folder (decay.json + protocol.json, "model": "expertium") and runs it through that
script's run/smoke/stop/save, so the epoch is resumable from snapshots exactly like the SM56 decays:

- weights: the shipped checkpoint; optimizer: a fresh AdamW (upstream rwkv.train_rwkv.get_optimizer groups);
- learning rate: linear warmup over --warmup-updates updates, times upstream's decay shape (TRAIN_MODE "D",
  `1 + cos(pi/2 (1 + t))`, our one_minus_sin) from --peak (3.5e-4, user 10-25) to 0 over the epoch's reviews;
- users: the fork's own training users 5000-10000 (5,001 users, 370.3M reviews), shuffled with --seed, whole histories;
- batches: --batch-chunks strided lanes of --chunk-reviews windows per update with carried state (the full retrain's
  recipe, SM56: each lane continues from its previous window's end state; lane starts from zero), so the model trains
  on whole histories in short windows. --fresh-state = the earlier recipe: independent fresh-state sections through
  the baseline class (upstream data_processing cuts 32,768-review sections, each from a fresh state);
- inputs: the process must have RWKV_DELETED_CARDS=neighbour (checked), so get_rwkv_data applies the SM71 transform.

  check   [--users 4676,516]          CPU: the shipped weights load into the trainer's model bit for bit, and the
                                       scorer gives identical rows for the shipped file and the trainer's re-saved copy
  prepare --out DIR [--warmup-updates 200 --chunk-reviews 2048 --batch-chunks 16 --peak 3.5e-4 --fresh-state
                     --seed 1 --snapshot-users 400 --memory 0.8 --probe]
                     (--probe: 4 users cut to 131,072 reviews each, a GPU memory/speed probe)
  run | smoke | stop | save --out DIR  (scripts/expertium_decay.py commands on that folder)
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_full_training import SNAPSHOT_USERS, atomic_json  # noqa: E402

FORK = Path("C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth")
RUN8K = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/full-run/run-8k-20261013/run.json")  # review_counts of all 10k
DATA = Path("C:/Users/jonat/workspace/anki-rwkv/data/anki-revlogs-10k-original-20261011")
USERS = range(5000, 10001)  # upstream TRAIN_USERS_START / TRAIN_USERS_END (inclusive)
TRACK = "big-leakfree-1ep"
MODE = "neighbour"
PEAK = 3.5e-4  # user 10-25: half the upstream peak (the update holds ~half of upstream's 66k-review batch)
# The full retrain's decay (read only): its R01 speed options are the carry recipe's base.
SM56_DECAY = Path("C:/Users/jonat/workspace/wt/sm56/artifacts/small-model/scale-w64kd-final-sc-lanes-dec15-1ep32k-muon/decay.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(users):
    """Bit-exact load + identical scorer rows (CPU, fp32 and w8+s8) for the shipped file and the trainer's copy."""
    import tempfile
    import torch
    from scripts.benchmark_gpu_trial import fresh_checkpoint
    from scripts.compare import BASELINE_CONFIG, BaselineModel
    shipped = torch.load(FORK, map_location="cpu", weights_only=True)
    fresh = fresh_checkpoint("baseline")["model"]
    keys_match = list(fresh) == list(shipped) and all(fresh[k].shape == shipped[k].shape for k in fresh)
    master = BaselineModel(BASELINE_CONFIG).float()
    master.load_state_dict(shipped)  # strict
    copy = master.state_dict()
    exact = all(torch.equal(copy[k], shipped[k]) and copy[k].dtype == shipped[k].dtype for k in shipped)
    params = sum(p.numel() for p in master.parameters())
    report = {"shipped": str(FORK), "shipped_sha256": sha256(FORK), "parameters": params,
              "trainer_keys_and_shapes_match": keys_match, "state_dict_bit_exact": exact}
    with tempfile.TemporaryDirectory(dir=ROOT / "artifacts/small-model/sm72") as tmp:
        resaved = Path(tmp) / "trainer-copy.pth"
        torch.save(copy, resaved)
        out = Path(tmp) / "rows.jsonl"
        users_file = Path(tmp) / "users.json"
        users_file.write_text(json.dumps({"users": users}))
        cmd = [sys.executable, "-X", "utf8", str(ROOT / "scripts/small_model_heldout.py"), "--pool", "732",
               "--users-file", str(users_file), "--device", "cpu", "--model", f"shipped=fork::{FORK}",
               "--model", f"copy=fork::{resaved}", "--variants", "fp32,w8+s8", "--out", str(out)]
        env = {k: v for k, v in os.environ.items() if k != "RWKV_DELETED_CARDS"}
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        report["scorer_exit"] = proc.returncode
        if proc.returncode:
            report["scorer_tail"] = (proc.stdout + proc.stderr)[-2000:]
        rows = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
    by = {}
    volatile = {"label", "tag", "checkpoint", "prepare_seconds", "model_seconds", "provenance_commit", "provenance_dirty"}
    for row in rows:
        if "/" not in row["label"]:
            by.setdefault((row["user_id"], "error"), {})["error"] = row
            continue
        tag, variant = row["label"].split("/", 1)
        by.setdefault((row["user_id"], variant), {})[tag] = {k: v for k, v in row.items() if k not in volatile}
    pairs = {f"{u}/{v}": d.get("shipped") == d.get("copy") and "shipped" in d for (u, v), d in sorted(by.items())}
    report["scorer_rows_identical"] = pairs
    report["pass"] = bool(keys_match and exact and params == 2_762_884 and pairs and all(pairs.values()))
    path = ROOT / "artifacts/small-model/sm72/check-big-load-sm72.json"
    atomic_json(path, report)
    print(json.dumps(report, indent=1))
    return 0 if report["pass"] else 1


def carry_options(chunk, batch_chunks, memory):
    """SM72 carry recipe: the full retrain's decay options (SM56) without its teacher and Muon, the fork's architecture
    through the variant class ("fork": stream-major, so the carried-state hook applies), the switches that
    architecture cannot run removed, strided lanes with carried state."""
    from scripts.expertium_options import compatible_options
    base = json.loads(SM56_DECAY.read_text())["options"]
    options = {k: v for k, v in base.items() if k not in ("distillation", "optimizer")}
    options.update(architecture="fork", chunk_reviews=chunk, batch_chunks=batch_chunks, gpu_memory_fraction=memory,
                   state_carry={"lanes": "strided"}, training_seed=2025)
    return compatible_options(options)


def prepare(out, warmup, chunk, batch_chunks, seed, snapshot_users, memory, probe, peak=PEAK, carry=True):
    if out.exists():
        raise FileExistsError(f"{out} exists; pick a new output")
    run = json.loads(RUN8K.read_text())
    counts = run["review_counts"]
    users = [u for u in USERS if str(u) in counts]
    random.Random(seed).shuffle(users)
    if probe:  # the 4 longest users, each cut to 131,072 reviews (2 updates at 2 x 32,768): memory and speed only
        users = sorted(users, key=lambda u: -counts[str(u)])[:4]
    cut = 131_072 if probe else None
    rows = [{"user_id": u, "available_reviews": counts[str(u)], "train_range": [0, min(counts[str(u)], cut or counts[str(u)])],
             "validation_range": [0, counts[str(u)]]} for u in users]
    reviews = sum(r["train_range"][1] for r in rows)
    dropped = []
    if carry:
        options, dropped = carry_options(chunk, batch_chunks, memory)
    else:  # fallback: fresh-state sections through the baseline class (the first prepared recipe)
        options = {"prefetch_users": True, "preparation_workers": 5, "chunk_reviews": chunk, "batch_chunks": batch_chunks,
                   "gpu_memory_fraction": memory, "training_seed": 2025}
    schedule = {"peak": peak, "warmup_updates": warmup, "decay_start": 0.0, "decay_end": 1.0, "shape": "one_minus_sin",
                "final": 0.0}
    full_run = {"epoch": 0, "evaluate": False, "init_master": str(FORK), "update_offset": 0, "lr_schedule": schedule,
                "evaluation_set": "none", "snapshot_users": snapshot_users, "optimizer_state": "fresh",
                "control_dir": (out / "control").relative_to(ROOT).as_posix()}
    protocol = {"rows": rows, "evaluation_rows": [], "dataset_path": str(DATA), "dataset_revision": run["dataset_revision"],
                "training": {"epochs": 1, "chunk_reviews": chunk, "gpu_memory_fraction": memory},
                "execution_mode": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}, "full_run": full_run}
    decay = {"created_utc": datetime.now(timezone.utc).isoformat(), "kind": "sm72-big-leakfree-epoch",
             "source_case": str(FORK), "source_sha256": sha256(FORK), "model": "expertium" if carry else "baseline",
             "track": TRACK, "options": options, "options_dropped": dropped, "deleted_cards": MODE, "probe": probe,
             "training_users": len(rows), "training_reviews": reviews,
             "train_users_from": "5000-10000 (upstream train_rwkv_config.toml)", "shuffle_seed": seed,
             "lr": f"{peak:g} -> 0, warmup {warmup} updates, shape one_minus_sin, over all {reviews:,} reviews",
             "optimizer": "fresh AdamW (rwkv.train_rwkv.get_optimizer)",
             "batches": (f"{batch_chunks} strided lanes x {chunk:,}-review windows per update, carried state, one user "
                         "per update" if carry else
                         f"{batch_chunks} x {chunk:,} reviews per update, fresh state, one user per update")}
    out.mkdir(parents=True)
    atomic_json(out / "decay.json", decay)
    atomic_json(out / "protocol.json", protocol)
    print(json.dumps({k: decay[k] for k in ("training_users", "training_reviews", "lr", "batches", "probe")}))
    return 0


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "prepare", "run", "smoke", "stop", "save"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--users", default="4676,516")
    parser.add_argument("--warmup-updates", type=int, default=200)
    parser.add_argument("--chunk-reviews", type=int, default=2048)
    parser.add_argument("--batch-chunks", type=int, default=16)
    parser.add_argument("--peak", type=float, default=PEAK)
    parser.add_argument("--fresh-state", action="store_true", help="fallback: fresh-state sections, baseline class")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--snapshot-users", type=int, default=SNAPSHOT_USERS)
    parser.add_argument("--memory", type=float, default=0.8)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.command == "check":
        return check([int(u) for u in args.users.split(",")])
    if args.out is None:
        parser.error("--out is required")
    out = (ROOT / args.out).resolve()
    if not out.is_relative_to(ROOT / "artifacts"):
        parser.error("--out must be under this checkout's artifacts/")
    if args.command in ("prepare", "run", "smoke") and os.environ.get("RWKV_DELETED_CARDS") != MODE:
        parser.error(f"set RWKV_DELETED_CARDS={MODE} (leak-free training inputs)")
    if args.command == "prepare":
        return prepare(out, args.warmup_updates, args.chunk_reviews, args.batch_chunks, args.seed, args.snapshot_users,
                       args.memory, args.probe, args.peak, not args.fresh_state)
    from scripts import expertium_decay
    if args.command in ("stop", "save"):
        (out / "control").mkdir(exist_ok=True)
        (out / "control" / args.command.upper()).touch()
        print(f"[sm72-big] {args.command.upper()} requested for {out.name}: snapshot at the next user boundary", flush=True)
        return 0
    decay = json.loads((out / "decay.json").read_text())
    if decay.get("deleted_cards") != os.environ.get("RWKV_DELETED_CARDS"):
        raise SystemExit(f"{out} was prepared for deleted_cards={decay.get('deleted_cards')}")
    return expertium_decay.smoke(out) if args.command == "smoke" else expertium_decay.run(out)


if __name__ == "__main__":
    raise SystemExit(main())
