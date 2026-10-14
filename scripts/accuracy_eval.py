#!/usr/bin/env python
"""Held-out scoring for scripts/accuracy_loop.py: one GPU process, one or more checkpoints, resumable.

Reproduces the full run's in-training evaluation (scripts/gpu_r01_training.py with a benchmark_db): per user, the
whole history as 1024-review blocks of independent 128-review samples (seed 2025 + user + offset), float32
master weights, the R01 all-label losses and the upstream benchmark metrics (rwkv.get_result.get_stats on the
label-DB benchmark test reviews: LogLoss, RMSE(bins), AUC, curve = "ahead" and instant heads).

Several checkpoints are scored on the same prepared batches in one pass (identical inputs by construction).
Per-user rows are appended to each checkpoint's per_user.jsonl as they finish, so a stopped job resumes. The job
stops starting new users after spec["max_minutes"] and exits 3 (incomplete), keeping every queue job bounded.

  accuracy_eval.py SPEC_JSON     (written by scripts/accuracy_loop.py evaluate)

spec["mode"] = "fast" (the loop's default since 2026-10-14) delegates to the EV01 fast path in
scripts/fast_evaluation.py instead: inputs bitwise identical, outputs within FP32 rounding of this reference path.

Hook: spec["evaluator"] = "module:function" replaces the default scorer. It is called as
function(models, rows, context) and must yield (name, per_user_row) pairs with the same row schema;
context carries device, label lookup, deadline, done-sets and the spec.
"""
from contextlib import ExitStack
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def labels_reader(db_path):
    """Read-only, lock-free access to the benchmark label LMDB (never writes to the main checkout)."""
    import lmdb
    import torch
    env = lmdb.open(str(db_path), readonly=True, lock=False, readahead=False, subdir=True)

    def read(user):
        with env.begin() as txn:
            raw_ths = txn.get(f"{user}_review_ths".encode())
            raw_bins = txn.get(f"{user}_rmse_bins".encode())
        if raw_ths is None:
            return [], []
        return (torch.load(io.BytesIO(raw_ths), weights_only=True).tolist(),
                torch.load(io.BytesIO(raw_bins), weights_only=True).tolist())
    return read


def benchmark(user, labels, predictions):
    """scripts/full_run_metrics.benchmark_metrics with labels already loaded."""
    import numpy as np
    from rwkv.get_result import get_stats
    review_ths, bins = labels
    if not review_ths:
        return {"size": 0, "skipped": "no benchmark test reviews"}
    result = {"size": len(review_ths)}
    for head in ("ahead", "instant"):
        preds = predictions[head]
        kept = [i for i, th in enumerate(review_ths) if th in preds and th in predictions["rating"]]
        result[f"{head}_coverage"] = len(kept) / len(review_ths)
        if not kept:
            continue
        ths = [review_ths[i] for i in kept]
        bins_dict = {review_ths[i]: bins[i] for i in kept}
        with contextlib.redirect_stdout(io.StringIO()):
            stats, _ = get_stats(user, ths, bins_dict, {th: np.float64(preds[th]) for th in ths},
                                 {th: np.float64(predictions["rating"][th]) for th in ths})
        result[head] = {**stats["metrics"], "size": stats["size"]}
    return result


class Accumulator:
    """scripts/full_run_metrics.evaluate_with_predictions, split per batch so several models share the inputs."""

    def __init__(self):
        self.totals = {"curve_labels": 0, "instant_labels": 0, "curve_log_loss": 0.0, "curve_raw_log_loss": 0.0,
                       "instant_log_loss": 0.0, "instant_rating_cross_entropy": 0.0}
        self.predictions = {"ahead": {}, "instant": {}, "rating": {}}
        self.seconds = 0.0

    def add(self, model, batch):
        import torch
        import torch.nn.functional as F
        from scripts.full_run_metrics import collect_predictions
        started = time.perf_counter()
        with torch.no_grad():
            stats = model.get_loss(batch)
        if stats is None or not torch.isfinite(stats.average_loss):
            raise RuntimeError("Non-finite evaluation loss")
        mask = stats.is_query.bool() & stats.has_label.bool()
        losses = F.binary_cross_entropy(stats.p_imm.clamp(1e-5, 1 - 1e-5), batch.labels[:, :, 2], reduction="none")
        t = self.totals
        t["curve_labels"] += stats.ahead_n
        t["instant_labels"] += stats.imm_n
        t["curve_log_loss"] += stats.ahead_avg.item() * stats.ahead_n
        t["curve_raw_log_loss"] += stats.ahead_raw_avg.item() * stats.ahead_n
        t["instant_rating_cross_entropy"] += stats.imm_avg.item() * stats.imm_n
        t["instant_log_loss"] += losses[mask].sum().item()
        collect_predictions(stats, self.predictions)
        self.seconds += time.perf_counter() - started

    def result(self):
        t = dict(self.totals)
        assert t["curve_labels"] > 0 and t["instant_labels"] > 0
        for key in ("curve_log_loss", "curve_raw_log_loss"):
            t[key] /= t["curve_labels"]
        for key in ("instant_log_loss", "instant_rating_cross_entropy"):
            t[key] /= t["instant_labels"]
        return t


def default_scorer(models, rows, context):
    from rwkv.data_processing import get_rwkv_data
    from scripts.gpu_batch1 import DATA, split_frame
    from scripts.gpu_r01_training import evaluation_job
    from scripts.trial_batches import preparation_pool
    device = context["device"]
    with preparation_pool(context["workers"]) as pool:
        for row in rows:
            todo = [name for name in models if row["user_id"] not in context["done"][name]]
            if not todo:
                continue
            if time.time() > context["deadline"]:
                context["stopped_early"] = True
                return
            started = time.perf_counter()
            user = row["user_id"]
            full = dict(row, train_range=[0, row["available_reviews"]], validation_range=[0, row["available_reviews"]])
            section = split_frame(get_rwkv_data(DATA, user), full, "validation")
            offsets = range(0, len(section), 1024)
            seeds = [2025 + user + 0 + offset for offset in offsets]
            futures = [pool.submit(evaluation_job, section.iloc[offset:offset + 1024], user, seed) for offset, seed in zip(offsets, seeds)]
            accumulators = {name: Accumulator() for name in todo}
            for future in futures:
                batch = future.result().to(device)
                for name in todo:
                    accumulators[name].add(models[name], batch)
            labels = context["labels"](user)
            for name in todo:
                values = accumulators[name].result()
                values["benchmark"] = benchmark(user, labels, accumulators[name].predictions)
                yield name, {"user_id": user, "reviews": row["available_reviews"], "model_seconds": round(accumulators[name].seconds, 3),
                             "seconds": round(time.perf_counter() - started, 3), "shared_models": len(todo), **values}


FAST_LABEL = "fast (EV01, c6530b9)"


def main_fast(spec):
    """EV01 fast path (scripts/fast_evaluation.py, merged from eval-speed): bitwise-identical inputs, FP32 rounding
    differences only (aggregate LogLoss within 2e-10 of the reference). One checkpoint at a time; a checkpoint's
    missing users are scored in one call, so the time budget is checked between checkpoints."""
    from scripts.fast_evaluation import evaluate_checkpoint
    started = time.time()
    deadline = started + 60 * float(spec.get("max_minutes") or 1e9)
    counts = {str(r["user_id"]): r["available_reviews"] for r in spec["users"]}
    users = [r["user_id"] for r in spec["users"]]
    complete = True
    for name, checkpoint in spec["checkpoints"].items():
        output = Path(spec["outputs"][name])
        output.parent.mkdir(parents=True, exist_ok=True)
        done = {json.loads(line)["user_id"] for line in output.read_text(encoding="utf-8").splitlines() if line.strip()}             if output.exists() else set()
        todo = [u for u in users if u not in done]
        if not todo:
            continue
        if time.time() > deadline:
            complete = False
            break
        t0 = time.time()
        report = evaluate_checkpoint(checkpoint, todo, output.parent / f"fast-{time.strftime('%Y%m%d-%H%M%S')}",
                                     model=spec["model"], mode="fast", data_path=spec["dataset_path"], label_db=spec["label_db"],
                                     counts=counts, workers=int(spec.get("workers") or 5), batch_chunks=int(spec.get("batch_chunks") or 512),
                                     threads=4, quiet=True, extra={"accuracy_loop_experiment": name},
                                     device=os.environ.get("ACCURACY_EVAL_DEVICE", "cuda:0"))
        with output.open("a", encoding="utf-8") as f:
            for row in report["per_user"]:
                f.write(json.dumps({**row, "reviews": counts[str(row["user_id"])], "eval_mode": FAST_LABEL}) + "\n")
        print(f"[eval-fast] {name}: {len(todo)} users, {report['validation_reviews']:,} reviews in {report['evaluation_seconds']:.0f}s "
              f"({report['reviews_per_second']:,.0f} reviews/s; {time.time() - t0:.0f}s with setup)", flush=True)
    print(f"[eval] {'complete' if complete else 'incomplete'} after {time.time() - started:.0f}s", flush=True)
    return 0 if complete else 3


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if spec.get("mode") == "fast":
        return main_fast(spec)
    options = spec.get("options") or {}
    os.environ["GPU_BATCH1_DATA_PATH"] = spec["dataset_path"]
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ.setdefault("CUPY_CACHE_DIR", str(ROOT / "artifacts/cache/cupy-cache"))
    import torch
    from rwkv.model import nvrtc
    from scripts.benchmark_gpu_trial import benchmark_models
    torch._C._set_graph_executor_optimize(bool(options.get("jit_optimize", False)))
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_num_threads(int(options.get("main_threads", 1)))
    torch.cuda.set_per_process_memory_fraction(float(options.get("gpu_memory_fraction", 0.8)), 0)
    device = torch.device("cuda:0")
    started = time.time()
    with ExitStack() as stack:
        nvrtc.register()
        nvrtc.set_checkpoint_length(32)
        if options.get("row_tiles_exact"):
            # The trainer keeps this pinned-arithmetic recurrence backend entered through its held-out evaluation.
            from scripts.gpu_row_tiles import row_tiled_backend
            stack.enter_context(row_tiled_backend(rows=options["row_tiles_exact"], tile=32, exact=True,
                                                  compact_previous=bool(options.get("compact_previous_state", False)),
                                                  next_step_prefetch=bool(options.get("recurrence_prefetch", False)),
                                                  load_pipeline=int(options.get("recurrence_load_pipeline", 0)),
                                                  wave_aware=bool(options.get("wave_aware_backward", False))))
        models = {}
        for name, path in spec["checkpoints"].items():
            master, child, _ = benchmark_models({"MODEL": spec["model"], "DTYPE": "bfloat16", "SEED": 2025}, device)
            del child
            master.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            master.eval()
            models[name] = master
        outputs = {name: Path(path) for name, path in spec["outputs"].items()}
        done = {}
        for name, path in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            done[name] = {json.loads(line)["user_id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()} \
                if path.exists() else set()
        context = {"device": device, "labels": labels_reader(spec["label_db"]), "done": done, "spec": spec,
                   "deadline": started + 60 * float(spec.get("max_minutes") or 1e9), "workers": int(spec.get("workers") or 5),
                   "stopped_early": False}
        scorer = default_scorer
        if spec.get("evaluator"):
            import importlib
            module, _, function = spec["evaluator"].partition(":")
            scorer = getattr(importlib.import_module(module), function)
        total = len(spec["users"])
        reviews = 0
        for name, row in scorer(models, spec["users"], context):
            with outputs[name].open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            done[name].add(row["user_id"])
            if name == next(iter(models)):
                reviews += row["reviews"]
                count = len(done[name])
                if count % 20 == 0 or count == total:
                    elapsed = time.time() - started
                    print(f"[eval] {count}/{total} users | {reviews:,} reviews this job | {elapsed:.0f}s | "
                          f"{reviews / max(elapsed, 1e-9):,.0f} reviews/s", flush=True)
    complete = all(len(done[name] & {r["user_id"] for r in spec["users"]}) == total for name in models)
    print(f"[eval] {'complete' if complete else 'incomplete'} after {time.time() - started:.0f}s", flush=True)
    return 0 if complete else 3


if __name__ == "__main__":
    raise SystemExit(main())
