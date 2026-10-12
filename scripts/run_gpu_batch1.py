"""Authorized 500-user paired temporal comparisons for isolated F01–F08 tracks."""

from argparse import ArgumentParser, Namespace
from dataclasses import replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.experiment_provenance import process_provenance


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def cohort(output):
    import numpy as np
    counts = json.loads((ROOT / "artifacts/diagnostics/check-temp/first-5000-review-counts.json").read_text())
    data = ROOT / "data/anki-revlogs-10k"
    manifest = json.loads((data / "download_manifest.json").read_text())
    if manifest["revision"] != counts["revision"]:
        backups = []
        for candidate in sorted((ROOT / "data").glob("anki-revlogs-10k-original-*")):
            saved_manifest = json.loads((candidate / "download_manifest.json").read_text())
            if saved_manifest["revision"] == counts["revision"]:
                backups.append(candidate)
        if len(backups) != 1:
            raise ValueError("Cannot resolve one preserved dataset matching the saved per-user counts")
        data = backups[0]
        manifest = json.loads((data / "download_manifest.json").read_text())
    rng = np.random.default_rng(20251011)
    ordered = sorted(range(1, 5001), key=lambda user: (counts["per_user"][str(user)], user))
    selected, final = [], []
    for decile, group in enumerate(np.array_split(ordered, 10)):
        shuffled = rng.permutation(group)
        for user in shuffled[:50]:
            user = int(user)
            length = counts["per_user"][str(user)]
            train_end = int(length * 0.8) // 128 * 128
            if train_end < 128 or length - train_end < 128:
                raise ValueError(f"User {user} is too short for this protocol")
            selected.append({"user_id": user, "history_decile": decile + 1,
                             "available_reviews": length, "train_range": [0, train_end],
                             "validation_range": [train_end, length]})
        final.extend(int(user) for user in shuffled[50:60])
    selected.sort(key=lambda row: row["user_id"])
    protocol = {"user_authorization": "2026-10-11: first eight GPU optimizations, 500 users for each",
                "evaluation": "later reviews of the same 500 gradient-training users; temporal validation",
                "dataset_revision": counts["revision"], "evaluation_users": 500,
                "dataset_repository": manifest["repository"], "dataset_path": data.relative_to(ROOT).as_posix(),
                "training_users": 500, "sampling_seed": 20251011, "seeds": [2025],
                "sampling": "50 users per history-size decile across IDs 1–5000; fixed for every case",
                "rows": selected, "independent_final_test_users": sorted(final),
                "final_test_policy": "100 disjoint users reserved, not loaded or scored in this suite",
                "holdout": "first 80% (rounded down to 128 reviews) train, remaining full history evaluate; cross-split next targets masked",
                "training_reviews": sum(row["train_range"][1] for row in selected),
                "validation_reviews": sum(row["validation_range"][1] - row["validation_range"][0] for row in selected),
                "training": {"epochs": 1, "chunk_reviews": 128, "baseline_chunks_workers": [128, 4],
                             "expertium_chunks_workers": [256, 6], "dtype": "BF16", "master_dtype": "FP32",
                             "peak_lr": 7e-4, "warmup_updates": 5, "clip": 0.5,
                             "initialization": "fresh paired weights and AdamW; no checkpoint reuse"},
                "evaluation_arithmetic": "original preparation, dropout off, FP32 master weights, standard WKV tile32",
                "acceptance": {"max_binary_logloss_regression": 0.001,
                               "quality": "paired user bootstrap upper bounds for both binary losses, label-weighted and user-averaged; Bonferroni 64 comparisons across both models; 10000 resamples, seed 20251012",
                               "speed": "complete training incl user loads/workers/transfers/capture/logging faster than both matched controls; F02 includes cache construction for cold acceptance",
                               "control_stability": "both controls must agree within 0.001 on all four aggregate loss scores; otherwise inconclusive",
                               "interpretation": "one training seed; bootstrap describes user uncertainty conditional on that seed"},
                "timing": "train wall time covers one complete selected-history pass, CPU loads/preparation/worker startup, transfers, all updates, graph captures and metrics logging; model startup/primer/cache construction/validation separately reported; child process wall also reported"}
    atomic_json(output / "protocol.json", protocol)
    return protocol


def record_protocol(output, protocol):
    path = ROOT / "docs/GPU_OPTIMIZATIONS.md"
    text = path.read_text(encoding="utf-8")
    link = output.relative_to(ROOT).as_posix()
    section = f"""\n<!-- gpu-batch1:start -->
## Authorized first batch: F01–F08, 500 users each

The user explicitly authorized this batch on 2026-10-11. Each track is isolated
against fresh controls for Default and Expertium; no component inherits another
track's validation. Implementation commits: **uncommitted**. Status: **pending
completion of the approved comparisons; live progress below**.

The fixed cohort contains **500 distinct users**, 50 per history-size decile from
IDs 1–5000 (sampling seed 20251011). Train on the first 80% of each complete history,
rounded down to 128 reviews, and score the remaining reviews. Mask next-review
targets crossing the split. This is temporal validation of known users. Reserve
100 disjoint users for a final test and leave them unscored. Exact IDs, ranges,
revision and protocol are in [protocol.json](../{link}/protocol.json).
The data path is pinned to `{protocol['dataset_path']}` at revision
`{protocol['dataset_revision']}`; replacing the active dataset pointer cannot change
this batch. The first short runner check stopped before training when that pointer
was replaced with the timestamped export; [its failure is retained](../artifacts/diagnostics/gpu-batch1-smoke/baseline/control/result.json).

Training seed **2025**, one full selected-history epoch: **{protocol['training_reviews']:,}
raw training reviews** and **{protocol['validation_reviews']:,} held-out raw reviews**
per case. BF16 computation/FP32 master, 128-review chunks; Default 128 chunks/4
workers, Expertium 256 chunks/6 workers. LR 7e-4 with five-update warmup, AdamW,
clip 0.5. Within each model, update order and exposure are identical across cases.
Evaluation uses original features, dropout off, FP32 master and WKV tile32.

Acceptance requires both binary loss regressions' paired-user bootstrap upper
bounds ≤0.001 for label-weighted and user-averaged scores (10,000 resamples;
Bonferroni across 64 model/track/metric comparisons). Both controls must agree within
0.001. Complete training must be faster than both controls; F02's cold decision
includes cache creation. A warm-cache result is reported separately. User-level
uncertainty is conditional on the single training seed; this does not measure
training-seed uncertainty. Inconclusive or failed trials stay visible.

Training wall time includes user loads, worker startup/preparation, transfers,
optimizer updates, graph capture and log writes. Startup, discarded JIT primer,
cache construction, validation and complete child-process wall time are separate.
F05 uses one exact-shape graph slot and eager fallback, releasing the graph on a
shape miss for Default to leave room for eager work; F08 initially fuses the
eight time-shift/interpolation inputs, with an analytical Torch backward.

[Raw results and progress](../{link}/suite.json),
[paired quality/time report](../{link}/summary.md).
<!-- gpu-batch1:end -->
"""
    if "<!-- gpu-batch1:start -->" in text:
        begin = text.index("<!-- gpu-batch1:start -->")
        end = text.index("<!-- gpu-batch1:end -->", begin) + len("<!-- gpu-batch1:end -->")
        text = text[:begin] + section.strip() + text[end:]
    else:
        text += section
    path.write_text(text, encoding="utf-8")


def tensor_hash(state):
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class SourceWatch:
    def __init__(self, provenance):
        from zipfile import ZipFile
        with ZipFile(ROOT / provenance["source_archive"]) as snapshot:
            all_hashes = json.loads(snapshot.read("source-manifest.json"))["files"]
        used_scripts = {"run_gpu_batch1.py", "gpu_batch1.py", "report_gpu_batch1.py",
                        "benchmark_gpu_trial.py", "train_gpu_trial.py", "gpu_optimizations.py",
                        "gpu_packed_encodings.py", "experiment_provenance.py", "trial_batches.py",
                        "compare.py", "gpu_power.py"}
        # Other chats can add unrelated audit scripts while this suite runs.
        # Every case captures a fresh full archive; enforce identical files that
        # this training pipeline can import, rather than unrelated source files.
        self.hashes = {name: digest for name, digest in all_hashes.items()
                       if name.startswith(("rwkv/", "expertium/"))
                       or name in ("pyproject.toml", "uv.lock")
                       or (name.startswith("scripts/") and Path(name).name in used_scripts)}
        self.implementation_sha256 = hashlib.sha256(json.dumps(self.hashes, sort_keys=True).encode()).hexdigest()
        self.stats = {}
        self.check()

    def check(self):
        for name, digest in self.hashes.items():
            path = ROOT / name
            stat = path.stat()
            identity = (stat.st_mtime_ns, stat.st_size)
            if self.stats.get(name) != identity:
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise RuntimeError(f"Source changed during the experiment: {name}; restart in a new suite")
                self.stats[name] = identity


def run_case(args):
    started = time.perf_counter()
    provenance = process_provenance()
    watch = SourceWatch(provenance)
    if args.expected_source and watch.implementation_sha256 != args.expected_source:
        failure = {"model": args.model, "track": args.track, "status": "failed",
                   "provenance": provenance, "error": "Training implementation changed between cases; restart from the preserved source"}
        atomic_json(ROOT / args.output / args.model / args.track / "result.json", failure)
        return 1
    output = ROOT / args.output
    protocol = json.loads((output / "protocol.json").read_text())
    os.environ["GPU_BATCH1_DATA_PATH"] = protocol["dataset_path"]
    import torch
    from rwkv.model import nvrtc
    from rwkv.train_rwkv import CLIP, get_optimizer, transfer_child_grad_to_master
    from scripts.benchmark_gpu_trial import benchmark_models, fresh_checkpoint
    from scripts.compare import evaluate
    from scripts.gpu_batch1 import (DATA, TRACKS, DynamicGraphs, ReducedSyncLoss,
                                    build_sample_cache, make_samples, split_frame,
                                    training_batches, transfer_batches)
    from scripts.gpu_power import limited_gpu
    from scripts.trial_batches import preparation_pool
    from rwkv.data_processing import get_rwkv_data
    from rwkv.prepare_batch import prepare

    path = output / args.model / args.track
    path.mkdir(parents=True, exist_ok=True)
    rows = protocol["rows"]
    if args.smoke:
        rows = [dict(row, train_range=[0, 512], validation_range=[512, 1024]) for row in rows[:2]]
    options = TRACKS[args.track]
    chunks, workers = (128, 4) if args.model == "baseline" else (256, 6)
    config = {"MODEL": args.model, "DTYPE": "bfloat16", "WKV_BACKEND": "nvrtc", "SEED": 2025,
              "BATCH_CHUNKS": chunks, "CHUNK_REVIEWS": 128, "PREFETCH_WORKERS": workers,
              "PREFETCH_BATCHES": workers * 2, "CPU_THREADS": 4, "EPOCHS": 1,
              "DATA_PATH": protocol["dataset_path"], "DATASET_REVISION": protocol["dataset_revision"],
              "GPU_MEMORY_FRACTION": 0.72, "PEAK_LR": 7e-4, "WARMUP_STEPS": 5}
    report = {"model": args.model, "track": args.track, "options": options, "config": config,
              "torch": str(torch.__version__), "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0), "training_dropout": "enabled",
              "master_dtype": "float32", "float32_matmul_precision": torch.get_float32_matmul_precision(),
              "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "rss_sampling": "process tree at user transitions; sampled maximum, not continuous peak",
              "provenance": provenance, "status": "running", "smoke_only": args.smoke,
              "implementation_sha256": watch.implementation_sha256,
              "evaluation_users": len(rows), "progress_users": 0, "updates": 0,
              "review_presentations": 0, "training_curve_labels": 0, "training_instant_labels": 0,
              "hierarchical_padded_positions": 0, "peak_process_tree_rss_mib": 0.0,
              "gpu_copy_milliseconds": 0.0, "gpu_copy_overlap_training_milliseconds": 0.0}
    atomic_json(path / "result.json", report)
    try:
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(0.72, 0)
        nvrtc.register()
        nvrtc.set_checkpoint_length(32)
        nvrtc.set_raw_pointer_launch(options.get("dynamic_graphs", False))
        device = torch.device("cuda:0")
        checkpoint = fresh_checkpoint(args.model, 2025)
        report["initial_parameter_sha256"] = tensor_hash(checkpoint["model"])
        master, base, dtype = benchmark_models(config, device)
        master.load_state_dict(checkpoint["model"])
        base.copy_downcast_(master, dtype)
        for module in base.modules():
            if hasattr(module, "fused_shift"):
                module.fused_shift = options.get("fused_shift", False)
        optimizer = get_optimizer(Namespace(PEAK_LR=7e-4), master)
        # Discarded forward/backward primes JIT without performing optimizer updates.
        prime_start = time.perf_counter()
        first = split_frame(get_rwkv_data(DATA, rows[0]["user_id"]), rows[0], "train").iloc[:1024]
        prime = prepare(make_samples(first, rows[0]["user_id"], 128), seed=2025).to(device)
        prime.start = prime.start.to(dtype)
        for _ in range(3):
            base.zero_grad(set_to_none=True)
            stats = base.get_loss(prime)
            if stats is None or not torch.isfinite(stats.average_loss):
                raise RuntimeError("Non-finite JIT primer")
            stats.average_loss.backward()
        torch.cuda.synchronize()
        base.zero_grad(set_to_none=True)
        del prime, stats, first
        report["discarded_primer_seconds"] = time.perf_counter() - prime_start
        report["model_setup_seconds"] = prime_start - started
        master.load_state_dict(checkpoint["model"])
        base.copy_downcast_(master, dtype)
        graphs = DynamicGraphs(base, lambda: benchmark_models(config, device)[1],
                               retain_on_miss=args.model == "expertium") if options.get("dynamic_graphs") else None
        reduced = ReducedSyncLoss(base) if options.get("reduced_sync") else None
        cache_paths = None
        if options.get("sample_cache"):
            cache_paths, report["cache_setup"] = build_sample_cache(rows, chunks * 128, output / "sample-cache")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        train_started = time.perf_counter()
        last_user = None
        previous_compute = None
        origin = torch.cuda.Event(enable_timing=True)
        origin.record()
        coverage = {}
        with limited_gpu(0, 0, path), preparation_pool(workers) as pool, (path / "updates.jsonl").open("w", encoding="utf-8") as log:
            source = training_batches(rows, chunks * 128, 2025, pool, options, cache_paths, workers * 2)
            transferred = transfer_batches(source, dtype, options.get("async_transfer", False))
            try:
                for row, item, batch in transferred:
                    if (output / "STOP").exists():
                        raise InterruptedError("STOP requested; case not complete, restart it from fresh weights")
                    user = row["user_id"]
                    if user != last_user:
                        watch.check()
                        import psutil
                        process = psutil.Process()
                        processes = [process, *process.children(recursive=True)]
                        rss = sum(p.memory_info().rss for p in processes if p.is_running()) / 2**20
                        report["peak_process_tree_rss_mib"] = max(report["peak_process_tree_rss_mib"], rss)
                        if last_user is not None:
                            report["progress_users"] += 1
                            report["training_elapsed_seconds"] = time.perf_counter() - train_started
                            atomic_json(path / "result.json", report)
                            print(f"{args.model}/{args.track}: {report['progress_users']}/{len(rows)} users, {report['updates']} updates", flush=True)
                        last_user = user
                    cpu_labels = item.batch.labels
                    curve_n = int(((1 - cpu_labels[:, :, 6]) * cpu_labels[:, :, 4]).sum())
                    instant_n = int((cpu_labels[:, :, 6] * cpu_labels[:, :, 4]).sum())
                    if curve_n + instant_n == 0:
                        raise RuntimeError(f"Unlabelled batch user {user}, offset {item.offset}")
                    torch.manual_seed(item.seed)
                    active = graphs.select(batch) if graphs else base
                    # Graph construction creates a fresh model and consumes RNG.
                    # Reset after admission so an eager or replayed update sees
                    # the same declared dropout seed as its paired control.
                    torch.manual_seed(item.seed)
                    optimizer.zero_grad(set_to_none=True)
                    if options.get("async_transfer"):
                        compute_started = torch.cuda.Event(enable_timing=True)
                        compute_started.record()
                    active.zero_grad(set_to_none=True)
                    active.train()
                    active.copy_downcast_(master, dtype)
                    capture_start = time.perf_counter()
                    if isinstance(active, __import__("scripts.gpu_optimizations", fromlist=["GraphTrainingAdapter"]).GraphTrainingAdapter):
                        is_capture = active.graph is None
                        stats = active.graph_update(batch, item.seed)
                        if is_capture:
                            torch.cuda.synchronize()
                            graphs.metrics["capture_seconds"] += time.perf_counter() - capture_start
                    else:
                        stats = reduced.get_loss(batch) if reduced else active.get_loss(batch)
                    if stats is None or not torch.isfinite(stats.average_loss):
                        raise RuntimeError(f"Non-finite training loss user {user}")
                    if not graphs or active is base:
                        stats.average_loss.backward()
                    transfer_child_grad_to_master(master, active)
                    norm = torch.nn.utils.clip_grad_norm_(master.parameters(), CLIP, error_if_nonfinite=True)
                    lr = 7e-4 * min(1.0, (report["updates"] + 1) / 5)
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.step()
                    if options.get("async_transfer"):
                        compute_ended = torch.cuda.Event(enable_timing=True)
                        compute_ended.record()
                    if reduced:
                        loss_value, norm_value = torch.stack((stats.average_loss.detach(), norm)).cpu().tolist()
                    else:
                        loss_value, norm_value = stats.average_loss.item(), norm.item()
                    report["updates"] += 1
                    report["review_presentations"] += item.reviews
                    report["hierarchical_padded_positions"] += sum(t.numel() for group in item.batch.sub_gather for t in group)
                    report["training_curve_labels"] += curve_n
                    report["training_instant_labels"] += instant_n
                    record = coverage.setdefault(str(user), {"reviews": 0, "updates": 0})
                    if item.offset != record["reviews"]:
                        raise RuntimeError("Review coverage gap/repeat")
                    record["reviews"] += item.reviews
                    record["updates"] += 1
                    transfer_metrics = {}
                    if options.get("async_transfer"):
                        copy_begin = origin.elapsed_time(item.copy_started)
                        copy_end = origin.elapsed_time(item.copy_ended)
                        overlap = max(0.0, min(copy_end, previous_compute[1]) - max(copy_begin, previous_compute[0])) if previous_compute else 0.0
                        previous_compute = (origin.elapsed_time(compute_started), origin.elapsed_time(compute_ended))
                        report["gpu_copy_milliseconds"] += copy_end - copy_begin
                        report["gpu_copy_overlap_training_milliseconds"] += overlap
                        transfer_metrics = {"gpu_copy_ms": copy_end - copy_begin, "gpu_copy_overlap_previous_training_ms": overlap}
                    log.write(json.dumps({"step": report["updates"], "user_id": user, "offset": item.offset,
                                          "reviews": item.reviews, "seed": item.seed, "loss": loss_value,
                                          "gradient_norm": norm_value, "curve_labels": curve_n,
                                          "instant_labels": instant_n, "lr": lr,
                                          "preparation_seconds": item.preparation_seconds,
                                          "queue_wait_seconds": item.wait_seconds, **transfer_metrics}) + "\n")
                    if not reduced:
                        log.flush()
                    # Let cache eviction release all graph-private outputs and
                    # parameter gradients before the next eager shape allocates.
                    if graphs:
                        active = stats = None
            finally:
                transferred.close()
                source.close()
        torch.cuda.synchronize()
        report["training_seconds"] = time.perf_counter() - train_started
        report["cold_training_seconds"] = report["training_seconds"] + report.get("cache_setup", {}).get("elapsed_seconds", 0)
        report["reviews_per_second"] = report["review_presentations"] / report["training_seconds"]
        report["cold_reviews_per_second"] = report["review_presentations"] / report["cold_training_seconds"]
        report["progress_users"] = len(rows)
        report["coverage"] = coverage
        for row in rows:
            if coverage[str(row["user_id"])]["reviews"] != row["train_range"][1]:
                raise RuntimeError("Incomplete user training coverage")
        report["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        report["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        if graphs:
            report["graphs"] = graphs.metrics
        report["trained_parameter_sha256"] = tensor_hash(master.state_dict())
        watch.check()
        # Preserve weights in a new experiment location, never the existing trial.
        torch.save(master.state_dict(), path / "master.pth")
        active = graphs = reduced = base = optimizer = stats = batch = item = None
        master.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        nvrtc.set_checkpoint_length(32)
        nvrtc.set_raw_pointer_launch(False)
        eval_started = time.perf_counter()
        per_user = []
        validation_hash = hashlib.sha256()
        report["status"] = "evaluating"
        atomic_json(path / "result.json", report)
        for row in rows:
            watch.check()
            user = row["user_id"]
            section = split_frame(get_rwkv_data(DATA, user), row, "validation")
            def heldout():
                for offset in range(0, len(section), 1024):
                    batch = prepare(make_samples(section.iloc[offset:offset + 1024], user, 128),
                                    seed=2025 + user + row["validation_range"][0] + offset)
                    for tensor in (batch.start, batch.labels, batch.label_review_th):
                        validation_hash.update(tensor.numpy().tobytes())
                    yield {"batch": batch.to(device)}
            values = evaluate(master, heldout())
            per_user.append({"user_id": user, **values})
            if len(per_user) % 10 == 0:
                print(f"{args.model}/{args.track}: evaluated {len(per_user)}/{len(rows)} users", flush=True)
        report["validation_seconds"] = time.perf_counter() - eval_started
        report["validation_input_sha256"] = validation_hash.hexdigest()
        report["per_user"] = per_user
        report["evaluation"] = aggregate(per_user)
        report["status"] = "complete"
        watch.check()
    except Exception as exc:
        report.update(status="stopped" if isinstance(exc, InterruptedError) else "failed",
                      error=str(exc), error_type=type(exc).__name__)
        traceback.print_exc()
    report["case_elapsed_seconds"] = time.perf_counter() - started
    atomic_json(path / "result.json", report)
    return 0 if report["status"] == "complete" else 1


def aggregate(rows):
    import numpy as np
    values = {}
    for metric, count in (("curve_log_loss", "curve_labels"), ("instant_log_loss", "instant_labels")):
        weights = np.array([row[count] for row in rows], dtype=float)
        loss = np.array([row[metric] for row in rows])
        values[count] = int(weights.sum())
        values[metric + "_label_weighted"] = float(np.average(loss, weights=weights))
        values[metric + "_user_average"] = float(loss.mean())
    return values


def report_suite(output, report):
    from scripts.report_gpu_batch1 import write_report
    write_report(output, report)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/benchmarks/gpu-batch1-500")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--case", action="store_true")
    parser.add_argument("--model", choices=("baseline", "expertium"))
    parser.add_argument("--track")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected-source")
    parser.add_argument("--in-process", action="store_true", help="Short functional smoke only; validation cases remain isolated processes")
    args = parser.parse_args()
    if args.in_process and not args.smoke:
        parser.error("--in-process is limited to the functional smoke")
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Output must be below workspace artifacts")
    output.mkdir(parents=True, exist_ok=True)
    if args.case:
        return run_case(args)
    if args.prepare:
        if (output / "suite.json").exists():
            raise FileExistsError("Suite already exists; preserve its original protocol")
        protocol = cohort(output)
        if not args.smoke:
            record_protocol(output, protocol)
        print(json.dumps({"users": 500, "training_reviews": protocol["training_reviews"],
                          "validation_reviews": protocol["validation_reviews"], "protocol": str(output / "protocol.json")}))
        return 0
    protocol = json.loads((output / "protocol.json").read_text())
    os.environ["GPU_BATCH1_DATA_PATH"] = protocol["dataset_path"]
    provenance = process_provenance()
    implementation = SourceWatch(provenance).implementation_sha256
    suite_path = output / "suite.json"
    if suite_path.exists() and not args.resume:
        raise FileExistsError("Suite exists; use --resume to continue completed cases")
    report = json.loads(suite_path.read_text()) if args.resume else {"protocol": protocol, "results": [], "provenance": provenance}
    if args.resume and report.get("implementation_sha256") != implementation:
        raise ValueError("Source changed: prepare a new suite rather than mixing revisions")
    import psutil
    lock = ROOT / "artifacts/gpu-batch1.lock"
    if lock.exists():
        owner = json.loads(lock.read_text())
        if psutil.pid_exists(owner["pid"]):
            raise RuntimeError(f"GPU batch already active: {owner}")
        lock.unlink()
    with lock.open("x") as file:
        json.dump({"pid": os.getpid(), "output": str(output)}, file)
    report.update(status="running", pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat())
    report["implementation_sha256"] = implementation
    atomic_json(suite_path, report)
    try:
        from scripts.gpu_batch1 import TRACKS
        for model in ("baseline", "expertium"):
            for track in TRACKS:
                if (output / "STOP").exists():
                    report["status"] = "stopped"
                    return 1
                if any(r["model"] == model and r["track"] == track and r["status"] == "complete" for r in report["results"]):
                    continue
                folder = output / model / track
                if (folder / "result.json").exists():
                    # Never overwrite failed/stopped evidence when resuming.
                    attempt = folder / f"previous-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
                    attempt.mkdir()
                    for path in list(folder.iterdir()):
                        if path.is_file():
                            path.replace(attempt / path.name)
                folder.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--case", "--model", model,
                           "--track", track, "--output", args.output,
                           "--expected-source", implementation]
                if args.smoke:
                    command.append("--smoke")
                report["active_case"] = {"model": model, "track": track}
                atomic_json(suite_path, report)
                report_suite(output, report)
                child_start = time.perf_counter()
                with (folder / "console.log").open("w", encoding="utf-8") as console:
                    if args.in_process:
                        from contextlib import redirect_stdout, redirect_stderr
                        case_args = Namespace(**{**vars(args), "model": model, "track": track,
                                                 "expected_source": implementation})
                        with redirect_stdout(console), redirect_stderr(console):
                            code = run_case(case_args)
                    else:
                        code = subprocess.run(command, cwd=ROOT, stdout=console, stderr=subprocess.STDOUT).returncode
                raw = folder / "result.json"
                value = json.loads(raw.read_text()) if raw.exists() else {"model": model, "track": track, "status": "failed", "error": f"child exit {code} before result creation"}
                value["process_wall_seconds"] = time.perf_counter() - child_start
                atomic_json(raw, value)
                report["results"] = [r for r in report["results"] if not (r["model"] == model and r["track"] == track)] + [value]
                atomic_json(suite_path, report)
                report_suite(output, report)
                print(f"{model}/{track}: {value['status']}", flush=True)
                if value["status"] != "complete" and track in ("control", "control-repeat"):
                    report["status"] = "failed"
                    return 1
                if value.get("error", "").startswith("Training implementation changed"):
                    report["status"] = "failed"
                    return 1
        report["status"] = "complete" if all(r["status"] == "complete" for r in report["results"]) else "complete-with-failures"
        report.pop("active_case", None)
        return 0 if report["status"] == "complete" else 1
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        atomic_json(suite_path, report)
        report_suite(output, report)
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
