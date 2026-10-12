"""R01 training with independent gradient-training and evaluation users.

Adapted from the preserved batch runner; its historical protocols remain intact.
"""

from argparse import Namespace
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance
from scripts.run_gpu_batch1 import SourceWatch, tensor_hash, aggregate


def atomic_json(path, value):
    """Windows readers can briefly deny deletion/replacement of progress JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(26):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 25:
                raise
            time.sleep(0.2)


def sampled_tree_rss(process):
    """A short-lived monitor child can exit between discovery and RSS sampling."""
    import psutil
    total, skipped = 0, 0
    for item in (process, *process.children(recursive=True)):
        try:
            total += item.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            skipped += 1
    return total / 2**20, skipped


class Progress:
    """Live one-line training/evaluation status from values the loop already holds on the host."""

    def __init__(self, label, users, reviews, interval=1.0):
        self.label, self.users, self.reviews, self.interval = label, users, reviews, interval
        self.started = self.shown = time.perf_counter()
        self.window = []
        sys.stdout.write(f"{label}: {users} users, {reviews:,} reviews\n")
        sys.stdout.flush()

    def training(self, users_done, reviews_done, updates, loss):
        now = time.perf_counter()
        self.window = [(t, r) for t, r in self.window if now - t <= 10.0] + [(now, reviews_done)]
        if now - self.shown < self.interval and reviews_done < self.reviews:
            return
        self.shown = now
        elapsed = max(now - self.started, 1e-9)
        first, last = self.window[0], self.window[-1]
        recent = (last[1] - first[1]) / (last[0] - first[0]) if last[0] > first[0] else reviews_done / elapsed
        average = reviews_done / elapsed
        fraction = reviews_done / max(self.reviews, 1)
        eta = (self.reviews - reviews_done) / average if average > 0 else float("inf")
        filled = int(20 * fraction)
        sys.stdout.write(f"\r[{'#' * filled}{'.' * (20 - filled)}] {100 * fraction:5.1f}% {users_done:>3}/{self.users} users "
                         f"{reviews_done / 1e6:5.2f}M | {average:>7,.0f} rev/s avg {recent:>7,.0f} now | "
                         f"{elapsed:5.0f}s ETA {eta:4.0f}s ")
        sys.stdout.flush()

    def evaluation(self, users_done, total):
        elapsed = time.perf_counter() - self.started
        sys.stdout.write(f"\reval {users_done:>3}/{total} held-out users | {elapsed:5.1f}s ")
        sys.stdout.flush()

    def end(self):
        sys.stdout.write("\n")
        sys.stdout.flush()


def evaluation_job(section, user, seed):
    """F21: the same held-out preparation call, run in a preparation worker."""
    from rwkv.prepare_batch import prepare
    from scripts.gpu_batch1 import make_samples
    return prepare(make_samples(section, user, 128), seed=seed)


def place_host_threads(worker_pids):
    """F36: give the dispatching main thread both SMT siblings of one physical core.

    The core is the one Windows ranks most preferred (highest CPU-set scheduling
    class). Preparation workers are restricted to the other cores, so they no longer
    share the dispatch thread's core. Placement only; arithmetic and order are unchanged."""
    import ctypes
    import struct
    from ctypes import wintypes
    import psutil

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    size = wintypes.ULONG(0)
    kernel32.GetSystemCpuSetInformation(None, 0, ctypes.byref(size), None, 0)
    buffer = ctypes.create_string_buffer(size.value)
    if not kernel32.GetSystemCpuSetInformation(buffer, size, ctypes.byref(size), None, 0):
        raise OSError(ctypes.get_last_error(), "GetSystemCpuSetInformation failed")
    cores, offset = {}, 0
    while offset < size.value:
        length, = struct.unpack_from("<I", buffer.raw, offset)
        _, _, cpu, core, _, _, _, _, rank = struct.unpack_from("<IHBBBBBBB", buffer.raw, offset + 8)
        cores.setdefault(core, {"cpus": [], "rank": rank})["cpus"].append(cpu)
        offset += length
    allowed = set(psutil.Process().cpu_affinity())
    choices = [(-entry["rank"], core) for core, entry in cores.items() if set(entry["cpus"]) <= allowed]
    if len(cores) < 2 or not choices:
        raise RuntimeError("F36 host placement needs at least two physical cores in the process affinity")
    core = min(choices)[1]
    main_cpus = sorted(cores[core]["cpus"])
    worker_cpus = sorted(allowed - set(main_cpus))
    kernel32.GetCurrentThread.restype = wintypes.HANDLE
    kernel32.SetThreadAffinityMask.restype = ctypes.c_size_t
    kernel32.SetThreadAffinityMask.argtypes = (wintypes.HANDLE, ctypes.c_size_t)
    if not kernel32.SetThreadAffinityMask(kernel32.GetCurrentThread(), sum(1 << cpu for cpu in main_cpus)):
        raise OSError(ctypes.get_last_error(), "SetThreadAffinityMask failed")
    for pid in worker_pids:
        psutil.Process(pid).cpu_affinity(worker_cpus)
    return {"main_thread_cpus": main_cpus, "worker_cpus": worker_cpus, "workers": len(worker_pids),
            "physical_cores": len(cores), "core_scheduling_class": cores[core]["rank"]}


@contextmanager
def prestarted_pool(stack, pool):
    """Yield an already started preparation pool and shut it down with the training block."""
    try:
        yield pool
    finally:
        stack.close()


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
    os.environ["GPU_BATCH1_DATA_PATH"] = os.environ.get("GPU_R01_SHARED_DATA_PATH", protocol["dataset_path"])
    import torch
    from rwkv.model import nvrtc
    from rwkv.train_rwkv import CLIP, get_optimizer, transfer_child_grad_to_master
    from scripts.benchmark_gpu_trial import benchmark_models, fresh_checkpoint
    from scripts.compare import evaluate
    from scripts.gpu_batch1 import (DATA, TRACKS, DynamicGraphs,
                                    build_sample_cache, make_samples, split_frame,
                                    training_batches, transfer_batches)
    from scripts.gpu_power import limited_gpu
    from scripts.trial_batches import preparation_pool
    from rwkv.data_processing import get_rwkv_data
    from rwkv.prepare_batch import prepare
    from scripts.gpu_r01_sync import ReducedSyncLoss

    path = output / args.model / args.track
    path.mkdir(parents=True, exist_ok=True)
    rows = protocol["rows"]
    evaluation_rows = protocol.get("evaluation_rows", rows)
    if args.smoke:
        rows = [dict(row, train_range=[0, 512], validation_range=[512, 1024]) for row in rows[:2]]
        evaluation_rows = [dict(row, validation_range=[512, 1024]) for row in evaluation_rows[:2]]
    options = TRACKS[args.track]
    chunks, workers = (128, 4) if args.model == "baseline" else (256, 6)
    # F17 declares a changed update schedule; the default keeps the accepted batch size.
    chunks = options.get("batch_chunks", chunks)
    if type(chunks) is not int or not 1 <= chunks <= 2048:
        raise ValueError("Batch chunks must be an integer between 1 and 2048")
    memory_fraction = options.get("gpu_memory_fraction", protocol["training"].get("gpu_memory_fraction", 0.95))
    if not 0.1 <= memory_fraction <= 0.95:
        raise ValueError("GPU memory fraction must stay between 0.1 and 0.95")
    workers = options.get("preparation_workers", workers)
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Preparation workers must be an integer between 1 and 8")
    # Seed studies vary initialization, dropout and batch encodings; held-out inputs keep seed 2025.
    training_seed = options.get("training_seed", 2025)
    config = {"MODEL": args.model, "DTYPE": "bfloat16", "WKV_BACKEND": "nvrtc", "SEED": training_seed,
              "BATCH_CHUNKS": chunks, "CHUNK_REVIEWS": 128, "PREFETCH_WORKERS": workers,
              "PREFETCH_BATCHES": workers * 2, "CPU_THREADS": 4, "EPOCHS": 1,
              "DATA_PATH": protocol["dataset_path"], "DATASET_REVISION": protocol["dataset_revision"],
              "GPU_MEMORY_FRACTION": memory_fraction, "PEAK_LR": 7e-4, "WARMUP_STEPS": 5}
    report = {"model": args.model, "track": args.track, "options": options, "config": config,
              "torch": str(torch.__version__), "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0), "training_dropout": "enabled",
              "master_dtype": "float32", "float32_matmul_precision": torch.get_float32_matmul_precision(),
              "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "rss_sampling": "process tree at user transitions; sampled maximum, not continuous peak",
              "provenance": provenance, "status": "running", "smoke_only": args.smoke,
              "implementation_sha256": watch.implementation_sha256,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "training_users": len(rows), "evaluation_users": len(evaluation_rows), "progress_users": 0, "updates": 0,
              "review_presentations": 0, "training_curve_labels": 0, "training_instant_labels": 0,
              "hierarchical_padded_positions": 0, "peak_process_tree_rss_mib": 0.0,
              "gpu_copy_milliseconds": 0.0, "gpu_copy_overlap_training_milliseconds": 0.0}
    atomic_json(path / "result.json", report)
    kernel_stack = ExitStack()
    try:
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(memory_fraction, 0)
        nvrtc.register()
        nvrtc.set_checkpoint_length(32)
        nvrtc.set_raw_pointer_launch(options.get("dynamic_graphs", False))
        if options.get("row_tiles_exact"):
            # F31: pinned-arithmetic row-tiled recurrence for training only, at the
            # accepted tile 32; the primer compiles it and evaluation restores production.
            from scripts.gpu_row_tiles import row_tiled_backend
            report["recurrence_kernel"] = kernel_stack.enter_context(
                row_tiled_backend(rows=options["row_tiles_exact"], tile=32, exact=True))
        device = torch.device("cuda:0")
        checkpoint = fresh_checkpoint(args.model, training_seed)
        report["initial_parameter_sha256"] = tensor_hash(checkpoint["model"])
        master, base, dtype = benchmark_models(config, device)
        master.load_state_dict(checkpoint["model"])
        base.copy_downcast_(master, dtype)
        for module in base.modules():
            if hasattr(module, "fused_shift"):
                module.fused_shift = options.get("fused_shift", False)
            if hasattr(module, "exact_shift"):
                # F26: sort-free shift-gather backward, bitwise equal to deterministic PyTorch.
                module.exact_shift = options.get("exact_shift", False)
            if hasattr(module, "exact_index"):
                # F27: sort-free hierarchy row-selection backward, also bitwise equal.
                module.exact_index = options.get("exact_index", False)
            if hasattr(module, "exact_group_norm"):
                # F32: GroupNorm weight/bias gradients spread across SMs, bitwise equal.
                module.exact_group_norm = options.get("exact_group_norm", False)
        if options.get("sync_free_loss"):
            # F37: no host syncs inside the training loss; F33's deferred finite check covers its NaN checks.
            if not options.get("deferred_reads") or not hasattr(base, "sync_free_loss"):
                raise ValueError("F37 sync-free loss needs F33 deferred reads and the Expertium loss")
            base.sync_free_loss = True
        optimizer = get_optimizer(Namespace(PEAK_LR=7e-4), master)
        warm_pool = None
        if options.get("prestart_pool"):
            # F28: spawn and import the preparation workers while the GPU primer runs.
            from scripts.gpu_r01_packing import worker_warmup, warm_preparation_pool
            pool_stack = ExitStack()
            ready_directory = path / "worker-ready" / uuid4().hex
            warm_pool = pool_stack.enter_context(warm_preparation_pool(workers, ready_directory))
            pool_started = time.perf_counter()
            warm_futures = [warm_pool.submit(worker_warmup) for _ in range(workers)]
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
        grouped = None
        if options.get("grouped_sync"):
            if graphs:
                raise ValueError("Grouped master/child synchronization is not combined with graph replicas")
            from scripts.gpu_r01_update import MasterChildSync
            grouped = MasterChildSync(master, base)
        cache_paths = None
        if options.get("sample_cache"):
            cache_paths, report["cache_setup"] = build_sample_cache(rows, chunks * 128, output / "sample-cache")
        if options.get("preprocessing_cache"):
            from scripts.gpu_preprocessing_cache import plan_cache
            cache_paths, report["cache_setup"] = plan_cache(
                rows, chunks * 128, Path(os.environ.get("GPU_R01_SHARED_CACHE_PATH", str(ROOT / "artifacts/cache/r01-preprocessing-v1"))) / args.model,
                DATA, protocol["dataset_revision"], options.get("preprocessing_cache_bytes", 512 * 1024**2))
            report["cache_stats"] = {"hits": 0, "misses": 0, "not_persisted": 0,
                                     "read_seconds": 0.0, "build_seconds": 0.0, "write_seconds": 0.0}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if options.get("main_threads"):
            # The training thread only counts labels on the CPU; idle intra-op threads compete with workers.
            torch.set_num_threads(options["main_threads"])
        progress = Progress(f"{args.model}/{args.track}", len(rows),
                            sum(row["train_range"][1] - row["train_range"][0] for row in rows))
        loss_value = float("nan")
        if warm_pool is not None:
            waited = time.perf_counter()
            from scripts.gpu_r01_packing import wait_ready_workers
            report["warmup_job_workers"] = len({future.result() for future in warm_futures})
            report["prestarted_worker_pids"] = wait_ready_workers(ready_directory, workers)
            report["prestarted_workers"] = len(report["prestarted_worker_pids"])
            report["pool_ready_wait_seconds"] = time.perf_counter() - waited
            report["pool_prestart_seconds"] = time.perf_counter() - pool_started
        if options.get("host_placement"):
            if warm_pool is None:
                raise ValueError("F36 host placement needs the prestarted worker pool")
            report["host_placement"] = place_host_threads(report["prestarted_worker_pids"])
        train_started = time.perf_counter()
        report["training_started_utc"] = datetime.now(timezone.utc).isoformat()
        last_user = None
        previous_compute = None
        origin = torch.cuda.Event(enable_timing=True)
        origin.record()
        coverage = {}
        pool_context = prestarted_pool(pool_stack, warm_pool) if warm_pool is not None else preparation_pool(workers)
        with limited_gpu(0, 0, path), pool_context as pool, (path / "updates.jsonl").open("w", encoding="utf-8") as log:
            source = training_batches(rows, chunks * 128, training_seed, pool, options, cache_paths, workers * 2)
            transferred = transfer_batches(source, dtype, options.get("async_transfer", False),
                                           **({"bulk": True} if options.get("bulk_transfer") else {}))
            deferred = options.get("deferred_reads", False)
            pending = None

            def finish(update):
                nonlocal loss_value, previous_compute
                item = update["item"]
                if "host" in update:
                    update["ready"].synchronize()
                    loss_value, norm_value = update["host"].tolist()
                    if not (math.isfinite(loss_value) and math.isfinite(norm_value)):
                        raise RuntimeError(f"Non-finite training loss or gradient at update {update['step']}, user {update['user']}")
                else:
                    loss_value, norm_value = update["values"]
                transfer_metrics = {}
                if update["compute"] is not None:
                    copy_begin = origin.elapsed_time(item.copy_started)
                    copy_end = origin.elapsed_time(item.copy_ended)
                    overlap = max(0.0, min(copy_end, previous_compute[1]) - max(copy_begin, previous_compute[0])) if previous_compute else 0.0
                    previous_compute = tuple(origin.elapsed_time(event) for event in update["compute"])
                    report["gpu_copy_milliseconds"] += copy_end - copy_begin
                    report["gpu_copy_overlap_training_milliseconds"] += overlap
                    transfer_metrics = {"gpu_copy_ms": copy_end - copy_begin, "gpu_copy_overlap_previous_training_ms": overlap}
                parts = update["parts"]
                log.write(json.dumps({"step": update["step"], "user_id": update["user"], "offset": item.offset,
                                      "reviews": item.reviews, "seed": item.seed, "loss": loss_value,
                                      "gradient_norm": norm_value, "curve_labels": update["curve_n"],
                                      "instant_labels": update["instant_n"], "lr": update["lr"],
                                      "preparation_seconds": item.preparation_seconds,
                                      "queue_wait_seconds": item.wait_seconds, **transfer_metrics,
                                      **({"parts": parts} if len(parts) > 1 else {})}) + "\n")
                if not reduced:
                    log.flush()
                progress.training(report["progress_users"], report["review_presentations"], report["updates"], loss_value)

            try:
                for row, item, batch in transferred:
                    cache_info = getattr(item, "cache_info", None)
                    if cache_info:
                        counters = report["cache_stats"]
                        counters["hits" if cache_info["hit"] else "misses"] += 1
                        counters["not_persisted"] += int(not cache_info["persisted"])
                        for field in ("read_seconds", "build_seconds", "write_seconds"):
                            counters[field] += cache_info[field]
                    if (output / "STOP").exists():
                        raise InterruptedError("STOP requested; case not complete, restart it from fresh weights")
                    user = row["user_id"]
                    # F29 batches may span users; single-user batches have one part.
                    parts = getattr(item, "parts", None) or [(user, item.offset, item.reviews)]
                    for part_user, _, _ in parts:
                        if part_user == last_user:
                            continue
                        watch.check()
                        import psutil
                        process = psutil.Process()
                        rss, skipped = sampled_tree_rss(process)
                        report["rss_sampling_skipped_processes"] = report.get("rss_sampling_skipped_processes", 0) + skipped
                        report["peak_process_tree_rss_mib"] = max(report["peak_process_tree_rss_mib"], rss)
                        if last_user is not None:
                            report["progress_users"] += 1
                            report["training_elapsed_seconds"] = time.perf_counter() - train_started
                            atomic_json(path / "result.json", report)
                        last_user = part_user
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
                    if grouped:
                        grouped.downcast()
                    else:
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
                    if stats is None or (not deferred and not torch.isfinite(stats.average_loss)):
                        raise RuntimeError(f"Non-finite training loss user {user}")
                    if not graphs or active is base:
                        stats.average_loss.backward()
                    if grouped:
                        grouped.transfer_grads()
                    else:
                        transfer_child_grad_to_master(master, active)
                    # F33 checks the same finite loss/norm after the next update is queued.
                    norm = torch.nn.utils.clip_grad_norm_(master.parameters(), CLIP, error_if_nonfinite=not deferred)
                    lr = 7e-4 * min(1.0, (report["updates"] + 1) / 5)
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.step()
                    if options.get("async_transfer"):
                        compute_ended = torch.cuda.Event(enable_timing=True)
                        compute_ended.record()
                    report["updates"] += 1
                    report["review_presentations"] += item.reviews
                    report["hierarchical_padded_positions"] += sum(t.numel() for group in item.batch.sub_gather for t in group)
                    report["training_curve_labels"] += curve_n
                    report["training_instant_labels"] += instant_n
                    for part_user, part_offset, part_reviews in parts:
                        record = coverage.setdefault(str(part_user), {"reviews": 0, "updates": 0})
                        if part_offset != record["reviews"]:
                            raise RuntimeError("Review coverage gap/repeat")
                        record["reviews"] += part_reviews
                        record["updates"] += 1
                    update = {"step": report["updates"], "user": user, "item": item, "curve_n": curve_n,
                              "instant_n": instant_n, "lr": lr, "parts": parts,
                              "compute": (compute_started, compute_ended) if options.get("async_transfer") else None}
                    if deferred:
                        # F33: queue a pinned copy of loss/norm and read it once the next update is queued.
                        values = torch.stack((stats.average_loss.detach().float(), norm.detach().float()))
                        update["host"] = torch.empty(2, dtype=torch.float32, pin_memory=True)
                        update["host"].copy_(values, non_blocking=True)
                        update["ready"] = torch.cuda.Event()
                        update["ready"].record()
                        if pending is not None:
                            finish(pending)
                        pending = update
                    else:
                        if reduced:
                            update["values"] = torch.stack((stats.average_loss.detach(), norm)).cpu().tolist()
                        else:
                            update["values"] = stats.average_loss.item(), norm.item()
                        finish(update)
                    # Let cache eviction release all graph-private outputs and
                    # parameter gradients before the next eager shape allocates.
                    if graphs:
                        active = stats = None
                if pending is not None:
                    finish(pending)
                    pending = None
            finally:
                transferred.close()
                source.close()
        torch.cuda.synchronize()
        report["training_seconds"] = time.perf_counter() - train_started
        kernel_stack.close()
        progress.training(len(rows), report["review_presentations"], report["updates"], loss_value)
        progress.end()
        if options.get("main_threads"):
            torch.set_num_threads(4)
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
        progress = Progress(f"{args.model}/{args.track}", len(evaluation_rows), 0)
        with preparation_pool(workers if options.get("parallel_evaluation") else 0) as evaluation_pool:
            for row in evaluation_rows:
                watch.check()
                user = row["user_id"]
                section = split_frame(get_rwkv_data(DATA, user), row, "validation")
                def heldout():
                    offsets = range(0, len(section), 1024)
                    seeds = [2025 + user + row["validation_range"][0] + offset for offset in offsets]
                    if evaluation_pool is None:
                        batches = (prepare(make_samples(section.iloc[offset:offset + 1024], user, 128), seed=seed)
                                   for offset, seed in zip(offsets, seeds))
                    else:
                        # Futures are consumed in submission order, so hashing and evaluation order are unchanged.
                        futures = [evaluation_pool.submit(evaluation_job, section.iloc[offset:offset + 1024], user, seed)
                                   for offset, seed in zip(offsets, seeds)]
                        batches = (future.result() for future in futures)
                    for batch in batches:
                        for tensor in (batch.start, batch.labels, batch.label_review_th):
                            validation_hash.update(tensor.numpy().tobytes())
                        yield {"batch": batch.to(device)}
                values = evaluate(master, heldout())
                per_user.append({"user_id": user, **values})
                progress.evaluation(len(per_user), len(evaluation_rows))
        progress.end()
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
    finally:
        kernel_stack.close()
    report["case_elapsed_seconds"] = time.perf_counter() - started
    atomic_json(path / "result.json", report)
    return 0 if report["status"] == "complete" else 1
