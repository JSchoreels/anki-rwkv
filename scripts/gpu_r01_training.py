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


# Long-context training option (chunk_reviews) is wired through; runbook-next.sh `context` checks for this flag.
LONG_CHUNKS_READY = True


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


def cpu_set_cores():
    """Physical cores as {core index: {"cpus": logical CPUs, "rank": CPU-set scheduling class}}."""
    import ctypes
    import struct
    from ctypes import wintypes

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
    return cores


def place_host_cores(worker_pids):
    """F71: one preparation worker per physical core; the main process keeps the remaining cores.

    Windows can pack the workers two per physical core (SMT siblings) while other cores idle,
    which slowed preparation about 1.4x and main-thread dispatch about 1.2x in the 2026-10-13
    baseline-drift diagnostic. Each worker gets the first logical CPU of its own core; the main
    process (dispatch, driver and telemetry threads) gets both siblings of the best-ranked
    remaining cores. Returns the report and the previous main-process affinity, which must be
    restored before any later pool starts, because child processes inherit it.
    Placement only; arithmetic and order are unchanged."""
    import psutil

    cores = cpu_set_cores()
    process = psutil.Process()
    previous = process.cpu_affinity()
    usable = sorted((-entry["rank"], core) for core, entry in cores.items() if set(entry["cpus"]) <= set(previous))
    if len(usable) <= len(worker_pids):
        raise RuntimeError(f"F71 placement needs more physical cores than workers: {len(usable)} cores, {len(worker_pids)} workers")
    main_cores = [core for _, core in usable[:len(usable) - len(worker_pids)]]
    worker_cores = [core for _, core in usable[len(usable) - len(worker_pids):]]
    main_cpus = sorted(cpu for core in main_cores for cpu in cores[core]["cpus"])
    process.cpu_affinity(main_cpus)
    worker_cpus = []
    for pid, core in zip(sorted(worker_pids), worker_cores):
        cpu = min(cores[core]["cpus"])
        psutil.Process(pid).cpu_affinity([cpu])
        worker_cpus.append(cpu)
    return {"mode": "physical", "main_process_cpus": main_cpus, "worker_cpus": worker_cpus, "workers": len(worker_pids),
            "physical_cores": len(cores), "main_core_scheduling_classes": [cores[core]["rank"] for core in main_cores]}, previous


def set_host_qos(worker_pids, main_priority=None, worker_qos=None):
    """F86: opt the main process and preparation workers out of Windows power throttling (EcoQoS)
    and raise them to above-normal priority.

    Windows 11 may lower the clock and priority of background console processes, which the
    agent-launched runs are; the slow host state ran dispatch about 1.2x slower with unchanged
    GPU clocks. Records each process's previous throttling and priority state and the system
    timer resolution (100 ns units). F93b `worker_qos="eco"` instead leaves the preparation workers
    under EcoQoS (execution-speed throttling on) at normal priority, keeping power and boost headroom
    for the unthrottled main process. Scheduling only; arithmetic and order are unchanged."""
    import ctypes
    from ctypes import wintypes
    import psutil

    class Throttling(ctypes.Structure):
        _fields_ = [("version", wintypes.ULONG), ("control", wintypes.ULONG), ("state", wintypes.ULONG)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    for name in ("GetProcessInformation", "SetProcessInformation"):
        getattr(kernel32, name).restype = wintypes.BOOL
        getattr(kernel32, name).argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    power_throttling, set_information, query_limited = 4, 0x0200, 0x1000
    if worker_qos not in (None, "eco"):
        raise ValueError("F93b worker_qos supports only 'eco'")
    eco_workers = worker_qos == "eco"
    processes = []
    for pid in [os.getpid(), *sorted(worker_pids)]:
        handle = kernel32.OpenProcess(set_information | query_limited, False, pid)
        if not handle:
            raise OSError(ctypes.get_last_error(), f"OpenProcess failed for {pid}")
        try:
            before = Throttling(1, 0, 0)
            queried = kernel32.GetProcessInformation(handle, power_throttling, ctypes.byref(before), ctypes.sizeof(before))
            # Take control of EXECUTION_SPEED and IGNORE_TIMER_RESOLUTION with both state bits clear: never throttle.
            wanted = Throttling(1, 0x1 | 0x4, 0)
            if eco_workers and pid != os.getpid():
                wanted = Throttling(1, 0x1, 0x1)
            if not kernel32.SetProcessInformation(handle, power_throttling, ctypes.byref(wanted), ctypes.sizeof(wanted)):
                raise OSError(ctypes.get_last_error(), f"SetProcessInformation(ProcessPowerThrottling) failed for {pid}")
        finally:
            kernel32.CloseHandle(handle)
        process = psutil.Process(pid)
        previous_priority = int(process.nice())
        # F86b: optionally raise only the dispatching main process to high priority (never realtime).
        high = main_priority == "high" and pid == os.getpid()
        eco = eco_workers and pid != os.getpid()
        process.nice(psutil.HIGH_PRIORITY_CLASS if high else psutil.NORMAL_PRIORITY_CLASS if eco else psutil.ABOVE_NORMAL_PRIORITY_CLASS)
        processes.append({"pid": pid, "role": "main" if pid == os.getpid() else "worker",
                          "previous_throttling": {"control_mask": before.control, "state_mask": before.state} if queried else None,
                          "previous_priority_class": previous_priority, "priority_class": int(process.nice()),
                          **({"throttling": "eco"} if eco else {})})
    coarsest, finest, current = wintypes.ULONG(), wintypes.ULONG(), wintypes.ULONG()
    status = ctypes.WinDLL("ntdll").NtQueryTimerResolution(ctypes.byref(coarsest), ctypes.byref(finest), ctypes.byref(current))
    return {"mode": "main_high_unthrottled" if main_priority == "high" else "above_normal_unthrottled", "processes": processes,
            "timer_resolution_100ns": {"coarsest": coarsest.value, "finest": finest.value, "current": current.value} if status == 0 else None}


def place_host_threads(worker_pids):
    """F36: give the dispatching main thread both SMT siblings of one physical core.

    The core is the one Windows ranks most preferred (highest CPU-set scheduling
    class). Preparation workers are restricted to the other cores, so they no longer
    share the dispatch thread's core. Placement only; arithmetic and order are unchanged."""
    import ctypes
    from ctypes import wintypes
    import psutil

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    cores = cpu_set_cores()
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
    # Long-context training (docs/SMALL_MODEL.md 5.3): reviews per independent training chunk. Unset = 128, the
    # accepted R01 layout, unchanged. Evaluation keeps its 128-review samples. Up to 32,768 = the fork's own context.
    chunk_reviews = options.get("chunk_reviews", 128)
    if type(chunk_reviews) is not int or not 16 <= chunk_reviews <= 32768:
        raise ValueError("chunk_reviews must be an integer between 16 and 32768")
    if chunk_reviews != 128 and (options.get("sample_cache") or options.get("preprocessing_cache") or options.get("pack_users")):
        raise ValueError("chunk_reviews other than 128 does not support sample/preprocessing caches or pack_users")
    config = {"MODEL": args.model, "DTYPE": "bfloat16", "WKV_BACKEND": "nvrtc", "SEED": training_seed,
              "BATCH_CHUNKS": chunks, "CHUNK_REVIEWS": chunk_reviews, "PREFETCH_WORKERS": workers,
              "PREFETCH_BATCHES": workers * 2, "CPU_THREADS": 4, "EPOCHS": 1,
              "DATA_PATH": protocol["dataset_path"], "DATASET_REVISION": protocol["dataset_revision"],
              "GPU_MEMORY_FRACTION": memory_fraction, "PEAK_LR": 7e-4, "WARMUP_STEPS": 5}
    report = {"model": args.model, "track": args.track, "options": options, "config": config,
              "torch": str(torch.__version__), "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0), "training_dropout": "enabled",
              # Includes this process's CUDA context; the rest is other processes' VRAM.
              "device_used_at_start_mib": (lambda free, total: (total - free) / 2**20)(*torch.cuda.mem_get_info()),
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
        # F38: one shift-backward sort per split and stream-argument raw launches, bitwise equal.
        from rwkv.model import exact_index
        nvrtc.set_stream_launch(options.get("stream_launches", False))
        nvrtc.set_driver_launch(options.get("driver_launch", False))
        exact_index.set_shared_orders(options.get("shared_shift_orders", False))
        # F35: row-selection backward with pipelined loads; same sequential sums, bitwise equal.
        exact_index.set_prefetch_rows(options.get("prefetch_rows", False))
        exact_index.set_prefetch_shift(options.get("prefetch_shift", False))
        if options.get("row_tiles_exact"):
            # F31: pinned-arithmetic row-tiled recurrence for training only, at the
            # accepted tile 32; the primer compiles it and evaluation restores production.
            from scripts.gpu_row_tiles import row_tiled_backend
            report["recurrence_kernel"] = kernel_stack.enter_context(
                row_tiled_backend(rows=options["row_tiles_exact"], tile=32, exact=True,
                                  compact_previous=bool(options.get("compact_previous_state", False)),
                                  next_step_prefetch=bool(options.get("recurrence_prefetch", False)),
                                  load_pipeline=int(options.get("recurrence_load_pipeline", 0)),
                                  wave_aware=bool(options.get("wave_aware_backward", False))))
        device = torch.device("cuda:0")
        # Expertium catch-up switches (docs/EXPERTIUM_CATCH_UP.md); a track without them keeps this path unchanged.
        catch_up = None
        if any(key in options for key in ("architecture", "lr_schedule", "optimizer", "end_to_start_intervals",
                                          "distillation")) or (protocol.get("full_run") or {}).get("lr_schedule"):
            from scripts.expertium_options import CatchUp
            catch_up = CatchUp(options, protocol.get("full_run") or {}, rows)
        variant = catch_up is not None and catch_up.spec is not None
        checkpoint = catch_up.fresh_checkpoint(training_seed) if variant else fresh_checkpoint(args.model, training_seed)
        # Full runs (scripts/run_full_training.py) continue from the previous epoch; R01 protocols have no "full_run".
        full_run = protocol.get("full_run") or {}
        if full_run.get("init_master"):
            checkpoint = {"model": torch.load(ROOT / full_run["init_master"], map_location="cpu", weights_only=True)}
        report["initial_parameter_sha256"] = tensor_hash(checkpoint["model"])
        master, base, dtype = catch_up.models(config, device) if variant else benchmark_models(config, device)
        master.load_state_dict(checkpoint["model"])
        base.copy_downcast_(master, dtype)
        if options.get("native_wkv") or options.get("native_group_norm") or options.get("native_index") or options.get("native_glue"):
            from rwkv.model.native_launch import configure
            report["native_extension"] = configure(base, options)
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
        if options.get("instant_loss_weight") is not None:
            # SM26: scales only the instant (rating) term of the training loss; evaluation is unaffected.
            if not hasattr(base, "instant_loss_weight"):
                raise ValueError("instant_loss_weight needs the Expertium loss")
            base.instant_loss_weight = float(options["instant_loss_weight"])
        if options.get("wddm_flush_streams"):
            # F77b: identity flush points between hierarchy streams in forward and backward.
            if not hasattr(base, "wddm_flush_streams"):
                raise ValueError("F77b needs the Expertium stream loop")
            base.wddm_flush_streams = True
        if options.get("skip_padding_sources"):
            if not options.get("exact_index") or not hasattr(base, "skip_padding_sources"):
                raise ValueError("F50 requires Expertium exact row selection")
            base.skip_padding_sources = True
        for module in base.modules():
            if hasattr(module, "partial_shift_lerp"):
                module.partial_shift_lerp = bool(options.get("partial_shift_lerp", False))
            if hasattr(module, "explicit_shift_lerp"):
                module.explicit_shift_lerp = bool(options.get("explicit_shift_lerp", False))
            if hasattr(module, "exact_shift_lerp"):
                module.exact_shift_lerp = bool(options.get("exact_shift_lerp", False))
            if hasattr(module, "exact_bonus_products"):
                module.exact_bonus_products = bool(options.get("exact_bonus_products", False))
            if hasattr(module, "exact_gate_glue"):
                module.exact_gate_glue = int(options.get("exact_gate_glue", 0))
            if hasattr(module, "exact_channel_glue"):
                module.exact_channel_glue = bool(options.get("exact_channel_glue", False))
        base.batched_split_gather = bool(options.get("batched_split_gather", False))
        base.fused_padded_gather = bool(options.get("fused_padded_gather", False))
        if hasattr(base, "concurrent_split_limit"):
            base.concurrent_split_limit = int(options.get("concurrent_split_limit", 0))
        if options.get("concurrent_splits"):
            if not hasattr(base, "concurrent_splits"):
                raise ValueError("F49 requires Expertium")
            base.concurrent_splits = True
        if options.get("state_vq") is not None:
            # SM32b: card/note state through the SM32 VQ codec at every stored step (fake-quant, STE, frozen
            # codebooks); absent = no mixer is touched.
            if options.get("fused_shift") or not hasattr(base, "rwkv_modules"):
                raise ValueError("state_vq needs the Expertium model without fused_shift")
            from scripts.state_vq import install_training
            report["state_vq"] = install_training(base, options["state_vq"], device)
        optimizer = catch_up.optimizer(master) if catch_up else get_optimizer(Namespace(PEAK_LR=7e-4), master)
        if catch_up:
            catch_up.load_teacher(device, dtype)
            report["expertium_catch_up"] = catch_up.describe(master)
            if catch_up.teacher is not None and catch_up.teacher.cache is not None:
                report["kd_cache"] = catch_up.teacher.cache_stats  # live counters, saved with every result.json
        if full_run.get("init_optimizer"):
            # Load on CPU: load_state_dict moves moments to the parameters' device but keeps AdamW's `step` where it
            # is, and a CUDA `step` makes non-capturable AdamW sync once per parameter (429 per update).
            optimizer.load_state_dict(torch.load(ROOT / full_run["init_optimizer"], map_location="cpu", weights_only=True))
        learned_books = None
        if options.get("state_vq") is not None:
            from scripts import state_vq
            learned_books = state_vq.LEARNED
            if learned_books is not None:
                # SM32c: the learned catalogs train in their own AdamW group (after the source optimizer state is
                # loaded, so that file keeps its group layout); the loop sets every group's LR to the schedule.
                if not isinstance(optimizer, torch.optim.AdamW):
                    raise ValueError("state_vq learn: the catalog group needs an AdamW optimizer")
                optimizer.add_param_group({"params": list(learned_books.parameters()), "weight_decay": 0.0})
                report["state_vq"]["learned_parameters"] = sum(p.numel() for p in learned_books.parameters())
        update_offset = int(full_run.get("update_offset", 0))
        if options.get("group_adam"):
            from scripts.gpu_r01_group_adam import install_group_adam
            install_group_adam(optimizer, reusable=options["group_adam"] == 2)
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
        # F74 T5 (pending approval, off): a frozen protocol may shorten the discarded primer.
        for _ in range((protocol.get("turnaround") or {}).get("primer_passes") or 3):
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
            if options.get("flat_buffers"):
                from scripts.gpu_r01_flat import FlatMasterChildSync
                grouped = FlatMasterChildSync(master, base, optimizer)
            else:
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
        restore_affinity = None
        if options.get("host_placement"):
            if warm_pool is None:
                raise ValueError("F36/F71 host placement needs the prestarted worker pool")
            if options["host_placement"] == "physical":
                report["host_placement"], restore_affinity = place_host_cores(report["prestarted_worker_pids"])
            else:
                report["host_placement"] = place_host_threads(report["prestarted_worker_pids"])
        if options.get("process_qos"):
            if warm_pool is None:
                raise ValueError("F86 process QoS needs the prestarted worker pool")
            if options.get("main_priority", "high") != "high":
                raise ValueError("F86b main_priority supports only 'high'")
            report["process_qos"] = set_host_qos(report["prestarted_worker_pids"], options.get("main_priority"),
                                                 options.get("worker_qos"))
        elif options.get("worker_qos"):
            raise ValueError("F93b worker_qos needs F86 process_qos")
        # Host garbage-collection time inside the training loop (accounting only).
        gc_totals, gc_started = {0: [0, 0.0], 1: [0, 0.0], 2: [0, 0.0]}, {}

        def gc_account(phase, info):
            generation = info["generation"]
            if phase == "start":
                gc_started[generation] = time.perf_counter()
            elif generation in gc_started:
                gc_totals[generation][0] += 1
                gc_totals[generation][1] += time.perf_counter() - gc_started.pop(generation)

        gc.callbacks.append(gc_account)
        train_started = time.perf_counter()
        report["training_started_utc"] = datetime.now(timezone.utc).isoformat()
        last_user = None
        previous_compute = None
        origin = torch.cuda.Event(enable_timing=True)
        origin.record()
        coverage = {}
        pool_context = prestarted_pool(pool_stack, warm_pool) if warm_pool is not None else preparation_pool(workers)
        with limited_gpu(0, 0, path), pool_context as pool, (path / "updates.jsonl").open("w", encoding="utf-8") as log:
            if chunk_reviews > 191:
                # A sample holds its reviews plus query rows (up to about 2x); scripts/check_long_wkv.py covers it.
                from rwkv.model import nvrtc as nvrtc_limits
                report["nvrtc_max_timesteps"] = 4 * chunk_reviews
                kernel_stack.callback(nvrtc_limits.set_max_timesteps, nvrtc_limits.set_max_timesteps(4 * chunk_reviews))
            source = training_batches(rows, chunks * chunk_reviews, training_seed, pool, options, cache_paths, workers * 2)
            transferred = transfer_batches(source, dtype, options.get("async_transfer", False),
                                           **({"bulk": True} if options.get("bulk_transfer") else {}))
            deferred = options.get("deferred_reads", False)
            # F33b: number of queued updates whose loss/norm reads stay outstanding (1 = original F33).
            read_depth = int(options.get("deferred_read_depth", 1))
            if read_depth < 1 or (read_depth > 1 and not deferred):
                raise ValueError("F33b deferred_read_depth needs F33 deferred reads and a positive depth")
            report["deferred_read_depth"] = read_depth
            pending = []

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
                # The values above were read after this update's end event, so both events have completed.
                timing_start, timing_end, host_started, host_dispatch = update["timing"]
                transfer_metrics.update(host_started_s=host_started, host_dispatch_s=host_dispatch,
                                        gpu_started_ms=origin.elapsed_time(timing_start),
                                        gpu_ms=timing_start.elapsed_time(timing_end))
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

            # F72a: reset gradients through cached parameter lists and set train mode once, instead of
            # walking the scripted module tree three times per update. Same None gradients and flags.
            lean_reset = bool(options.get("lean_step_reset"))
            if lean_reset:
                if graphs:
                    raise ValueError("F72a lean_step_reset does not support graph adapters")
                lean_master = [p for group in optimizer.param_groups for p in group["params"]]
                lean_child = list(base.parameters())
                base.train()
                report["lean_step_reset"] = {"master_parameters": len(lean_master), "child_parameters": len(lean_child)}
            bookkeeping_stride = int(options.get("bookkeeping_stride", 1))
            if bookkeeping_stride < 1:
                raise ValueError("F57 bookkeeping_stride must be a positive integer")
            report["bookkeeping_stride"] = bookkeeping_stride
            transitions = 0
            # Full runs: weight snapshots every `snapshot_users` users, on a SAVE request, and before a STOP request
            # (control files in the run directory), so an interrupted epoch can resume from its latest snapshot.
            snapshot_users = int(full_run.get("snapshot_users", 0))
            control = ROOT / full_run["control_dir"] if full_run.get("control_dir") else None
            row_end = {row["user_id"]: row["train_range"][1] for row in rows}

            def save_snapshot(reason):
                folder = path / "snapshots" / f"users-{report['progress_users']:05d}"
                folder.mkdir(parents=True, exist_ok=True)
                torch.save(master.state_dict(), folder / "master.pth")
                torch.save(optimizer.state_dict(), folder / "optimizer.pth")
                if learned_books is not None:
                    learned_books.export(folder / "state-vq-books.npz")
                done = sorted(int(user) for user, record in coverage.items() if record["reviews"] == row_end[int(user)])
                meta = {"reason": reason, "utc": datetime.now(timezone.utc).isoformat(), "progress_users": report["progress_users"],
                        "updates": report["updates"], "update_offset_end": update_offset + report["updates"],
                        "review_presentations": report["review_presentations"], "done_users": done,
                        "folder": folder.relative_to(ROOT).as_posix()}
                atomic_json(folder / "snapshot.json", meta)
                atomic_json(path / "snapshots" / "latest.json", meta)
                report["latest_snapshot"] = {key: meta[key] for key in ("reason", "utc", "progress_users", "folder")}
                atomic_json(path / "result.json", report)
            try:
                for row, item, batch in transferred:
                    # Per-update timing (accounting only): host start and device span, read with the update's values.
                    update_started = time.perf_counter()
                    timing_start = torch.cuda.Event(enable_timing=True)
                    timing_start.record()
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
                        transitions += 1
                        # F57: bookkeeping stride; RSS and progress JSON every `stride` users (1 = every user).
                        if transitions % bookkeeping_stride == 1 or bookkeeping_stride == 1:
                            import psutil
                            process = psutil.Process()
                            rss, skipped = sampled_tree_rss(process)
                            report["rss_sampling_skipped_processes"] = report.get("rss_sampling_skipped_processes", 0) + skipped
                            report["peak_process_tree_rss_mib"] = max(report["peak_process_tree_rss_mib"], rss)
                        if last_user is not None:
                            report["progress_users"] += 1
                            if report["progress_users"] % bookkeeping_stride == 0:
                                report["training_elapsed_seconds"] = time.perf_counter() - train_started
                                atomic_json(path / "result.json", report)
                            if full_run:
                                if control is not None and (control / "STOP").exists():
                                    save_snapshot("stop")
                                    raise InterruptedError("STOP requested; snapshot saved, `run` resumes from it")
                                if control is not None and (control / "SAVE").exists():
                                    (control / "SAVE").unlink(missing_ok=True)
                                    save_snapshot("save")
                                elif snapshot_users and report["progress_users"] % snapshot_users == 0:
                                    save_snapshot("periodic")
                        last_user = part_user
                    cpu_labels = item.batch.labels
                    curve_n = int(((1 - cpu_labels[:, :, 6]) * cpu_labels[:, :, 4]).sum())
                    instant_n = int((cpu_labels[:, :, 6] * cpu_labels[:, :, 4]).sum())
                    if curve_n + instant_n == 0:
                        raise RuntimeError(f"Unlabelled batch user {user}, offset {item.offset}")
                    torch.manual_seed(item.seed)
                    active = graphs.select(batch) if graphs else base
                    if options.get("worker_scatter_orders"):
                        from scripts.gpu_scatter_orders import install_orders
                        install_orders(batch)
                    # Graph construction creates a fresh model and consumes RNG.
                    # Reset after admission so an eager or replayed update sees
                    # the same declared dropout seed as its paired control.
                    torch.manual_seed(item.seed)
                    if lean_reset:
                        for parameter in lean_master:
                            parameter.grad = None
                    else:
                        optimizer.zero_grad(set_to_none=True)
                    if options.get("async_transfer"):
                        compute_started = torch.cuda.Event(enable_timing=True)
                        compute_started.record()
                    if lean_reset and active is base:
                        for parameter in lean_child:
                            parameter.grad = None
                    else:
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
                        if catch_up is not None and catch_up.carry is not None:
                            catch_up.carry.attach(item, batch, user)  # SM21: entity rows; table reset per user
                        stats = (reduced.get_loss(batch) if reduced else
                                 catch_up.loss(active, batch, report["review_presentations"],
                                               (user, item.offset, item.reviews, item.seed) if len(parts) == 1 else None,
                                               cpu_labels) if catch_up else
                                 active.get_loss(batch))
                    if stats is None or (not deferred and not torch.isfinite(stats.average_loss)):
                        raise RuntimeError(f"Non-finite training loss user {user}")
                    wddm_flush = options.get("wddm_flush")
                    if wddm_flush:
                        # F77a: cudaStreamQuery submits the WDDM software queue, so the GPU starts the
                        # forward while the host records the backward. Scheduling only; no arithmetic change.
                        torch.cuda.current_stream().query()
                    if not graphs or active is base:
                        stats.average_loss.backward()
                    if wddm_flush:
                        torch.cuda.current_stream().query()
                    if options.get("concurrent_splits"):
                        from expertium.model.srs_model import wait_split_streams
                        wait_split_streams()
                    if grouped:
                        grouped.transfer_grads()
                    else:
                        transfer_child_grad_to_master(master, active)
                    # F33 checks the same finite loss/norm after the next update is queued.
                    norm = torch.nn.utils.clip_grad_norm_(master.parameters(), CLIP, error_if_nonfinite=not deferred)
                    lr = (catch_up.learning_rate(report["updates"], update_offset, report["review_presentations"])
                          if catch_up else 7e-4 * min(1.0, (report["updates"] + 1 + update_offset) / 5))
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
                    timing_end = torch.cuda.Event(enable_timing=True)
                    timing_end.record()
                    update = {"step": report["updates"], "user": user, "item": item, "curve_n": curve_n,
                              "instant_n": instant_n, "lr": lr, "parts": parts,
                              "compute": (compute_started, compute_ended) if options.get("async_transfer") else None,
                              "timing": (timing_start, timing_end, update_started - train_started,
                                         time.perf_counter() - update_started)}
                    if deferred:
                        # F33: queue a pinned copy of loss/norm and read it once the next update is queued.
                        values = torch.stack((stats.average_loss.detach().float(), norm.detach().float()))
                        update["host"] = torch.empty(2, dtype=torch.float32, pin_memory=True)
                        update["host"].copy_(values, non_blocking=True)
                        update["ready"] = torch.cuda.Event()
                        update["ready"].record()
                        pending.append(update)
                        while len(pending) > read_depth:
                            finish(pending.pop(0))
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
                while pending:
                    finish(pending.pop(0))
            finally:
                transferred.close()
                source.close()
        torch.cuda.synchronize()
        report["training_seconds"] = time.perf_counter() - train_started
        if restore_affinity is not None:
            # F71: the evaluation pool would inherit the main-process cores; return to the original set.
            import psutil
            psutil.Process().cpu_affinity(restore_affinity)
            report["host_placement"]["restored_main_process_cpus"] = len(restore_affinity)
        gc.callbacks.remove(gc_account)
        report["training_gc"] = {f"generation_{g}": {"collections": n, "seconds": t} for g, (n, t) in gc_totals.items()}
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
        allocator = torch.cuda.memory_stats()
        report["allocator_stats"] = {key: allocator.get(key) for key in
                                     ("num_alloc_retries", "num_ooms", "num_device_alloc", "num_device_free")}
        if graphs:
            report["graphs"] = graphs.metrics
        report["trained_parameter_sha256"] = tensor_hash(master.state_dict())
        watch.check()
        # Preserve weights in a new experiment location, never the existing trial.
        torch.save(master.state_dict(), path / "master.pth")
        if learned_books is not None:
            report["state_vq"]["learned_books"] = {"file": "state-vq-books.npz",
                                                   "sha256": learned_books.export(path / "state-vq-books.npz")}
        if full_run:
            torch.save(optimizer.state_dict(), path / "optimizer.pth")
            report["update_offset_end"] = update_offset + report["updates"]
        active = graphs = reduced = base = optimizer = stats = batch = item = None
        master.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        nvrtc.set_checkpoint_length(32)
        nvrtc.set_raw_pointer_launch(False)
        nvrtc.set_stream_launch(False)
        nvrtc.set_driver_launch(False)
        exact_index.set_shared_orders(False)
        exact_index.set_prefetch_rows(False)
        exact_index.set_prefetch_shift(False)
        if full_run and not full_run.get("evaluate", True):
            report["status"] = "complete"
            report["case_elapsed_seconds"] = time.perf_counter() - started
            atomic_json(path / "result.json", report)
            return 0
        eval_started = time.perf_counter()
        per_user = []
        validation_hash = hashlib.sha256()
        report["status"] = "evaluating"
        atomic_json(path / "result.json", report)
        progress = Progress(f"{args.model}/{args.track}", len(evaluation_rows), 0)
        # F74 T1: an exact replica may reuse an evaluated origin's scores after the fail-closed F67 proof.
        from scripts import r01_evaluation_reuse as reuse
        origin = reuse.origin_request(os.environ.get("GPU_R01_CONTROLLER_ROOT")) \
            if (protocol.get("turnaround") or {}).get("evaluation_reuse") == "exact_replica" else None
        if isinstance(origin, dict):
            report["evaluation_reuse"], origin = origin, None
        elif origin is not None:
            reasons = reuse.precheck(report, origin)
            if reasons:
                report["evaluation_reuse"], origin = {"status": "refused", "reasons": reasons, "origin": str(origin)}, None

        def heldout_batches(row, evaluation_pool):
            user = row["user_id"]
            section = split_frame(get_rwkv_data(DATA, user), row, "validation")
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
                yield batch
        # EV01 (docs/EVAL_SPEED.md): opt-in merged-batch evaluation of full runs, same inputs and per-section sums.
        fast = bool(full_run.get("fast_evaluation") and full_run.get("benchmark_db")) and origin is None
        if fast:
            from scripts.fast_evaluation import evaluate_fast, supports_fast
            fast = supports_fast(master)
            report["evaluation_mode"] = "fast" if fast else "reference (fast evaluation unsupported for this model)"
        if full_run:
            report.setdefault("evaluation_mode", "reference")
        if fast:
            report["evaluation_batch_chunks"] = int(full_run.get("fast_evaluation_chunks", 512))

            def fast_progress(done, total):
                progress.evaluation(done, total)
                if done % 20 == 0:
                    report["evaluation_progress_users"] = done
                    atomic_json(path / "result.json", report)
            per_user, report["fast_evaluation"], _ = evaluate_fast(
                master, evaluation_rows, ROOT / full_run["benchmark_db"], workers=workers, progress=fast_progress,
                batch_chunks=int(full_run.get("fast_evaluation_chunks", 512)),
                db_size=full_run.get("benchmark_db_size", 10_000_000_000), input_hash=True)
            watch.check()
        with preparation_pool(0 if fast else workers if options.get("parallel_evaluation") else 0) as evaluation_pool:
            if origin is not None:
                # Prepare and hash the held-out inputs exactly as evaluation would, without the GPU pass.
                for row in evaluation_rows:
                    watch.check()
                    for _ in heldout_batches(row, evaluation_pool):
                        pass
                report["validation_input_sha256"] = validation_hash.hexdigest()
                report["evaluation_context"] = reuse.evaluation_context(ROOT, report, provenance)
                try:
                    proof = reuse.prove_reuse(report, protocol, origin, json.loads((origin.parents[1] / "protocol.json").read_text()),
                                              candidate_checkpoint=path / "master.pth")
                except Exception as exc:  # noqa: BLE001 - any proof error falls back to a normal evaluation
                    proof = {"status": "refused", "reasons": [f"{type(exc).__name__}: {exc}"]}
                if proof["status"] == "reusable":
                    per_user = proof["per_user"]
                    report["evaluation_reuse"] = {"status": "reused", "label": "held-out scores reused from an exact replica",
                                                  "proof_seconds": time.perf_counter() - eval_started,
                                                  "loss_provenance": proof["loss_provenance"]}
                else:
                    report["evaluation_reuse"] = {"status": "refused", "reasons": proof["reasons"], "origin": str(origin),
                                                  "proof_seconds": time.perf_counter() - eval_started}
                    validation_hash = hashlib.sha256()
                    report.pop("evaluation_context")
            for row in evaluation_rows if not fast and (report.get("evaluation_reuse") or {}).get("status") != "reused" else ():
                watch.check()
                if full_run.get("benchmark_db"):
                    # Full runs also score the upstream benchmark test reviews (LogLoss, RMSE(bins), AUC).
                    from scripts.full_run_metrics import benchmark_metrics, evaluate_with_predictions
                    values, predictions = evaluate_with_predictions(
                        master, ({"batch": batch.to(device)} for batch in heldout_batches(row, evaluation_pool)))
                    values["benchmark"] = benchmark_metrics(row["user_id"], ROOT / full_run["benchmark_db"],
                                                            full_run.get("benchmark_db_size", 10_000_000_000), predictions)
                    del predictions
                else:
                    values = evaluate(master, ({"batch": batch.to(device)} for batch in heldout_batches(row, evaluation_pool)),
                                      deferred_metric_reads=bool(options.get("deferred_evaluation_metrics", False)))
                per_user.append({"user_id": row["user_id"], **values})
                progress.evaluation(len(per_user), len(evaluation_rows))
                if full_run and len(per_user) % 20 == 0:
                    report["evaluation_progress_users"] = len(per_user)
                    atomic_json(path / "result.json", report)
        progress.end()
        report["validation_seconds"] = time.perf_counter() - eval_started
        report["validation_input_sha256"] = report["fast_evaluation"]["validation_input_sha256"] if fast else validation_hash.hexdigest()
        if "evaluation_context" not in report:
            report["evaluation_context"] = reuse.evaluation_context(ROOT, report, provenance)
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
