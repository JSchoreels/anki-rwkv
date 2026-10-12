"""Compare preparation, prefetch, batch size, and precision on real histories."""

from argparse import ArgumentParser, Namespace
import copy
import gc
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import pandas as pd
from rwkv import data_processing as processing
from rwkv.train_rwkv import CLIP, get_optimizer, transfer_child_grad_to_master
from scripts.trial_batches import batches, preparation_pool
from train_gpu_trial import training_models
from gpu_power import limited_gpu
from scripts.experiment_provenance import process_provenance


def benchmark_models(config, device):
    if config.get("MODEL", "baseline") == "baseline":
        return training_models(config, device)
    from rwkv.model.nvrtc import register
    from scripts.compare import SMALL_CONFIG, SmallModel, training_models as make_models
    register()
    dtype = getattr(torch, config["DTYPE"])
    master, model = make_models(SmallModel, SMALL_CONFIG, device, dtype, config["SEED"])
    return master, model, dtype


def fresh_checkpoint(model_name, seed=2025):
    from scripts.compare import BASELINE_CONFIG, SMALL_CONFIG, BaselineModel, SmallModel
    torch.manual_seed(seed)
    model_class, architecture = ((BaselineModel, BASELINE_CONFIG) if model_name == "baseline"
                                else (SmallModel, SMALL_CONFIG))
    master = model_class(architecture).float()
    optimizer = get_optimizer(Namespace(PEAK_LR=7e-4), master)
    return {"model": master.state_dict(), "optimizer": optimizer.state_dict(),
            "summary": {"steps": 0, "parameters": sum(p.numel() for p in master.parameters())}}


def legacy_locations(frame, submodule):
    grouped = frame.groupby(submodule, observed=True)
    return [grouped.get_group(name)["index"].values for name in set(frame[submodule].values)]


def update(master, model, optimizer, item, dtype, device, fast_copies=False):
    torch.manual_seed(item.seed)
    batch = item.batch.to(device)
    batch.start = batch.start.to(dtype)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    if model is not master:
        model.zero_grad(set_to_none=True)
        if fast_copies:
            from scripts.gpu_optimizations import copy_downcast_fast
            copy_downcast_fast(master, model)
        else:
            model.copy_downcast_(master, dtype)
    graph_update = getattr(model, "graph_update", None)
    stats = graph_update(batch, item.seed) if graph_update else model.get_loss(batch)
    if stats is None or not torch.isfinite(stats.average_loss) or stats.ahead_n + stats.imm_n == 0:
        raise RuntimeError("Invalid benchmark training loss or labels")
    if not graph_update:
        stats.average_loss.backward()
    if model is not master:
        if fast_copies:
            from scripts.gpu_optimizations import transfer_grads_fast
            transfer_grads_fast(master, model)
        else:
            transfer_child_grad_to_master(master, model)
    norm = torch.nn.utils.clip_grad_norm_(master.parameters(), CLIP, error_if_nonfinite=True)
    optimizer.step()
    return stats.average_loss.item(), norm.item()


def run_variant(name, frame, config, checkpoint, output, warmup, steps, pool=None, profile=False,
                after_training=None):
    provenance = process_provenance()
    device = torch.device("cuda:0")
    from rwkv.model import nvrtc
    nvrtc.set_checkpoint_length(config.get("CHECKPOINT_LENGTH", 32))
    nvrtc.set_raw_pointer_launch(config.get("RAW_POINTERS", False) or config.get("CUDA_GRAPH", False))
    torch.manual_seed(config.get("SEED", 2025))
    master, model, dtype = benchmark_models(config, device)
    master.load_state_dict(checkpoint["model"])
    if model is not master:
        model.copy_downcast_(master, dtype)
    optimizer = get_optimizer(Namespace(PEAK_LR=7e-4), master)
    # AdamW's CPU scalar step tensors can alias a loaded state dictionary.
    # Every scenario must begin with an independent copy of that same state.
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    if config.get("FUSED_ADAMW"):
        if optimizer.state:
            raise ValueError("Fused optimizer experiment requires fresh initialization")
        groups = [{**group, "fused": True, "foreach": None} for group in optimizer.param_groups]
        optimizer = torch.optim.AdamW(groups, eps=optimizer.defaults["eps"],
                                      betas=optimizer.defaults["betas"], fused=True)
    if config.get("CUDA_GRAPH"):
        from scripts.gpu_optimizations import GraphTrainingAdapter
        model = GraphTrainingAdapter(model)
    def perform_update(item):
        return update(master, model, optimizer, item, dtype, device, config.get("FAST_COPIES", False))
    span = config["CHUNK_REVIEWS"] * config["BATCH_CHUNKS"]
    selection = frame.iloc[:(warmup + steps + 1) * span]
    if len(selection) < (warmup + steps + 1) * span:
        raise ValueError("Benchmark selection is too short")
    iterator = iter(batches(selection, 17, 0, 0, config, pool))
    try:
        for _ in range(warmup):
            perform_update(next(iterator))
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        durations, preparation, waits, train_times = [], [], [], []
        for _ in range(steps):
            started = time.perf_counter()
            item = next(iterator)
            prepared = time.perf_counter()
            loss, norm = perform_update(item)
            torch.cuda.synchronize(device)
            ended = time.perf_counter()
            durations.append(ended - started)
            waits.append(prepared - started)
            preparation.append(item.preparation_seconds)
            train_times.append(ended - prepared)
        result = {"name": name, "model": config.get("MODEL", "baseline"),
                  "provenance": provenance,
                  "parameters": sum(p.numel() for p in master.parameters()),
                  "batch_chunks": config["BATCH_CHUNKS"], "dtype": config["DTYPE"],
                  "prefetch_workers": config["PREFETCH_WORKERS"], "measured_steps": steps,
                  "reviews_per_update": span, "reviews_per_second": steps * span / sum(durations),
                  "median_seconds_per_update": statistics.median(durations),
                  "median_preparation_seconds": statistics.median(preparation),
                  "median_wait_seconds": statistics.median(waits),
                  "median_training_seconds": statistics.median(train_times),
                  "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                  "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
                  "last_loss": loss, "last_gradient_norm": norm,
                  "experiment": {key: config.get(key) for key in ("PREPARATION", "CHECKPOINT_LENGTH", "FUSED_ADAMW", "CUDA_GRAPH", "FAST_COPIES", "RAW_POINTERS")},
                  "training_only_reviews_per_second": steps * span / sum(train_times),
                  "update_seconds": durations}
        if config.get("CUDA_GRAPH"):
            result["memory_reporting_note"] = "Use peak_reserved_mib: replay retains private-pool buffers after their temporary tensors are released"
        if profile:
            item = next(iterator)
            started = time.perf_counter()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                perform_update(item)
                torch.cuda.synchronize(device)
            result["profiled_update_wall_seconds"] = time.perf_counter() - started
            trace_path = output / f"{name}-trace.json"
            profiler.export_chrome_trace(str(trace_path))
            events = json.loads(trace_path.read_text())["traceEvents"]
            kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
            result["profiled_cuda_kernel_seconds"] = sum(e["dur"] for e in kernels) / 1e6
            result["profiled_cuda_kernel_count"] = len(kernels)
            if kernels:
                intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in kernels)
                busy, begin, end = 0, *intervals[0]
                for a, b in intervals[1:]:
                    if a > end:
                        busy += end - begin
                        begin, end = a, b
                    else:
                        end = max(end, b)
                busy += end - begin
                window = max(b for a, b in intervals) - min(a for a, b in intervals)
                result["profiled_kernel_busy_fraction"] = busy / window
        if after_training is not None:
            # Release graph pools before evaluating the updated FP32 master.
            # Evaluation is outside all benchmark timings.
            iterator.close()
            model = optimizer = None
            master.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            result["quality"] = after_training(master)
    finally:
        iterator.close()
        del optimizer, model, master
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps({k: v for k, v in result.items() if k != "update_seconds"}), flush=True)
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prefetch-only", action="store_true")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"))
    parser.add_argument("--batch-chunks", type=int)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--model", choices=("baseline", "expertium"), default="baseline")
    parser.add_argument("--fresh", action="store_true", help="Start every scenario from seeded fresh weights")
    parser.add_argument("--memory-fraction", type=float, default=0.72)
    parser.add_argument("--output-dir", default="artifacts/benchmarks/perf-trial")
    args = parser.parse_args()
    provenance = process_provenance()
    if args.batch_chunks is not None and args.batch_chunks <= 0:
        parser.error("--batch-chunks must be positive")
    if not 0 < args.memory_fraction <= 1:
        parser.error("--memory-fraction must be in (0, 1]")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    fresh = args.fresh or args.model == "expertium"
    checkpoint = (fresh_checkpoint(args.model) if fresh else
                  torch.load(ROOT / "checkpoints/gpu-trial/latest.pth", weights_only=True, map_location="cpu"))
    frame = processing.get_rwkv_data(ROOT / "data/anki-revlogs-10k", 17)
    # Reuse the same 64 complete history chunks at every batch size. A longer
    # window also exposes queue starvation instead of timing only prefetched work.
    common = frame.iloc[4096:4096 + 64 * 128]
    needed = (args.warmup + args.steps + 1) * max(32, args.batch_chunks or 32) * 128
    frame = pd.concat([common] * ((needed + len(common) - 1) // len(common)), ignore_index=True)
    config = {"WKV_BACKEND": "nvrtc", "CHUNK_REVIEWS": 128, "REVIEWS_PER_USER": 0,
              "SEED": 2025, "PREFETCH_WORKERS": 0, "PREFETCH_BATCHES": 4, "DTYPE": "float32",
              "MODEL": args.model}
    results = []
    optimized_locations = processing._module_group_locations
    with limited_gpu(0, 0, output):
        if not args.prefetch_only:
            try:
                processing._module_group_locations = legacy_locations
                results.append(run_variant("original-8-fp32", frame, {**config, "BATCH_CHUNKS": 8},
                                           checkpoint, output, args.warmup, args.steps))
            finally:
                processing._module_group_locations = optimized_locations
            results.append(run_variant("indices-8-fp32", frame, {**config, "BATCH_CHUNKS": 8},
                                       checkpoint, output, args.warmup, args.steps))
        with preparation_pool(args.workers) as pool:
            for dtype in ([args.dtype] if args.dtype else ("float32", "bfloat16")):
                for chunks in ([args.batch_chunks] if args.batch_chunks else (8, 16, 32)):
                    scenario = {**config, "PREFETCH_WORKERS": args.workers, "PREFETCH_BATCHES": max(4, args.workers * 2),
                                "BATCH_CHUNKS": chunks, "DTYPE": dtype}
                    name = f"prefetch-{chunks}-{dtype}" + (f"-{args.workers}-workers" if args.workers != 2 else "")
                    if fresh:
                        name += f"-{args.model}-fresh"
                    try:
                        results.append(run_variant(name, frame, scenario,
                            checkpoint, output, args.warmup, args.steps, pool,
                            profile=args.profile or chunks == 32))
                    except torch.OutOfMemoryError as exc:
                        results.append({"name": name, "status": "out_of_memory", "error": str(exc),
                                        "batch_chunks": chunks, "dtype": dtype, "model": args.model})
                        exc.__traceback__ = None
                        gc.collect()
                        torch.cuda.empty_cache()
                        print(f"{name}: exceeded allocator memory budget", flush=True)
    report = {"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
              "provenance": provenance,
              "checkpoint_provenance": checkpoint["summary"].get("provenance"),
              "cuda": torch.version.cuda, "checkpoint_step": checkpoint["summary"]["steps"],
              "parameters": checkpoint["summary"]["parameters"], "model": args.model,
              "initialization": "fresh-seed-2025" if fresh else "training-checkpoint",
              "gpu_memory_fraction": args.memory_fraction, "user": 17,
              "repeated_common_window_reviews": len(common),
              "warmup_steps": args.warmup, "results": results,
              "best": max((r for r in results if "reviews_per_second" in r),
                          key=lambda r: r["reviews_per_second"], default={"name": None})["name"]}
    filename = f"benchmark-{args.workers}-workers.json" if args.prefetch_only else "benchmark.json"
    if args.prefetch_only and args.batch_chunks is not None:
        filename = f"benchmark-{args.workers}-workers-{args.batch_chunks}-{args.dtype or 'both'}.json"
    if fresh:
        filename = filename.removesuffix(".json") + f"-{args.model}-fresh.json"
    (output / filename).write_text(json.dumps(report, indent=2) + "\n")
    print(f"Best measured pipeline: {report['best']}", flush=True)


if __name__ == "__main__":
    main()
