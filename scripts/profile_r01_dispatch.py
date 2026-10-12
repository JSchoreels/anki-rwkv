"""Main-thread dispatch profile of the accepted R01 update path; not a training run.

Reuses the fixed component-screen windows and preprepared GPU batches, then
attributes host time per update stage (no added synchronization, so each stage is
the time the training thread spends issuing it; host reads absorb GPU waits),
counts CUDA launches/ATen dispatches with torch.profiler and ranks Python
functions with cProfile. Results select candidates; they do not replace
full-history timing or quality.
"""

from argparse import ArgumentParser, Namespace
import cProfile
import io
import json
import os
from pathlib import Path
import pstats
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--options", type=Path, required=True, help="JSON options of the accepted case (selection.json entry)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--extra-options", help="JSON options merged over the selection entry")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Output must remain under workspace artifacts")
    if output.exists():
        raise FileExistsError("Preserve the existing profile")
    options = json.loads((ROOT / args.options).read_text(encoding="utf-8"))
    options = options.get("expertium", options).get("options", options)
    if args.extra_options:
        options = {**options, **json.loads(args.extra_options)}
    from scripts.benchmark_r01_components import windows
    data_path, selected = windows("expertium")
    os.environ["GPU_BATCH1_DATA_PATH"] = str(ROOT / data_path)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["CUPY_CACHE_DIR"] = str(ROOT / "artifacts/cache/cupy-cache")
    import torch
    from scripts.experiment_provenance import process_provenance
    provenance = process_provenance()
    torch._C._set_graph_executor_optimize(False)
    torch.use_deterministic_algorithms(True, warn_only=False)
    from rwkv.model import nvrtc
    from rwkv.train_rwkv import CLIP, get_optimizer
    from rwkv.data_processing import get_rwkv_data
    from scripts.benchmark_gpu_trial import benchmark_models, fresh_checkpoint
    from scripts.gpu_batch1 import split_frame
    from scripts.gpu_b01 import prepare_job, map_batch, materialize
    from scripts.gpu_r01_update import MasterChildSync

    if options.get("row_tiles_exact"):
        # F31: hold the pinned row-tiled recurrence for the whole profiled process.
        from contextlib import ExitStack
        from scripts.gpu_row_tiles import row_tiled_backend
        kernels = ExitStack()
        kernels.enter_context(row_tiled_backend(rows=options["row_tiles_exact"], tile=32, exact=True))
    span = options["batch_chunks"] * 128
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(options["gpu_memory_fraction"], 0)
    nvrtc.register()
    nvrtc.set_checkpoint_length(32)
    device = torch.device("cuda:0")
    config = {"MODEL": "expertium", "DTYPE": "bfloat16", "SEED": 2025, "WKV_BACKEND": "nvrtc"}
    prepared = []
    for row, length in selected:
        frame = split_frame(get_rwkv_data(Path(os.environ["GPU_BATCH1_DATA_PATH"]), row["user_id"]), row, "train")
        for offset in range(0, length, span):
            end = min(length, offset + span)
            prepared.append(prepare_job(row, offset, end, 2025 + row["user_id"] + offset, options, frame.iloc[offset:end]))
    checkpoint = fresh_checkpoint("expertium", 2025)
    master, base, dtype = benchmark_models(config, device)
    master.load_state_dict(checkpoint["model"])
    base.copy_downcast_(master, dtype)
    for module in base.modules():
        if hasattr(module, "exact_shift"):
            module.exact_shift = options.get("exact_shift", False)
        if hasattr(module, "exact_index"):
            module.exact_index = options.get("exact_index", False)
        if hasattr(module, "exact_group_norm"):
            module.exact_group_norm = options.get("exact_group_norm", False)
    optimizer = get_optimizer(Namespace(PEAK_LR=7e-4), master)
    grouped = MasterChildSync(master, base)
    torch.set_num_threads(options.get("main_threads", 4))
    reviews = sum(item.reviews for item in prepared)
    stages = {}
    updates = 0

    def timed(name, function):
        begin = time.perf_counter()
        value = function()
        stages[name] = stages.get(name, 0.0) + time.perf_counter() - begin
        return value

    def update(item):
        nonlocal updates
        # Same order as gpu_r01_training: host label counts, then transfer/expansion, then the update.
        labels = item.batch.labels
        timed("host label counts", lambda: (int(((1 - labels[:, :, 6]) * labels[:, :, 4]).sum()),
                                            int((labels[:, :, 6] * labels[:, :, 4]).sum())))
        batch = timed("transfer and GPU feature expansion",
                      lambda: materialize(map_batch(item.batch, lambda x: x.to(device, non_blocking=False)), dtype))
        torch.manual_seed(item.seed)
        timed("zero grads/train mode", lambda: (optimizer.zero_grad(set_to_none=True), base.zero_grad(set_to_none=True), base.train()))
        timed("master to child downcast", grouped.downcast)
        stats = timed("forward and loss", lambda: base.get_loss(batch))
        deferred = options.get("deferred_checks", False)
        if not deferred:
            timed("finite-loss check", lambda: bool(torch.isfinite(stats.average_loss)))
        timed("backward", stats.average_loss.backward)
        timed("child to master gradients", grouped.transfer_grads)
        norm = timed("gradient clipping", lambda: torch.nn.utils.clip_grad_norm_(master.parameters(), CLIP, error_if_nonfinite=not deferred))
        updates += 1
        for group in optimizer.param_groups:
            group["lr"] = 7e-4 * min(1.0, updates / 5)
        timed("optimizer step", optimizer.step)
        if deferred:
            # F33: one host read after the update carries loss, norm and both finite checks.
            return timed("host loss/norm reads", lambda: torch.stack((stats.average_loss.detach().float(), norm.float())).tolist())
        return timed("host loss/norm reads", lambda: (stats.average_loss.item(), norm.item()))

    result = {"options": options, "provenance": provenance, "batches_per_pass": len(prepared), "reviews_per_pass": reviews,
              "windows": [{"user_id": row["user_id"], "reviews": length} for row, length in selected],
              "scope": "accepted update path on fixed windows; host transfers synchronous here (asynchronous in training); first pass warms JIT and is excluded",
              "passes": []}
    for _ in range(1):
        for item in prepared:
            update(item)
    torch.cuda.synchronize()
    stages.clear()
    begin = time.perf_counter()
    for _ in range(args.passes):
        for item in prepared:
            update(item)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - begin
    count = args.passes * len(prepared)
    result["timed_updates"] = count
    result["wall_ms_per_update"] = 1000 * elapsed / count
    result["reviews_per_second"] = reviews * args.passes / elapsed
    result["host_stage_ms_per_update"] = {name: 1000 * value / count for name, value in sorted(stages.items(), key=lambda kv: -kv[1])}

    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        for item in prepared:
            update(item)
        torch.cuda.synchronize()
    averages = profiler.key_averages()
    per = len(prepared)
    rows = [{"name": event.key, "calls_per_update": event.count / per,
             "self_cpu_ms_per_update": event.self_cpu_time_total / 1000 / per,
             "device_ms_per_update": getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0)) / 1000 / per}
            for event in averages]
    result["profiler_top_self_cpu"] = sorted(rows, key=lambda r: -r["self_cpu_ms_per_update"])[:40]
    result["profiler_top_device"] = sorted(rows, key=lambda r: -r["device_ms_per_update"])[:25]
    launches = [r for r in rows if r["name"] in ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernelEx")]
    result["launches_per_update"] = sum(r["calls_per_update"] for r in launches)
    result["launch_cpu_ms_per_update"] = sum(r["self_cpu_ms_per_update"] for r in launches)
    result["aten_dispatches_per_update"] = sum(r["calls_per_update"] for r in rows if r["name"].startswith("aten::"))
    result["device_ms_per_update"] = sum(r["device_ms_per_update"] for r in rows)

    python_profile = cProfile.Profile()
    python_profile.enable()
    for item in prepared:
        update(item)
    torch.cuda.synchronize()
    python_profile.disable()
    stream = io.StringIO()
    pstats.Stats(python_profile, stream=stream).sort_stats("tottime").print_stats(45)
    result["cprofile_tottime_top"] = stream.getvalue().splitlines()
    stream = io.StringIO()
    pstats.Stats(python_profile, stream=stream).sort_stats("cumulative").print_stats(60)
    result["cprofile_cumulative_top"] = stream.getvalue().splitlines()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("wall_ms_per_update", "reviews_per_second", "host_stage_ms_per_update",
                                                    "launches_per_update", "launch_cpu_ms_per_update", "aten_dispatches_per_update",
                                                    "device_ms_per_update")}, indent=1))


if __name__ == "__main__":
    main()
