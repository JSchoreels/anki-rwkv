"""Diagnostic (no training): anatomy of one E33 update on real batches.

1. Attribute aten::fill_ / cudaMemsetAsync launches to their nearest named ancestor op.
2. Check that gradients reaching padded row-select sources (split_gather == -1) are exactly +-0.
3. Time rows_backward with and without padded sources (CUDA events) and compare outputs bitwise.
4. Record WKV forward/backward kernel durations per (stream, split) to bound split concurrency.
"""
import collections, json, os, sys, time
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2])
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from scripts.r01_workflow import configure_runtime
configure_runtime(ROOT, 1)
cohort = json.loads((ROOT / "artifacts/retraining/gpu-retrain-r01/cohorts-user-holdout-v1.json").read_text(encoding="utf-8"))
os.environ["GPU_BATCH1_DATA_PATH"] = str(ROOT / cohort["dataset_path"])
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import numpy as np
import torch
torch.set_num_threads(1)
from rwkv.data_processing import get_rwkv_data
from rwkv.model import exact_index, nvrtc
from scripts.benchmark_gpu_trial import benchmark_models, fresh_checkpoint
from scripts.gpu_batch1 import split_frame
from scripts.gpu_b01 import transfer_batches
from scripts.gpu_preprocessing_stage import prepare_job
from scripts.gpu_row_tiles import row_tiled_backend
import expertium.model.srs_model as srs

nvrtc.register()
nvrtc.set_checkpoint_length(32)
OPTIONS = json.loads((ROOT / "artifacts/retraining/gpu-retrain-r01/codex-round-0037-0039-handoff.json").read_text(encoding="utf-8"))["accepted_options"]
prep = {k: OPTIONS[k] for k in ("prefetch_users", "gpu_packed_features", "packed_ipc", "fast_samples", "batched_id_encodings", "columnar_samples")}
prep["preparation_workers"] = 6
protocol = json.loads((ROOT / "artifacts/retraining/gpu-retrain-r01/iteration-0033/protocol.json").read_text(encoding="utf-8"))
rows = {r["user_id"]: r for r in protocol["rows"]}
picks = [(2723, 0, 65536), (609, 0, 18812)]
jobs = []
for user, offset, end in picks:
    row = rows[user]
    frame = split_frame(get_rwkv_data(Path(os.environ["GPU_BATCH1_DATA_PATH"]), user), row, "train")
    end = min(end, len(frame))
    jobs.append((row, prepare_job(row, offset, end, 2025 + user + offset, prep, frame.iloc[offset:end])))
batches = [(row, item, batch) for (row, _), (_, item, batch) in zip(jobs, transfer_batches(iter(jobs), torch.bfloat16, False))]
torch._C._set_graph_executor_optimize(False)
torch.use_deterministic_algorithms(True)
master, model, dtype = benchmark_models({"MODEL": "expertium", "DTYPE": "bfloat16", "SEED": 2025, "WKV_BACKEND": "nvrtc"}, torch.device("cuda:0"))
master.load_state_dict(fresh_checkpoint("expertium", 2025)["model"])
model.copy_downcast_(master, dtype)
for module in model.modules():
    for flag in ("exact_shift", "exact_index", "exact_group_norm"):
        if hasattr(module, flag):
            setattr(module, flag, bool(OPTIONS.get(flag, False)))
model.sync_free_loss = True
model.train()
stack = ExitStack()
stack.enter_context(row_tiled_backend(rows=OPTIONS["row_tiles_exact"], tile=32, exact=True))
nvrtc.set_stream_launch(True)
exact_index.set_shared_orders(True)
exact_index.set_prefetch_rows(True)

# Row-select capture: forward calls happen in (stream, split) order; remember each call's padding mask.
captured = []
Original = exact_index.ExactIndexSelect


orig_forward, orig_backward = Original.forward, Original.backward


def capture_forward(ctx, x, index):
    ctx.slot = len(captured)
    captured.append({"rows": x.shape[0], "index": index})
    return orig_forward(ctx, x, index)


def capture_backward(ctx, grad):
    captured[ctx.slot]["grad"] = grad.detach().clone()
    return orig_backward(ctx, grad)


def update(batch, seed):
    torch.manual_seed(seed)
    model.zero_grad(set_to_none=True)
    stats = model.get_loss(batch)
    stats.average_loss.backward()
    return stats


result = {"options": OPTIONS, "batches": []}
for row, item, batch in batches:
    entry = {"user": row["user_id"], "reviews": int(batch.num_data)}
    for _ in range(2):
        update(batch, item.seed)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(3):
        update(batch, item.seed)
    torch.cuda.synchronize()
    entry["eager_ms_per_update"] = (time.perf_counter() - started) / 3 * 1000
    # 1. Profile fill_/memset attribution.
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        update(batch, item.seed)
        torch.cuda.synchronize()
    events = prof.events()
    targets = collections.Counter()
    totals = collections.Counter()
    for ev in events:
        if ev.name in ("aten::fill_", "cudaMemsetAsync", "aten::zero_", "cudaLaunchKernel", "cuLaunchKernel"):
            totals[ev.name] += 1
        if ev.name not in ("aten::fill_", "cudaMemsetAsync"):
            continue
        chain, parent = [], ev.cpu_parent
        while parent is not None and len(chain) < 6:
            chain.append(parent.name)
            parent = parent.cpu_parent
        if ev.name == "aten::fill_" and chain and chain[0] in ("aten::fill_",):
            continue
        named = [c for c in chain if c not in ("aten::fill_", "aten::zero_", "aten::zeros", "aten::zeros_like", "aten::empty", "aten::new_zeros", "aten::full", "aten::full_like")]
        key = ev.name + " <- " + " < ".join(chain[:3]) + " || " + (named[0] if named else "-")
        targets[key] += 1
    entry["launch_totals"] = dict(totals)
    entry["fill_memset_sources"] = targets.most_common(25)
    kernels = [ev for ev in events if ev.device_type == torch.autograd.DeviceType.CUDA]
    entry["device_kernel_ms"] = sum(ev.device_time for ev in kernels) / 1000 if kernels and hasattr(kernels[0], "device_time") else None
    wkv = [(ev.name[:40], ev.device_time / 1000) for ev in kernels if "rwkv_exact" in ev.name]
    entry["wkv_kernels_ms"] = [round(v, 4) for _, v in wkv]
    entry["wkv_kernel_names"] = sorted(set(n for n, _ in wkv))
    # 2/3. Padding gradients and rows_backward with/without padding sources.
    captured.clear()
    Original.forward, Original.backward = staticmethod(capture_forward), staticmethod(capture_backward)
    try:
        update(batch, item.seed)
    finally:
        Original.forward, Original.backward = staticmethod(orig_forward), staticmethod(orig_backward)
    torch.cuda.synchronize()
    gathers = [g for stream in batch.sub_gather for g in stream]
    if len(gathers) != len(captured):
        raise RuntimeError(f"capture mismatch {len(gathers)} vs {len(captured)}")
    calls = []
    for gather, rec in zip(gathers, captured):
        gather = gather.reshape(-1).long()
        pad = gather < 0
        grad = rec["grad"]
        padded_grad = grad[pad]
        nonzero = int((padded_grad != 0).sum())
        nonfinite = int((~torch.isfinite(padded_grad)).sum())
        negzero = int(((padded_grad == 0) & torch.signbit(padded_grad.float())).sum())
        index = rec["index"]
        timings = {}
        outs = {}
        for name, idx in (("clamped", index), ("skip_padding", gather)):
            for _ in range(3):
                outs[name] = exact_index.rows_backward(grad, idx, rec["rows"])
            torch.cuda.synchronize()
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):
                outs[name] = exact_index.rows_backward(grad, idx, rec["rows"])
            stop.record()
            torch.cuda.synchronize()
            timings[name] = start.elapsed_time(stop) / 10
        a, b = outs["clamped"], outs["skip_padding"]
        equal = bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))
        calls.append({"rows": rec["rows"], "sources": int(gather.numel()), "padded_sources": int(pad.sum()),
                      "padded_grad_nonzero": nonzero, "padded_grad_nonfinite": nonfinite, "padded_grad_negzero": negzero,
                      "ms_clamped": round(timings["clamped"], 4), "ms_skip_padding": round(timings["skip_padding"], 4),
                      "bitwise_equal": equal})
    entry["row_select_calls"] = calls
    entry["row_select_ms_clamped"] = round(sum(c["ms_clamped"] for c in calls), 3)
    entry["row_select_ms_skip_padding"] = round(sum(c["ms_skip_padding"] for c in calls), 3)
    entry["split_shapes"] = [[[int(L), int(g.numel() // L)] for g, L in zip(gs, ls)] for gs, ls in zip(batch.sub_gather, batch.sub_gather_lens)]
    result["batches"].append(entry)
    print(json.dumps({k: entry[k] for k in ("user", "reviews", "eager_ms_per_update", "launch_totals", "device_kernel_ms", "row_select_ms_clamped", "row_select_ms_skip_padding")}), flush=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
