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



import torch.utils.deterministic as det
gather_queue = []


def skip_forward(ctx, x, index):
    gather = gather_queue.pop(0)
    ctx.save_for_backward(gather)
    ctx.rows = x.shape[0]
    return torch.index_select(x, 0, index)


def set_variant(fill, skip):
    det.fill_uninitialized_memory = fill
    if skip:
        Original.forward = staticmethod(skip_forward)
    else:
        Original.forward = staticmethod(orig_forward)


def run(batch, seed, skip):
    if skip:
        gather_queue[:] = [g.reshape(-1).long() for stream in batch.sub_gather for g in stream]
    stats = update(batch, seed)
    return stats


def snapshot():
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def same(a, b):
    bits = {torch.bfloat16: torch.int16, torch.float32: torch.int32}[a.dtype]
    return a.shape == b.shape and torch.equal(a.view(bits), b.view(bits))


variants = {"fill_on": (True, False), "fill_off": (False, False), "fill_off_skip_pad": (False, True), "fill_on_skip_pad": (True, True)}
result = {"variants": variants, "batches": []}
for row, item, batch in batches:
    entry = {"user": row["user_id"], "samples": int(batch.num_data), "ms": {k: [] for k in variants}, "equal_to_fill_on": {}}
    ref = None
    for name, (fill, skip) in variants.items():
        set_variant(fill, skip)
        stats = run(batch, item.seed, skip)
        torch.cuda.synchronize()
        loss, grads = stats.average_loss.detach().clone(), snapshot()
        if ref is None:
            ref = (loss, grads)
        else:
            entry["equal_to_fill_on"][name] = bool(same(loss, ref[0]) and sorted(grads) == sorted(ref[1]) and all(same(grads[n], ref[1][n]) for n in grads))
    for rep in range(4):
        for name, (fill, skip) in (list(variants.items()) if rep % 2 == 0 else list(reversed(variants.items()))):
            set_variant(fill, skip)
            run(batch, item.seed, skip)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(4):
                run(batch, item.seed, skip)
            torch.cuda.synchronize()
            entry["ms"][name].append((time.perf_counter() - started) / 4 * 1000)
    from torch.profiler import profile, ProfilerActivity
    for name in ("fill_on", "fill_off"):
        set_variant(*variants[name])
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            run(batch, item.seed, False)
            torch.cuda.synchronize()
        counts = collections.Counter(ev.name for ev in prof.events() if ev.name in ("cudaLaunchKernel", "cuLaunchKernel", "aten::fill_", "cudaMemsetAsync"))
        entry.setdefault("launches", {})[name] = dict(counts)
    set_variant(True, False)
    entry["median_ms"] = {k: sorted(v)[len(v) // 2] for k, v in entry["ms"].items()}
    result["batches"].append(entry)
    print(json.dumps({k: entry[k] for k in ("user", "samples", "median_ms", "equal_to_fill_on", "launches")}), flush=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
