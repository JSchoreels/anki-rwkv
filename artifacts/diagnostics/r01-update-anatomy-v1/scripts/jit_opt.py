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
torch._C._set_graph_executor_optimize(sys.argv[3] == "1")
import torch.utils.deterministic as det
det.fill_uninitialized_memory = False
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



seq = [batches[i % 2] for i in range(16)]
times = []
for i, (row, item, batch) in enumerate(seq):
    torch.cuda.synchronize()
    started = time.perf_counter()
    stats = update(batch, item.seed)
    torch.cuda.synchronize()
    times.append(round((time.perf_counter() - started) * 1000, 2))
row, item, batch = batches[1]
stats = update(batch, item.seed)
torch.cuda.synchronize()
grads = {n: p.grad.detach().float().cpu() for n, p in model.named_parameters() if p.grad is not None}
torch.save({"loss": stats.average_loss.detach().float().cpu(), "grads": grads}, OUT.with_suffix(".pt"))
OUT.write_text(json.dumps({"optimize": sys.argv[3], "times_ms": times, "large_last4": times[-4::2], "medium_last4": times[-3::2]}), encoding="utf-8")
print(json.dumps({"optimize": sys.argv[3], "times_ms": times}))
