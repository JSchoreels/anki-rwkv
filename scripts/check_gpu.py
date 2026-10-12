"""Verify CUDA recurrence and train the complete model on synthetic data."""

from pathlib import Path
import sys
import tomllib
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from rwkv.model.rwkv_ops import reference_rwkv7
from train_gpu_trial import run


def main():
    torch.set_num_threads(4)
    torch.manual_seed(2025)
    inputs = [torch.randn(2, 8, 4, 32) * 0.1 for _ in range(6)]
    inputs[3] = inputs[3].sigmoid()
    skip = torch.zeros(2, 8, dtype=torch.bool)
    skip[:, 2] = True
    cpu = [x.clone().requires_grad_() for x in inputs]
    gpu = [x.cuda().requires_grad_() for x in inputs]
    out_cpu = reference_rwkv7(*cpu, skip)
    out_gpu = reference_rwkv7(*gpu, skip.cuda())
    torch.testing.assert_close(out_gpu.cpu(), out_cpu, rtol=1e-4, atol=1e-6)
    out_cpu.square().sum().backward()
    out_gpu.square().sum().backward()
    for left, right in zip(cpu, gpu):
        torch.testing.assert_close(right.grad.cpu(), left.grad, rtol=1e-4, atol=1e-6)
    print("Reference CUDA forward/backward matches CPU (including skipped state updates)", flush=True)
    data = ROOT / "artifacts" / "gpu-smoke" / "data"
    for table in ("cards", "decks", "revlogs"):
        (data / table / "user_id=1").mkdir(parents=True, exist_ok=True)
    reviews = pd.DataFrame({
        "card_id": [i % 4 for i in range(64)], "day_offset": [i // 4 for i in range(64)],
        "rating": [1 + i % 4 for i in range(64)], "state": [0 if i < 4 else 2 for i in range(64)],
        "duration": [3000] * 64, "elapsed_days": [-1 if i < 4 else 1 for i in range(64)],
        "elapsed_seconds": [-1 if i < 4 else 86400 for i in range(64)],
    })
    reviews.to_parquet(data / "revlogs" / "user_id=1" / "data.parquet", index=False)
    pd.DataFrame({"card_id": range(4), "note_id": range(4), "deck_id": [0] * 4}).to_parquet(
        data / "cards" / "user_id=1" / "data.parquet", index=False)
    pd.DataFrame({"deck_id": [0], "parent_id": [-1], "preset_id": [0]}).to_parquet(
        data / "decks" / "user_id=1" / "data.parquet", index=False)
    with (ROOT / "configs" / "train-gpu-trial.toml").open("rb") as source:
        config = tomllib.load(source)
    if config["WKV_BACKEND"] == "nvrtc":
        from rwkv.model.nvrtc import register
        from scripts.check_cuda import check_recurrence, check_training
        register()
        for heads in (1, 4):
            check_recurrence(torch.device("cuda:0"), torch.float32, heads)
        # Also cross several state checkpoints and the trial's maximum length.
        from rwkv.model.rwkv_ops import RWKV7_WKV
        for length in (32, 64, 256):
            shape = (2, length, 4, 32)
            values = [torch.randn(shape) * 0.05 for _ in range(6)]
            values[3] = torch.full(shape, 0.9)
            skip = torch.zeros(2, length, dtype=torch.bool)
            skip[:, ::7] = True
            skip[:, -1] = True
            left = [x.clone().requires_grad_() for x in values]
            right = [x.cuda().requires_grad_() for x in values]
            expected = reference_rwkv7(*left, skip)
            actual = RWKV7_WKV.apply(*right, skip.cuda())
            gradient = torch.randn(shape) * 0.1
            (expected * gradient).sum().backward()
            (actual * gradient.cuda()).sum().backward()
            torch.testing.assert_close(actual.cpu(), expected.detach(), atol=2e-5, rtol=2e-3)
            for a, b in zip(left, right):
                torch.testing.assert_close(b.grad.cpu(), a.grad, atol=2e-5, rtol=2e-3)
        check_training(torch.device("cuda:0"), torch.float32)
        print("NVRTC float32 outputs/gradients match reference, including 256 timesteps", flush=True)
    run_id = time.time_ns()
    config.update(DATA_PATH=str(data), USER_END=1, CHUNK_REVIEWS=16, BATCH_CHUNKS=2,
                  OUTPUT_DIR=f"artifacts/diagnostics/gpu-smoke/run-{run_id}", CHECKPOINT_DIR=f"artifacts/diagnostics/gpu-smoke/checkpoints-{run_id}",
                  CHECKPOINT_EVERY_STEPS=1, WARMUP_STEPS=2)
    summary = run(config)
    assert summary["status"] == "complete" and summary["steps"] == 2
    assert summary["selected_review_presentations"] == 64
    assert summary["head_max_weight_change"] > 0
    print("Synthetic CUDA model training passed: finite gradients, changed weights, saved checkpoint", flush=True)
    # A completed checkpoint resumes at the end without training the data again.
    resumed = run(config, resume=True)
    assert resumed["steps"] == summary["steps"]
    assert resumed["selected_review_presentations"] == 64
    print("Resume passed without repeating completed reviews", flush=True)


if __name__ == "__main__":
    main()
