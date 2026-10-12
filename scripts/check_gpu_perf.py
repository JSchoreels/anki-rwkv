"""Check exact preparation, ordered prefetch, BF16, and interrupted resume."""

from dataclasses import asdict, is_dataclass
from pathlib import Path
import sys
import time
import tomllib
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch

from rwkv import data_processing as processing
from rwkv.prepare_batch import prepare
from scripts.trial_batches import batches, preparation_pool
import train_gpu_trial as training


def legacy_locations(frame, submodule):
    grouped = frame.groupby(submodule, observed=True)
    return [grouped.get_group(name)["index"].values for name in set(frame[submodule].values)]


def assert_equal(left, right):
    if is_dataclass(left):
        assert_equal(asdict(left), asdict(right))
    elif isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right, (left, right)


def check_preparation():
    legacy_sample = FunctionType(processing.create_sample.__code__,
        {**processing.create_sample.__globals__, "_module_group_locations": legacy_locations})
    for user in (1, 10, 92):
        frame = processing.get_rwkv_data(ROOT / "data/anki-revlogs-10k", user)
        for offset in (0, 2048, len(frame) - 128):
            section = frame.iloc[offset:offset + 128]
            old = legacy_sample(user, section, [], torch.float32, torch.device("cpu"))
            new = processing.create_sample(user, section, [], torch.float32, torch.device("cpu"))
            assert_equal(old, new)
            assert_equal(prepare([old], seed=2025 + offset), prepare([new], seed=2025 + offset))
    config = {"CHUNK_REVIEWS": 128, "BATCH_CHUNKS": 2, "REVIEWS_PER_USER": 1024,
              "SEED": 2025, "PREFETCH_WORKERS": 2, "PREFETCH_BATCHES": 4}
    serial = list(batches(frame.iloc[:1024], 92, 0, 0, config))
    with preparation_pool(2) as pool:
        parallel = list(batches(frame.iloc[:1024], 92, 0, 0, config, pool))
    assert [item.offset for item in parallel] == [0, 256, 512, 768]
    for a, b in zip(serial, parallel):
        assert_equal(a.batch, b.batch)
        assert (a.seed, a.offset, a.reviews, a.chunks) == (b.seed, b.offset, b.reviews, b.chunks)
    print("Samples and seeded batches match the original exactly; prefetch preserves inputs and order", flush=True)


def check_resume(config):
    config = {**config, "DATA_PATH": str(ROOT / "artifacts/diagnostics/gpu-smoke/data"),
              "USER_START": 1, "USER_END": 1, "CHUNK_REVIEWS": 16, "BATCH_CHUNKS": 2,
              "REVIEWS_PER_USER": 0, "CHECKPOINT_EVERY_STEPS": 1, "WARMUP_STEPS": 2,
              "PREFETCH_WORKERS": 0, "PREFETCH_BATCHES": 4, "POWER_LIMIT_WATTS": 0,
              "DTYPE": "float32"}
    identifier = time.time_ns()

    def locations(name):
        return {**config, "OUTPUT_DIR": f"artifacts/benchmarks/perf-trial/{name}-{identifier}",
                "CHECKPOINT_DIR": f"artifacts/benchmarks/perf-trial/checkpoints-{name}-{identifier}"}

    uninterrupted_config = locations("serial")
    reference = training.run(uninterrupted_config)
    interrupted_config = {**locations("prefetch"), "PREFETCH_WORKERS": 2}
    original_optimizer = training.get_optimizer

    def stopping_optimizer(*args):
        optimizer = original_optimizer(*args)
        original_step = optimizer.step

        def step(*step_args, **step_kwargs):
            result = original_step(*step_args, **step_kwargs)
            (Path(interrupted_config["OUTPUT_DIR"]) / "STOP").touch()
            return result

        optimizer.step = step
        return optimizer

    try:
        training.get_optimizer = stopping_optimizer
        stopped = training.run(interrupted_config)
        assert stopped["status"] == "stopped" and stopped["selected_review_presentations"] == 32
    finally:
        training.get_optimizer = original_optimizer
    (Path(interrupted_config["OUTPUT_DIR"]) / "STOP").unlink()
    resumed = training.run(interrupted_config, resume=True)
    assert resumed["status"] == "complete" and resumed["steps"] == reference["steps"] == 2
    assert resumed["selected_review_presentations"] == 64
    expected = torch.load(reference["checkpoint"], weights_only=True, map_location="cpu")
    actual = torch.load(resumed["checkpoint"], weights_only=True, map_location="cpu")
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=2e-6, rtol=1e-5)
    mixed = {**locations("bf16"), "PREFETCH_WORKERS": 2, "DTYPE": "bfloat16"}
    summary = training.run(mixed)
    assert summary["status"] == "complete" and summary["steps"] == 2
    tensors = torch.load(summary["checkpoint"], weights_only=True, map_location="cpu")
    assert all(x.dtype == torch.float32 and torch.isfinite(x).all() for x in tensors.values())
    assert training.run(mixed, resume=True)["steps"] == 2
    print("Interrupted prefetch resume matches synchronous training; BF16 keeps float32 master checkpoints", flush=True)


def main():
    torch.set_num_threads(4)
    check_preparation()
    from rwkv.model.nvrtc import register
    from scripts.check_cuda import check_recurrence
    register()
    for dtype in (torch.float32, torch.bfloat16):
        for heads in (1, 4):
            check_recurrence(torch.device("cuda:0"), dtype, heads, length=256)
    print("Float32/BF16 kernel outputs and gradients match the reference at 256 timesteps", flush=True)
    config = tomllib.loads((ROOT / "configs/train-gpu-trial.toml").read_text())
    check_resume(config)


if __name__ == "__main__":
    main()
