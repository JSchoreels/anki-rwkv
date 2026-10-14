"""Shared data helpers for the compression experiments (read-only access to the main checkout's data and run)."""
import json
import random
from pathlib import Path

import torch

MAIN = Path(r"C:\Users\jonat\workspace\anki-rwkv")
RUN = MAIN / "artifacts" / "full-run" / "run-8k-20261013"
FORK_TRAIN_RANGE = (5000, 10000)  # the fork's RWKV_trained_on_5000_10000 saw these dataset users


def run_info():
    return json.loads((RUN / "run.json").read_text())


def data_path(run=None):
    run = run or run_info()
    return MAIN / run["dataset_path"]


def outside_fork_range(users):
    lo, hi = FORK_TRAIN_RANGE
    return [u for u in users if not lo <= u <= hi]


def heldout_users(count, seed=0, run=None):
    """`count` validation users of the 8k/2k run outside the fork's training range (seeded sample, sorted)."""
    run = run or run_info()
    pool = sorted(outside_fork_range(run["validation_users"]))
    if count >= len(pool):
        return pool
    return sorted(random.Random(seed).sample(pool, count))


def make_samples(frame, user, chunk):
    from rwkv.data_processing import create_sample
    return [create_sample(user, frame.iloc[i:i + chunk], [], torch.float32, torch.device("cpu"))
            for i in range(0, len(frame), chunk)]


def full_history_batch(data, user, seed=None):
    """Whole history as one sample (upstream benchmark test-DB layout; the fork's replay)."""
    from rwkv.data_processing import get_rwkv_data
    from rwkv.prepare_batch import prepare
    frame = get_rwkv_data(data, user)
    return prepare(make_samples(frame, user, len(frame)), seed=2025 + user if seed is None else seed)


def segment_batch(data, user, start, length, seed):
    """One contiguous history segment [start, start+length) as one sample (training)."""
    from rwkv.data_processing import get_rwkv_data
    from rwkv.prepare_batch import prepare
    frame = get_rwkv_data(data, user)
    return prepare(make_samples(frame.iloc[start:start + length], user, length), seed=seed)
