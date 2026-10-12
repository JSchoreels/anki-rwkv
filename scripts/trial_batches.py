"""Ordered, bounded CPU preparation for the GPU trial and performance checks."""

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import multiprocessing
import time

import torch

from rwkv.data_processing import create_sample
from rwkv.prepare_batch import prepare


@dataclass
class TrialBatch:
    batch: object
    offset: int
    reviews: int
    chunks: int
    seed: int
    preparation_seconds: float
    wait_seconds: float = 0.0


def worker_init():
    # Keep CPU preparation independent of the parent's CUDA/dropout generator.
    torch.set_num_threads(1)


@contextmanager
def preparation_pool(workers):
    if not workers:
        yield None
        return
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                               initializer=worker_init)
    try:
        yield pool
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def prepare_section(section, user, offset, chunk_reviews, seed, preparation="original"):
    if preparation not in ("original", "cpu", "gpu", "gpu-packed"):
        raise ValueError("Unknown preparation experiment")
    started = time.perf_counter()
    samples = [create_sample(user, section.iloc[start:start + chunk_reviews], [],
                             torch.float32, torch.device("cpu"))
               for start in range(0, len(section), chunk_reviews)]
    if preparation == "original":
        batch = prepare(samples, seed=seed)
    elif preparation == "gpu-packed":
        from scripts.gpu_packed_encodings import prepare_gpu_packed
        batch = prepare_gpu_packed(samples, seed=seed)
    else:
        from scripts.gpu_optimizations import prepare_fast
        batch = prepare_fast(samples, seed=seed, gpu_encodings=preparation == "gpu")
    return TrialBatch(batch, offset, len(section), len(samples), seed, time.perf_counter() - started)


def batches(frame, user, epoch, offset_start, config, pool=None):
    chunk = config["CHUNK_REVIEWS"]
    span = chunk * config.get("BATCH_CHUNKS", 1)
    cap = config["REVIEWS_PER_USER"]
    used = min(len(frame), cap) if cap else len(frame)
    offsets = iter(range(offset_start, used, span))

    def submit(offset):
        section = frame.iloc[offset:min(used, offset + span)]
        seed = config["SEED"] + epoch * 10001 + user + offset
        args = (section, user, offset, chunk, seed, config.get("PREPARATION", "original"))
        return pool.submit(prepare_section, *args) if pool else prepare_section(*args)

    if pool is None:
        for offset in offsets:
            item = submit(offset)
            item.wait_seconds = item.preparation_seconds
            yield item
        return
    pending = deque()
    try:
        for _ in range(config.get("PREFETCH_BATCHES", max(2, config.get("PREFETCH_WORKERS", 0) * 2))):
            offset = next(offsets, None)
            if offset is None:
                break
            pending.append(submit(offset))
        while pending:
            started = time.perf_counter()
            item = pending.popleft().result()
            item.wait_seconds = time.perf_counter() - started
            offset = next(offsets, None)
            if offset is not None:
                pending.append(submit(offset))
            # Consuming in submission order preserves every review position.
            yield item
    finally:
        for future in pending:
            future.cancel()
