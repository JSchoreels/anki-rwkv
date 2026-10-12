"""F28 worker prestart and F29 cross-user batch packing for R01 training.

F29 keeps user order and every review exactly once, but a batch may hold the end
of one user's history and the start of the next, so batches stop ending half
empty at user boundaries. It changes update membership, update count and batch
seeds; quality is judged with the measured loss ceilings, not parity.
"""

import time
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

from scripts.trial_batches import TrialBatch


def pack_jobs(rows, frames, chunks, chunk_reviews=128):
    """Yield lists of (row, offset, end, section) parts filling `chunks` chunks per batch in user order."""
    parts, free = [], chunks
    for row, section in frames:
        length = row["train_range"][1] - row["train_range"][0]
        offset = 0
        while offset < length:
            take = min(length - offset, free * chunk_reviews)
            end = offset + take
            parts.append((row, offset, end, section.iloc[offset:end] if section is not None else None))
            free -= -(-take // chunk_reviews)
            offset = end
            if free == 0:
                yield parts
                parts, free = [], chunks
    if parts:
        yield parts


def prepare_packed_job(parts, seed, options):
    """Prepare one packed batch with the same preparation path as single-user jobs."""
    from scripts.gpu_b01 import pack_batch, split_settings
    from scripts.gpu_r01_samples import samples_for
    started = time.perf_counter()
    samples = [sample for row, offset, end, section in parts for sample in samples_for(section, row["user_id"], options)]
    if options.get("gpu_packed_features"):
        from scripts.gpu_packed_encodings import prepare_gpu_packed
        batch = prepare_gpu_packed(samples, seed, batched_id_encodings=options.get("batched_id_encodings", False),
                                   **split_settings(options))
    elif options.get("batched_id_encodings"):
        from scripts.gpu_optimizations import prepare_fast
        batch = prepare_fast(samples, seed=seed, batched_id_encodings=True, **split_settings(options))
    else:
        from rwkv.prepare_batch import prepare
        batch = prepare(samples, seed=seed, **split_settings(options))
    if options.get("packed_ipc"):
        batch = pack_batch(batch)
    item = TrialBatch(batch, parts[0][1], sum(end - offset for _, offset, end, _ in parts), len(samples), seed,
                      time.perf_counter() - started)
    item.parts = [(row["user_id"], offset, end - offset) for row, offset, end, _ in parts]
    return item


def worker_warmup():
    """F28: spawn a worker and import the preparation stack before the training timer starts."""
    import os
    import scripts.gpu_b01  # noqa: F401
    import scripts.gpu_preprocessing_stage  # noqa: F401
    import scripts.gpu_packed_encodings  # noqa: F401
    import rwkv.prepare_batch  # noqa: F401
    return os.getpid()


def worker_warmup_ready(ready_directory):
    """Run once in every spawned worker, before it can accept preparation jobs."""
    from scripts.trial_batches import worker_init
    worker_init()
    pid = worker_warmup()
    (Path(ready_directory) / f"{pid}.ready").write_text(str(pid), encoding="ascii")


@contextmanager
def warm_preparation_pool(workers, ready_directory):
    ready_directory = Path(ready_directory)
    ready_directory.mkdir(parents=True, exist_ok=False)
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                               initializer=worker_warmup_ready, initargs=(str(ready_directory),))
    try:
        yield pool
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def wait_ready_workers(ready_directory, workers, timeout=120):
    """Count initializer acknowledgements, rather than repeat jobs on one worker."""
    deadline = time.perf_counter() + timeout
    while True:
        pids = sorted(int(path.stem) for path in Path(ready_directory).glob("*.ready"))
        if len(pids) == workers:
            return pids
        if len(pids) > workers or time.perf_counter() >= deadline:
            raise RuntimeError(f"Preparation pool readiness: {len(pids)}/{workers} workers")
        time.sleep(0.01)
