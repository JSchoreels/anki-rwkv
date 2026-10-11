"""F01: bound CPU preparation across users without changing update order."""

from collections import deque
import time

from scripts.gpu_batch1 import frame_stream, prepare_job, split_frame


def training_batches(rows, span, seed, pool, options, cache_paths=None, depth=8):
    if options != {"prefetch_users": True} or cache_paths is not None or depth < 1:
        raise ValueError("R01 F01 is an isolated ordered-prefetch experiment")

    def jobs():
        for row, frame in frame_stream(rows, prefetch=True):
            section = split_frame(frame, row, "train")
            length = row["train_range"][1] - row["train_range"][0]
            for offset in range(0, length, span):
                end = min(length, offset + span)
                yield row, (row, offset, end, seed + row["user_id"] + offset,
                            options, section.iloc[offset:end], None)

    iterator = iter(jobs())
    pending = deque()
    try:
        for _ in range(depth):
            entry = next(iterator, None)
            if entry is None:
                break
            row, args = entry
            pending.append((row, pool.submit(prepare_job, *args)))
        while pending:
            row, future = pending.popleft()
            started = time.perf_counter()
            item = future.result()
            item.wait_seconds = time.perf_counter() - started
            entry = next(iterator, None)
            if entry is not None:
                next_row, args = entry
                pending.append((next_row, pool.submit(prepare_job, *args)))
            yield row, item
    finally:
        for _, future in pending:
            future.cancel()
        iterator.close()
