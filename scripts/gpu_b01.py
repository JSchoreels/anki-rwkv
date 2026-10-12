"""Isolated B01 staging, including deferred F14 metadata in IPC and transfers.

These execution helpers preserve the original sample/chunk/seed ordering. The
bundle membership is selected separately after functional checks, then frozen.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import time

import torch
from rwkv.data_processing import get_rwkv_data
from rwkv.model.srs_model import PreparedBatch
from rwkv.prepare_batch import prepare
from scripts.gpu_batch1 import (DATA, _WORKER_FRAMES, batch_tensors as base_tensors,
                               frame_stream, make_samples, map_batch as base_map, split_frame)
from scripts.gpu_packed_encodings import PackedPreparedBatch, prepare_gpu_packed
from scripts.trial_batches import TrialBatch


def batch_tensors(batch):
    values = base_tensors(batch)
    if isinstance(batch, PackedPreparedBatch):
        values += list(batch.packed.values())
    return values


def map_batch(batch, function):
    moved = base_map(batch, function)
    if isinstance(batch, PackedPreparedBatch):
        moved = replace(moved, packed={name: function(value) for name, value in batch.packed.items()})
    return moved


def pack_batch(batch):
    """One aligned CPU byte storage for both hierarchy and feature metadata."""
    tensors = batch_tensors(batch)
    offsets, size = [], 0
    for value in tensors:
        size = (size + 7) // 8 * 8
        offsets.append(size)
        size += value.numel() * value.element_size()
    storage = torch.empty(size, dtype=torch.uint8)
    views = []
    for value, offset in zip(tensors, offsets):
        target = storage[offset:offset + value.numel() * value.element_size()]
        target.copy_(value.contiguous().view(torch.uint8).flatten())
        views.append(target.view(value.dtype).view(value.shape))
    iterator = iter(views)
    return map_batch(batch, lambda _: next(iterator))


def materialize(batch, dtype):
    """Expand already transferred metadata on the caller's current CUDA stream."""
    if not isinstance(batch, PackedPreparedBatch):
        batch.start = batch.start.to(dtype)
        return batch
    packed = batch.packed
    columns = [batch.start]
    for index in range(4):
        columns.append(packed[f"table{index}"][packed[f"indices{index}"]])
    angles = packed["frequencies"] * ((packed["baselines"] + packed["days"][:, None]) % packed["periods"])
    first_angles = packed["frequencies"] * ((packed["baselines"] + packed["first"][:, None]) % packed["periods"])
    columns.append(torch.stack((angles.sin(), angles.cos(), first_angles.sin(), first_angles.cos()), -1).flatten(1))
    start = torch.cat(columns, -1)
    start.masked_fill_(packed["padded"][:, None], 0)
    # Return a normal PreparedBatch so later graph keys describe the expanded
    # model inputs and never silently retain unconsumed packed metadata.
    return PreparedBatch(start=start.to(dtype), labels=batch.labels, label_review_th=batch.label_review_th,
                         sub_gather=batch.sub_gather, sub_gather_lens=batch.sub_gather_lens,
                         time_shift_selects=batch.time_shift_selects, skips=batch.skips, num_data=batch.num_data)


def split_settings(options):
    """F24 merges hierarchy splits; defaults reproduce the accepted layout."""
    return {"padding_factor": options.get("padding_factor", 0.9),
            "allowed_excess_in_one_step": options.get("split_excess", 20000)}


def prepare_job(row, offset, end, seed, options, section=None):
    started = time.perf_counter()
    user = row["user_id"]
    if section is None:
        if user not in _WORKER_FRAMES:
            _WORKER_FRAMES.clear()
            _WORKER_FRAMES[user] = split_frame(get_rwkv_data(DATA, user), row, "train")
        section = _WORKER_FRAMES[user].iloc[offset:end]
    from scripts.gpu_r01_samples import samples_for
    samples = samples_for(section, user, options)
    batch = (prepare_gpu_packed(samples, seed, **split_settings(options)) if options.get("gpu_packed_features")
             else prepare(samples, seed=seed, **split_settings(options)))
    if options.get("packed_ipc"):
        batch = pack_batch(batch)
    return TrialBatch(batch, offset, end - offset, len(samples), seed, time.perf_counter() - started)


def training_batches(rows, span, seed, pool, options, depth):
    def jobs(selected_rows):
        source = ((row, None) for row in selected_rows) if options.get("worker_local") else frame_stream(selected_rows, options.get("prefetch_users", False))
        for row, frame in source:
            section = split_frame(frame, row, "train") if frame is not None else None
            length = row["train_range"][1] - row["train_range"][0]
            for offset in range(0, length, span):
                end = min(length, offset + span)
                yield row, (row, offset, end, seed + row["user_id"] + offset, options,
                            section.iloc[offset:end] if section is not None else None)

    def consume(iterator):
        pending = deque()
        for _ in range(depth):
            entry = next(iterator, None)
            if entry is None:
                break
            row, args = entry
            pending.append((row, pool.submit(prepare_job, *args)))
        try:
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

    if options.get("prefetch_users"):
        yield from consume(iter(jobs(rows)))
    else:
        for row in rows:
            yield from consume(iter(jobs([row])))


def transfer_batches(source, dtype, asynchronous=False, bulk=False):
    device = torch.device("cuda:0")
    if bulk and not asynchronous:
        raise ValueError("Bulk staging extends the asynchronous transfer stage")
    if not asynchronous:
        for row, item in source:
            moved = map_batch(item.batch, lambda x: x.to(device))
            yield row, item, materialize(moved, dtype)
        return
    iterator = iter(source)
    stream = torch.cuda.Stream()
    with ThreadPoolExecutor(max_workers=1) as stage:
        def pin_next():
            entry = next(iterator, None)
            if entry is None:
                return None
            row, item = entry
            if bulk:
                # F20: one pinned copy per packed worker storage instead of one per tensor view.
                from scripts.gpu_r01_update import bulk_pin
                return row, item, bulk_pin(item.batch, map_batch, batch_tensors)
            return row, item, map_batch(item.batch, lambda x: x.pin_memory())

        def copy(entry):
            if entry is None:
                return None
            row, item, pinned = entry
            with torch.cuda.stream(stream):
                begin = torch.cuda.Event(enable_timing=True)
                begin.record(stream)
                if bulk:
                    from scripts.gpu_r01_update import bulk_to_device
                    pinned_batch, storages = pinned
                    moved = bulk_to_device(pinned_batch, map_batch, storages, device)
                else:
                    moved = map_batch(pinned, lambda x: x.to(device, non_blocking=True))
                batch = materialize(moved, dtype)
                # Includes deferred feature arithmetic, not just DMA. Waiting on
                # this event makes the fully expanded inputs safe to consume.
                ready = torch.cuda.Event(enable_timing=True)
                ready.record(stream)
                item.copy_started, item.copy_ended = begin, ready
            return row, item, batch, ready, pinned

        current = copy(stage.submit(pin_next).result())
        pending = stage.submit(pin_next)
        while current is not None:
            row, item, batch, ready, pinned = current
            consumer = torch.cuda.current_stream()
            consumer.wait_event(ready)
            for tensor in base_tensors(batch):
                tensor.record_stream(consumer)
            following = copy(pending.result())
            pending = stage.submit(pin_next) if following is not None else None
            yield row, item, batch
            current = following
