"""F01: bound CPU preparation across users without changing update order."""

from collections import deque
import time

import numpy as np

from scripts.gpu_batch1 import frame_stream, prepare_job, split_frame


def training_batches(rows, span, seed, pool, options, cache_paths=None, depth=8):
    if (options.get("prefetch_users") is not True
            or set(options) - {"prefetch_users", "reduced_sync", "preparation_workers", "fused_shift", "padding_factor", "async_transfer", "packed_ipc", "gpu_packed_features", "preprocessing_cache", "batched_id_encodings", "preprocessing_cache_bytes", "preprocessing_cache_mode",
                                          "split_excess", "batch_chunks", "gpu_memory_fraction", "grouped_sync", "main_threads", "driver_launch", "deferred_evaluation_metrics", "native_wkv", "native_group_norm",
                                         "bulk_transfer", "parallel_evaluation", "jit_optimize", "main_priority", "wddm_flush", "wddm_flush_streams", "wkv_rows", "lean_step_reset", "nondeterministic", "skip_uninitialized_fill", "skip_padding_sources", "concurrent_splits", "concurrent_split_limit", "exact_bonus_products", "batched_split_gather", "fused_padded_gather", "exact_gate_glue", "exact_channel_glue", "worker_scatter_orders", "recurrence_prefetch", "recurrence_load_pipeline", "wave_aware_backward", "flat_buffers", "group_adam", "partial_shift_lerp", "explicit_shift_lerp", "exact_shift_lerp", "compact_previous_state", "exact_shift", "exact_index", "shared_shift_orders", "stream_launches", "prefetch_rows", "prefetch_shift", "training_seed", "pack_users", "prestart_pool", "fast_samples", "columnar_samples", "batch_segment_features", "direct_sample_matrix", "row_tiles_exact", "exact_group_norm", "deferred_reads", "host_placement", "sync_free_loss", "process_qos", "native_index", "native_glue", "deferred_read_depth", "wkv_tile", "bookkeeping_stride", "worker_frames", "worker_qos",
                                         "architecture", "lr_schedule", "optimizer", "end_to_start_intervals", "distillation", "chunk_reviews", "instant_loss_weight", "state_carry"}
            or bool(options.get("preprocessing_cache")) != (cache_paths is not None) or depth < 1):
        raise ValueError("Unsupported R01 ordered-prefetch preparation options")
    carry = bool(options.get("state_carry"))
    universes, lanes = {}, {}
    if carry:
        # SM21: day features relative to the user's whole training section and the user's entity ids.
        from scripts.state_carry_train import (entity_universe, lane_universe, strided, strided_layout,
                                               user_segment_features, validate_options)
        validate_options(options)
        laned = strided(options["state_carry"])
        chunk = options.get("chunk_reviews", 128)
        if laned and span % chunk:
            raise ValueError("state_carry strided lanes need span = batch_chunks x chunk_reviews")

    def jobs():
        # F93a: with worker_frames each worker loads and splits its own user history, so the main
        # process neither runs the history-loading thread nor pickles DataFrame slices.
        descriptors_only = ((cache_paths is not None and options.get("preprocessing_cache_mode") != "cold_build")
                            or bool(options.get("worker_frames")))
        source = ((row, None) for row in rows) if descriptors_only else frame_stream(rows, prefetch=True)
        for row, frame in source:
            section = split_frame(frame, row, "train") if frame is not None else None
            if carry:
                section = user_segment_features(section)
            length = row["train_range"][1] - row["train_range"][0]
            if carry and laned:
                # SM21 v2: update u holds window u of every lane (scripts/state_carry_train.py docstring); its
                # offset (seed, dropout, logs) stays u x span as today, its section is the selected windows.
                layout = strided_layout(-(-length // chunk), span // chunk)
                universes[row["user_id"]] = lane_universe(section, layout, chunk)
                for u, update in enumerate(layout):
                    picked = np.concatenate([np.arange(w * chunk, min(length, (w + 1) * chunk)) for _, w in update])
                    lanes[row["user_id"], u * span] = [lane for lane, _ in update]
                    yield row, (row, u * span, u * span + len(picked), seed + row["user_id"] + u * span,
                                options, section.iloc[picked], None)
                continue
            if carry:
                universes[row["user_id"]] = entity_universe(section)
            for offset in range(0, length, span):
                end = min(length, offset + span)
                yield row, (row, offset, end, seed + row["user_id"] + offset,
                            options, section.iloc[offset:end] if section is not None else None,
                            cache_paths[row["user_id"], offset] if cache_paths is not None else None)

    def packed_jobs():
        from scripts.gpu_r01_packing import pack_jobs
        frames = ((row, split_frame(frame, row, "train")) for row, frame in frame_stream(rows, prefetch=True))
        for parts in pack_jobs(rows, frames, span // 128):
            first_row, first_offset = parts[0][0], parts[0][1]
            yield first_row, (parts, seed + first_row["user_id"] + first_offset, options)

    if options.get("worker_frames") and (options.get("pack_users") or cache_paths is not None):
        raise ValueError("F93a worker_frames does not combine with F29 packing or the preprocessing cache")
    if options.get("pack_users") and cache_paths is not None:
        raise ValueError("F29 packing prepares from history slices; it does not combine with the preprocessing cache")
    iterator = iter(packed_jobs() if options.get("pack_users") else jobs())
    preparation = prepare_job
    if options.get("pack_users"):
        from scripts.gpu_r01_packing import prepare_packed_job as preparation
    elif options.get("preprocessing_cache") or options.get("batched_id_encodings"):
        from scripts.gpu_preprocessing_stage import prepare_job as preparation
    pending = deque()
    try:
        for _ in range(depth):
            entry = next(iterator, None)
            if entry is None:
                break
            row, args = entry
            pending.append((row, pool.submit(preparation, *args)))
        while pending:
            row, future = pending.popleft()
            started = time.perf_counter()
            item = future.result()
            item.wait_seconds = time.perf_counter() - started
            if carry and item.offset == 0:
                item.carry_universe = universes.pop(row["user_id"])
            if carry and laned:
                item.carry_lanes = lanes.pop((row["user_id"], item.offset))
            entry = next(iterator, None)
            if entry is not None:
                next_row, args = entry
                pending.append((next_row, pool.submit(preparation, *args)))
            yield row, item
    finally:
        for _, future in pending:
            future.cancel()
        iterator.close()
