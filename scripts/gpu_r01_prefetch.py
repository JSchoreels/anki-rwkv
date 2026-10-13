"""F01: bound CPU preparation across users without changing update order."""

from collections import deque
import time

from scripts.gpu_batch1 import frame_stream, prepare_job, split_frame


def training_batches(rows, span, seed, pool, options, cache_paths=None, depth=8):
    if (options.get("prefetch_users") is not True
            or set(options) - {"prefetch_users", "reduced_sync", "preparation_workers", "fused_shift", "padding_factor", "async_transfer", "packed_ipc", "gpu_packed_features", "preprocessing_cache", "batched_id_encodings", "preprocessing_cache_bytes", "preprocessing_cache_mode",
                                          "split_excess", "batch_chunks", "gpu_memory_fraction", "grouped_sync", "main_threads", "driver_launch", "deferred_evaluation_metrics",
                                         "bulk_transfer", "parallel_evaluation", "jit_optimize", "nondeterministic", "skip_uninitialized_fill", "skip_padding_sources", "concurrent_splits", "concurrent_split_limit", "exact_bonus_products", "batched_split_gather", "fused_padded_gather", "exact_gate_glue", "exact_channel_glue", "worker_scatter_orders", "recurrence_prefetch", "recurrence_load_pipeline", "wave_aware_backward", "flat_buffers", "group_adam", "partial_shift_lerp", "explicit_shift_lerp", "exact_shift_lerp", "compact_previous_state", "exact_shift", "exact_index", "shared_shift_orders", "stream_launches", "prefetch_rows", "prefetch_shift", "training_seed", "pack_users", "prestart_pool", "fast_samples", "columnar_samples", "batch_segment_features", "direct_sample_matrix", "row_tiles_exact", "exact_group_norm", "deferred_reads", "host_placement", "sync_free_loss"}
            or bool(options.get("preprocessing_cache")) != (cache_paths is not None) or depth < 1):
        raise ValueError("Unsupported R01 ordered-prefetch preparation options")

    def jobs():
        descriptors_only = cache_paths is not None and options.get("preprocessing_cache_mode") != "cold_build"
        source = ((row, None) for row in rows) if descriptors_only else frame_stream(rows, prefetch=True)
        for row, frame in source:
            section = split_frame(frame, row, "train") if frame is not None else None
            length = row["train_range"][1] - row["train_range"][0]
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
            entry = next(iterator, None)
            if entry is not None:
                next_row, args = entry
                pending.append((next_row, pool.submit(preparation, *args)))
            yield row, item
    finally:
        for _, future in pending:
            future.cancel()
        iterator.close()
