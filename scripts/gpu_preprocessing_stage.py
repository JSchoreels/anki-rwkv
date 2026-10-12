"""Opt-in F02 preprocessing cache and F19 batched CPU encoding staging."""
import time
import json
import hashlib
from pathlib import Path
import torch
from scripts.gpu_batch1 import make_samples, split_frame
from scripts.gpu_b01 import pack_batch
from scripts.gpu_optimizations import prepare_fast
from scripts.gpu_packed_encodings import prepare_gpu_packed
from scripts.gpu_preprocessing_cache import (PreprocessingCache, cache_context, deterministic_layout,
                                              source_identity, user_data_identity)
from scripts.trial_batches import TrialBatch

_FRAMES = {}


def prepare_job(row, offset, end, seed, options, section=None, cache_path=None):
    started = time.perf_counter()
    if options.get("padding_factor", 0.9) != 0.9:
        raise ValueError("The deterministic cache currently preserves padding factor 0.9")
    layout, cache_info = None, None
    if options.get("preprocessing_cache"):
        if not cache_path:
            raise ValueError("Cached preparation requires a descriptor")
        context = cache_path["context"]
        expected = cache_context(context["dataset_revision"], context["data_files"], source_identity(), row, offset, end)
        if context != expected:
            raise ValueError("Cache descriptor does not match the current preparation request/source")
        mode = options.get("preprocessing_cache_mode", "cold_or_warm")
        if mode not in ("cold_or_warm", "cold_build", "warm_required"):
            raise ValueError("Unknown preprocessing cache mode")
        if mode == "cold_build" and (section is None or len(section) != end - offset):
            raise ValueError("Cold builds require the ordered main-process history slice")
        if mode != "cold_build" and section is not None:
            raise ValueError("Descriptor-only cache preparation must not receive a history slice")
        def build():
            if mode == "warm_required":
                raise RuntimeError("Warm-cache trial requires a complete preexisting entry; no implicit rebuild")
            from rwkv.data_processing import get_rwkv_data
            if user_data_identity(cache_path["dataset"], row["user_id"]) != context["data_files"]:
                raise RuntimeError("Dataset changed after the cache manifest was prepared")
            if section is None:
                identity = json.dumps({"row": row, "data_files": context["data_files"], "sources": context["sources"]}, sort_keys=True)
                key = cache_path["dataset"], hashlib.sha256(identity.encode("utf-8")).hexdigest()
                if key not in _FRAMES:
                    _FRAMES.clear()
                    _FRAMES[key] = split_frame(get_rwkv_data(Path(cache_path["dataset"]), row["user_id"]), row, "train")
                selected = _FRAMES[key].iloc[offset:end]
            else:
                selected = section
            samples = make_samples(selected, row["user_id"], context["chunk_reviews"])
            return {"samples": samples, "layout": deterministic_layout(samples)}
        cache = PreprocessingCache(cache_path["directory"], cache_path["max_bytes"])
        saved, cache_info = cache.get_or_build(context, build)
        samples, layout = saved["samples"], saved["layout"]
    else:
        if cache_path is not None or section is None:
            raise ValueError("Uncached CPU extensions require the selected DataFrame")
        samples = make_samples(section, row["user_id"], 128)
    batched = options.get("batched_id_encodings", False)
    if options.get("gpu_packed_features"):
        batch = prepare_gpu_packed(samples, seed, batched_id_encodings=batched, layout=layout)
    else:
        batch = prepare_fast(samples, seed=seed, batched_id_encodings=batched, layout=layout)
    if options.get("packed_ipc"):
        batch = pack_batch(batch)
    item = TrialBatch(batch, offset, end - offset, len(samples), seed, time.perf_counter() - started)
    item.cache_info = cache_info
    return item
