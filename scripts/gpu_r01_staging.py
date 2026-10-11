"""Opt-in F14 staging with R01's ordered, bounded user prefetch."""

from scripts import gpu_r01_prefetch
from scripts.gpu_b01 import prepare_job


def prepare_packed_job(row, offset, end, seed, options, section, cache_path=None):
    if cache_path is not None:
        raise ValueError("Packed R01 staging does not use a sample cache")
    return prepare_job(row, offset, end, seed, options, section)


def training_batches(rows, span, seed, pool, options, cache_paths=None, depth=8):
    if options.get("gpu_packed_features") is not True:
        raise ValueError("F14 staging requires deferred packed features")
    previous = gpu_r01_prefetch.prepare_job
    gpu_r01_prefetch.prepare_job = prepare_packed_job
    try:
        yield from gpu_r01_prefetch.training_batches(rows, span, seed, pool, options, cache_paths, depth)
    finally:
        gpu_r01_prefetch.prepare_job = previous
