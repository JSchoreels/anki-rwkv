"""CPU-only F02 cache/F19 encoding parity, integrity, ordering and timing checks."""
from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import fields, is_dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance


def same(left, right):
    import numpy as np
    import torch
    if is_dataclass(left):
        assert type(left) is type(right)
        for field in fields(left):
            same(getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape
        torch.testing.assert_close(left, right, atol=0, rtol=0, equal_nan=True)
    elif isinstance(left, np.ndarray):
        assert left.dtype == right.dtype
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    else:
        assert left == right


def fingerprint(batch):
    from scripts.gpu_b01 import batch_tensors
    digest = hashlib.sha256(json.dumps([int(batch.num_data), [[int(n) for n in g] for g in batch.sub_gather_lens]]).encode())
    for tensor in batch_tensors(batch):
        digest.update(f"{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.contiguous().reshape(-1).view(__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def cpu_worker_init():
    import torch
    torch.set_num_threads(1)
    def reject(*args, **kwargs):
        raise AssertionError("Spawned CPU check must not initialize CUDA")
    torch.cuda._lazy_init = reject


def worker_prepare(*args):
    import torch
    from scripts.gpu_preprocessing_stage import prepare_job
    item = prepare_job(*args)
    assert not torch.cuda.is_initialized()
    return {"fingerprint": fingerprint(item.batch), "offset": item.offset, "seed": item.seed,
            "reviews": item.reviews, "cache": item.cache_info,
            "rng": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}


def run_checks(output, protocol, report):
    import numpy as np
    import torch
    from scripts.gpu_b01 import materialize, prepare_job as packed_reference
    from scripts.gpu_batch1 import prepare_job as cpu_reference, make_samples, split_frame
    from scripts.gpu_optimizations import prepare_fast
    from scripts.gpu_preprocessing_cache import (PreprocessingCache, cache_context, context_key,
                                                 deterministic_layout, plan_cache, source_identity, user_data_identity)
    from scripts.gpu_preprocessing_stage import prepare_job
    from rwkv.data_processing import get_rwkv_data
    dataset = ROOT / protocol["dataset_path"]
    selected = protocol["rows"][:2]
    sections, contexts = {}, []
    report["checks"], report["parity_cases"] = [], []
    def check(name, condition=True):
        assert condition, name
        report["checks"].append(name)

    for row in selected:
        section = split_frame(get_rwkv_data(dataset, row["user_id"]), row, "train")
        sections[row["user_id"]] = section
        data = user_data_identity(dataset, row["user_id"])
        for length in (129, 512, 4096):
            end = min(length, len(section))
            context = cache_context(protocol["dataset_revision"], data, source_identity(), row, 0, end)
            contexts.append(context)
            descriptor = {"directory": str(output / "cache"), "max_bytes": 32 * 1024**2,
                          "dataset": str(dataset), "context": context}
            for packed in (False, True):
                for seed in (2025 + row["user_id"], 9091):
                    original = (packed_reference if packed else cpu_reference)(
                        row, 0, end, seed, {"gpu_packed_features": True} if packed else {}, section.iloc[:end]).batch
                    expected_rng = torch.get_rng_state().clone()
                    options = {"gpu_packed_features": packed, "batched_id_encodings": True}
                    direct = prepare_job(row, 0, end, seed, options, section.iloc[:end])
                    same(direct.batch, original)
                    check("F19 exact inputs and CPU RNG", torch.equal(expected_rng, torch.get_rng_state()))
                    options["preprocessing_cache"] = True
                    cold_descriptor = dict(descriptor, directory=str(output / "main-slice-cache"))
                    with patch("rwkv.data_processing.get_rwkv_data", side_effect=AssertionError("Cold workers must not reload a complete history")):
                        staged = prepare_job(row, 0, end, seed, dict(options, preprocessing_cache_mode="cold_build"),
                                             section.iloc[:end], cold_descriptor)
                    same(staged.batch, original)
                    check("Main-slice cold cache exact inputs and CPU RNG without worker history loads",
                          torch.equal(expected_rng, torch.get_rng_state()))
                    cached = prepare_job(row, 0, end, seed, options, cache_path=descriptor)
                    same(cached.batch, original)
                    check("F02+F19 exact inputs and CPU RNG", torch.equal(expected_rng, torch.get_rng_state()))
                    options["batched_id_encodings"] = False
                    cache_only = prepare_job(row, 0, end, seed, options, cache_path=descriptor)
                    same(cache_only.batch, original)
                    check("F02 exact inputs and CPU RNG", torch.equal(expected_rng, torch.get_rng_state()))
                    check("Warm cache hit", cache_only.cache_info["hit"])
                    if packed:
                        same(materialize(direct.batch, torch.bfloat16), materialize(original, torch.bfloat16))
                    report["parity_cases"].append({"user_id": row["user_id"], "reviews": end,
                                                  "seed": seed, "packed": packed,
                                                  "input_sha256": fingerprint(original),
                                                  "initial_cache_hit": cached.cache_info["hit"]})
    print("CPU reference parity: two users, partial/full chunks, two seeds and both feature paths passed", flush=True)

    context = contexts[0]
    samples = make_samples(sections[selected[0]["user_id"]].iloc[:129], selected[0]["user_id"], 128)
    before = torch.get_rng_state().clone()
    layout = deterministic_layout(samples)
    check("Cache template construction preserves CPU RNG", torch.equal(before, torch.get_rng_state()))
    payload = {"samples": samples, "layout": layout}
    cache = PreprocessingCache(output / "integrity-cache", 8 * 1024**2)
    saved, info = cache.get_or_build(context, lambda: payload)
    check("Cold cache build persisted", not info["hit"] and info["persisted"])
    loaded, info = cache.get_or_build(context, lambda: (_ for _ in ()).throw(AssertionError("Hit rebuilt")))
    same(saved, loaded)
    check("Memory-mapped samples/layout round trip", info["hit"])
    loaded["samples"][0].card_features[0, 0] += 123
    reread, _ = cache.get_or_build(context, lambda: payload)
    same(reread, payload)
    check("Private mappings cannot mutate persisted cache")
    del loaded, reread
    gc.collect()
    key = context_key(context)
    metadata_path = cache.directory / (key + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    blob = cache.directory / metadata["payload"]["data_file"]
    with blob.open("r+b") as stream:
        byte = stream.read(1)
        stream.seek(0)
        stream.write(bytes([byte[0] ^ 1]))
    recovered, info = cache.get_or_build(context, lambda: payload)
    same(recovered, payload)
    check("Corrupted binary is rebuilt", not info["hit"])
    metadata_path.write_text("{incomplete", encoding="utf-8")
    recovered, info = cache.get_or_build(context, lambda: payload)
    same(recovered, payload)
    check("Incomplete manifest is rebuilt", not info["hit"])
    check("Corruption recovery keeps disk budget", cache.disk_bytes() <= cache.max_bytes)
    for label, field in (("source", "sources"), ("data", "data_files"), ("revision", "dataset_revision"),
                         ("split", "row"), ("chunk", "chunk_reviews"), ("end", "end"), ("layout", "policy")):
        changed = deepcopy(context)
        if field == "row":
            changed[field]["validation_range"] = [1, 2]
        elif field in ("chunk_reviews", "end"):
            changed[field] += 1
        elif isinstance(changed[field], dict):
            changed[field]["test_change"] = "different"
        else:
            changed[field] += "-different"
        check(f"{label} changes invalidate cache key", context_key(changed) != key)
    tiny = PreprocessingCache(output / "tiny-cache", 128)
    result, info = tiny.get_or_build(context, lambda: payload)
    same(result, payload)
    check("Oversized entries return uncached without exceeding budget", not info["persisted"] and tiny.disk_bytes() == 0)
    row = selected[0]
    descriptor = {"directory": str(cache.directory), "max_bytes": cache.max_bytes, "dataset": str(dataset), "context": context}
    options = {"preprocessing_cache": True, "batched_id_encodings": True, "gpu_packed_features": True,
               "preprocessing_cache_mode": "warm_required"}
    warm = prepare_job(row, 0, 129, 2025, options, cache_path=descriptor)
    check("Warm-required path consumes existing entry", warm.cache_info["hit"])
    empty = {**descriptor, "directory": str(output / "empty-warm-cache")}
    try:
        prepare_job(row, 0, 129, 2025, options, cache_path=empty)
    except RuntimeError:
        check("Warm-required path refuses implicit rebuild")
    else:
        raise AssertionError("Missing warm cache entry was silently built")
    # Use scalar fixtures to check LRU pressure without keeping file maps alive.
    bounded = PreprocessingCache(output / "bounded-cache", 8000)
    for index in range(12):
        changed = {**context, "test_index": index}
        bounded.get_or_build(changed, lambda: {"tensor": torch.arange(128, dtype=torch.float32)})
        check("LRU disk budget enforced", bounded.disk_bytes() <= bounded.max_bytes)
    check("LRU removed old entries", len(list(bounded.directory.glob("*.json"))) < 12)
    fixture = output / "data-fingerprint"
    for table in ("revlogs", "cards", "decks"):
        partition = fixture / table / "user_id=1"
        partition.mkdir(parents=True)
        (partition / "fixture.parquet").write_bytes(b"original")
    first = user_data_identity(fixture, 1)
    file = fixture / "cards/user_id=1/fixture.parquet"
    stat = file.stat()
    file.write_bytes(b"modified")
    os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    check("Data fingerprint uses contents even with unchanged size/mtime", first != user_data_identity(fixture, 1))
    print("CPU cache round-trip, RNG preservation, invalidation, corruption and byte-budget checks passed", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from scripts.gpu_r01_prefetch import training_batches
    boundary_rows = [dict(row, train_range=[0, 384]) for row in selected]
    descriptors, report["boundary_identity"] = plan_cache(boundary_rows, 256, output / "worker-cache", dataset,
                                                           protocol["dataset_revision"], 8 * 1024**2)
    options = {"prefetch_users": True, "preprocessing_cache": True, "batched_id_encodings": True,
               "gpu_packed_features": True, "packed_ipc": True, "preprocessing_cache_mode": "cold_build"}
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"), initializer=cpu_worker_init) as pool:
        actual = []
        for row, item in training_batches(boundary_rows, 256, 2025, pool, options, descriptors, 2):
            end = item.offset + item.reviews
            section = split_frame(get_rwkv_data(dataset, row["user_id"]), row, "train").iloc[item.offset:end]
            reference = packed_reference(row, item.offset, end, item.seed, {"gpu_packed_features": True}, section).batch
            check("Spawned worker exact inputs", fingerprint(item.batch) == fingerprint(reference))
            actual.append([row["user_id"], item.offset, item.reviews, item.seed])
        expected = [[row["user_id"], offset, min(256, 384 - offset), 2025 + row["user_id"] + offset]
                    for row in boundary_rows for offset in (0, 256)]
        check("Ordered queue preserves users, offsets, reviews and seeds", actual == expected)
        warm_options = dict(options, preprocessing_cache_mode="warm_required")
        with patch("scripts.gpu_r01_prefetch.frame_stream", side_effect=AssertionError("Warm cache must skip complete history loads")):
            warm_actual = []
            for row, item in training_batches(boundary_rows, 256, 2025, pool, warm_options, descriptors, 2):
                section = split_frame(get_rwkv_data(dataset, row["user_id"]), row, "train").iloc[item.offset:item.offset + item.reviews]
                reference = packed_reference(row, item.offset, item.offset + item.reviews, item.seed,
                                             {"gpu_packed_features": True}, section).batch
                check("Warm spawned worker exact inputs without history loads", fingerprint(item.batch) == fingerprint(reference))
                warm_actual.append([row["user_id"], item.offset, item.reviews, item.seed])
            check("Warm queue preserves user boundaries and order", warm_actual == expected)
        row = boundary_rows[0]
        descriptor = descriptors[row["user_id"], 0]
        args = (row, 0, 256, 2025 + row["user_id"], warm_options, None, descriptor)
        first, second = [future.result() for future in (pool.submit(worker_prepare, *args), pool.submit(worker_prepare, *args))]
        check("Concurrent cache readers preserve values and RNG", first["fingerprint"] == second["fingerprint"] and first["rng"] == second["rng"])
    report["boundary_queue"] = actual
    print("Two spawned CPU workers and ordered user-boundary queue passed", flush=True)

    # Small local CPU preparation timing screen, no cProfile and no GPU.
    row = selected[0]
    end = min(8192, len(sections[row["user_id"]]))
    section = sections[row["user_id"]].iloc[:end]
    descriptor = {"directory": str(output / "timing-cache"), "max_bytes": 32 * 1024**2, "dataset": str(dataset),
                  "context": cache_context(protocol["dataset_revision"], user_data_identity(dataset, row["user_id"]), source_identity(), row, 0, end)}
    seed = 2025 + row["user_id"]
    variants = {"control": {"gpu_packed_features": True},
                "F19 (batched ID-encoding draws)": {"gpu_packed_features": True, "batched_id_encodings": True},
                "F02 (preprocessing cache)": {"gpu_packed_features": True, "preprocessing_cache": True},
                "F02+F19 (cache and batched encoding draws)": {"gpu_packed_features": True, "preprocessing_cache": True, "batched_id_encodings": True}}
    cold = prepare_job(row, 0, end, seed, variants["F02 (preprocessing cache)"], cache_path=descriptor)
    report["cpu_timing"] = {"reviews": end, "user_id": row["user_id"], "cold_cache_seconds": cold.preparation_seconds,
                           "cold_cache": cold.cache_info, "variants": {},
                           "scope": "Three alternating timings per variant, one CPU thread/user/prefix; excludes dataset load, manifest fingerprints, IPC, pinning, GPU and training"}
    values = {name: [] for name in variants}
    reference_hash = fingerprint(cold.batch)
    for cycle in range(3):
        names = list(variants)
        if cycle % 2:
            names.reverse()
        for name in names:
            options = variants[name]
            item = (packed_reference(row, 0, end, seed, options, section) if name == "control" else
                    prepare_job(row, 0, end, seed, options, section if not options.get("preprocessing_cache") else None,
                                descriptor if options.get("preprocessing_cache") else None))
            check("Timing-screen inputs remain exact", fingerprint(item.batch) == reference_hash)
            values[name].append(item.preparation_seconds)
    report["cpu_timing"]["variants"] = {name: {"seconds": durations, "median_seconds": statistics.median(durations)}
                                              for name, durations in values.items()}
    from scripts.run_gpu_r01 import OPTIONS
    from scripts.gpu_r01_policy import require_active
    check("Accepted presets stay unchanged", not OPTIONS["F01"].get("preprocessing_cache") and not OPTIONS["F01-F16-F14-F03-F04"].get("batched_id_encodings"))
    for model in ("baseline", "expertium"):
        try:
            require_active(model, 3)
        except RuntimeError:
            check("Stopped loop stays stopped")
        else:
            raise AssertionError("Stopped model was reopened")
    from scripts.gpu_r01_policy import latest_streak, round_boundary
    round_root = output / "round-fixture"
    old = round_root / "artifacts/retraining/gpu-retrain-r01/iteration-0006/comparison.json"
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps({"comparisons": {"baseline": {"no_20_percent_gain_streak": 3}}}))
    check("Historical stopping state remains visible", latest_streak(round_root, "baseline") == 3)
    check("Explicit new round has fresh stopping state", latest_streak(round_root, "baseline", 0, "0007") == 0)
    current = round_root / "artifacts/retraining/gpu-retrain-r01/iteration-0007/comparison.json"
    current.parent.mkdir(parents=True)
    current.write_text(json.dumps({"comparisons": {"baseline": {"no_20_percent_gain_streak": 3}}}))
    check("Resumed round still enforces three-miss stopping", latest_streak(round_root, "baseline", 0, "0007") == 3)
    try:
        round_boundary({"iteration": "0007", "round_start_iteration": "0007"})
    except ValueError:
        check("Round reset without authorization is rejected")
    else:
        raise AssertionError("Missing resumption authorization accepted")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        parser.error("Use a fresh directory below workspace artifacts")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    os.environ["GPU_BATCH1_DATA_PATH"] = str(ROOT / protocol["dataset_path"])
    output.mkdir(parents=True)
    report = {"status": "running", "provenance": process_provenance(), "protocol": str(args.protocol),
              "scope": "CPU staging checks and local preparation timings only; no CUDA, training or held-out quality"}
    started = time.perf_counter()
    try:
        import torch
        torch.set_num_threads(1)
        with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU suite must not initialize CUDA")) as guard:
            run_checks(output, protocol, report)
            report["cuda_initialization_attempts"] = guard.call_count
            assert guard.call_count == 0 and not torch.cuda.is_initialized()
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        report["diagnostic_seconds"] = time.perf_counter() - started
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Passed {len(report['checks'])} CPU checks: {output / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
