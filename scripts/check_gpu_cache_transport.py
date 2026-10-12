"""CPU transport of real warm R01 cache entries and cache-maintenance profiling."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
import cProfile
import json
import multiprocessing
import os
from pathlib import Path
import pstats
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def cpu_init():
    import torch
    torch.set_num_threads(1)
    def reject(*args, **kwargs):
        raise AssertionError("Cache transport check must not initialize CUDA")
    torch.cuda._lazy_init = reject


def warm_job(*args):
    import torch
    from scripts.gpu_preprocessing_stage import prepare_job
    item = prepare_job(*args)
    assert not torch.cuda.is_initialized()
    return item


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        parser.error("Use a fresh workspace artifact directory")
    output.mkdir(parents=True)
    from scripts.experiment_provenance import process_provenance
    report = {"status": "running", "provenance": process_provenance(),
              "scope": "CPU warm mapped-storage transport and existing cache metadata profile; no GPU, training or quality validation",
              "checks": []}
    try:
        import torch
        torch.set_num_threads(1)
        protocol = json.loads(args.protocol.read_text())
        dataset = (ROOT / protocol["dataset_path"]).resolve()
        os.environ["GPU_BATCH1_DATA_PATH"] = str(dataset)
        from scripts.gpu_preprocessing_cache import PreprocessingCache, plan_cache
        from scripts.check_gpu_preprocessing_cpu import fingerprint
        from scripts.gpu_batch1 import prepare_job as reference_job, split_frame
        from rwkv.data_processing import get_rwkv_data
        rows = [dict(row, train_range=[0, 512], validation_range=[512, 1024]) for row in protocol["rows"][:2]]
        with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU only")) as guard:
            descriptors, report["identity"] = plan_cache(rows, 16384, args.cache, dataset,
                                                        protocol["dataset_revision"], 16 * 2**30)
            options = {"preprocessing_cache": True, "preprocessing_cache_mode": "warm_required",
                       "batched_id_encodings": True}
            with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"), initializer=cpu_init) as pool:
                futures = [(row, pool.submit(warm_job, row, 0, 512, 2025 + row["user_id"], options,
                                              None, descriptors[row["user_id"], 0])) for row in rows]
                for row, future in futures:
                    item = future.result()
                    frame = split_frame(get_rwkv_data(dataset, row["user_id"]), row, "train")
                    reference = reference_job(row, 0, 512, item.seed, {}, frame).batch
                    assert item.cache_info["hit"] and fingerprint(item.batch) == fingerprint(reference)
                    report["checks"].append("Warm CPU-expanded mapped batch crosses spawned-worker IPC with exact inputs")
            cache = PreprocessingCache(args.cache, 16 * 2**30)
            before = cache.disk_bytes()
            profile = cProfile.Profile()
            started = time.perf_counter()
            with cache._lock():
                result = profile.runcall(cache._evict, 0)
            seconds = time.perf_counter() - started
            assert result and cache.disk_bytes() == before
            profile.dump_stats(str(output / "metadata-profile.prof"))
            with (output / "metadata-profile.txt").open("w") as stream:
                pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(25)
            report["metadata_maintenance"] = {"seconds": seconds, "entries": len(list(cache.directory.glob("*.json"))),
                                               "disk_bytes": before, "scope": "One profiled no-eviction capacity check; includes profiling overhead; not a training saving"}
            assert guard.call_count == 0 and not torch.cuda.is_initialized()
            report["cuda_initialization_attempts"] = guard.call_count
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "checks", "metadata_maintenance", "cuda_initialization_attempts")}))


if __name__ == "__main__":
    main()
