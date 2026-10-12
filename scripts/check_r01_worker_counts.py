"""Compare E33 packed bytes and post-preparation RNG across spawned worker counts."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields, is_dataclass
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(value):
    import numpy as np
    import torch
    result = hashlib.sha256()
    def add(item):
        if isinstance(item, torch.Tensor):
            result.update(str((item.dtype, tuple(item.shape))).encode())
            result.update(item.contiguous().view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            result.update(str((item.dtype, item.shape)).encode())
            result.update(item.tobytes())
        elif is_dataclass(item):
            for field in fields(item):
                result.update(field.name.encode())
                add(getattr(item, field.name))
        elif isinstance(item, dict):
            for key, child in item.items():
                add(key)
                add(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                add(child)
        else:
            result.update(repr(item).encode())
    add(value)
    return result.hexdigest()


def checked_prepare(arguments):
    import torch
    from scripts.gpu_preprocessing_stage import prepare_job
    item = prepare_job(*arguments)
    return {"offset": item.offset, "seed": item.seed, "reviews": item.reviews,
            "batch_sha256": digest(item.batch), "rng_sha256": digest(torch.random.get_rng_state())}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists() or not output.resolve().is_relative_to(ROOT / "artifacts"):
        parser.error("Use a fresh artifacts output")
    cohort = json.loads((ROOT / "artifacts/retraining/gpu-retrain-r01/cohorts-user-holdout-v1.json").read_text())
    os.environ["GPU_BATCH1_DATA_PATH"] = str(ROOT / cohort["dataset_path"])
    import torch
    torch.set_num_threads(1)
    from rwkv.data_processing import get_rwkv_data
    from scripts.gpu_batch1 import split_frame
    from scripts.trial_batches import worker_init
    from scripts.experiment_provenance import process_provenance
    reference = json.loads((ROOT / "artifacts/retraining/gpu-retrain-r01/iteration-0033/expertium/E30-F37-F42/result.json").read_text())
    rows = sorted(cohort["stages"]["100"]["rows"], key=lambda row: row["available_reviews"])
    picks = [(rows[-1], 0, 65536), (rows[-3], 32768, 65536),
             (rows[len(rows)//2], 0, 32769), (rows[0], 0, 127), (rows[10], 0, 129)]
    jobs = []
    for row, offset, end in picks:
        frame = split_frame(get_rwkv_data(ROOT / cohort["dataset_path"], row["user_id"]), row, "train")
        end = min(len(frame), end)
        jobs.append((row, offset, end, 2025 + row["user_id"] + offset, reference["options"], frame.iloc[offset:end]))
    result = {"provenance": process_provenance(), "scope": "E33; six/five/four spawned workers; ordered results on five real edge windows repeated twice; packed batch bytes and post-preparation RNG; CPU only", "worker_counts": {}}
    for workers in (6, 5, 4):
        candidates = [(*job[:4], {**job[4], "preparation_workers": workers}, job[5]) for job in jobs] * 2
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=worker_init) as pool:
            result["worker_counts"][str(workers)] = list(pool.map(checked_prepare, candidates))
        print(f"[workers] {workers}: {len(candidates)} packed/RNG digests captured", flush=True)
    control = result["worker_counts"]["6"]
    result["status"] = "passed" if all(items == control for items in result["worker_counts"].values()) else "failed"
    result["summary"] = {"jobs_per_count": len(control), "worker_counts": [6, 5, 4], "all_equal": result["status"] == "passed"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "summary": result["summary"]}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
