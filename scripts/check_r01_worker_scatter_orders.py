"""F53 spawned-worker batch/RNG bytes and stable order/start equivalence."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import multiprocessing
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def checked_pair(arguments):
    import torch
    from scripts.gpu_preprocessing_stage import prepare_job
    from scripts.check_r01_worker_counts import digest
    baseline = prepare_job(*arguments)
    rng_a = digest(torch.random.get_rng_state())
    args = (*arguments[:4], {**arguments[4], "worker_scatter_orders": True}, arguments[5])
    candidate = prepare_job(*args)
    rng_b = digest(torch.random.get_rng_state())
    plain = replace(candidate.batch, packed={k: v for k, v in candidate.batch.packed.items() if not k.startswith("scatter_")})
    equal = digest(baseline.batch) == digest(plain)
    orders_equal, count = True, 0
    rows = candidate.batch.start.shape[0]
    for i, (gathers, shifts, lengths) in enumerate(zip(candidate.batch.sub_gather, candidate.batch.time_shift_selects, candidate.batch.sub_gather_lens)):
        for j, (gather, shift, length) in enumerate(zip(gathers, shifts, lengths)):
            prefix = f"scatter_{i}_{j}_"
            keys = (torch.arange(shift.numel() // length)[:, None] * length + shift.view(-1, length)).reshape(-1)
            for name, values, destinations in (("row", gather, rows), ("shift", keys, keys.numel())):
                sorted_values, order = torch.sort(values, stable=True)
                starts = torch.searchsorted(sorted_values, torch.arange(destinations + 1))
                orders_equal &= torch.equal(order, candidate.batch.packed[prefix + name + "_order"])
                orders_equal &= torch.equal(starts, candidate.batch.packed[prefix + name + "_starts"])
                count += 1
        rows = sum(gather.numel() for gather in gathers)
    return {"user": arguments[0]["user_id"], "offset": arguments[1], "reviews": baseline.reviews,
            "batch_equal": equal, "rng_equal": rng_a == rng_b, "orders_equal": orders_equal, "orders": count}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--options", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists() or not output.resolve().is_relative_to(ROOT / "artifacts"):
        parser.error("Use a new artifact output")
    cohort = json.loads((ROOT / "artifacts/retraining/gpu-retrain-r01/cohorts-user-holdout-v1.json").read_text())
    os.environ["GPU_BATCH1_DATA_PATH"] = str(ROOT / cohort["dataset_path"])
    import torch
    torch.set_num_threads(1)
    from rwkv.data_processing import get_rwkv_data
    from scripts.gpu_batch1 import split_frame
    from scripts.trial_batches import worker_init
    from scripts.experiment_provenance import process_provenance
    options = json.loads((ROOT / args.options).read_text())["c"]
    rows = sorted(cohort["stages"]["100"]["rows"], key=lambda row: row["available_reviews"])
    picks = [(rows[-1], 0, 65536), (rows[-3], 32768, 65536),
             (rows[len(rows) // 2], 0, 32769), (rows[0], 0, 127), (rows[10], 0, 129)]
    jobs = []
    for row, offset, end in picks:
        frame = split_frame(get_rwkv_data(ROOT / cohort["dataset_path"], row["user_id"]), row, "train")
        end = min(len(frame), end)
        jobs.append((row, offset, end, 2025 + row["user_id"] + offset, options, frame.iloc[offset:end]))
    with ProcessPoolExecutor(max_workers=6, mp_context=multiprocessing.get_context("spawn"), initializer=worker_init) as pool:
        results = list(pool.map(checked_pair, jobs * 2))
    passed = all(r["batch_equal"] and r["rng_equal"] and r["orders_equal"] for r in results)
    result = {"status": "passed" if passed else "failed", "provenance": process_provenance(), "options": options,
              "summary": {"jobs": len(results), "all_equal": passed, "orders": sum(r["orders"] for r in results)}, "jobs": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"]))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
