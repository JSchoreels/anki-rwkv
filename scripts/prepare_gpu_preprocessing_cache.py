"""Build F02 preprocessing entries on CPU; optionally check F19 encoding staging."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--users", type=int, nargs="+", required=True)
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--max-reviews", type=int)
    budget.add_argument("--full-history", action="store_true")
    parser.add_argument("--span", type=int, default=32768)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--max-cache-mib", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batched-id-encodings", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists() or args.span < 128 or args.span % 128:
        parser.error("Use a fresh artifact output and a positive span divisible by 128")
    if args.max_reviews is not None and args.max_reviews < 1:
        parser.error("Review limit must be positive")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    selected = {row["user_id"]: row for row in protocol["rows"]}
    if len(set(args.users)) != len(args.users) or any(user not in selected for user in args.users):
        parser.error("Select distinct training users from the supplied protocol")
    rows = [dict(selected[user]) for user in args.users]
    if args.max_reviews is not None:
        for row in rows:
            begin, end = row["train_range"]
            row["train_range"] = [begin, min(end, begin + args.max_reviews)]
    output.mkdir(parents=True)
    report = {"status": "running", "provenance": process_provenance(), "rows": rows,
              "scope": "CPU cache preparation only; no CUDA, training, predictive quality or training throughput",
              "cases": []}
    started = time.perf_counter()
    try:
        import torch
        torch.set_num_threads(1)
        with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU preparation must not initialize CUDA")) as guard:
            dataset = (ROOT / protocol["dataset_path"]).resolve()
            os.environ["GPU_BATCH1_DATA_PATH"] = str(dataset)
            from scripts.gpu_preprocessing_cache import plan_cache
            from scripts.gpu_preprocessing_stage import prepare_job
            paths, report["identity"] = plan_cache(rows, args.span, args.cache, dataset,
                                                   protocol["dataset_revision"], args.max_cache_mib * 1024**2)
            options = {"preprocessing_cache": True, "gpu_packed_features": True,
                       "batched_id_encodings": args.batched_id_encodings}
            for row in rows:
                length = row["train_range"][1] - row["train_range"][0]
                for offset in range(0, length, args.span):
                    end = min(length, offset + args.span)
                    seed = 2025 + row["user_id"] + offset
                    item = prepare_job(row, offset, end, seed, options, cache_path=paths[row["user_id"], offset])
                    report["cases"].append({"user_id": row["user_id"], "offset": offset, "end": end,
                                            "seed": seed, "preparation_seconds": item.preparation_seconds,
                                            "cache": item.cache_info})
                    print(f"CPU cache user {row['user_id']}, offset {offset}: {'hit' if item.cache_info['hit'] else 'built'}", flush=True)
            report["cuda_initialization_attempts"] = guard.call_count
            assert guard.call_count == 0 and not torch.cuda.is_initialized()
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        report["complete_cpu_seconds"] = time.perf_counter() - started
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"CPU cache prepared: {output / 'result.json'}")


if __name__ == "__main__":
    main()
