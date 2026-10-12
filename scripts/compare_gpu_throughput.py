"""Compare default and Expertium training throughput on identical real histories."""

from argparse import ArgumentParser
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
import torch
from rwkv import data_processing as processing
from benchmark_gpu_trial import fresh_checkpoint, run_variant
from gpu_power import limited_gpu
from scripts.trial_batches import preparation_pool
from scripts.experiment_provenance import process_provenance


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--batch-chunks", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tune-workers", type=int, default=6)
    parser.add_argument("--tune-max", type=int, default=512)
    parser.add_argument("--memory-fraction", type=float, default=0.72)
    args = parser.parse_args()
    provenance = process_provenance()
    if min(args.steps, args.warmup, args.batch_chunks, args.workers, args.tune_workers) <= 0:
        parser.error("steps, warmup, chunks, and workers must be positive")
    if not 0 < args.memory_fraction <= 1:
        parser.error("memory fraction must be in (0, 1]")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    output = ROOT / "artifacts/benchmarks/perf-comparison"
    output.mkdir(parents=True, exist_ok=True)
    source = processing.get_rwkv_data(ROOT / "data/anki-revlogs-10k", 17)
    common = source.iloc[4096:4096 + 8192].copy()
    del source
    checkpoints = {name: fresh_checkpoint(name) for name in ("baseline", "expertium")}
    records = []

    def measure(model, dtype, chunks, workers, pool):
        name = f"{model}-{dtype}-{chunks}-chunks-{workers}-workers"
        needed = (args.warmup + args.steps + 1) * chunks * 128
        frame = pd.concat([common] * ((needed + len(common) - 1) // len(common)), ignore_index=True)
        config = {"MODEL": model, "DTYPE": dtype, "WKV_BACKEND": "nvrtc", "CHUNK_REVIEWS": 128,
                  "BATCH_CHUNKS": chunks, "PREFETCH_WORKERS": workers, "PREFETCH_BATCHES": workers * 2,
                  "REVIEWS_PER_USER": 0, "SEED": 2025}
        try:
            result = run_variant(name, frame, config, checkpoints[model], output,
                                 args.warmup, args.steps, pool)
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower():
                raise
            result = {"name": name, "model": model, "dtype": dtype, "batch_chunks": chunks,
                      "prefetch_workers": workers, "status": "out_of_memory", "error": str(exc)}
            exc.__traceback__ = None
            gc.collect()
            torch.cuda.empty_cache()
            print(f"{name}: exceeds memory budget", flush=True)
        records.append(result)
        (output / "progress.json").write_text(json.dumps(records, indent=2) + "\n")
        return result

    with limited_gpu(0, 0, output):
        with preparation_pool(args.workers) as pool:
            baseline = measure("baseline", "bfloat16", args.batch_chunks, args.workers, pool)
            small = measure("expertium", "bfloat16", args.batch_chunks, args.workers, pool)
            float_small = measure("expertium", "float32", args.batch_chunks, args.workers, pool)
        fastest = max((r for r in (small, float_small) if "reviews_per_second" in r),
                      key=lambda r: r["reviews_per_second"])
        with preparation_pool(args.tune_workers) as pool:
            low = args.batch_chunks
            prior = fastest
            chunks = min(low * 2, args.tune_max)
            while chunks > low:
                candidate = measure("expertium", fastest["dtype"], chunks, args.tune_workers, pool)
                improves = candidate.get("reviews_per_second", 0) > prior["reviews_per_second"] * 1.05
                if not improves:
                    # Refine the first plateau or memory ceiling with its midpoint.
                    midpoint = (low + chunks) // 2
                    if midpoint not in (low, chunks):
                        measure("expertium", fastest["dtype"], midpoint, args.tune_workers, pool)
                    break
                low, prior = chunks, candidate
                chunks = min(chunks * 2, args.tune_max)
    best_small = max((r for r in records if r["model"] == "expertium" and "reviews_per_second" in r),
                     key=lambda r: r["reviews_per_second"])
    report = {"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
              "provenance": provenance,
              "cuda": torch.version.cuda, "initialization": "fresh weights, seed 2025, independent optimizer per scenario",
              "data": "user 17 reviews 4097-12288, repeated; same CPU preparation and labels for both models",
              "chunk_reviews": 128, "gpu_memory_fraction": args.memory_fraction,
              "warmup_updates": args.warmup, "measured_updates": args.steps,
              "timing": "complete updates including CPU preparation waits, transfer, forward/backward and AdamW; excludes file loads and cold setup",
              "matched_baseline": baseline, "matched_expertium": small,
              "matched_speedup": small["reviews_per_second"] / baseline["reviews_per_second"],
              "best_expertium": best_small,
              "tuned_speedup": best_small["reviews_per_second"] / baseline["reviews_per_second"],
              "results": records}
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key in ("matched_speedup", "tuned_speedup")}), flush=True)


if __name__ == "__main__":
    main()
