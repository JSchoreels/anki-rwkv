#!/usr/bin/env python
"""Score a checkpoint on the user's own Anki collection, full layout, CPU only (for scripts/accuracy_loop.py).

Adapted from the gitignored artifacts/collection-eval/2026-10-14/score_collection.py + common.py in the main
checkout (protocol "full": the whole history as one sequence, as the fork replays RWKV). Inputs are read
read-only from that folder: the converted one-user dataset, its FSRS-6 benchmark label LMDB and, when present,
the cached prepared batch (prepared-full-s0.pt). Metrics are upstream rwkv.get_result.get_stats on the 173,690
benchmark test reviews, for the curve ("ahead") and instant heads. The fork's 2.76M RWKV scores curve 0.4417,
instant 0.4366 on the same reviews (full layout).

  accuracy_collection.py --checkpoint master.pth --out collection.json [--threads 2] [--label NAME]
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--threads", type=int, default=2)
parser.add_argument("--label", default="")
parser.add_argument("--collection-dir", default=os.environ.get(
    "ACCURACY_COLLECTION_DIR", r"C:\Users\jonat\workspace\anki-rwkv\artifacts\collection-eval\2026-10-14"))
args = parser.parse_args()
THREADS = max(1, min(4, args.threads))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = str(THREADS)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

torch.set_num_threads(THREADS)
assert not torch.cuda.is_available(), "CUDA must stay hidden"
from scripts.accuracy_eval import benchmark, labels_reader  # noqa: E402
from scripts.full_run_metrics import evaluate_with_predictions  # noqa: E402

COLLECTION = Path(args.collection_dir)
USER = 1
FORK = {"curve": 0.4417, "instant": 0.4366}


def prepared_full():
    cache = COLLECTION / "prepared-full-s0.pt"
    if cache.exists():
        return torch.load(cache, weights_only=False), str(cache)
    from rwkv.data_processing import create_sample, get_rwkv_data
    from rwkv.prepare_batch import prepare
    frame = get_rwkv_data(COLLECTION / "collection-dataset", USER)
    sample = create_sample(USER, frame, [], torch.float32, torch.device("cpu"))
    return [prepare([sample], seed=2025 + USER)], "rebuilt (seed 2026)"


def load_model(path):
    from expertium.architecture import DEFAULT_ANKI_RWKV_CONFIG as config
    from expertium.model.srs_model import SrsRWKV
    model = SrsRWKV(config).float()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    for module in model.modules():
        if hasattr(module, "use_reference_wkv"):
            module.use_reference_wkv = True  # CPU recurrence (the CUDA kernels are unavailable)
    model.eval()
    return model


def main():
    if os.name == "nt":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)  # below normal
    started = time.perf_counter()
    batches, source = prepared_full()
    prep = time.perf_counter() - started
    model = load_model(args.checkpoint)
    t0 = time.perf_counter()
    with torch.inference_mode():
        values, predictions = evaluate_with_predictions(model, ({"batch": b} for b in batches))
    model_seconds = time.perf_counter() - t0
    bench = benchmark(USER, labels_reader(COLLECTION / "collection-label-db")(USER), predictions)
    row = {"label": args.label, "checkpoint": str(Path(args.checkpoint).resolve()), "protocol": "full", "seed": 0,
           "prepared": source, "threads": THREADS, "benchmark_test_reviews": bench["size"],
           "curve": bench.get("ahead"), "instant": bench.get("instant"),
           "coverage": {"curve": bench.get("ahead_coverage"), "instant": bench.get("instant_coverage")},
           "fork_full": FORK, "delta_pct_vs_fork": {h: 100 * (bench[k]["LogLoss"] / FORK[h] - 1) for h, k in (("curve", "ahead"), ("instant", "instant"))},
           "r01_all_labels": {k: values[k] for k in ("curve_labels", "instant_labels", "curve_log_loss", "instant_log_loss")},
           "prepare_seconds": round(prep, 1), "model_seconds": round(model_seconds, 1),
           "total_seconds": round(time.perf_counter() - started, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    Path(args.out).write_text(json.dumps(row, indent=1) + "\n", encoding="utf-8")
    c, i = row["curve"], row["instant"]
    print(f"{args.label} collection full layout ({row['benchmark_test_reviews']} reviews): curve LL {c['LogLoss']:.4f} "
          f"({row['delta_pct_vs_fork']['curve']:+.2f}% vs fork) | instant LL {i['LogLoss']:.4f} "
          f"({row['delta_pct_vs_fork']['instant']:+.2f}% vs fork) | {row['total_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
