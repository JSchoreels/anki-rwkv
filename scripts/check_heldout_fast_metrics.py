"""SM48 check: scripts/heldout_fast_metrics.py (array path) vs the dict path on real users' labels (CPU, no model).

For each user: the full-history batch (real labels, review_th, has_label, is_query), synthetic float32 head
predictions (seeded), then
  dict path : full_run_metrics.collect_predictions + compress_metrics.metrics (upstream get_stats)
  array path: heldout_fast_metrics.collect_arrays + metrics_arrays
and compares the prediction maps (exact) and the benchmark dicts (==, i.e. bitwise on the rounded floats; the raw
unrounded sklearn values are compared too through get_stats_arrays' inputs). Also times both (CPU, one thread).

usage: check_heldout_fast_metrics.py --users 2388,1425,... --out artifacts/small-model/sm48/check-fast-metrics.json
"""
import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def synthetic_stats(batch, seed):
    labels = batch.labels.float()
    g = torch.Generator().manual_seed(seed)
    shape = labels.shape[:2]
    return SimpleNamespace(has_label=labels[..., 4].int(), is_query=labels[..., 6].int(),
                           label_review_th=batch.label_review_th, label_rating=torch.clamp(labels[..., 3] - 1, min=0),
                           p_curve=torch.rand(shape, generator=g).clamp(1e-4, 1 - 1e-4),
                           p_imm=torch.rand(shape, generator=g).clamp(1e-5, 1 - 1e-5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    import lmdb
    from scripts import compress_common as cc
    from scripts import heldout_fast_metrics as fm
    from scripts.compress_metrics import metrics, read_labels
    from scripts.full_run_metrics import collect_predictions
    run = cc.run_info()
    env = lmdb.open(str(cc.RUN / "label-db"), readonly=True, lock=False, map_size=10_000_000_000)
    with contextlib.redirect_stdout(io.StringIO()):   # warm imports (sklearn/pandas) outside the timings
        metrics(0, ([1, 2], [0, 1]), {"ahead": {1: 0.5, 2: 0.4}, "instant": {1: 0.5, 2: 0.4}, "rating": {1: 1.0, 2: 0.0}})
    rows, ok = [], True
    for user in map(int, args.users.split(",")):
        batch = cc.full_history_batch(cc.data_path(run), user)
        with env.begin() as txn:
            labels = read_labels(txn, user)
        stats = synthetic_stats(batch, user)
        t0 = time.perf_counter()
        old_pred = {"ahead": {}, "instant": {}, "rating": {}}
        collect_predictions(stats, old_pred)
        t1 = time.perf_counter()
        old = metrics(user, labels, old_pred)
        t2 = time.perf_counter()
        new_pred = fm.collect_arrays(stats)
        t3 = time.perf_counter()
        new = fm.metrics_arrays(user, labels, new_pred)
        t4 = time.perf_counter()
        same_pred = new_pred.to_dicts() == old_pred
        same_metrics = json.dumps(new, sort_keys=True) == json.dumps(old, sort_keys=True) and new == old
        ok &= same_pred and same_metrics
        n_bench = sum(old.get(h, {}).get("size", 0) for h in ("ahead", "instant"))
        row = {"user": user, "reviews": run["review_counts"][str(user)], "labelled": int(stats.has_label.sum()),
               "benchmark_entries": n_bench, "same_predictions": same_pred, "same_metrics": same_metrics,
               "dict_collect_s": round(t1 - t0, 4), "dict_metrics_s": round(t2 - t1, 4),
               "array_collect_s": round(t3 - t2, 4), "array_metrics_s": round(t4 - t3, 4),
               "benchmark": old}
        rows.append(row)
        print(f"user {user} ({row['reviews']} reviews): same predictions {same_pred}, same metrics {same_metrics}; "
              f"dict {t2 - t0:.3f}s (collect {t1 - t0:.3f}), array {t4 - t2:.3f}s (collect {t3 - t2:.3f})", flush=True)
    env.close()
    dict_s = sum(r["dict_collect_s"] + r["dict_metrics_s"] for r in rows)
    array_s = sum(r["array_collect_s"] + r["array_metrics_s"] for r in rows)
    reviews = sum(r["reviews"] for r in rows)
    summary = {"passed": ok, "users": len(rows), "reviews": reviews, "dict_seconds": round(dict_s, 3),
               "array_seconds": round(array_s, 3), "dict_us_per_review": round(1e6 * dict_s / reviews, 3),
               "array_us_per_review": round(1e6 * array_s / reviews, 3), "rows": rows}
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
