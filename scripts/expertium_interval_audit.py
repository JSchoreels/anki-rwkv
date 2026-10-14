#!/usr/bin/env python
"""Item 4 audit (docs/EXPERTIUM_CATCH_UP.md): does this repository's feature pipeline leak the outcome through clocks?

Expertium's clock leak (RWKV_CLOCK_AT_PREV_ANSWER, upstream commit 8914d1ab) lives in his rebuilt "-id" layout,
whose clock columns are measured to a reconstructed show time. This repository trains on the published layout
(rwkv/data_processing.CARD_FEATURE_COLUMNS, 24 card columns), where the interval columns of a query row
(scaled_elapsed_seconds, its sin/cos, the cumulative interval and its sin/cos) are derived from `elapsed_seconds`,
which in the published revlogs runs answer-to-answer and so contains the queried review's own `duration`.
This script measures, on real users, how much of the query rows' interval is that duration and whether the
duration carries outcome information, i.e. whether the same class of leak exists here.

  --users N --seed S --out artifacts/expertium-catch-up/interval-audit.json   (CPU only, one process)
"""
from argparse import ArgumentParser
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_DATA = Path(r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011")


def entropy(p):
    import numpy as np
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--users", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--out", type=Path, default=Path("artifacts/expertium-catch-up/interval-audit.json"))
    args = parser.parse_args()
    import numpy as np
    import pandas as pd
    users = sorted(random.Random(args.seed).sample(range(1, 10001), args.users))
    frames = []
    for user in users:
        df = pd.read_parquet(args.data / "revlogs" / f"user_id={user}", columns=["card_id", "rating", "duration",
                                                                                  "elapsed_days", "elapsed_seconds"])
        df["user_id"] = user
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    query = df[df["elapsed_days"] != -1].copy()            # add_queries makes a query row for every non-first review
    query["fail"] = (query["rating"] == 1).astype(float)
    seconds = query["elapsed_seconds"].astype(float).to_numpy()
    duration = query["duration"].astype(float).to_numpy() / 1000.0
    share = np.where(seconds > 0, duration / np.maximum(seconds, 1e-9), 0.0)
    query["share"] = share
    query["corrected_seconds"] = np.floor(np.maximum(seconds - duration, 0.0))
    out = {"users": users, "data": str(args.data), "query_rows": int(len(query)),
           "published_columns_measured_to_answer_time": ["scaled_elapsed_seconds", "elapsed_seconds_sin",
                                                         "elapsed_seconds_cos", "scaled_elapsed_seconds_cumulative",
                                                         "elapsed_seconds_cumulative_sin", "elapsed_seconds_cumulative_cos"],
           "id_layout_clock_columns_present": False,
           "query_rows_zero_duration_and_state": "rwkv/data_processing.add_queries rejects duration, scaled_duration, "
                                                 "state, scaled_state on query rows"}
    out["fraction_query_rows_duration_ge_10pct_of_interval"] = float((share >= 0.10).mean())
    out["fraction_query_rows_duration_ge_50pct_of_interval"] = float((share >= 0.50).mean())
    short = seconds < 3600
    out["fraction_query_rows_interval_under_1h"] = float(short.mean())
    out["median_duration_share_under_1h"] = float(np.median(share[short])) if short.any() else None
    out["fraction_query_rows_corrected_interval_clamped_to_0"] = float(((seconds - duration) < 0).mean())
    # Answer-to-answer intervals can never be shorter than the review's own duration, except by the floor to whole
    # seconds (< 1 s). Show-to-show intervals would be shorter whenever this review took longer than the last one.
    out["fraction_query_rows_interval_shorter_than_duration_by_over_1s"] = float(((seconds + 1.0) < duration).mean())
    out["fraction_query_rows_interval_shorter_than_duration_by_over_1s_under_1h"] = float(((seconds + 1.0) < duration)[short].mean()) if short.any() else None
    # Outcome information in the duration, among in-session (< 1 h) query rows: P(fail) by duration quartile.
    q = query[short].copy()
    q["duration_quartile"] = pd.qcut(q["duration"].rank(method="first"), 4, labels=False)
    table = q.groupby("duration_quartile").agg(rows=("fail", "size"), p_fail=("fail", "mean"),
                                                median_duration_s=("duration", lambda x: float(np.median(x)) / 1000))
    out["short_interval_p_fail_by_duration_quartile"] = table.reset_index().to_dict(orient="records")
    # Plug-in mutual information between the duration quartile and the outcome (nats per query row, all rows).
    query["duration_decile"] = pd.qcut(query["duration"].rank(method="first"), 10, labels=False)
    base = entropy(query["fail"].mean())
    conditional = sum(len(g) / len(query) * entropy(g["fail"].mean()) for _, g in query.groupby("duration_decile"))
    out["mutual_information_duration_decile_outcome_nats_per_query_row"] = float(base - conditional)
    out["p_fail_query_rows"] = float(query["fail"].mean())
    args.out = args.out if args.out.is_absolute() else ROOT / args.out
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("users",)}, indent=1))


if __name__ == "__main__":
    main()
