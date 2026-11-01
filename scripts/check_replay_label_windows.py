#!/usr/bin/env python
"""SM55: before a decay starts, check that every training update of every user has labels (CPU, ~10 s for 8,000 users).

The trainer (scripts/gpu_r01_training.py, run_case) stops with "Unlabelled batch user U, offset O" when an update
holds no labelled review. With chunk_reviews = span (batch_chunks 1, strided lanes: 1 lane) update u of a user is
reviews [u x span, min(n, (u + 1) x span)) of its train_range. A review has a label unless it is a card's first review,
and the SM11 replay files store NaN exactly at the reviews without a label (scripts/check_sm11_replay_targets.py
checks this), so a window is unlabelled iff all its replay rows are NaN in both the curve and the rating columns.
SM55 ticket 000596 hit this after 2 h 14 min: user 5599 (32,782 reviews) ends with 14 first reviews, its update 2
was reviews 32,768..32,781 with no label.

Checks per user: replay file present, rows == available_reviews, 5 columns; train_range [0, n] (or a recorded trim);
non-NaN curve/rating counts == the export's progress.jsonl record when there is one; no unlabelled window.

--trim-unlabelled-tail: an unlabelled LAST window holds only first reviews at the end of the user's history; it
carries no loss and feeds no later review, so the user's train_range is cut to the window's start. The original
protocol is kept as protocol.untrimmed.json and decay.json records the trims ("trimmed_unlabelled_tails").
An unlabelled window anywhere else is a failure (exit 1), never trimmed.

  python -X utf8 scripts/check_replay_label_windows.py --decay DIR [--trim-unlabelled-tail] [--out report.json]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np


def scan(job):
    user, n, start, end, replay, span = job
    path = Path(replay) / "users" / f"{user}.npz"
    if not path.exists():
        return {"user_id": user, "error": "replay file missing"}
    targets = np.load(path)["targets"]
    if targets.ndim != 2 or targets.shape != (n, 5):
        return {"user_id": user, "error": f"replay shape {targets.shape}, expected ({n}, 5)"}
    curve, rating = ~np.isnan(targets[:, 0]), ~np.isnan(targets[:, 1])
    labelled = curve | rating
    empty = [o for o in range(start, end, span) if not labelled[o:min(end, o + span)].any()]
    return {"user_id": user, "reviews": n, "train_range": [start, end], "curve": int(curve.sum()),
            "rating": int(rating.sum()), "unlabelled_windows": empty, "last_window": start + (end - start - 1) // span * span}


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--decay", required=True, type=Path)
    parser.add_argument("--trim-unlabelled-tail", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    decay = json.loads((args.decay / "decay.json").read_text(encoding="utf-8"))
    protocol = json.loads((args.decay / "protocol.json").read_text(encoding="utf-8"))
    options = decay["options"]
    replay = Path(options["distillation"]["replay"])
    span = int(options["chunk_reviews"]) * int(options.get("batch_chunks", 1))
    lanes = (options.get("state_carry") or {}).get("lanes")
    if lanes == "strided" and span != int(options["chunk_reviews"]):
        raise SystemExit("strided lanes with several chunks per update: window layout not covered by this check")
    trimmed = {t["user_id"]: t for t in decay.get("trimmed_unlabelled_tails", [])}
    progress = {}
    if (replay / "progress.jsonl").exists():
        for line in (replay / "progress.jsonl").read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            progress[record["user_id"]] = record
    fail, rows = [], protocol["rows"]
    for row in rows:
        allowed_end = trimmed[row["user_id"]]["to"] if row["user_id"] in trimmed else row["available_reviews"]
        if row["train_range"] != [0, allowed_end]:
            fail.append(f"user {row['user_id']}: train_range {row['train_range']}, expected [0, {allowed_end}]")
    jobs = [(r["user_id"], r["available_reviews"], r["train_range"][0], r["train_range"][1], str(replay), span) for r in rows]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(scan, jobs, chunksize=64))
    tails = []
    for res in results:
        if "error" in res:
            fail.append(f"user {res['user_id']}: {res['error']}")
            continue
        record = progress.get(res["user_id"])
        if record and (record.get("curve_targets"), record.get("rating_targets")) != (res["curve"], res["rating"]):
            fail.append(f"user {res['user_id']}: label counts {res['curve']}/{res['rating']} vs export record "
                        f"{record.get('curve_targets')}/{record.get('rating_targets')}")
        for offset in res["unlabelled_windows"]:
            if offset == res["last_window"] and offset > res["train_range"][0]:
                tails.append({"user_id": res["user_id"], "from": res["train_range"][1], "to": offset,
                              "dropped_reviews": res["train_range"][1] - offset})
            else:
                fail.append(f"user {res['user_id']}: unlabelled window at offset {offset} (not a trimmable tail)")
    applied = []
    if tails and args.trim_unlabelled_tail and not fail:
        untrimmed = args.decay / "protocol.untrimmed.json"
        if not untrimmed.exists():
            atomic_json(untrimmed, protocol)
        by_user = {t["user_id"]: t for t in tails}
        for row in rows:
            if row["user_id"] in by_user:
                row["train_range"] = [row["train_range"][0], by_user[row["user_id"]]["to"]]
        atomic_json(args.decay / "protocol.json", protocol)
        decay["trimmed_unlabelled_tails"] = list(trimmed.values()) + tails
        decay["training_reviews_after_trims"] = sum(r["train_range"][1] - r["train_range"][0] for r in rows)
        atomic_json(args.decay / "decay.json", decay)
        applied, tails = tails, []
    elif tails:
        fail.extend(f"user {t['user_id']}: unlabelled last window at {t['to']} ({t['dropped_reviews']} reviews); "
                    f"rerun with --trim-unlabelled-tail" for t in tails)
    report = {"decay": str(args.decay), "replay": str(replay), "span": span, "users": len(rows),
              "reviews_in_train_ranges": sum(r["train_range"][1] - r["train_range"][0] for r in rows),
              "labelled_reviews": sum(r.get("curve", 0) for r in results),
              "trims_recorded": sorted(trimmed), "trims_applied_now": applied, "failures": fail, "ok": not fail}
    if args.out:
        atomic_json(args.out, report)
    print(json.dumps({k: v for k, v in report.items() if k != "failures"} | {"failures": fail[:20]}))
    return 0 if not fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
