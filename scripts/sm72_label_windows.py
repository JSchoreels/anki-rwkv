#!/usr/bin/env python
"""SM72 pre-flight (CPU, ~1 min): every update of the big model's leak-free epoch has a labelled review.

The trainer (scripts/gpu_r01_training.py, run_case) stops with "Unlabelled batch user U, offset O" when an update holds
no labelled review. The SM72 run has no replay targets (no teacher), so the label mask is read from the data itself:
rwkv.data_processing.get_rwkv_data keeps the revlog rows in file order (no row dropped, review_th = position + 1) and
sets has_label = 0 exactly at each card's LAST review of the whole history (cumcount(ascending=False) == 0); the
curve and instant masks are both has_label times a query flag, so an update is unlabelled iff every review of its
windows is a last review. The deleted-card transform (RWKV_DELETED_CARDS) changes no card id and no row order.
Updates are laid out as the trainer does: chunk_reviews windows of train_range cut into strided lanes
(scripts/state_carry_train.strided_layout, lanes = batch_chunks).

The first SM72 launch (11:33) stopped at user 8279 (16,387 reviews = 8,192 + 8,192 + 3): its update 1 holds only the
final 3-review window, all last reviews. As in SM55/SM56 (scripts/check_replay_label_windows.py, scripts/sm56_epochs.py),
an unlabelled LAST update made of the user's final window only carries no loss and feeds no later window (its carried
state is never read), so --trim cuts the user's train_range to that window's start; protocol.untrimmed.json keeps the
original and decay.json records "trimmed_unlabelled_tails". Any other unlabelled update is a failure (exit 1).

Also checked per user (get_rwkv_data's assertions and the run's counts): revlog rows == available_reviews, day_offset
non-decreasing, duration >= 0, rating in 1..4.

  python -X utf8 scripts/sm72_label_windows.py --decay DIR [--trim] [--out report.json]
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def scan(job):
    import pyarrow.parquet as pq
    from scripts.state_carry_train import strided_layout
    user, n, start, end, data, chunk, lanes = job
    table = pq.read_table(Path(data) / "revlogs" / f"user_id={user}", columns=["card_id", "day_offset", "duration", "rating"])
    out = {"user_id": user}
    errors = []
    if table.num_rows != n:
        errors.append(f"revlog rows {table.num_rows} != available_reviews {n}")
    day = table["day_offset"].to_numpy()
    if len(day) > 1 and (np.diff(day) < 0).any():
        errors.append("day_offset decreases")
    if len(day) and table["duration"].to_numpy().min() < 0:
        errors.append("negative duration")
    rating = table["rating"].to_numpy()
    if len(rating) and (rating.min() < 1 or rating.max() > 4):
        errors.append("rating outside 1..4")
    cards = table["card_id"].to_numpy()
    last = np.zeros(len(cards), dtype=bool)
    _, first_from_end = np.unique(cards[::-1], return_index=True)
    last[len(cards) - 1 - first_from_end] = True
    labelled = ~last[start:end]
    windows = -(-(end - start) // chunk)
    empty = []
    layout = strided_layout(windows, lanes) if windows else []
    for u, update in enumerate(layout):
        if not any(labelled[w * chunk:(w + 1) * chunk].any() for _, w in update):
            empty.append({"update": u, "windows": [w for _, w in update], "last_update": u == len(layout) - 1})
    if empty:
        tail = empty[-1]
        if (len(empty) == 1 and tail["last_update"] and tail["windows"] == [windows - 1]):
            out["trim_to"] = start + (windows - 1) * chunk
            out["trimmed_reviews"] = end - out["trim_to"]
        else:
            errors.append(f"unlabelled updates {empty}")
    if errors:
        out["errors"] = errors
    return out


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--decay", type=Path, required=True)
    parser.add_argument("--trim", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    decay = json.loads((args.decay / "decay.json").read_text())
    protocol = json.loads((args.decay / "protocol.json").read_text())
    chunk, lanes = decay["options"]["chunk_reviews"], decay["options"]["batch_chunks"]
    if (decay["options"].get("state_carry") or {}).get("lanes") != "strided":
        raise SystemExit("sm72_label_windows covers the carried-state (strided lanes) recipe only")
    jobs = [(r["user_id"], r["available_reviews"], r["train_range"][0], r["train_range"][1], protocol["dataset_path"],
             chunk, lanes) for r in protocol["rows"]]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(scan, jobs, chunksize=16))
    errors = [r for r in results if r.get("errors")]
    trims = [r for r in results if "trim_to" in r]
    report = {"decay": str(args.decay), "users": len(results), "chunk_reviews": chunk, "lanes": lanes,
              "unlabelled_tails": [{k: r[k] for k in ("user_id", "trim_to", "trimmed_reviews")} for r in trims],
              "errors": errors, "pass": not errors}
    if args.trim and not errors:  # records the check even without trims (the launcher refuses to train without it)
        decay = json.loads((args.decay / "decay.json").read_text())
        decay["label_windows"] = {"script": "scripts/sm72_label_windows.py", "users": len(results),
                                  "unlabelled_tails_trimmed": len(trims)}
        (args.decay / "decay.json").write_text(json.dumps(decay, indent=1) + "\n")
    if args.trim and trims and not errors:
        if (args.decay / "protocol.untrimmed.json").exists():
            raise SystemExit("protocol.untrimmed.json exists: trims were applied already")
        shutil.copyfile(args.decay / "protocol.json", args.decay / "protocol.untrimmed.json")
        by_user = {r["user_id"]: r["trim_to"] for r in trims}
        for row in protocol["rows"]:
            if row["user_id"] in by_user:
                row["train_range"] = [row["train_range"][0], by_user[row["user_id"]]]
        (args.decay / "protocol.json").write_text(json.dumps(protocol, indent=1) + "\n")
        decay["trimmed_unlabelled_tails"] = report["unlabelled_tails"]
        decay["training_reviews_after_trims"] = sum(r["train_range"][1] - r["train_range"][0] for r in protocol["rows"])
        (args.decay / "decay.json").write_text(json.dumps(decay, indent=1) + "\n")
        report["trimmed"] = True
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print(json.dumps({k: (v if k != "unlabelled_tails" else len(v)) for k, v in report.items()}, indent=1)[:3000])
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
