#!/usr/bin/env python
"""SM56: label windows of the constant-LR epochs (2,048 x 16, strided lanes) on the 8k final tier, and the epoch hook.

The trainer stops with "Unlabelled batch user U, offset O" when an update holds no labelled review (SM55 ticket 000596).
With "state_carry": {"lanes": "strided"} and batch_chunks L > 1, a user's N windows of chunk_reviews are laid out by
scripts/state_carry_train.strided_layout: update u holds window u of every lane, and the last update holds the last
window of the last m lanes (N = L (U - 1) + m). A window is unlabelled iff all its replay rows are NaN in both the
curve and the rating columns (the SM11 files hold NaN exactly at a card's first review).

  check --experiment DIR [--out report.json]   CPU (~1 min for 8,000 users): per training user of DIR/run/run.json:
        replay file present, rows == review_counts, 5 columns, non-NaN curve/rating counts == the export's progress.jsonl
        record (8k folder for the 3,000 new users, gate5k folder for the 5,000 hard links), and no unlabelled update.
        An unlabelled LAST update made of the user's final window only (m == 1) holds only first reviews at the end of
        the history: it carries no loss and feeds no later review, so the user's train_range is cut to the window's
        start, as SM55's decay did (scripts/check_replay_label_windows.py). Writes DIR/label-window-trims.json.
        Any other unlabelled update is a failure (exit 1).
  hook  "--epoch-runner scripts.sm56_epochs:run_epoch" (scripts/accuracy_loop.py prepare): wraps the per-epoch
        protocol writer to apply DIR/label-window-trims.json to the rows (recorded in protocol["label_window_trims"]),
        then runs the epoch through scripts/run_full_training.run_epoch unchanged. Refuses to run without the trims file.
        SM56_PROTOCOL_ONLY=1 writes the protocol and returns 0 without training (CPU check of the epoch-1 build).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TRIMS = "label-window-trims.json"


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def scan(job):
    from scripts.state_carry_train import strided_layout
    user, n, replay, chunk, lanes = job
    path = Path(replay) / "users" / f"{user}.npz"
    if not path.exists():
        return {"user_id": user, "error": "replay file missing"}
    targets = np.load(path)["targets"]
    if targets.ndim != 2 or targets.shape != (n, 5):
        return {"user_id": user, "error": f"replay shape {targets.shape}, expected ({n}, 5)"}
    curve, rating = ~np.isnan(targets[:, 0]), ~np.isnan(targets[:, 1])
    labelled = curve | rating
    windows = -(-n // chunk)
    layout = strided_layout(windows, lanes)
    bad = []
    for u, update in enumerate(layout):
        if not any(labelled[w * chunk:min(n, (w + 1) * chunk)].any() for _, w in update):
            bad.append({"update": u, "windows": [w for _, w in update], "last": u == len(layout) - 1})
    return {"user_id": user, "reviews": n, "curve": int(curve.sum()), "rating": int(rating.sum()), "windows": windows,
            "updates": len(layout), "unlabelled": bad}


def progress_records(*folders):
    out = {}
    for folder in folders:
        path = Path(folder) / "progress.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if "curve_targets" in record:  # skip failed attempts (an "error" record before the retry's record)
                    out.setdefault(record["user_id"], record)
    return out


def check(args):
    run = json.loads((args.experiment / "run" / "run.json").read_text(encoding="utf-8"))
    options = run["options"]
    replay = Path(options["distillation"]["replay"])
    chunk, lanes = int(options["chunk_reviews"]), int(options.get("batch_chunks", 1))
    if (options.get("state_carry") or {}).get("lanes") != "strided":
        raise SystemExit("this check covers the strided-lane layout only")
    counts = run["review_counts"]
    progress = progress_records(replay, args.gate_replay)
    jobs = [(u, counts[str(u)], str(replay), chunk, lanes) for u in run["train_users"]]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(scan, jobs, chunksize=64))
    fail, trims, no_record = [], [], 0
    for res in results:
        if "error" in res:
            fail.append(f"user {res['user_id']}: {res['error']}")
            continue
        record = progress.get(res["user_id"])
        if record is None:
            no_record += 1
        elif (record.get("curve_targets"), record.get("rating_targets"), record.get("reviews")) != (res["curve"], res["rating"], res["reviews"]):
            fail.append(f"user {res['user_id']}: label counts {res['curve']}/{res['rating']} ({res['reviews']} rows) vs export "
                        f"record {record.get('curve_targets')}/{record.get('rating_targets')} ({record.get('reviews')})")
        for bad in res["unlabelled"]:
            if bad["last"] and bad["windows"] == [res["windows"] - 1] and res["updates"] > 1:
                start = (res["windows"] - 1) * chunk
                trims.append({"user_id": res["user_id"], "from": res["reviews"], "to": start,
                              "dropped_reviews": res["reviews"] - start, "updates_before": res["updates"]})
            else:
                fail.append(f"user {res['user_id']}: unlabelled update {bad['update']} (windows {bad['windows']}), not a trimmable tail")
    if no_record:
        fail.append(f"{no_record} users have no export progress record")
    report = {"experiment": str(args.experiment), "replay": str(replay), "chunk_reviews": chunk, "lanes": lanes,
              "users": len(results), "reviews": sum(r.get("reviews", 0) for r in results),
              "labelled_curve": sum(r.get("curve", 0) for r in results), "labelled_rating": sum(r.get("rating", 0) for r in results),
              "updates_per_epoch_before_trims": sum(r.get("updates", 0) for r in results),
              "trims": trims, "failures": fail, "ok": not fail}
    if not fail:
        atomic_json(args.experiment / TRIMS, {"rule": "unlabelled last update = the user's final window only: train_range cut to "
                                              "its start (scripts/sm56_epochs.py check)", "chunk_reviews": chunk, "lanes": lanes,
                                              "trims": trims})
    if args.out:
        atomic_json(args.out, report)
    print(json.dumps({k: v for k, v in report.items() if k != "failures"} | {"failures": fail[:20]}))
    return 0 if not fail else 1


def run_epoch(run_dir, epoch, experiment, rft):
    """accuracy_loop epoch_runner: the stock epoch with the recorded label-window trims applied to the rows."""
    trims_file = Path(run_dir).parent / TRIMS
    if not trims_file.exists():
        raise SystemExit(f"{trims_file} missing: run `scripts/sm56_epochs.py check --experiment {Path(run_dir).parent}` first")
    trims = {t["user_id"]: t for t in json.loads(trims_file.read_text(encoding="utf-8"))["trims"]}
    inner = rft.write_protocol

    def write_protocol(run, ep, config):
        folder = inner(run, ep, config)
        path = folder / "protocol.json"
        protocol = json.loads(path.read_text(encoding="utf-8"))
        applied = []
        for row in protocol["rows"]:
            t = trims.get(row["user_id"])
            if t:
                if row["train_range"] != [0, t["from"]]:
                    raise RuntimeError(f"user {row['user_id']}: train_range {row['train_range']}, trim expects [0, {t['from']}]")
                row["train_range"] = [0, t["to"]]
                applied.append(t)
        if len(applied) != len(trims):
            raise RuntimeError(f"trims for users not in the epoch rows: {sorted(set(trims) - {t['user_id'] for t in applied})}")
        protocol["label_window_trims"] = applied
        rft.atomic_json(path, protocol)
        return folder

    rft.write_protocol = write_protocol
    if os.environ.get("SM56_PROTOCOL_ONLY") == "1":
        config = json.loads((Path(run_dir) / "run.json").read_text())
        rft.write_protocol(Path(run_dir), epoch, config)
        return 0
    return rft.run_epoch(Path(run_dir), epoch)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check")
    p.add_argument("--experiment", required=True, type=Path)
    p.add_argument("--gate-replay", type=Path,
                   default=Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/fork-fp32-gate5k"))
    p.add_argument("--out", type=Path)
    p.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    return check(args)


if __name__ == "__main__":
    raise SystemExit(main())
