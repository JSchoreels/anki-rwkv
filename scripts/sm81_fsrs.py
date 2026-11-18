"""Scheduling-time vs review-time accuracy (SM81): FSRS-7 and FSRS7-Instant per-review predictions on the 732 set.

Runs with srs-benchmark's own code and venv (C:/Users/jonat/workspace/srs-benchmark, branch fsrs-instant, read only:
imported, never modified; PYTHONDONTWRITEBYTECODE=1) on the SAME dataset copy as the RWKV scorer
(data/anki-revlogs-10k-original-20261011), so review_th (the row number in the user's revlog file) is the same key on
both sides. Per user (the published FSRS-7 protocol: --algo FSRS-7 --short --secs --equalize_test_with_non_secs):
  test reviews  the benchmark's 5 time-series test folds (TimeSeriesSplit over the non-seconds rows; so only reviews
                whose card was last reviewed on an earlier day), union, by review_th;
  fsrs7_fit     per fold, FSRS-7 fitted on the user's reviews before that fold (fsrs-rs trainer tools/fit-v7, recency
                weight 0.0667 + 0.9333 (i/N)^11.25 = the published FSRS-7 recency row), predicting the fold;
  fsrs7_default FSRS-7 with its default parameters (no fit);
  fi_*          FSRS7-Instant as published in runs/shared-5k (protocol B: one shared FSRS-7 fit on 100 training users,
                runs/shared-5k/shared.json; online updates over the user's whole stream; config "FSRS7-Instant" of
                runs/shared.json), in its three modes: instant (review time), scheduling (scheduling time), start of
                day (parameters / offsets frozen at the first review of the review's day; the card's own history is
                always complete).
All FSRS predictions are scheduling-time predictions except fi_instant / fi_sod (FSRS-7 has no updates: one prediction).
Output: <out>/<user>.npz (review_th, y, fold, elapsed_days, p_* float64) and <out>/users.jsonl (counts, seconds).

  env SM81_OUT=... SM81_USERS=file|ids [SM81_PROCESSES=6] <srs-benchmark venv python> sm81_fsrs.py
"""
import json
import os
import sys
import time
from pathlib import Path

SRS = Path(os.environ.get("SM81_SRS", "C:/Users/jonat/workspace/srs-benchmark"))
FSRS_RS = Path(os.environ.get("SM81_FSRS_RS", "C:/Users/jonat/workspace/fsrs-rs"))
os.environ.setdefault("FI_DATA", "C:/Users/jonat/workspace/anki-rwkv/data/anki-revlogs-10k-original-20261011")
os.environ.setdefault("FI_FITTER", "rs")
os.environ.setdefault("FR_FIT", str(FSRS_RS / "tools/instant-replay/target/release/fit-v7.exe"))
os.environ.setdefault("FI_REPLAY", str(FSRS_RS / "tools/instant-replay/target/release/instant-replay.exe"))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(SRS))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import fsrs_instant as fi  # noqa: E402  (sets the FSRS-7 benchmark flags; argv is replaced)

SHARED = SRS / "runs/shared-5k/shared.json"
FI_CONFIG = next(c for c in json.loads((SRS / "runs/shared.json").read_text()) if c["name"] == "FSRS7-Instant")


def predict(weights, frame):
    return np.asarray(fi.Collection(fi.create_model(fi.config, weights), fi.config).batch_predict(frame)[0], dtype=np.float64)


def process_user(user):
    t0 = time.perf_counter()
    data = fi.UserDataLoader(fi.config).load_user_data(user)
    t_load = time.perf_counter() - t0
    # FSRS-7 per-user fit, the published protocol (fsrs_instant._fold_weights with FI_FITTER=rs)
    w_list, testsets = fi._fold_weights(data)
    t_fit = time.perf_counter() - t0 - t_load
    parts = []
    for k, (w, test) in enumerate(zip(w_list, testsets)):
        part = test[["review_th", "y", "elapsed_days"]].copy()
        part["fold"] = k
        part["p_fsrs7_fit"] = predict(w, test)
        part["p_fsrs7_default"] = predict(None, test)
        parts.append(part)
    test_df = pd.concat(parts)
    assert test_df["review_th"].is_unique and test_df["review_th"].is_monotonic_increasing
    # FSRS7-Instant, protocol B (fsrs_instant_shared.process_user): shared start, the whole stream, three modes
    params = np.array([json.loads(SHARED.read_text())["parameters"]], dtype=np.float32)
    raw_day = pd.read_parquet(Path(fi.config.data_path) / "revlogs" / f"user_id={user}",
                              columns=["day_offset"])["day_offset"].to_numpy()
    first = int(data["review_th"].min())
    st = fi._stream(user, data, [None], [data[data["review_th"] == first]], params)
    st["day"] = raw_day[st["review_th"] - 1]
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
        fi._write_input(src, st, [FI_CONFIG])
        subprocess.run([fi.REPLAY, str(src), str(dst)], check=True)
        raw = np.fromfile(dst, dtype="<f8")
    n = len(st["review_th"])
    modes = raw[: 3 * n].reshape(3, n)
    pos = {r: i for i, r in enumerate(st["review_th"])}
    idx = np.array([pos[r] for r in test_df["review_th"]])
    out = {"review_th": test_df["review_th"].to_numpy(np.int64), "y": test_df["y"].to_numpy(np.int8),
           "fold": test_df["fold"].to_numpy(np.int8), "elapsed_days": test_df["elapsed_days"].to_numpy(np.int64),
           "p_fsrs7_default": test_df["p_fsrs7_default"].to_numpy(), "p_fsrs7_fit": test_df["p_fsrs7_fit"].to_numpy(),
           "p_fi_instant": modes[0][idx], "p_fi_sched": modes[1][idx], "p_fi_sod": modes[2][idx]}
    info = {"user": user, "test_reviews": int(len(test_df)), "stream_reviews": int(n), "load_secs": round(t_load, 2),
            "fit_secs": round(t_fit, 2), "secs": round(time.perf_counter() - t0, 2),
            "fold_sizes": [int(len(t)) for t in testsets], "parameters_fit": w_list}
    return user, out, info


def main():
    out = Path(os.environ["SM81_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    spec = os.environ["SM81_USERS"]
    if os.path.exists(spec):
        text = Path(spec).read_text()
        users = json.loads(text)["users"] if text.lstrip().startswith("{") else [int(x) for x in text.split()]
    else:
        users = [int(x) for x in spec.split(",") if x]
    done = set()
    log = out / "users.jsonl"
    if log.exists():
        done = {json.loads(line)["user"] for line in log.open()}
    todo = [u for u in users if u not in done]
    print(f"[sm81-fsrs] {len(users)} users, {len(todo)} to do, data {fi.config.data_path}, fitter {os.environ['FI_FITTER']}",
          flush=True)
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    mp.set_start_method("spawn", force=True)
    try:
        import psutil
        psutil.Process().nice(getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 10))
    except Exception:  # noqa: BLE001
        pass
    failed = 0
    with ProcessPoolExecutor(int(os.environ.get("SM81_PROCESSES", "6"))) as ex:
        futures = {ex.submit(process_user, u): u for u in todo}
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                user, arrays, info = fut.result()
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"[sm81-fsrs] error user {futures[fut]}: {e!r}", flush=True)
                with open(out / "errors.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"user": futures[fut], "error": repr(e)}) + "\n")
                continue
            np.savez_compressed(out / f"{user}.npz", **arrays)
            with open(log, "a", encoding="utf-8") as f:   # written last: a user in users.jsonl has its file
                f.write(json.dumps(info) + "\n")
            print(f"[sm81-fsrs] {k}/{len(todo)} user {user}: {info['test_reviews']} test reviews, {info['secs']} s", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
