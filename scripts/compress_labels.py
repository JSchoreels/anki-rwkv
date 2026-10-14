"""Build benchmark test-review labels (FSRS-6 label filter, as the 8k run's label-db) for cohort-B users that the run's
label DB lacks (its train users). Writes artifacts/compression/label-db-b (LMDB); resumable (process() skips users
already present). CPU only, low priority.

usage: compress_labels.py --count 1000 --processes 2
"""
import argparse
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

TOML = ROOT / "artifacts" / "compression" / "labels-b.toml"
DB = ROOT / "artifacts" / "compression" / "label-db-b"


def job(user):
    sys.argv = [sys.argv[0], "--config", str(TOML)]
    from rwkv import find_equalize_test_reviews as labels
    labels.process(user)
    return user


def init():
    sys.argv = [sys.argv[0], "--config", str(TOML)]
    from rwkv import find_equalize_test_reviews as labels
    labels.set_low_priority()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--processes", type=int, default=2)
    args = parser.parse_args()
    import lmdb
    from scripts import compress_common as cc
    run = cc.run_info()
    TOML.parent.mkdir(parents=True, exist_ok=True)
    TOML.write_text(f'DATA_PATH = "{cc.data_path(run).as_posix()}"\nLABEL_FILTER_LMDB_PATH = "{DB.as_posix()}"\n'
                    'LABEL_FILTER_LMDB_SIZE = 10_000_000_000\nALGO = "FSRS-6"\nSHORT = true\nSECS = true\n'
                    'USER_START = 1\nUSER_END = 10000\nPROCESSES = 2\n', encoding="utf-8")
    env = lmdb.open(str(cc.RUN / "label-db"), readonly=True, lock=False)
    with env.begin() as txn:
        users = [u for u in cc.b_users(run=run)[:args.count] if txn.get(f"{u}_review_ths".encode()) is None]
    env.close()
    started = time.time()
    print(f"[labels] {len(users)} users to label", flush=True)
    with multiprocessing.Pool(args.processes, initializer=init) as pool:
        for done, _ in enumerate(pool.imap_unordered(job, users, chunksize=2), 1):
            if done % 50 == 0:
                print(f"[labels] {done}/{len(users)} users, {time.time() - started:.0f}s", flush=True)
    print(json.dumps({"labelled": len(users), "seconds": round(time.time() - started)}), flush=True)


if __name__ == "__main__":
    main()
