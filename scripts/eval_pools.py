"""Evaluation user pools for the small-model track (docs/SMALL_MODEL.md).

  464  : the screen eval users with id < 5000 (artifacts/accuracy/tiers.json), the historical standard.
  732  : every validation user of the 8k split with id < 5000 (fork-unseen). The new standard.
  2225 : the 732 + the 1,493 id < 5000 users of the split's training users that are in the `rest` part of the tier split,
         i.e. not in the gate training tier. INVALID for any model trained on the final tier (all 8,000 train users).
The 500 final_test_users are sealed and never included.

usage: eval_pools.py --write   (rewrites artifacts/small-model/eval-users-{732,2225}.json)
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TIERS = ROOT / "artifacts" / "accuracy" / "tiers.json"
POOLS = (464, 732, 2225)
FORK_SEEN_FROM = 5000


def split_info():
    tiers = json.loads(TIERS.read_text())
    path = Path(tiers["split"])
    return path, json.loads(path.read_text()), tiers


def gate_train_users(split):
    from scripts.accuracy_loop import TIER_SEED
    from scripts.run_full_training import stratified_split
    train = split["train_users"]
    counts = {int(k): v for k, v in split["review_counts"].items()}
    parts = stratified_split(train, counts, {"screen": 2000, "gate_extra": 3000, "rest": len(train) - 5000}, TIER_SEED)
    return set(parts["screen"] + parts["gate_extra"]), set(parts["rest"])


def pool_users(pool):
    """Sorted user ids of the pool (before any ordering)."""
    path, split, tiers = split_info()
    if pool == 464:
        screen = tiers["tiers"]["screen"]
        return sorted(u for u in screen["eval_users"] if u < FORK_SEEN_FROM)
    if pool not in (732, 2225):
        raise SystemExit(f"unknown pool {pool}; choose from {POOLS}")
    users = {u for u in split["validation_users"] if u < FORK_SEEN_FROM}
    if pool == 2225:
        _, rest = gate_train_users(split)
        users |= {u for u in rest if u < FORK_SEEN_FROM}
    return sorted(users - set(split["final_test_users"]))


def write_files():
    path, split, _ = split_info()
    counts = split["review_counts"]
    for pool in (732, 2225):
        users = pool_users(pool)
        doc = {
            "note": "Evaluation pool for full-history scoring (scripts/eval_pools.py, small_model_heldout.py --pool). "
                    "Public benchmark user ids only.",
            "pool": pool,
            "rule": ("every validation user of the 8k split with id < 5000 (fork-unseen)" if pool == 732 else
                     "the 732 validation users plus the split's training users with id < 5000 that are not in the gate "
                     "training tier (the `rest` part of build_tiers, TIER_SEED). Valid only for models trained on the "
                     "gate tier (5,000 users) or less; INVALID for any model trained on the final tier (all 8,000 train "
                     "users). For future reference / decisive checks."),
            "sealed": "final_test_users (500) are never included",
            "split": str(path), "split_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "users_count": len(users), "reviews_total": sum(counts[str(u)] for u in users),
            "users": users,
        }
        out = ROOT / "artifacts" / "small-model" / f"eval-users-{pool}.json"
        out.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"{out}: {doc['users_count']} users, {doc['reviews_total']:,} reviews")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    if ap.parse_args().write:
        write_files()
