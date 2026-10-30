#!/usr/bin/env python
"""Data augmentation (SM67): the note split (artifacts/small-model/sm67/DESIGN.md on main).

A large training user becomes two pseudo-users, each holding a random half of the user's notes (a deleted card, which
has no note, counts as a note of its own and is split by card id). Every card keeps its whole review sequence, so its
own inputs and its labels are unchanged; only the cross-card context (reviews today, other reviews since, the deck,
preset and user streams) sees half the collection.

Trainer option (decay.json / run.json "options"; default absent = today's training, bit for bit):

    "augment": {"kind": "note_split", "min_reviews": 65536, "share": 0.5, "seed": 67}

- Candidates: training users with >= min_reviews reviews that are in no evaluation set (the 464 / 732 / 2,225 pools,
  the 8k split's validation and final-test users). The 2,225 pool's 1,493 extra users are final-tier training users:
  they stay in training whole, as in the control, but are never split.
- `share` of the candidates (ranked by a hash of seed and user, exact count round(share x n)) are split; the others
  stay whole. Each split row is replaced in place by its two halves (part 0 then part 1), so the user order is kept.
- Pseudo-user ids are 10,000,000 + 4 x user + part (never a real id). A pseudo row carries
  "augment": {"kind", "seed", "source_user", "part"}; the frame is get_rwkv_data(source_user, keep=<half mask>), the
  item seed (id codes, day baseline, dropout) uses the real user (scripts/gpu_r01_prefetch.py), and the replay
  targets are the real user's, looked up through the half's original review numbers (scripts/expertium_options.py).

  python -X utf8 scripts/augment.py smoke-folder --decay DIR --out DIR2   (2 pseudo rows of one split user first;
                                                                          for `expertium_decay.py smoke`)
"""
from argparse import ArgumentParser
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PSEUDO_BASE = 10_000_000
SPEC_KEYS = {"kind": str, "min_reviews": int, "share": float, "seed": int}


def validate(spec):
    if not isinstance(spec, dict) or set(spec) != set(SPEC_KEYS):
        raise ValueError(f"augment needs exactly the keys {sorted(SPEC_KEYS)}: {spec}")
    if spec["kind"] != "note_split":
        raise ValueError(f"unknown augment kind {spec['kind']!r} (only note_split)")
    if type(spec["min_reviews"]) is not int or spec["min_reviews"] < 2 or type(spec["seed"]) is not int:
        raise ValueError(f"augment min_reviews / seed must be integers (min_reviews >= 2): {spec}")
    if not isinstance(spec["share"], (int, float)) or not 0 < spec["share"] <= 1:
        raise ValueError(f"augment share must lie in (0, 1]: {spec}")
    return spec


def encode(user, part):
    if not 0 <= int(user) < PSEUDO_BASE // 4 or part not in (0, 1):
        raise ValueError(f"cannot encode user {user} part {part}")
    return PSEUDO_BASE + 4 * int(user) + part


def decode(user_id):
    """(real user, part) of a pseudo id; (user, None) for a real id."""
    user_id = int(user_id)
    if user_id < PSEUDO_BASE:
        return user_id, None
    user, part = divmod(user_id - PSEUDO_BASE, 4)
    if part not in (0, 1):
        raise ValueError(f"{user_id} is not a pseudo-user id")
    return user, part


def _hash(text):
    return hashlib.blake2b(text.encode(), digest_size=8).digest()


def held_out_users(include_2225=False):
    """Held-out users, never in training: the split's validation and final-test users and the 464 / 732 pools. With
    include_2225 also the 2,225 pool's extra users (final-tier training users: trained whole, never split)."""
    sys.path.insert(0, str(ROOT))
    from scripts.eval_pools import pool_users, split_info
    _, split, _ = split_info()
    held = set(split["validation_users"]) | set(split["final_test_users"]) | set(split.get("preserved_final_test_users", []))
    for pool in (464, 732) + ((2225,) if include_2225 else ()):
        held |= set(pool_users(pool))
    return {int(u) for u in held}


def selected_users(counts, spec, held_out):
    """The users to split: share of the eligible (>= min_reviews, not held out), ranked by a hash of (seed, user)."""
    eligible = sorted(int(u) for u, n in counts.items() if n >= spec["min_reviews"] and int(u) not in held_out)
    ranked = sorted(eligible, key=lambda u: _hash(f"{spec['seed']}:select:{u}"))
    return set(ranked[:int(round(spec["share"] * len(eligible)))]), eligible


@lru_cache(maxsize=8)
def review_parts(data, user, seed):
    """0 / 1 per raw revlog row of `user` (the order get_rwkv_data reads): a hash of (seed, user, note); a card
    without a note in the cards table (deleted) is split by its card id."""
    import pandas as pd
    from rwkv.data_processing import read_user_metadata
    data = Path(data)
    cards = pd.read_parquet(data / "revlogs" / f"user_id={user}", columns=["card_id"])["card_id"].to_numpy()
    table = read_user_metadata(data / "cards", user)
    note_of = {int(c): n for c, n in zip(table["card_id"].to_numpy(), table["note_id"].to_numpy())}
    unique, inverse = np.unique(cards, return_inverse=True)
    part = np.empty(len(unique), dtype=np.int8)
    for i, card in enumerate(unique.tolist()):
        note = note_of.get(int(card))
        key = f"{seed}:{user}:n{int(note)}" if note is not None and note == note else f"{seed}:{user}:c{int(card)}"
        part[i] = _hash(key)[0] & 1
    out = part[inverse]
    out.flags.writeable = False
    return out


def keep_mask(data, user, part, seed):
    return review_parts(str(data), int(user), int(seed)) == part


def kept_review_ths(data, user, part, seed):
    """Original review numbers (1-based) of the half's reviews, in order: half review k is original kept[k - 1]."""
    return np.flatnonzero(keep_mask(data, user, part, seed)) + 1


def row_keep(data, row):
    """The keep mask of a pseudo row (checked against its id and length)."""
    spec = row["augment"]
    if spec.get("kind") != "note_split" or encode(spec["source_user"], spec["part"]) != row["user_id"]:
        raise ValueError(f"inconsistent augment row {row}")
    mask = keep_mask(data, spec["source_user"], spec["part"], spec["seed"])
    if int(mask.sum()) != row["available_reviews"]:
        raise ValueError(f"pseudo-user {row['user_id']}: {int(mask.sum())} reviews, row says {row['available_reviews']}")
    return mask


def expand_rows(rows, spec, data, held_out, never_split, workers=1):
    """Replace each selected row by its two halves (same position). held_out: refused in the rows; never_split (a
    superset: every evaluation-set user, the 2,225 pool included): never split. Returns (rows, report)."""
    validate(spec)
    if not held_out <= never_split:
        raise ValueError("never_split must contain every held-out user")
    counts = {int(r["user_id"]): r["available_reviews"] for r in rows}
    bad = sorted(set(counts) & held_out)
    if bad:
        raise ValueError(f"{len(bad)} training rows are held-out users, e.g. {bad[:5]}")
    if any(decode(u)[1] is not None for u in counts):
        raise ValueError("rows are already augmented")
    chosen, eligible = selected_users(counts, spec, never_split)
    sizes = {}
    todo = sorted(chosen)
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as pool:
            for user, n0 in zip(todo, pool.map(_part0_size, [(str(data), u, spec["seed"]) for u in todo], chunksize=16)):
                sizes[user] = n0
    else:
        sizes = {u: _part0_size((str(data), u, spec["seed"])) for u in todo}
    out = []
    for row in rows:
        user = int(row["user_id"])
        if user not in chosen:
            out.append(row)
            continue
        n = row["available_reviews"]
        if row["train_range"] != [0, n] or row.get("validation_range", [0, n]) != [0, n]:
            raise ValueError(f"user {user}: a split row must cover the whole history ({row})")
        for part, m in ((0, sizes[user]), (1, n - sizes[user])):
            if m == 0:
                raise ValueError(f"user {user}: empty half {part}")
            out.append({"user_id": encode(user, part), "available_reviews": m, "train_range": [0, m], "validation_range": [0, m],
                        "augment": {"kind": spec["kind"], "seed": spec["seed"], "source_user": user, "part": part}})
    halves = [r["available_reviews"] for r in out if "augment" in r]
    report = {"spec": spec, "candidates": len(eligible), "split_users": len(chosen), "rows": len(out),
              "pseudo_users": len(halves), "split_reviews": int(sum(halves)),
              "split_review_share": sum(halves) / sum(counts.values()),
              "smallest_half": min(halves) if halves else None, "halves_under_32768": sum(h < 32768 for h in halves),
              "reviews": int(sum(r["available_reviews"] for r in out)),
              "held_out_users_checked": len(held_out), "held_out_overlap": 0, "evaluation_users_never_split": len(never_split),
              "split_users_in_any_evaluation_set": len(chosen & never_split),
              "candidates_excluded_as_held_out": sum(1 for u, n in counts.items() if n >= spec["min_reviews"] and u in never_split)}
    if report["reviews"] != sum(counts.values()):
        raise RuntimeError("the expanded rows do not hold the same reviews")
    return out, report


def _part0_size(job):
    data, user, seed = job
    return int((review_parts(data, int(user), int(seed)) == 0).sum())


def smoke_folder(decay_dir, out):
    """A copy of a prepared decay whose first two rows are the two halves of one split user (the trainer's smoke mode
    trains rows[:2] for 512 reviews each), so `expertium_decay.py smoke --out OUT` exercises the pseudo-user path."""
    import shutil
    decay_dir, out = Path(decay_dir), Path(out)
    protocol = json.loads((decay_dir / "protocol.json").read_text())
    pseudo = [r for r in protocol["rows"] if "augment" in r]
    first = next(r for r in pseudo if r["augment"]["part"] == 0 and r["available_reviews"] >= 1024)
    pair = [first, next(r for r in pseudo if r["user_id"] == first["user_id"] + 1)]
    if pair[1]["available_reviews"] < 1024:
        raise RuntimeError("smoke halves need >= 1,024 reviews each")
    protocol["rows"] = pair + [r for r in protocol["rows"] if r not in pair]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy(decay_dir / "decay.json", out / "decay.json")
    (out / "protocol.json").write_text(json.dumps(protocol, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"smoke_folder": str(out), "rows": [r["user_id"] for r in pair], "source_user": first["augment"]["source_user"]}))
    return 0


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke-folder",))
    parser.add_argument("--decay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    return smoke_folder(args.decay, args.out)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
