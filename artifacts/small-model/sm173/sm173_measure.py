#!/usr/bin/env python
"""Deleted-card note ids (SM173), phase 1a (CPU): size of the placeholder note-id collapse.

rwkv/data_processing.get_rwkv_data merges the revlogs with the cards table (left join), so a user with at least one card
missing from the cards table (deleted before the export) gets a float64 note_id column; the deleted cards' note id
ID_PLACEHOLDER + card_id (about 3.14e17) is then rounded to a multiple of 64 (the float64 spacing there). The dataset's
card ids are small per-user integers (0, 1, 2, ...), so deleted cards whose ids share a 64-wide block share one note
lane. In addition create_sample casts the float64 column to int32 for the random id codes: every placeholder value is
out of int32 range and becomes -2^31, so ALL deleted cards of a user share one note code (the carried-state trainer
uses int64 ids: one code per 64-block there).

Per user (aggregate counts only, never card ids): reviews, deleted-card reviews and cards, placeholder note lanes, the
deleted cards / reviews whose lane is shared with another card, the largest lane. Pools: the 8,000 training users of
the 8k split (the release models' training users), the 732 set, the 1,500 pool.

  cd C:/Users/jonat/workspace/wt/sm173-run
  python -X utf8 <this file> --out <folder>/sm173-measure.json
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

CODE = Path.cwd()
sys.path.insert(0, str(CODE))


def user_stats(data, user):
    from rwkv.data_processing import ID_PLACEHOLDER, read_user_metadata
    cid = pd.read_parquet(data / "revlogs" / f"user_id={user}", columns=["card_id"])["card_id"].to_numpy()
    cards = read_user_metadata(data / "cards", user)
    deleted = ~np.isin(cid, cards["card_id"].to_numpy())
    out = {"user": int(user), "reviews": int(len(cid)), "deleted_reviews": int(deleted.sum())}
    if not deleted.any():
        return {**out, "deleted_cards": 0, "lanes": 0, "shared_cards": 0, "shared_reviews": 0, "max_lane": 0,
                "all_deleted": False}
    dc = cid[deleted]
    uniq, per_card = np.unique(dc, return_counts=True)
    # exactly what get_rwkv_data stores: int64 ID_PLACEHOLDER + card_id assigned into a float64 column
    lane = (np.int64(ID_PLACEHOLDER) + uniq.astype(np.int64)).astype(np.float64)
    lanes, inv, lane_cards = np.unique(lane, return_inverse=True, return_counts=True)
    shared = lane_cards[inv] > 1
    return {**out, "deleted_cards": int(len(uniq)), "lanes": int(len(lanes)), "shared_cards": int(shared.sum()),
            "shared_reviews": int(per_card[shared].sum()), "max_lane": int(lane_cards.max()),
            "all_deleted": bool(deleted.all())}


def summary(rows):
    n = len(rows)
    rev = sum(r["reviews"] for r in rows)
    aff = [r for r in rows if r["deleted_cards"]]
    shared_users = [r for r in rows if r["shared_cards"]]
    dc = sum(r["deleted_cards"] for r in rows)
    lanes = sum(r["lanes"] for r in rows)
    out = {
        "users": n, "reviews": rev,
        "users_with_deleted_cards": len(aff), "users_with_shared_placeholder": len(shared_users),
        "users_all_cards_deleted": sum(r["all_deleted"] for r in rows),
        "deleted_reviews": sum(r["deleted_reviews"] for r in rows),
        "shared_placeholder_reviews": sum(r["shared_reviews"] for r in rows),
        "reviews_of_users_with_deleted_cards": sum(r["reviews"] for r in aff),
        "deleted_cards": dc, "placeholder_lanes": lanes,
        "shared_placeholder_cards": sum(r["shared_cards"] for r in rows),
        "cards_per_placeholder_lane": round(dc / lanes, 3) if lanes else None,
        "max_cards_on_one_lane": max([r["max_lane"] for r in rows] or [0]),
    }
    out["share_users_with_deleted_cards_pct"] = round(100 * len(aff) / n, 2)
    out["share_users_with_shared_placeholder_pct"] = round(100 * len(shared_users) / n, 2)
    out["share_reviews_deleted_pct"] = round(100 * out["deleted_reviews"] / rev, 3)
    out["share_reviews_shared_placeholder_pct"] = round(100 * out["shared_placeholder_reviews"] / rev, 3)
    out["share_reviews_in_users_with_deleted_cards_pct"] = round(100 * out["reviews_of_users_with_deleted_cards"] / rev, 2)
    out["share_deleted_cards_on_shared_lane_pct"] = round(100 * out["shared_placeholder_cards"] / dc, 2) if dc else None
    per_user = sorted(100 * r["deleted_reviews"] / r["reviews"] for r in aff)
    if per_user:
        out["deleted_review_share_per_affected_user_pct"] = {
            "median": round(float(np.median(per_user)), 2), "p90": round(float(np.percentile(per_user, 90)), 2),
            "max": round(per_user[-1], 2)}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from scripts import compress_common as cc
    from scripts.eval_pools import pool_users
    run = cc.run_info()
    data = cc.data_path(run)
    pools = {"train_8000": sorted(run["train_users"]), "pool_732": pool_users(732), "pool_1500": pool_users(1500)}
    users = sorted(set().union(*pools.values()))
    rows = {}
    for i, u in enumerate(users):
        rows[u] = user_stats(data, u)
        if i % 500 == 0:
            print(f"[sm173] {i}/{len(users)}", flush=True)
    doc = {"note": "Deleted-card note ids (SM173) phase 1a: aggregate counts of the placeholder note-id collapse "
                   "(sm173_measure.py). No card ids.",
           "dataset": str(data), "pools": {k: summary([rows[u] for u in v]) for k, v in pools.items()}}
    sets = {k: set(v) for k, v in pools.items()}
    Path(args.out).write_text(json.dumps(doc, indent=1))
    with open(Path(args.out).with_name("sm173-measure-users.jsonl"), "w") as f:
        for u in users:
            r = rows[u]
            f.write(json.dumps({**r, "pools": [k for k, v in sets.items() if u in v]}) + "\n")
    print(json.dumps(doc["pools"], indent=1))


if __name__ == "__main__":
    main()
