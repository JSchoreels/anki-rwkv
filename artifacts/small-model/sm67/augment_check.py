"""Data augmentation (SM67): CPU-only sanity check of candidate augmentations (design note DESIGN.md, section 6).

Light by design (one process, 4 threads, ~25 training users, no GPU). It reads the dataset and the leak-free feature
code of the leak-free small model's worktree (wt/sm72s, imported read-only with no bytecode written) and reports:

  A. review_counts only (all 8,000 final-tier training users): updates and lane-start share for the 2,048 x 16 strided
     lanes (constant-LR epochs) and the 32k x 1 cool-down, plain vs note-split users.
  B. per sampled user, on the full leak-free frame (get_rwkv_data with RWKV_DELETED_CARDS=neighbour):
     - note split into 2 complementary halves: every kept card's own inputs and labels identical to the full frame?
       which card-feature columns change (share of rows, mean |change|)? labels kept?
     - time shift (day_offset + 3): a no-op on every input?
     - prefix (first 60% of reviews): identical rows (causal model -> no new information)?
     - review drop (10% of reviews at random): share of remaining labelled rows whose input interval or label changes.
     - start crop (drop the first 30% of reviews, cut at a day boundary): share of cards whose first visible review is not a new-card review,
       vs the natural rate in full histories.
     - decks per user (feasibility of a deck split).
Output: augment-check.json next to this file.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["RWKV_DELETED_CARDS"] = "neighbour"
sys.dont_write_bytecode = True
WT = Path("C:/Users/jonat/workspace/wt/sm72s")  # read only
sys.path.insert(0, str(WT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import rwkv.data_processing as dp  # noqa: E402

DATA = Path("C:/Users/jonat/workspace/anki-rwkv/data/anki-revlogs-10k-original-20261011")
RUN8K = Path("C:/Users/jonat/workspace/wt/sm55/artifacts/small-model/sm55-final8k-run.json")
SPLIT = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/full-run/run-8k-20261013/run.json")
E732 = WT / "artifacts/small-model/sm72s/eval-users-732-fit.json"
E464 = WT / "artifacts/small-model/sm72s/eval-users-464-fit.json"
OUT = Path(__file__).with_name("augment-check.json")
SAMPLE = int(os.environ.get("SM67_SAMPLE", "25"))
MAX_REVIEWS = 250_000
COLS = dp.CARD_FEATURE_COLUMNS
LABELS = ["has_label", "label_elapsed_seconds", "label_elapsed_days", "label_y", "label_rating"]
IDS = ["card_id", "note_id", "deck_id", "preset_id"]

_read = pd.read_parquet
_filter = {"fn": None}


def _patched(path, *args, **kwargs):
    frame = _read(path, *args, **kwargs)
    if Path(path).parent.name == "revlogs" and _filter["fn"] is not None:
        frame = frame.reset_index(drop=True)
        frame["orig_th"] = np.arange(1, len(frame) + 1)
        frame = _filter["fn"](frame)
    return frame


pd.read_parquet = _patched  # this process only; get_rwkv_data reads through it


def frame(user, fn=None):
    _filter["fn"] = fn or (lambda f: f)
    try:
        return dp.get_rwkv_data(DATA, user)
    finally:
        _filter["fn"] = None


def note_of(user):
    cards = dp.read_user_metadata(DATA / "cards", user)
    return dict(zip(cards["card_id"].to_numpy(), cards["note_id"].to_numpy()))


def part_of(user, raw, seed=0):
    """0/1 per raw review: a hash of (seed, user, note), deleted cards (no note) by card."""
    notes = note_of(user)
    keys = [f"{seed}:{user}:n{notes[c]}" if c in notes and notes[c] == notes[c] else f"{seed}:{user}:c{c}"
            for c in raw["card_id"].to_numpy()]
    cache = {}
    out = np.empty(len(keys), dtype=np.int8)
    for i, k in enumerate(keys):
        if k not in cache:
            cache[k] = hashlib.blake2b(k.encode(), digest_size=2).digest()[0] & 1
        out[i] = cache[k]
    return out


def compare(full, aug):
    """Rows of `aug` matched to `full` by original review number; per column the share of rows that differ."""
    ref = full.set_index("orig_th").loc[aug["orig_th"].to_numpy()]
    out = {}
    for col in COLS + LABELS + IDS:
        a, b = aug[col].to_numpy(), ref[col].to_numpy()
        diff = ~np.isclose(a.astype(float), b.astype(float), rtol=0, atol=1e-9) if col not in IDS else a != b
        out[col] = {"share": float(diff.mean()), "mean_abs": float(np.abs(a.astype(float) - b.astype(float)).mean())
                    if col not in IDS else None}
    return out


def windows_stats(counts, split_users=None, span=2048, lanes=16, big=32768):
    """Updates for 2,048 x 16 strided lanes and the 32k cool-down, and the share of reviews in lane-start windows."""
    up_lanes = up_big = lane_start = total = 0
    for n in counts:
        w = -(-n // span)
        up_lanes += -(-w // lanes)
        up_big += -(-n // big)
        lane_start += min(n, min(w, lanes) * span)
        total += n
    return {"pseudo_users": len(counts), "updates_2048x16": up_lanes, "updates_32k": up_big,
            "lane_start_review_share": lane_start / total}


def main():
    t0 = time.time()
    run = json.loads(RUN8K.read_text())
    split = json.loads(SPLIT.read_text())
    held = set(split["validation_users"]) | set(split["final_test_users"]) | set(split.get("preserved_final_test_users", []))
    held |= set(json.loads(E732.read_text())["users"]) | set(json.loads(E464.read_text())["users"])
    train = [int(u) for u in run["train_users"]]
    assert not (set(train) & held), "a held-out user is in the training list"
    counts = {u: run["review_counts"][str(u)] for u in train}
    result = {"held_out_users_checked": len(held), "training_users": len(train), "overlap_with_held_out": 0}

    # A. review_counts only: halves of size n/2 (note split is near-even on average; measured below per user)
    plain = list(counts.values())
    for name, thr in (("split_all", 0), ("split_if_>=65536", 65536)):
        halves = []
        for n in plain:
            if n >= max(thr, 2):
                halves += [n // 2, n - n // 2]
            else:
                halves.append(n)
        result[f"A_{name}"] = windows_stats(halves)
    result["A_plain"] = windows_stats(plain)
    result["A_reviews_in_users_>=65536"] = sum(n for n in plain if n >= 65536) / sum(plain)
    result["A_users_>=65536"] = sum(n >= 65536 for n in plain)

    # B. sampled users (stratified by size, <= MAX_REVIEWS so the CPU stays light)
    rng = np.random.default_rng(67)
    pool = sorted((u for u in train if 2_000 <= counts[u] <= MAX_REVIEWS), key=lambda u: counts[u])
    picks = [pool[int(i)] for i in np.linspace(0, len(pool) - 1, SAMPLE)]
    per_user = []
    for user in picks:
        rec = {"user": user, "reviews": counts[user]}
        full = frame(user)
        rec["cards"], rec["notes"] = int(full["card_id"].nunique()), int(full["note_id"].nunique())
        rec["decks"] = int(full["deck_id"].nunique())
        share = full.groupby("deck_id").size() / len(full)
        rec["decks_ge_10pct"] = int((share >= 0.10).sum())
        rec["labelled"] = int(full["has_label"].sum())
        firsts = full.groupby("card_id").head(1)
        rec["natural_first_not_new"] = float((firsts["elapsed_days"] != -1).mean())

        # note split
        raw = _read(DATA / "revlogs" / f"user_id={user}").reset_index(drop=True)
        parts = part_of(user, raw)
        halves = []
        for h in (0, 1):
            keep = parts == h
            halves.append(frame(user, lambda f, keep=keep: f[keep]))
        rec["split_sizes"] = [len(x) for x in halves]
        rec["split_labelled"] = [int(x["has_label"].sum()) for x in halves]
        rec["split_labels_kept"] = sum(rec["split_labelled"]) == rec["labelled"]
        diffs = [compare(full, x) for x in halves]
        rec["split_changed"] = {c: {"share": float(np.mean([d[c]["share"] for d in diffs])),
                                    "mean_abs": None if diffs[0][c]["mean_abs"] is None else
                                    float(np.mean([d[c]["mean_abs"] for d in diffs]))} for c in diffs[0]}

        # time shift
        shifted = frame(user, lambda f: f.assign(day_offset=f["day_offset"] + 3))
        rec["shift_changed_cols"] = [c for c, v in compare(full, shifted).items() if v["share"] > 0]

        # prefix (first 60%)
        cut = int(0.6 * len(raw))
        pre = frame(user, lambda f: f.iloc[:cut])
        pc = compare(full, pre)
        rec["prefix_changed_cols"] = {c: v["share"] for c, v in pc.items() if v["share"] > 0}

        # review drop 10% (raw level: what would have to change)
        drop = rng.random(len(raw)) < 0.10
        kept = raw[~drop].reset_index(drop=True)
        prev_full = raw.groupby("card_id").cumcount()  # 0 = first review of the card
        # a kept review whose previous review of the same card was dropped: its interval input is wrong unless recomputed
        prev_dropped = np.zeros(len(raw), dtype=bool)
        last_seen = {}
        for i, (c, d) in enumerate(zip(raw["card_id"].to_numpy(), drop)):
            if c in last_seen and last_seen[c]:
                prev_dropped[i] = True
            last_seen[c] = d
        rec["drop_rows_interval_changes"] = float(prev_dropped[~drop].mean())
        nxt = np.zeros(len(raw), dtype=bool)
        nxt_seen = {}
        cards = raw["card_id"].to_numpy()
        for i in range(len(raw) - 1, -1, -1):
            c = cards[i]
            if c in nxt_seen and nxt_seen[c]:
                nxt[i] = True
            nxt_seen[c] = drop[i]
        rec["drop_rows_label_changes"] = float(nxt[~drop].mean())
        rec["drop_first_review_lost_cards"] = float(raw[drop & (prev_full.to_numpy() == 0)]["card_id"].nunique()
                                                    / raw["card_id"].nunique())
        del kept

        # elapsed_days == day_offset gap of the same card (so a recomputation after a drop would be possible)
        gap = raw.groupby("card_id")["day_offset"].diff()
        ok = raw["elapsed_days"].to_numpy()[gap.notna().to_numpy()] == gap.dropna().to_numpy()
        rec["elapsed_days_equals_day_gap"] = float(ok.mean()) if len(ok) else None

        # start crop: drop the first 30% of the user's reviews
        d0 = raw["day_offset"].iloc[int(0.3 * len(raw))]
        crop = raw[raw["day_offset"] >= d0]
        cf = crop.groupby("card_id").head(1)
        rec["crop_first_not_new"] = float((cf["elapsed_days"] != -1).mean()) if len(cf) else None
        rec["crop_review_share"] = len(crop) / len(raw)
        per_user.append(rec)
        print(f"user {user}: {len(raw):,} reviews, split {rec['split_sizes']}, labels kept {rec['split_labels_kept']}, "
              f"{time.time() - t0:.0f} s", flush=True)

    result["B_users"] = per_user
    w = np.array([r["reviews"] for r in per_user], dtype=float)
    agg = {}
    for col in COLS + LABELS + IDS:
        agg[col] = {"share": float(np.average([r["split_changed"][col]["share"] for r in per_user], weights=w))}
        if per_user[0]["split_changed"][col]["mean_abs"] is not None:
            agg[col]["mean_abs"] = float(np.average([r["split_changed"][col]["mean_abs"] for r in per_user], weights=w))
    result["B_split_changed_weighted"] = agg
    result["B_summary"] = {
        "users": len(per_user), "reviews": int(w.sum()),
        "split_labels_kept_all": all(r["split_labels_kept"] for r in per_user),
        "split_half_share_mean": float(np.mean([min(r["split_sizes"]) / r["reviews"] for r in per_user])),
        "shift_noop_all": all(not r["shift_changed_cols"] for r in per_user),
        "prefix_changed_cols_union": sorted({c for r in per_user for c in r["prefix_changed_cols"]}),
        "drop_rows_interval_changes": float(np.average([r["drop_rows_interval_changes"] for r in per_user], weights=w)),
        "drop_rows_label_changes": float(np.average([r["drop_rows_label_changes"] for r in per_user], weights=w)),
        "elapsed_days_equals_day_gap": float(np.mean([r["elapsed_days_equals_day_gap"] for r in per_user])),
        "natural_first_not_new": float(np.mean([r["natural_first_not_new"] for r in per_user])),
        "crop_first_not_new": float(np.mean([r["crop_first_not_new"] for r in per_user])),
        "users_one_deck_ge_10pct": sum(r["decks_ge_10pct"] <= 1 for r in per_user),
        "decks_median": float(np.median([r["decks"] for r in per_user])),
    }
    result["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "B_users" and k != "B_split_changed_weighted"}, indent=1))
    print(json.dumps({c: v for c, v in agg.items() if v["share"] > 0}, indent=1))


if __name__ == "__main__":
    main()
