#!/usr/bin/env python
"""Tiered card state by answer count (SM141): answer-count distribution of cards and notes, tier shares and bytes
(CPU, a few minutes; reads only the revlog columns card_id / day_offset / rating / state / elapsed_days and the cards
table card_id / note_id; no model output). Aggregates only, no raw rows are written.

Users: the 723 fit users of the 732 set (wt/sm72s eval-users-732-fit.json), the standard readout pool.

Definitions (match the scorer and the fork):
  answers of a card  = its revlog rows (every revlog row is one answer = one card-state update; the scorer's frame order,
                       scripts/small_model_heldout.card_prior_answers). End-of-history count = rows of the card.
  answers of a note  = the revlog rows of all its cards (the note state is updated on every answer of any of its cards).
                       A revlog card missing from the cards table (a deleted card) is its own note (as the scorer's
                       deleted-card neighbour rule cannot be reproduced here; counted and reported).
  never-answered card = in the cards table, no revlog row. The fork keeps no state for it (engine.rs state_ref: the card
                       map has no entry, a fresh zero state is used), so it costs 0 bytes in every scheme.
  stored tier of an entity with n answers: young (<= M) / int4 (M < n < N) / int8 (>= N).
  review-weighted   = per predicted review (a revlog row with >= 1 earlier answer of its card), the earlier answers p of
                       its card (and of its note): the state it reads was written after p updates, so its tier is the
                       tier of p. This is where the LogLoss cost of a tier shows.

  python sm141_dist.py [--out sm141-dist.json]
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
U732FIT = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json")
PCTS = [10, 25, 50, 75, 90, 95, 99]
CAP = 20000   # histogram cap (answers); larger counts are clipped into the last bin (only for the histograms)
# bytes per entity by tier, compact layout (sm141_bytes.py, checked against SM136's encoder there)
BYTES = {"Big": {"card": {8: 13891, 4: 7747, 3: 6211, 2: 4675}, "note": {8: 9267, 4: 5171, 3: 4147, 2: 3123}},
         "Small": {"card": {8: 2545, 4: 1521, 3: 1265, 2: 1009}, "note": {8: 1315, 4: 803, 3: 675, 2: 547}},
         "Tiny": {"card": {8: 1315, 4: 803, 3: 675, 2: 547}, "note": {8: 1315, 4: 803, 3: 675, 2: 547}}}
YOUNG = [0, 2, 4, 8, 16]          # M (0 = no young tier)
INT8_FROM = [64, 128, 256, None]  # N (None = never int8: all int4 above M)
AGE_BUCKETS = [1, 2, 3, 5, 9, 17, 33, 128]   # bucket lower edges of the end-of-history answer count
AGE_NAMES = ["1", "2", "3-4", "5-8", "9-16", "17-32", "33-127", "128+"]


def tier_counts(hist, M, N):
    """(young, int4, int8) entity counts of a histogram over answer counts (index = n)."""
    n = np.arange(len(hist))
    young = hist[(n >= 1) & (n <= M)].sum()
    int8 = hist[n >= N].sum() if N is not None else 0
    return int(young), int(hist[1:].sum() - young - int8), int(int8)


def main():
    import pandas as pd
    from scripts import compress_common as cc
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "sm141-dist.json"))
    args = ap.parse_args()
    data = cc.data_path(cc.run_info())
    users = json.loads(U732FIT.read_text())["users"]
    card_hist = np.zeros(CAP + 1, np.int64)
    card_hist_live = np.zeros(CAP + 1, np.int64)   # cards still in the cards table (not deleted)
    per_user_live = []
    note_hist = np.zeros(CAP + 1, np.int64)
    rev_card = np.zeros(CAP + 1, np.int64)   # predicted reviews by earlier answers of the card
    rev_note = np.zeros(CAP + 1, np.int64)   # ... of the note (reviews whose note has >= 1 earlier answer)
    per_user = {"card": [], "note": []}
    never, answered, deleted, multi_card_notes, notes_total, no_cards_table = 0, 0, 0, 0, 0, 0
    age = {k: {"cards": 0, "lapsed": 0, "age_days": [], "recent30": 0, "last_ivl": [], "since_last": [], "easy_only": 0}
           for k in AGE_NAMES}
    for u in users:
        r = pd.read_parquet(data / "revlogs" / f"user_id={u}",
                            columns=["card_id", "day_offset", "rating", "state", "elapsed_days"])
        cp = data / "cards" / f"user_id={u}"
        if cp.exists():
            c = pd.read_parquet(cp, columns=["card_id", "note_id"])
        else:   # no cards table for this user: every card is its own note, never-answered cards unknown (counted)
            no_cards_table += 1
            c = pd.DataFrame({"card_id": pd.Series(dtype="int64"), "note_id": pd.Series(dtype="int64")})
        cid = r["card_id"]
        prior = cid.groupby(cid).cumcount().to_numpy()
        n_card = cid.value_counts()
        answered += len(n_card)
        never += int((~c["card_id"].isin(n_card.index)).sum())
        note_of = c.set_index("card_id")["note_id"]
        missing = ~n_card.index.isin(note_of.index)
        deleted += int(missing.sum())
        # deleted cards: their own note (negative ids cannot collide with real note ids)
        note_map = pd.Series(note_of.reindex(n_card.index).to_numpy(), index=n_card.index)
        note_map[missing] = -1 - np.arange(int(missing.sum()))
        n_note = n_card.groupby(note_map.to_numpy()).sum()
        notes_total += len(n_note)
        multi_card_notes += int((note_map.value_counts() > 1).sum())
        card_hist += np.bincount(np.minimum(n_card.to_numpy(), CAP), minlength=CAP + 1)
        live = n_card[~missing].to_numpy()
        if len(live):
            card_hist_live += np.bincount(np.minimum(live, CAP), minlength=CAP + 1)
            per_user_live.append(np.percentile(live, PCTS).tolist())
        note_hist += np.bincount(np.minimum(n_note.to_numpy(), CAP), minlength=CAP + 1)
        pp = prior[prior >= 1]
        rev_card += np.bincount(np.minimum(pp, CAP), minlength=CAP + 1)
        nid = note_map.reindex(cid.to_numpy()).to_numpy()
        nprior = pd.Series(nid).groupby(nid).cumcount().to_numpy()
        rev_note += np.bincount(np.minimum(nprior[prior >= 1], CAP), minlength=CAP + 1)
        per_user["card"].append(np.percentile(n_card.to_numpy(), PCTS).tolist())
        per_user["note"].append(np.percentile(n_note.to_numpy(), PCTS).tolist())
        # easy vs recently added, by end-of-history answer count
        end_day = int(r["day_offset"].max())
        g = r.groupby("card_id", sort=False)
        first = g["day_offset"].min()
        last = g["day_offset"].max()
        last_ivl = g["elapsed_days"].last()
        lapsed = r[(r["rating"] == 1) & (r["state"] == 2)].groupby("card_id").size().reindex(n_card.index).fillna(0) > 0
        easy_only = r.assign(e=r["rating"] >= 3).groupby("card_id")["e"].all()
        b = np.searchsorted(np.asarray(AGE_BUCKETS), n_card.to_numpy(), side="right") - 1
        for k, name in enumerate(AGE_NAMES):
            ids = n_card.index[b == k]
            if not len(ids):
                continue
            a = age[name]
            a["cards"] += len(ids)
            a["lapsed"] += int(lapsed.loc[ids].sum())
            a["easy_only"] += int(easy_only.loc[ids].sum())
            ad = end_day - first.loc[ids].to_numpy()
            a["age_days"].append(ad)
            a["recent30"] += int((ad <= 30).sum())
            a["last_ivl"].append(last_ivl.loc[ids].to_numpy())
            a["since_last"].append(end_day - last.loc[ids].to_numpy())

    def pct_hist(h):
        cum = np.cumsum(h[1:])
        tot = cum[-1]
        return {f"P{p}": int(np.searchsorted(cum, p / 100 * tot) + 1) for p in PCTS}

    def share(h, lo, hi=None):
        n = np.arange(len(h))
        m = (n >= lo) & ((n <= hi) if hi is not None else True)
        return float(h[m].sum() / h[1:].sum())

    out = {"note": "aggregates only; 723 fit users of the 732 set; counts at the end of each user's history",
           "users": len(users), "answered_cards": int(answered), "never_answered_cards": int(never),
           "deleted_cards_in_revlog": int(deleted), "users_without_cards_table": no_cards_table, "notes_with_answers": int(notes_total),
           "notes_with_several_answered_cards": int(multi_card_notes),
           "predicted_reviews": int(rev_card.sum()),
           "cards": {"pooled": pct_hist(card_hist),
                     "per_user_median": dict(zip([f"P{p}" for p in PCTS], np.median(np.array(per_user["card"]), 0).tolist())),
                     "mean": float((np.arange(CAP + 1) * card_hist).sum() / card_hist[1:].sum()),
                     "share_le": {str(k): share(card_hist, 1, k) for k in (1, 2, 3, 4, 8, 16, 32, 63, 127)}},
           "cards_not_deleted": {"pooled": pct_hist(card_hist_live), "cards": int(card_hist_live[1:].sum()),
                                 "per_user_median": dict(zip([f"P{p}" for p in PCTS], np.median(np.array(per_user_live), 0).tolist())),
                                 "share_le": {str(k): share(card_hist_live, 1, k) for k in (1, 2, 3, 4, 8, 16, 32, 63, 127)}},
           "notes": {"pooled": pct_hist(note_hist),
                     "per_user_median": dict(zip([f"P{p}" for p in PCTS], np.median(np.array(per_user["note"]), 0).tolist())),
                     "mean": float((np.arange(CAP + 1) * note_hist).sum() / note_hist[1:].sum()),
                     "share_le": {str(k): share(note_hist, 1, k) for k in (1, 2, 3, 4, 8, 16, 32, 63, 127)}},
           "reviews_by_card_prior": {"pooled": pct_hist(rev_card),
                                     "share_le": {str(k): share(rev_card, 1, k) for k in (1, 2, 4, 8, 16, 63, 127)}},
           "reviews_by_note_prior": {"pooled": pct_hist(rev_note) if rev_note[1:].sum() else None,
                                     "share_le": {str(k): share(rev_note, 1, k) for k in (1, 2, 4, 8, 16, 63, 127)}}}
    # tiers: shares of entities (end of history), of predicted reviews (read tier), bytes vs all-int8 per size
    tiers = []
    for young_bits in (3, 2):
        for M in YOUNG:
            if young_bits == 2 and M == 0:
                continue
            for N in INT8_FROM:
                row = {"young_bits": young_bits if M else None, "M": M, "N": N}
                for kind, h, rv in (("card", card_hist, rev_card), ("note", note_hist, rev_note)):
                    y, i4, i8 = tier_counts(h, M, N)
                    ry, ri4, ri8 = tier_counts(rv, M, N)
                    tot, rtot = y + i4 + i8, ry + ri4 + ri8
                    row[kind] = {"entities": [y / tot, i4 / tot, i8 / tot], "reviews": [ry / rtot, ri4 / rtot, ri8 / rtot]}
                    for size, bb in BYTES.items():
                        b = bb[kind]
                        mean = (y * b[young_bits if M else 4] + i4 * b[4] + i8 * b[8]) / tot
                        row[kind][size] = {"mean_bytes": round(mean, 1), "vs_int8": round(100 * (mean / b[8] - 1), 2)}
                tiers.append(row)
    out["tiers"] = tiers
    out["by_answer_count"] = {}
    for name, a in age.items():
        if not a["cards"]:
            continue
        ad, li, sl = (np.concatenate(a[k]) for k in ("age_days", "last_ivl", "since_last"))
        out["by_answer_count"][name] = {
            "cards_share": a["cards"] / answered, "lapsed_share": a["lapsed"] / a["cards"],
            "good_easy_only_share": a["easy_only"] / a["cards"], "first_answer_within_30d_of_end": a["recent30"] / a["cards"],
            "median_age_days": float(np.median(ad)), "median_last_elapsed_days": float(np.median(li)),
            "median_days_since_last": float(np.median(sl))}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("answered_cards", "never_answered_cards", "cards", "notes")})[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
