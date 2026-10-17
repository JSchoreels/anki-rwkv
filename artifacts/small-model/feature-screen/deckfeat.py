"""Deck / preset / collection size and progress features, added to fx/<user>.npz (user request 2026-10-17).

All in -id space (raw ids): card -> deck -> preset from the -id cards/decks tables; card creation time = raw card id
(ms) when plausible. For each review row (as of that review) and *_atprev (as of the card's previous review, where the
curve is forecast):
  deck_cards_created   cards of the deck created by then (raw card id <= review time)  - "deck size" without the future
  deck_cards_snapshot  cards of the deck in the export snapshot (includes cards created later: a mild future leak)
  {deck,preset,user}_introduced   distinct cards of the deck / preset / collection reviewed at least once by then
  deck_progress        deck_introduced / deck_cards_created
  {deck,preset}_reviews  reviews in the deck / preset so far
Counts are log1p; progress is a ratio in [0, 1].
"""
from pathlib import Path

import numpy as np
import pandas as pd

ORIG = Path(r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011")
IDD = Path(r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k")


def table(root, name, user):
    p = root / name / f"user_id={user}"
    return pd.read_parquet(p) if p.is_dir() else None


def running_distinct(group, item):
    """Number of distinct `item`s seen within `group` up to and including each row."""
    first = ~pd.DataFrame({"g": group, "i": item}).duplicated()
    return pd.Series(first.astype(int).to_numpy()).groupby(group).cumsum().to_numpy()


def add(user):
    path = Path("fx") / f"{user}.npz"
    z = dict(np.load(path))
    if "deck_cards_created" in z:
        return "skip"
    rv = pd.read_parquet(ORIG / "revlogs" / f"user_id={user}", columns=["card_id"])
    rid = pd.read_parquet(IDD / "revlogs" / f"user_id={user}", columns=["card_id", "review_time"]).loc[rv.index]
    ic, idk = table(IDD, "cards", user), table(IDD, "decks", user)
    if ic is None or idk is None:
        return "no cards/decks table"
    n = len(rv)
    card = rid["card_id"].to_numpy(); t = rid["review_time"].to_numpy().astype(np.int64)
    deck_of = ic.set_index("card_id")["deck_id"]
    preset_of = idk.set_index("deck_id")["preset_id"]
    deck = pd.Series(card).map(deck_of).fillna(-1).to_numpy().astype(np.int64)
    preset = pd.Series(deck).map(preset_of).fillna(-1).to_numpy().astype(np.int64)
    f = {}
    # deck size as of the review: cards of the deck whose creation ms <= review time
    ok = (ic["card_id"] > 10 ** 12) & (ic["card_id"] < 4 * 10 ** 12)          # plausible epoch-ms creation stamps
    created = np.full(n, np.nan)
    for d, cs in ic[ok].groupby("deck_id")["card_id"]:
        s = np.sort(cs.to_numpy()); m = deck == d
        created[m] = np.searchsorted(s, t[m], side="right")
    f["deck_cards_created"] = np.log1p(created)
    f["deck_cards_snapshot"] = np.log1p(pd.Series(deck).map(ic.groupby("deck_id").size()).to_numpy().astype(float))
    intro = {k: running_distinct(g, card) for k, g in (("deck", deck), ("preset", preset), ("user", np.zeros(n, np.int64)))}
    for k, v in intro.items():
        f[f"{k}_introduced"] = np.log1p(v)
    f["deck_progress"] = np.where(created > 0, np.minimum(intro["deck"] / np.maximum(created, 1), 1), np.nan)
    for k, g in (("deck", deck), ("preset", preset)):
        f[f"{k}_reviews"] = np.log1p(pd.Series(np.ones(n)).groupby(g).cumsum().to_numpy())
    # as of the card's previous review (curve forecast point); the original card ids give the per-card order
    prev = pd.Series(np.arange(n)).groupby(rv["card_id"].to_numpy()).shift(1).to_numpy()
    has = ~np.isnan(prev); pi = np.where(has, prev, 0).astype(int)
    for k in list(f):
        f[k + "_atprev"] = np.where(has, f[k][pi], np.nan)
    z.update(f)
    np.savez_compressed(path, **z)
    return "ok"


if __name__ == "__main__":
    import json
    users = [u for s in (0, 1) for u in json.load(open(f"shard{s}.json"))["users"]]
    out = {u: add(u) for u in users}
    print({k: sum(v == k for v in out.values()) for k in set(out.values())})
