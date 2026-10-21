"""SM21 v1 stale-state share (CPU, data only): with 16 consecutive 2,048-review windows per update all reading the carry
table as of the update start, which reviews start their entity's state from a stale point (an earlier window of the same
update touched the entity, so its carried state misses those reviews)? Per stream: card, note, deck, user (the user
stream is touched by every review). Classes per review: new (entity first seen; zero is exact), in-window (previous
review of the entity in the same window), exact-carry (previous review in an earlier update), stale (previous review in
an earlier window of the same update). Training users: every 17th of the SM11 replay export (5,000 gate training users)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011")
EXPORT = Path(r"C:\Users\jonat\workspace\anki-rwkv\artifacts\cache\kd-replay\fork-fp32-gate5k\users")
W, U = 2048, 32768
users = sorted(int(p.stem) for p in EXPORT.glob("*.npz"))[::17]
tot = {s: np.zeros(4) for s in ("card", "note", "deck", "user")}
same_day_stale = np.zeros(2)
for u in users:
    rv = pd.read_parquet(DATA / "revlogs" / f"user_id={u}", columns=["card_id", "day_offset"])
    cards = pd.read_parquet(DATA / "cards" / f"user_id={u}", columns=["card_id", "note_id", "deck_id"]).set_index("card_id")
    n = len(rv); i = np.arange(n); win = i // W; upd = i // U
    keys = {"card": rv["card_id"].to_numpy(), "note": cards["note_id"].reindex(rv["card_id"]).fillna(-1).to_numpy(),
            "deck": cards["deck_id"].reindex(rv["card_id"]).fillna(-1).to_numpy(), "user": np.zeros(n)}
    for s, k in keys.items():
        prev = pd.Series(i).groupby(k).shift(1).to_numpy()
        has = ~np.isnan(prev); p = np.where(has, prev, 0).astype(int)
        inwin = has & (win[p] == win); exact = has & (upd[p] < upd); stale = has & ~inwin & ~exact
        tot[s] += [(~has).sum(), inwin.sum(), exact.sum(), stale.sum()]
        if s == "card":
            same_day = has & (rv["day_offset"].to_numpy()[p] == rv["day_offset"].to_numpy())
            same_day_stale += [(stale & same_day).sum(), same_day.sum()]
print(f"{len(users)} training users, {int(tot['card'].sum()):,} reviews")
print("stream | new | in-window | exact carry | STALE (earlier window, same update)")
for s, a in tot.items():
    f = 100 * a / a.sum(); print(f"{s:5s} | {f[0]:5.1f}% | {f[1]:5.1f}% | {f[2]:5.1f}% | {f[3]:5.1f}%")
print(f"card: same-day reviews whose state is stale: {100 * same_day_stale[0] / max(same_day_stale[1], 1):.1f}% of same-day reviews")
# Dense streams (user, and in practice deck/preset): every window k > 0 of an update enters with the state as of the
# update start, so it skips the k x 2,048 reviews before it. Share of reviews that sit in such a window:
inside = skipped = total = 0
for u in users:
    n = len(pd.read_parquet(DATA / "revlogs" / f"user_id={u}", columns=["day_offset"]))
    k = (np.arange(n) % U) // W
    inside += (k > 0).sum(); skipped += (k * W).sum(); total += n
print(f"user stream: {100 * inside / total:.1f}% of reviews are in a window whose entering user state skips earlier windows "
      f"(mean skipped history {skipped / max(inside, 1):,.0f} reviews)")
