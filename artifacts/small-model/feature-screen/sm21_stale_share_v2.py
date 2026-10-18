"""SM21 v2 (strided lanes) stale-state share (CPU, data only), the v2 layout of sm21_stale_share.py (same 295 users:
every 17th of the SM11 replay export, whole revlogs, 2,048-review windows).

v2 layout (scripts/state_carry_train.strided_layout): the user's N windows are cut into 16 lanes of consecutive windows
(the first 16 - m lanes hold U - 1 windows, the last m hold U, U = ceil(N / 16)); update u holds window u of every lane
and each lane has its own carried state, so a window reads its lane predecessor's end state fresh. Classes per review:
  new          entity first seen (zero is exact);
  in-window    previous review of the entity in the same window;
  exact carry  previous review in an earlier window of the same lane (the lane table holds its end state);
  STALE        previous review in another window of the same lane that the table does not have yet (0 by construction);
  lane cut     previous review in an earlier lane: the lane table has no state for the entity, so it starts from zero
               (v2 zero lane starts; the stale-lane-start alternative would give these an epoch-old state). Split into
               reviews inside a lane-start window and reviews in a later window of the lane (entity first seen in the
               lane after its start).
Also: the user stream (dense) per window, and the per-(lane, entity) table size relative to v1's per-entity table."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.state_carry_train import strided_layout  # noqa: E402

DATA = Path(r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011")
EXPORT = Path(r"C:\Users\jonat\workspace\anki-rwkv\artifacts\cache\kd-replay\fork-fp32-gate5k\users")
W, LANES = 2048, 16
users = sorted(int(p.stem) for p in EXPORT.glob("*.npz"))[::17]
names = ("new", "in-window", "exact carry", "STALE", "lane cut (lane-start window)", "lane cut (later window)")
tot = {s: np.zeros(len(names)) for s in ("card", "note", "deck", "user")}
ratio = {"card": [], "note": []}
cut_users = start_rows = total_rows = 0
user_zero_windows = user_windows = 0
for u in users:
    rv = pd.read_parquet(DATA / "revlogs" / f"user_id={u}", columns=["card_id", "day_offset"])
    cards = pd.read_parquet(DATA / "cards" / f"user_id={u}", columns=["card_id", "note_id", "deck_id"]).set_index("card_id")
    n = len(rv)
    i = np.arange(n)
    win = i // W
    layout = strided_layout(int(win[-1]) + 1, LANES)
    lane_of_window = np.empty(int(win[-1]) + 1, dtype=np.int64)
    update_of_window = np.empty_like(lane_of_window)
    for update, pairs in enumerate(layout):
        for lane, window in pairs:
            lane_of_window[window], update_of_window[window] = lane, update
    lane, upd = lane_of_window[win], update_of_window[win]
    lane_start = upd == 0  # every lane starts at update 0
    start_rows += int(lane_start.sum())
    total_rows += n
    cut_users += int(len(layout) > 1)
    for window in range(len(lane_of_window)):
        user_windows += 1
        user_zero_windows += int(update_of_window[window] == 0 and window > 0)
    keys = {"card": rv["card_id"].to_numpy(), "note": cards["note_id"].reindex(rv["card_id"]).fillna(-1).to_numpy(),
            "deck": cards["deck_id"].reindex(rv["card_id"]).fillna(-1).to_numpy(), "user": np.zeros(n)}
    for s, k in keys.items():
        prev = pd.Series(i).groupby(k).shift(1).to_numpy()
        has = ~np.isnan(prev)
        p = np.where(has, prev, 0).astype(int)
        inwin = has & (win[p] == win)
        same_lane = has & ~inwin & (lane[p] == lane)
        exact = same_lane & (upd[p] < upd)
        stale = same_lane & ~exact
        cut = has & (lane[p] != lane)
        tot[s] += [(~has).sum(), inwin.sum(), exact.sum(), stale.sum(), (cut & lane_start).sum(), (cut & ~lane_start).sum()]
        if s in ratio:
            codes = np.unique(k, return_inverse=True)[1].reshape(-1)
            ratio[s].append((len(np.unique(lane * (codes.max() + 1) + codes)), len(np.unique(codes))))
print(f"{len(users)} training users, {int(tot['card'].sum()):,} reviews; {cut_users} users with more than one update "
      f"(the others: every window is a lane start)")
print("stream | " + " | ".join(names))
for s, a in tot.items():
    f = 100 * a / a.sum()
    print(f"{s:5s} | " + " | ".join(f"{x:5.1f}%" for x in f))
print(f"reviews in lane-start windows (update 0 of each user): {100 * start_rows / total_rows:.1f}%")
print(f"user stream: {100 * user_zero_windows / user_windows:.1f}% of windows enter with a zero user state although "
      f"earlier reviews exist (lane starts of lanes 1-15); every other window continues its lane exactly (v1: 92.2% of "
      f"reviews entered with a state that skipped 15,112 reviews on average)")
for s, rows in ratio.items():
    r = np.array(rows, dtype=float)
    per = r[:, 0] / r[:, 1]
    print(f"{s} table: (lane, entity) slots / entities = {r[:, 0].sum() / r[:, 1].sum():.2f} overall, "
          f"median user {np.median(per):.2f}, max {per.max():.2f}; largest user {int(r[:, 1].max()):,} entities -> "
          f"{int(r[r[:, 1].argmax(), 0]):,} slots")
