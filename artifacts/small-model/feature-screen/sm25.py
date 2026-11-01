"""SM25: review-type robustness. Score with the history review types rewritten as a non-SM-2 scheduler would label them.

Wraps scripts/small_model_heldout.py unchanged; patches rwkv.data_processing.base_transform_state (the only place
`scaled_state` = state - 2 is built; state = the card's state before the review: 0 new, 1 learning, 2 review,
3 relearning). Mode from env SM25_MODE (default nosteps):
  nosteps   FSRS without learning / relearning steps: a card's first review stays new (0), every later one is review (2)
  daysteps  steps shorter than a day: a learning / relearning review after a gap of >= 1 day becomes review (2)
The student never reads the input (zero_features=(22,)), so only the fork changes; query rows carry 0 for both.
"""
import os
import sys

import numpy as np

WT = r"C:\Users\jonat\workspace\wt\small-model"
sys.path.insert(0, WT)
MODE = os.environ.get("SM25_MODE", "nosteps")
assert MODE in ("nosteps", "daysteps")

from rwkv import data_processing as dp  # noqa: E402

_orig = dp.base_transform_state
TOTAL = [0, 0]


def patched(df):
    s = df["state"].to_numpy().copy()
    first = ~df["card_id"].duplicated().to_numpy()
    if MODE == "nosteps":
        new = np.where(first, 0, 2)
    else:
        gap = df.groupby("card_id")["day_offset"].diff().to_numpy()
        new = np.where((gap >= 1) & np.isin(s, (1, 3)), 2, s)
    TOTAL[0] += int((new != s).sum()); TOTAL[1] += len(s)
    print(f"[sm25 {MODE}] relabelled {(new != s).mean():.1%} of {len(s)} rows (run total {TOTAL[0] / TOTAL[1]:.1%})",
          flush=True)
    return _orig(df.assign(state=new))


dp.base_transform_state = patched  # module level: spawned data-loader workers re-import this file and get it too

if __name__ == "__main__":
    from scripts import small_model_heldout as sm
    sm.main()
