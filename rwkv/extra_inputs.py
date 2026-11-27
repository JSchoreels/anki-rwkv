"""Session inputs (SM179): optional extra input columns, gated by the environment variable SM179_INPUTS.

Unset or empty (the default) = no extra column anywhere: CARD_FEATURE_COLUMNS, add_queries, the fast sample builder
and the model width (92) are exactly today's, bit for bit. Set it to one mode for the whole process tree (the training
workers and the scorer inherit it, and read it when they import this module):

  session : 6 session and fatigue columns, computed from the revlog's show time (review_time) and answer time
            (review_time + duration) of this user's reviews, all known when the card is shown:
            sm179_gap       ln(1 + seconds from the previous review's answer of ANY card to this show), 0 on the first
            sm179_n10       ln(1 + reviews answered in the 10 minutes before this show)
            sm179_n60       ln(1 + reviews answered in the 60 minutes before this show)
            sm179_sess_min  ln(1 + minutes since the session's first show); a session starts at the user's first
                            review and at every review whose gap (above) is >= 30 minutes
            sm179_sess_pos  ln(1 + earlier reviews in this session)
            sm179_sess_again Again share of the last <= 10 earlier reviews of this session (0 when there is none)
            each z-scored with the constants below (the 944 training users of the screen, update rows).
  state   : 4 card-state columns (the information of input 22, which the model zeroes): one-hot of the revlog review
            kind of THIS review, known before the answer: learning (state 1), review (2), relearning (3), filtered or
            other (>= 4); all 0 on a card's first review (state 0, already flagged by elapsed = -1).

The columns are inserted after the 24 deterministic features (input indices 24 .. 24 + n - 1), so the card / note /
deck / preset codes and the calendar move to 24 + n .. 91 + n; the model's first Linear gets n more input columns
(sm179_prepare.py inserts zero columns there, so step 0 equals the release model). Every column has the same value on
the query row and the update row of a review (add_queries keeps it): it describes the moment the card is shown.

review_time is not in the training dataset (anki-revlogs-10k-original-20261011, answer-time layout); it is read from
the show-time export of the same revlogs (SM179_TIME_DATA, default <data_path>/../anki-revlogs-10k), aligned by the
parquet row index (shared by both exports), and checked against rating and duration of every row.
"""
import os
from pathlib import Path

import numpy as np

ENV = "SM179_INPUTS"
TIME_ENV = "SM179_TIME_DATA"
SESSION_GAP_SECONDS = 1800
AGAIN_WINDOW = 10

SESSION = ["sm179_gap", "sm179_n10", "sm179_n60", "sm179_sess_min", "sm179_sess_pos", "sm179_sess_again"]
STATE = ["sm179_st_learn", "sm179_st_review", "sm179_st_relearn", "sm179_st_other"]
MODES = {"session": SESSION, "state": STATE}

# (mean, std) of the raw transformed value over the update rows of the 944 training users of the screen
# (sm179_prepare.py stats; design.md "Inputs"). The state columns are plain 0/1.
SCALE = {
    "sm179_gap": (0.5174, 1.6065),
    "sm179_n10": (3.3827, 1.2234),
    "sm179_n60": (4.1561, 1.3916),
    "sm179_sess_min": (2.9301, 1.3054),
    "sm179_sess_pos": (4.2047, 1.5089),
    "sm179_sess_again": (0.1575, 0.2165),
}


def mode():
    value = os.environ.get(ENV, "") or ""
    if value and value not in MODES:
        raise ValueError(f"{ENV}={value!r}: expected one of {sorted(MODES)} or unset")
    return value


MODE = mode()
COLUMNS = list(MODES.get(MODE, []))
DIMS = len(COLUMNS)


def _time_frame(data_path, user_id):
    import pandas as pd
    root = Path(os.environ.get(TIME_ENV) or (Path(data_path).parent / "anki-revlogs-10k"))
    a = pd.read_parquet(Path(data_path) / "revlogs" / f"user_id={user_id}", columns=["rating", "duration"])
    b = pd.read_parquet(root / "revlogs" / f"user_id={user_id}", columns=["rating", "duration", "review_time"])
    if not (a.index.is_unique and b.index.is_unique and len(a) == len(b)):
        raise ValueError(f"SM179: user {user_id}: the two exports do not share a unique row index")
    b = b.reindex(a.index)
    if b["review_time"].isna().any() or not ((a["rating"].to_numpy() == b["rating"].to_numpy()).all()
                                             and (a["duration"].to_numpy() == b["duration"].to_numpy()).all()):
        raise ValueError(f"SM179: user {user_id}: rows of the two exports differ (rating / duration)")
    return b["review_time"].to_numpy().astype(np.int64), a["duration"].to_numpy().astype(np.int64)


def session_raw(show_ms, duration_ms):
    """Raw (untransformed) session values per review, in revlog order: gap s, n10, n60, minutes, position, again share
    needs ratings -> see session_values. Pure numpy; review k only reads reviews j < k."""
    show = np.asarray(show_ms, dtype=np.int64)
    answer = show + np.asarray(duration_ms, dtype=np.int64)
    n = len(show)
    k = np.arange(n)
    gap = np.zeros(n, dtype=np.float64)
    if n > 1:
        gap[1:] = np.maximum(show[1:] - answer[:-1], 0) / 1000.0
    start = np.ones(n, dtype=bool)
    start[1:] = gap[1:] >= SESSION_GAP_SECONDS
    first = np.maximum.accumulate(np.where(start, k, 0))        # index of the session's first review
    # earlier answers in the window [show - W, show]: answers are in revlog (answer) order; a running maximum keeps the
    # search exact if two answers are out of order (count of j < k with max(answer[0..j]) >= show - W).
    run = np.maximum.accumulate(answer)

    def within(window_ms):
        lo = np.searchsorted(run, show - window_ms, side="left")
        return np.maximum(k - lo, 0).astype(np.float64)

    minutes = np.maximum(show - show[first], 0) / 60000.0
    position = (k - first).astype(np.float64)
    return gap, within(600_000), within(3_600_000), minutes, position, first


def session_values(show_ms, duration_ms, rating):
    gap, n10, n60, minutes, position, first = session_raw(show_ms, duration_ms)
    again = (np.asarray(rating) == 1).astype(np.int64)
    c = np.concatenate([[0], np.cumsum(again)])
    k = np.arange(len(gap))
    m = np.minimum(k - first, AGAIN_WINDOW)
    share = np.where(m > 0, (c[k] - c[k - m]) / np.maximum(m, 1), 0.0)
    return {"sm179_gap": np.log1p(gap), "sm179_n10": np.log1p(n10), "sm179_n60": np.log1p(n60),
            "sm179_sess_min": np.log1p(minutes), "sm179_sess_pos": np.log1p(position), "sm179_sess_again": share}


def state_values(state):
    s = np.asarray(state)
    return {"sm179_st_learn": (s == 1).astype(np.int64), "sm179_st_review": (s == 2).astype(np.int64),
            "sm179_st_relearn": (s == 3).astype(np.int64), "sm179_st_other": (s >= 4).astype(np.int64)}


def add_columns(df, data_path, user_id):
    """The mode's columns on get_rwkv_data's frame (one row per revlog row, review_th = row + 1)."""
    if not COLUMNS:
        return df
    th = df["review_th"].to_numpy()
    if not (th == np.arange(1, len(df) + 1)).all():
        raise ValueError("SM179: frame rows are not in revlog order")
    if MODE == "session":
        show, duration = _time_frame(data_path, user_id)
        if len(show) != len(df) or not (duration == df["duration"].to_numpy()).all():
            raise ValueError(f"SM179: user {user_id}: frame and time export differ")
        values = session_values(show, duration, df["rating"].to_numpy())
        for name in SESSION:
            mean, std = SCALE[name]
            if mean is None:
                raise RuntimeError("SM179: session scale constants are not set")
            df[name] = (values[name] - mean) / std
    else:
        for name, value in state_values(df["state"].to_numpy()).items():
            df[name] = value
    return df
