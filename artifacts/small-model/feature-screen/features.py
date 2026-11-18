"""Candidate input features per review, for the residual screen (scratch; review times stay local).

For every review row r, features use only information available before the review is answered.
Original dataset = what the models read; the -id dataset adds review_time and raw Anki ids (creation ms).
Output: <out>/<user>.npz with review_th (1-based row index) and one array per feature.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ORIG = Path(r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011")
IDD = Path(r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k")
W = [0.212, 1.2931, 2.3065, 8.2956, 6.4133, 0.8334, 3.0194, 0.001, 1.8722, 0.1666, 0.796, 1.4835, 0.0614, 0.2629,
     1.6483, 0.6014, 1.8729, 0.5425, 0.0912, 0.0658, 0.1542]  # FSRS-6 defaults
DECAY = W[20]
FACTOR = 0.9 ** (-1 / DECAY) - 1


def table(root, name, user):
    p = root / name / f"user_id={user}"
    return pd.read_parquet(p) if p.is_dir() else None


def fsrs_states(card, rating, edays):
    """Default-parameter FSRS-6 (S, D, R) before each review; NaN on first reviews."""
    n = len(card)
    S = np.full(n, np.nan); D = np.full(n, np.nan); R = np.full(n, np.nan)
    st = {}
    for i in range(n):
        c, g, t = card[i], rating[i], edays[i]
        if c not in st or t < 0:
            s0 = W[g - 1]
            d0 = min(max(W[4] - np.exp(W[5] * (g - 1)) + 1, 1), 10)
            st[c] = (s0, d0)
            continue
        s, d = st[c]
        r = (1 + FACTOR * t / s) ** (-DECAY)
        S[i], D[i], R[i] = s, d, r
        if t == 0:
            ns = s * np.exp(W[17] * (g - 3 + W[18])) * s ** (-W[19])
            if g >= 3:
                ns = max(ns, s)
        elif g == 1:
            ns = W[11] * d ** (-W[12]) * ((s + 1) ** W[13] - 1) * np.exp(W[14] * (1 - r))
            ns = min(ns, s / np.exp(W[17] * W[18]))
        else:
            ns = s * (np.exp(W[8]) * (11 - d) * s ** (-W[9]) * (np.exp(W[10] * (1 - r)) - 1)
                      * (W[15] if g == 2 else 1) * (W[16] if g == 4 else 1) + 1)
        d0_easy = W[4] - np.exp(W[5] * 3) + 1
        nd = d + (-W[6] * (g - 3)) * (10 - d) / 9
        nd = min(max(W[7] * d0_easy + (1 - W[7]) * nd, 1), 10)
        st[c] = (max(ns, 0.001), nd)
    return S, D, R


def ewma_prev(x, half_life):
    """EWMA of x over previous rows only (value before row i)."""
    a = 1 - 0.5 ** (1 / half_life)
    out = np.empty(len(x)); m = np.nan
    for i, v in enumerate(x):
        out[i] = m
        m = v if np.isnan(m) else (1 - a) * m + a * v
    return out


def user_features(user):
    rv = pd.read_parquet(ORIG / "revlogs" / f"user_id={user}")
    rid = pd.read_parquet(IDD / "revlogs" / f"user_id={user}")
    n = len(rv)
    # same rows by pandas index (source row id), but ties within a day can be ordered differently, and the -id build
    # re-derives intervals from show times and keeps exact durations: align by index, check the card mapping is 1:1
    assert len(rid) == n and rid.index.is_unique and rv.index.is_unique
    rid = rid.loc[rv.index]
    # a few rows can land one day apart at the rollover edge (user 1901: 3 of 34,149)
    assert (rid["day_offset"].to_numpy() == rv["day_offset"].to_numpy()).mean() > 0.999, "datasets not row-aligned"
    assert (rid["rating"].to_numpy() == rv["rating"].to_numpy()).mean() > 0.999, "datasets not row-aligned"
    pairs = pd.DataFrame({"a": rv["card_id"].to_numpy(), "b": rid["card_id"].to_numpy()}).drop_duplicates()
    assert pairs["a"].is_unique and pairs["b"].is_unique, "card ids do not map 1:1"
    card = rv["card_id"].to_numpy(); rating = rv["rating"].to_numpy(); ed = rv["elapsed_days"].to_numpy()
    es = rv["elapsed_seconds"].to_numpy().astype(float); dur = rv["duration"].to_numpy().astype(float)
    day = rv["day_offset"].to_numpy(); state = rv["state"].to_numpy()
    y = (rating > 1).astype(float)
    df = pd.DataFrame({"card": card, "y": y, "rating": rating, "es": es})
    g = df.groupby("card", sort=False)
    f = {}
    f["card_prior_reviews"] = np.log1p(g.cumcount().to_numpy())
    fails = (rating == 1).astype(float)
    f["card_prior_lapses"] = np.log1p(g["y"].transform(lambda s: (1 - s).cumsum().shift(1).fillna(0)).to_numpy())
    # passes since the last Again on this card
    streak = np.zeros(n); last = {}
    for i in range(n):
        c = card[i]; streak[i] = last.get(c, 0)
        last[c] = 0 if rating[i] == 1 else last.get(c, 0) + 1
    f["card_streak"] = np.log1p(streak)
    prev_es = g["es"].shift(1).to_numpy()
    f["interval_ratio"] = np.where((ed >= 0) & (prev_es > 0), np.log((es + 60) / (prev_es + 60)), np.nan)
    S, D, R = fsrs_states(card, rating, ed)
    f["fsrs_R"] = R; f["fsrs_logS"] = np.log(S); f["fsrs_D"] = D
    # review type (SM-2 scheduler state): the query review's own pre-review state, and the card's previous one
    f["review_type"] = state.astype(float)
    f["prev_review_type"] = pd.Series(state).groupby(card).shift(1).to_numpy().astype(float)
    f["user_pass_ewma200"] = ewma_prev(y, 200)
    f["user_pass_ewma5000"] = ewma_prev(y, 5000)
    f["user_logdur_ewma200"] = ewma_prev(np.log(dur + 100), 200)
    f["card_reviews_today"] = np.log1p(pd.DataFrame({"c": card, "d": day}).groupby(["c", "d"]).cumcount().to_numpy())
    # note siblings (original ids): days since the last review of another card of the same note
    cards = table(ORIG, "cards", user)
    if cards is not None:
        note = pd.Series(card).map(cards.set_index("card_id")["note_id"]).to_numpy()
        nsz = cards.groupby("note_id").size()
        f["note_cards"] = np.log(pd.Series(note).map(nsz).to_numpy().astype(float))
        sib = np.full(n, np.nan); lastnote = {}
        for i in range(n):
            k = note[i]
            if k == k and k in lastnote and lastnote[k][0] != card[i]:
                sib[i] = day[i] - lastnote[k][1]
            if k == k:
                lastnote[k] = (card[i], day[i])
        f["sibling_gap_days"] = np.log1p(sib)
    # -id dataset: review clock, card creation, deck depth
    t = rid["review_time"].to_numpy().astype(np.int64) / 1000.0
    gap = np.r_[np.nan, np.diff(t)]
    f["gap_any_review_s"] = np.log1p(np.clip(gap, 0, None))
    sess = np.zeros(n); k = 0
    for i in range(n):
        k = 0 if (i == 0 or gap[i] > 1800) else k + 1
        sess[i] = k
    f["session_position"] = np.log1p(sess)
    # local clock relative to the day rollover: choose the shift that best reproduces day_offset changes
    best, bh = -1, 0
    dd = np.diff(day) > 0
    for h in np.arange(0, 24, 0.5):
        k2 = np.floor((t - h * 3600) / 86400)
        m = (np.diff(k2) > 0) == dd
        if m.mean() > best:
            best, bh = m.mean(), h
    f["hours_since_rollover"] = ((t - bh * 3600) % 86400) / 3600
    f["hours_since_rollover_sin"] = np.sin(f["hours_since_rollover"] * 2 * np.pi / 24)
    cid = rid["card_id"].to_numpy().astype(float) / 1000.0
    first_t = pd.Series(t).groupby(rid["card_id"].to_numpy()).transform("first").to_numpy()
    age_new = (first_t - cid) / 86400
    f["days_new_before_first_review"] = np.where((age_new >= 0) & (age_new < 365 * 30), np.log1p(age_new), np.nan)
    icards, idecks = table(IDD, "cards", user), table(IDD, "decks", user)
    if icards is not None and idecks is not None:
        par = idecks.set_index("deck_id")["parent_id"].to_dict()
        def depth(d):
            k = 0
            while d in par and par[d] not in (0, None) and par[d] == par[d] and k < 20:
                d = par[d]; k += 1
            return k
        dk = pd.Series(rid["card_id"].to_numpy()).map(icards.set_index("card_id")["deck_id"])
        f["deck_depth"] = dk.map(lambda d: depth(d) if d == d else np.nan).to_numpy().astype(float)
    return {"review_th": np.arange(1, n + 1), "rollover_match": np.array(best), **f}


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    for u in map(int, sys.argv[2:]):
        if (out / f"{u}.npz").exists():
            continue
        np.savez_compressed(out / f"{u}.npz", **user_features(u))
        print(u, flush=True)
