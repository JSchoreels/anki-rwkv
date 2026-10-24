"""SM70 feature discovery: candidate model inputs per benchmark test review, joined to per-review predictions.

For every held-out user with a per-review dump (scripts/small_model_heldout.py --dump), builds candidate features from the
raw anki-revlogs-10k tables (revlogs, cards, decks) using only information available before the review is answered
(no current rating, no current duration, nothing later in the history), and writes one parquet with
user_id, review_th, y, the SM56 predictions (and the big model's, when its dump exists) and the features.

Two versions of every time-varying feature:
  <name>       as of just before the current review (deployable for the instant head, next review),
  <name>@c     as of the card's previous review (deployable for the curve head, which is fixed when the previous review
               is answered and only takes the elapsed time as argument).
Card-history, static and interval features are identical for both heads (they are known after the previous answer or
are a function of the elapsed time).

usage: sm70_features.py --dump artifacts/small-model/sm70/dump --fork-dump <dir> --out artifacts/small-model/sm70/features.parquet
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path(r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011")
# the "-id" export of the same collections: raw Anki ids (creation timestamps in ms) and review_time (show time, ms)
DATA_ID = Path(r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k")
DAY_MS = 86_400_000

# name -> (group, kind); kind: dyn = time-varying (lagged @c version for the curve head), other = same for both heads
FEATURES = {
    # S: current review type (the revlog state of the queried review: learning / review / relearning / filtered)
    "s_state": ("state", "card"),
    # C: card history (all past reviews of this card)
    "c_n": ("card", "card"), "c_lapses": ("card", "card"), "c_fail_frac": ("card", "card"),
    "c_streak": ("card", "card"), "c_hard_frac": ("card", "card"), "c_easy_frac": ("card", "card"),
    "c_entropy": ("card", "card"), "c_growth": ("card", "interval"), "c_max_ratio": ("card", "interval"),
    "c_age": ("card", "interval"), "c_today": ("card", "interval"), "c_dur_mean": ("card", "card"),
    "c_dur_last": ("card", "card"), "c_first_rating": ("card", "card"),
    # N: note siblings
    "n_size": ("snapshot", "static"), "n_sib_n": ("note", "dyn"), "n_sib_fail": ("note", "dyn"),
    "n_sib_gap": ("note", "dyn"), "n_sib_last_y": ("note", "dyn"),
    # D: deck and preset
    "d_size": ("snapshot", "static"), "d_depth": ("snapshot", "static"), "p_ndecks": ("snapshot", "static"),
    "p_size": ("snapshot", "static"), "d_ret": ("deck", "dyn"), "d_today": ("deck", "dyn"),
    # K: creation order (card ids are ranked by creation time in the dataset)
    "k_rank": ("snapshot", "static"), "k_order_gap": ("creation", "card"), "k_intro_jump": ("creation", "card"),
    "k_nb_ret": ("creation", "dyn"), "k_nb_n": ("creation", "dyn"), "k_backlog": ("snapshot", "dyn"),
    # U: user running stats
    "u_n": ("user", "dyn"), "u_ret": ("user", "dyn"), "u_ret_rev": ("user", "dyn"), "u_ret_100": ("user", "dyn"),
    "u_today_fail": ("user", "dyn"), "u_gap": ("user", "dyn"), "u_yday": ("user", "dyn"), "u_load": ("user", "dyn"),
    "u_hard": ("user", "dyn"), "u_easy": ("user", "dyn"), "u_wd_ret": ("user", "dyn"),
    "u_dur_last": ("user", "dyn"), "u_dur_mean": ("user", "dyn"),
    # T: clock and session (from the -id export's review_time)
    # review_time = answer time - recorded duration, and the recorded duration is capped (60 s by default): when the
    # current review took longer than the cap, its estimated show time is late by the excess, so any gap ending at the
    # current show time carries the current review's (uncapped) thinking time. Those instant-head versions are leaky
    # ("clock-leaky"); the "@c" versions (as of the card's previous review) and the "_prev" ones are past information.
    "t_tod": ("clock", "dyn"), "t_sib_gap_h": ("clock", "dyn"), "t_gap_prev": ("clock", "dyn"),
    "t_break": ("clock", "dyn"), "t_sess_pos_prev": ("clock", "dyn"), "t_sess_min_prev": ("clock", "dyn"),
    "t_sess_fail_prev": ("clock", "dyn"),
    "t_since_any": ("clock-leaky", "dyn"), "t_sess_pos": ("clock-leaky", "dyn"),
    "t_sess_min": ("clock-leaky", "dyn"), "t_sess_fail": ("clock-leaky", "dyn"),
    # R: real creation times (raw Anki ids of the -id export)
    "r_card_age": ("created", "interval"), "r_note_age": ("created", "interval"), "r_deck_age": ("created", "interval"),
    "r_wait_new": ("created", "card"), "r_batch": ("created", "card"),
}
SESSION_GAP_MS = 20 * 60 * 1000
LEAKY_GROUPS = {"snapshot", "clock-leaky"}   # left out of the "all-deployable" bundles


def _past(df, keys, col):
    """Sum of `col` over earlier rows of the same group (strictly before the current row)."""
    return df.groupby(keys, sort=False)[col].cumsum() - df[col]


def _carry_other(df, group, member, value):
    """Per row: `value` at the latest earlier row of the same `group` whose `member` differs (NaN if none)."""
    g = df.groupby(group, sort=False)
    prev_member = g[member].shift(1)
    prev_value = g[value].shift(1)
    cand = prev_value.where(prev_member.notna() & (prev_member != df[member]))
    # a same-member previous row carries that row's own value: forward fill within the group
    return cand.groupby(df[group], sort=False).ffill()


def rollover_offset(rt, day_offset):
    """Quarter-hour offset q (ms = q * 900,000) such that floor((rt - offset) / day) - day_offset is constant: the
    user's day rollover in UTC. Every offset inside the user's nightly no-review window agrees, so the middle of the
    longest agreeing run is taken (local time of day relative to the middle of the quiet hours)."""
    agree = np.zeros(96)
    for q in range(96):
        dd = np.floor((rt - q * 900_000) / DAY_MS).astype(np.int64) - day_offset
        agree[q] = np.unique(dd, return_counts=True)[1].max() / len(dd)
    ok = agree >= agree.max() - 1e-12
    best, run, start, best_start = 0, 0, 0, int(np.argmax(ok))
    for i in range(192):
        if ok[i % 96]:
            if run == 0:
                start = i
            run += 1
            if run > best and run <= 96:
                best, best_start = run, start
        else:
            run = 0
    return ((best_start + best // 2) % 96) * 900_000


def build_clock(user, df):
    """-id export features aligned to the original rows of `df` (original card ids = ranks of the sorted raw ids)."""
    a = pd.read_parquet(DATA_ID / "revlogs" / f"user_id={user}").reset_index(drop=True)
    raw_ids = np.sort(a["card_id"].unique())
    a["cid"] = np.searchsorted(raw_ids, a["card_id"].values)
    a["k"] = a.groupby("cid").cumcount()
    k = df.groupby("card_id").cumcount()
    key_a = pd.MultiIndex.from_arrays([a["cid"], a["k"]])
    pos = key_a.get_indexer(pd.MultiIndex.from_arrays([df["card_id"], k]))
    assert (pos >= 0).all(), "row alignment with the -id export failed"
    m = a.iloc[pos].reset_index(drop=True)
    assert (m["rating"].values == df["rating"].values).mean() > 0.999
    rt = m["review_time"].to_numpy(np.int64)
    out = pd.DataFrame(index=df.index)
    off = rollover_offset(a["review_time"].to_numpy(np.int64), a["day_offset"].to_numpy(np.int64))
    out["t_tod"] = ((rt - off) % DAY_MS) / 3_600_000.0
    # previous answer of any card (original order): its show time + its duration
    prev_end = np.concatenate([[np.nan], (rt[:-1] + m["duration"].to_numpy(np.float64)[:-1])])
    gap = np.clip(rt - prev_end, 0, None)
    out["t_since_any"] = np.log1p(gap / 1000.0)
    new_sess = ~(gap <= SESSION_GAP_MS)            # NaN (first review) starts a session
    sess = np.cumsum(new_sess)
    sdf = pd.DataFrame({"s": sess, "rt": rt, "nf": df["nf"].values, "fail": df["fail"].values})
    out["t_sess_pos"] = np.log1p(sdf.groupby("s").cumcount().values.astype(float))
    out["t_sess_min"] = np.log1p(np.clip(rt - sdf.groupby("s")["rt"].transform("first").values, 0, None) / 60_000.0)
    s_nf = sdf.groupby("s")["nf"].cumsum().values - sdf["nf"].values
    s_fail = sdf.groupby("s")["fail"].cumsum().values - sdf["fail"].values
    out["t_sess_fail"] = (s_fail + 0.2) / (s_nf + 2.0)
    # past-only session state: the previous review's gap and session counters (no current show time involved), and a
    # session-break flag whose threshold (30 min) is far above the duration-cap error of a normal review
    out["t_gap_prev"] = np.concatenate([[np.nan], out["t_since_any"].values[:-1]])
    out["t_break"] = np.where(np.isnan(gap), np.nan, np.where(gap > 30 * 60_000, np.log1p(gap / 3_600_000.0), 0.0))
    for c in ("t_sess_pos", "t_sess_min", "t_sess_fail"):
        out[c + "_prev"] = np.concatenate([[np.nan], out[c].values[:-1]])
    tmp = pd.DataFrame({"note_id": df["note_id"].values, "card_id": df["card_id"].values, "rt": rt.astype(float)})
    sib_rt = _carry_other(tmp, "note_id", "card_id", "rt")
    out["t_sib_gap_h"] = np.log1p(np.clip(rt - sib_rt.values, 0, None) / 3_600_000.0)
    # creation times: raw ids are ms timestamps when plausible (2006 .. the review time)
    cards = pd.read_parquet(DATA_ID / "cards" / f"user_id={user}")
    raw_card = m["card_id"].to_numpy(np.int64)
    lo = 1_136_073_600_000   # 2006-01-01

    def age(ids):
        ids = np.asarray(ids, dtype=np.float64)
        ok = (ids >= lo) & (ids <= rt)
        return np.where(ok, np.log1p(np.clip(rt - ids, 0, None) / DAY_MS), np.nan)
    note_of = dict(zip(cards["card_id"], cards["note_id"]))
    deck_of = dict(zip(cards["card_id"], cards["deck_id"]))
    out["r_card_age"] = age(raw_card)
    out["r_note_age"] = age(pd.Series(raw_card).map(note_of).to_numpy(np.float64))
    out["r_deck_age"] = age(pd.Series(raw_card).map(deck_of).to_numpy(np.float64))
    first_rt = pd.Series(rt).groupby(df["card_id"].values).transform("first").to_numpy(np.float64)
    wait = (first_rt - raw_card) / DAY_MS
    out["r_wait_new"] = np.where((raw_card >= lo) & (wait >= 0), np.log1p(np.clip(wait, 0, None)), np.nan)
    # creation batch: cards (reviewed or in the cards table) created within 60 s of this card
    all_ids = np.unique(np.concatenate([cards["card_id"].to_numpy(np.int64), raw_ids]))
    left = np.searchsorted(all_ids, raw_card - 60_000, side="left")
    right = np.searchsorted(all_ids, raw_card + 60_000, side="right")
    out["r_batch"] = np.where(raw_card >= lo, np.log1p(right - left - 1), np.nan)
    return out


def build_user(user):
    df = pd.read_parquet(DATA / "revlogs" / f"user_id={user}").reset_index(drop=True)
    cards = pd.read_parquet(DATA / "cards" / f"user_id={user}")
    decks = pd.read_parquet(DATA / "decks" / f"user_id={user}")
    n = len(df)
    df["review_th"] = np.arange(1, n + 1)
    df = df.merge(cards, on="card_id", how="left").merge(decks[["deck_id", "parent_id", "preset_id"]], on="deck_id",
                                                         how="left")
    assert len(df) == n
    df["note_id"] = df["note_id"].fillna(-1 - df["card_id"])
    df["deck_id"] = df["deck_id"].fillna(-1)
    df["preset_id"] = df["preset_id"].fillna(-1)
    out = pd.DataFrame({"review_th": df["review_th"]})
    day = df["day_offset"]
    y = (df["rating"] > 1).astype(float)
    first = df["elapsed_days"] == -1
    df["y"] = y
    df["one"] = 1.0
    df["nf"] = (~first).astype(float)               # non-first reviews (the ones with a recall label)
    df["ynf"] = y * df["nf"]
    df["fail"] = (df["rating"] == 1).astype(float) * df["nf"]
    df["lapse"] = ((df["rating"] == 1) & (df["state"] == 2)).astype(float)
    for k in (1, 2, 3, 4):
        df[f"r{k}"] = (df["rating"] == k).astype(float)
    df["logdur"] = np.log1p(df["duration"].clip(0, 60000) / 1000.0)
    e_days = np.where(first, np.nan, df["elapsed_seconds"].clip(lower=0) / 86400.0)
    df["e"] = e_days

    # --- state of the queried review (known before answering: learning / review / relearning / filtered)
    out["s_state"] = df["state"].astype(float)

    # --- card history
    gc = df.groupby("card_id", sort=False)
    c_n = gc.cumcount().astype(float)
    out["c_n"] = np.log1p(c_n)
    out["c_lapses"] = _past(df, "card_id", "lapse")
    past_fail = _past(df, "card_id", "fail")
    past_nf = _past(df, "card_id", "nf")
    out["c_fail_frac"] = (past_fail / past_nf).where(past_nf > 0)
    cum_fail = gc["fail"].cumsum()
    streak_after = df.groupby([df["card_id"], cum_fail], sort=False)["ynf"].cumsum()
    out["c_streak"] = streak_after.groupby(df["card_id"], sort=False).shift(1).fillna(0)
    cnt = {k: _past(df, "card_id", f"r{k}") for k in (1, 2, 3, 4)}
    tot = sum(cnt.values())
    succ = cnt[2] + cnt[3] + cnt[4]
    out["c_hard_frac"] = (cnt[2] / succ).where(succ > 0)
    out["c_easy_frac"] = (cnt[4] / succ).where(succ > 0)
    ent = 0
    for k in (1, 2, 3, 4):
        q = (cnt[k] / tot).where(tot > 0)
        ent = ent - (q * np.log(q.where(q > 0, 1.0)))
    out["c_entropy"] = ent
    prev_e = gc["e"].shift(1)
    out["c_growth"] = np.log((df["e"] + 0.01) / (prev_e + 0.01))
    max_e = gc["e"].cummax()
    max_prev = max_e.groupby(df["card_id"], sort=False).shift(1)
    out["c_max_ratio"] = np.log((df["e"] + 0.01) / (max_prev + 0.01))
    out["c_age"] = np.log1p(day - gc["day_offset"].transform("first"))
    out["c_today"] = df.groupby(["card_id", "day_offset"], sort=False).cumcount().astype(float)
    out["c_dur_mean"] = (_past(df, "card_id", "logdur") / c_n).where(c_n > 0)
    out["c_dur_last"] = gc["logdur"].shift(1)
    out["c_first_rating"] = gc["rating"].transform("first").astype(float).where(c_n > 0)

    # --- note siblings
    note_size = cards.groupby("note_id").size()
    out["n_size"] = df["note_id"].map(note_size).fillna(1).astype(float)
    sib_n = _past(df, "note_id", "nf") - past_nf
    out["n_sib_n"] = np.log1p(_past(df, "note_id", "one") - c_n)
    sib_fail = _past(df, "note_id", "fail") - past_fail
    out["n_sib_fail"] = (sib_fail / sib_n).where(sib_n > 0)
    df["dayf"] = day.astype(float)
    sib_day = _carry_other(df, "note_id", "card_id", "dayf")
    out["n_sib_gap"] = np.log1p(day - sib_day)
    out["n_sib_last_y"] = _carry_other(df, "note_id", "card_id", "y")

    # --- deck and preset
    deck_size = cards.groupby("deck_id").size()
    out["d_size"] = np.log1p(df["deck_id"].map(deck_size).fillna(0).astype(float))
    parent = dict(zip(decks["deck_id"], decks["parent_id"]))
    depth = {}
    for d0 in decks["deck_id"]:
        k, cur, seen = 0, d0, set()
        while cur in parent and parent[cur] in parent and cur not in seen:
            seen.add(cur)
            cur = parent[cur]
            k += 1
        depth[d0] = k
    out["d_depth"] = df["deck_id"].map(depth).fillna(-1).astype(float)
    p_nd = decks.groupby("preset_id").size()
    out["p_ndecks"] = np.log1p(df["preset_id"].map(p_nd).fillna(0).astype(float))
    cards_p = cards.merge(decks[["deck_id", "preset_id"]], on="deck_id", how="left")
    p_size = cards_p.groupby("preset_id").size()
    out["p_size"] = np.log1p(df["preset_id"].map(p_size).fillna(0).astype(float))
    d_nf = _past(df, "deck_id", "nf") - _past(df, ["deck_id", "card_id"], "nf")
    d_y = _past(df, "deck_id", "ynf") - _past(df, ["deck_id", "card_id"], "ynf")
    out["d_ret"] = ((d_y + 1.8) / (d_nf + 2.0))
    out["d_today"] = np.log1p(df.groupby(["deck_id", "day_offset"], sort=False).cumcount().astype(float))

    # --- creation order (card ids are creation-ordered ranks over the export; snapshot = uses export-time tables)
    all_ids = np.union1d(cards["card_id"].values, df["card_id"].unique())
    nall = max(len(all_ids), 1)
    out["k_rank"] = (np.searchsorted(all_ids, df["card_id"].values) + 1) / nall
    # past-only: at a card's introduction, the fraction of earlier-introduced cards with a larger id (out-of-order
    # learning), and the id jump from the previously introduced card
    intro = df.loc[first, ["card_id"]].drop_duplicates("card_id")
    ids = intro["card_id"].values
    pos_all = np.searchsorted(all_ids, ids)
    tree = np.zeros(nall + 1, dtype=np.int64)
    later = np.zeros(len(ids))
    for j, q in enumerate(pos_all):
        k, s_ = q + 1, 0          # count earlier intros with position <= q
        while k > 0:
            s_ += tree[k]
            k -= k & -k
        later[j] = (j - s_) / max(j, 1)
        k = q + 1
        while k <= nall:
            tree[k] += 1
            k += k & -k
    jump = np.log1p(np.abs(np.diff(pos_all, prepend=pos_all[0] if len(pos_all) else 0)))
    out["k_order_gap"] = df["card_id"].map(dict(zip(ids, later))).astype(float)
    out["k_intro_jump"] = df["card_id"].map(dict(zip(ids, jump))).astype(float)
    # neighbours by creation order: buckets of 16 consecutive card ids (past labels, own card excluded)
    df["bucket"] = df["card_id"] // 16
    b_nf = _past(df, "bucket", "nf") - past_nf
    b_y = _past(df, "bucket", "ynf") - _past(df, "card_id", "ynf")
    out["k_nb_ret"] = ((b_y + 1.8) / (b_nf + 2.0))
    out["k_nb_n"] = np.log1p(b_nf)
    ncards = max(len(cards), 1)
    out["k_backlog"] = np.log1p((ncards - first.cumsum() + first.astype(int)).clip(lower=0).astype(float))

    # --- user running stats
    r = np.arange(n, dtype=float)
    u_nf = df["nf"].cumsum() - df["nf"]
    u_y = df["ynf"].cumsum() - df["ynf"]
    out["u_n"] = np.log1p(r)
    out["u_ret"] = (u_y + 1.8) / (u_nf + 2.0)
    rev = (df["state"] == 2).astype(float)
    ry = rev * y
    out["u_ret_rev"] = ((ry.cumsum() - ry) + 1.8) / ((rev.cumsum() - rev) + 2.0)
    # last 100 labelled reviews
    nf_idx = np.flatnonzero(df["nf"].values > 0)
    ynf_vals = y.values[nf_idx]
    cs = np.concatenate([[0.0], np.cumsum(ynf_vals)])
    pos = np.searchsorted(nf_idx, np.arange(n), side="left")   # labelled reviews strictly before row
    lo = np.maximum(pos - 100, 0)
    cnt100 = pos - lo
    out["u_ret_100"] = np.where(cnt100 > 0, (cs[pos] - cs[lo] + 1.8) / (cnt100 + 2.0), np.nan)
    t_nf = _past(df, "day_offset", "nf")
    t_fail = _past(df, "day_offset", "fail")
    out["u_today_fail"] = ((t_fail + 0.2) / (t_nf + 2.0))
    days = np.sort(day.unique())
    di = np.searchsorted(days, day.values)
    prev_day = np.where(di > 0, days[np.maximum(di - 1, 0)], np.nan)
    out["u_gap"] = np.log1p(day.values - prev_day)
    per_day = day.value_counts()
    out["u_yday"] = np.log1p(pd.Series(prev_day).map(per_day).values.astype(float))
    today_so_far = df.groupby("day_offset", sort=False).cumcount().values
    past_days = np.maximum(di, 1)
    past_reviews = np.concatenate([[0], np.cumsum(per_day.reindex(days).values)])[di]
    out["u_load"] = np.log((today_so_far + 1.0) / (past_reviews / past_days + 1.0))
    sc = df["r2"] + df["r3"] + df["r4"]
    s_cum = sc.cumsum() - sc
    out["u_hard"] = ((df["r2"].cumsum() - df["r2"]) + 0.1) / (s_cum + 1.0)
    out["u_easy"] = ((df["r4"].cumsum() - df["r4"]) + 0.1) / (s_cum + 1.0)
    df["wd"] = day % 7
    w_nf = _past(df, "wd", "nf")
    w_y = _past(df, "wd", "ynf")
    out["u_wd_ret"] = (w_y + 1.8) / (w_nf + 2.0) - out["u_ret"]
    out["u_dur_last"] = df["logdur"].shift(1)
    out["u_dur_mean"] = ((df["logdur"].cumsum() - df["logdur"]) / r.clip(min=1)).where(r > 0)

    clock = build_clock(user, df)
    for c in clock.columns:
        out[c] = clock[c].values

    # curve-head versions of the time-varying features: the value at the card's previous review
    prev_row = gc["review_th"].shift(1)
    for name, (_, kind) in FEATURES.items():
        if kind == "dyn":
            vals = out[name].values
            idx = prev_row.values
            lag = np.full(n, np.nan)
            ok = ~np.isnan(idx)
            lag[ok] = vals[idx[ok].astype(int) - 1]
            out[name + "@c"] = lag
    return out


def one_user(args):
    user, dump, fork_dump = args
    z = np.load(Path(dump) / f"{user}_sm56_w8+s8.npz")
    rows = pd.DataFrame({"user_id": user, "review_th": z["review_th"], "y": z["y"], "bin": z["bin"],
                         "p_a": z["p_ahead"], "p_i": z["p_instant"]})
    if fork_dump:
        f = Path(fork_dump) / f"{user}_fork_w8+s8.npz"
        if f.exists():
            zf = np.load(f)
            fk = pd.DataFrame({"review_th": zf["review_th"], "f_a": zf["p_ahead"], "f_i": zf["p_instant"]})
            rows = rows.merge(fk, on="review_th", how="left")
    feats = build_user(user)
    rows = rows.merge(feats, on="review_th", how="left", validate="one_to_one")
    for c in rows.columns:
        if rows[c].dtype == np.float64 and c not in ("p_a", "p_i", "f_a", "f_i"):
            rows[c] = rows[c].astype(np.float32)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--fork-dump", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    users = sorted(int(p.name.split("_")[0]) for p in Path(a.dump).glob("*_sm56_w8+s8.npz"))
    with ProcessPoolExecutor(a.workers) as ex:
        parts = list(ex.map(one_user, [(u, a.dump, a.fork_dump) for u in users]))
    allrows = pd.concat(parts, ignore_index=True)
    allrows.to_parquet(a.out)
    print(f"[sm70 features] {len(users)} users, {len(allrows)} test reviews -> {a.out}")


if __name__ == "__main__":
    main()
