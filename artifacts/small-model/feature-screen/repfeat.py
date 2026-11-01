"""Per-user rep-number statistics (user idea 2026-10-17): with SM-2 a card's learning steps are its reviews 2, 3 (4),
and relearning steps the first reviews after an Again, so a user's own pass rate at "rep k of a card" (learned from
their other cards) may tell a model what review type told the fork.

Keys (per review: the predicted review's position, known at the card's previous review):
  rep      k = the card's review number (2, 3, ... capped at 10)
  phase    (ever had an interval >= 1 day before, reviews since the last Again or since the first review, capped at 6)
  preset / deck   (preset or deck, k)
Per key, causal running statistics over the user's earlier reviews (row index < t):
  rp_<key>          smoothed pass rate minus the user's running pass rate (m = 10 pseudo-reviews at that rate)
  rp_<key>_n        log1p of reviews with that key so far
  rr_<model>_<head>_<key>   running mean residual y - p of the model on scored reviews with that key (shrunk, m = 10)
t = the predicted review's row (instant) or the card's previous review's row (curve: the forecast point); *_atprev
holds the curve version. Written into fx/<user>.npz.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ORIG = Path(r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011")
IDD = Path(r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k")
M = 10.0


def keyed_running(key, val, query_t, query_key, mask=None):
    """sum and count of val over rows r < t with key[r] == query_key, for each query (t, key)."""
    n = len(key); mask = np.ones(n, bool) if mask is None else mask
    s_out = np.zeros(len(query_t)); c_out = np.zeros(len(query_t))
    rows = np.flatnonzero(mask)
    order = pd.Series(rows).groupby(key[rows]).indices
    qgroups = pd.Series(np.arange(len(query_t))).groupby(query_key).indices
    for k, qi in qgroups.items():
        if k not in order:
            continue
        r = rows[order[k]]; cs = np.concatenate([[0], np.cumsum(val[r])])
        pos = np.searchsorted(r, query_t[qi], side="left")
        s_out[qi] = cs[pos]; c_out[qi] = pos
    return s_out, c_out


def add(user):
    path = Path("fx") / f"{user}.npz"
    z = dict(np.load(path))
    rv = pd.read_parquet(ORIG / "revlogs" / f"user_id={user}", columns=["card_id", "rating", "elapsed_days"])
    n = len(rv); card = rv["card_id"].to_numpy(); y = (rv["rating"].to_numpy() > 1).astype(float)
    g = rv.groupby("card_id", sort=False)
    rep = np.minimum(g.cumcount().to_numpy() + 1, 10)
    grad = g["elapsed_days"].transform(lambda s: (s.shift(1) >= 1).cummax()).fillna(False).to_numpy().astype(int)
    since = np.zeros(n, int); last = {}
    for i in range(n):
        c = card[i]; since[i] = min(last.get(c, 0) + 1, 6)
        last[c] = 0 if y[i] == 0 else last.get(c, 0) + 1
    keys = {"rep": rep, "phase": grad * 10 + since}
    ic = pd.read_parquet(IDD / "cards" / f"user_id={user}") if (IDD / "cards" / f"user_id={user}").is_dir() else None
    idk = pd.read_parquet(IDD / "decks" / f"user_id={user}") if (IDD / "decks" / f"user_id={user}").is_dir() else None
    if ic is not None and idk is not None:
        rid = pd.read_parquet(IDD / "revlogs" / f"user_id={user}", columns=["card_id"]).loc[rv.index]
        deck = pd.Series(rid["card_id"].to_numpy()).map(ic.set_index("card_id")["deck_id"]).fillna(-1).to_numpy().astype(np.int64)
        preset = pd.Series(deck).map(idk.set_index("deck_id")["preset_id"]).fillna(-1).to_numpy().astype(np.int64)
        keys["deck"] = deck * 16 + rep; keys["preset"] = preset * 16 + rep
    prev = pd.Series(np.arange(n)).groupby(card).shift(1).to_numpy()
    has = ~np.isnan(prev); tprev = np.where(has, prev, 0).astype(int)
    t_now = np.arange(n)
    allkey = np.zeros(n, int)
    f = {}
    for when, t in (("", t_now), ("_atprev", tprev)):
        us, uc = keyed_running(allkey, y, t, allkey)
        urate = (us + 0.85 * 5) / (uc + 5)
        for name, key in keys.items():
            s, c = keyed_running(key, y, t, key)
            v = (s + M * urate) / (c + M) - urate
            f[f"rp_{name}{when}"] = np.where(has, v, np.nan); f[f"rp_{name}_n{when}"] = np.where(has, np.log1p(c), np.nan)
    # model residuals on scored reviews
    for m in ("fork", "st"):
        d = np.load(Path("dump") / f"{user}_{m}_w8+s8.npz")
        idx = d["review_th"] - 1
        for h, when, t in (("ahead", "_atprev", tprev), ("instant", "", t_now)):
            res = np.zeros(n); res[idx] = d["y"] - d[f"p_{h}"]
            mask = np.zeros(n, bool); mask[idx] = True
            for name, key in keys.items():
                s, c = keyed_running(key, res, t, key, mask)
                f[f"rr_{m}_{h}_{name}{when}"] = np.where(has, s / (c + M), np.nan)
    z.update(f)
    np.savez_compressed(path, **z)
    return "ok" if "deck" in keys else "no deck"


if __name__ == "__main__":
    import json
    import sys
    from concurrent.futures import ProcessPoolExecutor
    users = [u for s in (0, 1) for u in json.load(open(f"shard{s}.json"))["users"]]
    with ProcessPoolExecutor(4) as ex:
        out = list(ex.map(add, users))
    print({k: out.count(k) for k in set(out)})
