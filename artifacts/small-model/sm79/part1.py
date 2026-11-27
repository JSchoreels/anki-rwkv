#!/usr/bin/env python
"""What makes cards stick (SM79), part 1: note structure (A), same-day siblings (B), new-card load (C).
CPU only, read only on the benchmark dataset. Training users of the 8k/2k run inside 5000-10000 (never validation,
the 464/732/2,225 eval pools, or final test). Within-user (and within user x deck) residuals, bootstrap over users.
usage: part1.py --users N --procs P --deadline HH:MM --out JSON"""
import json, sys, time, random, datetime as dt
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd

MAIN = Path(__file__).resolve().parents[3]
RUN = json.load(open(MAIN / "artifacts/full-run/run-8k-20261013/run.json"))
DATA = MAIN / RUN["dataset_path"]
FOLLOW = 180          # card's first review must be >= 180 days before the user's last review day
NOTE_B = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 9, "5-9"), (10, 10**9, "10+")]
NEW_B = [(1, 10, "1-10"), (11, 30, "11-30"), (31, 100, "31-100"), (101, 10**9, "100+")]
OUT = ["reach21_10", "lapses10", "leech", "reviews180"]


def bucket(x, B):
    o = np.full(len(x), -1)
    for k, (lo, hi, _) in enumerate(B):
        o[(x >= lo) & (x <= hi)] = k
    return o


def resid(df, cols, strat):
    g = df.groupby(strat, sort=False)
    return {c: (df[c] - g[c].transform("mean")).to_numpy() for c in cols}


def agg(keys, nb, vals):
    """per-bucket count, raw sum, residual sum per outcome."""
    r = {"n": np.bincount(keys, minlength=nb).tolist()}
    for name, (raw, res) in vals.items():
        r[name] = [np.bincount(keys, raw, nb).tolist(), np.bincount(keys, res, nb).tolist()]
    return r


def one(u):
    rv = pd.read_parquet(DATA / "revlogs" / f"user_id={u}", columns=["card_id", "day_offset", "rating", "elapsed_days"])
    rv = rv[rv.rating.between(1, 4)].reset_index(drop=True)
    if len(rv) < 200:
        return None
    cd = pd.read_parquet(DATA / "cards" / f"user_id={u}", columns=["card_id", "note_id", "deck_id"])
    cd = cd.assign(nsize=cd.groupby("note_id").card_id.transform("size").to_numpy())
    rv = rv.merge(cd, on="card_id", how="left")
    rv = rv[rv.note_id.notna()].reset_index(drop=True)
    rv["note_id"] = rv.note_id.astype(np.int64)
    rv["pos"] = np.arange(len(rv))
    rv["i"] = rv.groupby("card_id").cumcount()
    last = rv.day_offset.max()
    lapse = (rv.rating == 1) & (rv.elapsed_days >= 1)
    hit = (rv.rating > 1) & (rv.elapsed_days >= 21)
    # siblings same day: distinct other cards of the note reviewed earlier / only later that day
    fp = rv.groupby(["note_id", "day_offset", "card_id"]).pos.min().rename("fp").reset_index()
    fp["g"] = fp.groupby(["note_id", "day_offset"]).ngroup()
    rv = rv.merge(fp, on=["note_id", "day_offset", "card_id"], how="left").sort_values("pos").reset_index(drop=True)
    gs = fp.groupby("g").size()
    BIG = len(rv) + 1
    keys = np.sort(fp.g.to_numpy() * BIG + fp.fp.to_numpy())
    gv, pv, fv = rv.g.to_numpy(), rv.pos.to_numpy(), rv.fp.to_numpy()
    nbefore = np.searchsorted(keys, gv * BIG + pv) - np.searchsorted(keys, gv * BIG) - (fv < pv)
    ndist = gs.reindex(gv).to_numpy()
    rv["sib"] = np.where(nbefore > 0, 1, np.where(ndist > 1, 2, 0))  # 0 none that day, 1 earlier, 2 only later
    # most recent earlier row of a different sibling the same day: row gap and whether it passed
    o = rv.sort_values(["g", "pos"]).reset_index()
    og, oc = o.g.to_numpy(), o.card_id.to_numpy()
    newrun = np.r_[True, (og[1:] != og[:-1]) | (oc[1:] != oc[:-1])]
    start = np.maximum.accumulate(np.where(newrun, np.arange(len(o)), 0))
    prev = start - 1
    ok = (prev >= 0) & (og[np.maximum(prev, 0)] == og)
    gap = np.where(ok, o.pos.to_numpy() - o.pos.to_numpy()[np.maximum(prev, 0)], -1)
    ppass = np.where(ok, o.rating.to_numpy()[np.maximum(prev, 0)] > 1, False)
    s2 = np.zeros(len(o), int)
    s2[ok & (gap <= 5) & ppass] = 1
    s2[ok & (gap <= 5) & ~ppass] = 2
    s2[ok & (gap > 5) & ppass] = 3
    s2[ok & (gap > 5) & ~ppass] = 4
    rv["sib2"] = pd.Series(s2, index=o["index"].to_numpy()).sort_index().to_numpy()
    rv.loc[(rv.sib == 2).to_numpy(), "sib2"] = 5
    lapse = (rv.rating == 1) & (rv.elapsed_days >= 1)
    hit = (rv.rating > 1) & (rv.elapsed_days >= 21)
    # cards: first review in data, enough follow-up
    first = rv[rv.i == 0].set_index("card_id")
    c = pd.DataFrame(index=first.index)
    c["deck"] = first.deck_id
    c["nsize"] = first.nsize
    c["day0"] = first.day_offset
    c["sib0"] = first.sib
    c["isnew"] = first.elapsed_days == -1
    newday = c[c.isnew].groupby("day0").size()
    c["newload"] = newday.reindex(c.day0).fillna(0).to_numpy()
    c["total"] = rv.groupby("card_id").size()
    c["lapses_all"] = lapse.groupby(rv.card_id).sum()
    c["lapses10"] = (lapse & (rv.i < 10)).groupby(rv.card_id).sum()
    c["first21"] = rv.i.where(hit).groupby(rv.card_id).min().reindex(c.index)
    c["reviews180"] = ((rv.day_offset - rv.card_id.map(c.day0)) < 180).groupby(rv.card_id).sum()
    c = c[c.isnew & (c.day0 <= last - FOLLOW)].copy()
    if len(c) < 50:
        return None
    c["reach21_10"] = (c.first21 < 10).astype(float)
    c["leech"] = (c.lapses_all >= 8).astype(float)
    c["lapses10"] = c.lapses10.astype(float)
    c["reviews180"] = c.reviews180.astype(float)
    res = {"user": u, "cards": len(c), "reviews": int(len(rv))}
    # prize: share of the included cards' reviews going to the 5% / 10% most-reviewed cards (within user)
    t = np.sort(c.total.to_numpy())[::-1]
    T = t.sum()
    res["prize"] = {f"top{p}": float(t[:max(1, int(round(len(t) * p / 100)))].sum() / T) for p in (5, 10)}
    res["prize_total"] = float(T)
    r = c.first21.dropna() + 1
    res["first21_hist"] = np.bincount(np.minimum(r.astype(int), 31), minlength=32).tolist()
    res["first21_reached_any"] = int(c.first21.notna().sum())
    # A: note size, residual within user x deck, and within user only
    kb = bucket(c.nsize.to_numpy(), NOTE_B)
    R = resid(c, OUT, ["deck"])
    R0 = resid(c, OUT, [np.zeros(len(c))])
    res["A_deck"] = agg(kb, 5, {o: (c[o].to_numpy(), R[o]) for o in OUT})
    res["A_user"] = agg(kb, 5, {o: (c[o].to_numpy(), R0[o]) for o in OUT})
    # B card-level: card introduced the day a sibling was reviewed (earlier / only later), within user x deck x note size
    cs = c[(c.nsize >= 2).to_numpy()]
    if len(cs) >= 20:
        R = resid(cs, OUT, ["deck", "nsize"])
        res["B_card"] = agg(cs.sib0.to_numpy().astype(int), 3, {o: (cs[o].to_numpy(), R[o]) for o in OUT})
    # B review-level: pass rate of cross-day reviews (elapsed >= 1) of cards with siblings,
    # strata user x log2(elapsed) x note-size bucket x review-index bucket
    rr = rv[(rv.elapsed_days >= 1) & (rv.nsize >= 2)].copy()
    if len(rr) >= 50:
        rr["pass"] = (rr.rating > 1).astype(float)
        rr["eb"] = np.floor(np.log2(rr.elapsed_days)).astype(int)
        rr["nb"] = bucket(rr.nsize.to_numpy(), NOTE_B)
        rr["ib"] = np.minimum(rr.i, 10)
        R = resid(rr, ["pass"], ["eb", "nb", "ib"])
        res["B_review"] = agg(rr.sib.to_numpy().astype(int), 3, {"pass": (rr["pass"].to_numpy(), R["pass"])})
        res["B_review2"] = agg(rr.sib2.to_numpy().astype(int), 6, {"pass": (rr["pass"].to_numpy(), R["pass"])})
    # C: new cards started that day, within user x deck
    kc = bucket(c.newload.to_numpy(), NEW_B)
    R = resid(c, OUT, ["deck"])
    res["C"] = agg(kc, 4, {o: (c[o].to_numpy(), R[o]) for o in OUT})
    return res


def run_one(u):
    try:
        return one(u)
    except Exception as e:  # noqa: BLE001
        return {"user": u, "error": repr(e)}


def summarize(rows, names, key, outcomes, B=1000):
    rows = [r for r in rows if key in r]
    idx = np.random.default_rng(0).integers(0, len(rows), (B, len(rows)))
    n = np.array([r[key]["n"] for r in rows], float)
    tab = []
    for k, name in enumerate(names):
        N = n[:, k]
        row = {"bucket": name, "n": int(N.sum()), "users": int((N > 0).sum())}
        for o in outcomes:
            raw = np.array([r[key][o][0][k] for r in rows])
            res = np.array([r[key][o][1][k] for r in rows])
            den = max(N.sum(), 1)
            b = res[idx].sum(1) / np.maximum(N[idx].sum(1), 1)
            row[o] = {"raw": raw.sum() / den, "within": res.sum() / den, "ci": np.percentile(b, [2.5, 97.5]).tolist()}
        tab.append(row)
    return {"users": len(rows), "rows": tab}


def contrast(rows, key, o, a, b, B=1000):
    rows = [r for r in rows if key in r]
    idx = np.random.default_rng(1).integers(0, len(rows), (B, len(rows)))
    n = np.array([r[key]["n"] for r in rows], float)
    res = np.array([r[key][o][1] for r in rows])
    f = lambda I: res[I, a].sum(-1) / np.maximum(n[I, a].sum(-1), 1) - res[I, b].sum(-1) / np.maximum(n[I, b].sum(-1), 1)
    return {"diff": float(f(np.arange(len(rows)))), "ci": np.percentile(f(idx), [2.5, 97.5]).tolist()}


def main():
    a = sys.argv[1:]
    g = lambda k, d: a[a.index(k) + 1] if k in a else d
    nu, procs, out = int(g("--users", 600)), int(g("--procs", 8)), g("--out", "part1.json")
    hh, mm = map(int, g("--deadline", "21:48").split(":"))
    dl = dt.datetime.now().replace(hour=hh, minute=mm, second=0)
    pool_u = sorted(u for u in RUN["train_users"] if 5000 <= u <= 10000)
    users = random.Random(79).sample(pool_u, nu)
    rows, t0 = [], time.time()
    with Pool(procs) as p:
        for r in p.imap_unordered(run_one, users):
            if r is not None:
                rows.append(r)
            if dt.datetime.now() >= dl:
                print("deadline hit", flush=True)
                p.terminate()
                break
    errs = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    print(f"{len(rows)} users ok, {len(errs)} errors {errs[:3]}, {time.time() - t0:.0f}s", flush=True)
    W = np.array([r["prize_total"] for r in rows])
    S = {"sample": {"pool": "8k/2k run train_users inside 5000-10000 (random.Random(79).sample)", "requested": nu,
                    "users": len(rows), "cards": int(sum(r["cards"] for r in rows)),
                    "reviews": int(sum(r["reviews"] for r in rows)), "errors": len(errs), "follow_up_days": FOLLOW}}
    pr = {}
    for p in ("top5", "top10"):
        v = np.array([r["prize"][p] for r in rows])
        I = np.random.default_rng(2).integers(0, len(v), (1000, len(v)))
        pr[p] = {"review_weighted": float((v * W).sum() / W.sum()), "median_user": float(np.median(v)),
                 "ci_weighted": np.percentile((v[I] * W[I]).sum(1) / W[I].sum(1), [2.5, 97.5]).tolist()}
    S["prize"] = pr
    h = np.sum([r["first21_hist"] for r in rows], 0)
    nc = S["sample"]["cards"]
    S["first21"] = {"cards": nc, "reached_ever": int(sum(r["first21_reached_any"] for r in rows)),
                    "reached_within": {k: float(h[1:k + 1].sum() / nc) for k in (3, 5, 7, 10, 15, 20)},
                    "median_reviews_if_reached": int(np.searchsorted(np.cumsum(h), h.sum() / 2))}
    NB, CB = [b[2] for b in NOTE_B], [b[2] for b in NEW_B]
    S["A_deck"] = summarize(rows, NB, "A_deck", OUT)
    S["A_user"] = summarize(rows, NB, "A_user", OUT)
    S["A_contrasts_deck"] = {o: {NB[k]: contrast(rows, "A_deck", o, k, 0) for k in range(1, 5)} for o in OUT}
    SB = ["no sibling that day", "sibling earlier that day", "sibling only later that day"]
    S["B_review"] = summarize(rows, SB, "B_review", ["pass"])
    S["B_review_contrasts"] = {"earlier_vs_none": contrast(rows, "B_review", "pass", 1, 0),
                               "later_vs_none": contrast(rows, "B_review", "pass", 2, 0)}
    SB2 = ["no sibling that day", "sibling <=5 rows earlier, passed", "sibling <=5 rows earlier, failed",
           "sibling >5 rows earlier, passed", "sibling >5 rows earlier, failed", "sibling only later that day"]
    S["B_review2"] = summarize(rows, SB2, "B_review2", ["pass"])
    S["B_review2_contrasts"] = {SB2[k]: contrast(rows, "B_review2", "pass", k, 0) for k in range(1, 6)}
    S["B_card"] = summarize(rows, SB, "B_card", OUT)
    S["B_card_contrasts"] = {o: {"earlier_vs_none": contrast(rows, "B_card", o, 1, 0),
                                 "later_vs_none": contrast(rows, "B_card", o, 2, 0)} for o in OUT}
    S["C"] = summarize(rows, CB, "C", OUT)
    S["C_contrasts"] = {o: {CB[k]: contrast(rows, "C", o, k, 0) for k in range(1, 4)} for o in OUT}
    json.dump(S, open(out, "w"), indent=1, default=float)
    print("wrote", out)


if __name__ == "__main__":
    main()
