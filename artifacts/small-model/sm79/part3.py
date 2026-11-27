#!/usr/bin/env python
"""What makes cards stick (SM79), part 3: does a same-day RELEARNING review after a lapse help?
Mirror of the part 2 "slip" design for lapses. CPU only, read only on the benchmark dataset. Same sample and filters
as parts 1-2 (training users of the 8k/2k run inside 5000-10000, random.Random(79).sample(1000), >= 200 reviews,
>= 50 included new cards).
usage: part3.py --users N --procs P --deadline HH:MM --out JSON"""
import json, sys, time, random, datetime as dt
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd

MAIN = Path(__file__).resolve().parents[3]
RUN = json.load(open(MAIN / "artifacts/full-run/run-8k-20261013/run.json"))
DATA = MAIN / RUN["dataset_path"]
FOLLOW = 180
MIN_DECK = 10            # lapse events a deck needs for its "usually shows a same-day step" share
THR = (0.5, 0.7, 0.9)    # share of the deck's lapses that got a same-day relearning review; 0.7 = main (part 2)
OUTS = ["next_pass", "log_next_ed", "lap180", "rev_cross180", "rev_tot180", "log_rec_ed", "reach21"]
BAL = ["prior_lapses", "log_last_ivl", "prior_pass", "log_load", "rank_frac", "prev_rating", "log_card_i"]
_DECKS = None


def presets():
    global _DECKS
    if _DECKS is None:
        _DECKS = pd.read_parquet(DATA / "decks", columns=["deck_id", "preset_id", "user_id"])
    return _DECKS


def grp(df, strat, gcol, ng, outs):
    """per group: for each outcome, non-NaN count, raw sum, residual sum (residual within strata, NaN-aware)."""
    g = df[gcol].to_numpy().astype(int)
    r = {}
    for o in outs:
        v = df[o].to_numpy(float)
        ok = ~np.isnan(v)
        if ok.sum() == 0:
            r[o] = [[0] * ng, [0.0] * ng, [0.0] * ng]
            continue
        d = df.loc[ok, strat].copy()
        d["_v"] = v[ok]
        m = d.groupby(strat, sort=False)["_v"].transform("mean").to_numpy()
        res = v[ok] - m
        gg = g[ok]
        r[o] = [np.bincount(gg, minlength=ng).tolist(), np.bincount(gg, v[ok], ng).tolist(),
                np.bincount(gg, res, ng).tolist()]
    return r


def mixed(df, strat, gcol):
    a = df.groupby(strat, sort=False)[gcol].transform(lambda s: (s == 0).any() & (s != 0).any())
    return df[a.to_numpy().astype(bool)]


def one(u):
    rv = pd.read_parquet(DATA / "revlogs" / f"user_id={u}",
                         columns=["card_id", "day_offset", "rating", "elapsed_days"])
    rv = rv[rv.rating.between(1, 4)].reset_index(drop=True)
    if len(rv) < 200:
        return None
    cd = pd.read_parquet(DATA / "cards" / f"user_id={u}", columns=["card_id", "note_id", "deck_id"])
    rv = rv.merge(cd[["card_id", "note_id", "deck_id"]], on="card_id", how="left")
    rv = rv[rv.note_id.notna()].reset_index(drop=True)
    n = len(rv)
    last = int(rv.day_offset.max())
    # part 2 inclusion: >= 50 new cards (first review in data, >= 180 days before the last review day)
    first = rv.groupby("card_id").head(1)
    inc = ((first.elapsed_days == -1) & (first.day_offset <= last - FOLLOW)).sum()
    if inc < 50:
        return None
    res = {"user": u, "reviews": n}

    # per-day load and position in the day (original, chronological order)
    day = rv.day_offset.to_numpy()
    dsize = rv.groupby("day_offset").day_offset.transform("size").to_numpy()
    dpos = rv.groupby("day_offset").cumcount().to_numpy()
    rank_frac = (dpos + 0.5) / dsize
    # card-grouped chronological arrays
    cn = pd.factorize(rv.card_id)[0].astype(np.int64)
    o = np.lexsort((np.arange(n), cn))
    cs, ds = cn[o], day[o]
    rat = rv.rating.to_numpy()[o]
    ed = rv.elapsed_days.to_numpy()[o].astype(float)
    deck = rv.deck_id.to_numpy()[o]
    dl, rf = dsize[o], rank_frac[o]
    key = cs.astype(np.float64) * 1e6 + ds
    pas = rat > 1
    xd = ed >= 1
    lap = (~pas) & xd
    cum = lambda x: np.r_[0, np.cumsum(x.astype(np.int64))]
    cl, cx, cxp = cum(lap), cum(xd), cum(pas & xd)
    cstart = np.r_[0, np.flatnonzero(np.diff(cs)) + 1]
    start_of = np.repeat(cstart, np.diff(np.r_[cstart, n]))   # first row index of each row's card
    J = np.flatnonzero(lap & (ds <= last - FOLLOW))
    if len(J) < 50:
        return res
    hs = np.searchsorted(key, key[J], "right")                 # first row after the lapse day (same card or beyond)
    h180 = np.searchsorted(key, key[J] + FOLLOW, "right")
    nsd = hs - J - 1                                           # same-day reviews after the lapse
    same_next = hs < n
    nxt_ok = same_next.copy()
    nxt_ok[same_next] = cs[hs[same_next]] == cs[J][same_next]
    hsc = np.minimum(hs, n - 1)
    BIG = n + 5
    idx = np.arange(n)
    nxt_pass = np.minimum.accumulate(np.where(pas, idx, BIG)[::-1])[::-1]
    nxt_r21 = np.minimum.accumulate(np.where(pas & (ed >= 21), idx, BIG)[::-1])[::-1]
    q = np.where(hs < n, nxt_pass[hsc], BIG)
    q21 = np.where(hs < n, nxt_r21[hsc], BIG)
    ev = pd.DataFrame({"deck": deck[J], "nsd": nsd})
    first_sd = np.where(nsd > 0, rat[np.minimum(J + 1, n - 1)], 0)
    last_sd = np.where(nsd > 0, rat[np.maximum(hs - 1, 0)], 0)
    ev["first_sd_pass"] = np.where(nsd > 0, first_sd > 1, False)
    ev["last_sd_pass"] = np.where(nsd > 0, last_sd > 1, False)
    # outcomes
    ev["next_pass"] = np.where(nxt_ok, pas[hsc].astype(float), np.nan)
    ev["log_next_ed"] = np.where(nxt_ok, np.log(np.maximum(ed[hsc], 1)), np.nan)
    ev["lap180"] = (cl[h180] - cl[hs]).astype(float)
    ev["rev_cross180"] = (h180 - hs).astype(float)
    ev["rev_tot180"] = ev.rev_cross180 + nsd
    qq = np.minimum(q, n - 1)
    ok_q = q < h180
    ev["log_rec_ed"] = np.where(ok_q, np.log(np.maximum(ed[qq], 1)), np.nan)
    ev["reach21"] = (q21 < h180).astype(float)
    # pre-lapse history / context
    j0 = start_of[J]
    pl = (cl[J] - cl[j0]).astype(float)
    ev["prior_lapses"] = pl
    ev["log_last_ivl"] = np.log(np.maximum(ed[J], 1))
    nx_, np_ = (cx[J] - cx[j0]), (cxp[J] - cxp[j0])
    ev["prior_pass"] = np.where(nx_ > 0, np_ / np.maximum(nx_, 1), np.nan)
    ev["log_load"] = np.log2(dl[J])
    ev["rank_frac"] = rf[J]
    ev["prev_rating"] = np.where(J > j0, rat[np.maximum(J - 1, 0)], np.nan)
    ev["log_card_i"] = np.log1p(J - j0)
    # strata
    ev["lb"] = np.digitize(pl, [1, 2, 3, 4, 5, 7, 10, 15, 25])   # fine prior-lapse bins (heavy tail of leech cards)
    ev["lo"] = np.digitize(pl, [1, 2, 4, 8])                      # ordinal bucket: 0 | 1 | 2-3 | 4-7 | 8+
    ev["ivb"] = np.floor(np.log2(np.maximum(ed[J], 1))).astype(int).clip(0, 9)
    pp = ev.prior_pass.to_numpy()
    ev["ppb"] = np.where(np.isnan(pp), -1, np.digitize(np.nan_to_num(pp), [0.8, 0.9, 0.95]))
    ev["ldb"] = np.clip(np.floor(ev.log_load.to_numpy()).astype(int), 2, 12)
    ev["rkb"] = np.digitize(ev.rank_frac.to_numpy(), [0.3, 0.5, 0.7, 0.85, 0.95, 0.99])
    ev["prb"] = ev.prev_rating.fillna(0).astype(int)
    # deck habit: share of the deck's lapses that got a same-day relearning review
    ev["got"] = (ev.nsd > 0).astype(int)
    dn = ev.groupby("deck").got.transform("size")
    ev["share"] = ev.groupby("deck").got.transform("mean").where(dn >= MIN_DECK, np.nan)
    res["n_events"] = len(ev)
    res["share_hist"] = np.histogram(ev.share.dropna(), bins=[0, .1, .3, .5, .7, .9, 1.0001])[0].tolist()
    res["got_n"] = np.bincount(np.minimum(ev.nsd, 3), minlength=4).tolist()
    A, B = ["deck"], ["deck", "lb", "ivb", "ppb"]
    C = B + ["ldb", "rkb"]
    for thr in THR:
        hv = ev[ev.share >= thr].copy()
        hv["g"] = hv.got
        for nm, st in (("A", A), ("B", B), ("C", C)):
            if thr != 0.7 and nm != "B":
                continue
            mm = mixed(hv, st, "g")
            if len(mm) > 20:
                res[f"slip_{nm}_{thr}"] = grp(mm, st, "g", 2, OUTS)
        if thr == 0.7:
            for nm, st in (("A", A), ("B", B)):
                mm = mixed(hv, st, "g")
                if len(mm) > 20:
                    res[f"bal_{nm}"] = grp(mm, st, "g", 2, BAL)
            # by which lapse of the card this is (0 = first lapse ...), stratified without the lapse-count bins
            for lo in range(5):
                fo = hv[hv.lo == lo]
                for nm, st in (("B", ["deck", "ivb", "ppb"]), ("C", ["deck", "ivb", "ppb", "ldb", "rkb"])):
                    mm = mixed(fo, st, "g")
                    if len(mm) > 20:
                        res[f"slip_lapse{lo}_{nm}"] = grp(mm, st, "g", 2, OUTS)
                if lo == 0:
                    mm = mixed(fo, ["deck", "ivb", "ppb"], "g")
                    if len(mm) > 20:
                        res["bal_lapse0_B"] = grp(mm, ["deck", "ivb", "ppb"], "g", 2, BAL)
            # 1 vs 2+ same-day relearning reviews
            h2 = hv.copy()
            h2["g"] = np.minimum(h2.nsd, 2)
            mm = mixed(h2, B, "g")
            if len(mm) > 20:
                res["count_B"] = grp(mm, B, "g", 3, OUTS)
            # first same-day review passed vs failed (among those with a step)
            h3 = hv.copy()
            h3["g"] = np.where(h3.nsd == 0, 0, np.where(h3.first_sd_pass, 1, 2))
            mm = mixed(h3, B, "g")
            if len(mm) > 20:
                res["passfail_B"] = grp(mm, B, "g", 3, OUTS)
            # last same-day review passed vs not (the state the card was left in)
            h4 = hv.copy()
            h4["g"] = np.where(h4.nsd == 0, 0, np.where(h4.last_sd_pass, 1, 2))
            mm = mixed(h4, B, "g")
            if len(mm) > 20:
                res["lastpass_B"] = grp(mm, B, "g", 3, OUTS)
    return res


def run_one(u):
    try:
        return one(u)
    except Exception:  # noqa: BLE001
        import traceback
        return {"user": u, "error": traceback.format_exc()[-800:]}


def summarize(rows, key, names, outs, B=1000):
    rows = [r for r in rows if key in r]
    if not rows:
        return None
    idx = np.random.default_rng(0).integers(0, len(rows), (B, len(rows)))
    tab = {"users": len(rows), "rows": []}
    for k, name in enumerate(names):
        row = {"bucket": name}
        for o in outs:
            N = np.array([r[key][o][0][k] for r in rows], float)
            raw = np.array([r[key][o][1][k] for r in rows])
            rs = np.array([r[key][o][2][k] for r in rows])
            N0 = np.array([r[key][o][0][0] for r in rows], float)
            rs0 = np.array([r[key][o][2][0] for r in rows])
            f = lambda I: rs[I].sum(-1) / np.maximum(N[I].sum(-1), 1) - rs0[I].sum(-1) / np.maximum(N0[I].sum(-1), 1)
            row[o] = {"n": int(N.sum()), "users": int((N > 0).sum()), "raw": float(raw.sum() / max(N.sum(), 1)),
                      "vs_ref": float(f(np.arange(len(rows)))) if k else 0.0,
                      "ci": np.percentile(f(idx), [2.5, 97.5]).tolist() if k else [0, 0]}
        tab["rows"].append(row)
    return tab


def main():
    a = sys.argv[1:]
    g = lambda k, d: a[a.index(k) + 1] if k in a else d
    nu, procs, out = int(g("--users", 1000)), int(g("--procs", 8)), g("--out", "part3.json")
    hh, mm = map(int, g("--deadline", "22:45").split(":"))
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
    rows = [r for r in rows if "error" not in r and "n_events" in r]
    print(f"{len(rows)} users ok, {len(errs)} errors {errs[:2]}, {time.time() - t0:.0f}s", flush=True)
    S = {"started": dt.datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
         "sample": {"requested": nu, "users": len(rows), "reviews": int(sum(r["reviews"] for r in rows)),
                    "lapse_events": int(sum(r["n_events"] for r in rows)), "errors": len(errs)},
         "share_hist_bins": [0, .1, .3, .5, .7, .9, 1.0], "share_hist": np.sum([r["share_hist"] for r in rows], 0).tolist(),
         "got_n_0_1_2_3plus": np.sum([r["got_n"] for r in rows], 0).tolist()}
    G2 = ["slipped (no further review that day)", "same-day relearning review(s)"]
    for kk in [k for k in rows[0] if k.startswith("slip_") or k.startswith("bal_")] :
        pass
    keys = sorted({k for r in rows for k in r if k.startswith(("slip_", "bal_", "count_", "passfail_", "lastpass_"))})
    for kk in keys:
        outs = BAL if kk.startswith("bal_") else OUTS
        names = G2
        if kk.startswith("count_"):
            names = ["slipped", "1 same-day review", "2+ same-day reviews"]
        elif kk.startswith("passfail_"):
            names = ["slipped", "step, first same-day review passed", "step, first same-day review Again"]
        elif kk.startswith("lastpass_"):
            names = ["slipped", "step, last same-day review passed", "step, last same-day review Again"]
        t = summarize(rows, kk, names, outs)
        if t:
            S[kk] = t
    S["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    json.dump(S, open(out, "w"), indent=1, default=float)
    print("wrote", out, time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
