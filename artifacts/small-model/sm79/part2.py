#!/usr/bin/env python
"""What makes cards stick (SM79), part 2: early "won't stick" flag (1), missed-days timing natural experiment (2),
early spacing of new cards (3). CPU only, read only on the benchmark dataset. Same sample and filters as part 1
(training users of the 8k/2k run inside 5000-10000, random.Random(79).sample(1000), >= 200 reviews, >= 50 cards).
usage: part2.py --users N --procs P --deadline HH:MM --out JSON"""
import json, sys, time, random, datetime as dt
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd

MAIN = Path(__file__).resolve().parents[3]
RUN = json.load(open(MAIN / "artifacts/full-run/run-8k-20261013/run.json"))
DATA = MAIN / RUN["dataset_path"]
FOLLOW = 180
CAP = 600            # cards per user kept for the flag model (labels computed on all included cards)
KS = (1, 2, 3, 5)    # flag after the card's k-th review
LAPSE_N = 4          # "won't stick" label B: >= 4 cross-day lapses over the whole history
GAP_B = [(0, 0, "on time (active previous 3 days)"), (2, 3, "after 2-3 idle days"), (4, 7, "after 4-7 idle days"),
         (8, 30, "after 8-30 idle days")]
SP_B = ["next exposure next day or later", "1 extra same-day exposure", "2+ extra same-day exposures"]
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
    """keep strata where the reference group (0) and at least one other group both appear."""
    a = df.groupby(strat, sort=False)[gcol].transform(lambda s: (s == 0).any() & (s != 0).any())
    return df[a.to_numpy().astype(bool)]


def one(u):
    rv = pd.read_parquet(DATA / "revlogs" / f"user_id={u}",
                         columns=["card_id", "day_offset", "rating", "duration", "elapsed_days", "elapsed_seconds"])
    rv = rv[rv.rating.between(1, 4)].reset_index(drop=True)
    if len(rv) < 200:
        return None
    cd = pd.read_parquet(DATA / "cards" / f"user_id={u}", columns=["card_id", "note_id", "deck_id"])
    cd = cd.assign(nsize=cd.groupby("note_id").card_id.transform("size").to_numpy())
    dk = presets()
    dk = dk[dk.user_id.astype(int) == u][["deck_id", "preset_id"]].drop_duplicates("deck_id")
    cd = cd.merge(dk, on="deck_id", how="left")
    rv = rv.merge(cd, on="card_id", how="left")
    rv = rv[rv.note_id.notna()].reset_index(drop=True)
    rv["note_id"] = rv.note_id.astype(np.int64)
    rv["preset_id"] = rv.preset_id.fillna(-1).astype(np.int64)
    n = len(rv)
    rv["i"] = rv.groupby("card_id").cumcount()
    last = int(rv.day_offset.max())
    p = (rv.rating > 1).to_numpy()
    xd = (rv.elapsed_days >= 1).to_numpy()
    ag = (rv.rating == 1).to_numpy()
    lapse = ag & xd
    rv["pass"] = p.astype(float)
    rv["lapse"] = lapse.astype(float)

    # ---------- running context, strictly before each row ----------
    def before(key, x):
        x = pd.Series(x.astype(float))
        if key is None:
            return (x.cumsum() - x).to_numpy()
        return (x.groupby(key).cumsum() - x).to_numpy()
    upx, ux = before(None, p & xd), before(None, xd)
    dpx, dx = before(rv.deck_id, p & xd), before(rv.deck_id, xd)
    npx, nx, nag = before(rv.note_id, p & xd), before(rv.note_id, xd), before(rv.note_id, ag)
    cpx, cx, cag = before(rv.card_id, p & xd), before(rv.card_id, xd), before(rv.card_id, ag)
    ld = np.log1p(np.clip(rv.duration.to_numpy(), 0, 600000) / 1000.0)
    uld = before(None, ld) / np.maximum(np.arange(n), 1)
    newtoday = before(rv.day_offset, (rv.elapsed_days == -1).to_numpy())
    ctx = pd.DataFrame({
        "u_pass": (upx + 4.5) / (ux + 5), "u_log_n": np.log1p(np.arange(n)),
        "d_pass": (dpx + 4.5) / (dx + 5), "d_log_n": np.log1p(dx),
        "s_n": nx - cx, "s_pass": (npx - cpx + 0.9) / (nx - cx + 1), "s_again": nag - cag,
        "u_logdur": uld, "new_today": np.log1p(newtoday)})

    # ---------- cards (part 1 filters) ----------
    first = rv[rv.i == 0].set_index("card_id")
    c = pd.DataFrame(index=first.index)
    c["deck"], c["preset"], c["nsize"], c["day0"] = first.deck_id, first.preset_id, first.nsize, first.day_offset
    c["isnew"] = first.elapsed_days == -1
    c["total"] = rv.groupby("card_id").size()
    c["lapses_all"] = rv.groupby("card_id").lapse.sum()
    c["r180"] = ((rv.day_offset - rv.card_id.map(c.day0)) < FOLLOW).groupby(rv.card_id).sum()
    c = c[c.isnew & (c.day0 <= last - FOLLOW)].copy()
    if len(c) < 50:
        return None
    res = {"user": u, "cards": len(c), "reviews": n}

    # ---------- 1. early flag features ----------
    rng = np.random.default_rng(u)
    tot = c.total.to_numpy().astype(float)
    order = np.lexsort((rng.random(len(c)), -tot))
    worst = np.zeros(len(c), bool)
    worst[order[:max(1, int(round(len(c) * 0.10)))]] = True
    c["worst10"] = worst
    c["lapseN"] = c.lapses_all >= LAPSE_N
    t180 = c.r180.to_numpy().astype(float)
    o180 = np.lexsort((rng.random(len(c)), -t180))
    w180 = np.zeros(len(c), bool)
    w180[o180[:max(1, int(round(len(c) * 0.10)))]] = True
    c["worst10_180"] = w180
    keep = c.index.to_numpy()
    if len(keep) > CAP:
        keep = np.sort(rng.choice(keep, CAP, replace=False))
    cs = c.loc[keep]
    sub = rv[rv.card_id.isin(keep) & (rv.i < max(KS))]
    sctx = ctx.loc[sub.index]
    ci = pd.Index(keep).get_indexer(sub.card_id)
    ii = sub.i.to_numpy()
    K = max(KS)
    A = {k: np.zeros((len(keep), K)) for k in ("rating", "ed", "es", "ld", "day")}
    for k, v in (("rating", sub.rating.to_numpy()), ("ed", np.log1p(np.maximum(sub.elapsed_days.to_numpy(), 0))),
                 ("es", np.log1p(np.maximum(sub.elapsed_seconds.to_numpy(), 0))),
                 ("ld", ld[sub.index.to_numpy()] - uld[sub.index.to_numpy()]),
                 ("day", (sub.day_offset.to_numpy() - cs.day0.to_numpy()[ci]))):
        A[k][ci, ii] = v
    C = {k: np.zeros((len(keep), K)) for k in ctx.columns}
    for k in ctx.columns:
        C[k][ci, ii] = sctx[k].to_numpy()
    nrev = np.minimum(cs.total.to_numpy(), K)
    res["flag"] = {"user": u, "worst10": cs.worst10.to_numpy().astype(np.int8).tolist(),
                   "lapseN": cs.lapseN.to_numpy().astype(np.int8).tolist(),
                   "worst10_180": cs.worst10_180.to_numpy().astype(np.int8).tolist(), "r180": cs.r180.astype(int).tolist(),
                   "total": cs.total.astype(int).tolist(),
                   "nrev": nrev.astype(int).tolist(), "nsize": np.log1p(cs.nsize.to_numpy()).tolist(),
                   "A": {k: np.round(v, 4).tolist() for k, v in A.items()},
                   "C": {k: np.round(v, 4).tolist() for k, v in C.items()},
                   "all_total": float(tot.sum()), "all_cards": len(c)}

    # ---------- 2. missed days ----------
    days = np.unique(rv.day_offset.to_numpy())
    idle_before = np.r_[0, np.diff(days) - 1]          # idle days right before each active day
    ib = pd.Series(idle_before, index=days)
    act3 = pd.Series(np.r_[np.zeros(3, bool), (days[3:] - days[:-3]) == 3] if len(days) > 3 else np.zeros(len(days), bool),
                     index=days)
    prevday_active = np.r_[-1, days[:-1]]
    prev_active = pd.Series(prevday_active, index=days)
    rv["idle"] = ib.reindex(rv.day_offset).to_numpy()
    # user's active days in the 180 days after each day (future-activity confound check)
    act = pd.Series(np.searchsorted(days, days + FOLLOW, "right") - np.arange(len(days)) - 1, index=days)
    rv["act180"] = act.reindex(rv.day_offset).to_numpy().astype(float)
    rv["act3"] = act3.reindex(rv.day_offset).to_numpy()
    rv["prev_active_day"] = prev_active.reindex(rv.day_offset).to_numpy()
    g = rv.groupby("card_id")
    rv["prev_ed"] = g.elapsed_days.shift(1)
    rv["prev_rating"] = g.rating.shift(1)
    rv["prev_day"] = g.day_offset.shift(1)
    rv["next_ed"] = g.elapsed_days.shift(-1)
    rv["next_pass"] = g["pass"].shift(-1)
    # reviews / lapses of this card in the 180 days after this review (strictly after this row)
    dd = rv.day_offset.to_numpy()
    cnum = pd.factorize(rv.card_id)[0].astype(np.int64)
    o = np.lexsort((np.arange(n), cnum))     # rows grouped by card, chronological
    cn, dn = cnum[o], dd[o]
    key = cn.astype(np.float64) * 1e6 + dn
    hi = np.searchsorted(key, cn.astype(np.float64) * 1e6 + dn + FOLLOW, "left")
    posn = np.arange(n)
    cumlap = np.r_[0, np.cumsum(rv.lapse.to_numpy()[o])]
    rev180 = np.empty(n)
    lap180 = np.empty(n)
    rev180[o] = hi - posn - 1
    lap180[o] = cumlap[hi] - cumlap[posn + 1]
    rv["rev180"], rv["lap180"] = rev180, lap180
    m = (rv.elapsed_days >= 1) & rv.prev_ed.notna() & (rv.prev_ed >= 1) & (dd <= last - FOLLOW)
    late = m & (rv.idle >= 2) & (rv.prev_day < rv.prev_active_day)     # interval spans the gap, first day back
    ontime = m & rv.act3 & (rv.idle == 0)
    t = rv[late | ontime].copy()
    if len(t) > 50:
        gb = np.full(len(t), -1)
        for k, (lo, hi_, _) in enumerate(GAP_B):
            if k == 0:
                gb[(t.idle == 0).to_numpy()] = 0
            else:
                gb[((t.idle >= lo) & (t.idle <= hi_)).to_numpy()] = k
        t["gb"] = gb
        t = t[t.gb >= 0]
        t["peb"] = np.floor(np.log2(t.prev_ed)).astype(int)
        t["ib"] = np.minimum(t.i, 10)
        t["log_ed"] = np.log(t.elapsed_days)
        t["log_next"] = np.log(np.maximum(t.next_ed, 1))
        t.loc[t.next_ed.isna(), "log_next"] = np.nan
        t["next_growth"] = t.log_next - t.log_ed
        strat = ["peb", "prev_rating", "ib"]
        t = mixed(t, strat, "gb")
        if len(t) > 20:
            t["rev_per_act"] = t.rev180 / np.maximum(t.act180, 1) * 100
            O = ["log_ed", "pass", "rev180", "lap180", "next_pass", "log_next", "next_growth", "act180", "rev_per_act"]
            res["T_all"] = grp(t, strat, "gb", 4, O)
            tp = t[t["pass"] == 1]
            res["T_pass"] = grp(tp, strat, "gb", 4, O)
            tf = t[t["pass"] == 0]
            res["T_fail"] = grp(tf, strat, "gb", 4, O)

    # ---------- 3. early spacing of new cards ----------
    r2 = rv[rv.card_id.isin(c.index)]
    fr = r2[r2.i == 0].set_index("card_id").rating
    sec = r2[r2.i == 1].set_index("card_id")
    sameday = r2[(r2.i > 0) & (r2.day_offset == r2.card_id.map(c.day0))].groupby("card_id").size()
    s = c.loc[c.index.isin(sec.index)].copy()
    s["extra"] = sameday.reindex(s.index).fillna(0).to_numpy()
    s["sp"] = np.minimum(s.extra, 2).astype(int)
    s["r1"] = fr.reindex(s.index).to_numpy()
    s["step_s"] = sec.elapsed_seconds.reindex(s.index).to_numpy()
    x = r2[(r2.i > 0) & (r2.elapsed_days >= 1)]
    fx = x.groupby("card_id").first()
    s["pass_first_xd"] = fx["pass"].reindex(s.index).to_numpy()
    w = r2[(r2.day_offset - r2.card_id.map(c.day0)) < FOLLOW]
    s["rev180"] = w.groupby("card_id").size().reindex(s.index).fillna(0).to_numpy()
    s["xd180"] = (w.elapsed_days >= 1).groupby(w.card_id).sum().reindex(s.index).fillna(0).to_numpy()
    s["lap180"] = w.groupby("card_id").lapse.sum().reindex(s.index).fillna(0).to_numpy()
    s["reach21"] = ((w.rating > 1) & (w.elapsed_days >= 21)).groupby(w.card_id).any().reindex(s.index) \
        .fillna(False).to_numpy().astype(float)
    s["leech"] = (s.lapses_all >= 8).astype(float)
    O3 = ["pass_first_xd", "lap180", "reach21", "rev180", "xd180", "leech"]
    res["S_n"] = np.bincount(s.sp, minlength=3).tolist()
    for name, strat, ss in (("S_deck", ["deck"], s), ("S_preset", ["preset"], s),
                            ("S_deck_r1", ["deck", "r1"], s), ("S_deck_good", ["deck"], s[s.r1 == 3])):
        mm = mixed(ss, strat, "sp")
        if len(mm) > 20:
            res[name] = grp(mm, strat, "sp", 3, O3)
    # within deck x first rating, cards whose 2nd exposure slipped to the next day although the deck usually
    # shows it the same day (session ended before the learning step was due)
    g3 = s[s.r1 == 3].copy()
    if len(g3) > 20:
        share = g3.groupby("deck").sp.transform(lambda v: (v > 0).mean())
        hv = g3[(share >= 0.7).to_numpy()].copy()
        hv["sp"] = np.minimum(hv.sp, 1)
        mm = mixed(hv, ["deck"], "sp")
        if len(mm) > 20:
            res["S_slip"] = grp(mm, ["deck"], "sp", 3, O3)
    return res


def run_one(u):
    try:
        return one(u)
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"user": u, "error": traceback.format_exc()[-600:]}


# ======================= aggregation =======================
def summarize(rows, key, names, outs, B=1000):
    rows = [r for r in rows if key in r]
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


FEATS = {}


def build(fl, k, groups=None):
    """feature matrix after the card's k-th review; durations of reviews 1..k-1 only (review k's duration = leak)."""
    A, C = fl["A"], fl["C"]
    cols, names = [], []

    def add(name, v, grp_):
        if groups is None or grp_ in groups:
            cols.append(np.asarray(v, float))
            names.append(name)
    for j in range(k):
        add(f"rating{j + 1}", np.asarray(A["rating"])[:, j], "ratings")
    R = np.asarray(A["rating"])[:, :k]
    add("n_again", (R == 1).sum(1), "ratings")
    add("n_hard", (R == 2).sum(1), "ratings")
    add("n_easy", (R == 4).sum(1), "ratings")
    for j in range(1, k):
        add(f"log_ed{j + 1}", np.asarray(A["ed"])[:, j], "intervals")
        add(f"log_es{j + 1}", np.asarray(A["es"])[:, j], "intervals")
    add("days_since_intro", np.asarray(A["day"])[:, k - 1], "intervals")
    for j in range(k - 1):
        add(f"rel_logdur{j + 1}", np.asarray(A["ld"])[:, j], "durations")
    Ck = {kk: np.asarray(v)[:, k - 1] for kk, v in C.items()}
    add("nsize", fl["nsize"], "siblings")
    for kk in ("s_n", "s_pass", "s_again"):
        add(kk, Ck[kk], "siblings")
    for kk in ("d_pass", "d_log_n"):
        add(kk, Ck[kk], "deck")
    for kk in ("u_pass", "u_log_n", "u_logdur", "new_today"):
        add(kk, Ck[kk], "user")
    C0 = {kk: np.asarray(v)[:, 0] for kk, v in C.items()}
    add("new_today_at_intro", C0["new_today"], "user")
    return np.column_stack(cols), names


def flag_eval(score, users, y, total, nrev_rem, B=300, rates=(5, 10), seed=0):
    """per-user flag of the top r% of cards by score; precision, recall, prize share (all reviews / reviews after k)."""
    from sklearn.metrics import roc_auc_score
    uu, inv = np.unique(users, return_inverse=True)
    flagged = {}
    for r in rates:
        f = np.zeros(len(score), bool)
        for j in range(len(uu)):
            ix = np.where(inv == j)[0]
            m = max(1, int(round(len(ix) * r / 100)))
            f[ix[np.argsort(-score[ix], kind="stable")[:m]]] = True
        flagged[r] = f
    per = []
    for j in range(len(uu)):
        ix = inv == j
        d = {"n": ix.sum(), "y": y[ix].sum(), "T": total[ix].sum(), "Trem": nrev_rem[ix].sum()}
        for r in rates:
            f = flagged[r][ix]
            d[f"f{r}"] = f.sum()
            d[f"tp{r}"] = (f & (y[ix] == 1)).sum()
            d[f"tot{r}"] = total[ix][f].sum()
            d[f"rem{r}"] = nrev_rem[ix][f].sum()
        per.append(d)
    P = {kk: np.array([d[kk] for d in per], float) for kk in per[0]}

    def stats(I):
        o = {}
        for r in rates:
            o[f"precision@{r}%"] = P[f"tp{r}"][I].sum() / P[f"f{r}"][I].sum()
            o[f"recall@{r}%"] = P[f"tp{r}"][I].sum() / P["y"][I].sum()
            o[f"prize_all@{r}%"] = P[f"tot{r}"][I].sum() / P["T"][I].sum()
            o[f"prize_after_k@{r}%"] = P[f"rem{r}"][I].sum() / P["Trem"][I].sum()
        return o
    point = stats(np.arange(len(uu)))
    point["auc"] = roc_auc_score(y, score)
    rng = np.random.default_rng(seed)
    bs = {kk: [] for kk in point}
    rowsof = [np.where(inv == j)[0] for j in range(len(uu))]
    for b in range(B):
        I = rng.integers(0, len(uu), len(uu))
        st = stats(I)
        if b < 50:
            ix = np.concatenate([rowsof[j] for j in I])
            st["auc"] = roc_auc_score(y[ix], score[ix])
        for kk, v in st.items():
            bs[kk].append(v)
    return {kk: {"v": float(v), "ci": np.percentile(bs[kk], [2.5, 97.5]).tolist()} for kk, v in point.items()}


def flag_analysis(rows, procs):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    fls = [r["flag"] for r in rows if "flag" in r]
    test_u = {f["user"] for f in fls if (f["user"] * 2654435761) % 1000 < 500}
    out = {"cards_per_user_cap": CAP, "lapse_label": f">= {LAPSE_N} cross-day lapses ever",
           "train_users": len(fls) - len(test_u), "test_users": len(test_u)}
    U = np.concatenate([[f["user"]] * len(f["total"]) for f in fls])
    te = np.isin(U, list(test_u))
    Y = {"worst10": np.concatenate([f["worst10"] for f in fls]), "lapseN": np.concatenate([f["lapseN"] for f in fls]),
         "worst10_180": np.concatenate([f["worst10_180"] for f in fls])}
    r180 = np.concatenate([f["r180"] for f in fls]).astype(float)
    total = np.concatenate([f["total"] for f in fls]).astype(float)
    nrev = np.concatenate([f["nrev"] for f in fls])
    out["base_rate"] = {kk: float(v.mean()) for kk, v in Y.items()}
    out["oracle_prize_10pct_sample"] = None
    feats = {}
    for k in KS:
        Xs, names = zip(*[build(f, k) for f in fls])
        feats[k] = (np.vstack(Xs), names[0])
    res = {}
    for lab in ("worst10", "lapseN", "worst10_180"):
        y = Y[lab]
        res[lab] = {}
        Tw = r180 if lab == "worst10_180" else total   # prize measured on the label's own review count
        for k in KS:
            X, names = feats[k]
            ok = nrev >= k
            tr = ~te & ok
            rem = np.maximum(Tw - k, 0)
            R = {}
            for mname, mdl in (("logistic", make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=1.0))),
                               ("boosting", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                                                           max_leaf_nodes=31, random_state=0))):
                mdl.fit(X[tr], y[tr])
                sc = np.full(len(y), -1e9)
                sc[ok] = mdl.predict_proba(X[ok])[:, 1]
                R[mname] = flag_eval(sc[te], U[te], y[te], Tw[te], rem[te])
                if mname == "boosting":
                    best = mdl
            # simple rule: Again count in the first k reviews, ties broken by lower mean rating
            Rr = np.asarray(feats[k][0][:, names.index("n_again")]) - 0.01 * X[:, :k].mean(1)
            sc = np.where(ok, Rr, -1e9)
            R["rule_again_count"] = flag_eval(sc[te], U[te], y[te], Tw[te], rem[te], B=200)
            R["cards_with_k_reviews_share"] = float(ok.mean())
            res[lab][f"after_{k}"] = R
            print(lab, k, {m: (round(R[m]["auc"]["v"], 3), round(R[m]["prize_all@10%"]["v"], 3))
                           for m in ("logistic", "boosting", "rule_again_count")}, flush=True)
    # oracle (perfect flag of the worst 10% by total reviews) on the same sampled cards
    sc = total[te] + np.random.default_rng(5).random(te.sum())
    res["oracle_by_total"] = flag_eval(sc, U[te], Y["worst10"][te], total[te], np.maximum(total[te] - 3, 0), B=100)
    sc = r180[te] + np.random.default_rng(6).random(te.sum())
    res["oracle_by_r180"] = flag_eval(sc, U[te], Y["worst10_180"][te], r180[te], np.maximum(r180[te] - 3, 0), B=100)
    # ablation at k = 3 (boosting), worst10 label: drop each signal group / only that group
    abl = {}
    k = 3
    ok = nrev >= k
    for mode in ("drop", "only"):
        for grp_ in ("ratings", "intervals", "durations", "siblings", "deck", "user"):
            allg = {"ratings", "intervals", "durations", "siblings", "deck", "user"}
            gs = allg - {grp_} if mode == "drop" else {grp_}
            Xs = np.vstack([build(f, k, gs)[0] for f in fls])
            for lab in ("worst10",):
                y = Y[lab]
                mdl = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(Xs[~te & ok], y[~te & ok])
                sc = np.full(len(y), -1e9)
                sc[ok] = mdl.predict_proba(Xs[ok])[:, 1]
                e = flag_eval(sc[te], U[te], y[te], total[te], np.maximum(total[te] - k, 0), B=100)
                abl[f"{mode}_{grp_}"] = {m: e[m] for m in ("auc", "prize_all@10%", "precision@10%")}
                print("abl", mode, grp_, round(e["auc"]["v"], 3), round(e["prize_all@10%"]["v"], 3), flush=True)
    res["ablation_k3_boosting_worst10"] = abl
    out.update(res)
    return out


def main():
    a = sys.argv[1:]
    g = lambda k, d: a[a.index(k) + 1] if k in a else d
    nu, procs, out = int(g("--users", 1000)), int(g("--procs", 8)), g("--out", "part2.json")
    hh, mm = map(int, g("--deadline", "22:40").split(":"))
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
    print(f"{len(rows)} users ok, {len(errs)} errors {errs[:2]}, {time.time() - t0:.0f}s", flush=True)
    S = {"sample": {"pool": "8k/2k run train_users inside 5000-10000 (random.Random(79).sample)", "requested": nu,
                    "users": len(rows), "cards": int(sum(r["cards"] for r in rows)),
                    "reviews": int(sum(r["reviews"] for r in rows)), "errors": len(errs)}}
    GN = [b[2] for b in GAP_B]
    O = ["log_ed", "pass", "rev180", "lap180", "next_pass", "log_next", "next_growth", "act180", "rev_per_act"]
    for kk in ("T_all", "T_pass", "T_fail"):
        S[kk] = summarize(rows, kk, GN, O)
    O3 = ["pass_first_xd", "lap180", "reach21", "rev180", "xd180", "leech"]
    S["S_n"] = np.sum([r["S_n"] for r in rows], 0).tolist()
    for kk in ("S_deck", "S_preset", "S_deck_r1", "S_deck_good", "S_slip"):
        S[kk] = summarize(rows, kk, SP_B if kk != "S_slip" else ["slipped to next day", "same day as the deck usually does", "-"], O3)
    S["S_slip"]["rows"] = S["S_slip"]["rows"][:2]
    if "--skip-flag" in a:   # refresh items 2 and 3 only, keep the flag analysis already in the output file
        S["flag"] = json.load(open(out))["flag"]
    json.dump(S, open(out, "w"), indent=1, default=float)
    print("wrote 2+3", time.time() - t0, flush=True)
    if "--skip-flag" in a:
        return
    S["flag"] = flag_analysis(rows, procs)
    json.dump(S, open(out, "w"), indent=1, default=float)
    print("wrote", out, time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
