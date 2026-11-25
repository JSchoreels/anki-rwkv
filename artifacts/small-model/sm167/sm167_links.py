#!/usr/bin/env python
"""Links between cards (SM167): is there card-to-card signal left in the residuals of our models, beyond same note,
same deck and same-session user state? CPU only, reads the SM160 prediction dumps and the dataset read-only.
Pre-registered design: design.md (written before any result).

  python sm167_links.py features --pool wfit|e732 [--workers 4] [--users 1007,5022]
        per-user feature arrays -> work/sm167/<pool>/<user>.npz (not committed),
        per-user aggregate rows -> rows-<pool>.jsonl (sums only, no card ids, no raw rows)
  python sm167_links.py report     -> sm167-result.{json,md}
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WT = HERE.parents[2]
MAIN = Path("C:/Users/jonat/workspace/anki-rwkv")
DATA = MAIN / "data" / "anki-revlogs-10k-original-20261011"
DUMP = Path("C:/Users/jonat/workspace/wt/sm160/work/sm160-dump")
WORK = WT / "work" / "sm167"
LABEL = "w8+lr2.16.4q8x8@e2s-query"   # T24, show-time
MODELS = {   # key: (dump tag, plain name)
    "small_io": ("sm113-inst-e35st-st-dec2ep32k-t99", "Small Instant-Only (SM113)"),
    "small_th": ("sm118-e26-dec2ep32k-t91s2", "Small two-head (SM118), instant head"),
    "big_io": ("sm99-big-inst-e2", "Big Instant-Only (SM99)"),
}
MAIN_MODEL = "small_io"
W_SESSION = 50       # window (review positions, same day) for co-failure links and session baselines
W_ORDER = 10         # window for confusable-order links
MAX_DAY_FAILS = 150  # days with more failed cards add no co-failure pairs
LINKS = ("a", "a1", "b")
FEATS0 = ("f_sess", "f_deck")
FEATS1 = FEATS0 + ("f_a", "i_a", "f_b", "i_b")
CLIP_LOGIT = 1e-7
DRAWS = 2000
MIN_GAIN = 0.05


def dump_file(pool, user, tag):
    return DUMP / pool / f"{user}_{tag}_{LABEL}.npz"


def pool_users(pool):
    tag = MODELS[MAIN_MODEL][0]
    return sorted(int(f.name.split("_")[0]) for f in (DUMP / pool).glob(f"*_{tag}_{LABEL}.npz"))


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), CLIP_LOGIT, 1 - CLIP_LOGIT)
    return np.log(p) - np.log1p(-p)


def loss_vec(y, p):
    eps = np.finfo(np.float64).eps
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def count_before(sorted_enc, key, dm, day):
    """Number of encoded events key * dm + e_day with e_day < day."""
    base = key * dm
    return np.searchsorted(sorted_enc, base + day, "left") - np.searchsorted(sorted_enc, base, "left")


def load_user(user):
    import pandas as pd
    r = pd.read_parquet(DATA / "revlogs" / f"user_id={user}", columns=["card_id", "day_offset", "rating"])
    r = r.reset_index(drop=True)
    cdir = DATA / "cards" / f"user_id={user}"
    if cdir.exists():
        cards = pd.read_parquet(cdir, columns=["card_id", "note_id", "deck_id"])
    else:   # no card metadata: every card its own note, one shared deck placeholder (as the scorer's loader)
        cards = pd.DataFrame({"card_id": pd.Series([], dtype="int64"), "note_id": pd.Series([], dtype="float64"),
                              "deck_id": pd.Series([], dtype="float64")})
    m = r[["card_id"]].merge(cards, on="card_id", how="left", validate="many_to_one")
    assert len(m) == len(r)
    card = pd.factorize(r["card_id"])[0].astype(np.int64)
    note = m["note_id"].to_numpy(dtype=np.float64)
    nan = np.isnan(note)
    note = np.where(nan, -1.0 - card, note).astype(np.int64)        # missing note: one per card
    deck = m["deck_id"].to_numpy(dtype=np.float64)
    deck = np.where(np.isnan(deck), -1.0, deck).astype(np.int64)    # missing deck: one shared placeholder
    day = r["day_offset"].to_numpy(dtype=np.int64)
    assert np.all(np.diff(day) >= 0)
    day = day - day.min()
    rating = r["rating"].to_numpy()
    first = r.groupby("card_id").cumcount().to_numpy() == 0
    return card, note, deck, day, rating, first


def user_features(pool, user):
    card, note, deck, day, rating, first = load_user(user)
    n = len(card)
    nc = int(card.max()) + 1
    dm = int(day.max()) + 2
    fail = (rating == 1) & ~first

    # --- per-model predictions on the full row index (NaN = not a benchmark review) ---
    preds, base_th = {}, None
    for key, (tag, _) in MODELS.items():
        f = dump_file(pool, user, tag)
        if not f.exists():
            continue
        z = np.load(f)
        th = z["review_th"].astype(np.int64)
        if base_th is None:
            if key != MAIN_MODEL:
                return None
            base_th, y = th, np.asarray(z["y"], dtype=np.float64)
            rows = th - 1
            assert np.array_equal((rating[rows] > 1).astype(np.float64), y), user
        elif not np.array_equal(th, base_th):
            continue
        p = np.full(n, np.nan)
        p[th - 1] = z["p_instant"]
        preds[key] = p
    if base_th is None:
        return None
    T = base_th - 1
    nt = len(T)
    has = ~np.isnan(preds[MAIN_MODEL])

    # --- link (a): co-failure days (undirected), learned from days strictly before ---
    fk = np.unique(card[fail] * dm + day[fail])        # unique (card, day) failures, sorted by card then day
    f_card, f_day = fk // dm, fk % dm
    order = np.argsort(f_day, kind="stable")
    dcard, dday = f_card[order], f_day[order]
    bounds = np.flatnonzero(np.diff(dday)) + 1
    pair_chunks = []
    for grp in np.split(np.arange(len(dday)), bounds):
        k = len(grp)
        if k < 2 or k > MAX_DAY_FAILS:
            continue
        cs = dcard[grp]
        i, j = np.triu_indices(k, 1)
        lo, hi = np.minimum(cs[i], cs[j]), np.maximum(cs[i], cs[j])
        pair_chunks.append((lo * nc + hi) * dm + dday[grp[0]])
    enc_a = np.sort(np.concatenate(pair_chunks)) if pair_chunks else np.zeros(0, np.int64)
    fail_days = np.unique(f_day)

    # --- link (b): confusable order events A at s-o -> X at s (directed), learned from days strictly before ---
    ev_all, ev_fail = [], []
    for o in range(1, W_ORDER + 1):
        s = np.arange(o, n)
        j = s - o
        ok = (day[j] == day[s]) & (card[j] != card[s]) & (note[j] != note[s])
        s, j = s[ok], j[ok]
        e = (card[j] * nc + card[s]) * dm + day[s]
        ev_all.append(e)
        ev_fail.append(e[fail[s]])
    enc_bn = np.sort(np.concatenate(ev_all))
    enc_bk = np.sort(np.concatenate(ev_fail))
    nf = ~first
    enc_rev = np.sort(card[nf] * dm + day[nf])          # non-first reviews per card, with multiplicity
    enc_fail = np.sort(card[fail] * dm + day[fail])

    # --- queries: target t, window card j (same day, earlier, benchmark, other card, other note) ---
    qt, qj, qo = [], [], []
    for o in range(1, W_SESSION + 1):
        j = T - o
        ok = j >= 0
        idx = np.flatnonzero(ok)
        jj = j[ok]
        tt = T[ok]
        ok2 = (day[jj] == day[tt]) & (card[jj] != card[tt]) & (note[jj] != note[tt]) & has[jj]
        qt.append(idx[ok2])
        qj.append(jj[ok2])
        qo.append(np.full(int(ok2.sum()), o, np.int64))
    qt, qj, qo = np.concatenate(qt), np.concatenate(qj), np.concatenate(qo)
    tr = T[qt]
    dt = day[tr]
    ca, cx = card[qj], card[tr]
    same_deck = deck[qj] == deck[tr]

    key_a = np.minimum(ca, cx) * nc + np.maximum(ca, cx)
    C = count_before(enc_a, key_a, dm, dt)
    FA = count_before(fk, ca, dm, dt)
    FX = count_before(fk, cx, dm, dt)
    D = np.searchsorted(fail_days, dt, "left")
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.where(C > 0, C * D / np.maximum(FA * FX, 1), 0.0)
    link = {"a": (C >= 2) & (lift >= 2), "a1": C >= 1}
    inb = qo <= W_ORDER
    key_b = ca * nc + cx
    NB = np.where(inb, count_before(enc_bn, key_b, dm, dt), 0)
    KB = np.where(inb, count_before(enc_bk, key_b, dm, dt), 0)
    fx = (count_before(enc_fail, cx, dm, dt) + 1) / (count_before(enc_rev, cx, dm, dt) + 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        link["b"] = inb & (KB >= 2) & (KB / np.maximum(NB, 1) >= 2 * fx)

    # matched controls: same (target, same-deck flag), not linked, nearest offset (ties: larger offset = earlier)
    ctrl = {}
    for L in LINKS:
        scope = inb if L == "b" else np.ones(len(qt), bool)
        g = (qt * 2 + same_deck) * 64 + qo
        cand = np.flatnonzero(scope & ~link[L])
        cand = cand[np.argsort(g[cand], kind="stable")]
        gc = g[cand]
        lk = np.flatnonzero(link[L])
        pos = np.searchsorted(gc, g[lk])
        grp = g[lk] // 64
        best = np.full(len(lk), -1)
        bestd = np.full(len(lk), 10 ** 9)
        for side in (pos, pos - 1):        # right (larger offset) first so it wins ties
            okk = (side >= 0) & (side < len(cand))
            sidx = np.where(okk, side, 0)
            okk &= (gc[sidx] // 64) == grp
            d = np.abs(qo[cand[sidx]] - qo[lk])
            better = okk & (d < bestd)
            best = np.where(better, cand[sidx], best)
            bestd = np.where(better, d, bestd)
        ctrl[L] = (lk, best)

    out = {"user": user, "n_rows": n, "n_targets": nt, "models": {}}
    arrays = {"y": y.astype(np.int8)}
    for L in LINKS:
        anyl = np.bincount(qt[link[L]], minlength=nt) > 0
        lk, best = ctrl[L]
        kept = np.bincount(qt[lk[best >= 0]], minlength=nt) > 0
        out[f"cov_{L}"] = int(anyl.sum())
        out[f"cov_{L}_matched"] = int(kept.sum())
        out[f"links_{L}_per_covered"] = float(link[L].sum() / max(anyl.sum(), 1))
    arrays["i_a"] = (np.bincount(qt[link["a"]], minlength=nt) > 0).astype(np.float32)
    arrays["i_b"] = (np.bincount(qt[link["b"]], minlength=nt) > 0).astype(np.float32)

    def mean_by(mask, vals):
        cnt = np.bincount(qt[mask], minlength=nt)
        s = np.bincount(qt[mask], weights=vals[mask], minlength=nt)
        return np.where(cnt > 0, s / np.maximum(cnt, 1), 0.0)

    for key, p in preds.items():
        res_full = np.zeros(n)
        okp = ~np.isnan(p)
        res_full[okp] = (rating[okp] > 1) - p[okp]
        pt = p[T]
        e = y - pt
        rq = res_full[qj]
        if key != MAIN_MODEL:
            # the window set uses the main model's benchmark rows; identical review sets checked above
            assert np.all(~np.isnan(p[qj]))
        arrays[f"lp_{key}"] = logit(pt).astype(np.float32)
        arrays[f"f_sess_{key}"] = mean_by(np.ones(len(qt), bool), rq).astype(np.float32)
        arrays[f"f_deck_{key}"] = mean_by(same_deck, rq).astype(np.float32)
        arrays[f"f_a_{key}"] = mean_by(link["a"], rq).astype(np.float32)
        arrays[f"f_b_{key}"] = mean_by(link["b"], rq).astype(np.float32)
        m = {"n": nt, "loss_raw": float(loss_vec(y, pt).sum())}
        for L in LINKS:
            lk, best = ctrl[L]
            okb = best >= 0
            lk, best = lk[okb], best[okb]
            t_ids = qt[lk]
            cnt = np.bincount(t_ids, minlength=nt)
            sel = cnt > 0
            rl = np.bincount(t_ids, weights=rq[lk], minlength=nt)[sel] / cnt[sel]
            rc = np.bincount(t_ids, weights=rq[best], minlength=nt)[sel] / cnt[sel]
            ex = e[sel]
            m[L] = {"n": int(sel.sum()), "se": float(ex.sum()), "see": float((ex * ex).sum()),
                    "sl": float(rl.sum()), "sll": float((rl * rl).sum()), "sel": float((ex * rl).sum()),
                    "sc": float(rc.sum()), "scc": float((rc * rc).sum()), "sec": float((ex * rc).sum())}
        out["models"][key] = m
    return out, arrays


def worker(args):
    pool, user = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.time()
    try:
        res = user_features(pool, user)
    except Exception as exc:   # noqa: BLE001 - report and continue
        return {"user": user, "error": repr(exc)}
    if res is None:
        return {"user": user, "error": "no main-model dump"}
    row, arrays = res
    (WORK / pool).mkdir(parents=True, exist_ok=True)
    np.savez(WORK / pool / f"{user}.npz", **arrays)
    row["sec"] = round(time.time() - t0, 2)
    return row


def cmd_features(a):
    users = [int(u) for u in a.users.split(",")] if a.users else pool_users(a.pool)
    out = HERE / f"rows-{a.pool}.jsonl" if not a.users else WORK / f"rows-{a.pool}-test.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists() and not a.users:
        done = {json.loads(l)["user"] for l in open(out) if l.strip() and "error" not in json.loads(l)}
    todo = [(a.pool, u) for u in users if u not in done]
    print(f"[sm167] {a.pool}: {len(users)} users, {len(todo)} to do", flush=True)
    t0 = time.time()
    with open(out, "a") as fh:
        if a.workers > 1:
            from multiprocessing import Pool
            with Pool(a.workers) as pl:
                it = pl.imap_unordered(worker, todo)
                for k, row in enumerate(it, 1):
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    if k % 25 == 0 or "error" in row:
                        print(f"[sm167] {k}/{len(todo)} {time.time() - t0:.0f}s {row.get('error', '')}", flush=True)
        else:
            for k, t in enumerate(todo, 1):
                row = worker(t)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(f"[sm167] {k}/{len(todo)} user {t[1]} {row.get('sec')}s {row.get('error', '')}", flush=True)
    print(f"[sm167] done {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ report

def read_rows(pool):
    rows = {}
    for l in open(HERE / f"rows-{pool}.jsonl"):
        r = json.loads(l)
        if "error" not in r:
            rows[r["user"]] = r
    return rows


def load_arrays(pool, users, key, feats):
    ys, lps, Xs, uid = [], [], [], []
    for k, u in enumerate(users):
        z = np.load(WORK / pool / f"{u}.npz")
        if f"lp_{key}" not in z.files:
            continue
        ys.append(z["y"].astype(np.float64))
        lps.append(z[f"lp_{key}"].astype(np.float64))
        Xs.append(np.stack([z[f if f.startswith("i_") else f"{f}_{key}"] for f in feats], 1).astype(np.float64))
        uid.append(np.full(len(z["y"]), k, np.int32))
    return np.concatenate(ys), np.concatenate(lps), np.concatenate(Xs), np.concatenate(uid)


def fit_newton(y, z0, X, iters=30):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(z0 + X @ b)))
        g = X.T @ (p - y)
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        b -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return b


def per_user_loss(y, z, uid, nu):
    return np.bincount(uid, weights=loss_vec(y, 1 / (1 + np.exp(-z))), minlength=nu)


def boot_ratio(num, den, draws=DRAWS, seed=0):
    rng = np.random.default_rng(seed)
    nu = len(num)
    idx = rng.integers(0, nu, size=(draws, nu))
    vals = 100 * (num[idx].sum(1) / den[idx].sum(1) - 1)
    return float(100 * (num.sum() / den.sum() - 1)), [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def corr_from(S):
    n = S["n"]
    cov_l = S["sel"] - S["se"] * S["sl"] / n
    cov_c = S["sec"] - S["se"] * S["sc"] / n
    ve = S["see"] - S["se"] ** 2 / n
    vl = S["sll"] - S["sl"] ** 2 / n
    vc = S["scc"] - S["sc"] ** 2 / n
    rl = cov_l / np.sqrt(ve * vl)
    rc = cov_c / np.sqrt(ve * vc)
    return rl, rc, cov_l / vl, cov_c / vc


def corr_table(rows, key):
    users = [u for u in rows if key in rows[u]["models"]]
    out = {}
    for L in LINKS:
        arr = {f: np.array([rows[u]["models"][key][L][f] for u in users], dtype=np.float64)
               for f in ("n", "se", "see", "sl", "sll", "sel", "sc", "scc", "sec")}
        tot = {f: v.sum() for f, v in arr.items()}
        rl, rc, bl, bc = corr_from(tot)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(users), size=(DRAWS, len(users)))
        bs = {f: v[idx].sum(1) for f, v in arr.items()}
        brl, brc, bbl, bbc = corr_from(bs)
        d = brl - brc
        out[L] = {"users": len(users), "pairs_targets": int(tot["n"]), "corr_link": float(rl), "corr_ctrl": float(rc),
                  "diff": float(rl - rc), "diff_ci": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))],
                  "corr_link_ci": [float(np.nanpercentile(brl, 2.5)), float(np.nanpercentile(brl, 97.5))],
                  "corr_ctrl_ci": [float(np.nanpercentile(brc, 2.5)), float(np.nanpercentile(brc, 97.5))],
                  "slope_link": float(bl), "slope_ctrl": float(bc)}
    return out


def cmd_report(a):
    rows_w, rows_e = read_rows("wfit"), read_rows("e732")
    result = {"design": "design.md", "label": LABEL, "models": {}}
    tot_t = sum(r["n_targets"] for r in rows_e.values())
    result["coverage_e732"] = {L: {"any": sum(r[f"cov_{L}"] for r in rows_e.values()) / tot_t,
                                   "matched": sum(r[f"cov_{L}_matched"] for r in rows_e.values()) / tot_t,
                                   "links_per_covered": float(np.average(
                                       [r[f"links_{L}_per_covered"] for r in rows_e.values()],
                                       weights=[max(r[f"cov_{L}"], 1) for r in rows_e.values()]))}
                               for L in LINKS}
    result["coverage_e732"]["targets"] = tot_t
    result["users"] = {"wfit": len(rows_w), "e732": len(rows_e)}
    for key in MODELS:
        uw = sorted(u for u in rows_w if key in rows_w[u]["models"])
        ue = sorted(u for u in rows_e if key in rows_e[u]["models"])
        if len(uw) < 90 or len(ue) < 700:
            result["models"][key] = {"skipped": f"users wfit {len(uw)}, e732 {len(ue)}"}
            continue
        y, z0, X, _ = load_arrays("wfit", uw, key, FEATS1)
        b0 = fit_newton(y, z0, X[:, :2])
        b1 = fit_newton(y, z0, X)
        del y, z0, X
        y, z0, X, uid = load_arrays("e732", ue, key, FEATS1)
        nu = len(ue)
        L_raw = per_user_loss(y, z0, uid, nu)
        L0 = per_user_loss(y, z0 + X[:, :2] @ b0, uid, nu)
        L1 = per_user_loss(y, z0 + X @ b1, uid, nu)
        bo0 = fit_newton(y, z0, X[:, :2])
        bo1 = fit_newton(y, z0, X)
        Lo0 = per_user_loss(y, z0 + X[:, :2] @ bo0, uid, nu)
        Lo1 = per_user_loss(y, z0 + X @ bo1, uid, nu)
        del y, z0, X, uid
        g_link, ci_link = boot_ratio(L1, L0)
        g_sess, ci_sess = boot_ratio(L0, L_raw)
        g_all, ci_all = boot_ratio(L1, L_raw)
        g_or, ci_or = boot_ratio(Lo1, Lo0)
        result["models"][key] = {
            "name": MODELS[key][1], "users_wfit": len(uw), "users_e732": nu,
            "beta_M0": dict(zip(FEATS0, map(float, b0))), "beta_M1": dict(zip(FEATS1, map(float, b1))),
            "beta_M1_oracle": dict(zip(FEATS1, map(float, bo1))),
            "logloss_raw": float(L_raw.sum() / sum(rows_e[u]["n_targets"] for u in ue)),
            "gain_link_pct": g_link, "gain_link_ci": ci_link,
            "gain_session_pct": g_sess, "gain_session_ci": ci_sess,
            "gain_all_pct": g_all, "gain_all_ci": ci_all,
            "gain_link_oracle_pct": g_or, "gain_link_oracle_ci": ci_or,
            "corr": corr_table(rows_e, key),
        }
        print(f"[sm167] {key}: link {g_link:+.4f}% {ci_link}, session {g_sess:+.4f}%, oracle {g_or:+.4f}%", flush=True)
    m = result["models"][MAIN_MODEL]
    go = m["gain_link_ci"][1] < 0 and m["gain_link_pct"] <= -MIN_GAIN
    result["decision"] = "GO" if go else "NO GO"
    (HERE / "sm167-result.json").write_text(json.dumps(result, indent=1))
    write_md(result)
    print(f"[sm167] decision {result['decision']}")


def write_md(R):
    cov = R["coverage_e732"]
    names = {"a": "co-failure (a), C >= 2 and lift >= 2", "a1": "co-failure loose (a1), C >= 1",
             "b": "confusable order (b), A -> X"}
    L = ["# Links between cards (SM167): result", "",
         f"Label T24 (`{R['label']}`), training users {R['users']['wfit']}, 732 set users {R['users']['e732']}, "
         f"{cov['targets']:,} target reviews. Design: design.md (pre-registered).", "",
         "## Coverage (732 set, share of target reviews)", "",
         "| link type | has a linked card in the window | with a matched control | linked cards per covered review |",
         "|---|---:|---:|---:|"]
    for k in LINKS:
        L.append(f"| {names[k]} | {100 * cov[k]['any']:.2f}% | {100 * cov[k]['matched']:.2f}% | "
                 f"{cov[k]['links_per_covered']:.2f} |")
    for key, m in R["models"].items():
        if "skipped" in m:
            L += ["", f"## {MODELS[key][1]}: skipped ({m['skipped']})"]
            continue
        L += ["", f"## {m['name']}", "",
              f"Raw benchmark instant LogLoss {m['logloss_raw']:.5f} on {m['users_e732']} users.", "",
              "| link type | target reviews | corr(linked) | corr(matched control) | difference [95% CI] | "
              "slope linked | slope control |", "|---|---:|---:|---:|---:|---:|---:|"]
        for k in LINKS:
            c = m["corr"][k]
            L.append(f"| {names[k]} | {c['pairs_targets']:,} | {c['corr_link']:+.4f} | {c['corr_ctrl']:+.4f} | "
                     f"{c['diff']:+.4f} [{c['diff_ci'][0]:+.4f}, {c['diff_ci'][1]:+.4f}] | "
                     f"{c['slope_link']:+.4f} | {c['slope_ctrl']:+.4f} |")
        L += ["", "| correction (beta fit on the 100 training users) | LogLoss change [95% CI] |", "|---|---:|",
              f"| links beyond session baseline (M1 vs M0), the go statistic | {m['gain_link_pct']:+.4f}% "
              f"[{m['gain_link_ci'][0]:+.4f}%, {m['gain_link_ci'][1]:+.4f}%] |",
              f"| session baseline vs raw model (M0 vs raw), information | {m['gain_session_pct']:+.4f}% "
              f"[{m['gain_session_ci'][0]:+.4f}%, {m['gain_session_ci'][1]:+.4f}%] |",
              f"| everything vs raw model (M1 vs raw), information | {m['gain_all_pct']:+.4f}% "
              f"[{m['gain_all_ci'][0]:+.4f}%, {m['gain_all_ci'][1]:+.4f}%] |",
              f"| links beyond session, beta fit on the 732 set itself (in-sample upper bound) | "
              f"{m['gain_link_oracle_pct']:+.4f}% [{m['gain_link_oracle_ci'][0]:+.4f}%, "
              f"{m['gain_link_oracle_ci'][1]:+.4f}%] |", "",
              "Betas M1 (training users): " + ", ".join(f"{k} {v:+.4f}" for k, v in m["beta_M1"].items()) + "."]
    L += ["", f"## Decision: {R['decision']}", "",
          "Go rule (design.md): for Small Instant-Only (SM113), the link gain M1 vs M0 needs its 95% CI upper bound "
          "below 0 and a point gain of at least 0.05%."]
    (HERE / "sm167-result.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("features")
    f.add_argument("--pool", choices=("wfit", "e732"), required=True)
    f.add_argument("--workers", type=int, default=1)
    f.add_argument("--users", default="")
    sp.add_parser("report")
    a = ap.parse_args()
    {"features": cmd_features, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
