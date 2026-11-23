#!/usr/bin/env python
"""Answer-time spread shrink (SM153). Pre-registered in artifacts/small-model/sm153/design.md. CPU only.

Rescoring of the saved per-review time distributions of answer-time expressiveness (SM142): each card's predicted
distribution (per grade) is shifted toward the user's running per-grade centre by a factor beta in ln time and/or mixed
with the user's running per-grade distribution, p' = alpha shift_beta(p_card) + (1 - alpha) p_user. Fitted on half the
users, read out on the other half.

The SM142 / SM125 code is imported read only from the sm142 worktree (branch sm142, commit c8764a2e; env SM153_LIB).

  grid     --model KEY      every user: exact raw time LogLoss + the (beta, alpha) grid sums -> rows/sm153-KEY-grid.jsonl
  fit                       grid argmin on the fit users per model and arm -> sm153-fit.json
  apply    --model KEY      held-out users: exact LogLoss per arm, SM142 analyse_user per arm, raw and shrunk timed events
  recal    --model KEY      held-out users: the existing rolling recalibration on the shrunk events (primary arm)
  report                    -> sm153-result.json (+ tables printed for RESULT.md)
  check                     C1-C4 on a few users -> check-cpu.json
"""
from argparse import ArgumentParser
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

LIB = Path(os.environ.get("SM153_LIB", "C:/Users/jonat/workspace/wt/sm142"))
sys.path.insert(0, str(LIB))
os.environ.setdefault("SM125_TAG", "sm125")

from scripts import sm142_expr as x142  # noqa: E402
from scripts import sm125_recal as r125  # noqa: E402
from scripts.sm117_head import MODELS, bootstrap_idx  # noqa: E402
from scripts.sm117_timelib import (BINS, LOG_EDGES, TIME_BIN_EDGES_MS, TIME_CAP_MS, censored_ll, median_log_s,  # noqa: E402
                                   time_labels)

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "artifacts/small-model/sm153"
ROWS = OUT / "rows"
WORK = OUT / "work"
WTD = Path("C:/Users/jonat/workspace/wt")
DUMP142 = LIB / "artifacts/small-model/sm142/work"
MODELS_153 = ("sm99", "sm113", "sm118", "sm130th", "sm130io")
RECAL_SRC = {"sm113": ("sm125", "sm125"), "sm118": ("sm125", "sm125"), "sm99": ("sm128", "sm128"),
             "sm130th": ("sm140", "sm140"), "sm130io": ("sm140", "sm140")}
ALPHAS = np.round(np.r_[np.arange(0.50, 0.8999, 0.05), np.arange(0.90, 1.0001, 0.01)], 4)
BETAS = np.round(np.arange(0.40, 1.1001, 0.05), 4)
KAPPA_GRADE = 4.0
W15 = float(LOG_EDGES[1] - LOG_EDGES[0])                       # bin width in ln s (0.342)
KNOTS = np.r_[math.log(0.25), LOG_EDGES, math.log(60.0) + W15]  # 17 knots: bin k spans [KNOTS[k], KNOTS[k+1]]
S_MAIN = (2, 3, 4, 5, 6, 8, 10)
CELLS = [f"fast/{sn}/{s}" for sn in ("GE", "Rec", "E") for s in S_MAIN]
AUC_S = ("2", "4", "8")


def jl(path):
    p = Path(path)
    return {r["user_id"]: r for r in map(json.loads, p.read_text().splitlines()) if "error" not in r} if p.exists() else {}


def split_users():
    users = sorted(jl(LIB / "artifacts/small-model/sm142/gpu/dump-sm113.jsonl"))
    perm = np.random.default_rng(153).permutation(len(users))
    half = len(users) // 2
    return sorted(users[i] for i in perm[:half]), sorted(users[i] for i in perm[half:])


# ------------------------------------------------------------------ the user's running per-grade distribution
def user_dist(user, ti):
    """p_user [n, 4, 16] and centre c [n, 4] (ln s) at history rows ti (0-based), from strictly earlier labelled rows."""
    import pandas as pd
    fr = pd.read_parquet(Path(x142.data_path()) / "revlogs" / f"user_id={user}", columns=["rating", "duration", "elapsed_days"])
    lower, cens, cap = time_labels(fr["duration"].to_numpy(), fr["elapsed_days"].to_numpy(), same_day=True)
    g = fr["rating"].to_numpy(np.int64) - 1
    ok = (lower >= 0) & (g >= 0) & (g <= 3)
    code = np.where(ok, g * BINS + np.minimum(lower, BINS - 1), -1)
    onehot = np.zeros((len(code), 4 * BINS), np.int32)
    onehot[np.flatnonzero(ok), code[ok]] = 1
    cum = np.cumsum(onehot, axis=0)
    before = (cum[ti] - onehot[ti]).reshape(-1, 4, BINS).astype(np.float64)      # strictly earlier rows only
    n_all = before.sum(1)                                                          # [n, 16]
    h = (n_all + 1.0) / (n_all.sum(-1, keepdims=True) + BINS)
    pu = (before + KAPPA_GRADE * h[:, None, :]) / (before.sum(-1, keepdims=True) + KAPPA_GRADE)
    c = median_log_s(pu.reshape(-1, BINS)).reshape(-1, 4)
    return pu, c, int(cap)


# ------------------------------------------------------------------ shrink operators
def shift(p, delta):
    """Shift [M, 16] bin distributions by delta [M] in ln time (piecewise uniform per bin, bin 15 over [60 s, 60 s e^w]
    for the shift only; mass below the first edge stays in bin 0, above 60 s goes to bin 15)."""
    if not np.any(delta):
        return p.copy()
    C = np.c_[np.zeros(len(p)), np.cumsum(p, axis=1)]                              # [M, 17]
    z = LOG_EDGES[None, :] - delta[:, None]                                        # [M, 15] interior edges
    j = np.clip(np.searchsorted(KNOTS, z, side="right") - 1, 0, BINS - 1)
    lo, hi = KNOTS[j], KNOTS[j + 1]
    frac = np.clip((z - lo) / (hi - lo), 0.0, 1.0)
    G = np.take_along_axis(C, j, 1) + frac * np.take_along_axis(p, j, 1)
    G = np.clip(G, 0.0, 1.0)
    G = np.maximum.accumulate(G, axis=1)
    out = np.diff(np.c_[np.zeros(len(p)), G, np.ones(len(p))], axis=1)
    return np.maximum(out, 0.0)


def shrink_all(pt, pu, c, alpha, beta):
    """[n, 4, 16] shrunk distributions for every grade."""
    n = len(pt)
    flat = pt.reshape(-1, BINS)
    if beta != 1.0:
        m = median_log_s(flat)
        flat = shift(flat, (beta - 1.0) * (m - c.reshape(-1)))
    q = flat.reshape(n, 4, BINS)
    return q if alpha == 1.0 else alpha * q + (1.0 - alpha) * pu


def label_prob(p, lower, cens):
    tail = np.cumsum(p[:, ::-1], axis=1)[:, ::-1]
    return np.where(cens > 0, np.take_along_axis(tail, lower[:, None], 1)[:, 0], np.take_along_axis(p, lower[:, None], 1)[:, 0])


# ------------------------------------------------------------------ per-user loading
def load_user(key, user):
    with np.load(DUMP142 / f"dump-{key}" / f"{user}.npz") as zz:
        z = {k: zz[k] for k in zz.files}
    ti = z["th"].astype(np.int64) - 1
    pu, c, cap = user_dist(user, ti)
    d = z["d"].astype(np.int64)
    # labels of the test rows: SM117's time_labels rule with the user's own cap (from the whole history)
    lower = np.searchsorted(np.asarray(TIME_BIN_EDGES_MS), d.astype(np.float64), side="right").astype(np.int64)
    cens = ((d >= cap) & (cap < TIME_CAP_MS)).astype(np.int64)
    return z, pu, c, cap, lower, cens


def grid_user(key, user):
    z, pu, c, cap, lower, cens = load_user(key, user)
    gr = z["grade"].astype(np.int64)
    rows = np.arange(len(gr))
    pc = z["pt"].astype(np.float64)[rows, gr]
    pug = pu[rows, gr]
    cg = c[rows, gr]
    m = median_log_s(pc)
    raw_ll = float(censored_ll(pc, lower, cens).sum())
    u = label_prob(pug, lower, cens)
    out = np.empty((len(BETAS), len(ALPHAS)))
    for i, b in enumerate(BETAS):
        q = pc if b == 1.0 else shift(pc, (b - 1.0) * (m - cg))
        a = label_prob(q, lower, cens)
        mix = ALPHAS[None, :] * a[:, None] + (1.0 - ALPHAS[None, :]) * u[:, None]
        out[i] = -np.log(np.maximum(mix, 1e-7)).sum(0)
    user_ll = float(censored_ll(pug, lower, cens).sum())
    return {"user_id": int(user), "n": int(len(gr)), "cap": cap, "raw_ll": raw_ll, "user_only_ll": user_ll,
            "grid": np.round(out, 6).tolist()}


def cmd_grid(args):
    key = args.model
    idx = jl(LIB / f"artifacts/small-model/sm142/gpu/dump-{key}.jsonl")
    out = ROWS / f"sm153-{key}-grid.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set(jl(out))
    users = [u for u in sorted(idx) if u not in done]
    t0 = time.time()
    print(f"[sm153-grid] {key}: {len(idx)} users, {len(users)} to do", flush=True)
    with out.open("a", encoding="utf-8") as f:
        for k, u in enumerate(users):
            rec = grid_user(key, u)
            ref = idx[u]["sums"]["model"]
            rec["c1_rel"] = abs(rec["raw_ll"] - ref["ll"]) / max(ref["ll"], 1e-9)
            rec["c1_n"] = ref["n"] == rec["n"]
            f.write(json.dumps(rec) + "\n"); f.flush()
            if (k + 1) % 50 == 0:
                print(f"[sm153-grid] {k + 1}/{len(users)} users, {time.time() - t0:.0f} s", flush=True)
    print(f"[sm153-grid] done {len(users)} users in {time.time() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------ fit
ARMS = ("combined", "mixture", "shift")


def cmd_fit(args):
    fit_u, held_u = split_users()
    res = {"fit_users": len(fit_u), "held_users": len(held_u), "alphas": ALPHAS.tolist(), "betas": BETAS.tolist(), "models": {}}
    ib1, ia1 = int(np.flatnonzero(BETAS == 1.0)[0]), int(np.flatnonzero(ALPHAS == 1.0)[0])
    for key in MODELS_153:
        g = jl(ROWS / f"sm153-{key}-grid.jsonl")
        if not g:
            continue
        rows = [g[u] for u in fit_u if u in g]
        G = np.sum([r["grid"] for r in rows], axis=0)
        n = sum(r["n"] for r in rows)
        raw = sum(r["raw_ll"] for r in rows) / n
        e = {"fit_users": len(rows), "fit_rows": n, "fit_raw_ll": raw, "c1_max_rel": max(r["c1_rel"] for r in g.values()),
             "c1_counts_ok": all(r["c1_n"] for r in g.values())}
        sub = {"combined": (slice(None), slice(None)), "mixture": (slice(ib1, ib1 + 1), slice(None)), "shift": (slice(None), slice(ia1, ia1 + 1))}
        for arm, (sb, sa) in sub.items():
            M = G[sb, sa]
            i, j = np.unravel_index(np.argmin(M), M.shape)
            b = float(BETAS[sb][i]); a = float(ALPHAS[sa][j])
            e[arm] = {"alpha": a, "beta": b, "fit_ll": float(M[i, j] / n), "fit_rel_pct": 100 * (M[i, j] / n - raw) / raw}
        res["models"][key] = e
        print(key, json.dumps({k: e[k] for k in ARMS}), f"C1 max rel {e['c1_max_rel']:.2e}", flush=True)
    (OUT / "sm153-fit.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------ apply (held-out users)
def cmd_apply(args):
    key = args.model
    fit = json.loads((OUT / "sm153-fit.json").read_text())["models"][key]
    _, held = split_users()
    idx = jl(LIB / f"artifacts/small-model/sm142/gpu/dump-{key}.jsonl")
    users = [u for u in held if u in idx]
    if args.limit:
        users = users[:args.limit]
    out = ROWS / f"sm153-{key}-apply.jsonl"
    done = set(jl(out))
    arms = {a: (fit[a]["alpha"], fit[a]["beta"]) for a in ARMS}
    for a in ARMS:
        (WORK / a / "rows").mkdir(parents=True, exist_ok=True)
        (WORK / a / f"sample-{key}").mkdir(parents=True, exist_ok=True)
    ev_dir = WORK / f"events-{key}"
    ev_dir.mkdir(parents=True, exist_ok=True)
    src_tag, _ = RECAL_SRC[key]
    t0 = time.time()
    print(f"[sm153-apply] {key}: {len(users)} held-out users, {len(done)} done, arms {arms}", flush=True)
    with out.open("a", encoding="utf-8") as f:
        for k, u in enumerate(users):
            if u in done:
                continue
            z, pu, c, cap, lower, cens = load_user(key, u)
            gr = z["grade"].astype(np.int64)
            rows = np.arange(len(gr))
            pt = z["pt"].astype(np.float64)
            base = x142.user_base(u)
            rec = {"user_id": int(u), "n": int(len(gr)), "ll": {"raw": float(censored_ll(pt[rows, gr], lower, cens).sum()),
                                                               "user_only": float(censored_ll(pu[rows, gr], lower, cens).sum())}}
            # C2: raw events from the SM142 distributions vs the recalibration dump of the same rows
            P_raw = r125.event_matrix(z["pg"].astype(np.float64), pt)
            rz = np.load(WTD / src_tag / f"artifacts/small-model/{src_tag}/work/dump-{key}/{u}.npz")
            same = len(rz["grade"]) == len(gr) and np.array_equal(rz["grade"], z["grade"]) and np.array_equal(rz["d"], z["d"])
            rec["c2_same_rows"] = bool(same)
            rec["c2_max_abs"] = float(np.abs(P_raw - rz["p"].astype(np.float64)).max()) if same else None
            day = rz["day"].astype(np.int64)
            for a, (al, be) in arms.items():
                q = shrink_all(pt, pu, c, al, be)
                rec["ll"][a] = float(censored_ll(q[rows, gr], lower, cens).sum())
                rec.setdefault("c3", {})[a] = [float(np.abs(q.sum(-1) - 1).max()), float(q.min())]
                zz = dict(z); zz["pt"] = q.astype(np.float32)
                r, sample = x142.analyse_user(zz, base, u)
                r["model"] = key
                with (WORK / a / "rows" / f"sm142-{key}-users.jsonl").open("a", encoding="utf-8") as g:
                    g.write(json.dumps(r) + "\n")
                if sample is not None:
                    x142.atomic_npz(WORK / a / f"sample-{key}" / f"{u}.npz", **sample)
                if a == "combined":
                    P = r125.event_matrix(z["pg"].astype(np.float64), q)
                    x142.atomic_npz(ev_dir / f"{u}.npz", p=P.astype(np.float32), grade=z["grade"], d=z["d"], day=day.astype(np.int32))
            f.write(json.dumps(rec) + "\n"); f.flush()
            if (k + 1) % 25 == 0:
                print(f"[sm153-apply] {k + 1}/{len(users)} users, {time.time() - t0:.0f} s", flush=True)
    print(f"[sm153-apply] done in {time.time() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------ recalibration of the shrunk events (held-out users)
_ORIG_VALID = r125.valid_cols
_CELL_COL = np.array([c[1] in S_MAIN for c in r125.COLS])


def lean_valid(cap):
    """Only the 21 cells are fitted (Platt fits are independent per event, the repair of a cell only reads lower S,
    the same S and the raw P(set)), so the cells equal the full scheme's; the other columns stay raw."""
    return _ORIG_VALID(cap) & _CELL_COL


r125.valid_cols = lean_valid


def cmd_recal(args):
    key = args.model
    src_tag, _ = RECAL_SRC[key]
    idx = jl(LIB / f"artifacts/small-model/sm142/gpu/dump-{key}.jsonl")
    _, held = split_users()
    users = [u for u in held if u in idx]
    if args.limit:
        users = users[:args.limit]
    out = ROWS / f"sm153-{key}-recal-{args.arm}.jsonl"
    done = set(jl(out))
    t0 = time.time()
    print(f"[sm153-recal] {key}: {len(users)} users, {len(done)} done ({args.arm})", flush=True)
    with out.open("a", encoding="utf-8") as f:
        for k, u in enumerate(users):
            if u in done:
                continue
            if args.arm == "raw":
                rz = np.load(WTD / src_tag / f"artifacts/small-model/{src_tag}/work/dump-{key}/{u}.npz")
            else:
                rz = np.load(WORK / f"events-{key}" / f"{u}.npz")
            arms, Y, V, active, st = r125.recal_user(rz["p"], rz["grade"].astype(np.int64), rz["d"].astype(np.int64),
                                                    rz["day"].astype(np.int64), idx[u]["user_cap_ms"])
            sums = {}
            for arm in ("raw", "cal"):
                sums[arm] = {}
                for j, cc in enumerate(r125.COLS):
                    kk = r125.key_of(cc)
                    if kk in CELLS or cc[1] == 0:
                        sums[arm][kk] = r125.binary_sums(arms[arm][:, j], Y[:, j]) if V[j] else [0, 0.0, 0.0, 0.0, 0.0]
            f.write(json.dumps({"user_id": int(u), "arm": args.arm, "active_rows": st["active_rows"], "sums": sums}) + "\n"); f.flush()
            if (k + 1) % 25 == 0:
                print(f"[sm153-recal] {k + 1}/{len(users)} users, {time.time() - t0:.0f} s", flush=True)
    print(f"[sm153-recal] done in {time.time() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------ report
def paired(a, b, w):
    a, b, w = (np.asarray(v, np.float64) for v in (a, b, w))
    idx = bootstrap_idx(len(a))
    d = float((a - b).sum() / w.sum())
    bs = (a - b)[idx].sum(1) / w[idx].sum(1)
    ref = float(b.sum() / w.sum())
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    return {"delta": d, "ci": [lo, hi], "rel_pct": 100 * d / ref, "rel_ci_pct": [100 * lo / ref, 100 * hi / ref], "ref": ref,
            "users": int(len(a)), "users_better": int((a < b).sum())}


def go_of(p):
    return p["ci"][1] < 0 and -p["rel_pct"] >= 0.05


def agg_expr(key, rows_dir, work_dir, users):
    """SM142's aggregate on a subset of users (rows from rows_dir, samples from work_dir)."""
    src = jl(Path(rows_dir) / f"sm142-{key}-users.jsonl")
    tmp = OUT / "work" / "agg-tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / f"sm142-{key}-users.jsonl").write_text("".join(json.dumps(src[u]) + "\n" for u in users if u in src))
    old_rows, old_work = x142.ROWS, x142.WORK
    x142.ROWS, x142.WORK = tmp, Path(work_dir)
    try:
        return x142.aggregate(key), src
    finally:
        x142.ROWS, x142.WORK = old_rows, old_work


def expr_summary(res):
    st = res["auc"]["strat"]
    cl = res["card"]
    sh = res["shape"]
    return {"auc": {s: st[s]["head"] for s in AUC_S if s in st},
            "review_slope": res["slope"]["head"]["pooled"], "between_slope": cl["head"]["between_slope"],
            "spread_ratio": cl["head"]["spread_ratio"], "explained": cl["head"]["explained"],
            "head_minus_card_median_share": cl["head-a"],
            "curves_loc_0.1": sh["head_loc_0.1"]["median"], "shapes_norm_0.1": sh["head_norm_0.1"]["median"],
            "sd_log_median": sh["sd_x"]["median"], "sd_log_iqr": sh["sd_w"]["median"],
            "pca_k95_norm": sh["pooled"]["head_norm"]["k95"], "pca_pr_norm": sh["pooled"]["head_norm"]["pr"],
            "card_shape_cal": sh["calibration"].get("card_head"), "coverage_all": sh["calibration"]["coverage_all"],
            "label": res["labels"]["label"]}


def cmd_report(args):
    fitj = json.loads((OUT / "sm153-fit.json").read_text())
    _, held = split_users()
    res = {"name": "Answer-time spread shrink (SM153)", "fit": fitj, "models": {}}
    for key in MODELS_153:
        ap = jl(ROWS / f"sm153-{key}-apply.jsonl")
        if not ap:
            continue
        us = [u for u in held if u in ap]
        n = [ap[u]["n"] for u in us]
        raw = [ap[u]["ll"]["raw"] for u in us]
        e = {"name": MODELS[key]["name"], "users": len(us), "rows": int(sum(n)), "raw_ll": sum(raw) / sum(n), "arms": {}}
        for a in ARMS + ("user_only",):
            p = paired([ap[u]["ll"][a] for u in us], raw, n)
            p["go"] = go_of(p)
            if a in ARMS:
                p["alpha"], p["beta"] = fitj["models"][key][a]["alpha"], fitj["models"][key][a]["beta"]
            e["arms"][a] = p
        e["c2_same_rows"] = all(ap[u]["c2_same_rows"] for u in us)
        e["c2_max_abs"] = max(ap[u]["c2_max_abs"] or 0 for u in us)
        e["c3_max_sum_err"] = max(max(v[0] for v in ap[u]["c3"].values()) for u in us)
        e["c3_min"] = min(min(v[1] for v in ap[u]["c3"].values()) for u in us)
        # expressiveness: raw (SM142 rows) vs each arm, held-out users
        raw_res, raw_rows = agg_expr(key, LIB / "artifacts/small-model/sm142/rows", DUMP142, us)
        e["expr"] = {"raw": expr_summary(raw_res)}
        for a in ARMS:
            ar, arm_rows = agg_expr(key, WORK / a / "rows", WORK / a, us)
            e["expr"][a] = expr_summary(ar)
            dd = {}
            for s in AUC_S:
                pr = [(arm_rows[u]["auc"][s]["strat"]["head"], raw_rows[u]["auc"][s]["strat"]["head"]) for u in us
                      if u in arm_rows and u in raw_rows and "strat" in arm_rows[u]["auc"].get(s, {}) and "strat" in raw_rows[u]["auc"].get(s, {})]
                v = np.array([p0 - p1 for p0, p1 in pr])
                idx = bootstrap_idx(len(v))
                bs = v[idx].mean(1)
                dd[s] = [float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(v)]
            e["expr"][a]["auc_minus_raw"] = dd
            e["expr"][a]["kills_expressiveness"] = bool(any(dd[s][0] < -0.005 and dd[s][2] < 0 for s in dd)
                                                         or e["expr"][a]["shapes_norm_0.1"] < 0.5 * e["expr"]["raw"]["shapes_norm_0.1"])
        # recalibration (combined arm)
        rc = jl(ROWS / f"sm153-{key}-recal-combined.jsonl")
        src_tag, _ = RECAL_SRC[key]
        rec_rows = jl(WTD / src_tag / f"artifacts/small-model/{src_tag}/rows/{src_tag}-{key}-recal.jsonl")
        ur = [u for u in us if u in rc and u in rec_rows]
        if ur:
            cells = {}
            tot = {k2: [] for k2 in ("shr_raw", "base_raw", "shr_cal", "base_cal")}
            for cell in CELLS:
                nn = [rec_rows[u]["sums"]["cal"][cell][0] for u in ur]
                if sum(nn) == 0:
                    continue
                vals = {"shr_raw": [rc[u]["sums"]["raw"][cell][1] for u in ur], "base_raw": [rec_rows[u]["sums"]["raw"][cell][1] for u in ur],
                        "shr_cal": [rc[u]["sums"]["cal"][cell][1] for u in ur], "base_cal": [rec_rows[u]["sums"]["cal"][cell][1] for u in ur]}
                for k2 in tot:
                    tot[k2].append(vals[k2])
                cells[cell] = {"before": paired(vals["shr_raw"], vals["base_raw"], nn), "after": paired(vals["shr_cal"], vals["base_cal"], nn),
                               "recal_vs_raw": paired(vals["base_cal"], vals["base_raw"], nn)}
            ones = [1] * len(ur)
            e["recal"] = {"users": len(ur), "cells": cells,
                          "total_before": paired(np.sum(tot["shr_raw"], 0), np.sum(tot["base_raw"], 0), [rec_rows[u]["sums"]["cal"]["fast/GE/4"][0] for u in ur]),
                          "total_after": paired(np.sum(tot["shr_cal"], 0), np.sum(tot["base_cal"], 0), [rec_rows[u]["sums"]["cal"]["fast/GE/4"][0] for u in ur]),
                          "after_worse_cells": sum(1 for c in cells.values() if c["after"]["ci"][0] > 0),
                          "after_better_cells": sum(1 for c in cells.values() if c["after"]["ci"][1] < 0)}
            del ones
        res["models"][key] = e
        print(key, "done", flush=True)
    (OUT / "sm153-result.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------ checks
def cmd_check(args):
    out = {}
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(BINS), size=500)
    q0 = shift(p, np.zeros(500))
    q1 = shift(p, rng.normal(0, 0.8, 500))
    pu = rng.dirichlet(np.ones(BINS), size=(500, 4)); pt = rng.dirichlet(np.ones(BINS), size=(500, 4)); c = rng.normal(1, 0.3, (500, 4))
    out["C3"] = {"shift0_max_abs": float(np.abs(q0 - p).max()), "shift_sum_err": float(np.abs(q1.sum(1) - 1).max()), "shift_min": float(q1.min()),
                 "alpha1_beta1_exact": bool(np.array_equal(shrink_all(pt, pu, c, 1.0, 1.0), pt)),
                 "shift_roundtrip_mean_abs": float(np.abs(shift(shift(p, np.full(500, 0.1)), np.full(500, -0.1)) - p)[:, 1:-1].mean())}
    # C4 causality: p_user at row i unchanged when rows >= i are altered (patched reader)
    import pandas as pd
    user = 4
    fr = pd.read_parquet(Path(x142.data_path()) / "revlogs" / f"user_id={user}", columns=["rating", "duration", "elapsed_days"])
    n = len(fr); cut = n // 2
    ti = np.arange(n)
    pu_a, c_a, _ = user_dist(user, ti)
    orig = pd.read_parquet

    def patched(path, columns=None):
        f2 = orig(path, columns=columns).copy()
        f2.loc[f2.index[cut:], "duration"] = 1234
        f2.loc[f2.index[cut:], "rating"] = 1
        return f2
    pd.read_parquet = patched
    try:
        pu_b, c_b, _ = user_dist(user, ti)
    finally:
        pd.read_parquet = orig
    out["C4"] = {"user": user, "rows": n, "cut": cut, "before_cut_equal": bool(np.array_equal(pu_a[:cut + 1], pu_b[:cut + 1])),
                 "after_cut_changed": bool(not np.array_equal(pu_a[cut + 2:], pu_b[cut + 2:]))}
    # C1 on 3 users of the smallest model
    idx = jl(LIB / "artifacts/small-model/sm142/gpu/dump-sm130th.jsonl")
    us = sorted(idx, key=lambda u: idx[u]["reviews"])[:3]
    out["C1"] = []
    for u in us:
        g = grid_user("sm130th", u)
        ib1, ia1 = int(np.flatnonzero(BETAS == 1.0)[0]), int(np.flatnonzero(ALPHAS == 1.0)[0])
        out["C1"].append({"user": u, "raw_ll": g["raw_ll"], "recorded": idx[u]["sums"]["model"]["ll"],
                          "rel": abs(g["raw_ll"] - idx[u]["sums"]["model"]["ll"]) / idx[u]["sums"]["model"]["ll"],
                          "grid_identity_rel": abs(g["grid"][ib1][ia1] - g["raw_ll"]) / g["raw_ll"]})
    print(json.dumps(out, indent=1))
    (OUT / "check-cpu.json").write_text(json.dumps(out, indent=1))


def main():
    ap = ArgumentParser()
    ap.add_argument("cmd", choices=["grid", "fit", "apply", "recal", "report", "check"])
    ap.add_argument("--model")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arm", default="combined")
    a = ap.parse_args()
    {"grid": cmd_grid, "fit": cmd_fit, "apply": cmd_apply, "recal": cmd_recal, "report": cmd_report, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
