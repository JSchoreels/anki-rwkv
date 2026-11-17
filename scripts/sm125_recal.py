#!/usr/bin/env python
"""Rolling recalibration on the 732 set (SM125): does a causal per-user rolling Platt scaling (refitted every 7 days on
the user's latest 30,000 earlier reviews, raw model before 1,000 earlier reviews) make the timed events of fast Good/Easy
prediction (SM120) reliable on the 732 set without losing their LogLoss lead? Pre-registered in
artifacts/small-model/sm125/design.md.

Answer time = front + back (question shown -> grade pressed, capped at the preset's maximum answer seconds): "within
S s" always means answered within S s, reading the back included.

The forward pass, head inputs and event definitions are SM120's / SM117's (scripts/sm120_fast.py, scripts/sm117_head.py,
imported, not edited); the baselines are SM120's recorded per-user sums (artifacts/small-model/sm120/gpu, same rows,
checked by the review sha256), not recalibrated.

  dump    --model KEY [--users-file F] [--device cuda]   per-review raw p (33 events) + labels -> work/dump-KEY/<user>.npz,
                                                          index gpu/dump-KEY.jsonl (raw sums per user, checked vs SM120)
  recal   --model KEY [--until EPOCH]                    per-user rolling Platt (CPU) -> rows/sm125-KEY-recal.jsonl +
                                                          work/recal-KEY.hist/<user>.npz (resumable, stops before EPOCH)
  check   [--out F]                                      CPU proofs P1-P8 (small users, CPU, no GPU) -> check-cpu.json
  report                                                 all recalibrated models -> sm125-result.{md,json}
"""
from argparse import ArgumentParser, Namespace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import sm120_fast as f120  # noqa: E402  (read only: S grid, sets, head dirs)
from scripts.sm117_head import (MODELS, ORDER, HIST_BINS, U732, verified_master, build_model, Frozen, load_head,  # noqa: E402
                                ScoreBatches, cdf_at, binary_sums, hist_of, ece_auc_from_hist, pool_users, bootstrap_idx, _diff)
from scripts.sm117_timelib import BINS, label_db, test_reviews, first, atomic_json  # noqa: E402

OUT = ROOT / "artifacts/small-model/sm125"
WORK = OUT / "work"
GPU = OUT / "gpu"
ROWS = OUT / "rows"
SM120_ROWS = ROOT / "artifacts/small-model/sm120/gpu"       # SM120's committed per-user sums (baselines), read only
S_GRID = f120.S_GRID                                        # (2, 3, 4, 5, 6, 8, 10, 15, 20, 30)
S_MAIN = (2, 3, 4, 5, 6, 8, 10)                             # the 2-10 s range of the go rule
SETS = f120.SETS                                            # Rec, GE, E (0-based grades)
SET_ORDER = ("E", "GE", "Rec")                              # nested: E within GE within Rec
SET_NAMES = f120.SET_NAMES
COLS = [(sn, 0) for sn in SETS] + [(sn, s) for sn in SETS for s in S_GRID]   # s = 0: grade only, P(set)
CI = {c: i for i, c in enumerate(COLS)}
MAIN_MODELS = ("sm113", "sm115", "sm118", "sm98", "sm106")   # the release family candidates
ROLL_DAYS, ROLL_ROWS = 7, 30000
N_MIN, N_MIN_SENS = 1000, 5000                              # earlier rows needed before recalibrating (primary, sensitivity)
MIN_CLASS = 20                                              # positives and negatives needed in the fit rows, else identity
A_RANGE, B_RANGE = (0.05, 20.0), (-15.0, 15.0)              # Platt slope / intercept clamps (safety; hits are counted)
EPS = 1e-6
MIN_REL_PCT = 0.05                                          # every go difference at least 0.05% of the reference LogLoss
ARMS = ("raw", "cal", "cal5k", "cal_norepair", "raw@active", "cal@active", "cal_grade")
ARM_NAMES = {"raw": "raw model (SM120)", "cal": "rolling Platt per user (primary)", "cal5k": "rolling Platt per user, 5,000-review minimum",
             "cal_norepair": "rolling Platt per user, before the repair", "raw@active": "raw model, recalibrated rows only",
             "cal@active": "rolling Platt per user, recalibrated rows only",
             "cal_grade": "diagnostic: grade-only events recalibrated too"}


def key_of(c):
    return f"grade/{c[0]}" if c[1] == 0 else f"fast/{c[0]}/{c[1]}"


def head_dir(key):
    return f120.HEAD_DIRS[key]


# ------------------------------------------------------------------ dump (GPU pass; also CPU for the check)
def event_matrix(pg, pt):
    """[n, 33] float64: P(set) and P(set and answered within S s) = sum_{g in set} P(g) P(t <= S | g), SM120's formulas."""
    out = np.empty((len(pg), len(COLS)))
    cdfs = {s: cdf_at(pt, s) for s in S_GRID}
    for j, (sn, s) in enumerate(COLS):
        gs = list(SETS[sn])
        out[:, j] = pg[:, gs].sum(1) if s == 0 else (pg[:, gs] * cdfs[s][:, gs]).sum(1)
    return out


def labels(gr, d):
    """[n, 33] float64 labels of the 33 events (grade only; set and d <= S)."""
    y = np.empty((len(gr), len(COLS)))
    for j, (sn, s) in enumerate(COLS):
        y[:, j] = np.isin(gr, SETS[sn]) & ((s == 0) | (d <= 1000 * s))
    return y


def valid_cols(cap):
    """S counts only below the user's own maximum answer seconds (SM117's `valid`); grade-only events always."""
    return np.array([s == 0 or cap > 1000 * s for _, s in COLS])


def dump(args):
    """Every SM117 time-labelled test review of the users (fp32, full history, show-time inputs, leak-free), exactly as
    SM120's score: the 33 event probabilities (float32), grade, duration, day -> <out-dir>/<user>.npz; one index record
    per user with the float64 raw sums (checked against SM120's recorded model sums by `recal`)."""
    from scripts import small_model_heldout as smh  # noqa: F401  first
    import torch
    import pandas as pd
    from scripts import compress_common as cc
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("HELDOUT_GPU_MEMORY_FRACTION", "0.8")))
    else:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    info = verified_master(args.model)
    hd = Path(args.head_dir or head_dir(args.model))
    trunk_path = hd / "master.pth"
    trunk = torch.load(trunk_path, map_location="cpu", weights_only=True)
    released = torch.load(info["master"], map_location="cpu", weights_only=True)
    if set(trunk) != set(released) or not all(torch.equal(trunk[k].float(), released[k].float()) for k in released):
        raise SystemExit(f"{trunk_path} is not the released model of {args.model}")
    frozen = Frozen(build_model(args.model, trunk, device), MODELS[args.model]["kind"])
    head = load_head(hd / "head.pth", frozen.inputs).to(device)
    users = json.loads(Path(args.users_file).read_text())["users"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = Path(args.index)
    index.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["user_id"] for line in index.read_text().splitlines() if '"error"' not in line} if index.exists() else set()
    todo = [u for u in users if u not in done]
    data = cc.data_path()
    loader = torch.utils.data.DataLoader(ScoreBatches(todo, data, cc.run_info()["review_counts"]), batch_size=1,
                                         num_workers=args.workers, collate_fn=first, prefetch_factor=1 if args.workers else None)
    env = label_db()
    started = time.perf_counter()
    nbytes = 0
    print(f"[sm125-dump] {MODELS[args.model]['name']}: {len(users)} users, {len(todo)} to dump", flush=True)
    f = index.open("a", encoding="utf-8")
    try:
        for k, (user, batch, error, prep, table, rev) in enumerate(loader):
            if batch is None:
                f.write(json.dumps({"user_id": user, "error": error}) + "\n"); f.flush()
                continue
            with env.begin() as txn:
                ths, frame, lower, cens, cap = test_reviews(data, user, txn, same_day=True)
            if ths is None or len(ths) == 0:
                f.write(json.dumps({"user_id": user, "error": "no time-labelled test reviews"}) + "\n"); f.flush()
                continue
            try:
                gb = batch.to(device)
                with torch.inference_mode():
                    qi, q_th, feats, o = frozen.forward(gb)
                    p_logits = o[3][0].float()
                    order = torch.argsort(q_th)
                    q_th, qi, feats = q_th[order], qi[order], feats[order]
                    want = torch.as_tensor(ths, device=device)
                    pos = torch.searchsorted(q_th, want).clamp(max=len(q_th) - 1)
                    found = q_th[pos] == want
                    if not bool(found.all()):
                        raise RuntimeError(f"{int((~found).sum())} test reviews without an instant row")
                    x = torch.cat([feats[pos], torch.as_tensor(table, device=device)[want - 1]], -1)
                    pg = torch.softmax(p_logits[qi[pos]], -1).double().cpu().numpy()
                    pt = torch.softmax(head(x).float().view(-1, 4, BINS), -1).double().cpu().numpy()
                del gb, o
            except torch.OutOfMemoryError as oom:
                frozen.clear()
                torch.cuda.empty_cache()
                f.write(json.dumps({"user_id": user, "error": f"OOM: {oom}"[:300]}) + "\n"); f.flush()
                continue
            except RuntimeError as err:   # as SM120: recorded, the user is left out (pass 2 retries)
                frozen.clear()
                if "out of memory" in str(err):
                    torch.cuda.empty_cache()
                print(f"[sm125-dump] user {user}: {err}", flush=True)
                f.write(json.dumps({"user_id": user, "error": f"{err}"[:300]}) + "\n"); f.flush()
                continue
            t = ths - 1
            d = frame["duration"].to_numpy(np.int64)[t]
            gr = frame["rating"].to_numpy(np.int64)[t] - 1
            day = pd.read_parquet(Path(data) / "revlogs" / f"user_id={user}", columns=["day_offset"])["day_offset"].to_numpy(np.int64)[t]
            P = event_matrix(pg, pt)
            Y = labels(gr, d)
            V = valid_cols(cap)
            sums = {key_of(c): binary_sums(P[:, j], Y[:, j]) if V[j] else [0, 0.0, 0.0, 0.0, 0.0] for j, c in enumerate(COLS)}
            tmp = out_dir / f"{user}.tmp.npz"
            np.savez(tmp, p=P.astype(np.float32), grade=gr.astype(np.int8), d=d.astype(np.int32), day=day.astype(np.int32))
            os.replace(tmp, out_dir / f"{user}.npz")
            nbytes += (out_dir / f"{user}.npz").stat().st_size
            rec = {"user_id": user, "reviews": int(len(ths)), "review_th_sha256": hashlib.sha256(np.asarray(ths, np.int64).tobytes()).hexdigest(),
                   "model": args.model, "trunk": str(trunk_path), "head_dir": str(hd), "precision": "fp32", "device": args.device,
                   "user_cap_ms": int(cap), "days": [int(day.min()), int(day.max())], "day_sorted": bool((np.diff(day) >= 0).all()),
                   "grade_counts": np.bincount(gr, minlength=4).tolist(), "raw_sums": sums}
            f.write(json.dumps(rec) + "\n"); f.flush()
            if (k + 1) % 50 == 0:
                print(f"[sm125-dump] {k + 1}/{len(todo)} users, {time.perf_counter() - started:.0f} s, {nbytes / 1e9:.2f} GB", flush=True)
    finally:
        f.close()
        env.close()
    print(f"[sm125-dump] done {len(todo)} users in {time.perf_counter() - started:.0f} s, {nbytes / 1e9:.2f} GB -> {out_dir}", flush=True)
    return 0


# ------------------------------------------------------------------ Platt (batched Newton) and the rolling scheme
def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def fit_platt_batch(X, Y, w0, iters=50, tol=1e-12):
    """Per row k of X, Y ([K, n]): (a, b) minimising the mean LogLoss of sigmoid(a x + b), Newton with per-row step
    halving (SM123's fit_platt, batched over events); w0 [K, 2] warm start. Returns w [K, 2]."""
    w = w0.astype(np.float64).copy()

    def loss(w, X, Y):
        z = w[:, :1] * X + w[:, 1:]
        return np.mean(np.logaddexp(0, z) - Y * z, axis=1)
    cur = loss(w, X, Y)
    act = np.ones(len(X), bool)
    for _ in range(iters):
        if not act.any():
            break
        a = np.flatnonzero(act)
        Xa, Ya, wa = X[a], Y[a], w[a]
        z = wa[:, :1] * Xa + wa[:, 1:]
        q = 0.5 * (1 + np.tanh(0.5 * z))
        r = q * (1 - q)
        g = np.stack([np.mean((q - Ya) * Xa, 1), np.mean(q - Ya, 1)], 1)
        h11, h12, h22 = np.mean(r * Xa * Xa, 1) + 1e-9, np.mean(r * Xa, 1), np.mean(r, 1) + 1e-9
        det = h11 * h22 - h12 * h12
        step = np.stack([(h22 * g[:, 0] - h12 * g[:, 1]) / det, (h11 * g[:, 1] - h12 * g[:, 0]) / det], 1)
        lam = np.ones(len(a))
        new_w = wa - step
        nl = loss(new_w, Xa, Ya)
        bad = nl > cur[a] + 1e-15
        while bad.any() and lam.min() > 1e-4:
            lam[bad] /= 2
            new_w[bad] = wa[bad] - lam[bad, None] * step[bad]
            nl[bad] = loss(new_w[bad], Xa[bad], Ya[bad])
            bad = (nl > cur[a] + 1e-15) & (lam > 1e-4)
        stuck = nl > cur[a] + 1e-15
        ok = ~stuck
        conv = ok & (np.abs(cur[a] - nl) < tol)
        w[a[ok]] = new_w[ok]
        cur[a[ok]] = nl[ok]
        act[a[stuck | conv]] = False
    return w


def blocks_of(day):
    """Rolling blocks of one user's rows (time-sorted): block k holds the rows with (day - d0) // 7 == k; its fit rows
    are the latest ROLL_ROWS rows of strictly earlier days. Returns [(i0, i1, lo)] = applied rows [i0, i1), fit [lo, i0)."""
    assert (np.diff(day) >= 0).all(), "rows are not in time order"
    if len(day) == 0:
        return []
    b = (day - day[0]) // ROLL_DAYS
    cuts = np.r_[0, np.flatnonzero(np.diff(b)) + 1, len(day)]
    return [(int(i0), int(i1), max(0, int(i0) - ROLL_ROWS)) for i0, i1 in zip(cuts[:-1], cuts[1:])]


def repair(P, Praw):
    """Running maximum over increasing S per set, then the nesting E <= GE <= Rec at every S, then each timed event
    capped by the raw P(set) (grade only, not recalibrated). P, Praw [n, 33]; returns the repaired copy."""
    out = P.copy()
    for sn in SET_ORDER:
        j = [CI[(sn, s)] for s in S_GRID]
        out[:, j] = np.maximum.accumulate(out[:, j], axis=1)
    for lo, hi in (("E", "GE"), ("GE", "Rec")):
        for s in S_GRID:
            out[:, CI[(hi, s)]] = np.maximum(out[:, CI[(hi, s)]], out[:, CI[(lo, s)]])
    for sn in SETS:
        j = [CI[(sn, s)] for s in S_GRID]
        out[:, j] = np.minimum(out[:, j], Praw[:, [CI[(sn, 0)]]])
        out[:, CI[(sn, 0)]] = Praw[:, CI[(sn, 0)]]
    return out


def repair_grade(P):
    """Diagnostic arm: grade-only events recalibrated too; nesting E <= GE <= Rec of P(set), then the timed repair
    against the recalibrated P(set)."""
    G = P.copy()
    G[:, CI[("GE", 0)]] = np.maximum(G[:, CI[("GE", 0)]], G[:, CI[("E", 0)]])
    G[:, CI[("Rec", 0)]] = np.maximum(G[:, CI[("Rec", 0)]], G[:, CI[("GE", 0)]])
    return repair(G, G)


def recal_user(P32, gr, d, day, cap, n_min=N_MIN, n_min_sens=N_MIN_SENS):
    """One user's arms (n x 33 float64 each) and fit statistics. Causal: block rows only see fits on rows of earlier
    days (indices < i0, day < the block's first day)."""
    Praw = P32.astype(np.float64)
    Y = labels(gr, d)
    V = valid_cols(cap)
    X = logit(Praw)
    n = len(Praw)
    cal = Praw.copy()             # raw before n_min, recalibrated after
    cal5 = Praw.copy()
    active = np.zeros(n, bool)
    w = np.tile([1.0, 0.0], (len(COLS), 1))
    st = {"blocks": 0, "fitted_blocks": 0, "identity_fits": 0, "clamped": 0, "max_fit_day_gap_min": None, "causal": True,
          "last_GE4": None}
    jv = np.flatnonzero(V)
    for i0, i1, lo in blocks_of(day):
        st["blocks"] += 1
        nfit = i0 - lo
        if nfit < n_min:
            continue
        if day[i0 - 1] >= day[i0] or i0 > i1:
            st["causal"] = False
        gap = int(day[i0] - day[i0 - 1])
        st["max_fit_day_gap_min"] = gap if st["max_fit_day_gap_min"] is None else min(st["max_fit_day_gap_min"], gap)
        Yf = Y[lo:i0, jv]
        npos = Yf.sum(0)
        ok = (npos >= MIN_CLASS) & (nfit - npos >= MIN_CLASS)
        wk = np.tile([1.0, 0.0], (len(jv), 1))
        if ok.any():
            jo = jv[ok]
            wf = fit_platt_batch(X[lo:i0, jo].T, Y[lo:i0, jo].T, w[jo])
            cl = np.c_[np.clip(wf[:, 0], *A_RANGE), np.clip(wf[:, 1], *B_RANGE)]
            st["clamped"] += int((cl != wf).any(1).sum())
            w[jo] = cl
            wk[ok] = cl
        st["identity_fits"] += int((~ok).sum())
        st["fitted_blocks"] += 1
        z = wk[:, :1].T * X[i0:i1, jv] + wk[:, 1:].T
        pc = 0.5 * (1 + np.tanh(0.5 * z))
        cal[i0:i1, jv] = pc
        active[i0:i1] = True
        if nfit >= n_min_sens:
            cal5[i0:i1, jv] = pc
        if ("GE", 4) in [COLS[j] for j in jv]:
            st["last_GE4"] = [round(float(v), 4) for v in w[CI[("GE", 4)]]]
    arms = {"raw": Praw, "cal_norepair": cal, "cal": repair(cal, Praw), "cal5k": repair(cal5, Praw), "cal_grade": repair_grade(cal)}
    ch = arms["cal"][:, 3:] != cal[:, 3:]
    st["repair_changed_share"] = round(float(ch[active].mean()), 6) if active.any() else 0.0
    st["repair_mean_abs"] = float(f"{np.abs(arms['cal'][active, 3:] - cal[active, 3:]).mean():.3g}") if active.any() else 0.0
    st["active_rows"] = int(active.sum())
    return arms, Y, V, active, st


def user_sums(arms, Y, V, active):
    """{arm: {event key: binary_sums}} and fine histograms {arm/event key: (n, pos, sp)} on the valid rows."""
    sums, hists = {}, {}
    for arm in ARMS:
        src = arms[arm.split("@")[0]]
        rows = active if arm.endswith("@active") else slice(None)
        sums[arm] = {}
        for j, c in enumerate(COLS):
            k = key_of(c)
            if not V[j]:
                sums[arm][k] = [0, 0.0, 0.0, 0.0, 0.0]
                continue
            p, y = src[rows, j], Y[rows, j]
            sums[arm][k] = binary_sums(p, y)
            hists[f"{arm}|{k}"] = hist_of(p, y)
    return sums, hists


def load_dump(key, user, dump_dir=None):
    z = np.load(Path(dump_dir or WORK / f"dump-{key}") / f"{user}.npz")
    return z["p"], z["grade"].astype(np.int64), z["d"].astype(np.int64), z["day"].astype(np.int64)


def sm120_rows(key):
    return {r["user_id"]: r for r in map(json.loads, (SM120_ROWS / f"sm120-{key}-rows.jsonl").read_text().splitlines()) if "error" not in r}


def dump_matches_sm120(rec, old, tol=1e-5):
    """Rows identical (review sha256, counts) and the float64 raw sums equal SM120's recorded model sums."""
    if rec["review_th_sha256"] != old["review_th_sha256"]:
        return False, float("inf")
    worst = 0.0
    for c in COLS:
        x = rec["raw_sums"][key_of(c)]
        y = old["grade"][c[0]]["model"] if c[1] == 0 else old["fast"][c[0]]["model"][str(c[1])]
        if x[0] != y[0] or x[4] != y[4]:
            return False, float("inf")
        for i in (1, 2, 3):
            worst = max(worst, abs(x[i] - y[i]) / max(abs(y[i]), 1e-9))
    return worst <= tol, worst


def recal(args):
    """Per user (resumable): dump vs SM120 row check, the rolling Platt arms, sums -> rows jsonl, histograms -> npz."""
    key = args.model
    index = Path(args.index or GPU / f"dump-{key}.jsonl")
    dump_dir = Path(args.dump_dir or WORK / f"dump-{key}")
    out = Path(args.out or ROWS / f"sm125-{key}-recal.jsonl")
    hd = Path(args.hist_dir) if args.hist_dir else WORK / f"recal-{key}.hist"
    out.parent.mkdir(parents=True, exist_ok=True)
    hd.mkdir(parents=True, exist_ok=True)
    idx = {r["user_id"]: r for r in map(json.loads, index.read_text().splitlines()) if "error" not in r}
    old = sm120_rows(key)
    done = {json.loads(line)["user_id"] for line in out.read_text().splitlines()} if out.exists() else set()
    users = [u for u in (args.users or pool_users()) if u in idx and u not in done]
    print(f"[sm125-recal] {MODELS[key]['name']}: {len(idx)} dumped, {len(done)} done, {len(users)} to do", flush=True)
    t0 = time.time()
    with out.open("a", encoding="utf-8") as f:
        for k, u in enumerate(users):
            if args.until and time.time() > args.until:
                print(f"[sm125-recal] stopping before the window: {k} of {len(users)} done this run", flush=True)
                return 3
            rec = idx[u]
            same, worst = dump_matches_sm120(rec, old[u]) if u in old else (False, float("inf"))
            P32, gr, d, day = load_dump(key, u, dump_dir)
            arms, Y, V, active, st = recal_user(P32, gr, d, day, rec["user_cap_ms"], n_min=args.n_min)
            sums, hists = user_sums(arms, Y, V, active)
            ks = sorted(hists)
            np.savez_compressed(hd / f"{u}.npz", keys=np.array(ks), n=np.array([hists[q][0] for q in ks]),
                                pos=np.array([hists[q][1] for q in ks]), sp=np.array([hists[q][2] for q in ks]))
            f.write(json.dumps({"user_id": u, "reviews": int(len(gr)), "review_th_sha256": rec["review_th_sha256"],
                                "dump_matches_sm120": bool(same), "dump_vs_sm120_max_rel": worst if math.isfinite(worst) else None,
                                "user_cap_ms": rec["user_cap_ms"], "stats": st, "sums": sums}) + "\n")
            f.flush()
            if (k + 1) % 50 == 0:
                print(f"[sm125-recal] {k + 1}/{len(users)} users, {time.time() - t0:.0f} s", flush=True)
    print(f"[sm125-recal] done {len(users)} users in {time.time() - t0:.0f} s -> {out}", flush=True)
    return 0


# ------------------------------------------------------------------ evaluation + report (CPU)
def evaluate(key, rows_path=None, hist_dir=None, users_only=None):
    rows_path = Path(rows_path or ROWS / f"sm125-{key}-recal.jsonl")
    hist_dir = Path(hist_dir or WORK / f"recal-{key}.hist")
    pool = users_only or pool_users()
    rows = {r["user_id"]: r for r in map(json.loads, rows_path.read_text().splitlines())}
    old = sm120_rows(key)
    users = [u for u in pool if u in rows and u in old and rows[u]["reviews"] > 0]
    res = {"model": key, "name": MODELS[key]["name"], "pool": len(pool), "users": len(users),
           "missing": [u for u in pool if u not in rows][:30],
           "dump_mismatch": [u for u in users if not rows[u]["dump_matches_sm120"]][:30],
           "dump_vs_sm120_max_rel": max((rows[u]["dump_vs_sm120_max_rel"] or 0.0) for u in users) if users else None}
    H = {}
    for u in users:
        z = np.load(hist_dir / f"{u}.npz")
        for i, q in enumerate(z["keys"]):
            q = str(q)
            if q not in H:
                H[q] = [np.zeros(HIST_BINS), np.zeros(HIST_BINS), np.zeros(HIST_BINS)]
            H[q][0] += z["n"][i]; H[q][1] += z["pos"][i]; H[q][2] += z["sp"][i]
    st = [rows[u]["stats"] for u in users]
    res["coverage"] = {"rows": int(sum(rows[u]["reviews"] for u in users)), "active_rows": int(sum(s["active_rows"] for s in st)),
                       "users_with_recalibration": int(sum(s["active_rows"] > 0 for s in st)),
                       "blocks": int(sum(s["blocks"] for s in st)), "fitted_blocks": int(sum(s["fitted_blocks"] for s in st)),
                       "identity_fits": int(sum(s["identity_fits"] for s in st)), "clamped_fits": int(sum(s["clamped"] for s in st)),
                       "causal_all": all(s["causal"] for s in st),
                       "repair_changed_share_mean": round(float(np.mean([s["repair_changed_share"] for s in st if s["active_rows"]] or [0])), 6),
                       "repair_mean_abs_mean": float(f"{np.mean([s.get('repair_mean_abs', 0.0) for s in st if s['active_rows']] or [0]):.3g}")}
    res["coverage"]["active_share"] = round(res["coverage"]["active_rows"] / max(res["coverage"]["rows"], 1), 4)
    boot = {}

    def bidx(n):
        if n not in boot:
            boot[n] = bootstrap_idx(n)
        return boot[n]

    def arr(get):
        return np.array([get(u) for u in users], float)

    def bar(d, ref_ll):
        rel = 100 * d["delta"] / ref_ll
        return {**d, "relative_pct_exact": round(rel, 4), "passes": bool(d["ci"][1] < 0 and rel <= -MIN_REL_PCT)}

    out = {}
    for c in COLS:
        k = key_of(c)
        o = old
        base_names = ["base_b"] if c[1] == 0 else ["base_a", "base_b"]
        B = {b: arr(lambda u, b=b: o[u]["grade"][c[0]][b] if c[1] == 0 else o[u]["fast"][c[0]][b][str(c[1])]) for b in base_names}
        A = {a: arr(lambda u, a=a: rows[u]["sums"][a][k]) for a in ARMS}
        n = A["raw"][:, 0]
        if n.sum() == 0:
            continue
        keep = n > 0
        ent = {"rows": int(n.sum()), "event_rate": round(float(A["raw"][:, 4].sum() / n.sum()), 5), "arms": {}, "vs": {}}
        for a in ARMS:
            na = A[a][:, 0]
            if na.sum() == 0:
                continue
            e = ece_auc_from_hist(*H[f"{a}|{k}"]) if f"{a}|{k}" in H else None
            ent["arms"][a] = {"rows": int(na.sum()), "logloss": round(float(A[a][:, 1].sum() / na.sum()), 6),
                              "brier": round(float(A[a][:, 2].sum() / na.sum()), 6), "mean_p": round(float(A[a][:, 3].sum() / na.sum()), 5),
                              "event_rate": round(float(A[a][:, 4].sum() / na.sum()), 5),
                              "ece": round(e["ece"], 5) if e else None, "auc": round(e["auc"], 5) if e else None,
                              "max_bin_gap": round(e["max_gap"], 5) if e else None}
        for b in base_names:
            bl = float(B[b][keep, 1].sum() / n[keep].sum())
            for a in ("raw", "cal", "cal5k", "cal_grade"):
                ent["vs"][f"{a}_vs_{b}"] = bar(_diff(A[a][keep, 1], B[b][keep, 1], n[keep], bidx(int(keep.sum()))), bl)
        rl = float(A["raw"][keep, 1].sum() / n[keep].sum())
        for a in ("cal", "cal5k", "cal_norepair", "cal_grade"):
            ent["vs"][f"{a}_vs_raw"] = bar(_diff(A[a][keep, 1], A["raw"][keep, 1], n[keep], bidx(int(keep.sum()))), rl)
        nact = A["raw@active"][:, 0]
        ka = nact > 0
        if ka.sum() >= 2:
            ral = float(A["raw@active"][ka, 1].sum() / nact[ka].sum())
            ent["vs"]["cal@active_vs_raw@active"] = bar(_diff(A["cal@active"][ka, 1], A["raw@active"][ka, 1], nact[ka], bidx(int(ka.sum()))), ral)
        m120 = arr(lambda u: (o[u]["grade"][c[0]]["model"] if c[1] == 0 else o[u]["fast"][c[0]]["model"][str(c[1])])[1])
        ent["raw_ll_matches_sm120"] = bool(abs(A["raw"][keep, 1].sum() - m120[keep].sum()) <= 1e-5 * A["raw"][keep, 1].sum())
        for a in ("raw", "cal", "cal5k", "cal_grade"):
            e = ent["arms"][a]["ece"]
            ent[f"reliable_{a}"] = bool(e is not None and e < 0.01 and all(ent["vs"][f"{a}_vs_{b}"]["passes"] for b in base_names))
        out[k] = ent
    res["events"] = out
    # pre-registered rules (design.md)
    main_cells = [key_of((sn, s)) for sn in ("GE", "Rec", "E") for s in S_MAIN]
    g1 = all(out[k]["reliable_cal"] for k in main_cells if k in out)
    g2 = out["fast/GE/4"]["vs"]["cal_vs_raw"]["passes"] if "fast/GE/4" in out else False
    harm = [k for k in main_cells if k in out and out[k]["vs"]["cal_vs_raw"]["ci"][0] > 0]
    g4 = all(out[key_of((sn, 0))]["reliable_raw"] for sn in ("GE", "E"))
    res["rules"] = {"G1_all_2_10s_cells_reliable_after_recalibration": g1,
                    "G2_GE4_recalibrated_better_than_raw": g2,
                    "G3_no_2_10s_cell_worse_than_raw": not harm, "G3_worse_cells": harm,
                    "G4_grade_only_GE_E_reliable": g4,
                    "go": bool(g1 and g2 and not harm and g4),
                    "cells_reliable_raw": sum(out[k]["reliable_raw"] for k in main_cells if k in out),
                    "cells_reliable_cal": sum(out[k]["reliable_cal"] for k in main_cells if k in out),
                    "cells_reliable_cal5k": sum(out[k]["reliable_cal5k"] for k in main_cells if k in out),
                    "cells": len(main_cells),
                    "GE4_reliable_raw": out["fast/GE/4"]["reliable_raw"] if "fast/GE/4" in out else None,
                    "GE4_reliable_cal": out["fast/GE/4"]["reliable_cal"] if "fast/GE/4" in out else None}
    res["reliable_S"] = {f"{a}/{sn}": [s for s in S_GRID if out.get(key_of((sn, s)), {}).get(f"reliable_{a}")]
                         for a in ("raw", "cal", "cal5k") for sn in SETS}
    return res


def fmt(d):
    return f"{d['delta']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}] ({d['relative_pct_exact']:+.2f}%)"


def s_list(lst):
    return ", ".join(f"{s} s" for s in lst) if lst else "none"


def write_md(res, path):
    L = ["# Rolling recalibration on the 732 set (SM125): result", "",
         "**What answer time means.** The answer time is the whole time on the card, front and back together: from the moment the question is shown to the moment a grade button is pressed, capped at the preset's maximum answer seconds. Every \"within S s\" below means answered within S s, reading the back included.", "",
         f"Scoring only (no training), CPU recalibration of per-review outputs re-dumped on the GPU with fast Good/Easy prediction (SM120)'s code, on the 732 set (723 users, show-time inputs, leak-free, SM117's time-labelled test reviews). Pre-registered in `design.md`. Scheme: per user, Platt scaling per event refitted every {ROLL_DAYS} days on the user's latest {ROLL_ROWS:,} earlier reviews (earlier days only); raw model until {N_MIN:,} earlier reviews; repair (non-decreasing in S, Easy within Good or Easy within recalled, each timed event at most the raw P(set)). Grade-only P(set) is not recalibrated. Reliable = ECE (20 equal-mass bins) below 0.01 and LogLoss below every baseline of SM120 (not recalibrated), each difference with the 95% CI (2,000 user bootstraps, seed 0, review-weighted) below 0 and at least 0.05%.", ""]
    L += ["## Verdict per model (pre-registered rules)", "",
          "| model | go | 2-10 s cells reliable raw / recalibrated (of 21) | Good or Easy within 4 s reliable raw / recalibrated | G1 all 21 | G2 4 s better than raw | G3 no cell worse | G4 P(Good or Easy), P(Easy) | rows recalibrated |",
          "|---|---|---|---|---|---|---|---|---|"]
    for key, r in res["models"].items():
        if "rules" not in r:
            L.append(f"| {r['name']} | not scored | | | | | | | |")
            continue
        g = r["rules"]
        L.append(f"| {r['name']} | {'go' if g['go'] else 'no go'} | {g['cells_reliable_raw']} / {g['cells_reliable_cal']} | {'yes' if g['GE4_reliable_raw'] else 'no'} / {'yes' if g['GE4_reliable_cal'] else 'no'} | "
                 f"{'yes' if g['G1_all_2_10s_cells_reliable_after_recalibration'] else 'no'} | {'yes' if g['G2_GE4_recalibrated_better_than_raw'] else 'no'} | "
                 f"{'yes' if g['G3_no_2_10s_cell_worse_than_raw'] else 'no: ' + ', '.join(g['G3_worse_cells'])} | {'yes' if g['G4_grade_only_GE_E_reliable'] else 'no'} | {100 * r['coverage']['active_share']:.1f}% |")
    L += ["", "## Good or Easy within 4 s (answered within 4 s, reading the back included)", "",
          "| model | ECE raw / recalibrated | LogLoss raw / recalibrated | recalibrated vs raw | recalibrated vs (a) | recalibrated vs (b) | raw vs (a) | raw vs (b) |", "|---|---|---|---|---|---|---|---|"]
    for key, r in res["models"].items():
        e = r.get("events", {}).get("fast/GE/4")
        if not e:
            continue
        a = e["arms"]
        L.append(f"| {r['name']} | {a['raw']['ece']:.4f} / {a['cal']['ece']:.4f} | {a['raw']['logloss']:.4f} / {a['cal']['logloss']:.4f} | {fmt(e['vs']['cal_vs_raw'])} | "
                 f"{fmt(e['vs']['cal_vs_base_a'])} | {fmt(e['vs']['cal_vs_base_b'])} | {fmt(e['vs']['raw_vs_base_a'])} | {fmt(e['vs']['raw_vs_base_b'])} |")
    L += ["", "## Where each timed event is reliable (values of S), raw -> recalibrated", "",
          "| model | Good or Easy within S s | recalled within S s | Easy within S s |", "|---|---|---|---|"]
    for key, r in res["models"].items():
        if "reliable_S" not in r:
            continue
        rs = r["reliable_S"]
        L.append(f"| {r['name']} | {s_list(rs['raw/GE'])} -> {s_list(rs['cal/GE'])} | {s_list(rs['raw/Rec'])} -> {s_list(rs['cal/Rec'])} | {s_list(rs['raw/E'])} -> {s_list(rs['cal/E'])} |")
    L += ["", "## ECE per S, raw -> recalibrated (Good or Easy within S s)", "", "| model | " + " | ".join(f"{s} s" for s in S_GRID) + " |", "|---|" + "---|" * len(S_GRID)]
    for key, r in res["models"].items():
        ev = r.get("events", {})
        if not ev:
            continue
        L.append(f"| {r['name']} | " + " | ".join(f"{ev[key_of(('GE', s))]['arms']['raw']['ece']:.4f} -> {ev[key_of(('GE', s))]['arms']['cal']['ece']:.4f}" if key_of(('GE', s)) in ev else "" for s in S_GRID) + " |")
    L += ["", "## Grade only: P(Good or Easy) and P(Easy) (shipped raw; diagnostic: recalibrated too)", "",
          "| model | event | ECE raw | raw vs (b) | reliable raw | ECE recalibrated (diagnostic) | recalibrated vs raw (diagnostic) |", "|---|---|---|---|---|---|---|"]
    for key, r in res["models"].items():
        for sn in ("GE", "E"):
            e = r.get("events", {}).get(key_of((sn, 0)))
            if not e:
                continue
            L.append(f"| {r['name']} | {SET_NAMES[sn]} | {e['arms']['raw']['ece']:.4f} | {fmt(e['vs']['raw_vs_base_b'])} | {'yes' if e['reliable_raw'] else 'no'} | {e['arms']['cal_grade']['ece']:.4f} | {fmt(e['vs']['cal_grade_vs_raw'])} |")
    L += ["", "## Coverage and fits", "", "| model | users | rows | recalibrated rows | users with recalibration | refits | identity fits | clamped fits | repair changed | dump equals SM120 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for key, r in res["models"].items():
        if "coverage" not in r:
            continue
        c = r["coverage"]
        L.append(f"| {r['name']} | {r['users']} | {c['rows']:,} | {c['active_rows']:,} ({100 * c['active_share']:.1f}%) | {c['users_with_recalibration']} | {c['fitted_blocks']:,} | {c['identity_fits']:,} | {c['clamped_fits']:,} | {100 * c['repair_changed_share_mean']:.2f}% | {'yes' if not r['dump_mismatch'] else 'no (' + str(len(r['dump_mismatch'])) + ' users)'} |")
    L += ["", "Every per-S number, the sensitivity arm (5,000-review minimum), the arm before the repair and the recalibrated-rows-only comparison are in `sm125-result.json`."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def report(args):
    res = {"design": {"S": list(S_GRID), "S_main": list(S_MAIN), "roll_days": ROLL_DAYS, "roll_rows": ROLL_ROWS, "n_min": N_MIN,
                      "n_min_sensitivity": N_MIN_SENS, "min_class": MIN_CLASS, "a_range": A_RANGE, "b_range": B_RANGE,
                      "bootstrap": 2000, "seed": 0, "min_rel_pct": MIN_REL_PCT, "arms": ARM_NAMES}, "models": {}}
    for key in args.models:
        p = ROWS / f"sm125-{key}-recal.jsonl"
        if not p.exists():
            res["models"][key] = {"name": MODELS[key]["name"]}
            continue
        print(f"[sm125-report] {key}", flush=True)
        res["models"][key] = evaluate(key)
    atomic_json(OUT / "sm125-result.json", res)
    write_md(res, OUT / "sm125-result.md")
    gos = [(k, r.get("rules", {}).get("go")) for k, r in res["models"].items()]
    ge4 = [(k, r.get("rules", {}).get("GE4_reliable_raw"), r.get("rules", {}).get("GE4_reliable_cal")) for k, r in res["models"].items()]
    print(f"[sm125-report] {len(gos)} models: go {gos}; Good or Easy within 4 s reliable raw/recalibrated {ge4}")
    return 0


# ------------------------------------------------------------------ CPU proofs
def check(args):
    """CPU proofs (no GPU): P1 dump reproduces SM120's recorded per-user model sums (CPU vs GPU) on small users;
    P2 batched Platt equals scipy's optimum; P3 causality (labels of later days cannot move earlier predictions);
    P4 identity (no history -> raw exactly); P5 repair invariants; P6 a synthetic miscalibrated user is repaired
    (slope / intercept recovered, ECE and LogLoss drop); P7 end-to-end recal + evaluate on the check users; P8 timing
    of a large synthetic user (CPU cost of the full run)."""
    key = args.model
    work = WORK / "check"
    work.mkdir(parents=True, exist_ok=True)
    old = sm120_rows(key)
    pool = set(pool_users())
    small = sorted((r["reviews"], u) for u, r in old.items() if u in pool and r["reviews"] >= 1500)
    users = [u for _, u in small[:3]] + [u for n, u in small if n >= 6000][:1]
    checks = {}
    # P1 dump on CPU
    idx_path, dd = work / f"dump-{key}.jsonl", work / f"dump-{key}"
    if idx_path.exists():
        idx_path.unlink()
    uf = work / "users.json"
    uf.write_text(json.dumps({"users": users}))
    t0 = time.time()
    dump(Namespace(model=key, head_dir="", users_file=str(uf), out_dir=str(dd), index=str(idx_path), device="cpu", workers=0))
    dump_s = round(time.time() - t0, 1)
    idx = {r["user_id"]: r for r in map(json.loads, idx_path.read_text().splitlines())}
    per = {}
    for u in users:
        ok, worst = dump_matches_sm120(idx[u], old[u], tol=1e-4)
        P32, gr, d, day = load_dump(key, u, dd)
        Y = labels(gr, d)
        V = valid_cols(idx[u]["user_cap_ms"])
        f32 = max(abs(binary_sums(P32[:, j].astype(np.float64), Y[:, j])[1] - idx[u]["raw_sums"][key_of(c)][1]) / max(idx[u]["raw_sums"][key_of(c)][1], 1e-9)
                  for j, c in enumerate(COLS) if V[j])
        nest = bool((P32[:, CI[("E", 4)]] <= P32[:, CI[("GE", 4)]] + 1e-6).all() and (P32[:, CI[("GE", 4)]] <= P32[:, CI[("Rec", 4)]] + 1e-6).all()
                    and (P32[:, CI[("GE", 30)]] <= P32[:, CI[("GE", 0)]] + 1e-6).all())
        per[str(u)] = {"rows": int(len(gr)), "same_rows_and_sums": ok, "max_rel_vs_sm120_gpu": worst, "float32_dump_max_rel": f32,
                       "days": idx[u]["days"], "day_sorted": idx[u]["day_sorted"], "raw_nested": nest}
    checks["P1_dump_reproduces_sm120"] = {"pass": all(v["same_rows_and_sums"] and v["float32_dump_max_rel"] < 1e-5 and v["day_sorted"] and v["raw_nested"] for v in per.values()),
                                          "tolerance_vs_sm120": 1e-4, "tolerance_float32": 1e-5, "users": per, "dump_seconds_cpu": dump_s}
    # P2 Platt vs scipy
    from scipy.optimize import minimize
    rng = np.random.default_rng(0)
    worst = 0.0
    for t in range(6):
        n = int(rng.integers(500, 20000))
        x = rng.normal(0, 1.5, n)
        y = (rng.random(n) < 1 / (1 + np.exp(-(rng.uniform(0.5, 2) * x + rng.normal(0, 1))))).astype(float)
        w = fit_platt_batch(x[None], y[None], np.array([[1.0, 0.0]]))[0]
        f = lambda v: np.mean(np.logaddexp(0, v[0] * x + v[1]) - y * (v[0] * x + v[1]))  # noqa: E731
        ws = minimize(f, np.array([1.0, 0.0]), method="BFGS", options={"gtol": 1e-10}).x
        worst = max(worst, float(np.abs(w - ws).max()))
    checks["P2_platt_equals_scipy"] = {"pass": worst < 1e-4, "max_abs_param_diff": worst}
    # P3 causality: flip every label from day D on; recalibrated p before D unchanged (bitwise)
    u = users[-1]
    P32, gr, d, day = load_dump(key, u, dd)
    cap = idx[u]["user_cap_ms"]
    A0, *_ = recal_user(P32, gr, d, day, cap)
    D = int(np.quantile(day, 0.7))
    gr2, d2 = gr.copy(), d.copy()
    late = day >= D
    gr2[late] = 3 - gr2[late]
    d2[late] = 60000 - np.minimum(d2[late], 59000)
    A1, *_ = recal_user(P32, gr2, d2, day, cap)
    same_before = all(np.array_equal(A0[a][~late], A1[a][~late]) for a in A0)
    moved_after = not np.array_equal(A0["cal"][late], A1["cal"][late])
    checks["P3_causal"] = {"pass": bool(same_before and moved_after), "user": u, "day_D": D, "rows_before": int((~late).sum()),
                           "earlier_predictions_bitwise_unchanged": bool(same_before), "later_predictions_moved": bool(moved_after)}
    # P4 identity without history
    A2, _, _, act2, _ = recal_user(P32, gr, d, day, cap, n_min=10 ** 9, n_min_sens=10 ** 9)
    checks["P4_identity_without_history"] = {"pass": bool(not act2.any() and np.array_equal(A2["cal_norepair"], P32.astype(np.float64))
                                                          and np.array_equal(A2["cal"], repair(P32.astype(np.float64), P32.astype(np.float64)))
                                                          and float(np.abs(A2["cal"] - P32).max()) < 1e-6),
                                             "max_abs_change_by_repair_of_raw": float(np.abs(A2["cal"] - P32).max())}
    # P5 repair invariants on the recalibrated arm of every check user
    inv = True
    for u in users:
        P32, gr, d, day = load_dump(key, u, dd)
        A, *_ = recal_user(P32, gr, d, day, idx[u]["user_cap_ms"])
        for a in ("cal", "cal5k", "cal_grade"):
            Q = A[a]
            for sn in SETS:
                j = [CI[(sn, s)] for s in S_GRID]
                inv &= bool((np.diff(Q[:, j], axis=1) >= 0).all() and (Q[:, j] <= Q[:, [CI[(sn, 0)]]] + 1e-12).all())
            for lo, hi in (("E", "GE"), ("GE", "Rec")):
                inv &= bool(all((Q[:, CI[(lo, s)]] <= Q[:, CI[(hi, s)]]).all() for s in (0,) + S_GRID))
    checks["P5_repair_invariants"] = {"pass": inv, "note": "non-decreasing in S, nested E <= GE <= Rec, timed <= P(set)"}
    # P6 synthetic miscalibrated user: y ~ sigmoid(1.6 logit p + 0.4)
    n, days = 60000, 400
    day_s = np.sort(rng.integers(0, days, n))
    p = 1 / (1 + np.exp(-rng.normal(-1, 1.3, n)))
    Ps = np.repeat(p[:, None], len(COLS), 1).astype(np.float32)
    ytrue = rng.random(n) < 1 / (1 + np.exp(-(1.6 * logit(p) + 0.4)))
    X, Yv = logit(Ps[:, 0].astype(np.float64)), ytrue.astype(float)
    blocks = blocks_of(day_s)
    cal = Ps[:, 0].astype(np.float64).copy()
    last = None
    w = np.array([[1.0, 0.0]])
    for i0, i1, lo in blocks:
        if i0 - lo < N_MIN:
            continue
        w = fit_platt_batch(X[None, lo:i0], Yv[None, lo:i0], w)
        cal[i0:i1] = 1 / (1 + np.exp(-(w[0, 0] * X[i0:i1] + w[0, 1])))
        last = w[0]
    e_raw = ece_auc_from_hist(*hist_of(Ps[:, 0].astype(np.float64), Yv))["ece"]
    e_cal = ece_auc_from_hist(*hist_of(cal, Yv))["ece"]
    ll_raw, ll_cal = binary_sums(Ps[:, 0].astype(np.float64), Yv)[1] / n, binary_sums(cal, Yv)[1] / n
    checks["P6_synthetic_recovery"] = {"pass": bool(abs(last[0] - 1.6) < 0.08 and abs(last[1] - 0.4) < 0.08 and e_cal < e_raw / 3 and ll_cal < ll_raw),
                                       "last_fit": [round(float(v), 4) for v in last], "true": [1.6, 0.4], "ece_raw": round(e_raw, 5),
                                       "ece_cal": round(e_cal, 5), "logloss_raw": round(ll_raw, 5), "logloss_cal": round(ll_cal, 5)}
    # P7 end-to-end recal + evaluate on the check users (bootstrap over 4 users: pipeline only, not a result)
    rp = work / f"recal-{key}.jsonl"
    if rp.exists():
        rp.unlink()
    rc = recal(Namespace(model=key, index=str(idx_path), dump_dir=str(dd), out=str(rp), hist_dir=str(work / f"recal-{key}.hist"),
                         users=users, until=0, n_min=N_MIN))
    ev = evaluate(key, rows_path=rp, hist_dir=work / f"recal-{key}.hist", users_only=users)
    ge4 = ev["events"]["fast/GE/4"]
    checks["P7_end_to_end"] = {"pass": bool(rc == 0 and ev["users"] == len(users) and not ev["dump_mismatch"] and ev["coverage"]["causal_all"]
                                            and all(e["raw_ll_matches_sm120"] for e in ev["events"].values())),
                               "users": ev["users"], "coverage": ev["coverage"], "rules": ev["rules"],
                               "GE4": {"ece_raw": ge4["arms"]["raw"]["ece"], "ece_cal": ge4["arms"]["cal"]["ece"],
                                       "ll_raw": ge4["arms"]["raw"]["logloss"], "ll_cal": ge4["arms"]["cal"]["logloss"],
                                       "cal_vs_raw": ge4["vs"]["cal_vs_raw"]}}
    # P8 timing: a large synthetic user (200,000 rows over 1,500 days, all 33 events)
    n, days = 200000, 1500
    day_s = np.sort(rng.integers(0, days, n))
    base = 1 / (1 + np.exp(-rng.normal(0, 1.5, (n, 1))))
    Pb = np.clip(base * np.linspace(1, 0.3, len(COLS))[None], 1e-4, 1 - 1e-4).astype(np.float32)
    grs = rng.integers(0, 4, n)
    ds = rng.integers(500, 30000, n)
    t0 = time.time()
    A, Y, V, act, st = recal_user(Pb, grs, ds, day_s, 60000)
    user_sums(A, Y, V, act)
    sec = time.time() - t0
    tot = sum(r["reviews"] for u, r in old.items() if u in pool)
    est_h = sec / n * tot / 3600
    checks["P8_timing"] = {"pass": True, "rows": n, "seconds": round(sec, 1), "blocks": st["blocks"], "rows_per_second": round(n / sec),
                           "model_rows_732": tot, "estimated_cpu_hours_per_model": round(est_h, 2)}
    res = {"model": MODELS[key]["name"], "device": "cpu", "users": users, "pass": all(c["pass"] for c in checks.values()), "checks": checks}
    atomic_json(Path(args.out), res)
    print(f"[sm125-check] {'PASS' if res['pass'] else 'FAIL'}: " + ", ".join(f"{k} {v['pass']}" for k, v in checks.items()))
    return 0 if res["pass"] else 1


def main():
    p0 = ArgumentParser(description=__doc__.split("\n")[0])
    sub = p0.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dump")
    p.add_argument("--model", required=True, choices=ORDER)
    p.add_argument("--head-dir", default="")
    p.add_argument("--users-file", default=str(U732))
    p.add_argument("--out-dir", default="")
    p.add_argument("--index", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=3)
    p = sub.add_parser("recal")
    p.add_argument("--model", required=True, choices=ORDER)
    p.add_argument("--index", default="")
    p.add_argument("--dump-dir", default="")
    p.add_argument("--out", default="")
    p.add_argument("--hist-dir", default=None)
    p.add_argument("--users", type=int, nargs="*", default=None)
    p.add_argument("--until", type=float, default=0, help="epoch seconds: stop (exit 3) before starting a user after this")
    p.add_argument("--n-min", type=int, default=N_MIN)
    p = sub.add_parser("check")
    p.add_argument("--model", default="sm106", choices=ORDER)
    p.add_argument("--out", default=str(OUT / "check-cpu.json"))
    p = sub.add_parser("report")
    p.add_argument("--models", nargs="*", default=list(MAIN_MODELS))
    a = p0.parse_args()
    if a.cmd == "dump":
        a.out_dir = a.out_dir or str(WORK / f"dump-{a.model}")
        a.index = a.index or str(GPU / f"dump-{a.model}.jsonl")
    return {"dump": dump, "recal": recal, "check": check, "report": report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
