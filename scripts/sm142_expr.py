#!/usr/bin/env python
"""How expressive the answer-time predictions are (SM142). Pre-registered in artifacts/small-model/sm142/design.md.

Does the answer-time head give card-specific predicted time distributions (location and shape) that track the real
differences between one user's cards, or only a few templates around the user's average? Description, not a gate.

Answer time = front + back (question shown -> grade pressed, capped at the preset's maximum answer seconds): "within
S s" always means answered within S s, reading the back included.

The forward pass, rows, heads and bins are SM117's (scripts/sm117_head.py, imported, not edited); head directories are
SM120's table (scripts/sm120_fast.py HEAD_DIRS).

  dump     --model KEY [--users-file F] [--device cuda]   per review: card, grade, duration, P(grade), the head's and the
                                                           ablation's 4 x 16 distributions -> work/dump-KEY/<user>.npz,
                                                           index gpu/dump-KEY.jsonl (sums checked vs the recorded scoring)
  verify   --model KEY                                    dump index vs the model's recorded scoring rows (exit 1 if < 700)
  analyse  --model KEY [--users ...]                      CPU, per user -> rows/sm142-KEY-users.jsonl + work/sample-KEY/
  report                                                  all analysed models -> sm142-result.{md,json}
  check    [--out F]                                      CPU proof (2 small users, CPU, no GPU) -> check-cpu.json
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

from scripts.sm117_head import (MODELS, ORDER, U732, Frozen, build_model, verified_master, load_head, ScoreBatches,  # noqa: E402
                                cdf_at, pool_users, bootstrap_idx)
from scripts.sm117_timelib import (BINS, LOG_EDGES, EXPLICIT_DIM, RunningMedian, first, label_db, test_reviews,  # noqa: E402
                                   user_sums, median_log_s, time_user_cap)
from scripts.sm120_fast import HEAD_DIRS  # noqa: E402

OUT = ROOT / "artifacts/small-model/sm142"
WORK = OUT / "work"
GPU = OUT / "gpu"
ROWS = OUT / "rows"
MAIN = Path(r"C:\Users\jonat\workspace\anki-rwkv")
RUN = MAIN / "artifacts" / "full-run" / "run-8k-20261013"
WTD = Path("C:/Users/jonat/workspace/wt")
MODELS_142 = ("sm130th", "sm130io", "sm113", "sm118", "sm99")
RECORDED = {"sm113": WTD / "sm119/artifacts/small-model/sm119/gpu/sm119-sm113-rows.jsonl",
            "sm118": WTD / "sm119/artifacts/small-model/sm119/gpu/sm119-sm118-rows.jsonl",
            "sm99": WTD / "sm117/artifacts/small-model/sm117/gpu/sm117-sm99-rows.jsonl",
            "sm130th": WTD / "sm138/artifacts/small-model/sm138/gpu/sm138-sm130th-rows.jsonl",
            "sm130io": WTD / "sm138/artifacts/small-model/sm138/gpu/sm138-sm130io-rows.jsonl"}
S_AUC = (2, 3, 4, 6, 8, 10)
QS = (0.10, 0.25, 0.50, 0.75, 0.90)
LN60 = math.log(60.0)
LN_LO = math.log(0.25)
LO = np.r_[LN_LO, LOG_EDGES]                     # bin k's lower ln-second edge
HI = np.r_[LOG_EDGES, LOG_EDGES[-1]]             # bin 15 (60 s or more) = a point mass at 60 s (median_log_s rule)
U_GRID = np.round(np.arange(-2.5, 2.5 + 1e-9, 0.05), 10)              # location-normalised CDF offsets (101)
T_GRID = np.arange(math.log(0.5), LN60 + 1e-9, 0.05)                  # location-including ln-time grid
DELTAS = (0.1, 0.2)
SAMPLE_PER_USER = 200
MIN_AUC_CLASS = 10
MIN_CARDS = 20
MIN_SLOPE_ROWS = 200
MIN_ROWS_SHAPE = 50
MIN_BUCKET = 20
BUCKETS = ("1", "2-3", "4-7", "8-15", "16-31", "32+")
BUCKET_EDGES = (2, 4, 8, 16, 32)
PREDS = ("head", "abl", "a", "b", "c")
GRADE_NAMES = ("Again", "Hard", "Good", "Easy")
NAME = "Answer-time expressiveness (SM142)"


def data_path():
    return MAIN / json.loads((RUN / "run.json").read_text())["dataset_path"]


def atomic_npz(path, **arrays):
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


# ------------------------------------------------------------------ dump (GPU pass; CPU for the check)
def dump(args):
    """Every SM117 time-labelled test review of the users (fp32, full history, show-time inputs, leak-free), exactly as
    SM117's score: per review the review index, card id, grade, duration, elapsed days, P(grade), the head's 4 x 16
    distribution (float32) and the ablation's (float16) -> <out-dir>/<user>.npz; one index record per user with the
    float64 true-grade sums (SM117 user_sums) of both heads, checked against the recorded scoring rows by `verify`."""
    from scripts import small_model_heldout as smh  # noqa: F401  first
    import torch
    from scripts import compress_common as cc
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("HELDOUT_GPU_MEMORY_FRACTION", "0.8")))
    else:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    info = verified_master(args.model)
    hd = Path(args.head_dir or HEAD_DIRS[args.model])
    trunk_path = hd / "master.pth"
    trunk = torch.load(trunk_path, map_location="cpu", weights_only=True)
    released = torch.load(info["master"], map_location="cpu", weights_only=True)
    if set(trunk) != set(released) or not all(torch.equal(trunk[k].float(), released[k].float()) for k in released):
        raise SystemExit(f"{trunk_path} is not the released model of {args.model}")
    frozen = Frozen(build_model(args.model, trunk, device), MODELS[args.model]["kind"])
    head = load_head(hd / "head.pth", frozen.inputs).to(device)
    abl = load_head(hd / "ablation-head.pth", EXPLICIT_DIM).to(device)
    users = json.loads(Path(args.users_file).read_text())["users"]
    out_dir = Path(args.out_dir or WORK / f"dump-{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    index = Path(args.index or GPU / f"dump-{args.model}.jsonl")
    index.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["user_id"] for line in index.read_text().splitlines() if '"error"' not in line} if index.exists() else set()
    todo = [u for u in users if u not in done]
    data = cc.data_path()
    loader = torch.utils.data.DataLoader(ScoreBatches(todo, data, cc.run_info()["review_counts"]), batch_size=1,
                                         num_workers=args.workers, collate_fn=first, prefetch_factor=1 if args.workers else None)
    env = label_db()
    started = time.perf_counter()
    nbytes = 0
    print(f"[sm142-dump] {MODELS[args.model]['name']}: {len(users)} users, {len(todo)} to dump", flush=True)
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
                    pa = torch.softmax(abl(x[:, frozen.frozen_dim:]).float().view(-1, 4, BINS), -1).double().cpu().numpy()
                del gb, o
            except torch.OutOfMemoryError as oom:
                frozen.clear()
                torch.cuda.empty_cache()
                f.write(json.dumps({"user_id": user, "error": f"OOM: {oom}"[:300]}) + "\n"); f.flush()
                continue
            except RuntimeError as err:   # as SM117/SM125: recorded, the user is left out (pass 2 retries)
                frozen.clear()
                if "out of memory" in str(err):
                    torch.cuda.empty_cache()
                print(f"[sm142-dump] user {user}: {err}", flush=True)
                f.write(json.dumps({"user_id": user, "error": f"{err}"[:300]}) + "\n"); f.flush()
                continue
            ti = ths - 1
            d = frame["duration"].to_numpy(np.int64)
            gr = frame["rating"].to_numpy(np.int64)[ti] - 1
            ed = frame["elapsed_days"].to_numpy(np.int64)[ti]
            card = frame["card_id"].to_numpy(np.int64)[ti]
            t = {"grade": gr.astype(np.int8), "lower": lower[ti].astype(np.int8), "cens": cens[ti].astype(np.int8),
                 "logd": np.log(np.maximum(d[ti], 1) / 1000.0), "uncensored": (d[ti] < 60000) & (cens[ti] == 0)}
            rows_idx = np.arange(len(ths))
            sums = {"model": user_sums(pt[rows_idx, gr], t), "ablation": user_sums(pa[rows_idx, gr], t)}
            atomic_npz(out_dir / f"{user}.npz", th=ths.astype(np.int32), card=card, grade=gr.astype(np.int8),
                       d=d[ti].astype(np.int32), ed=ed.astype(np.int32), pg=pg.astype(np.float32),
                       pt=pt.astype(np.float32), pa=pa.astype(np.float16))
            nbytes += (out_dir / f"{user}.npz").stat().st_size
            rec = {"user_id": user, "reviews": int(len(ths)), "review_th_sha256": hashlib.sha256(np.asarray(ths, np.int64).tobytes()).hexdigest(),
                   "model": args.model, "trunk": str(trunk_path), "head_dir": str(hd), "precision": "fp32", "device": args.device,
                   "user_cap_ms": int(cap), "later_day_reviews": int((ed >= 1).sum()), "sums": sums}
            f.write(json.dumps(rec) + "\n"); f.flush()
            if (k + 1) % 50 == 0:
                print(f"[sm142-dump] {k + 1}/{len(todo)} users, {time.perf_counter() - started:.0f} s, {nbytes / 1e9:.2f} GB", flush=True)
    finally:
        f.close()
        env.close()
    print(f"[sm142-dump] done {len(todo)} users in {time.perf_counter() - started:.0f} s, {nbytes / 1e9:.2f} GB -> {out_dir}", flush=True)
    return 0


def sums_match(rec, old, tol=1e-6):
    """A dump index record vs the recorded scoring row of the same user: same reviews, counts exact, LogLoss and |ln
    error| sums of the head and the ablation (true grade, all rows) within tol relative."""
    if rec.get("review_th_sha256") != old.get("review_th_sha256"):
        return False, "review sha256 differs"
    worst = 0.0
    for mine, theirs in (("model", "model"), ("ablation", "ablation")):
        a, b = rec["sums"][mine], old["time"][theirs]["cond"]
        if a["n"] != b["n"] or a["n_unc"] != b["n_unc"]:
            return False, f"{mine} counts differ"
        for fld in ("ll", "ae"):
            worst = max(worst, abs(a[fld] - b[fld]) / max(abs(b[fld]), 1e-12))
        for g in range(1, 5):
            if a[f"g{g}"][0] != b[f"g{g}"][0]:
                return False, f"{mine} grade {g} counts differ"
    return worst <= tol, f"worst relative {worst:.2e}"


def load_jsonl(path):
    p = Path(path)
    return {r["user_id"]: r for r in map(json.loads, p.read_text(encoding="utf-8").splitlines()) if "error" not in r} if p.exists() else {}


def verify(args):
    idx = load_jsonl(args.index or GPU / f"dump-{args.model}.jsonl")
    old = load_jsonl(RECORDED[args.model])
    bad, ok, worst = [], 0, []
    for u, rec in idx.items():
        if u not in old:
            bad.append([u, "no recorded row"]); continue
        good, why = sums_match(rec, old[u], tol=args.tol)
        worst.append(why)
        if good:
            ok += 1
        else:
            bad.append([u, why])
    res = {"model": args.model, "dumped": len(idx), "recorded": len(old), "match": ok, "bad": bad, "tol": args.tol,
           "worst": max((float(w.split()[-1]) for w in worst if w.startswith("worst")), default=None)}
    out = Path(args.out or GPU / f"verify-{args.model}.json")
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"verify {args.model}: {ok} of {len(idx)} dumped users match the recorded rows (worst {res['worst']}), {len(bad)} bad")
    return 0 if ok >= args.min_users else 1


# ------------------------------------------------------------------ baselines (CPU, model-independent, cached)
def log_time(d, cap):
    return np.log(np.minimum(np.maximum(d, 1), min(cap, 60000)) / 1000.0)


def baselines_full(card, rating, duration):
    """Per review i of the history (before review i is seen): (a) the card's running median of earlier timed log times,
    (b) the card's last timed log time, (c) the user's running median of earlier timed log times of review i's grade,
    (c_all) of every grade, cnt = the number of earlier reviews of the card (all). NaN = none yet."""
    cap = time_user_cap(duration)
    y = log_time(np.asarray(duration, np.int64), cap)
    n = len(duration)
    a = np.full(n, np.nan); b = np.full(n, np.nan); c = np.full(n, np.nan); call = np.full(n, np.nan)
    cnt = np.zeros(n, np.int32)
    cards = {}
    per = {g: RunningMedian() for g in (1, 2, 3, 4)}
    every = RunningMedian()
    for i in range(n):
        cid = int(card[i]); r = int(rating[i])
        st = cards.get(cid)
        if st is None:
            st = [RunningMedian(), None, 0]
            cards[cid] = st
        m = st[0].median()
        if m is not None:
            a[i] = m
            b[i] = st[1]
        cnt[i] = st[2]
        mg = per[r].median() if r in per else None
        me = every.median()
        if me is not None:
            call[i] = me
        c[i] = mg if mg is not None else (me if me is not None else np.nan)
        st[2] += 1
        if duration[i] > 0:
            x = float(y[i])
            st[0].push(x); st[1] = x
            if r in per:
                per[r].push(x)
            every.push(x)
    return a, b, c, call, cnt, cap


def user_base(user):
    """Cached baselines of a user (work/base/<user>.npz, every review of the history)."""
    p = WORK / "base" / f"{user}.npz"
    if p.exists():
        z = np.load(p)
        return {k: z[k] for k in z.files}
    import pandas as pd
    fr = pd.read_parquet(Path(data_path()) / "revlogs" / f"user_id={user}", columns=["card_id", "rating", "duration"])
    a, b, c, call, cnt, cap = baselines_full(fr["card_id"].to_numpy(np.int64), fr["rating"].to_numpy(np.int64),
                                             fr["duration"].to_numpy(np.int64))
    p.parent.mkdir(parents=True, exist_ok=True)
    out = {"a": a.astype(np.float32), "b": b.astype(np.float32), "c": c.astype(np.float32), "call": call.astype(np.float32),
           "cnt": cnt, "cap": np.array(cap)}
    atomic_npz(p, **out)
    return out


# ------------------------------------------------------------------ distribution helpers
def quantiles(prob, qs=QS):
    """[N, len(qs)] ln-second quantiles of [N, 16] bin distributions, linear in ln time inside the crossing bin (bin 0
    from 0.25 s, bin 15 = 60 s), the median_log_s rule for every q."""
    cdf = np.cumsum(prob, axis=1)
    out = np.empty((len(prob), len(qs)))
    for j, q in enumerate(qs):
        b = np.minimum((cdf < q).sum(axis=1), BINS - 1)
        before = np.where(b > 0, np.take_along_axis(cdf, np.maximum(b - 1, 0)[:, None], 1)[:, 0], 0.0)
        pb = np.take_along_axis(prob, b[:, None], 1)[:, 0]
        frac = np.clip((q - before) / np.maximum(pb, 1e-12), 0, 1)
        out[:, j] = LO[b] + frac * (HI[b] - LO[b])
    return out


def cdf_grid(prob, z):
    """CDF of [N, 16] bin distributions at ln-second points z ([N, G] or [G]): 0 below 0.25 s, 1 at 60 s or more, linear
    in ln time inside a bin."""
    z = np.broadcast_to(z, (len(prob), z.shape[-1])) if z.ndim == 1 else z
    cum = np.c_[np.zeros(len(prob)), np.cumsum(prob, axis=1)]          # cum[:, k] = mass of bins < k
    b = np.searchsorted(LOG_EDGES, z, side="right")                    # bin holding z (0..15)
    bb = np.minimum(b, BINS - 1)
    lo, hi = LO[bb], HI[bb]
    frac = np.clip((z - lo) / np.where(hi > lo, hi - lo, 1.0), 0, 1)
    pk = np.take_along_axis(prob, bb, 1)
    f = np.take_along_axis(cum, bb, 1) + frac * pk
    f = np.where(z < LN_LO, 0.0, f)
    f = np.where(z >= LN60, 1.0, f)
    return np.clip(f, 0, 1)


def midrank(x):
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    starts = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1]])
    counts = np.diff(np.r_[starts, len(xs)])
    r = np.empty(len(x))
    r[order] = np.repeat(starts + 1 + (counts - 1) / 2.0, counts)
    return r


def auc_pairs(score, y):
    """(AUC with midranks, positives x negatives); higher score = more likely positive."""
    y = np.asarray(y, bool)
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan"), 0
    r = midrank(score)
    return float((r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)), npos * nneg


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra, rb = midrank(a), midrank(b)
    ra -= ra.mean(); rb -= rb.mean()
    den = math.sqrt(float((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def eff_number(v, delta):
    """exp(Shannon entropy) of the occupancy of the cells of rows v rounded to a delta grid."""
    if len(v) == 0:
        return float("nan")
    cells = np.round(np.asarray(v) / delta).astype(np.int64)
    _, cnt = np.unique(cells, axis=0, return_counts=True)
    p = cnt / cnt.sum()
    return float(math.exp(-(p * np.log(p)).sum()))


def fnan(x):
    x = float(x)
    return None if not math.isfinite(x) else x


# ------------------------------------------------------------------ per-user analysis (CPU)
def strat_center(v, g):
    out = np.empty(len(v))
    for k in range(4):
        m = g == k
        if m.any():
            out[m] = v[m] - v[m].mean()
    return out


def analyse_user(z, base, user):
    """Every per-user statistic of the design (metrics 1-5) from one user's dump and cached baselines."""
    th = z["th"].astype(np.int64)
    ti = th - 1
    cap = int(base["cap"])
    gr = z["grade"].astype(np.int64)
    d = z["d"].astype(np.int64)
    later = z["ed"] >= 1
    a, b, c, call, cnt = (base[k][ti].astype(np.float64) for k in ("a", "b", "c", "call", "cnt"))
    have = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & np.isfinite(call)
    keep = later & have
    rec = {"user_id": int(user), "rows": int(len(th)), "later": int(later.sum()), "later_no_card_history": int((later & ~have).sum()),
           "scored": int(keep.sum()), "cap_ms": cap}
    if keep.sum() < 2:
        return rec, None
    pt = z["pt"].astype(np.float64)[keep]
    pa = z["pa"].astype(np.float64)[keep]
    pa /= pa.sum(-1, keepdims=True)
    pg = z["pg"].astype(np.float64)[keep]
    gr, d, th, card, a, b, c, call, cnt = gr[keep], d[keep], th[keep], z["card"][keep], a[keep], b[keep], c[keep], call[keep], cnt[keep]
    n = len(gr)
    rows = np.arange(n)
    ph, pab = pt[rows, gr], pa[rows, gr]                 # true-grade distributions
    qh, qa = quantiles(ph), quantiles(pab)
    y = log_time(d, cap)
    x = {"head": qh[:, 2], "abl": qa[:, 2], "a": a, "b": b, "c": c}
    w = qh[:, 3] - qh[:, 1]
    rec["grade_counts"] = np.bincount(gr, minlength=4).tolist()

    # 1. AUC within user x grade (primary), grades pooled (secondary i), the head's grade mix (secondary ii)
    auc = {}
    for s in S_AUC:
        if cap <= 1000 * s:
            continue
        lab = d <= 1000 * s
        sc = {"head": cdf_at(ph, s), "abl": cdf_at(pab, s), "a": -a, "b": -b, "c": -c}
        strat = {p: [0.0, 0] for p in PREDS}
        npos = nneg = 0
        for k in range(4):
            m = gr == k
            yk = lab[m]
            if yk.all() or not yk.any():
                continue
            npos += int(yk.sum()); nneg += int((~yk).sum())
            for p in PREDS:
                v, pr = auc_pairs(sc[p][m], yk)
                strat[p][0] += v * pr; strat[p][1] += pr
        e = {"pos": npos, "neg": nneg}
        if npos >= MIN_AUC_CLASS and nneg >= MIN_AUC_CLASS:
            e["strat"] = {p: strat[p][0] / strat[p][1] for p in PREDS}
        tp, tn = int(lab.sum()), int((~lab).sum())
        e["pos_all"], e["neg_all"] = tp, tn
        if tp >= MIN_AUC_CLASS and tn >= MIN_AUC_CLASS:
            e["pooled"] = {p: auc_pairs(sc[p], lab)[0] for p in PREDS}
            mix = (pg * cdf_at(pt, s)).sum(-1)
            e["mix"] = {"head_mix": auc_pairs(mix, lab)[0], "a": e["pooled"]["a"], "b": e["pooled"]["b"],
                        "c_all": auc_pairs(-call, lab)[0]}
        auc[str(s)] = e
    rec["auc"] = auc

    # 2. range: review-level slopes within user x grade (+ the head grades pooled)
    yc = strat_center(y, gr)
    xc = {p: strat_center(x[p], gr) for p in PREDS}
    rec["slope"] = {p: {"cov": float((xc[p] * yc).mean()), "var": float((xc[p] ** 2).mean())} for p in PREDS}
    xp, yp = x["head"] - x["head"].mean(), y - y.mean()
    rec["slope"]["head_pooled"] = {"cov": float((xp * yp).mean()), "var": float((xp ** 2).mean())}
    rec["slope_n"] = n

    # card split: chronological A = first floor(n_c / 2) scored rows of the card, B = the rest
    _, cid = np.unique(card, return_inverse=True)
    order = np.lexsort((th, cid))
    cs = cid[order]
    nc = np.bincount(cid)
    start = np.r_[0, np.cumsum(nc)[:-1]]
    kpos = np.arange(n) - start[cs]
    inA = np.empty(n, bool); inA[order] = kpos < (nc[cs] // 2)
    ncard = len(nc)
    nA = np.bincount(cid[inA], minlength=ncard); nB = np.bincount(cid[~inA], minlength=ncard)

    def cmean(v, mask):
        s = np.bincount(cid[mask], weights=v[mask], minlength=ncard)
        k = np.bincount(cid[mask], minlength=ncard)
        return s / np.maximum(k, 1)

    two = (nA >= 1) & (nB >= 1)
    crd = {"cards2": int(two.sum())}
    if two.sum() >= MIN_CARDS:
        yA, yB = cmean(yc, inA)[two], cmean(yc, ~inA)[two]
        cov = lambda u, v: float(((u - u.mean()) * (v - v.mean())).mean())
        crd["Cyy"] = cov(yA, yB)
        crd["Vy"] = cov(yB, yB)
        for p in PREDS:
            xA = cmean(xc[p], inA)[two]
            crd[p] = {"Vp": cov(xA, xA), "Cpy": cov(xA, yB)}
    # within cards (cards with >= 3 scored rows)
    m3 = nc[cid] >= 3
    if (nc >= 3).sum() >= MIN_CARDS:
        dev = lambda v: v - (np.bincount(cid, weights=v, minlength=ncard) / nc)[cid]
        dy = dev(yc)[m3]
        crd["Vyw"] = float((dy ** 2).mean())
        crd["cards3"] = int((nc >= 3).sum())
        for p in PREDS:
            dx = dev(xc[p])[m3]
            crd.setdefault("within", {})[p] = {"Cw": float((dx * dy).mean()), "Vpw": float((dx ** 2).mean())}
    rec["card"] = crd

    # 4. shapes: per-user effective numbers, SDs; shape calibration
    sh = {"n": n, "sd_x": float(x["head"].std()), "sd_w": float(w.std())}
    for tag, q in (("head", qh), ("abl", qa)):
        norm = q[:, [0, 1, 3, 4]] - q[:, [2]]
        for dl in DELTAS:
            sh[f"{tag}_norm_{dl}"] = eff_number(norm, dl)
            sh[f"{tag}_loc_{dl}"] = eff_number(q, dl)
    rec["shape"] = sh
    cal = {}
    five = (nA >= 2) & (nB >= 3)
    cal["cards5"] = int(five.sum())
    if five.sum() >= MIN_CARDS:
        wA = cmean(w, inA)[five]
        sdB = np.zeros(ncard); sdA = np.zeros(ncard)
        for v_mask, out in ((~inA, sdB), (inA, sdA)):
            s1 = np.bincount(cid[v_mask], weights=yc[v_mask], minlength=ncard)
            s2 = np.bincount(cid[v_mask], weights=yc[v_mask] ** 2, minlength=ncard)
            k = np.bincount(cid[v_mask], minlength=ncard).astype(np.float64)
            var = (s2 - s1 ** 2 / np.maximum(k, 1)) / np.maximum(k - 1, 1)
            out[:] = np.sqrt(np.maximum(var, 0))
        cal["card_head"] = spearman(wA, sdB[five])
        cal["card_own"] = spearman(sdA[five], sdB[five])
    if n >= MIN_ROWS_SHAPE:
        cal["review"] = spearman(w, np.abs(y - x["head"]))
        r = midrank(w)
        quint = np.minimum((5 * (r - 1) / n).astype(int), 4)
        inside = (y >= qh[:, 1]) & (y <= qh[:, 3])
        cal["coverage"] = [fnan(inside[quint == k].mean()) if (quint == k).any() else None for k in range(5)]
        cal["coverage_all"] = float(inside.mean())
    rec["shape_cal"] = cal

    # 5. life of a card
    bk = np.searchsorted(np.asarray(BUCKET_EDGES), cnt, side="right")   # cnt 1 -> 0, 2-3 -> 1, ...
    ybar = y.mean()
    life = {}
    for tag, gm in (("all", np.ones(n, bool)), ("good", gr == 2)):
        for k, name in enumerate(BUCKETS):
            m = gm & (bk == k) & (cnt >= 1)
            if m.sum() < MIN_BUCKET:
                continue
            res = y[m] - x["head"][m]
            life.setdefault(tag, {})[name] = {"n": int(m.sum()), "pred": float((x["head"][m] - ybar).mean()),
                                              "act": float((y[m] - ybar).mean()), "w": float(w[m].mean()),
                                              "res_iqr": float(np.percentile(res, 75) - np.percentile(res, 25)),
                                              "grades": np.bincount(gr[m], minlength=4).tolist()}
    rec["life"] = life
    rec["cnt0"] = int((cnt < 1).sum())

    # shape sample for the pooled PCA / pooled distinct counts
    rng = np.random.default_rng([0, int(user)])
    pick = np.sort(rng.choice(n, size=min(SAMPLE_PER_USER, n), replace=False))
    sample = {}
    for tag, pp, q in (("head", ph, qh), ("abl", pab, qa)):
        sample[f"{tag}_norm"] = cdf_grid(pp[pick], q[pick, 2][:, None] + U_GRID[None, :]).astype(np.float32)
        sample[f"{tag}_loc"] = cdf_grid(pp[pick], T_GRID).astype(np.float32)
        sample[f"{tag}_q"] = q[pick].astype(np.float32)
    return rec, sample


def analyse(args):
    key = args.model
    dump_dir = Path(args.dump_dir or WORK / f"dump-{key}")
    out = Path(args.out or ROWS / f"sm142-{key}-users.jsonl")
    sdir = Path(args.sample_dir or WORK / f"sample-{key}")
    out.parent.mkdir(parents=True, exist_ok=True); sdir.mkdir(parents=True, exist_ok=True)
    idx = load_jsonl(args.index or GPU / f"dump-{key}.jsonl")
    bad = set()
    vf = GPU / f"verify-{key}.json"
    if vf.exists() and not args.users:
        bad = {u for u, _ in json.loads(vf.read_text())["bad"]}
    users = args.users or sorted(u for u in idx if u not in bad)
    done = set(load_jsonl(out)) if out.exists() else set()
    todo = [u for u in users if u not in done]
    started = time.perf_counter()
    print(f"[sm142-analyse] {key}: {len(users)} users, {len(todo)} to analyse", flush=True)
    with out.open("a", encoding="utf-8") as f:
        for k, u in enumerate(todo):
            if args.until and time.time() > args.until:
                print(f"[sm142-analyse] paused before user {u} ({k} of {len(todo)})", flush=True)
                return 3
            zp = dump_dir / f"{u}.npz"
            if not zp.exists():
                f.write(json.dumps({"user_id": u, "error": "no dump"}) + "\n"); f.flush()
                continue
            with np.load(zp) as zz:
                z = {kk: zz[kk] for kk in zz.files}
            base = user_base(u)
            rec, sample = analyse_user(z, base, u)
            rec["model"] = key
            if sample is not None:
                atomic_npz(sdir / f"{u}.npz", **sample)
            f.write(json.dumps(rec) + "\n"); f.flush()
            if (k + 1) % 50 == 0:
                print(f"[sm142-analyse] {k + 1}/{len(todo)} users, {time.perf_counter() - started:.0f} s", flush=True)
    print(f"[sm142-analyse] done {len(todo)} users in {time.perf_counter() - started:.0f} s -> {out}", flush=True)
    return 0


# ------------------------------------------------------------------ aggregation (report)
def boot_mean(v, idx=None):
    v = np.asarray(v, np.float64)
    idx = bootstrap_idx(len(v)) if idx is None else idx
    bs = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_ratio(num, den, f=lambda n, d: n / d):
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    idx = bootstrap_idx(len(num))
    bs = f(num[idx].sum(1), den[idx].sum(1))
    return float(f(num.sum(), den.sum())), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_share_diff(cpy1, vp1, cpy2, vp2, cyy):
    """Explained between-card share sum(C^2/V)/sum(Cyy) of predictor 1 minus predictor 2, paired over users."""
    t1 = np.asarray(cpy1) ** 2 / np.asarray(vp1); t2 = np.asarray(cpy2) ** 2 / np.asarray(vp2); cyy = np.asarray(cyy)
    idx = bootstrap_idx(len(cyy))
    bs = (t1[idx].sum(1) - t2[idx].sum(1)) / cyy[idx].sum(1)
    return float((t1.sum() - t2.sum()) / cyy.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def pca_stats(m):
    m = np.asarray(m, np.float64)
    m = m - m.mean(0)
    ev = np.clip(np.linalg.eigvalsh(m.T @ m / len(m))[::-1], 0, None)
    tot = ev.sum()
    cum = np.cumsum(ev) / tot
    return {"k90": int(np.searchsorted(cum, 0.90) + 1), "k95": int(np.searchsorted(cum, 0.95) + 1),
            "k99": int(np.searchsorted(cum, 0.99) + 1), "pr": float(tot ** 2 / (ev ** 2).sum()),
            "top5_share": [float(e / tot) for e in ev[:5]], "total_var": float(tot), "rows": int(len(m))}


def aggregate(key):
    rows = [r for r in load_jsonl(ROWS / f"sm142-{key}-users.jsonl").values() if r.get("scored", 0) >= 2]
    res = {"model": key, "name": MODELS[key]["name"], "users": len(rows),
           "rows": {"later": int(sum(r["later"] for r in rows)), "scored": int(sum(r["scored"] for r in rows)),
                    "later_no_card_history": int(sum(r["later_no_card_history"] for r in rows))}}
    # 1. AUC
    auc = {}
    for s in S_AUC:
        for mode, preds, ref in (("strat", PREDS, "head"), ("pooled", PREDS, "head"), ("mix", ("head_mix", "a", "b", "c_all"), "head_mix")):
            us = [r["auc"][str(s)][mode] for r in rows if mode in r["auc"].get(str(s), {})]
            if len(us) < 20:
                continue
            idx = bootstrap_idx(len(us))
            e = {"users": len(us)}
            for p in preds:
                e[p] = boot_mean([u[p] for u in us], idx)
            for p in preds:
                if p != ref:
                    e[f"{ref}-{p}"] = boot_mean([u[ref] - u[p] for u in us], idx)
            auc.setdefault(mode, {})[str(s)] = e
    res["auc"] = auc
    # 2. slopes
    sl = {}
    for p in list(PREDS) + ["head_pooled"]:
        cov = [r["slope"][p]["cov"] for r in rows]; var = [r["slope"][p]["var"] for r in rows]
        per = [r["slope"][p]["cov"] / r["slope"][p]["var"] for r in rows if r["slope_n"] >= MIN_SLOPE_ROWS and r["slope"][p]["var"] > 0]
        sl[p] = {"pooled": boot_ratio(cov, var), "median_user": float(np.median(per)) if per else None, "users_median": len(per)}
    res["slope"] = sl
    # 2/3. card level
    cr = [r["card"] for r in rows if "Cyy" in r["card"] and all(r["card"][p]["Vp"] > 0 for p in PREDS)]
    cl = {"users": len(cr), "cards": int(sum(c["cards2"] for c in cr))}
    cyy = [c["Cyy"] for c in cr]
    for p in PREDS:
        vp = [c[p]["Vp"] for c in cr]; cpy = [c[p]["Cpy"] for c in cr]
        cl[p] = {"between_slope": boot_ratio(cpy, vp), "spread_ratio": boot_ratio(vp, cyy, lambda a, b: np.sqrt(a / b)),
                 "spread_ratio_median_user": float(np.median([math.sqrt(c[p]["Vp"] / c["Cyy"]) for c in cr if c["Cyy"] > 0])),
                 "explained": boot_ratio([cp ** 2 / v for cp, v in zip(cpy, vp)], cyy)}
    for p in ("a", "b", "abl", "c"):
        cl[f"head-{p}"] = boot_share_diff([c["head"]["Cpy"] for c in cr], [c["head"]["Vp"] for c in cr],
                                          [c[p]["Cpy"] for c in cr], [c[p]["Vp"] for c in cr], cyy)
    wr = [r["card"] for r in rows if "within" in r["card"] and all(r["card"]["within"][p]["Vpw"] > 0 for p in PREDS)]
    cl["within_users"] = len(wr)
    vyw = [c["Vyw"] for c in wr]
    for p in PREDS:
        cl[p]["within_explained"] = boot_ratio([c["within"][p]["Cw"] ** 2 / c["within"][p]["Vpw"] for c in wr], vyw)
    both = [r["card"] for r in rows if "Cyy" in r["card"] and "Vyw" in r["card"]]
    cl["persistent_between_share_median_user"] = float(np.median([c["Cyy"] / (c["Cyy"] + c["Vyw"]) for c in both if c["Cyy"] + c["Vyw"] > 0]))
    res["card"] = cl
    # 4. shapes
    sh = {}
    for fld in [f"{t}_{k}_{dl}" for t in ("head", "abl") for k in ("loc", "norm") for dl in DELTAS] + ["sd_x", "sd_w", "n"]:
        v = np.array([r["shape"][fld] for r in rows if r["shape"].get(fld) is not None], np.float64)
        sh[fld] = {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
    sdir = WORK / f"sample-{key}"
    samples = [np.load(sdir / f"{r['user_id']}.npz") for r in rows if (sdir / f"{r['user_id']}.npz").exists()]
    pooled = {}
    for tag in ("head", "abl"):
        for rep in ("norm", "loc"):
            pooled[f"{tag}_{rep}"] = pca_stats(np.concatenate([z[f"{tag}_{rep}"] for z in samples]))
        q = np.concatenate([z[f"{tag}_q"] for z in samples]).astype(np.float64)
        for dl in DELTAS:
            pooled[f"{tag}_q_loc_{dl}"] = eff_number(q, dl)
            pooled[f"{tag}_q_norm_{dl}"] = eff_number(q[:, [0, 1, 3, 4]] - q[:, [2]], dl)
    sh["pooled"] = pooled
    sh["sample_users"] = len(samples)
    cal = [r["shape_cal"] for r in rows]
    cc = [c for c in cal if c.get("card_head") is not None and c.get("card_own") is not None
          and math.isfinite(c["card_head"]) and math.isfinite(c["card_own"])]
    sc = {"card_users": len(cc)}
    if len(cc) >= 20:
        idx = bootstrap_idx(len(cc))
        sc["card_head"] = boot_mean([c["card_head"] for c in cc], idx)
        sc["card_own"] = boot_mean([c["card_own"] for c in cc], idx)
        sc["card_head-own"] = boot_mean([c["card_head"] - c["card_own"] for c in cc], idx)
    rv = [c["review"] for c in cal if c.get("review") is not None and math.isfinite(c["review"])]
    sc["review_users"] = len(rv)
    if rv:
        sc["review"] = boot_mean(rv)
    cov_rows = [c["coverage"] for c in cal if c.get("coverage")]
    sc["coverage_by_width_quintile"] = [float(np.mean([cv[k] for cv in cov_rows if cv[k] is not None])) for k in range(5)]
    sc["coverage_all"] = float(np.mean([c["coverage_all"] for c in cal if "coverage_all" in c]))
    sh["calibration"] = sc
    res["shape"] = sh
    # 5. life of a card
    life = {}
    for tag in ("all", "good"):
        for name in BUCKETS:
            us = [r["life"][tag][name] for r in rows if name in r["life"].get(tag, {})]
            if len(us) < 20:
                continue
            idx = bootstrap_idx(len(us))
            g = np.sum([u["grades"] for u in us], axis=0)
            life.setdefault(tag, {})[name] = {"users": len(us), "pred": boot_mean([u["pred"] for u in us], idx),
                                              "act": boot_mean([u["act"] for u in us], idx),
                                              "pred-act": boot_mean([u["pred"] - u["act"] for u in us], idx),
                                              "w": boot_mean([u["w"] for u in us], idx),
                                              "res_iqr": boot_mean([u["res_iqr"] for u in us], idx),
                                              "grade_share": (g / g.sum()).tolist(), "rows": int(sum(u["n"] for u in us))}
    res["life"] = life
    res["labels"] = labels_of(res)
    return res


def labels_of(res):
    st = res["auc"].get("strat", {})
    e1 = all(str(s) in st and st[str(s)]["head-a"][1] > 0 for s in (2, 3, 4, 6))
    sl = res["slope"]["head"]["pooled"][0]
    bs = res["card"]["head"]["between_slope"][0]
    e2 = sl >= 0.8 and 0.8 <= bs <= 1.25
    e3 = res["card"]["head-a"][1] > 0
    sc = res["shape"]["calibration"]
    e4 = res["shape"]["pooled"]["head_norm"]["k95"] >= 2 and "card_head" in sc and sc["card_head"][1] > 0
    lab = "template-like" if not e1 else ("shape-expressive" if (e1 and e2 and e3 and e4) else
                                          ("card-expressive" if (e1 and e2 and e3) else "partly expressive"))
    return {"E1": e1, "E2": e2, "E3": e3, "E4": e4, "label": lab}


def pct(v):
    return f"{100 * (math.exp(v) - 1):+.0f}%"


def f3(t, nd=3):
    return f"{t[0]:.{nd}f} [{t[1]:.{nd}f}, {t[2]:.{nd}f}]"


def write_md(allres, path):
    L = [f"# {NAME}: result", "",
         "Answer time = front + back together, from the moment the question is shown to the moment a grade button is pressed, capped at the maximum answer seconds; \"answered within S s\" means answered within S s, reading the back included. Rows: the 732 set (722 users with time rows), show-time inputs, leak-free, fp32; reviews on a later day than the card's previous review whose card has an earlier timed review. Pre-registered methods: `design.md`. Every CI: 2,000 paired user bootstraps, seed 0, 95%. Description, not a gate.", ""]
    names = {k: r["name"] for k, r in allres.items()}
    L += ["## Summary", "", "| model | label | E1 AUC vs card median | E2 slope / between-card slope | E3 between-card explained head vs (a) | E4 shapes (PCA k95, card shape calibration) |", "|---|---|---|---|---|---|"]
    for k, r in allres.items():
        lb = r["labels"]; c = r["card"]; sc = r["shape"]["calibration"]
        L.append(f"| {names[k]} | {lb['label']} | {lb['E1']} | {lb['E2']} ({r['slope']['head']['pooled'][0]:.2f} / {c['head']['between_slope'][0]:.2f}) | "
                 f"{lb['E3']} ({c['head']['explained'][0]:.1%} vs {c['a']['explained'][0]:.1%}) | {lb['E4']} ({r['shape']['pooled']['head_norm']['k95']}, "
                 f"{sc['card_head'][0]:.3f} [{sc['card_head'][1]:.3f}, {sc['card_head'][2]:.3f}]) |" if "card_head" in sc else
                 f"| {names[k]} | {lb['label']} | {lb['E1']} | {lb['E2']} | {lb['E3']} | {lb['E4']} |")
    L += ["", "## 1. Within-user, within-grade AUC (P(answered within S s | true grade))", "",
          "Each user equal; strata = (user, true grade). (a) the card's running median, (b) the card's last time, (c) the user's running median of the grade; ablation = the explicit-features-only head.", ""]
    for k, r in allres.items():
        L += [f"### {names[k]}", "", "| S | users | head | ablation | (a) card median | (b) card last | (c) user grade median | head - (a) | head - ablation | head - (b) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for s, e in r["auc"].get("strat", {}).items():
            L.append(f"| {s} s | {e['users']} | {e['head'][0]:.4f} | {e['abl'][0]:.4f} | {e['a'][0]:.4f} | {e['b'][0]:.4f} | {e['c'][0]:.4f} | {f3(e['head-a'], 4)} | {f3(e['head-abl'], 4)} | {f3(e['head-b'], 4)} |")
        L += ["", "Secondary, grades pooled within user: head on the true grade / the head's own grade mix vs (a), (b), the user's running median:", ""]
        L += ["| S | head (true grade) | head - (a) | head (grade mix) | grade mix - (a) | (c) all-grade user median |", "|---|---|---|---|---|---|"]
        for s, e in r["auc"].get("pooled", {}).items():
            m = r["auc"].get("mix", {}).get(s)
            L.append(f"| {s} s | {e['head'][0]:.4f} | {f3(e['head-a'], 4)} | {m['head_mix'][0]:.4f} | {f3(m['head_mix-a'], 4)} | {m['c_all'][0]:.4f} |" if m else f"| {s} s | {e['head'][0]:.4f} | {f3(e['head-a'], 4)} | | | |")
        L.append("")
    L += ["## 2-3. Range and variance explained", "",
          "Review-level slope: actual log time on the predicted log median within (user, grade), each user weighted by its predicted variance. Card level (out of sample: predictions on a card's first half, actual times on its second half): between-card slope, spread ratio = SD of predicted card means / SD of the persistent actual card means, explained share of the persistent between-card variance. Within cards: share of the within-card variance explained.", "",
          "| model | predictor | review slope | median user slope | between-card slope | spread ratio | between-card explained | within-card explained |", "|---|---|---|---|---|---|---|---|"]
    pn = {"head": "head", "abl": "ablation", "a": "(a) card median", "b": "(b) card last", "c": "(c) user grade median"}
    for k, r in allres.items():
        for p in PREDS:
            c = r["card"][p]; s = r["slope"][p]
            L.append(f"| {names[k]} | {pn[p]} | {f3(s['pooled'])} | {s['median_user']:.3f} | {f3(c['between_slope'])} | {f3(c['spread_ratio'])} | {f3(c['explained'])} | {f3(c['within_explained'])} |")
        c = r["card"]
        L.append(f"| {names[k]} | head - (a), head - ablation (between-card explained) | | | | | {f3(c['head-a'])}; {f3(c['head-abl'])} | |")
        L.append(f"| {names[k]} | head, grades pooled within user | {f3(r['slope']['head_pooled']['pooled'])} | | | | | |")
    L += ["", "Persistent between-card share of the within-(user, grade) variance (median user): " +
          ", ".join(f"{names[k]} {r['card']['persistent_between_share_median_user']:.1%}" for k, r in allres.items()), ""]
    L += ["## 4. Shapes", "", "Effective number of distinct curves = exp(entropy) of the cells of the quantile vectors (q10, q25, q50, q75, q90 of the true-grade distribution) rounded to 0.1 ln-units (about 10% in time) / 0.2; location-normalised = quantiles minus the median. Per user: median [p10, p90] over users. PCA on CDF grids of a pooled sample (200 rows per user).", "",
          "| model | head | per-user curves, location incl. 0.1 / 0.2 | per-user shapes, normalised 0.1 / 0.2 | pooled PCA normalised k90/k95/k99, PR | pooled PCA location incl. k90/k95/k99, PR | per-user SD of predicted log median / of log IQR |", "|---|---|---|---|---|---|---|"]
    for k, r in allres.items():
        s = r["shape"]
        for t in ("head", "abl"):
            pn_ = s["pooled"][f"{t}_norm"]; pl = s["pooled"][f"{t}_loc"]
            sdx = f"{s['sd_x']['median']:.3f} / {s['sd_w']['median']:.3f}" if t == "head" else ""
            L.append(f"| {names[k]} | {'head' if t == 'head' else 'ablation'} | {s[f'{t}_loc_0.1']['median']:.0f} [{s[f'{t}_loc_0.1']['p10']:.0f}, {s[f'{t}_loc_0.1']['p90']:.0f}] / {s[f'{t}_loc_0.2']['median']:.0f} | "
                     f"{s[f'{t}_norm_0.1']['median']:.0f} [{s[f'{t}_norm_0.1']['p10']:.0f}, {s[f'{t}_norm_0.1']['p90']:.0f}] / {s[f'{t}_norm_0.2']['median']:.0f} | "
                     f"{pn_['k90']}/{pn_['k95']}/{pn_['k99']}, {pn_['pr']:.2f} | {pl['k90']}/{pl['k95']}/{pl['k99']}, {pl['pr']:.2f} | {sdx} |")
    L += ["", f"Median scored rows per user: " + ", ".join(f"{names[k]} {r['shape']['n']['median']:.0f}" for k, r in allres.items()), ""]
    L += ["Shape calibration: card level = Spearman, within user, of the predicted IQR (first half of the card) with the realised SD of the card's later times (second half), against the card's own earlier SD; review level = Spearman of the predicted IQR with |actual - predicted median|; coverage = share of answers inside the predicted IQR by within-user quintile of the predicted IQR (ideal 50%).", "",
          "| model | card: head | card: own earlier SD | head - own | review level | coverage by width quintile (narrow to wide) | coverage all |", "|---|---|---|---|---|---|---|"]
    for k, r in allres.items():
        sc = r["shape"]["calibration"]
        if "card_head" in sc:
            L.append(f"| {names[k]} | {f3(sc['card_head'])} | {f3(sc['card_own'])} | {f3(sc['card_head-own'])} | {f3(sc['review'])} | "
                     f"{' / '.join(f'{v:.0%}' for v in sc['coverage_by_width_quintile'])} | {sc['coverage_all']:.1%} |")
    L += ["", "## 5. Life of a card", "", "Answer count = earlier reviews of the card. Levels as % of the user's typical time (geometric mean of its scored answers); each user equal.", ""]
    for k, r in allres.items():
        L += [f"### {names[k]}", "", "| grades | earlier reviews | users | predicted median | actual mean | predicted - actual (ln) | predicted IQR (ln) | realised IQR of residuals (ln) | Again / Hard / Good / Easy share |", "|---|---|---|---|---|---|---|---|---|"]
        for tag in ("all", "good"):
            for name, e in r["life"].get(tag, {}).items():
                L.append(f"| {tag} | {name} | {e['users']} | {pct(e['pred'][0])} | {pct(e['act'][0])} | {f3(e['pred-act'])} | {e['w'][0]:.3f} | {e['res_iqr'][0]:.3f} | {' / '.join(f'{v:.0%}' for v in e['grade_share'])} |")
        L.append("")
    L += ["## 6. Plain-language read-out", ""]
    for k, r in allres.items():
        s = r["shape"]; st = r["auc"].get("strat", {}); c = r["card"]
        def d(sv):
            e = st.get(sv)
            return f"{e['head-a'][0]:+.3f}" if e else "n/a"
        L.append(f"- {names[k][0].upper() + names[k][1:]}: a typical user gets about {s['head_loc_0.1']['median']:.0f} distinguishable time curves (location included, 10% grid) and about {s['head_norm_0.1']['median']:.0f} distinguishable shapes once the location is removed. "
                 f"Within a grade, the head ranks this user's reviews by speed with AUC {st['4']['head'][0]:.3f} at 4 s against {st['4']['a'][0]:.3f} for the card's own running median ({d('2')} at 2 s, {d('4')} at 4 s, {d('8')} at 8 s). "
                 f"It explains {c['head']['explained'][0]:.0%} of the persistent card-to-card differences against {c['a']['explained'][0]:.0%} for the card's own median and {c['abl']['explained'][0]:.0%} for the explicit-features-only head. Label: {r['labels']['label']}." if "4" in st else f"- {names[k]}: not enough users.")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def report(args):
    allres = {}
    for key in MODELS_142:
        if (ROWS / f"sm142-{key}-users.jsonl").exists():
            allres[key] = aggregate(key)
    if not allres:
        raise SystemExit("no analysed model")
    (OUT / "sm142-result.json").write_text(json.dumps(allres, indent=1), encoding="utf-8")
    write_md(allres, OUT / "sm142-result.md")
    print("report: " + "; ".join(f"{k} {r['labels']['label']}" for k, r in allres.items()))
    return 0


# ------------------------------------------------------------------ CPU proof
def brute_baselines(card, rating, duration):
    cap = time_user_cap(duration)
    y = log_time(np.asarray(duration, np.int64), cap)
    n = len(card)
    a = np.full(n, np.nan); b = np.full(n, np.nan); c = np.full(n, np.nan); cnt = np.zeros(n, np.int32)
    for i in range(n):
        prev = np.arange(i)
        same = prev[(card[:i] == card[i])]
        cnt[i] = len(same)
        ts = same[duration[same] > 0]
        if len(ts):
            a[i] = np.median(y[ts]); b[i] = y[ts[-1]]
        tg = prev[(duration[:i] > 0) & (rating[:i] == rating[i])]
        tall = prev[duration[:i] > 0]
        c[i] = np.median(y[tg]) if len(tg) else (np.median(y[tall]) if len(tall) else np.nan)
    return a, b, c, cnt


def check(args):
    import pandas as pd
    t0 = time.time()
    res = {"checks": {}}
    ck = res["checks"]
    # quantile rule = median_log_s; cdf_grid agrees with cdf_at at S points and with quantiles
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.full(BINS, 0.3), size=2000)
    q = quantiles(p)
    ck["q50_equals_median_log_s"] = bool(np.allclose(q[:, 2], median_log_s(p), atol=1e-12, rtol=0))
    ck["cdf_grid_equals_cdf_at"] = bool(all(np.allclose(cdf_grid(p, np.array([math.log(s)]))[:, 0], cdf_at(p, s), atol=1e-12) for s in S_AUC))
    inner = (q[:, 1] > LN_LO) & (q[:, 1] < math.log(59.9))
    ck["cdf_at_quantile_is_q"] = bool(np.allclose(cdf_grid(p[inner], q[inner, 1][:, None])[:, 0], 0.25, atol=1e-9))
    # AUC midranks vs brute pairs
    sc = rng.integers(0, 20, 300).astype(float); yy = rng.random(300) < 0.4
    pos, neg = sc[yy], sc[~yy]
    brute = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))
    ck["auc_equals_brute_pairs"] = bool(abs(auc_pairs(sc, yy)[0] - brute) < 1e-12)
    ck["eff_number_uniform"] = bool(abs(eff_number(np.arange(8)[:, None] * 1.0, 1.0) - 8) < 1e-9)
    # users: the two smallest pool users with >= 500 recorded time rows of the six-layer Tiny two-head model (SM130)
    rec_rows = load_jsonl(RECORDED["sm130th"])
    small = sorted((r["reviews"], u) for u, r in rec_rows.items() if r["reviews"] >= 500)[:2]
    users = [u for _, u in small]
    res["users"] = users
    uf = WORK / "check" / "users.json"
    uf.parent.mkdir(parents=True, exist_ok=True)
    uf.write_text(json.dumps({"users": users}))
    dd, ix = WORK / "check" / "dump-sm130th", WORK / "check" / "dump-sm130th.jsonl"
    if ix.exists():
        ix.unlink()
    dump(Namespace(model="sm130th", head_dir="", users_file=str(uf), out_dir=str(dd), index=str(ix), device="cpu", workers=0))
    idx = load_jsonl(ix)
    ck["dump_all_users"] = sorted(idx) == sorted(users)
    m = [sums_match(idx[u], rec_rows[u], tol=1e-5) for u in users if u in idx]
    res["dump_vs_recorded_gpu"] = [why for _, why in m]
    ck["dump_equals_recorded_gpu_1e-5"] = bool(m and all(ok for ok, _ in m))
    # baselines vs brute force
    okb = True
    for u in users:
        fr = pd.read_parquet(Path(data_path()) / "revlogs" / f"user_id={u}", columns=["card_id", "rating", "duration"])
        cd, rt, du = (fr[c].to_numpy(np.int64) for c in ("card_id", "rating", "duration"))
        a, b, c, call, cnt, cap = baselines_full(cd, rt, du)
        ba, bb, bc, bcnt = brute_baselines(cd, rt, du)
        okb &= bool(np.allclose(a, ba, equal_nan=True) and np.allclose(b, bb, equal_nan=True) and np.allclose(c, bc, equal_nan=True) and (cnt == bcnt).all())
    ck["baselines_equal_brute_force"] = okb
    # dump rows match the frame (card ids, grades, durations)
    okf = True
    for u in users:
        z = np.load(dd / f"{u}.npz")
        fr = pd.read_parquet(Path(data_path()) / "revlogs" / f"user_id={u}", columns=["card_id", "rating", "duration", "elapsed_days"])
        ti = z["th"].astype(np.int64) - 1
        okf &= bool((fr["card_id"].to_numpy()[ti] == z["card"]).all() and (fr["rating"].to_numpy()[ti] - 1 == z["grade"]).all()
                    and (fr["duration"].to_numpy()[ti] == z["d"]).all() and (np.diff(ti) > 0).all())
        okf &= bool(np.allclose(z["pt"].sum(-1), 1, atol=1e-5) and np.allclose(z["pa"].astype(np.float64).sum(-1), 1, atol=2e-2))
    ck["dump_rows_match_revlog"] = okf
    # end to end on the two users (a pipeline check, not a result)
    old_base = WORK / "base"
    outj = WORK / "check" / "users-sm130th.jsonl"
    if outj.exists():
        outj.unlink()
    rc = analyse(Namespace(model="sm130th", dump_dir=str(dd), out=str(outj), sample_dir=str(WORK / "check" / "sample-sm130th"),
                           index=str(ix), users=users, until=0))
    recs = load_jsonl(outj)
    ck["analyse_end_to_end"] = rc == 0 and sorted(recs) == sorted(users) and all(r["scored"] > 0 for r in recs.values())
    finite = True
    for r in recs.values():
        for e in r["auc"].values():
            for mode in ("strat", "pooled"):
                if mode in e:
                    finite &= all(math.isfinite(v) for v in e[mode].values())
        finite &= all(math.isfinite(v["cov"]) and math.isfinite(v["var"]) for v in r["slope"].values())
    ck["analyse_finite"] = finite
    res["analyse_summary"] = {u: {"scored": r["scored"], "auc_keys": sorted(r["auc"]), "card": sorted(r["card"])} for u, r in recs.items()}
    res["base_cache"] = str(old_base)
    res["pass"] = all(ck.values())
    res["seconds"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({"pass": res["pass"], "checks": ck, "seconds": res["seconds"]}))
    return 0 if res["pass"] else 1


def info(args):
    print(f"MNAME='{MODELS[args.model]['name']}'")
    print(f"HEADDIR='{HEAD_DIRS[args.model].as_posix()}'")
    print(f"RECORDED='{RECORDED[args.model].as_posix()}'")
    return 0


def main():
    ap = ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dump")
    p.add_argument("--model", required=True, choices=MODELS_142)
    p.add_argument("--head-dir", default="")
    p.add_argument("--users-file", default=str(WORK / "users-722.json"))
    p.add_argument("--out-dir", default="")
    p.add_argument("--index", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=3)
    p = sub.add_parser("verify")
    p.add_argument("--model", required=True, choices=MODELS_142)
    p.add_argument("--index", default="")
    p.add_argument("--out", default="")
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--min-users", type=int, default=700)
    p = sub.add_parser("analyse")
    p.add_argument("--model", required=True, choices=MODELS_142)
    p.add_argument("--dump-dir", default="")
    p.add_argument("--index", default="")
    p.add_argument("--out", default="")
    p.add_argument("--sample-dir", default="")
    p.add_argument("--users", type=int, nargs="*", default=None)
    p.add_argument("--until", type=float, default=0, help="epoch seconds: stop (exit 3) before starting a user after this")
    sub.add_parser("report")
    p = sub.add_parser("check")
    p.add_argument("--out", default=str(OUT / "check-cpu.json"))
    p = sub.add_parser("users")
    p = sub.add_parser("info")
    p.add_argument("--model", required=True, choices=MODELS_142)
    a = ap.parse_args()
    if a.cmd == "users":
        WORK.mkdir(parents=True, exist_ok=True)
        u = pool_users()
        (WORK / "users-722.json").write_text(json.dumps({"users": u}))
        print(f"users {len(u)} -> {WORK / 'users-722.json'}")
        return 0
    return {"dump": dump, "verify": verify, "analyse": analyse, "report": report, "check": check, "info": info}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
