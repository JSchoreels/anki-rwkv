#!/usr/bin/env python
"""Response-time prediction (SM82): answer-time audit, warm start, CPU proof, time scoring and decision.

Stage 0 (CPU):
  audit      --out artifacts/small-model/sm82/stage0-audit   answer-time spread per grade and per user, the share at
             the 60 s cap and at exactly 0, per-user caps, how much of the spread is the card's own (text length) vs
             the user's, and the leak audit (a review's own time never enters the row that predicts it; the interval
             feature of that row is answer-to-answer, so it holds the review's own time: how much). Writes .json + .md.
Stage 1 (the answer-time head, expertium.variants preset "w64-sm82": per grade 16 log-time bins, see BIN_EDGES_MS):
  extend     --master M --out DIR     DIR/master.pth = M (w64) + the zero head (time_heads.0.linear.*), extend.json.
  cooldown   --start DIR --out DIR2   the quarter-epoch 32k Muon cool-down of the reference recipe (users, order,
             options, schedule from the leak-free small model's (SM72s) epoch 21, = the tiny architecture additions
             (SM77) reference's) with architecture "w64-sm82" and the start weights DIR/master.pth.
  check      --ext DIR --out report.json   the CPU proof (see `check`).
  baselines  --out FILE.npz   CPU: per test review of the 732 users (existing cards), the grade, the time label and
             the two baselines (the card's last time; the user's running median per grade), as log-time bin
             distributions with residual CDFs fit on the users' non-test reviews.
  score      --master M --baselines FILE --out FILE.jsonl   GPU (or CPU): the head's per-grade time distribution at
             every test review (fp32, leak-free inputs as the grade scorer), aligned with the baselines; one row per user.
  decide     --time-rows FILE --arm-rows FILE --arm-label L --ref-rows FILE --ref-label L   the stage-1 rule.

Bins (milliseconds, Anki's stored duration): 15 edges 500 x 120^(k/14), k = 0..14 (0.5 s .. 60 s, x1.41 per bin);
bin 0 = under 0.5 s, bins 1..14 between edges, bin 15 = 60 s and more (the dataset caps every time at 60 s, so bin
15 is "hit the cap"). A user whose own cap is lower (Anki's "maximum answer seconds") has times at that cap censored:
the label is "this bin or later". Duration 0 = no time recorded: never a label.
"""
from argparse import ArgumentParser
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HERE = ROOT / "artifacts/small-model/sm82"
S72 = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s")  # read only
U732, ULONG = S72 / "eval-users-732-fit.json", S72 / "users-732-long.json"
RUN72S = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes-leakfree/run")
TRACK = "E89-F65e-F77a"
ARCH = "w64-sm82"
BINS = 16
# the label definition lives with the trainer (scripts/expertium_options.py, a SourceWatch file); one copy only
from scripts.expertium_options import (TIME_BIN_EDGES_MS as BIN_EDGES_MS, TIME_CAP_MS as CAP_MS,  # noqa: E402
                                       time_labels, time_user_cap as user_cap)


def data_path():
    from scripts import compress_common as cc
    return cc.data_path()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------ stage 0 audit
def _audit_user(args):
    data, user = args
    import pandas as pd
    df = pd.read_parquet(data / "revlogs" / f"user_id={user}",
                         columns=["card_id", "rating", "duration", "elapsed_days", "elapsed_seconds"])
    d = df["duration"].to_numpy(np.int64)
    rating = df["rating"].to_numpy(np.int64)
    card = df["card_id"].to_numpy(np.int64)
    es = df["elapsed_seconds"].to_numpy(np.int64)
    ed = df["elapsed_days"].to_numpy(np.int64)
    n = len(d)
    first = ed == -1
    cap = user_cap(d)
    lo, cens, _ = time_labels(d)
    lo = lo.astype(np.int8)
    out = {"user": user, "reviews": n, "non_first": int((~first).sum()), "zero": int((d == 0).sum()),
           "zero_non_first": int(((d == 0) & ~first).sum()), "at_60s": int((d >= CAP_MS).sum()),
           "max_ms": int(d.max()) if n else 0, "cap_ms": cap, "censored": int(cens.sum())}
    timed = (d > 0) & ~first
    out["timed_non_first"] = int(timed.sum())
    out["bin_hist"] = np.bincount(lo[timed].astype(np.int64), minlength=BINS).tolist()
    logd = np.log(np.maximum(d, 1) / 1000.0)
    unc = timed & (d < cap)
    out["grade"] = {}
    for g in (1, 2, 3, 4):
        m = timed & (rating == g)
        out["grade"][g] = {"n": int(m.sum()), "hist": np.bincount(lo[m].astype(np.int64), minlength=BINS).tolist(),
                           "median_s": float(np.median(d[m]) / 1000) if m.any() else None}
    out["median_s"] = float(np.median(d[timed]) / 1000) if timed.any() else None
    # card vs user spread of log time (uncensored timed reviews of non-new cards; cards with >= 2 such reviews)
    if unc.sum() >= 50:
        x, c = logd[unc], card[unc]
        order = np.argsort(c, kind="stable")
        xs, cs = x[order], c[order]
        _, start, counts = np.unique(cs, return_index=True, return_counts=True)
        sums = np.add.reduceat(xs, start)
        means = np.repeat(sums / counts, counts)
        keep = np.repeat(counts >= 2, counts)
        if keep.sum() >= 50:
            within = float(((xs - means)[keep] ** 2).mean())
            total = float(((xs[keep] - xs[keep].mean()) ** 2).mean())
            out["var_total"], out["var_within_card"], out["n_var"] = total, within, int(keep.sum())
    # card's own previous time vs this time (log), grade-agnostic
    prev_idx = np.full(n, -1, np.int64)
    order = np.lexsort((np.arange(n), card))
    same = np.r_[False, card[order][1:] == card[order][:-1]]
    prev_idx[order[same]] = order[np.flatnonzero(same) - 1]
    pair = (prev_idx >= 0) & unc
    pair &= np.where(prev_idx >= 0, (d[np.maximum(prev_idx, 0)] > 0) & (d[np.maximum(prev_idx, 0)] < cap), False)
    if pair.sum() >= 50:
        a, b = logd[pair], logd[prev_idx[pair]]
        out["corr_prev_card"] = float(np.corrcoef(a, b)[0, 1])
        out["mae_prev_card"] = float(np.abs(a - b).mean())
        out["n_pair"] = int(pair.sum())
    # leak audit: is elapsed_seconds answer-to-answer? Then elapsed_j >= the times of every review answered after the
    # card's previous answer up to and including review j; answer-to-show would only bound the ones before j.
    cum = np.cumsum(d)
    has = prev_idx >= 0
    j = np.flatnonzero(has)
    i = prev_idx[j]
    incl = cum[j] - cum[i]                    # sum of d over (i, j]
    excl = incl - d[j]                        # sum of d over (i, j)
    short = es[j] < 3600
    slack = 1000                              # elapsed_seconds is whole seconds
    out["leak"] = {"pairs": int(j.size), "short_pairs": int(short.sum()),
                   "short_violate_incl": int(((es[j] * 1000 + slack < incl) & short).sum()),
                   "short_violate_excl": int(((es[j] * 1000 + slack < excl) & short).sum()),
                   "own_time_ge_10pct": int(((d[j] >= 0.1 * np.maximum(es[j], 1) * 1000) & (d[j] > 0)).sum()),
                   "own_time_ge_1pct": int(((d[j] >= 0.01 * np.maximum(es[j], 1) * 1000) & (d[j] > 0)).sum()),
                   "interval_ge_1_day": int((ed[j] >= 1).sum()),
                   "interval_ge_1_day_timed": int(((ed[j] >= 1) & (d[j] > 0)).sum()),
                   "back_to_back": int(((j == i + 1) & short).sum()),
                   "back_to_back_gap_within_1_5s_of_own_time": int(((j == i + 1) & short & (np.abs(es[j] * 1000 - d[j]) <= 1500)).sum()),
                   "short_gap_minus_incl_median_s": float(np.median((es[j] * 1000 - incl)[short]) / 1000) if short.any() else None,
                   "short_gap_minus_excl_median_s": float(np.median((es[j] * 1000 - excl)[short]) / 1000) if short.any() else None}
    return out


def _pooled_quantiles(hist, qs=(0.1, 0.5, 0.9)):
    hist = np.asarray(hist, float)
    cdf = np.cumsum(hist) / max(hist.sum(), 1)
    edges = [0.0] + [e / 1000 for e in BIN_EDGES_MS] + [float("inf")]
    out = []
    for q in qs:
        b = int(np.searchsorted(cdf, q))
        out.append(f"{edges[b]:.2g}-{edges[b + 1]:.2g}")
    return out


def audit(args):
    from multiprocessing import Pool
    data = data_path()
    users = json.loads(U732.read_text())["users"]
    started = time.time()
    with Pool(args.workers) as pool:
        rows = pool.map(_audit_user, [(data, u) for u in users], chunksize=4)
    tot = lambda k: sum(r[k] for r in rows)  # noqa: E731
    n, nf = tot("reviews"), tot("non_first")
    hist = np.sum([r["bin_hist"] for r in rows], axis=0)
    low_cap = [r for r in rows if r["cap_ms"] < CAP_MS]
    med = np.array([r["median_s"] for r in rows if r["median_s"] is not None])
    grade = {}
    for g in ("1", "2", "3", "4"):
        h = np.sum([r["grade"][int(g)]["hist"] for r in rows], axis=0)
        um = np.array([r["grade"][int(g)]["median_s"] for r in rows if r["grade"][int(g)]["median_s"] is not None
                       and r["grade"][int(g)]["n"] >= 20])
        grade[g] = {"reviews": int(h.sum()), "share_of_timed": round(float(h.sum() / hist.sum()), 4),
                    "pooled_bins_p10_p50_p90_s": _pooled_quantiles(h), "share_at_60s": round(float(h[-1] / max(h.sum(), 1)), 4),
                    "user_median_s_p10_p50_p90": [round(float(np.percentile(um, q)), 2) for q in (10, 50, 90)] if um.size else None,
                    "users": int(um.size)}
    var = [r for r in rows if "var_total" in r]
    within = np.array([r["var_within_card"] for r in var]); total = np.array([r["var_total"] for r in var])
    wts = np.array([r["n_var"] for r in var], float)
    # between-user share of log-time variance: pooled over users, user means vs grand mean (uncensored, timed)
    corr = np.array([r["corr_prev_card"] for r in rows if "corr_prev_card" in r])
    mae = np.array([r["mae_prev_card"] for r in rows if "mae_prev_card" in r])
    leak = {k: sum(r["leak"][k] for r in rows) for k in ("pairs", "short_pairs", "short_violate_incl", "short_violate_excl",
                                                         "own_time_ge_10pct", "own_time_ge_1pct", "interval_ge_1_day",
                                                         "interval_ge_1_day_timed", "back_to_back",
                                                         "back_to_back_gap_within_1_5s_of_own_time")}
    gm = [r["leak"]["short_gap_minus_incl_median_s"] for r in rows if r["leak"]["short_gap_minus_incl_median_s"] is not None]
    res = {"stage": "response-time prediction (SM82) stage 0 audit", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "users_file": str(U732), "users": len(rows), "dataset": str(data), "seconds": round(time.time() - started, 1),
           "reviews": n, "non_first_reviews": nf,
           "share_zero": round(tot("zero") / n, 5), "share_zero_non_first": round(tot("zero_non_first") / max(nf, 1), 5),
           "share_at_60s": round(tot("at_60s") / n, 5),
           "users_with_any_zero": sum(r["zero"] > 0 for r in rows),
           "users_with_lower_cap": len(low_cap), "lower_caps_s": sorted({r["cap_ms"] / 1000 for r in low_cap}),
           "reviews_censored_at_lower_cap": tot("censored"),
           "bin_edges_s": [round(e / 1000, 3) for e in BIN_EDGES_MS],
           "bin_share_non_first_timed": [round(float(x), 4) for x in hist / hist.sum()],
           "user_median_s_p10_p50_p90": [round(float(np.percentile(med, q)), 2) for q in (10, 50, 90)],
           "user_median_s_min_max": [round(float(med.min()), 2), round(float(med.max()), 2)],
           "grade": grade,
           "card_share_of_within_user_log_variance": round(float(1 - (within * wts).sum() / (total * wts).sum()), 3),
           "users_card_share_p10_p50_p90": [round(float(np.percentile(1 - within / total, q)), 3) for q in (10, 50, 90)],
           "corr_log_time_with_card_previous_p10_p50_p90": [round(float(np.percentile(corr, q)), 3) for q in (10, 50, 90)],
           "mae_log_s_card_previous_median_user": round(float(np.median(mae)), 3),
           "leak": {**leak, "short_violate_incl_share": round(leak["short_violate_incl"] / max(leak["short_pairs"], 1), 5),
                    "short_violate_excl_share": round(leak["short_violate_excl"] / max(leak["short_pairs"], 1), 5),
                    "own_time_ge_10pct_share": round(leak["own_time_ge_10pct"] / max(leak["pairs"], 1), 4),
                    "own_time_ge_1pct_share": round(leak["own_time_ge_1pct"] / max(leak["pairs"], 1), 4),
                    "interval_ge_1_day_share": round(leak["interval_ge_1_day"] / max(leak["pairs"], 1), 4),
                    "short_gap_minus_incl_median_s_users_median": round(float(np.median(gm)), 1) if gm else None},
           "per_user": [{k: r[k] for k in ("user", "reviews", "zero", "at_60s", "cap_ms", "median_s")} for r in rows]}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out.with_suffix(".json"), res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_user", "grade")}, indent=None)[:3000])
    print(json.dumps(grade)[:2000])
    return 0


# ------------------------------------------------------------------ extend / cool-down
def extend(args):
    import torch
    from expertium.variants import build
    old = torch.load(args.master, map_location="cpu", weights_only=True)
    base, model = build("w64").float(), build(ARCH).float()
    base.load_state_dict(old, strict=True)
    new_sd = model.state_dict()
    added = [k for k in new_sd if k not in old]
    if set(old) - set(new_sd) or any(not k.startswith("time_heads.") for k in added):
        raise SystemExit(f"{ARCH} is not w64 plus the time head: {sorted(set(old) ^ set(new_sd))[:10]}")
    generator = torch.Generator().manual_seed(args.seed)
    out_sd = {k: (old[k].clone() if k in old else
                  (torch.randn(v.shape, generator=generator) * args.perturb if args.perturb else torch.zeros_like(v)))
              for k, v in new_sd.items()}
    model.load_state_dict(out_sd, strict=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(out_sd, out / "master.pth")
    record = {"source_master": str(args.master), "source_master_sha256": sha256(args.master), "architecture": ARCH,
              "added": {k: list(new_sd[k].shape) for k in added}, "added_values": sum(new_sd[k].numel() for k in added),
              "init": f"normal(0, {args.perturb}) seed {args.seed} (check only)" if args.perturb else "zeros",
              "master_sha256": sha256(out / "master.pth"),
              "parameters": sum(v.numel() for v in out_sd.values()), "parameters_w64": sum(v.numel() for v in old.values())}
    atomic_json(out / "extend.json", record)
    print(json.dumps({k: v for k, v in record.items() if k != "added"}))
    return 0


def cooldown(args):
    """The reference's cool-down (scripts/sm77_warm.py cooldown with SM72s's epoch 21 as start) with ARCH and the
    extended start weights; the update offset is SM72s epoch 21's."""
    from scripts.expertium_decay import prepare
    out = (ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    extra = json.loads(args.extra_options)
    extra["architecture"] = ARCH
    code = prepare(RUN72S, args.epoch, out, args.fraction, "one_minus_sin", "none", 1, extra, 0.0, fresh_optimizer=True,
                   train_tier="final")
    if code:
        return code
    case = RUN72S / f"epoch-{args.epoch:02d}" / "expertium" / TRACK
    result = json.loads((case / "result.json").read_text())
    if result.get("status") != "complete":
        raise SystemExit(f"{case} is not a complete epoch")
    ext = Path(args.start).resolve()
    record = json.loads((ext / "extend.json").read_text())
    if Path(record["source_master"]).resolve() != (case / "master.pth").resolve() or record["init"] != "zeros":
        raise SystemExit(f"{ext} does not extend {case / 'master.pth'} with a zero head ({record['source_master']}, {record['init']})")
    protocol = json.loads((out / "protocol.json").read_text())
    decay = json.loads((out / "decay.json").read_text())
    protocol["full_run"].update(init_master=(ext / "master.pth").as_posix(), update_offset=result["update_offset_end"])
    decay.update(source_case=str(case), source_update_offset_end=result["update_offset_end"], start=str(ext),
                 start_master_sha256=sha256(ext / "master.pth"), architecture=ARCH,
                 sm82=f"users / order / schedule from {RUN72S.name} epoch {args.epoch}; start weights {case} + the zero "
                      f"answer-time head ({ext})")
    atomic_json(out / "protocol.json", protocol)
    atomic_json(out / "decay.json", decay)
    print(json.dumps({"out": out.name, "start": str(ext), "update_offset": result["update_offset_end"],
                      "training_users": decay["training_users"], "training_reviews": decay["training_reviews"]}))
    return 0


# ------------------------------------------------------------------ CPU proof
SM77_TREE = Path("C:/Users/jonat/workspace/wt/sm77")   # the code that trained the reference cool-down (read only)
DATA_DEFAULT = os.environ.get("GPU_BATCH1_DATA_PATH", r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011")

PROBE = r'''
import hashlib, json, os, sys
root = sys.argv[1]
sys.path.insert(0, root)
os.chdir(root)
import numpy as np
import torch
torch.set_num_threads(4)
from concurrent.futures import ThreadPoolExecutor
from scripts.gpu_r01_prefetch import training_batches
from scripts.expertium_options import CatchUp
from expertium.variants import build
spec = json.loads(open(sys.argv[2]).read())
model = build(spec["arch"]).float()
model.load_state_dict(torch.load(spec["checkpoint"], map_location="cpu", weights_only=True), strict=True)
params = dict(model.named_parameters())
catch_up = CatchUp(spec["options"], {}, spec["rows"])
catch_up.load_teacher(torch.device("cpu"), torch.float32)
with ThreadPoolExecutor(max_workers=1) as pool:
    prepared = list(training_batches(spec["rows"], spec["span"], spec["options"]["training_seed"], pool, spec["options"], None, 2))
digest = lambda t: hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()
frames = {}
out = []
for row, item in prepared:
    batch = item.batch.to(torch.device("cpu"))
    model.train()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(item.seed)
    catch_up.carry.attach(item, batch, row["user_id"])
    key = (row["user_id"], item.offset, item.reviews, item.seed)
    labels = item.batch.labels
    stats = catch_up.loss(model, batch, 0, key, labels)
    stats.average_loss.backward()
    rec = {"user": row["user_id"], "offset": item.offset, "loss_hex": float(stats.average_loss).hex(),
           "imm_hex": float(stats.imm_avg).hex(), "ahead_hex": float(stats.ahead_avg).hex(), "imm_n": stats.imm_n,
           "p_imm_all": digest(stats.p_imm_all), "p_curve": digest(stats.p_curve), "loss_tensor": digest(stats.loss_tensor),
           "time_avg": float(getattr(stats, "time_avg", torch.tensor(0.0))), "time_n": float(getattr(stats, "time_n", torch.tensor(0.0)))}
    torch.save({n: (p.grad.clone() if p.grad is not None else None) for n, p in params.items()}, spec["dump"] + f"-{len(out)}.pt")
    if spec.get("align"):
        # the time label of every labelled instant row = the time of the review it predicts (revlog order = review_th - 1)
        from scripts.expertium_options import time_labels
        import pandas as pd
        from scripts.gpu_batch1 import DATA
        user = row["user_id"]
        if user not in frames:
            frames[user] = pd.read_parquet(DATA / "revlogs" / f"user_id={user}", columns=["rating", "duration", "elapsed_days"])
        f = frames[user]
        lower, cens, _ = time_labels(f["duration"].to_numpy(), f["elapsed_days"].to_numpy())
        q = (batch.labels[..., 6] == 1) & (batch.labels[..., 4] == 1)
        th = batch.label_review_th[q].long().numpy()
        rating_ok = bool((batch.labels[..., 3][q].numpy() == f["rating"].to_numpy()[th - 1]).all())
        target = catch_up.time(batch, user)
        target_ok = bool((target[..., 0][q].numpy() == lower[th - 1]).all() and (target[..., 1][q].numpy() == cens[th - 1]).all())
        rec.update(align_rows=int(q.sum()), align_rating_ok=rating_ok, align_target_ok=target_ok,
                   labelled_rows=int((target[..., 0][q] >= 0).sum()))
    out.append(rec)
print("PROBE " + json.dumps(out))
'''

LEAK_PROBE = r'''
import json, os, sys
root = sys.argv[1]
sys.path.insert(0, root)
os.chdir(root)
os.environ["RWKV_NO_JIT"] = "1"
import numpy as np
import torch
torch.set_num_threads(4)
from expertium.variants import build
from rwkv.data_processing import get_rwkv_data, scale_duration
from rwkv.prepare_batch import prepare
from scripts import compress_common as cc
from scripts import sm71_leak
spec = json.loads(open(sys.argv[2]).read())
model = build(spec["arch"]).float().eval()
model.load_state_dict(torch.load(spec["checkpoint"], map_location="cpu", weights_only=True), strict=True)
user = spec["user"]
data = cc.data_path()
frame0 = get_rwkv_data(data, user)
frame0, _ = sm71_leak.neutralise(frame0, "neighbour", data, user)

def outputs(frame):
    batch = prepare(cc.make_samples(frame, user, len(frame)), seed=2025 + user)
    with torch.inference_mode():
        out = model.forward_batch_variant(batch.start, batch.sub_gather, batch.sub_gather_lens, batch.time_shift_selects,
                                          batch.skips, batch.num_data)
    return batch, out[3][0], out[6][0]

batch, p0, t0 = outputs(frame0)
th = batch.label_review_th[0].long().numpy()
is_query = batch.labels[0, :, 6].numpy() == 1
rng = np.random.default_rng(0)
cands = [j for j in range(len(frame0)) if frame0["elapsed_days"].iat[j] >= 0 and frame0["duration"].iat[j] > 0]
picks = sorted(rng.choice(cands, size=spec["picks"], replace=False).tolist())
res = []
for j in picks:
    frame = frame0.copy()
    col = frame.columns.get_loc("duration")
    new = int(frame0["duration"].iat[j]) * 3 + 1000
    frame.iloc[j, col] = new
    frame.iloc[j, frame.columns.get_loc("scaled_duration")] = scale_duration(np.float64(new))
    b2, p1, t1 = outputs(frame)
    review_th = j + 1
    qrow = int(np.flatnonzero(is_query & (th == review_th))[0])      # the instant row predicting review j
    upto = slice(0, qrow + 1)
    same_upto = bool(torch.equal(p0[upto], p1[upto]) and torch.equal(t0[upto], t1[upto]))
    later = slice(qrow + 1, None)
    changed_later = bool(not torch.equal(t0[later], t1[later])) if qrow + 1 < t0.shape[0] else None
    s0, s1 = batch.start.reshape(-1, batch.start.shape[-1]), b2.start.reshape(-1, b2.start.shape[-1])
    diff = s0 != s1
    changed_rows = torch.nonzero(diff.any(-1)).squeeze(-1).tolist()
    changed_cols = sorted(set(torch.nonzero(diff)[:, 1].tolist()))
    res.append({"review_th": review_th, "query_row": qrow, "rows": int(t0.shape[0]), "elapsed_days": int(frame0["elapsed_days"].iat[j]),
                "outputs_up_to_its_query_bitwise_equal": same_upto, "later_outputs_changed": changed_later,
                "input_rows_changed": changed_rows, "input_columns_changed": changed_cols,
                "changed_rows_after_query": all(r > qrow for r in changed_rows),
                "query_row_duration_input_zero": bool(float(s0[qrow, 8]) == 0.0)})
print("PROBE " + json.dumps(res))
'''


def run_probe(tree, code, spec, tmp, name, jit=False):
    probe = Path(tmp) / f"sm82_{name}.py"
    probe.write_text(code, encoding="utf-8")
    spec_file = Path(tmp) / f"spec-{name}.json"
    spec_file.write_text(json.dumps(spec), encoding="utf-8")
    env = dict(os.environ, GPU_BATCH1_DATA_PATH=DATA_DEFAULT, PYTHONPATH="", CUDA_VISIBLE_DEVICES="",
               PYTHONDONTWRITEBYTECODE="1", RWKV_DELETED_CARDS=spec.get("deleted_cards_env", "neighbour"))
    if spec.get("deleted_cards_env") == "":
        env.pop("RWKV_DELETED_CARDS")
    if jit:
        env.pop("RWKV_NO_JIT", None)
    else:
        env["RWKV_NO_JIT"] = "1"
    result = subprocess.run([sys.executable, "-X", "utf8", str(probe), str(tree), str(spec_file)], cwd=tree,
                            capture_output=True, text=True, env=env)
    line = next((x for x in result.stdout.splitlines() if x.startswith("PROBE ")), None)
    if line is None:
        raise RuntimeError(f"probe {name} failed in {tree}: {result.stderr[-3000:]}")
    return json.loads(line[6:])


def check(args):
    """CPU proof, before any GPU ticket:
    T0 flags off: this tree's w64 trainer path == the tiny architecture additions (SM77) tree's (the code that trained
       the reference cool-down): loss bits, outputs, every gradient, on real batches with the cool-down's options
       (KD replay targets, 2 x 128 strided lanes state carry; small chunks so it runs on CPU).
    T1 function preservation: w64-sm82 with the zero head (extend output) == w64: the grade terms (instant and curve
       averages), the per-row loss, the rating and curve outputs bitwise; every old weight's gradient bitwise
       (torch.equal); the head gets a gradient; the total loss = grade loss + 0.1 x time loss (about ln 16 at zero).
    T2 labels: every labelled instant row's time label is the time of the review it predicts (label_review_th into
       the revlog), its grade label matches the revlog rating, and only rows on a later day carry a time label.
    T3 no leak (scorer path, full history, random non-zero head): changing one review's answer time changes no
       output (grade or time) up to and including the instant row that predicts that review; later rows change.
    T4 JIT: the TorchScript build of w64-sm82 gives the same loss and outputs as eager on the first update.
    T5 optimizer: matrices of the head on Muon, its bias on AdamW (expertium-iter53 split)."""
    import tempfile
    import torch
    started = time.perf_counter()
    report = {"checks": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    ext = Path(args.ext).resolve()
    record = json.loads((ext / "extend.json").read_text())
    ck = Path(record["source_master"])
    report["extend"] = {k: v for k, v in record.items()}
    protocol = json.loads((RUN72S / "epoch-21" / "protocol.json").read_text())
    small = sorted((r for r in protocol["rows"] if 300 <= r["train_range"][1] <= 1500), key=lambda r: r["train_range"][1])
    rows = [small[0], next(r for r in small[len(small) // 2:] if r["user_id"] != small[0]["user_id"])]
    decay = json.loads(Path(args.decay_options).read_text()) if args.decay_options else None
    options = dict(decay["options"]) if decay else None
    if options is None:
        raise SystemExit("--decay-options DIR/decay.json (the reference cool-down's) is required")
    options = {k: v for k, v in options.items() if k not in ("packed_ipc",)}
    options.update(chunk_reviews=128, batch_chunks=2, architecture="w64")
    span = 256
    report["users"] = {str(r["user_id"]): r["train_range"][1] for r in rows}
    report["options"] = options
    fails = []
    with tempfile.TemporaryDirectory() as tmp:
        spec = {"arch": "w64", "checkpoint": str(ck), "options": options, "rows": rows, "span": span}
        ours = run_probe(ROOT, PROBE, dict(spec, dump=str(Path(tmp) / "ours")), tmp, "t0-ours")
        theirs = run_probe(SM77_TREE, PROBE, dict(spec, dump=str(Path(tmp) / "theirs")), tmp, "t0-theirs")
        names = [n for n in torch.load(Path(tmp) / "ours-0.pt", weights_only=True)]

        def grads_equal(a, b, n, only=None):
            out = []
            for i in range(n):
                ga, gb = (torch.load(Path(tmp) / f"{tag}-{i}.pt", weights_only=True) for tag in (a, b))
                ok = True
                for name in (only or names):
                    x, y = ga[name], gb[name]
                    if (x is None) != (y is None) or (x is not None and not torch.equal(x, y)):
                        ok = False
                out.append(ok)
            return out

        keys = ("user", "offset", "loss_hex", "imm_hex", "ahead_hex", "imm_n", "p_imm_all", "p_curve", "loss_tensor")
        same = [{k: u[k] for k in keys} for u in ours] == [{k: u[k] for k in keys} for u in theirs]
        g0 = grads_equal("ours", "theirs", len(ours))
        t0 = same and all(g0) and len(ours) > len(rows)
        report["checks"]["T0_flags_off_equals_sm77_tree"] = {"pass": t0, "updates": len(ours), "loss_outputs_bitwise": same,
                                                            "gradients_equal_per_update": g0,
                                                            "losses": [float.fromhex(u["loss_hex"]) for u in ours]}
        fails += [] if t0 else ["T0"]
        arm_options = dict(options, architecture=ARCH)
        arm = run_probe(ROOT, PROBE, dict(spec, arch=ARCH, checkpoint=str(ext / "master.pth"), options=arm_options,
                                          dump=str(Path(tmp) / "arm"), align=True), tmp, "t1-arm")
        gkeys = ("user", "offset", "imm_hex", "ahead_hex", "imm_n", "p_imm_all", "p_curve", "loss_tensor")
        same1 = [{k: u[k] for k in gkeys} for u in arm] == [{k: u[k] for k in gkeys} for u in ours]
        g1 = grads_equal("ours", "arm", len(arm))
        head_grad = []
        for i in range(len(arm)):
            g = torch.load(Path(tmp) / f"arm-{i}.pt", weights_only=True)
            head_grad.append({n: float(g[n].norm()) for n in g if n.startswith("time_heads.")})
        total_gap = [abs(float.fromhex(a["loss_hex"]) - float.fromhex(o["loss_hex"]) - 0.1 * a["time_avg"])
                     for a, o in zip(arm, ours)]
        labelled = [u for u in arm if u["time_n"] > 0]
        t1 = (same1 and all(g1) and all(h["time_heads.0.linear.bias"] > 0 for h, u in zip(head_grad, arm) if u["time_n"] > 0)
              and max(total_gap) < 1e-6 and len(labelled) > 0
              and all(abs(u["time_avg"] - math.log(BINS)) < 1e-5 for u in labelled))
        report["checks"]["T1_zero_head_grade_path_bitwise"] = {
            "pass": t1, "updates": len(arm), "grade_terms_outputs_bitwise": same1, "old_gradients_equal_per_update": g1,
            "head_gradient_norms": head_grad, "time_loss": [round(u["time_avg"], 7) for u in arm],
            "time_rows": [u["time_n"] for u in arm], "instant_rows": [u["imm_n"] for u in arm],
            "ln16": round(math.log(BINS), 7), "total_minus_grade_minus_0.1_time_max": max(total_gap)}
        fails += [] if t1 else ["T1"]
        t2 = all(u["align_rating_ok"] and u["align_target_ok"] for u in arm) and all(u["align_rows"] > 0 for u in arm)
        report["checks"]["T2_time_labels_aligned"] = {
            "pass": t2, "instant_rows": [u["align_rows"] for u in arm], "time_labelled_rows": [u["labelled_rows"] for u in arm],
            "rating_matches_revlog": [u["align_rating_ok"] for u in arm], "target_matches_revlog": [u["align_target_ok"] for u in arm]}
        fails += [] if t2 else ["T2"]
        leak = run_probe(ROOT, LEAK_PROBE, {"arch": ARCH, "checkpoint": str(Path(args.perturbed).resolve() / "master.pth"),
                                            "user": rows[1]["user_id"], "picks": 6, "deleted_cards_env": ""}, tmp, "t3-leak")
        t3 = (all(r["outputs_up_to_its_query_bitwise_equal"] and r["query_row_duration_input_zero"] for r in leak)
              and all(r["later_outputs_changed"] is not False for r in leak)
              and all(len(r["input_rows_changed"]) == 1 and r["input_columns_changed"] == [8] and r["changed_rows_after_query"]
                      for r in leak))
        report["checks"]["T3_no_time_leak_into_its_own_row"] = {"pass": t3, "user": rows[1]["user_id"], "picks": leak}
        fails += [] if t3 else ["T3"]
        jit_spec = dict(spec, arch=ARCH, checkpoint=str(ext / "master.pth"), options=arm_options,
                        rows=rows[:1], dump=str(Path(tmp) / "jit"))
        jit = run_probe(ROOT, PROBE, jit_spec, tmp, "t4-jit", jit=True)
        eager = arm[:len(jit)]
        jk = ("imm_hex", "ahead_hex", "p_imm_all", "p_curve")
        close = all(abs(float.fromhex(a["loss_hex"]) - float.fromhex(b["loss_hex"])) < 1e-6 and abs(a["time_avg"] - b["time_avg"]) < 1e-6
                    for a, b in zip(jit, eager))
        bit = [{k: a[k] for k in jk} == {k: b[k] for k in jk} for a, b in zip(jit, eager)]
        report["checks"]["T4_torchscript_build"] = {"pass": close, "updates": len(jit), "grade_bitwise_vs_eager": bit,
                                                    "loss_jit": [float.fromhex(u["loss_hex"]) for u in jit],
                                                    "loss_eager": [float.fromhex(u["loss_hex"]) for u in eager],
                                                    "time_jit": [u["time_avg"] for u in jit]}
        fails += [] if close else ["T4"]
    from expertium.model.variant_model import VariantSrsRWKV
    from expertium.variants import variant
    from scripts.expertium_options import CatchUp
    master = VariantSrsRWKV(variant(ARCH)).float()
    optimizer = CatchUp({"architecture": ARCH, "optimizer": "expertium-iter53"}, {}, [{"train_range": [0, 10]}]).optimizer(master)
    names = {id(p): n for n, p in master.named_parameters()}
    muon = [names[id(p)] for g in optimizer.muon.param_groups for p in g["params"] if names[id(p)].startswith("time_heads.")]
    adamw = [names[id(p)] for g in optimizer.adamw.param_groups for p in g["params"] if names[id(p)].startswith("time_heads.")]
    counts = [sum(p.numel() for g in optimizer.muon.param_groups for p in g["params"]),
              sum(p.numel() for g in optimizer.adamw.param_groups for p in g["params"])]
    t5 = muon == ["time_heads.0.linear.weight"] and adamw == ["time_heads.0.linear.bias"] and counts == [309888 + 8192, 21623 + 64]
    report["checks"]["T5_optimizer_split"] = {"pass": t5, "muon": muon, "adamw": adamw, "muon_adamw_values": counts,
                                              "parameters": sum(p.numel() for p in master.parameters())}
    fails += [] if t5 else ["T5"]
    report.update(passed=not fails, fails=fails, seconds=round(time.perf_counter() - started, 1))
    atomic_json(args.out, report)
    print(f"[sm82-check] {'PASS' if not fails else 'FAIL ' + ','.join(fails)} ({report['seconds']} s; {args.out})")
    return 0 if not fails else 1


# ------------------------------------------------------------------ test reviews, baselines, time scoring
def label_db():
    import lmdb
    from scripts import compress_common as cc
    return lmdb.open(str(cc.RUN / "label-db"), readonly=True, lock=False, map_size=10_000_000_000)


def test_reviews(data, user, txn):
    """The user's time-scored test reviews: the benchmark test reviews of the label DB (the grade scorer's "all") that
    carry a time label. Returns (review_th sorted, frame columns of the whole history, lower, censored, cap)."""
    import pandas as pd
    from scripts.compress_metrics import read_labels
    labels = read_labels(txn, user)
    frame = pd.read_parquet(data / "revlogs" / f"user_id={user}", columns=["card_id", "rating", "duration", "elapsed_days"])
    lower, cens, cap = time_labels(frame["duration"].to_numpy(), frame["elapsed_days"].to_numpy())
    if labels is None:
        return None, frame, lower, cens, cap
    ths = np.unique(np.asarray(labels[0], dtype=np.int64))
    ths = ths[(ths >= 1) & (ths <= len(frame))]
    ths = ths[lower[ths - 1] >= 0]
    return ths, frame, lower, cens, cap


LOG_EDGES = np.log(np.asarray(BIN_EDGES_MS) / 1000.0)   # ln seconds


def bin_log_centers():
    """ln-second point of each bin for medians: bin 0 = 0.35 s, bins 1..14 the geometric centres, bin 15 = 60 s."""
    e = LOG_EDGES
    return np.r_[math.log(0.35), (e[:-1] + e[1:]) / 2, e[-1]]


def median_log_s(prob):
    """Median of a [N, 16] bin distribution in ln seconds: linear in ln time inside the crossing bin (bin 0 from
    ln 0.25 s, bin 15 = its lower edge, 60 s)."""
    lo = np.r_[math.log(0.25), LOG_EDGES]
    hi = np.r_[LOG_EDGES, LOG_EDGES[-1]]
    cdf = np.cumsum(prob, axis=1)
    b = np.minimum((cdf < 0.5).sum(axis=1), BINS - 1)
    before = np.where(b > 0, np.take_along_axis(cdf, np.maximum(b - 1, 0)[:, None], 1)[:, 0], 0.0)
    pb = np.take_along_axis(prob, b[:, None], 1)[:, 0]
    frac = np.clip((0.5 - before) / np.maximum(pb, 1e-12), 0, 1)
    return lo[b] + frac * (hi[b] - lo[b])


def censored_ll(prob, lower, cens):
    """-ln P(label): the bin, or (censored) the bin and every later one; prob [N, 16], floored at 1e-7."""
    p = np.maximum(prob, 1e-7)
    p = p / p.sum(axis=1, keepdims=True)
    tail = np.cumsum(p[:, ::-1], axis=1)[:, ::-1]
    picked = np.where(cens > 0, np.take_along_axis(tail, lower[:, None], 1)[:, 0], np.take_along_axis(p, lower[:, None], 1)[:, 0])
    return -np.log(picked)


class RunningMedian:
    """Exact running median (two heaps) of a stream; median() of the values pushed so far (None if empty)."""

    def __init__(self):
        self.low, self.high = [], []   # max-heap (negated), min-heap

    def push(self, x):
        import heapq
        if not self.low or x <= -self.low[0]:
            heapq.heappush(self.low, -x)
        else:
            heapq.heappush(self.high, x)
        if len(self.low) > len(self.high) + 1:
            heapq.heappush(self.high, -heapq.heappop(self.low))
        elif len(self.high) > len(self.low):
            heapq.heappush(self.low, -heapq.heappop(self.high))

    def median(self):
        if not self.low:
            return None
        if len(self.low) > len(self.high):
            return -self.low[0]
        return (-self.low[0] + self.high[0]) / 2


def baseline_points(frame):
    """Per review (whole history, ln seconds, NaN = none): the card's last recorded time and the user's running median
    per grade of every earlier recorded time (falling back to the running median over all grades, then 8 s)."""
    d = frame["duration"].to_numpy(np.int64)
    card = frame["card_id"].to_numpy(np.int64)
    rating = frame["rating"].to_numpy(np.int64)
    n = len(d)
    logd = np.log(np.maximum(d, 1) / 1000.0)
    # card's last recorded (non-zero) time before this review
    last = np.full(n, np.nan)
    order = np.lexsort((np.arange(n), card))
    cs, ds = card[order], np.where(d[order] > 0, logd[order], np.nan)
    import pandas as pd
    prev = pd.Series(ds).groupby(cs).shift(1)
    prev = prev.groupby(cs).ffill().to_numpy()
    last[order] = prev
    med = np.full(n, np.nan)
    per = {g: RunningMedian() for g in (1, 2, 3, 4)}
    every = RunningMedian()
    for i in range(n):
        m = per[rating[i]].median() if rating[i] in per else None
        if m is None:
            m = every.median()
        med[i] = m if m is not None else math.log(8.0)
        if d[i] > 0:
            per.get(rating[i], every).push(logd[i])
            every.push(logd[i])
    last = np.where(np.isnan(last), med, last)
    return last, med


def _baseline_user(args):
    data, user = args
    env = label_db()
    with env.begin() as txn:
        ths, frame, lower, cens, cap = test_reviews(data, user, txn)
    env.close()
    last, med = baseline_points(frame)
    d = frame["duration"].to_numpy(np.int64)
    grade = frame["rating"].to_numpy(np.int64) - 1
    labelled = np.flatnonzero(lower >= 0)
    is_test = np.zeros(len(frame), bool)
    if ths is not None:
        is_test[ths - 1] = True
    fit = labelled[~is_test[labelled] & (d[labelled] < cap)]   # residuals from the non-test, uncensored labelled reviews
    logd = np.log(np.maximum(d, 1) / 1000.0)
    res = {"user": user, "cap_ms": cap}
    for name, mu in (("card_last", last), ("user_median", med)):
        res[f"fit_{name}"] = (grade[fit].astype(np.int8), (logd[fit] - mu[fit]).astype(np.float32))
    t = (ths - 1) if ths is not None else np.zeros(0, np.int64)
    res["test"] = {"review_th": ths if ths is not None else np.zeros(0, np.int64), "grade": grade[t].astype(np.int8),
                   "lower": lower[t].astype(np.int8), "cens": cens[t].astype(np.int8), "logd": logd[t].astype(np.float32),
                   "uncensored": (d[t] < CAP_MS) & (cens[t] == 0), "card_last": last[t].astype(np.float32),
                   "user_median": med[t].astype(np.float32)}
    return res


GRID = np.arange(-10.0, 10.0 + 1e-9, 0.02)   # residual CDF grid, ln seconds


def residual_cdf(residuals):
    hist = np.histogram(np.clip(residuals, GRID[0], GRID[-1] - 1e-9), bins=GRID)[0].astype(np.float64) + 0.5
    return np.r_[0.0, np.cumsum(hist) / hist.sum()]


def baseline_prob(mu, cdf):
    """[N, 16] bin probabilities of a point prediction mu (ln s) with a residual CDF on GRID."""
    edges = LOG_EDGES[None, :] - mu[:, None]
    f = np.interp(edges, GRID, cdf)
    return np.diff(np.c_[np.zeros(len(mu)), f, np.ones(len(mu))], axis=1)


def user_sums(prob, t, mask=None):
    """Per-user sums for the rule: censored LogLoss and |ln error| of the median (uncensored), overall and by grade."""
    ll = censored_ll(prob, t["lower"].astype(np.int64), t["cens"])
    err = np.abs(median_log_s(prob) - t["logd"])
    unc = t["uncensored"]
    out = {"n": int(len(ll)), "ll": float(ll.sum()), "n_unc": int(unc.sum()), "ae": float(err[unc].sum())}
    for g in range(4):
        m = t["grade"] == g
        out[f"g{g + 1}"] = [int(m.sum()), float(ll[m].sum()), int((m & unc).sum()), float(err[m & unc].sum())]
    return out


def baselines(args):
    from multiprocessing import Pool
    data = data_path()
    users = json.loads(U732.read_text())["users"]
    started = time.time()
    with Pool(args.workers) as pool:
        rows = pool.map(_baseline_user, [(data, u) for u in users], chunksize=2)
    cdfs = {}
    for name in ("card_last", "user_median"):
        g = np.concatenate([r[f"fit_{name}"][0] for r in rows]); x = np.concatenate([r[f"fit_{name}"][1] for r in rows])
        cdfs[name] = {k: residual_cdf(x[g == k]) for k in range(4)}
    out = []
    for r in rows:
        t = r["test"]
        rec = {"user_id": r["user"], "cap_ms": r["cap_ms"], "reviews": int(len(t["review_th"])),
               "review_th_sha256": hashlib.sha256(np.asarray(t["review_th"], np.int64).tobytes()).hexdigest()}
        for name in ("card_last", "user_median"):
            prob = np.zeros((len(t["grade"]), BINS))
            for k in range(4):
                m = t["grade"] == k
                if m.any():
                    prob[m] = baseline_prob(t[name][m].astype(np.float64), cdfs[name][k])
            rec[name] = user_sums(prob, t) if len(prob) else {"n": 0}
        out.append(rec)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    np.savez_compressed(path.with_suffix(".cdf.npz"), grid=GRID,
                        **{f"{n}_g{k + 1}": cdfs[n][k] for n in cdfs for k in range(4)})
    n = sum(r["reviews"] for r in out)
    summary = {name: {"ll": sum(r[name]["ll"] for r in out if r["reviews"]) / n,
                      "mae_ln_s": sum(r[name]["ae"] for r in out if r["reviews"]) / sum(r[name]["n_unc"] for r in out if r["reviews"])}
               for name in ("card_last", "user_median")}
    print(f"[sm82-baselines] {len(out)} users, {n:,} test reviews with a time label, {time.time() - started:.0f} s: "
          + json.dumps({k: {m: round(v, 4) for m, v in s.items()} for k, s in summary.items()}))
    return 0


def score(args):
    """The head's time distribution at every time-scored test review (GPU fp32, full history, leak-free inputs:
    --deleted-cards neighbour, id / day codes seed 2025 + user, as the grade scorer)."""
    import torch
    from scripts import small_model_heldout as smh   # sets RWKV_NO_JIT; the scorer's long WKV
    from scripts import compress_common as cc
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("HELDOUT_GPU_MEMORY_FRACTION", "0.8")))
    state = {k: v.float() for k, v in torch.load(args.master, map_location="cpu", weights_only=True).items()}
    model = smh.build_expertium(ARCH, state).to(device)
    smh.set_expertium_state_bits(model, 0)   # fp32 state (the time head is scored in float; the grade scorer uses w8+s8)
    users = json.loads(Path(args.users_file).read_text())["users"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["user_id"] for line in out.read_text().splitlines() if "error" not in line}
    base = {json.loads(line)["user_id"]: json.loads(line) for line in Path(args.baselines).read_text().splitlines()}
    todo = [u for u in users if u not in done]
    data = cc.data_path()
    counts = cc.run_info()["review_counts"]
    dataset = smh.LeakBatches(todo, data, counts, "neighbour", 0)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.workers, collate_fn=smh.first,
                                         prefetch_factor=1 if args.workers else None)
    env = label_db()
    started = time.perf_counter()
    print(f"[sm82-score] {len(users)} users, {len(todo)} to score, {args.master}", flush=True)
    with out.open("a", encoding="utf-8") as f:
        for k, (user, batch, error, prep) in enumerate(loader):
            if batch is None:
                f.write(json.dumps({"user_id": user, "error": error}) + "\n"); f.flush()
                continue
            with env.begin() as txn:
                ths, frame, lower, cens, cap = test_reviews(data, user, txn)
            try:
                gb = batch.to(device)
                with torch.inference_mode():
                    o = model.forward_batch_variant(gb.start, gb.sub_gather, gb.sub_gather_lens, gb.time_shift_selects,
                                                    gb.skips, gb.num_data)
                p_logits, t_logits = o[3][0].float(), o[6][0].float()
                labels = gb.labels[0]
                q = (labels[:, 6] == 1) & (labels[:, 4] == 1)
                rows_th = gb.label_review_th[0].long()
                qi = torch.nonzero(q).squeeze(-1)
                q_th = rows_th[qi]
                order = torch.argsort(q_th)
                q_th, qi = q_th[order], qi[order]
                want = torch.as_tensor(ths, device=device)
                pos = torch.searchsorted(q_th, want).clamp(max=len(q_th) - 1)
                found = q_th[pos] == want
                if not bool(found.all()):
                    raise RuntimeError(f"{int((~found).sum())} test reviews without an instant row")
                rows = qi[pos]
                grade = torch.as_tensor(frame["rating"].to_numpy()[ths - 1] - 1, device=device)
                pg = torch.softmax(p_logits[rows], -1)                                     # [N, 4]
                pt = torch.softmax(t_logits[rows].view(-1, 4, BINS), -1)                    # [N, 4, 16]
                cond = pt[torch.arange(len(rows), device=device), grade].double().cpu().numpy()
                marg = (pg.unsqueeze(-1) * pt).sum(1).double().cpu().numpy()
                under = cond[:, :7].sum(1)                                                  # P(t < 5.48 s | grade)
                del gb, o
            except torch.OutOfMemoryError as oom:
                torch.cuda.empty_cache()
                f.write(json.dumps({"user_id": user, "error": f"OOM: {oom}"[:300]}) + "\n"); f.flush()
                continue
            d = frame["duration"].to_numpy(np.int64)
            t = {"grade": grade.cpu().numpy().astype(np.int8), "lower": lower[ths - 1].astype(np.int8),
                 "cens": cens[ths - 1].astype(np.int8), "logd": np.log(np.maximum(d[ths - 1], 1) / 1000.0),
                 "uncensored": (d[ths - 1] < CAP_MS) & (cens[ths - 1] == 0)}
            sha = hashlib.sha256(np.asarray(ths, np.int64).tobytes()).hexdigest()
            b = base.get(user, {})
            rec = {"user_id": user, "reviews": int(len(ths)), "review_th_sha256": sha,
                   "baseline_same_reviews": b.get("review_th_sha256") == sha, "checkpoint": str(args.master),
                   "arch": ARCH, "precision": "fp32", "deleted_cards": "neighbour", "prepare_seconds": round(prep, 2),
                   "model": user_sums(cond, t), "model_marginal": user_sums(marg, t),
                   "under_5_5s": [[int((t["grade"] == g).sum()), float(under[t["grade"] == g].sum()),
                                   int(((t["grade"] == g) & (t["lower"] <= 6) & (t["cens"] == 0)).sum())] for g in range(4)]}
            f.write(json.dumps(rec) + "\n"); f.flush()
            if (k + 1) % 50 == 0:
                print(f"[sm82-score] {k + 1}/{len(todo)} users, {time.perf_counter() - started:.0f} s", flush=True)
    env.close()
    print(f"[sm82-score] done {len(todo)} users in {time.perf_counter() - started:.0f} s -> {out}", flush=True)
    return 0


# ------------------------------------------------------------------ decision
GRADE_MAX = 0.03   # points of % of the big model (1 leak-free epoch), each head


def decide(args):
    from scripts import sm77_rule
    users732 = set(json.loads(U732.read_text())["users"])
    big = sm77_rule.load(*sm77_rule.BIG, users732)
    pool = sorted(users732 & set(big))
    arm = sm77_rule.load(Path(args.arm_rows), args.arm_label, set(big)) if Path(args.arm_rows).exists() else {}
    ref = sm77_rule.load(Path(args.ref_rows), args.ref_label, set(big))
    missing = [u for u in pool if u not in arm or u not in ref]
    trows = {}
    if Path(args.time_rows).exists():
        trows = {r["user_id"]: r for r in map(json.loads, Path(args.time_rows).read_text().splitlines()) if "error" not in r}
    base = {r["user_id"]: r for r in map(json.loads, Path(args.baselines).read_text().splitlines())}
    tmissing = [u for u in pool if u not in trows]
    if missing or tmissing:
        print(f"[sm82-rule] grade rows missing for {len(missing)} users ({missing[:5]}), time rows for {len(tmissing)} ({tmissing[:5]})")
        return 2
    res = {"grade": sm77_rule.compare(arm, ref, big, pool)}
    tusers = [u for u in pool if trows[u]["reviews"] > 0]
    misaligned = [u for u in tusers if not trows[u]["baseline_same_reviews"]]
    if misaligned:
        print(f"[sm82-rule] REFUSE: {len(misaligned)} users' time-scored reviews differ from the baselines' ({misaligned[:5]})")
        return 1
    idx = np.random.default_rng(0).integers(0, len(tusers), size=(2000, len(tusers)))
    n = np.array([trows[u]["model"]["n"] for u in tusers], float)
    nu = np.array([trows[u]["model"]["n_unc"] for u in tusers], float)
    ll = {"model": np.array([trows[u]["model"]["ll"] for u in tusers]),
          "model_marginal": np.array([trows[u]["model_marginal"]["ll"] for u in tusers])}
    ae = {"model": np.array([trows[u]["model"]["ae"] for u in tusers]),
          "model_marginal": np.array([trows[u]["model_marginal"]["ae"] for u in tusers])}
    for b in ("card_last", "user_median"):
        ll[b] = np.array([base[u][b]["ll"] for u in tusers]); ae[b] = np.array([base[u][b]["ae"] for u in tusers])
    time_res = {"users": len(tusers), "reviews": int(n.sum()), "uncensored": int(nu.sum()),
                "time_logloss": {k: round(float(v.sum() / n.sum()), 5) for k, v in ll.items()},
                "mae_ln_s": {k: round(float(v.sum() / nu.sum()), 5) for k, v in ae.items()}, "vs": {}}
    beats = True
    for b in ("card_last", "user_median"):
        row = {}
        for metric, vals, w in (("time_logloss", ll, n), ("mae_ln_s", ae, nu)):
            delta = float((vals["model"] - vals[b]).sum() / w.sum())
            boot = ((vals["model"] - vals[b])[idx]).sum(1) / w[idx].sum(1)
            ci = [round(float(np.percentile(boot, q)), 5) for q in (2.5, 97.5)]
            better = int(((vals["model"] - vals[b]) / np.maximum(w, 1) < 0).sum())
            row[metric] = {"delta": round(delta, 5), "ci": ci, "relative_pct": round(100 * delta / float(vals[b].sum() / w.sum()), 2),
                           "users_better": better}
            beats &= delta < 0 and ci[1] < 0
        time_res["vs"][b] = row
    by_grade = {}
    for g in range(4):
        key = f"g{g + 1}"
        c = sum(trows[u]["model"][key][0] for u in tusers)
        by_grade[str(g + 1)] = {"reviews": c, **{k: round(sum((trows[u][k] if k.startswith("model") else base[u][k])[key][1] for u in tusers) / max(c, 1), 4)
                                               for k in ("model", "card_last", "user_median")}}
        under = [sum(trows[u]["under_5_5s"][g][i] for u in tusers) for i in range(3)]
        by_grade[str(g + 1)]["p_under_5_5s_predicted_vs_actual"] = [round(under[1] / max(under[0], 1), 4), round(under[2] / max(under[0], 1), 4)]
    time_res["by_grade_time_logloss"] = by_grade
    res["time"] = time_res
    grade_ok = all(res["grade"][h]["points"] <= GRADE_MAX for h in ("curve", "instant"))
    res["grade_ok"], res["time_beats_baselines"] = grade_ok, bool(beats)
    res["verdict"] = "pass" if grade_ok and beats else ("fail-grade" if not grade_ok else "fail-time")
    res["rule"] = {"grade": f"curve and instant each <= +{GRADE_MAX} points of % of the big model (1 leak-free epoch) vs the reference",
                   "time": "model beats both baselines on the censored time LogLoss and on the mean |ln error| (uncensored), "
                           "95% CI of the review-weighted difference clear of zero (paired user bootstrap 2,000, seed 0)",
                   "time_reviews": "benchmark test reviews of the 722 users with a recorded time, on a later day than the "
                                   "card's previous review; conditional on the true grade"}
    res.update(arm_rows=str(args.arm_rows), arm_label=args.arm_label, ref_rows=str(args.ref_rows), ref_label=args.ref_label,
               time_rows=str(args.time_rows), baselines=str(args.baselines))
    a, r = sm77_rule.first_row(args.arm_rows, args.arm_label), sm77_rule.first_row(args.ref_rows, args.ref_label)
    res["size"] = {"parameters_arm": a.get("parameters"), "parameters_reference": r.get("parameters"),
                   "weight_bytes_arm": a.get("weight_bytes"), "weight_bytes_reference": r.get("weight_bytes"),
                   "weight_bytes_change": (a.get("weight_bytes") or 0) - (r.get("weight_bytes") or 0)}
    HERE.mkdir(parents=True, exist_ok=True)
    atomic_json(HERE / "sm82-decision.json", res)
    gr, tr = res["grade"], res["time"]
    md = ["# Response-time prediction (SM82) stage 1: answer-time head vs the same recipe without it", "",
          "Grade: leak-free scoring, w8+s8, 722 users of the 732 set, review-weighted LogLoss, points of % of the big model "
          "(1 leak-free epoch), paired user bootstrap 2,000 seed 0, 95% CI. Arm = the leak-free small model (SM72s) epoch 21 "
          "+ the quarter-epoch 32k Muon cool-down with the answer-time head; reference = the same cool-down without it "
          "(the tiny architecture additions (SM77) reference).", "",
          "| head | arm | reference | delta | points [95% CI] | users better | bar |", "|---|---|---|---|---|---|---|"]
    for h in ("curve", "instant"):
        x = gr[h]
        md.append(f"| {h} | {x['arm']:.5f} | {x['reference']:.5f} | {x['delta']:+.6f} | {x['points']:+.2f} [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}] | "
                  f"{x['users_better']} | <= +{GRADE_MAX:.2f} |")
    md += ["", f"Time ({tr['users']} users, {tr['reviews']:,} test reviews with a recorded time on a later day than the card's "
           f"previous review, {tr['uncensored']:,} under the cap): censored 16-bin time LogLoss given the true grade, and the mean "
           "|ln error| of the predicted median (uncensored reviews). Baselines get a residual distribution per grade fit on the "
           "users' non-test reviews.", "",
           "| predictor | time LogLoss | mean abs ln error |", "|---|---|---|"]
    for k, label in (("model", "answer-time head (given the true grade)"), ("model_marginal", "answer-time head (grade mixed by the grade head)"),
                     ("card_last", "baseline: the card's last time"), ("user_median", "baseline: the user's running median per grade")):
        md.append(f"| {label} | {tr['time_logloss'][k]:.4f} | {tr['mae_ln_s'][k]:.4f} |")
    md += ["", "| head vs | LogLoss delta [95% CI] | relative | abs ln error delta [95% CI] | relative |", "|---|---|---|---|---|"]
    for b, label in (("card_last", "the card's last time"), ("user_median", "the user's running median per grade")):
        x, y = tr["vs"][b]["time_logloss"], tr["vs"][b]["mae_ln_s"]
        md.append(f"| {label} | {x['delta']:+.4f} [{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}] | {x['relative_pct']:+.1f}% | "
                  f"{y['delta']:+.4f} [{y['ci'][0]:+.4f}, {y['ci'][1]:+.4f}] | {y['relative_pct']:+.1f}% |")
    md += ["", "| grade | reviews | head LogLoss | card's last time | user's median per grade | P(under 5.5 s) predicted / actual |",
           "|---|---|---|---|---|---|"]
    for g, x in by_grade.items():
        md.append(f"| {g} | {x['reviews']:,} | {x['model']:.4f} | {x['card_last']:.4f} | {x['user_median']:.4f} | "
                  f"{x['p_under_5_5s_predicted_vs_actual'][0]:.3f} / {x['p_under_5_5s_predicted_vs_actual'][1]:.3f} |")
    s = res["size"]
    if s["parameters_arm"]:
        md += ["", f"Size: {s['parameters_reference']:,} -> {s['parameters_arm']:,} parameters; scorer w8 weight bytes "
               f"{s['weight_bytes_reference']:,} -> {s['weight_bytes_arm']:,} ({s['weight_bytes_change']:+,} B); state unchanged."]
    md += ["", f"Verdict: **{res['verdict']}** (grade within the bar: {grade_ok}; time head beats both baselines: {bool(beats)})."]
    (HERE / "sm82-decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[sm82-rule] {len(pool)} users: curve {gr['curve']['points']:+.2f} points {gr['curve']['ci']} instant "
          f"{gr['instant']['points']:+.2f} points {gr['instant']['ci']} (arm - reference); time LogLoss {tr['time_logloss']['model']:.4f} "
          f"vs card's last {tr['time_logloss']['card_last']:.4f} / user median {tr['time_logloss']['user_median']:.4f}, |ln err| "
          f"{tr['mae_ln_s']['model']:.4f} vs {tr['mae_ln_s']['card_last']:.4f} / {tr['mae_ln_s']['user_median']:.4f}: {res['verdict'].upper()}")
    return 0


def main():
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("audit")
    p.add_argument("--out", default=str(HERE / "stage0-audit"))
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("extend")
    p.add_argument("--master", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--perturb", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("cooldown")
    p.add_argument("--epoch", type=int, default=21)
    p.add_argument("--start", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fraction", type=float, default=0.25)
    p.add_argument("--extra-options", required=True)
    p = sub.add_parser("check")
    p.add_argument("--ext", required=True)
    p.add_argument("--perturbed", required=True, help="an extend --perturb output (random head) for the leak test")
    p.add_argument("--decay-options", required=True, help="the reference cool-down's decay.json")
    p.add_argument("--out", required=True)
    p = sub.add_parser("baselines")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("score")
    p.add_argument("--master", required=True)
    p.add_argument("--baselines", required=True)
    p.add_argument("--users-file", default=str(U732))
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=3)
    p = sub.add_parser("decide")
    p.add_argument("--time-rows", required=True)
    p.add_argument("--baselines", required=True)
    p.add_argument("--arm-rows", required=True)
    p.add_argument("--arm-label", required=True)
    p.add_argument("--ref-rows", required=True)
    p.add_argument("--ref-label", required=True)
    args = parser.parse_args()
    return {"audit": audit, "extend": extend, "cooldown": cooldown, "check": check, "baselines": baselines,
            "score": score, "decide": decide}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
