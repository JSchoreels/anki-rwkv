#!/usr/bin/env python
"""Fast Good/Easy prediction (SM120): scoring only, no training. The answer-time heads of the time heads for the family
(SM117) and of the time heads for the new candidates (SM119) on their frozen models, scored on the 732 set (723 users,
show-time inputs, leak-free, the SM117 time-labelled test reviews) for new events (artifacts/small-model/sm120/design.md):

  grade sets   Rec = {Hard, Good, Easy} (SM117), GE = {Good, Easy}, E = {Easy}
  1. grade     P(set) from the model's grade probabilities vs the user's running rate of the set
  2. fast      P(set and t <= S) = sum_{g in set} P(g) P(t <= S | g), S in S_GRID; baselines (a) model P(g) x the user's
               running share under S of grade g, summed over the set, and (b) the user's running rate of the event
  3. speed     on reviews of actual grade g (Hard, Good, Easy): P(t <= S | g) vs the user's running share under S of g;
               |ln error| of the predicted median vs the user's running median of g (SM117's user-median baseline)
  4. mixture   P(t) = sum_g P(g) P(t | g) (all grades) and P(t | GE) = sum_{g in GE} P(g) P(t | g) / P(GE): censored
               LogLoss over the 16 bins, |ln error| of the median and of the expected ln time, vs (a) the same mixture
               of the user's running per-grade bin distributions and (b) the user's running bin distribution
               (all grades / Good+Easy only); all-grade mixture on all rows and on Good/Easy-graded rows, GE mixture on
               Good/Easy-graded rows

Answer time = front + back: question shown -> grade pressed, capped at the preset's maximum answer seconds ("within S s"
always includes reading the back). S counts only below the user's own cap (SM117's `valid`). Grades are not touched
(the frozen trunk is checked equal to the release before scoring, as SM117).

  priors   --out F                  pooled priors of the running baselines on the SM117 plan's training users (CPU)
  score    --model KEY --head-dir D --out R    one JSON record per user + fine histograms in R.hist/<user>.npz
  check    --out F                  the CPU check (3 small users, Rec reproduces SM117's per-user sums)
  report                            all scored models -> artifacts/small-model/sm120/sm120-result.{md,json}
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

from scripts import sm117_head as h117  # noqa: E402
from scripts.sm117_head import (MODELS, ORDER, HIST_BINS, LOG_LO, LOG_HI, PRIOR_M, INPUTS, U732,  # noqa: E402
                                verified_master, build_model, Frozen, load_head, ScoreBatches, cdf_at, binary_sums,
                                hist_of, ece_auc_from_hist, pool_users, bootstrap_idx, _diff)
from scripts.sm117_timelib import (BINS, CAP_MS, EXPLICIT_DIM, label_db, test_reviews, first, median_log_s,  # noqa: E402
                                   censored_ll, atomic_json, time_labels)

SIN = ROOT / "artifacts/small-model/sm117"          # SM117 inputs (plan, baselines), read only
OUT = ROOT / "artifacts/small-model/sm120"
S_GRID = (2, 3, 4, 5, 6, 8, 10, 15, 20, 30)          # seconds, pre-registered (design.md)
S117 = h117.S_GRID
SETS = {"Rec": (1, 2, 3), "GE": (2, 3), "E": (3,)}  # 0-based grades: 0 Again, 1 Hard, 2 Good, 3 Easy
SET_NAMES = {"Rec": "recalled (Hard, Good or Easy)", "GE": "Good or Easy", "E": "Easy"}
SPEED_GRADES = {"Hard": 1, "Good": 2, "Easy": 3}
LN_VAL = np.r_[(LOG_LO[:-1] + LOG_HI[:-1]) / 2, math.log(60.0)]   # bin value in ln s (bin 15 = the 60 s cap)
HEAD_DIRS = {k: Path(f"C:/Users/jonat/workspace/wt/sm117/artifacts/small-model/sm117-{k}-head")
             for k in ("sm106", "sm98", "sm101", "sm99", "sm91s2", "sm92")}
HEAD_DIRS.update({k: Path(f"C:/Users/jonat/workspace/wt/sm119/artifacts/small-model/sm119-{k}-head") for k in ("sm113", "sm115", "sm118")})
SM117_ROWS = Path("C:/Users/jonat/workspace/wt/sm117/artifacts/small-model/sm117/gpu")


# ------------------------------------------------------------------ priors (CPU)
def _prior_user(args):
    data, user = args
    import pandas as pd
    f = pd.read_parquet(Path(data) / "revlogs" / f"user_id={user}", columns=["rating", "duration", "elapsed_days"])
    d, e, r = f["duration"].to_numpy(np.int64), f["elapsed_days"].to_numpy(np.int64), f["rating"].to_numpy(np.int64)
    lower, _, _ = time_labels(d, e, same_day=True)
    lab = lower >= 0
    return {"n": int(lab.sum()), "n_g": [int((lab & (r == g)).sum()) for g in (1, 2, 3, 4)],
            "under_g": [[int((lab & (r == g) & (d <= 1000 * s)).sum()) for s in S_GRID] for g in (1, 2, 3, 4)],
            "bins_g": [np.bincount(lower[lab & (r == g)], minlength=BINS).tolist() for g in (1, 2, 3, 4)]}


def priors(args):
    """Pooled rates of the running baselines, SM117's rule (the plan's 1,991 training users, none in the 732 pool,
    whole histories, time-labelled reviews), on the SM120 S grid, per grade set, plus per-grade bin distributions."""
    from multiprocessing import Pool
    from scripts import compress_common as cc
    order = [r["user_id"] for r in json.loads((SIN / "plan.json").read_text())["order"]]
    data = str(cc.data_path())
    started = time.time()
    with Pool(args.workers) as p:
        res = p.map(_prior_user, [(data, u) for u in order], chunksize=8)
    n = sum(x["n"] for x in res)
    ng = np.array([sum(x["n_g"][g] for x in res) for g in range(4)], np.float64)
    under = np.array([[sum(x["under_g"][g][k] for x in res) for k in range(len(S_GRID))] for g in range(4)], np.float64)
    bins = np.array([[sum(x["bins_g"][g][b] for x in res) for b in range(BINS)] for g in range(4)], np.float64)
    rec = {"users": len(order), "source": "the SM117 plan's training users (none in the 732 pool), whole histories, "
           "time-labelled reviews (SM117 rule)", "S": list(S_GRID), "pseudo_count": PRIOR_M, "reviews": int(n),
           "reviews_per_grade": ng.astype(int).tolist(),
           "F0_under_S_given_grade": (under / ng[:, None]).round(6).tolist(),
           "E0_event": {k: (under[list(g)].sum(0) / n).round(6).tolist() for k, g in SETS.items()},
           "R0_set": {k: round(float(ng[list(g)].sum() / n), 6) for k, g in SETS.items()},
           "B0_bins_given_grade": (bins / ng[:, None]).round(8).tolist(),
           "B0_bins_all": (bins.sum(0) / n).round(8).tolist(),
           "B0_bins_GE": (bins[[2, 3]].sum(0) / ng[[2, 3]].sum()).round(8).tolist(),
           "seconds": round(time.time() - started, 1)}
    atomic_json(args.out, rec)
    print(f"[sm120-priors] {len(order)} users, {n:,} reviews, rate GE {rec['R0_set']['GE']}, E {rec['R0_set']['E']} -> {args.out}")
    return 0


# ------------------------------------------------------------------ causal running baselines
def running(rev, t, pri):
    """Per test review t (revlog index), from the user's earlier time-labelled reviews (SM117 rule) only, with
    pseudo-count M at the pooled priors: F [n, 4, S] share under S of each grade, E {set: [n, S]} rate of
    "set and t <= S", R {set: [n]} rate of the set, Bg [n, 4, 16] bin distribution per grade, Ball / Bge [n, 16]
    bin distribution of all grades / of Good+Easy."""
    d = rev["duration"].to_numpy(np.int64); r = rev["rating"].to_numpy(np.int64) - 1; e = rev["elapsed_days"].to_numpy(np.int64)
    lower, _, _ = time_labels(d, e, same_day=True)
    lab = lower >= 0
    m = float(pri["pseudo_count"])
    ex = lambda a: (np.cumsum(a, 0) - a)[t].astype(np.float64)   # exclusive running count at the test reviews
    n_all = ex(lab.astype(np.int64))
    ng = np.stack([ex((lab & (r == g)).astype(np.int64)) for g in range(4)], 1)            # [n, 4]
    under = np.zeros((len(t), 4, len(S_GRID)))
    for g in range(4):
        for k, s in enumerate(S_GRID):
            under[:, g, k] = ex((lab & (r == g) & (d <= 1000 * s)).astype(np.int64))
    f0 = np.asarray(pri["F0_under_S_given_grade"])
    F = (under + m * f0[None]) / (ng[:, :, None] + m)
    E = {k: (under[:, list(g)].sum(1) + m * np.asarray(pri["E0_event"][k])[None]) / (n_all[:, None] + m) for k, g in SETS.items()}
    R = {k: (ng[:, list(g)].sum(1) + m * pri["R0_set"][k]) / (n_all + m) for k, g in SETS.items()}
    Bg = np.zeros((len(t), 4, BINS))
    cnt = np.zeros((len(t), 4, BINS))
    b0 = np.asarray(pri["B0_bins_given_grade"])
    for g in range(4):
        oh = np.zeros((len(d), BINS), np.int32)
        sel = lab & (r == g)
        oh[np.flatnonzero(sel), lower[sel]] = 1
        cnt[:, g] = ex(oh)
        Bg[:, g] = (cnt[:, g] + m * b0[g][None]) / (ng[:, g:g + 1] + m)
    Ball = (cnt.sum(1) + m * np.asarray(pri["B0_bins_all"])[None]) / (n_all[:, None] + m)
    Bge = (cnt[:, [2, 3]].sum(1) + m * np.asarray(pri["B0_bins_GE"])[None]) / (ng[:, [2, 3]].sum(1)[:, None] + m)
    return F, E, R, Bg, Ball, Bge


def dist_sums(prob, lower, cens, logd, unc):
    """Censored LogLoss over the 16 bins, |ln error| of the median and of the expected ln time (uncensored rows)."""
    if len(prob) == 0:
        return [0, 0.0, 0, 0.0, 0.0]
    ll = censored_ll(prob, lower.astype(np.int64), cens)
    med = np.abs(median_log_s(prob) - logd)
    mean = np.abs(prob @ LN_VAL - logd)
    return [int(len(ll)), float(ll.sum()), int(unc.sum()), float(med[unc].sum()), float(mean[unc].sum())]


# ------------------------------------------------------------------ scoring
def score_user(pg, pt, gr, d, lower, cens, cap, F, E, R, Bg, Ball, Bge):
    """All SM120 sums of one user (pg [n, 4] grade probabilities, pt [n, 4, 16] P(t | g) of the head) and the fine
    histograms {key: (n, pos, sp)}."""
    rec = {"grade": {}, "fast": {}, "speed": {}, "speed_median": {}, "mix": {}}
    hists = {}

    def add(key, p, y, store):
        store[:] = binary_sums(p, y)
        hists[key] = hist_of(p, y)

    for sn, gs in SETS.items():
        y = np.isin(gr, gs).astype(np.float64)
        rec["grade"][sn] = {}
        for pn, p in (("model", pg[:, list(gs)].sum(1)), ("base_b", R[sn])):
            rec["grade"][sn][pn] = []
            add(f"grade/{sn}/{pn}", p, y, rec["grade"][sn][pn])
    cdfs = {s: cdf_at(pt, s) for s in S_GRID}                      # [n, 4] P(t <= s | g)
    for sn, gs in SETS.items():
        rec["fast"][sn] = {"model": {}, "base_a": {}, "base_b": {}}
        for k, s in enumerate(S_GRID):
            valid = np.full(len(gr), cap > 1000 * s)
            y = (np.isin(gr, gs) & (d <= 1000 * s)).astype(np.float64)[valid]
            preds = {"model": (pg[:, list(gs)] * cdfs[s][:, list(gs)]).sum(1), "base_a": (pg[:, list(gs)] * F[:, list(gs), k]).sum(1),
                     "base_b": E[sn][:, k]}
            for pn, p in preds.items():
                rec["fast"][sn][pn][str(s)] = []
                add(f"fast/{sn}/{pn}/{s}", p[valid], y, rec["fast"][sn][pn][str(s)])
    unc = (d < CAP_MS) & (cens == 0)
    logd = np.log(np.maximum(d, 1) / 1000.0)
    for gn, g in SPEED_GRADES.items():
        m = gr == g
        rec["speed"][gn] = {"model": {}, "base": {}}
        for k, s in enumerate(S_GRID):
            valid = m & (cap > 1000 * s)
            y = (d[valid] <= 1000 * s).astype(np.float64)
            for pn, p in (("model", cdfs[s][valid, g]), ("base", F[valid, g, k])):
                rec["speed"][gn][pn][str(s)] = []
                add(f"speed/{gn}/{pn}/{s}", p, y, rec["speed"][gn][pn][str(s)])
        rec["speed_median"][gn] = {pn: dist_sums(p[m], lower[m], cens[m], logd[m], unc[m])
                                   for pn, p in (("model", pt[:, g]), ("base_running_bins", Bg[:, g]))}
    p_ge = pg[:, [2, 3]].sum(1, keepdims=True)
    mix_all = {"model": (pg[:, :, None] * pt).sum(1), "base_a": (pg[:, :, None] * Bg).sum(1), "base_b": Ball}
    mix_ge = {"model": (pg[:, [2, 3], None] * pt[:, [2, 3]]).sum(1) / p_ge,
              "base_a": (pg[:, [2, 3], None] * Bg[:, [2, 3]]).sum(1) / p_ge, "base_b": Bge}
    ge = np.isin(gr, (2, 3))
    allr = np.ones(len(gr), bool)
    for name, mix, subsets in (("all", mix_all, (("all_rows", allr), ("ge_rows", ge))), ("ge", mix_ge, (("ge_rows", ge),))):
        for sub, msk in subsets:
            rec["mix"][f"{name}@{sub}"] = {pn: dist_sums(p[msk], lower[msk], cens[msk], logd[msk], unc[msk]) for pn, p in mix.items()}
    return rec, hists


def score(args):
    """Every SM117 time-labelled test review of the users (fp32, full history, show-time inputs, leak-free): the
    SM120 sums per user (one JSON record each) and fine histograms in <out>.hist/<user>.npz. The forward pass and the
    head inputs are SM117's (scripts/sm117_head.py score), only the read-outs differ."""
    from scripts import small_model_heldout as smh  # noqa: F401  first
    import torch
    from scripts import compress_common as cc
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("HELDOUT_GPU_MEMORY_FRACTION", "0.8")))
    else:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    info = verified_master(args.model)
    hd = Path(args.head_dir)
    trunk_path = hd / "master.pth"
    trunk = torch.load(trunk_path, map_location="cpu", weights_only=True)
    released = torch.load(info["master"], map_location="cpu", weights_only=True)
    if set(trunk) != set(released) or not all(torch.equal(trunk[k].float(), released[k].float()) for k in released):
        raise SystemExit(f"{trunk_path} is not the released model of {args.model}")
    frozen = Frozen(build_model(args.model, trunk, device), MODELS[args.model]["kind"])
    head = load_head(hd / "head.pth", frozen.inputs).to(device)
    pri = json.loads(Path(args.priors).read_text())
    if tuple(pri["S"]) != S_GRID:
        raise SystemExit("priors S grid differs")
    users = json.loads(Path(args.users_file).read_text())["users"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    hist_dir = out.with_suffix(".hist")
    hist_dir.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["user_id"] for line in out.read_text().splitlines() if '"error"' not in line} if out.exists() else set()
    base = {json.loads(line)["user_id"]: json.loads(line) for line in (SIN / "time-baselines.jsonl").read_text().splitlines()}
    todo = [u for u in users if u not in done]
    data = cc.data_path()
    loader = torch.utils.data.DataLoader(ScoreBatches(todo, data, cc.run_info()["review_counts"]), batch_size=1,
                                         num_workers=args.workers, collate_fn=first, prefetch_factor=1 if args.workers else None)
    env = label_db()
    started = time.perf_counter()
    print(f"[sm120-score] {MODELS[args.model]['name']}: {len(users)} users, {len(todo)} to score", flush=True)
    f = out.open("a", encoding="utf-8")
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
                    nonfinite = int((~torch.isfinite(x)).any(-1).sum())
                del gb, o
            except torch.OutOfMemoryError as oom:
                frozen.clear()
                torch.cuda.empty_cache()
                f.write(json.dumps({"user_id": user, "error": f"OOM: {oom}"[:300]}) + "\n"); f.flush()
                continue
            except RuntimeError as err:   # as SM117: recorded, the user is left out (pass 2 retries)
                frozen.clear()
                if "out of memory" in str(err):
                    torch.cuda.empty_cache()
                print(f"[sm120-score] user {user}: {err}", flush=True)
                f.write(json.dumps({"user_id": user, "error": f"{err}"[:300]}) + "\n"); f.flush()
                continue
            t = ths - 1
            d = frame["duration"].to_numpy(np.int64)[t]
            gr = frame["rating"].to_numpy(np.int64)[t] - 1
            F, E, R, Bg, Ball, Bge = running(rev, t, pri)
            sums, hists = score_user(pg, pt, gr, d, lower[t], cens[t], cap, F, E, R, Bg, Ball, Bge)
            sha = hashlib.sha256(np.asarray(ths, np.int64).tobytes()).hexdigest()
            rec = {"user_id": user, "reviews": int(len(ths)), "review_th_sha256": sha,
                   "baseline_same_reviews": base.get(user, {}).get("review_th_sha256") == sha, "model": args.model,
                   "trunk": str(trunk_path), "head_dir": str(hd), "precision": "fp32", "inputs": INPUTS,
                   "deleted_cards": "neighbour", "user_cap_ms": int(cap), "prepare_seconds": round(prep, 2),
                   "nonfinite_inputs": nonfinite, "grade_counts": np.bincount(gr, minlength=4).tolist(),
                   "p_grade_sum_maxdev": float(np.abs(pg.sum(1) - 1).max()), "p_time_sum_maxdev": float(np.abs(pt.sum(2) - 1).max()),
                   **sums}
            keys = sorted(hists)
            np.savez_compressed(hist_dir / f"{user}.npz", keys=np.array(keys),
                                n=np.array([hists[q][0] for q in keys]), pos=np.array([hists[q][1] for q in keys]),
                                sp=np.array([hists[q][2] for q in keys]))
            f.write(json.dumps(rec) + "\n"); f.flush()
            if (k + 1) % 50 == 0:
                print(f"[sm120-score] {k + 1}/{len(todo)} users, {time.perf_counter() - started:.0f} s", flush=True)
    finally:
        f.close()
        env.close()
    print(f"[sm120-score] done {len(todo)} users in {time.perf_counter() - started:.0f} s -> {out}", flush=True)
    return 0


# ------------------------------------------------------------------ CPU check
def check(args):
    """3 small users of the 732 pool with the Tiny two-head model (SM106) on CPU: the Rec fast event reproduces SM117's
    per-user R-under-S sums (head, baselines (a) and (b)) at SM117's S; the running baselines reproduce
    sm117_head.running_baselines; the priors reproduce SM117's priors.json; label / probability sanity."""
    from scripts import compress_common as cc
    import pandas as pd  # noqa: F401
    key = args.model
    out = Path(args.out)
    work = OUT / "work"
    work.mkdir(parents=True, exist_ok=True)
    pri = json.loads(Path(args.priors).read_text())
    old_pri = json.loads((SIN / "priors.json").read_text())
    checks = {}
    # C1 priors on SM117's S equal SM117's
    ks = [S_GRID.index(s) for s in S117]
    d1 = max(abs(pri["F0_under_S_given_grade"][g][k] - old_pri["F0_under_S_given_grade"][g][j]) for g in range(4) for j, k in enumerate(ks))
    d2 = max(abs(pri["E0_event"]["Rec"][k] - old_pri["E0_event"][j]) for j, k in enumerate(ks))
    checks["C1_priors_match_sm117"] = {"pass": d1 <= 1e-6 and d2 <= 1e-6 and pri["reviews"] == old_pri["reviews"],
                                       "max_abs_F0": d1, "max_abs_E0": d2, "reviews": [pri["reviews"], old_pri["reviews"]]}
    b0 = np.asarray(pri["B0_bins_given_grade"])
    checks["C2_prior_sums"] = {"pass": bool(np.allclose(b0.sum(1), 1, atol=1e-6) and abs(sum(pri["B0_bins_all"]) - 1) < 1e-6
                                            and abs(sum(pri["B0_bins_GE"]) - 1) < 1e-6
                                            and pri["R0_set"]["E"] <= pri["R0_set"]["GE"] <= pri["R0_set"]["Rec"] <= 1),
                               "R0_set": pri["R0_set"]}
    # pick users
    old_rows = {r["user_id"]: r for r in map(json.loads, (SM117_ROWS / f"sm117-{key}-rows.jsonl").read_text().splitlines()) if "error" not in r}
    pool = set(pool_users())
    users = sorted((r["reviews"], u) for u, r in old_rows.items() if u in pool and r["reviews"] >= 300)[:3]
    users = [u for _, u in users]
    uf = work / "check-users.json"
    uf.write_text(json.dumps({"users": users}))
    rows_path = work / f"check-{key}-rows.jsonl"
    for p in (rows_path,):
        if p.exists():
            p.unlink()
    t0 = time.time()
    score(Namespace(model=key, head_dir=str(HEAD_DIRS[key]), out=str(rows_path), users_file=str(uf), device="cpu", workers=0,
                    priors=args.priors))
    secs = round(time.time() - t0, 1)
    new = {r["user_id"]: r for r in map(json.loads, rows_path.read_text().splitlines())}
    # C3 Rec reproduces SM117
    worst = {"model": 0.0, "base_a": 0.0, "base_b": 0.0}
    exact_counts = True
    per_user = {}
    for u in users:
        a, b = new[u], old_rows[u]
        per_user[u] = {"reviews": a["reviews"], "sm117_reviews": b["reviews"], "same_reviews": a["review_th_sha256"] == b["review_th_sha256"]}
        for pn in worst:
            for s in S117:
                x, y = a["fast"]["Rec"][pn][str(s)], b["rus"][pn][str(s)]
                exact_counts &= x[0] == y[0] and x[4] == y[4]
                for i in (1, 2, 3):
                    worst[pn] = max(worst[pn], abs(x[i] - y[i]) / max(abs(y[i]), 1e-9))
        per_user[u]["rec_5s_model_ll"] = [a["fast"]["Rec"]["model"]["5"][1], b["rus"]["model"]["5"][1]]
    checks["C3_rec_reproduces_sm117"] = {"pass": exact_counts and all(v <= 1e-4 for v in worst.values())
                                         and all(p["same_reviews"] for p in per_user.values()),
                                         "rows_and_event_counts_equal": exact_counts, "max_rel_diff_ll_brier_sump": worst,
                                         "tolerance": 1e-4, "users": per_user}
    # C4 running baselines equal sm117's on the SM117 grid (one user, raw arrays)
    from scripts.sm117_timelib import revlog
    data = cc.data_path()
    rev = revlog(data, users[0])
    env = label_db()
    with env.begin() as txn:
        ths, frame, lower, cens, cap = test_reviews(data, users[0], txn, same_day=True)
    env.close()
    t = ths - 1
    F, E, R, Bg, Ball, Bge = running(rev, t, pri)
    Fo, Eo = h117.running_baselines(rev, old_pri)
    dF = float(np.abs(F[:, :, ks] - Fo[t]).max()); dE = float(np.abs(E["Rec"][:, ks] - Eo[t]).max())
    checks["C4_running_baselines_match_sm117"] = {"pass": dF < 1e-5 and dE < 1e-5, "max_abs_F": dF, "max_abs_E": dE,
                                                  "note": "priors rounded to 6 decimals in both files"}
    checks["C5_running_distributions_sum_to_1"] = {"pass": bool(np.allclose(Bg.sum(2), 1) and np.allclose(Ball.sum(1), 1) and np.allclose(Bge.sum(1), 1))}
    # C6 label / probability sanity on the scored users
    san = {"pass": True}
    for u in users:
        a = new[u]
        rate = {sn: [a["fast"][sn]["model"][str(s)][4] / max(a["fast"][sn]["model"][str(s)][0], 1) for s in S_GRID] for sn in SETS}
        meanp = {sn: [a["fast"][sn]["model"][str(s)][3] / max(a["fast"][sn]["model"][str(s)][0], 1) for s in S_GRID] for sn in SETS}
        grade_rate = {sn: a["grade"][sn]["model"][4] / a["grade"][sn]["model"][0] for sn in SETS}
        grade_p = {sn: a["grade"][sn]["model"][3] / a["grade"][sn]["model"][0] for sn in SETS}
        nested = all(rate["E"][k] <= rate["GE"][k] + 1e-12 <= rate["Rec"][k] + 2e-12 for k in range(len(S_GRID))) \
            and all(meanp["E"][k] <= meanp["GE"][k] + 1e-12 <= meanp["Rec"][k] + 2e-12 for k in range(len(S_GRID)))
        mono = all(np.all(np.diff(rate[sn]) >= -1e-12) and np.all(np.diff(meanp[sn]) >= -1e-12) for sn in SETS)
        below_grade = all(meanp[sn][-1] <= grade_p[sn] + 1e-9 and rate[sn][-1] <= grade_rate[sn] + 1e-9 for sn in SETS)
        gc = a["grade_counts"]
        counts_ok = a["grade"]["GE"]["model"][4] == gc[2] + gc[3] and a["grade"]["E"]["model"][4] == gc[3] \
            and a["speed"]["Easy"]["model"]["30"][0] == (gc[3] if a["user_cap_ms"] > 30000 else 0)
        sums_ok = a["p_grade_sum_maxdev"] < 1e-5 and a["p_time_sum_maxdev"] < 1e-5   # float32 softmax
        mix_rows = a["mix"]["all@all_rows"]["model"][0] == a["reviews"] and a["mix"]["ge@ge_rows"]["model"][0] == gc[2] + gc[3]
        ok = nested and mono and below_grade and counts_ok and sums_ok and mix_rows
        san[str(u)] = {"ok": ok, "nested_E_GE_Rec": nested, "monotone_in_S": mono, "fast_below_grade_only": below_grade,
                       "label_counts": counts_ok, "probabilities_sum_to_1": sums_ok, "mixture_rows": mix_rows,
                       "user_cap_ms": a["user_cap_ms"], "grade_counts": gc,
                       "event_rate": {sn: [round(x, 4) for x in v] for sn, v in rate.items()},
                       "mean_p_model": {sn: [round(x, 4) for x in v] for sn, v in meanp.items()},
                       "grade_rate": {sn: round(v, 4) for sn, v in grade_rate.items()},
                       "grade_mean_p": {sn: round(v, 4) for sn, v in grade_p.items()},
                       "mean_p_base_a_GE": [round(a["fast"]["GE"]["base_a"][str(s)][3] / max(a["fast"]["GE"]["base_a"][str(s)][0], 1), 4) for s in S_GRID],
                       "mean_p_base_b_GE": [round(a["fast"]["GE"]["base_b"][str(s)][3] / max(a["fast"]["GE"]["base_b"][str(s)][0], 1), 4) for s in S_GRID]}
        san["pass"] &= ok
    checks["C6_labels_and_probabilities"] = san
    res = {"model": MODELS[key]["name"], "device": "cpu", "users": users, "score_seconds": secs, "S": list(S_GRID),
           "pass": all(c["pass"] for c in checks.values()), "checks": checks}
    atomic_json(out, res)
    print(f"[sm120-check] {'PASS' if res['pass'] else 'FAIL'}: " + ", ".join(f"{k} {v['pass']}" for k, v in checks.items()))
    return 0 if res["pass"] else 1


# ------------------------------------------------------------------ evaluation + report (CPU)
def _ece(H, key):
    st = ece_auc_from_hist(H[key][0], H[key][1], H[key][2])
    return st


def evaluate(key, rows_path):
    pool = pool_users()
    rows = {r["user_id"]: r for r in map(json.loads, Path(rows_path).read_text().splitlines()) if "error" not in r}
    users = [u for u in pool if u in rows and rows[u]["reviews"] > 0]
    res = {"model": key, "name": MODELS[key]["name"], "pool": len(pool), "missing": [u for u in pool if u not in rows],
           "users": len(users), "baseline_mismatch": [u for u in users if not rows[u]["baseline_same_reviews"]][:20]}
    if not users or res["baseline_mismatch"]:
        return res
    hd = Path(rows_path).with_suffix(".hist")
    H = {}
    for u in users:
        z = np.load(hd / f"{u}.npz")
        for i, q in enumerate(z["keys"]):
            q = str(q)
            if q not in H:
                H[q] = [np.zeros(HIST_BINS), np.zeros(HIST_BINS), np.zeros(HIST_BINS)]
            H[q][0] += z["n"][i]; H[q][1] += z["pos"][i]; H[q][2] += z["sp"][i]
    base = {r["user_id"]: r for r in map(json.loads, (SIN / "time-baselines.jsonl").read_text().splitlines())}

    def binary_block(get, preds, hist_key):
        arr = {p: np.array([get(rows[u], p) for u in users], float) for p in preds}
        n = arr["model"][:, 0]
        keep = n > 0
        out = {"rows": int(n.sum()), "users_with_rows": int(keep.sum()), "predictors": {}, "vs": {}}
        if n.sum() == 0:
            out["reliable"] = False
            return out
        out["event_rate"] = round(float(arr["model"][:, 4].sum() / n.sum()), 5)
        for p in preds:
            st = _ece(H, hist_key(p))
            out["predictors"][p] = {"logloss": round(float(arr[p][:, 1].sum() / n.sum()), 6),
                                    "brier": round(float(arr[p][:, 2].sum() / n.sum()), 6),
                                    "mean_p": round(float(arr[p][:, 3].sum() / n.sum()), 5),
                                    "ece": round(st["ece"], 5) if st else None, "auc": round(st["auc"], 5) if st else None,
                                    "max_bin_gap": round(st["max_gap"], 5) if st else None,
                                    "reliability": st["table"] if st else None}
        idx = bootstrap_idx(int(keep.sum()))
        for p in preds[1:]:
            out["vs"][f"model_vs_{p}"] = {m: _diff(arr["model"][keep, c], arr[p][keep, c], n[keep], idx) for m, c in (("logloss", 1), ("brier", 2))}
        out["reliable"] = bool(out["predictors"]["model"]["ece"] is not None and out["predictors"]["model"]["ece"] < 0.01
                               and all(out["vs"][f"model_vs_{p}"]["logloss"]["ci"][1] < 0 for p in preds[1:]))
        return out

    res["grade"] = {sn: binary_block(lambda r, p: r["grade"][sn][p], ["model", "base_b"], lambda p: f"grade/{sn}/{p}") for sn in SETS}
    res["fast"] = {sn: {str(s): binary_block(lambda r, p: r["fast"][sn][p][str(s)], ["model", "base_a", "base_b"],
                                             lambda p: f"fast/{sn}/{p}/{s}") for s in S_GRID} for sn in SETS}
    res["speed"] = {gn: {str(s): binary_block(lambda r, p: r["speed"][gn][p][str(s)], ["model", "base"],
                                              lambda p: f"speed/{gn}/{p}/{s}") for s in S_GRID} for gn in SPEED_GRADES}
    idx = bootstrap_idx(len(users))

    def dist_block(get, preds):
        arr = {p: np.array([get(rows[u], p) for u in users], float) for p in preds}   # [n, ll, n_unc, ae_med, ae_mean]
        n, nu = arr[preds[0]][:, 0], arr[preds[0]][:, 2]
        out = {"rows": int(n.sum()), "uncensored": int(nu.sum()),
               "time_logloss": {p: round(float(arr[p][:, 1].sum() / max(n.sum(), 1)), 5) for p in preds if arr[p][:, 1].any() or p == preds[0]},
               "mae_ln_median": {p: round(float(arr[p][:, 3].sum() / max(nu.sum(), 1)), 5) for p in preds},
               "mae_ln_mean": {p: round(float(arr[p][:, 4].sum() / max(nu.sum(), 1)), 5) for p in preds}, "vs": {}}
        for p in preds[1:]:
            v = {"mae_ln_median": _diff(arr[preds[0]][:, 3], arr[p][:, 3], nu, idx)}
            if arr[p][:, 1].any():
                v["time_logloss"] = _diff(arr[preds[0]][:, 1], arr[p][:, 1], n, idx)
                v["mae_ln_mean"] = _diff(arr[preds[0]][:, 4], arr[p][:, 4], nu, idx)
            out["vs"][f"{preds[0]}_vs_{p}"] = v
        out["beats_all"] = all(x["delta"] < 0 and x["ci"][1] < 0 for v in out["vs"].values() for x in v.values())
        return out

    sm = {}
    for gn, g in SPEED_GRADES.items():   # SM117's user-median baseline: the user's exact running median of the grade
        get = lambda r, p, gn=gn, g=g: (r["speed_median"][gn][p] if p != "user_median"
                                        else [base[r["user_id"]]["user_median"][f"g{g + 1}"][0], 0.0, base[r["user_id"]]["user_median"][f"g{g + 1}"][2],
                                              base[r["user_id"]]["user_median"][f"g{g + 1}"][3], 0.0])
        sm[gn] = dist_block(get, ["model", "base_running_bins", "user_median"])
    res["speed_median"] = sm
    res["mix"] = {name: dist_block(lambda r, p, name=name: r["mix"][name][p], ["model", "base_a", "base_b"])
                  for name in ("all@all_rows", "all@ge_rows", "ge@ge_rows")}
    res["reliable_S"] = {**{f"fast/{sn}": [s for s in S_GRID if res["fast"][sn][str(s)]["reliable"]] for sn in SETS},
                         **{f"speed/{gn}": [s for s in S_GRID if res["speed"][gn][str(s)]["reliable"]] for gn in SPEED_GRADES},
                         **{f"grade/{sn}": res["grade"][sn]["reliable"] for sn in SETS}}
    return res


def fmt_ci(d, pct=True):
    s = f"{d['delta']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]"
    return s + (f" ({d['relative_pct']:+.1f}%)" if pct else "")


def s_ranges(lst):
    if not lst:
        return "none"
    return ", ".join(f"{s} s" for s in lst)


def report(args):
    G = OUT / "gpu"
    res = {}
    for key in ORDER:
        p = G / f"sm120-{key}-rows.jsonl"
        if p.exists():
            res[key] = evaluate(key, p)
    L = ["# Fast Good/Easy prediction (SM120): result", "",
         "**What answer time means.** The answer time is the whole time on the card, front and back together: from the moment the "
         "question is shown to the moment a grade button is pressed, capped at the preset's maximum answer seconds. The \"Show answer\" "
         "moment is not stored, so front-only and back-only times cannot be computed. Every \"within S s\" below means answered within "
         "S s, reading the back included.", "",
         "Scoring only (no training): the answer-time heads of the time heads for the family (SM117) and of the time heads for the new "
         "candidates (SM119) on their frozen models, the 732 set (723 users, show-time inputs, leak-free, the SM117 time-labelled test "
         "reviews). Pre-registered in `design.md`. Reliable = ECE (20 equal-mass bins) below 0.01 and LogLoss below every baseline with "
         "the 95% CI (2,000 user bootstraps, seed 0, review-weighted) below zero. Deltas are model minus baseline. S counts only "
         "below the user's own maximum answer seconds.", "",
         "## Where each event is reliable (values of S)", "",
         "| model | P(Good or Easy) | P(Easy) | recalled within S s | Good or Easy within S s | Easy within S s | speed of a Good answer | speed of an Easy answer |",
         "|---|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        if "fast" not in r:
            L.append(f"| {r['name']} | not scored ({r['users']} users, baseline mismatch {r['baseline_mismatch'][:3]}) |||||||")
            continue
        q = r["reliable_S"]
        L.append(f"| {r['name']} | {'yes' if q['grade/GE'] else 'no'} | {'yes' if q['grade/E'] else 'no'} | {s_ranges(q['fast/Rec'])} | "
                 f"{s_ranges(q['fast/GE'])} | {s_ranges(q['fast/E'])} | {s_ranges(q['speed/Good'])} | {s_ranges(q['speed/Easy'])} |")
    L += ["", "## Good or Easy within 4 s (answered within 4 s, reading the back included)", "",
          "| model | rows | event rate | mean p | LogLoss model / (a) / (b) | model vs (a) | model vs (b) | ECE | AUC model / (a) / (b) | reliable |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        if "fast" not in r:
            continue
        x = r["fast"]["GE"]["4"]; p = x["predictors"]
        L.append(f"| {r['name']} | {x['rows']:,} | {x['event_rate']:.3f} | {p['model']['mean_p']:.3f} | {p['model']['logloss']:.4f} / "
                 f"{p['base_a']['logloss']:.4f} / {p['base_b']['logloss']:.4f} | {fmt_ci(x['vs']['model_vs_base_a']['logloss'])} | "
                 f"{fmt_ci(x['vs']['model_vs_base_b']['logloss'])} | {p['model']['ece']:.4f} | {p['model']['auc']:.3f} / {p['base_a']['auc']:.3f} / "
                 f"{p['base_b']['auc']:.3f} | {'yes' if x['reliable'] else 'no'} |")
    L += ["", "## 1. Grade-only events (no time): P(Good or Easy), P(Easy), P(recalled)", "",
          "Baseline (b): the user's running rate of the event (earlier time-labelled reviews, pseudo-count 10 at the pooled rate).", "",
          "| model | event | rows | event rate | LogLoss model / (b) | model vs (b) | ECE model / (b) | AUC model / (b) | reliable |", "|---|---|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        if "grade" not in r:
            continue
        for sn in SETS:
            x = r["grade"][sn]; p = x["predictors"]
            L.append(f"| {r['name']} | {SET_NAMES[sn]} | {x['rows']:,} | {x['event_rate']:.3f} | {p['model']['logloss']:.4f} / {p['base_b']['logloss']:.4f} | "
                     f"{fmt_ci(x['vs']['model_vs_base_b']['logloss'])} | {p['model']['ece']:.4f} / {p['base_b']['ece']:.4f} | "
                     f"{p['model']['auc']:.3f} / {p['base_b']['auc']:.3f} | {'yes' if x['reliable'] else 'no'} |")
    L += ["", "## 2. Fast events: P(set and answered within S s)", "",
          "p = sum over the grades of the set of P(grade) x P(t <= S | grade). Baseline (a): the model's own P(grade) x the user's running "
          "share of earlier answers of that grade within S s, summed over the set; baseline (b): the user's running rate of the event.", ""]
    for sn in SETS:
        L += [f"### {SET_NAMES[sn]} within S s", ""]
        for k, r in res.items():
            if "fast" not in r:
                continue
            L += [f"{r['name']}:", "", "| S | rows | event rate | mean p | LogLoss model | (a) | (b) | model vs (a) | model vs (b) | ECE model | ECE (a) | AUC model | AUC (a) | AUC (b) | reliable |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for s in S_GRID:
                x = r["fast"][sn][str(s)]
                if not x["rows"]:
                    continue
                p = x["predictors"]
                L.append(f"| {s} s | {x['rows']:,} | {x['event_rate']:.3f} | {p['model']['mean_p']:.3f} | {p['model']['logloss']:.4f} | "
                         f"{p['base_a']['logloss']:.4f} | {p['base_b']['logloss']:.4f} | {fmt_ci(x['vs']['model_vs_base_a']['logloss'])} | "
                         f"{fmt_ci(x['vs']['model_vs_base_b']['logloss'])} | {p['model']['ece']:.4f} | {p['base_a']['ece']:.4f} | "
                         f"{p['model']['auc']:.3f} | {p['base_a']['auc']:.3f} | {p['base_b']['auc']:.3f} | {'yes' if x['reliable'] else 'no'} |")
            L.append("")
    L += ["## 3. Speed given the grade: P(answered within S s | the answer was Hard / Good / Easy)", "",
          "Rows: reviews whose actual grade is that grade. Baseline: the user's running share of earlier answers of that grade within S s.", ""]
    for gn in SPEED_GRADES:
        L += [f"### {gn} answers", "", "| model | " + " | ".join(f"{s} s" for s in S_GRID) + " |", "|---|" + "---|" * len(S_GRID)]
        for k, r in res.items():
            if "speed" not in r:
                continue
            cells = []
            for s in S_GRID:
                x = r["speed"][gn][str(s)]
                if not x["rows"]:
                    cells.append("")
                    continue
                cells.append(f"{x['vs']['model_vs_base']['logloss']['relative_pct']:+.1f}%, ECE {x['predictors']['model']['ece']:.3f}"
                             f"{' (reliable)' if x['reliable'] else ''}")
            L.append(f"| {r['name']} | " + " | ".join(cells) + " |")
        L += ["", "Cells: LogLoss of the head vs the baseline (relative), the head's ECE; full numbers with CIs in `sm120-result.json` (`speed`).", ""]
    L += ["### Predicted median time of the grade vs the actual time (mean abs ln error, uncensored rows)", "",
          "| model | grade | rows | head | user's running median (SM117 baseline) | user's running bins | head vs running median | censored LogLoss head / running bins |",
          "|---|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        if "speed_median" not in r:
            continue
        for gn in SPEED_GRADES:
            x = r["speed_median"][gn]
            L.append(f"| {r['name']} | {gn} | {x['uncensored']:,} | {x['mae_ln_median']['model']:.4f} | {x['mae_ln_median']['user_median']:.4f} | "
                     f"{x['mae_ln_median']['base_running_bins']:.4f} | {fmt_ci(x['vs']['model_vs_user_median']['mae_ln_median'])} | "
                     f"{x['time_logloss']['model']:.4f} / {x['time_logloss']['base_running_bins']:.4f} |")
    L += ["", "## 4. Grade-weighted time: the mixture over the grade probabilities", "",
          "All grades: P(t) = sum_g P(g) P(t | g); success only: P(t | Good or Easy) = sum over Good, Easy of P(g) P(t | g) / P(Good or Easy). "
          "Baseline (a): the same mixture of the user's running per-grade bin distributions with the model's P(g); baseline (b): the user's "
          "running bin distribution (all grades, or Good and Easy only). Median and expected ln time from the 16 bins.", "",
          "| model | mixture @ rows | rows | censored LogLoss model / (a) / (b) | model vs (b) | abs ln error of the median model / (a) / (b) | model vs (b) | abs ln error of E[ln t] model / (b) | beats both |",
          "|---|---|---|---|---|---|---|---|---|"]
    lab = {"all@all_rows": "all grades @ all rows", "all@ge_rows": "all grades @ Good/Easy rows", "ge@ge_rows": "Good or Easy @ Good/Easy rows"}
    for k, r in res.items():
        if "mix" not in r:
            continue
        for name, x in r["mix"].items():
            tl, me, mm = x["time_logloss"], x["mae_ln_median"], x["mae_ln_mean"]
            L.append(f"| {r['name']} | {lab[name]} | {x['rows']:,} | {tl['model']:.4f} / {tl['base_a']:.4f} / {tl['base_b']:.4f} | "
                     f"{fmt_ci(x['vs']['model_vs_base_b']['time_logloss'])} | {me['model']:.4f} / {me['base_a']:.4f} / {me['base_b']:.4f} | "
                     f"{fmt_ci(x['vs']['model_vs_base_b']['mae_ln_median'])} | {mm['model']:.4f} / {mm['base_b']:.4f} | {'yes' if x['beats_all'] else 'no'} |")
    L += ["", "Reliability tables (20 equal-mass bins: rows, mean predicted, observed) per event and S: `sm120-result.json`."]
    (OUT / "sm120-result.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    atomic_json(OUT / "sm120-result.json", res)
    print(f"[sm120-report] {len(res)} models: Good or Easy within 4 s reliable "
          f"{[(k, r['fast']['GE']['4']['reliable']) for k, r in res.items() if 'fast' in r]} -> {OUT / 'sm120-result.md'}")
    return 0


def main():
    p0 = ArgumentParser(description=__doc__.split("\n")[0])
    sub = p0.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("priors")
    p.add_argument("--out", default=str(OUT / "priors.json"))
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("score")
    p.add_argument("--model", required=True, choices=ORDER)
    p.add_argument("--head-dir", default="")
    p.add_argument("--priors", default=str(OUT / "priors.json"))
    p.add_argument("--users-file", default=str(U732))
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=3)
    p = sub.add_parser("check")
    p.add_argument("--model", default="sm106", choices=ORDER)
    p.add_argument("--priors", default=str(OUT / "priors.json"))
    p.add_argument("--out", default=str(OUT / "check-cpu.json"))
    sub.add_parser("report")
    a = p0.parse_args()
    if a.cmd == "score" and not a.head_dir:
        a.head_dir = str(HEAD_DIRS[a.model])
    return {"priors": priors, "score": score, "check": check, "report": report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
