#!/usr/bin/env python
"""Fast-recall prediction (SM94), stage 1: is "pass AND answer under N seconds" predictable from the released small
model's existing heads, and does the answer-speed signal add anything to plain recall?

  dump     GPU: the answer-time head on richer frozen features (SM82c) scorer (`scripts/sm82c_head.py score` of the
           sm82c worktree, imported read only, unchanged), run on the released small model + its shipped answer-time
           head (SM83), but writing per-review arrays instead of per-user sums: one npz per user with the instant
           head's grade probabilities pg [n, 4] (f32), the answer-time head's per-grade 16-bin distributions pt
           [n, 4, 16] (f16), the 26 explicit causal time features ex [n, 26] (f32), and review_th / grade / duration /
           lower bin / censored / user cap. Same reviews as SM82c's time scoring (time-scored later-day test reviews).
           A per-user jsonl row keeps the identity sums (censored time LogLoss of the true grade) for the check
           against SM82c's recorded rows.
  analyze  CPU: the fixed stage 1 plan of artifacts/small-model/sm94/DESIGN.md (events, predictors, cross-fitted
           stacks, paired user bootstrap 2,000 seed 0) -> stage1-result.{json,md}.

The sm82c worktree is put first on sys.path so `scripts.*`, `expertium.*` and `rwkv.*` resolve to its committed code;
this file only adds the output format and the analysis.
"""
from argparse import ArgumentParser
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

SM82C_ROOT = Path("C:/Users/jonat/workspace/wt/sm82c")        # read only: the SM82c scorer and its model code
HERE = Path(__file__).resolve().parents[1] / "artifacts/small-model/sm94"
MASTER = SM82C_ROOT / "artifacts/small-model/sm82c-head-released/master.pth"   # release (SM63) + shipped head (SM83)
ABLATION = SM82C_ROOT / "artifacts/small-model/sm82c-head-released/ablation-head.pth"
S72 = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s")
USERS732 = S72 / "eval-users-732-fit.json"
SM82C_TIME = SM82C_ROOT / "artifacts/small-model/sm82c/gpu/sm82c-time.jsonl"   # recorded identity rows (model head)
RELEASE_ROWS = S72 / "gpu/small-lf-neighbour.jsonl"
RELEASE_LABEL = "s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/w8+s8"
BIG_ROWS = Path("C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/gpu-big/bigleak-neighbour.jsonl")
BINS = 16
EDGES_MS = [500.0 * 120.0 ** (k / 14) for k in range(15)]   # = scripts/expertium_options.TIME_BIN_EDGES_MS
FIXED_K = [6, 7, 9, 11]                                     # under 3.9 / 5.5 / 10.9 / 21.5 s (bin edges, no interpolation)
EPS = 1e-6


def _code_path():
    sys.path.insert(0, str(SM82C_ROOT))


# ------------------------------------------------------------------ dump (GPU)
def dump(args):
    _code_path()
    from scripts import small_model_heldout as smh  # noqa: F401  first: sets RWKV_NO_JIT
    import hashlib
    import torch
    from scripts import compress_common as cc
    from scripts.sm82_time import label_db, test_reviews, user_sums, CAP_MS
    from scripts.sm82c_head import (ScoreBatches, load_heads, Frozen, build_frozen, FROZEN_DIM, BINS as HB)
    from scripts.sm82b_head import first
    assert HB == BINS
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("HELDOUT_GPU_MEMORY_FRACTION", "0.8")))
    else:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    sd, _, main, _abl = load_heads(args.master, args.ablation, device)
    frozen = Frozen(build_frozen({k: v.float() for k, v in sd.items()}, device))
    users = json.loads(Path(args.users_file).read_text())["users"]
    if args.max_users:
        users = users[: args.max_users]
    out_dir = Path(args.out_dir)
    (out_dir / "users").mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "dump-rows.jsonl"
    done = set()
    if rows_path.exists():
        done = {json.loads(x)["user_id"] for x in rows_path.read_text().splitlines() if x and "error" not in x}
    todo = [u for u in users if u not in done]
    data = cc.data_path()
    loader = torch.utils.data.DataLoader(ScoreBatches(todo, data, cc.run_info()["review_counts"]), batch_size=1,
                                         num_workers=args.workers, collate_fn=first,
                                         prefetch_factor=1 if args.workers else None)
    env = label_db()
    started = time.perf_counter()
    scored = 0
    print(f"[sm94-dump] {len(users)} users, {len(todo)} to dump, {args.master}", flush=True)
    with rows_path.open("a", encoding="utf-8") as f:
        try:
            for k, (user, batch, error, prep, table) in enumerate(loader):
                if batch is None:
                    f.write(json.dumps({"user_id": user, "error": error}) + "\n"); f.flush()
                    continue
                with env.begin() as txn:
                    ths, frame, lower, cens, cap = test_reviews(data, user, txn)
                t0 = time.perf_counter()
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
                        ex = torch.as_tensor(table, device=device)[want - 1]
                        x = torch.cat([feats[pos], ex], -1)
                        pg = torch.softmax(p_logits[qi[pos]], -1)
                        pt = torch.softmax(main(x).float().view(-1, 4, BINS), -1)
                        pg_np = pg.cpu().numpy().astype(np.float32)
                        pt64 = pt.double().cpu().numpy()
                        ex_np = ex.float().cpu().numpy()
                    del gb, o
                except torch.OutOfMemoryError as oom:
                    frozen.hp.clear(); frozen.capture.snaps = []
                    torch.cuda.empty_cache()
                    f.write(json.dumps({"user_id": user, "error": f"OOM: {oom}"[:300]}) + "\n"); f.flush()
                    continue
                d = frame["duration"].to_numpy(np.int64)
                grade = frame["rating"].to_numpy()[ths - 1].astype(np.int8) - 1
                t = {"grade": grade, "lower": lower[ths - 1].astype(np.int8), "cens": cens[ths - 1].astype(np.int8),
                     "logd": np.log(np.maximum(d[ths - 1], 1) / 1000.0),
                     "uncensored": (d[ths - 1] < CAP_MS) & (cens[ths - 1] == 0)}
                cond = pt64[np.arange(len(ths)), grade.astype(np.int64)]
                np.savez(out_dir / "users" / f"{user}.npz", th=np.asarray(ths, np.int32), grade=grade,
                         dur=d[ths - 1].astype(np.int32), lower=t["lower"], cens=t["cens"], cap_ms=np.int32(cap),
                         elapsed=frame["elapsed_days"].to_numpy()[ths - 1].astype(np.int32),
                         pg=pg_np, pt=pt64.astype(np.float16), ex=ex_np.astype(np.float32))
                sha = hashlib.sha256(np.asarray(ths, np.int64).tobytes()).hexdigest()
                rec = {"user_id": user, "reviews": int(len(ths)), "review_th_sha256": sha, "checkpoint": str(args.master),
                       "precision": "fp32", "deleted_cards": "neighbour", "prepare_seconds": round(prep, 2),
                       "model_seconds": round(time.perf_counter() - t0, 3), "full_history_reviews": int(len(frame)),
                       "model": user_sums(cond, t)}
                f.write(json.dumps(rec) + "\n"); f.flush()
                scored += len(ths)
                if (k + 1) % 50 == 0:
                    s = time.perf_counter() - started
                    print(f"[sm94-dump] {k + 1}/{len(todo)} users, {s:.0f} s, {scored / s:,.0f} scored reviews/s", flush=True)
        finally:
            env.close()
    s = time.perf_counter() - started
    print(f"[sm94-dump] done {len(todo)} users in {s:.0f} s, {scored:,} scored reviews ({scored / max(s, 1e-9):,.0f}/s)"
          f" -> {out_dir}", flush=True)
    return 0


# ------------------------------------------------------------------ analysis (CPU)
def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def bll(p, y):
    """Per-row binary log loss, p clipped to [1e-6, 1 - 1e-6]."""
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def running_rate(hit, base, prior_rate, prior_n):
    """Causal running share of `hit` among rows with `base`, before each row: (hits so far + a r0) / (bases so far + a)."""
    h = np.concatenate([[0.0], np.cumsum(hit & base)[:-1]])
    b = np.concatenate([[0.0], np.cumsum(base)[:-1]])
    return (h + prior_n * prior_rate) / (b + prior_n)


def load_user(path):
    z = np.load(path)
    out = {k: z[k] for k in z.files}
    out["pt"] = out["pt"].astype(np.float32)
    return out


def user_k(u):
    """The user-relative threshold: the bin edge nearest in ln time to the user's median uncensored pass answer time."""
    m = (u["grade"] >= 1) & (u["cens"] == 0) & (u["dur"] > 0)
    if m.sum() < 20:
        return None
    med = float(np.median(u["dur"][m]))
    k = int(np.argmin(np.abs(np.log(np.asarray(EDGES_MS[1:14])) - math.log(med)))) + 1
    return k


def events(u, ks):
    """{name: (k, y, C, Ppass)} for each usable threshold (user cap >= the edge)."""
    out = {}
    passed = u["grade"] >= 1
    pp = u["pg"][:, 1:].sum(1)
    cdf = np.cumsum(u["pt"], axis=2)                       # [n, 4, 16]
    for name, k in ks:
        if k is None or u["cap_ms"] < EDGES_MS[k]:
            continue
        y = passed & (u["lower"] <= k) & (u["cens"] == 0)
        c = (u["pg"][:, 1:] * cdf[:, 1:, k]).sum(1)
        out[name] = (k, y.astype(np.float64), c.astype(np.float64), pp.astype(np.float64))
    return out


def stack_x(kind, c, pp, ex):
    """Design matrices of the cross-fitted stacks (intercept added by the fitter)."""
    if kind == "S0":
        return logit(c)[:, None]
    if kind == "S":
        cf = np.clip(c / np.maximum(pp, EPS), EPS, 1 - EPS)
        return np.column_stack([logit(c), logit(pp), logit(cf), ex])
    if kind == "Rb0":
        return logit(pp)[:, None]
    if kind == "Rb":
        lp = logit(pp)
        return np.column_stack([lp, ex, ex[:, :3] * lp[:, None]])
    raise ValueError(kind)


def fit_logistic(x, y, l2=1e-3, iters=30):
    """Newton / IRLS on standardised features; returns (mean, std, w) with w[0] the intercept."""
    mu, sd = x.mean(0), x.std(0)
    sd[sd < 1e-8] = 1.0
    z = np.column_stack([np.ones(len(x)), (x - mu) / sd])
    w = np.zeros(z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(z @ w, -30, 30)))
        g = z.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]]
        h = (z * (p * (1 - p))[:, None]).T @ z / len(y) + l2 * np.eye(len(w))
        step = np.linalg.solve(h, g)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return mu, sd, w


def apply_logistic(model, x):
    mu, sd, w = model
    z = np.column_stack([np.ones(len(x)), (x - mu) / sd])
    return 1 / (1 + np.exp(-np.clip(z @ w, -30, 30)))


def boot(per_user_a, per_user_b, n, reps=2000, seed=0):
    """Paired user bootstrap of the review-weighted mean difference (a - b); per-user sums and row counts."""
    rng = np.random.default_rng(seed)
    d = per_user_a - per_user_b
    m = len(d)
    idx = rng.integers(0, m, size=(reps, m))
    stats = d[idx].sum(1) / n[idx].sum(1)
    return float(d.sum() / n.sum()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def analyze(args):
    t_start = time.perf_counter()
    dump_dir = Path(args.dump_dir)
    rows = {}
    for line in (dump_dir / "dump-rows.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "error" not in r:
            rows[r["user_id"]] = r
    users = sorted(u for u in rows if (dump_dir / "users" / f"{u}.npz").exists())
    if args.max_users:
        users = users[: args.max_users]
    # identity check vs the recorded SM82c rows (same master, same reviews)
    ident = {"users": 0, "same_reviews": 0, "max_rel_gap": 0.0}
    if SM82C_TIME.exists():
        rec = {json.loads(x)["user_id"]: json.loads(x) for x in SM82C_TIME.read_text().splitlines() if "error" not in x}
        for u in users:
            if u in rec:
                ident["users"] += 1
                ident["same_reviews"] += int(rec[u]["review_th_sha256"] == rows[u]["review_th_sha256"])
                a, b = rec[u]["model"]["ll"], rows[u]["model"]["ll"]
                ident["max_rel_gap"] = max(ident["max_rel_gap"], abs(a - b) / max(abs(a), 1e-12))
    rng = np.random.default_rng(0)
    half = {u: i % 2 for i, u in enumerate(rng.permutation(users))}
    names = [f"under {EDGES_MS[k] / 1000:.1f} s" for k in FIXED_K] + ["under the user's typical time"]
    # pass 1: pooled rates and the fit samples
    pooled = {nm: [0.0, 0.0, 0.0] for nm in names}      # fast-pass hits, pass rows, rows
    sample = {(nm, h): [] for nm in names for h in (0, 1)}
    sample_b = {h: [] for h in (0, 1)}
    kus = {}
    for u in users:
        d = load_user(dump_dir / "users" / f"{u}.npz")
        ku = user_k(d)
        kus[u] = ku
        ev = events(d, [(nm, k) for nm, k in zip(names, FIXED_K + [ku])])
        n = len(d["grade"])
        take = rng.choice(n, size=min(n, args.fit_rows_per_user), replace=False)
        for nm, (k, y, c, pp) in ev.items():
            pooled[nm][0] += y.sum(); pooled[nm][1] += (d["grade"] >= 1).sum(); pooled[nm][2] += n
            sample[(nm, half[u])].append((y[take], c[take], pp[take], d["ex"][take]))
        pp = d["pg"][:, 1:].sum(1).astype(np.float64)
        sample_b[half[u]].append(((d["grade"] >= 1)[take].astype(np.float64), pp[take], d["ex"][take]))
    models = {}
    for nm in names:
        for h in (0, 1):
            parts = sample[(nm, h)]
            if not parts:
                continue
            y = np.concatenate([p[0] for p in parts]); c = np.concatenate([p[1] for p in parts])
            pp = np.concatenate([p[2] for p in parts]); ex = np.concatenate([p[3] for p in parts]).astype(np.float64)
            for kind in ("S0", "S"):
                models[(nm, kind, h)] = fit_logistic(stack_x(kind, c, pp, ex), y)
    for h in (0, 1):
        parts = sample_b[h]
        y = np.concatenate([p[0] for p in parts]); pp = np.concatenate([p[1] for p in parts])
        ex = np.concatenate([p[2] for p in parts]).astype(np.float64)
        for kind in ("Rb0", "Rb"):
            models[("pass", kind, h)] = fit_logistic(stack_x(kind, None, pp, ex), y)
    del sample, sample_b
    # pass 2: per-user sums of every predictor (stacks applied with the model fitted on the OTHER half)
    preds_a = ["C", "B0", "B1", "S0", "S"]
    per = {nm: {p: [] for p in preds_a + ["n", "user"]} for nm in names}
    calib = {nm: np.zeros((20, 3)) for nm in names}      # C: sum pred, sum y, rows per equal-width bin
    preds_b = ["R", "Rb0", "Rb"]
    perb = {p: [] for p in preds_b + ["n", "user"]}
    for u in users:
        d = load_user(dump_dir / "users" / f"{u}.npz")
        other = 1 - half[u]
        ev = events(d, [(nm, k) for nm, k in zip(names, FIXED_K + [kus[u]])])
        passed = (d["grade"] >= 1)
        ex = d["ex"].astype(np.float64)
        for nm, (k, y, c, pp) in ev.items():
            r0 = pooled[nm][0] / max(pooled[nm][2], 1)
            rf = pooled[nm][0] / max(pooled[nm][1], 1)
            fast = y > 0
            b0 = running_rate(fast, np.ones_like(fast), r0, args.prior_n)
            b1 = pp * running_rate(fast, passed, rf, args.prior_n)
            s0 = apply_logistic(models[(nm, "S0", other)], stack_x("S0", c, pp, ex))
            s = apply_logistic(models[(nm, "S", other)], stack_x("S", c, pp, ex))
            for name, p in (("C", c), ("B0", b0), ("B1", b1), ("S0", s0), ("S", s)):
                per[nm][name].append(float(bll(p, y).sum()))
            per[nm]["n"].append(len(y)); per[nm]["user"].append(u)
            bins = np.minimum((np.clip(c, 0, 1) * 20).astype(int), 19)
            np.add.at(calib[nm], bins, np.column_stack([c, y, np.ones_like(y)]))
        pp = d["pg"][:, 1:].sum(1).astype(np.float64)
        y = passed.astype(np.float64)
        perb["R"].append(float(bll(pp, y).sum()))
        perb["Rb0"].append(float(bll(apply_logistic(models[("pass", "Rb0", other)], stack_x("Rb0", None, pp, ex)), y).sum()))
        perb["Rb"].append(float(bll(apply_logistic(models[("pass", "Rb", other)], stack_x("Rb", None, pp, ex)), y).sum()))
        perb["n"].append(len(y)); perb["user"].append(u)
    # readout
    res = {"users": len(users), "identity_vs_sm82c": ident, "prior_n": args.prior_n,
           "fit_rows_per_user": args.fit_rows_per_user, "bootstrap": {"reps": 2000, "seed": 0, "unit": "user"},
           "thresholds": {}, "recall": {}}
    for nm in names:
        p = {k: np.asarray(v, dtype=np.float64) for k, v in per[nm].items()}
        if not len(p["n"]):
            continue
        n = p["n"]; ref = p["C"].sum() / n.sum()
        t = {"users": int(len(n)), "rows": int(n.sum()), "event_rate": float(pooled[nm][0] / max(pooled[nm][2], 1)),
             "logloss": {k: float(p[k].sum() / n.sum()) for k in preds_a},
             "user_avg_logloss": {k: float(np.mean(p[k] / n)) for k in preds_a}, "vs_C_points": {}}
        for k in ("B0", "B1", "S0", "S"):
            m, lo, hi = boot(p[k], p["C"], n)
            t["vs_C_points"][k] = [100 * m / ref, 100 * lo / ref, 100 * hi / ref]
        m, lo, hi = boot(p["S"], p["S0"], n)
        t["S_vs_S0_points"] = [100 * m / ref, 100 * lo / ref, 100 * hi / ref]
        cb = calib[nm]
        nz = cb[:, 2] > 0
        t["calibration_C"] = {"mean_pred": float(cb[:, 0].sum() / cb[:, 2].sum()), "mean_obs": float(cb[:, 1].sum() / cb[:, 2].sum()),
                              "ece": float((np.abs(cb[nz, 0] - cb[nz, 1])).sum() / cb[:, 2].sum()),
                              "bins": [[round(a / c, 4), round(b / c, 4), int(c)] for a, b, c in cb[nz]]}
        smodel = models.get((nm, "S0", 0))
        if smodel is not None:   # Platt slope on logit C (fold 0), unstandardised
            t["platt_slope_fold0"] = float(smodel[2][1] / smodel[1][0])
        res["thresholds"][nm] = t
    pb = {k: np.asarray(v, dtype=np.float64) for k, v in perb.items()}
    n = pb["n"]
    rec_ = {"rows": int(n.sum()), "logloss": {k: float(pb[k].sum() / n.sum()) for k in preds_b}}
    # points = % of the big model (1 leak-free epoch) instant LogLoss on all instant labels of the same users
    big, rel = _instant_totals(BIG_ROWS, None, set(pb["user"].astype(int))), _instant_totals(RELEASE_ROWS, RELEASE_LABEL, set(pb["user"].astype(int)))
    rec_["big_instant"] = big; rec_["release_instant"] = rel
    if big and big["labels"]:
        denom = big["logloss"] * big["labels"]
        for a, b in (("Rb0", "R"), ("Rb", "R"), ("Rb", "Rb0")):
            d = pb[a] - pb[b]
            rng2 = np.random.default_rng(0)
            idx = rng2.integers(0, len(d), size=(2000, len(d)))
            # the all-label instant LogLoss change if only these later-day rows changed, in % of the big model
            share = big["labels"] / max(n.sum(), 1)   # all instant labels per dumped row
            stats = d[idx].sum(1) / n[idx].sum(1) * (n.sum() / big["labels"])
            m = d.sum() / big["labels"]
            rec_[f"{a}_vs_{b}_points"] = [100 * m / big["logloss"], 100 * np.percentile(stats, 2.5) / big["logloss"],
                                         100 * np.percentile(stats, 97.5) / big["logloss"]]
            rec_["later_day_share_of_instant_labels"] = 1 / share
    res["recall"] = rec_
    res["seconds"] = round(time.perf_counter() - t_start, 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"[sm94-analyze] {len(users)} users in {res['seconds']} s -> {out}", flush=True)
    for nm, t in res["thresholds"].items():
        v = t["vs_C_points"]
        print(f"  {nm}: rate {t['event_rate']:.3f} LL C {t['logloss']['C']:.5f} | B1 {v['B1'][0]:+.2f} S0 {v['S0'][0]:+.3f} "
              f"S {v['S'][0]:+.3f} [{v['S'][1]:+.3f}, {v['S'][2]:+.3f}] | ECE {t['calibration_C']['ece']:.4f}", flush=True)
    r = res["recall"]
    if "Rb_vs_Rb0_points" in r:
        print(f"  recall: speed beyond recalibration {r['Rb_vs_Rb0_points'][0]:+.3f} [{r['Rb_vs_Rb0_points'][1]:+.3f}, "
              f"{r['Rb_vs_Rb0_points'][2]:+.3f}] points", flush=True)
    return 0


def _instant_totals(path, label, users):
    """Label-weighted instant LogLoss over `users` from a scorer rows file (first label if `label` is None)."""
    if not Path(path).exists():
        return None
    lab, n, s = label, 0, 0.0
    seen = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "instant_labels" not in r or r.get("user_id") not in users:
            continue
        if lab is None:
            lab = r.get("label")
        if r.get("label") != lab or r["user_id"] in seen:
            continue
        seen.add(r["user_id"])
        n += r["instant_labels"]; s += r["instant_log_loss"] * r["instant_labels"]
    return {"label": lab, "users": len(seen), "labels": n, "logloss": s / n if n else None}


def main():
    p = ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--master", default=str(MASTER))
    d.add_argument("--ablation", default=str(ABLATION))
    d.add_argument("--users-file", default=str(USERS732))
    d.add_argument("--out-dir", required=True)
    d.add_argument("--device", default="cuda")
    d.add_argument("--workers", type=int, default=2)
    d.add_argument("--max-users", type=int, default=0)
    a = sub.add_parser("analyze")
    a.add_argument("--dump-dir", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--fit-rows-per-user", type=int, default=3000)
    a.add_argument("--prior-n", type=float, default=10.0)
    a.add_argument("--max-users", type=int, default=0)
    args = p.parse_args()
    return dump(args) if args.cmd == "dump" else analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
