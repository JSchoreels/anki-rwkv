#!/usr/bin/env python
"""Response-time prediction (SM82) stage 2: answer-time curves, fast-Good patterns and a time-aware workload simulation.

The frozen released small model + the answer-time head on richer frozen features (SM82c) is queried "what if this card
were reviewed after t days instead of the actual interval": only the instant (query) row of the review changes; every
earlier real review, the user's state and the card's history stay as they were.

What changes in the query row (start tensor, the time scorer's inputs) for a shift of D = t - e days (e = the actual
interval the model saw): scaled_elapsed_days, scaled_elapsed_days_cumulative (+D days), scaled_elapsed_seconds,
scaled_elapsed_seconds_cumulative (+86,400 D s; time of day unchanged, so the sin/cos columns stay), day_of_week, and the
day-offset encodings of the review day (sin/cos rotated by 2 pi D / period; the card's first-day encodings stay). The
head's explicit feature elapsed_days_ln1p (column 23) gets ln(1 + t) / 3. Everything else (session counters, the
user's running medians, the card's last times) stays at the real review's values. Query rows are skip rows (their
outputs never feed another row; the CPU check C2 verifies it), so all reviews of a user are queried in one forward per
grid point.

  sample   --out users.json                          the user sample (seeded) from the SM82c time rows (732 set).
  check    --master M --user U --out check.json      CPU check (C1 identity = SM82c's row, C2 query-row independence,
                                                     C3 D = 0 leaves the start bitwise, D then -D round trip).
  query    --master M --users-file F --out-dir DIR   the curve queries (GPU), one npz per user, resumable.
  analyze  --out-dir DIR --result-dir R              curves, calibration, patterns, workload -> stage2-result.{md,json}.
"""
from argparse import ArgumentParser
from dataclasses import replace
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

NAME = "response-time prediction (SM82) stage 2"
SM82C_ROWS = Path("C:/Users/jonat/workspace/wt/sm82c/artifacts/small-model/sm82c/gpu/sm82c-time.jsonl")   # read only
SM82C_MASTER = Path("C:/Users/jonat/workspace/wt/sm82c/artifacts/small-model/sm82c-head-released/master.pth")
GRID = ([(f"x{k:g}", "mult", k) for k in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)]
        + [(f"d{d}", "abs", d) for d in (1, 3, 7, 14, 30, 90, 365)])
IDENTITY = GRID.index(("x1", "mult", 1.0))
FAST_MS = 500.0 * 120.0 ** (7 / 14)      # 5,477 ms: the top edge of bin 6 (SM82's "under 5.5 s")
N_CARD_FEATURES, ID_DIMS, DAY0 = 24, 40, 64
COL = {"ed": 0, "edc": 1, "es": 2, "esc": 5, "dow": 17, "is_query": 23}
EXPLICIT_ELAPSED = 23
SUB = 3000                                # reviews per user kept for the curve grid


def below_normal():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:  # noqa: BLE001  best effort
        pass


def bin_centres_s(cap_ms):
    """Seconds per bin for expected recorded time: bin 0 = 0.3 s, bins 1..14 the arithmetic midpoints, bin 15 = 60 s;
    every centre clipped at the user's own cap (Anki records at most the cap)."""
    from scripts.expertium_options import TIME_BIN_EDGES_MS
    e = np.asarray(TIME_BIN_EDGES_MS) / 1000.0
    c = np.r_[0.3, (e[:-1] + e[1:]) / 2, 60.0]
    return np.minimum(c, cap_ms / 1000.0)


# ------------------------------------------------------------------ counterfactual query rows
def counterfactual_start(start, rows, shift):
    """start [N, 92] -> a copy where query rows `rows` look reviewed `shift` days later (shift may be negative);
    rows with shift 0 keep their values bitwise."""
    import torch
    from rwkv.config import DAY_OFFSET_ENCODE_PERIODS
    from rwkv.data_processing import STATISTICS as ST
    out = start.clone()
    changed = shift != 0
    if not bool(changed.any()):
        return out
    v = start.index_select(0, rows).double()
    d = shift.double()

    def moved(col, key, unit):
        m, s = ST[key + "_mean"], ST[key + "_std"]
        x = torch.exp(v[:, col] * s + m) - 1 - 1e-5
        return (torch.log(1 + 1e-5 + (x + unit * d).clamp(min=0)) - m) / s

    new = {COL["ed"]: moved(COL["ed"], "elapsed_days", 1.0), COL["edc"]: moved(COL["edc"], "elapsed_days_cumulative", 1.0),
           COL["es"]: moved(COL["es"], "elapsed_seconds", 86400.0),
           COL["esc"]: moved(COL["esc"], "elapsed_seconds_cumulative", 86400.0)}
    dow = torch.round(v[:, COL["dow"]] * 3 + 3)
    new[COL["dow"]] = (torch.remainder(dow + d, 7) - 3) / 3
    for p, period in enumerate(DAY_OFFSET_ENCODE_PERIODS):
        a = 2 * math.pi / period * d
        c0 = DAY0 + 4 * p
        sn, cs = v[:, c0], v[:, c0 + 1]
        new[c0] = sn * torch.cos(a) + cs * torch.sin(a)
        new[c0 + 1] = cs * torch.cos(a) - sn * torch.sin(a)
    for col, val in new.items():
        out[rows, col] = torch.where(changed, val.to(out.dtype), start.index_select(0, rows)[:, col])
    return out


def model_elapsed(start, rows):
    import torch
    from rwkv.data_processing import STATISTICS as ST
    v = start.index_select(0, rows)[:, COL["ed"]].double()
    x = torch.exp(v * ST["elapsed_days_std"] + ST["elapsed_days_mean"]) - 1 - 1e-5
    return torch.round(x).long()


def targets(kind, value, e):
    if kind == "mult":
        return np.maximum(1, np.round(value * e)).astype(np.int64)
    return np.full_like(e, int(value))


class Querier:
    def __init__(self, master, device):
        import torch
        from scripts import sm82c_head as h
        self.h, self.device = h, torch.device(device)
        sd, _, self.main, _ = h.load_heads(master, Path(master).with_name("ablation-head.pth"), self.device)
        self.frozen = h.Frozen(h.build_frozen({k: v.float() for k, v in sd.items()}, self.device))

    def outputs(self, gb, rows_sel, table_sel, cents):
        """One forward -> (pg [n, 4], E sec per grade [n, 4], P(under 5.5 s) per grade [n, 4], pt [n, 4, 16]) at the
        instant rows `rows_sel` (canonical row ids), explicit features `table_sel`."""
        import torch
        h = self.h
        qi, q_th, feats, o = self.frozen.forward(gb)
        idx = torch.searchsorted(qi, rows_sel)          # qi ascending (nonzero); rows_sel are instant rows
        if not bool((qi[idx.clamp(max=len(qi) - 1)] == rows_sel).all()):
            raise RuntimeError("query rows outside the instant rows")
        x = torch.cat([feats[idx], table_sel], -1)
        pg = torch.softmax(o[3][0].float()[rows_sel], -1)
        pt = torch.softmax(self.main(x).float().view(-1, 4, h.BINS), -1)
        es = (pt * torch.as_tensor(cents, device=self.device, dtype=torch.float32)).sum(-1)
        pf = pt[..., :7].sum(-1)
        return pg, es, pf, pt


def user_query(q, batch, table, data, user, txn, sub_n=SUB, grid=GRID, only=None):
    """All grid points for one user. Returns the npz dict and a summary record."""
    import torch
    from scripts.sm82_time import test_reviews, user_sums, CAP_MS
    ths, frame, lower, cens, cap = test_reviews(data, user, txn)
    gb = batch.to(q.device)
    lab = gb.labels[0]
    qi = torch.nonzero((lab[:, 6] == 1) & (lab[:, 4] == 1)).squeeze(-1)
    th_all = gb.label_review_th[0].long()[qi]
    order = torch.argsort(th_all)
    th_sorted, qi_sorted = th_all[order], qi[order]
    want = torch.as_tensor(ths, device=q.device)
    pos = torch.searchsorted(th_sorted, want).clamp(max=len(th_sorted) - 1)
    if not bool((th_sorted[pos] == want).all()):
        raise RuntimeError("test reviews without an instant row")
    rows = qi_sorted[pos]
    start = gb.start
    if start.shape[1] != DAY0 + 28 or not bool((start[rows, COL["is_query"]] == 1).all()):
        raise RuntimeError(f"unexpected start layout {tuple(start.shape)}")
    e_model = model_elapsed(start, rows).cpu().numpy()
    e_rev = frame["elapsed_days"].to_numpy(np.int64)[ths - 1]
    tab = torch.as_tensor(table, device=q.device)[want - 1]
    cents = bin_centres_s(cap)
    rng = np.random.default_rng(user)
    sub = np.sort(rng.choice(len(ths), size=min(sub_n, len(ths)), replace=False)) if only is None else np.asarray(only)
    sub_t = torch.as_tensor(sub, device=q.device)
    out = {"th": ths.astype(np.int32), "e": e_model.astype(np.int32), "e_revlog": e_rev.astype(np.int32),
           "grade": (frame["rating"].to_numpy()[ths - 1]).astype(np.int8), "dur": frame["duration"].to_numpy(np.int64)[ths - 1].astype(np.int32),
           "lower": lower[ths - 1].astype(np.int8), "cens": cens[ths - 1].astype(np.int8), "cap_ms": np.int32(cap), "sub": sub.astype(np.int32)}
    G, S = len(grid), len(sub)
    tt = np.zeros((G, S), np.int32)
    spg, ses, spf = (np.zeros((G, S, 4), np.float32) for _ in range(3))
    rec = {"user_id": user, "reviews": int(len(ths)), "sub": int(S), "e_mismatch_revlog": int((e_model != e_rev).sum())}
    with torch.inference_mode():
        for g, (name, kind, value) in enumerate(grid):
            t_sub = targets(kind, value, e_model[sub])
            tt[g] = t_sub
            if g == IDENTITY:
                pg, es, pf, pt = q.outputs(gb, rows, tab, cents)
                out.update(full_pg=pg.cpu().numpy(), full_es=es.cpu().numpy(), full_pf=pf.cpu().numpy())
                grade = torch.as_tensor(out["grade"].astype(np.int64) - 1, device=q.device)
                cond = pt[torch.arange(len(rows), device=q.device), grade].double().cpu().numpy()
                d = out["dur"].astype(np.int64)
                t = {"grade": (out["grade"] - 1).astype(np.int8), "lower": out["lower"], "cens": out["cens"],
                     "logd": np.log(np.maximum(d, 1) / 1000.0), "uncensored": (d < CAP_MS) & (out["cens"] == 0)}
                rec["identity_sums"] = user_sums(cond, t)
                spg[g], ses[g], spf[g] = out["full_pg"][sub], out["full_es"][sub], out["full_pf"][sub]
                continue
            shift = torch.as_tensor(t_sub - e_model[sub], device=q.device)
            gb_g = replace(gb, start=counterfactual_start(start, rows[sub_t], shift))
            tab_g = tab[sub_t].clone()
            tab_g[:, EXPLICIT_ELAPSED] = torch.log1p(torch.as_tensor(t_sub, device=q.device, dtype=torch.float32)) / 3.0
            pg, es, pf, _ = q.outputs(gb_g, rows[sub_t], tab_g, cents)
            spg[g], ses[g], spf[g] = pg.cpu().numpy(), es.cpu().numpy(), pf.cpu().numpy()
            del gb_g
    out.update(grid_t=tt, grid_pg=spg, grid_es=ses, grid_pf=spf)
    return out, rec


def query(args):
    from scripts import small_model_heldout as smh  # noqa: F401  first: sets RWKV_NO_JIT
    import torch
    from scripts import compress_common as cc
    from scripts import sm82c_head as h
    from scripts.sm82_time import label_db
    from scripts.experiment_provenance import process_provenance
    below_normal()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("HELDOUT_GPU_MEMORY_FRACTION", "0.8")))
    else:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    out = Path(args.out_dir); (out / "users").mkdir(parents=True, exist_ok=True)
    index = out / "query.jsonl"
    done = {json.loads(l)["user_id"] for l in index.read_text().splitlines() if "error" not in l} if index.exists() else set()
    users = json.loads(Path(args.users_file).read_text())["users"]
    todo = [u for u in users if u not in done]
    prov = process_provenance()
    (out / f"provenance-{int(time.time())}.json").write_text(json.dumps(prov, indent=1))
    q = Querier(args.master, args.device)
    data = cc.data_path()
    loader = torch.utils.data.DataLoader(h.ScoreBatches(todo, data, cc.run_info()["review_counts"]), batch_size=1,
                                         num_workers=args.workers, collate_fn=lambda x: x[0],
                                         prefetch_factor=1 if args.workers else None)
    env = label_db()
    started = time.perf_counter()
    print(f"[sm82s2-query] {len(users)} users, {len(todo)} to query, grid {len(GRID)}, commit {prov['git_commit'][:8]}"
          f" dirty {prov['git_dirty']}", flush=True)
    with index.open("a", encoding="utf-8") as f:
        for k, (user, batch, error, prep, table) in enumerate(loader):
            t0 = time.perf_counter()
            if batch is None:
                f.write(json.dumps({"user_id": user, "error": error}) + "\n"); f.flush(); continue
            try:
                with env.begin() as txn:
                    arrays, rec = user_query(q, batch, table, data, user, txn)
            except torch.OutOfMemoryError as oom:
                q.frozen.hp.clear(); q.frozen.capture.snaps = []
                torch.cuda.empty_cache()
                f.write(json.dumps({"user_id": user, "error": f"OOM: {oom}"[:300]}) + "\n"); f.flush(); continue
            np.savez(out / "users" / f"{user}.npz", **arrays)
            rec.update(prepare_seconds=round(prep, 2), query_seconds=round(time.perf_counter() - t0, 2),
                       commit=prov["git_commit"], device=str(device))
            f.write(json.dumps(rec) + "\n"); f.flush()
            if (k + 1) % 10 == 0:
                print(f"[sm82s2-query] {k + 1}/{len(todo)} users, {time.perf_counter() - started:.0f} s", flush=True)
    env.close()
    print(f"[sm82s2-query] done {len(todo)} users in {time.perf_counter() - started:.0f} s -> {out}", flush=True)
    return 0


# ------------------------------------------------------------------ sample + CPU check
def sample(args):
    rows = [json.loads(l) for l in SM82C_ROWS.read_text().splitlines()]
    users = sorted({r["user_id"] for r in rows if "error" not in r and r.get("head") == "model"})
    rng = np.random.default_rng(args.seed)
    pick = sorted(int(u) for u in rng.choice(users, size=args.n, replace=False))
    Path(args.out).write_text(json.dumps({"users": pick, "rule": f"{args.n} of the {len(users)} users with SM82c time rows "
                                          f"(732 set minus never-fit), numpy default_rng({args.seed}).choice, no replacement"}, indent=1))
    print(f"[sm82s2-sample] {len(pick)} users -> {args.out}")


def check(args):
    from scripts import small_model_heldout as smh  # noqa: F401
    import torch
    from scripts import compress_common as cc
    from scripts import sm82c_head as h
    from scripts.sm82_time import label_db, test_reviews
    below_normal()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    q = Querier(args.master, "cpu")
    data = cc.data_path()
    user = args.user
    _, batch, error, _, table = h.ScoreBatches([user], data, cc.run_info()["review_counts"])[0]
    if batch is None:
        raise SystemExit(error)
    env = label_db()
    report = {"user": user, "checks": {}}
    with env.begin() as txn:
        arrays, rec = user_query(q, batch, table, data, user, txn, sub_n=args.sub)
        ths = test_reviews(data, user, txn)[0]
    ref = {json.loads(l)["user_id"]: json.loads(l) for l in SM82C_ROWS.read_text().splitlines()
           if '"head": "model"' in l and "error" not in l}.get(user)
    ll, ll_ref = rec["identity_sums"]["ll"], (ref or {}).get("model", {}).get("ll")
    c1 = ref is not None and abs(ll - ll_ref) <= 1e-5 * abs(ll_ref) and rec["identity_sums"]["n"] == ref["model"]["n"]
    report["checks"]["C1_identity_equals_sm82c_row"] = {"pass": bool(c1), "ll": ll, "ll_sm82c_gpu": ll_ref,
                                                         "n": rec["identity_sums"]["n"]}
    # C2: modify half of the subsample rows only (x2): the other half = identity outputs, the modified half = the run
    # that modifies all of them.
    gb = batch
    lab = gb.labels[0]
    qi = torch.nonzero((lab[:, 6] == 1) & (lab[:, 4] == 1)).squeeze(-1)
    th_all = gb.label_review_th[0].long()[qi]
    o = torch.argsort(th_all)
    rows = qi[o][torch.searchsorted(th_all[o], torch.as_tensor(ths))]
    sub = arrays["sub"]
    e = arrays["e"]
    tab = torch.as_tensor(table)[torch.as_tensor(ths) - 1]
    cents = bin_centres_s(int(arrays["cap_ms"]))
    g2 = [n for n, _, _ in GRID].index("x2")
    half = sub[::2]
    t_half = targets("mult", 2.0, e[half])
    with torch.inference_mode():
        gb_h = replace(gb, start=counterfactual_start(gb.start, rows[half], torch.as_tensor(t_half - e[half])))
        tab_s = tab[torch.as_tensor(sub)].clone()
        mod = np.isin(sub, half)
        tab_s[torch.as_tensor(mod), EXPLICIT_ELAPSED] = torch.log1p(torch.as_tensor(t_half, dtype=torch.float32)) / 3.0
        pg, es, _, _ = q.outputs(gb_h, rows[torch.as_tensor(sub)], tab_s, cents)
    pg, es = pg.numpy(), es.numpy()
    d_un = float(np.abs(pg[~mod] - arrays["full_pg"][sub[~mod]]).max()) if (~mod).any() else 0.0
    d_un_t = float(np.abs(es[~mod] - arrays["full_es"][sub[~mod]]).max()) if (~mod).any() else 0.0
    d_mod = float(np.abs(pg[mod] - arrays["grid_pg"][g2][mod]).max())
    d_mod_t = float(np.abs(es[mod] - arrays["grid_es"][g2][mod]).max())
    report["checks"]["C2_query_rows_independent"] = {"pass": max(d_un, d_mod) <= 1e-5 and max(d_un_t, d_mod_t) <= 1e-4,
                                                     "max_abs_unmodified_vs_identity_pg": d_un, "time_s": d_un_t,
                                                     "max_abs_modified_vs_all_modified_pg": d_mod, "time_s_mod": d_mod_t}
    # C3: zero shift leaves start bitwise; the shifted elapsed-days column = the scaler applied to the integer e + D
    # (float32 cast only); +D then -D comes back within the float32 storage error of elapsed seconds (2e-3 scaled)
    from rwkv.data_processing import scale_elapsed_days
    s0 = counterfactual_start(gb.start, rows, torch.zeros(len(rows), dtype=torch.long))
    sh = torch.as_tensor(np.random.default_rng(0).integers(1, 400, len(rows)))
    fwd = counterfactual_start(gb.start, rows, sh)
    want_ed = torch.as_tensor(scale_elapsed_days(e.astype(np.float64) + sh.numpy()), dtype=torch.float32)
    ed_gap = float((fwd[rows, COL["ed"]] - want_ed).abs().max())
    back = counterfactual_start(fwd, rows, -sh)
    rt = float((back - gb.start).abs().max())
    me = float(np.abs(arrays["e"] - arrays["e_revlog"]).max())
    zero = bool(torch.equal(s0, gb.start))
    report["checks"]["C3_shift_exact_and_round_trip"] = {"pass": zero and ed_gap <= 1e-6 and rt <= 2e-3, "zero_shift_bitwise": zero,
                                                        "shifted_elapsed_days_vs_scaler_max_abs": ed_gap, "round_trip_max_abs": rt,
                                                        "e_model_vs_revlog_mismatches": rec["e_mismatch_revlog"], "max_abs_e_gap_days": me}
    # C4 (report only): mean recall and time over the grid
    report["grid_means"] = {n: {"t_mean": float(arrays["grid_t"][g].mean()), "recall": float(1 - arrays["grid_pg"][g][:, 0].mean()),
                                "E_time_s": float((arrays["grid_pg"][g] * arrays["grid_es"][g]).sum(-1).mean())}
                            for g, (n, _, _) in enumerate(GRID)}
    report["pass"] = all(v["pass"] for v in report["checks"].values())
    env.close()
    if args.npz:
        np.savez(args.npz, **arrays)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"[sm82s2-check] {'PASS' if report['pass'] else 'FAIL'} user {user}: " +
          ", ".join(f"{k} {v['pass']}" for k, v in report["checks"].items()))
    return 0 if report["pass"] else 1


def main():
    ap = ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample"); p.add_argument("--out", required=True); p.add_argument("--n", type=int, default=150); p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("check"); p.add_argument("--master", default=str(SM82C_MASTER)); p.add_argument("--user", type=int, required=True)
    p.add_argument("--out", required=True); p.add_argument("--sub", type=int, default=400); p.add_argument("--npz")
    p = sub.add_parser("query"); p.add_argument("--master", default=str(SM82C_MASTER)); p.add_argument("--users-file", required=True)
    p.add_argument("--out-dir", required=True); p.add_argument("--device", default="cuda"); p.add_argument("--workers", type=int, default=3)
    p = sub.add_parser("analyze"); p.add_argument("--out-dir", required=True); p.add_argument("--result-dir", required=True)
    p.add_argument("--procs", type=int, default=6)
    args = ap.parse_args()
    if args.cmd == "analyze":
        from scripts.sm82s2_analyze import analyze
        return analyze(args)
    return {"sample": sample, "check": check, "query": query}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
