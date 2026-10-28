"""SM66: per-user adaptation. Does a tiny per-user calibration, fitted only on a user's earlier reviews, lower LogLoss on later ones?

Model corrected: the released small model (SM56, w8+s8) and, as a comparison, the big model (the fork's RWKV, same users, same reviews).
Correction: logit' = a * logit(p) + b, fitted separately for the curve head (p_ahead) and the instant head (p_instant), with an
L2 prior toward the identity (a=1, b=0). Two forms: B = bias only, BT = bias + temperature.
Prior strengths are chosen by 5-fold cross-validation ACROSS USERS (one value per model/head/scheme, never per user).

Protocols (every user's dumped benchmark test reviews, in time order, review_th):
  offline : fit on the first half, evaluate on the second half.
  online  : expanding window, refit after every N reviews on everything seen so far, predict only the next block
            (the first block is uncorrected). Scored on the same second half (comparable to offline) and on all blocks after the first.
Controls: global (a, b) fitted on the first halves of all users outside the user's CV fold, applied unchanged; and the per-user
correction stacked on top of that global one.

usage: sm66_per_user_calibration.py --out artifacts/small-model/sm66 [--online-block 500 --block-sens 2000]
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SM_DIR = Path("C:/Users/jonat/workspace/wt/sm70/artifacts/small-model/sm70/dump")
BIG_DIR = Path("C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/sm24-dump")
GRID = [3.0, 30.0, 300.0, 3000.0, 30000.0]   # prior strength (pseudo-reviews); inf = no correction
HEADS = ("curve", "instant")
PCOL = {"curve": "p_ahead", "instant": "p_instant"}
MODELS = ("small", "big")
NFOLD = 5
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def losses(u, y):
    return np.logaddexp(0.0, -np.abs(u)) + np.maximum(u, 0.0) - y * u   # = -[y log p + (1-y) log(1-p)] for p = sigmoid(u)


def load_user(user):
    d = {}
    for m, folder, name in (("small", SM_DIR, f"{user}_sm56_w8+s8.npz"), ("big", BIG_DIR, f"{user}_fork_w8+s8.npz")):
        z = np.load(folder / name)
        d[m] = {h: logit(z[PCOL[h]].astype(np.float64)) for h in HEADS}
        d[m]["y"] = np.clip(z["y"].astype(np.float64), 0, 1)
        d[m]["th"] = z["review_th"]
    assert (d["small"]["th"] == d["big"]["th"]).all() and (d["small"]["y"] == d["big"]["y"]).all()
    return d


def fit(z, y, lam_b, lam_a, w0=None, iters=25):
    """Newton for logit' = a z + b with ridge lam_b*b^2/2 + lam_a*(a-1)^2/2 (lam_a=None: bias only, a=1)."""
    bias_only = lam_a is None
    a, b = (1.0, 0.0) if w0 is None else w0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(a * z + b)))
        r = p - y
        w = p * (1 - p)
        gb = r.sum() + lam_b * b
        hb = w.sum() + lam_b
        if bias_only:
            step = gb / hb
            step = max(-2.0, min(2.0, step))
            b -= step
            if abs(step) < 1e-6:
                break
            continue
        ga = (r * z).sum() + lam_a * (a - 1.0)
        haa = (w * z * z).sum() + lam_a
        hab = (w * z).sum()
        det = haa * hb - hab * hab
        sa = (ga * hb - gb * hab) / det
        sb = (haa * gb - hab * ga) / det
        sa = max(-1.5, min(1.5, sa))
        sb = max(-2.0, min(2.0, sb))
        a -= sa
        b -= sb
        if max(abs(sa), abs(sb)) < 1e-6:
            break
    return a, b


def fit_global(z, y):
    return fit(z, y, 0.0, 0.0, iters=50)


COMBOS = [("B", lb, None) for lb in GRID] + [("BT", lb, la) for lb in GRID for la in GRID]


def combo_table(z_fit, y_fit, z_ev, y_ev):
    """Offline: fit every combo on (z_fit, y_fit), return summed eval loss per combo: B[len GRID], BT[len GRID, len GRID]."""
    B = np.zeros(len(GRID))
    BT = np.zeros((len(GRID), len(GRID)))
    for i, lb in enumerate(GRID):
        a, b = fit(z_fit, y_fit, lb, None)
        B[i] = losses(z_ev + b, y_ev).sum()
        for j, la in enumerate(GRID):
            a, b = fit(z_fit, y_fit, lb, la)
            BT[i, j] = losses(a * z_ev + b, y_ev).sum()
    return B, BT


def online_table(z, y, n, s, block):
    """Online: expanding-window refit every `block` reviews; per combo summed loss on [s:n] (second half) and on [block:n]."""
    nb = len(GRID)
    out = {"B2": np.zeros(nb), "BT2": np.zeros((nb, nb)), "BA": np.zeros(nb), "BTA": np.zeros((nb, nb))}
    if s < block:   # second-half reviews before the first corrected block stay uncorrected
        pre = float(losses(z[s:min(block, n)], y[s:min(block, n)]).sum())
        out["B2"] += pre
        out["BT2"] += pre
    state_b = [None] * nb
    state_bt = [[None] * nb for _ in range(nb)]
    for start in range(block, n, block):
        end = min(start + block, n)
        zf, yf = z[:start], y[:start]
        ze, ye = z[start:end], y[start:end]
        mask = np.arange(start, end) >= s
        for i, lb in enumerate(GRID):
            st = fit(zf, yf, lb, None, w0=state_b[i], iters=8)
            state_b[i] = st
            le = losses(st[0] * ze + st[1], ye)
            out["BA"][i] += le.sum()
            out["B2"][i] += le[mask].sum()
            for j, la in enumerate(GRID):
                st = fit(zf, yf, lb, la, w0=state_bt[i][j], iters=8)
                state_bt[i][j] = st
                le = losses(st[0] * ze + st[1], ye)
                out["BTA"][i, j] += le.sum()
                out["BT2"][i, j] += le[mask].sum()
    return out


def process_user(args):
    user, glob, blocks = args   # glob[model][head] = (a, b) global calibration for this user's CV fold
    d = load_user(user)
    y = d["small"]["y"]
    n = len(y)
    s = n // 2
    res = {"user": user, "n": n, "s": s}
    for m in MODELS:
        for h in HEADS:
            z = d[m][h]
            r = {}
            ev_y = y[s:]
            r["L0"] = float(losses(z[s:], ev_y).sum())
            ga, gb = glob[m][h]
            zg = ga * z + gb
            r["Lg"] = float(losses(zg[s:], ev_y).sum())
            r["off_B"], r["off_BT"] = combo_table(z[:s], y[:s], z[s:], ev_y)
            r["offg_B"], r["offg_BT"] = combo_table(zg[:s], y[:s], zg[s:], ev_y)
            for block in blocks:
                o = online_table(z, y, n, s, block)
                o["L0A"] = float(losses(z[block:], y[block:]).sum())
                o["nA"] = n - block
                r[f"on{block}"] = o
            res[(m, h)] = r
    return res


def build_folds(users, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(users))
    fold = np.empty(len(users), dtype=int)
    fold[perm] = np.arange(len(users)) % NFOLD
    return {u: int(f) for u, f in zip(users, fold)}


def global_params(users, fold):
    """For every fold, (a, b) per model/head fitted on the first halves of the users outside it."""
    zs = {(m, h): {} for m in MODELS for h in HEADS}
    ys = {}
    for u in users:
        d = load_user(u)
        s = len(d["small"]["y"]) // 2
        ys[u] = d["small"]["y"][:s]
        for m in MODELS:
            for h in HEADS:
                zs[(m, h)][u] = d[m][h][:s]
    out = {}
    for f in range(NFOLD):
        tr = [u for u in users if fold[u] != f]
        yy = np.concatenate([ys[u] for u in tr])
        out[f] = {m: {h: fit_global(np.concatenate([zs[(m, h)][u] for u in tr]), yy) for h in HEADS} for m in MODELS}
    return out


# ---------------------------------------------------------------- aggregation
def boot_idx(U, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, U, size=(B, U))


def ci(delta_num, den, idx):
    """pooled delta (sum num / sum den) with user-bootstrap 95% CI."""
    point = delta_num.sum() / den.sum()
    bs = delta_num[idx].sum(axis=1) / den[idx].sum(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def cv_select(table, nvec, fold_of):
    """table[u, combo...] summed eval loss; choose, per fold, the combo minimising the pooled loss of the OTHER folds' users.
    Returns the chosen per-user summed loss and the combo index chosen for every fold."""
    U = table.shape[0]
    flat = table.reshape(U, -1)
    chosen = np.zeros(U)
    picks = {}
    for f in range(NFOLD):
        tr = fold_of != f
        k = int(np.argmin(flat[tr].sum(axis=0)))
        picks[f] = k
        chosen[~tr] = flat[~tr, k]
    return chosen, picks


def pick_names(picks, kind):
    out = {}
    for f, k in picks.items():
        if kind == "B":
            out[f] = {"lam_b": GRID[k]}
        else:
            out[f] = {"lam_b": GRID[k // len(GRID)], "lam_a": GRID[k % len(GRID)]}
    return out


def aggregate(results, users, fold, blocks, tercile_of, idx):
    U = len(users)
    fold_of = np.array([fold[u] for u in users])
    n_ev = np.array([r["n"] - r["s"] for r in results], float)
    rep = {"users": U, "eval_reviews_second_half": int(n_ev.sum()), "schemes": {}}
    # per scheme: (model, head) -> dict(L0, Lc, n)
    sch = {}

    def stack(key, m, h):
        return np.stack([r[(m, h)][key] for r in results])

    for m in MODELS:
        for h in HEADS:
            L0 = np.array([r[(m, h)]["L0"] for r in results])
            Lg = np.array([r[(m, h)]["Lg"] for r in results])
            entries = {}
            entries["global_static"] = (L0, Lg, n_ev, {"note": "static global (a,b) from other folds' first halves"})
            for tag, key_prefix, base in (("raw", "off", L0), ("on_global", "offg", Lg)):
                for kind in ("B", "BT"):
                    t = stack(f"{key_prefix}_{kind}", m, h)
                    ch, picks = cv_select(t, n_ev, fold_of)
                    name = {"raw": f"offline_{kind}", "on_global": f"offline_{kind}_on_global"}[tag]
                    entries[name] = (L0, ch, n_ev, {"cv_prior": pick_names(picks, kind)})
            for block in blocks:
                for kind in ("B", "BT"):
                    t2 = np.stack([r[(m, h)][f"on{block}"][f"{kind}2"] for r in results])
                    ch, picks = cv_select(t2, n_ev, fold_of)
                    entries[f"online{block}_{kind}"] = (L0, ch, n_ev, {"cv_prior": pick_names(picks, kind)})
                    # all blocks after the first (own baseline and denominator)
                    tA = np.stack([r[(m, h)][f"on{block}"][f"{kind}A"] for r in results])
                    LA0 = np.array([r[(m, h)][f"on{block}"]["L0A"] for r in results])
                    nA = np.array([r[(m, h)][f"on{block}"]["nA"] for r in results], float)
                    chA, picksA = cv_select(tA, nA, fold_of)
                    entries[f"online{block}_{kind}_allblocks"] = (LA0, chA, nA, {"cv_prior": pick_names(picksA, kind)})
            sch[(m, h)] = entries
    names = list(sch[("small", "curve")].keys())
    for h in HEADS:
        for name in names:
            row = {}
            for m in MODELS:
                L0, Lc, nn, extra = sch[(m, h)][name]
                dnum = Lc - L0
                pt, lo, hi = ci(dnum, nn, idx)
                base = float(L0.sum() / nn.sum())
                row[m] = {"logloss_before": base, "delta": pt, "ci": [lo, hi], "delta_pct": 100 * pt / base,
                          "frac_users_improved": float((dnum < 0).mean()), **extra}
            # gap small - big, before/after, paired bootstrap
            L0s, Lcs, nn, _ = sch[("small", h)][name]
            L0b, Lcb, _, _ = sch[("big", h)][name]
            gap0 = float((L0s - L0b).sum() / nn.sum())
            pt, lo, hi = ci((Lcs - L0s) - (Lcb - L0b), nn, idx)
            row["gap"] = {"small_minus_big_before": gap0, "small_minus_big_after": gap0 + pt, "change": pt, "ci": [lo, hi]}
            # versus the global control (offline schemes share the second-half eval set)
            if name.startswith("offline") and name != "global_static":
                Lg = sch[("small", h)]["global_static"][1]
                row["small_vs_global_static"] = dict(zip(("delta", "lo", "hi"), ci(Lc_of(sch, "small", h, name) - Lg, n_ev, idx)))
                Lgb = sch[("big", h)]["global_static"][1]
                row["big_vs_global_static"] = dict(zip(("delta", "lo", "hi"), ci(Lc_of(sch, "big", h, name) - Lgb, n_ev, idx)))
            if name.startswith("online") and not name.endswith("allblocks"):
                Lg = sch[("small", h)]["global_static"][1]
                row["small_vs_global_static"] = dict(zip(("delta", "lo", "hi"), ci(Lc_of(sch, "small", h, name) - Lg, n_ev, idx)))
                Lgb = sch[("big", h)]["global_static"][1]
                row["big_vs_global_static"] = dict(zip(("delta", "lo", "hi"), ci(Lc_of(sch, "big", h, name) - Lgb, n_ev, idx)))
            # terciles by user size
            ter = {}
            for t in range(3):
                sel = tercile_of == t
                tr = {}
                for m in MODELS:
                    L0, Lc, nn_m, _ = sch[(m, h)][name]
                    sidx = boot_idx(int(sel.sum()), 2000, 0)
                    tr[m] = dict(zip(("delta", "lo", "hi"), ci((Lc - L0)[sel], nn_m[sel], sidx)))
                    tr[m]["logloss_before"] = float(L0[sel].sum() / nn_m[sel].sum())
                ter[f"T{t + 1}"] = tr
            row["terciles"] = ter
            rep["schemes"].setdefault(h, {})[name] = row
    return rep


def Lc_of(sch, m, h, name):
    return sch[(m, h)][name][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/small-model/sm66")
    ap.add_argument("--online-block", type=int, default=500)
    ap.add_argument("--block-sens", type=int, nargs="*", default=[2000])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-users", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    blocks = [args.online_block] + list(args.block_sens)
    users = sorted({p.name.split("_")[0] for p in SM_DIR.glob("*_sm56_w8+s8.npz")}, key=int)
    if args.max_users:
        users = users[: args.max_users]
    fold = build_folds(users)
    t0 = time.time()
    glob = global_params(users, fold)
    print("global params", {f: {m: {h: [round(x, 4) for x in glob[f][m][h]] for h in HEADS} for m in MODELS} for f in glob}, flush=True)
    sizes = {u: len(np.load(SM_DIR / f"{u}_sm56_w8+s8.npz")["y"]) for u in users}
    order = sorted(users, key=lambda u: -sizes[u])
    jobs = [(u, glob[fold[u]], blocks) for u in order]
    res = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(process_user, jobs, chunksize=1)):
            res[r["user"]] = r
            if i % 10 == 0:
                print(f"{i + 1}/{len(jobs)} users done, {time.time() - t0:.0f}s", flush=True)
    results = [res[u] for u in users]
    nsz = np.array([sizes[u] for u in users])
    cuts = np.percentile(nsz, [100 / 3, 200 / 3])
    tercile_of = np.digitize(nsz, cuts)
    idx = boot_idx(len(users))
    rep = aggregate(results, users, fold, blocks, tercile_of, idx)
    rep["tercile_review_ranges"] = [[int(nsz[tercile_of == t].min()), int(nsz[tercile_of == t].max()), int((tercile_of == t).sum())] for t in range(3)]
    rep["global_params"] = {str(f): {m: {h: list(map(float, glob[f][m][h])) for h in HEADS} for m in MODELS} for f in glob}
    rep["config"] = {"grid": GRID, "folds": NFOLD, "online_blocks": blocks, "bootstrap": "2000 user resamples, seed 0", "split": "first half / second half in time",
                     "users": users, "runtime_seconds": time.time() - t0}
    (out / "summary.json").write_text(json.dumps(rep, indent=1))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
