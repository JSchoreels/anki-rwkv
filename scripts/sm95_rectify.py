"""The grade order by construction (SM95): scorer probe rows and the order read-out of rectified curves.

  scorer   scripts/small_model_heldout.py --grade-rectify: every labelled committed row of the full-history batch gets
           the three what-if rows of the other grades right before it (install_scorer_probes, the pandas path of
           rwkv.data_processing.create_sample) and the model's curve on that row becomes its grade's value in the
           rectified 4-vector (expertium/model/grade_rectify.py). Probes are skip rows with has_label = False, so the
           state, the instant head and the set of scored reviews are unchanged; only the curve probability changes, to
           the one the scheduler uses (the four buttons' curves pooled to be ordered).
  derive   python scripts/sm95_rectify.py derive --measure DIR --out DIR2 [--rectify]
           from a scripts/sm33_grade_order.py measure folder (per sampled review: the four what-if curves' parameters),
           recompute per review: unrounded intervals at 0.80 / 0.90 / 0.95 (bisection on log t), curve values at
           1 / 7 / 30 / 180 days, and the fork's displayed intervals at 0.90 on its grid with and without its PAVA, for
           the raw curves or (--rectify) the rectified ones. -> DIR2/users/<user>.npz + DIR2/derive-log.json
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expertium.model.grade_rectify import pava4_numpy  # noqa: E402

SECONDS_PER_DAY = 86400.0
MAX_DAYS = 36500.0
TARGETS = (0.80, 0.90, 0.95)
HORIZONS = (1.0, 7.0, 30.0, 180.0)


# ---------------------------------------------------------------- scorer probe rows (pandas path)
def install_scorer_probes():
    """Wrap rwkv.data_processing.add_queries: after the query rows, insert the three other-grade probes before every
    labelled committed row (the training layout of scripts/gpu_r01_samples.add_rectify_probes at density 1)."""
    import rwkv.data_processing as dp
    if getattr(dp.add_queries, "_sm95", False):
        return
    original = dp.add_queries

    def add_queries_with_rectify_probes(section_df, equalize_review_ths):
        import pandas as pd
        df = original(section_df, equalize_review_ths)
        df["sm95_order"] = np.where(df["is_query"].to_numpy() == 1, 0, 5)
        real = df[(~df["skip"].astype(bool)) & df["has_label"].astype(bool)]
        copies = [df]
        for g in (1, 2, 3, 4):
            c = real[real["rating"].to_numpy() != g].copy()
            c["rating"] = g
            for i in (1, 2, 3, 4):
                c[f"rating_{i}"] = int(i == g)
            c["y"] = int(g > 1)
            c["skip"] = True
            c["has_label"] = False
            c["label_elapsed_days"] = float(-g)
            c["sm95_order"] = g
            copies.append(c)
        df = pd.concat(copies, ignore_index=True)
        df = df.sort_values(by=["review_th", "sm95_order"], kind="stable").reset_index(drop=True)
        df["index"] = range(0, len(df))
        return df.drop(columns=["sm95_order"])

    add_queries_with_rectify_probes._sm95 = True
    add_queries_with_rectify_probes._original = original
    dp.add_queries = add_queries_with_rectify_probes


def uninstall_scorer_probes():
    import rwkv.data_processing as dp
    if getattr(dp.add_queries, "_sm95", False):
        dp.add_queries = dp.add_queries._original


# ---------------------------------------------------------------- order read-out
def curves4(params, days):
    """params [n, 4, 3N] (w | s_raw | d_raw per grade), days broadcastable to [n, ...] -> R of all four grades at each
    days entry: [n, ..., 4] float64 (the model's gru_forgetting_curve)."""
    p = params.astype(np.float64)
    n3 = p.shape[-1] // 3
    w, s, d = p[..., :n3], np.exp(np.clip(p[..., n3:2 * n3], -25, 25)), np.exp(np.clip(p[..., 2 * n3:], -25, 25))
    days = np.asarray(days, dtype=np.float64)
    extra = days.ndim - 1
    shape = (p.shape[0],) + (1,) * extra + (4, n3)
    w, s, d = w.reshape(shape), s.reshape(shape), d.reshape(shape)
    t = np.maximum(1.0, days * SECONDS_PER_DAY)[..., None, None]
    r = np.sum(w * np.exp(-d * np.log1p(t / (1e-7 + s))), axis=-1)
    return 1e-5 + (1 - 2e-5) * r


def grade_curves(params, days, rectify):
    """days [n, 4] (one time per grade) -> R_g(days_g) [n, 4], raw or rectified (pooled over the four grades at that t)."""
    r = curves4(params, days)                     # [n, 4 (time of grade i), 4 (grade)]
    if rectify:
        r = pava4_numpy(r)
    return r[:, np.arange(4), np.arange(4)]


def intervals(params, target, rectify, iters=60):
    n = params.shape[0]
    lo = np.full((n, 4), np.log(1.0 / SECONDS_PER_DAY))
    hi = np.full((n, 4), np.log(MAX_DAYS))
    at_hi = grade_curves(params, np.exp(hi), rectify) > target
    at_lo = grade_curves(params, np.exp(lo), rectify) <= target
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        above = grade_curves(params, np.exp(mid), rectify) > target
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    out = np.where(at_hi, MAX_DAYS, np.exp(hi))
    return np.where(at_lo, 1.0 / SECONDS_PER_DAY, out)


def fork_display(params, target, rectify, pava, iters=50):
    """The fork's unrounded displayed intervals (scripts/sm33_grade_order.fork_intervals) on raw / rectified curves,
    with (pava=True) or without its grade-order PAVA on the margins."""
    from scripts.sm33_grade_order import fork_grid
    grid = fork_grid()

    def margins(days):  # days [n, 4] -> margins [n, 4] (grade i's margin at days[:, i])
        r = curves4(params, days)
        if rectify:
            r = pava4_numpy(r)
        m = r - target
        if pava:
            m = pava4_numpy(m)
        return m[:, np.arange(4), np.arange(4)]

    rg = curves4(params, grid[None, :])           # [n, G, 4]
    if rectify:
        rg = pava4_numpy(rg)
    mg = rg - target
    if pava:
        mg = pava4_numpy(mg)
    hit = mg <= 0
    any_hit = hit.any(axis=1)
    first = hit.argmax(axis=1)
    hi = grid[first]
    lo = np.where(first > 0, grid[np.maximum(first - 1, 0)], 0.0)
    exact = margins(lo) <= 0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        above = margins(mid) > 0
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    out = np.where(exact, grid[first], hi)
    return np.where(any_hit, np.minimum(out, MAX_DAYS), MAX_DAYS)


def derive_user(z, rectify):
    params = z["params"].astype(np.float64)
    iv = np.stack([intervals(params, t, rectify) for t in TARGETS], axis=1)                         # [n, 3, 4]
    cv = np.stack([grade_curves(params, np.full((len(params), 4), h), rectify) for h in HORIZONS], axis=1)
    fk = fork_display(params, 0.90, rectify, True)
    fk0 = fork_display(params, 0.90, rectify, False)
    return {"intervals": iv, "curve_at": cv, "fork_pava_09": fk, "fork_nopava_09": fk0}


def cmd_derive(args):
    src, out = Path(args.measure), Path(args.out)
    (out / "users").mkdir(parents=True, exist_ok=True)
    files = sorted((src / "users").glob("*.npz"))
    t0, done = time.time(), 0
    for f in files:
        target = out / "users" / f.name
        if target.exists():
            continue
        z = np.load(f)
        d = derive_user(z, args.rectify)
        np.savez_compressed(target, review_th=z["review_th"], rating=z["rating"], first=z["first"], n_reviews=z["n_reviews"],
                            **d)
        done += 1
    (out / "derive-log.json").write_text(json.dumps({"measure": str(src), "rectify": bool(args.rectify), "users": len(files),
                                                      "derived_now": done, "seconds": round(time.time() - t0, 1)}, indent=1) + "\n")
    print(f"[sm95-derive] {src.name} -> {out} ({'rectified' if args.rectify else 'raw'}): {len(files)} users, "
          f"{done} derived now, {time.time() - t0:.0f} s", flush=True)
    return 0


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("derive")
    d.add_argument("--measure", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--rectify", action="store_true")
    args = p.parse_args()
    return {"derive": cmd_derive}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
