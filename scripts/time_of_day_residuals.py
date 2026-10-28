"""Does the time of day (or the weekday) carry recall information the model does not use? (CPU, read only)

The model inputs have no clock: `day_of_week` is (day_offset % 7) from the user's first review, and the sin/cos
features encode elapsed time between reviews of a card. The -id dataset has `review_time` (UTC epoch ms), and
`day_offset` changes at the user's local day rollover, so the UTC time of that rollover is recoverable per user
(ambiguous inside nightly gaps without reviews); assuming Anki's default 4 am rollover gives a local clock.

Predictions: the leak-free big model's per-review replay targets (artifacts/cache/kd-replay/...; rows = review_th - 1)
on a random sample of the 8,000 training users. Curve p = sigmoid(curve_logit); instant p = 1 - p_again.
For each grouping (local hour, real weekday, hours since the day's first review) it reports:
  - recall rate, mean prediction and calibration by bin;
  - global gain: one logit offset per bin, fit on half the users, LogLoss change on the other half (2-fold);
  - per-user gain: per-user offsets by bin (L2 prior), fit on each user's first 70% of reviews, scored on the
    last 30%, minus the gain of a per-user constant offset (the part that is about the clock).
95% intervals: user bootstrap.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "artifacts/cache/kd-replay/big-leakfree-1ep-final8k-neighbour"
DATA = ROOT / "data/anki-revlogs-10k/revlogs"
DAY = 86_400_000
Q = 900_000  # 15-minute candidate rollover steps


def rollover(t, day_offset):
    """UTC ms-of-day of the user's rollover: middle of the run of 15-min shifts that reproduce day_offset best."""
    agree = np.empty(96)
    for b in range(96):
        d = (t - b * Q) // DAY - day_offset
        vals, counts = np.unique(d, return_counts=True)
        agree[b] = counts.max() / len(d)
    ok = agree >= agree.max() - 0.002
    # the feasible shifts form one circular run (a gap without reviews); take its middle
    idx = np.flatnonzero(ok)
    if ok.all():
        return 0, 24.0, agree.max()
    start = next(i for i in idx if not ok[(i - 1) % 96])
    n = 0
    while ok[(start + n) % 96]:
        n += 1
    return ((start + n // 2) % 96) * Q, n * 0.25, agree.max()


def user_frame(u):
    t = pq.read_table(DATA / f"user_id={u}/data.parquet", columns=["day_offset", "rating", "review_time"])
    day_offset = t["day_offset"].to_numpy(); rating = t["rating"].to_numpy(); rt = t["review_time"].to_numpy()
    z = np.load(REPLAY / f"users/{u}.npz")["targets"].astype(np.float32)
    if len(z) != len(rt):
        return None
    b, width, agree = rollover(rt, day_offset)
    tz = (4 * 3_600_000 - b) if b <= 16 * 3_600_000 else (28 * 3_600_000 - b)   # local offset, assuming 4 am rollover
    local = rt + tz
    hour = ((local % DAY) // 3_600_000).astype(np.int8)
    weekday = ((local // DAY + 3) % 7).astype(np.int8)                         # 0 = Monday (1970-01-01 was a Thursday)
    kday = (rt - b) // DAY
    # hours since the first review of the same Anki day
    starts = np.r_[0, np.flatnonzero(kday[1:] != kday[:-1]) + 1]
    day_first = np.repeat(rt[starts], np.diff(np.r_[starts, len(rt)]))
    since = (rt - day_first) / 3_600_000
    since_bin = np.digitize(since, [0.5, 2, 4, 8, 12]).astype(np.int8)
    m = np.isfinite(z[:, 0])
    y = (rating[m] > 1).astype(np.int8)
    logit_c = np.clip(z[m, 0], -12, 12)
    p_inst = np.clip(1 - z[m, 1], 1e-5, 1 - 1e-5)
    logit_i = np.log(p_inst / (1 - p_inst))
    return {"y": y, "z": {"curve": logit_c, "instant": logit_i.astype(np.float32)},
            "bins": {"hour": hour[m], "weekday": weekday[m], "since": since_bin[m]},
            "width": width, "agree": agree, "tz_h": tz / 3_600_000}


def ll(y, z):
    return np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y)


def fit_offsets(y, z, b, nb, lam=0.0, iters=8):
    d = np.zeros(nb)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(z + d[b])))
        g = np.bincount(b, y - p, nb) - lam * d
        h = np.bincount(b, p * (1 - p), nb) + lam + 1e-9
        d += g / h
    return d


NB = {"hour": 24, "weekday": 7, "since": 6}
COARSE = {"hour": lambda h: (h // 4).astype(np.int8), "weekday": lambda w: w, "since": lambda s: s}
NBC = {"hour": 6, "weekday": 7, "since": 6}


def boot(num, den, rng, n=2000):
    idx = rng.integers(0, len(num), (n, len(num)))
    s = num[idx].sum(1) / den[idx].sum(1)
    return float(num.sum() / den.sum()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=600)
    ap.add_argument("--max-reviews", type=int, default=30000, help="per user, random subsample for the global part")
    ap.add_argument("--max-width", type=float, default=3.0, help="hour analysis: users whose rollover is known within N h")
    ap.add_argument("--lam", type=float, default=200.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rng = np.random.default_rng(2026)
    all_users = sorted(int(p.stem) for p in (REPLAY / "users").glob("*.npz"))
    users = sorted(rng.choice(all_users, a.users, replace=False).tolist())
    per = {}
    for i, u in enumerate(users):
        f = user_frame(u)
        if f is not None and len(f["y"]) >= 200:
            per[u] = f
        if (i + 1) % 100 == 0:
            print(f"[tod] loaded {i + 1}/{len(users)}", flush=True)
    U = sorted(per)
    widths = np.array([per[u]["width"] for u in U])
    report = {"users": len(U), "labelled_reviews": int(sum(len(per[u]["y"]) for u in U)),
              "rollover_width_h": {"median": float(np.median(widths)), "share_le_max": float(np.mean(widths <= a.max_width))},
              "groupings": {}}
    fold = {u: k % 2 for k, u in enumerate(rng.permutation(U))}
    for g in ("hour", "weekday", "since"):
        GU = [u for u in U if g != "hour" or per[u]["width"] <= a.max_width]
        res = {"users": len(GU)}
        for head in ("curve", "instant"):
            # bins table (all reviews of GU)
            nb = NB[g]
            cnt = np.zeros(nb); ys = np.zeros(nb); ps = np.zeros(nb)
            sub = {}
            for u in GU:
                f = per[u]; y = f["y"]; z = f["z"][head]; b = f["bins"][g]
                p = 1 / (1 + np.exp(-z))
                cnt += np.bincount(b, minlength=nb); ys += np.bincount(b, y, nb); ps += np.bincount(b, p, nb)
                k = np.arange(len(y)) if len(y) <= a.max_reviews else np.sort(rng.choice(len(y), a.max_reviews, replace=False))
                sub[u] = (y[k], z[k], b[k])
            # global 2-fold offsets
            off = {}
            for k in (0, 1):
                tr = [u for u in GU if fold[u] == k]
                off[k] = fit_offsets(np.concatenate([sub[u][0] for u in tr]), np.concatenate([sub[u][1] for u in tr]),
                                     np.concatenate([sub[u][2] for u in tr]).astype(np.int64), nb)
            gnum, gden, base_ll = [], [], []
            for u in GU:
                y, z, b = sub[u]
                d = off[1 - fold[u]]
                gnum.append((ll(y, z + d[b.astype(np.int64)]) - ll(y, z)).sum()); gden.append(ll(y, z).sum())
            gnum = np.array(gnum); gden = np.array(gden)
            gl = boot(gnum, gden, rng)
            # per-user: bins vs constant, time split 70/30
            nbc = NBC[g]
            pnum_b, pnum_c, pden = [], [], []
            for u in GU:
                f = per[u]; y = f["y"]; z = f["z"][head]; b = COARSE[g](f["bins"][g]).astype(np.int64)
                cut = int(len(y) * 0.7)
                db = fit_offsets(y[:cut], z[:cut], b[:cut], nbc, a.lam)
                dc = fit_offsets(y[:cut], z[:cut], np.zeros(cut, np.int64), 1, a.lam)
                base = ll(y[cut:], z[cut:])
                pnum_b.append((ll(y[cut:], z[cut:] + db[b[cut:]]) - base).sum())
                pnum_c.append((ll(y[cut:], z[cut:] + dc[0]) - base).sum())
                pden.append(base.sum())
            pnum_b = np.array(pnum_b); pnum_c = np.array(pnum_c); pden = np.array(pden)
            res[head] = {
                "bins": [{"bin": i, "share": float(cnt[i] / cnt.sum()), "recall": float(ys[i] / max(cnt[i], 1)),
                          "pred": float(ps[i] / max(cnt[i], 1)), "offset_fold0": float(off[0][i]), "offset_fold1": float(off[1][i])}
                         for i in range(nb)],
                "global_gain": gl,                                  # relative LogLoss change, negative = better
                "per_user_bins_gain": boot(pnum_b, pden, rng),
                "per_user_const_gain": boot(pnum_c, pden, rng),
                "per_user_clock_part": boot(pnum_b - pnum_c, pden, rng),
                "users_better_bins_vs_const": float(np.mean(pnum_b < pnum_c)),
            }
            print(f"[tod] {g} {head}: global {gl[0]:+.4%} [{gl[1]:+.4%}, {gl[2]:+.4%}]  per-user clock part "
                  f"{res[head]['per_user_clock_part'][0]:+.4%} [{res[head]['per_user_clock_part'][1]:+.4%}, {res[head]['per_user_clock_part'][2]:+.4%}]", flush=True)
        report["groupings"][g] = res
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("users", "labelled_reviews", "rollover_width_h")}))


if __name__ == "__main__":
    main()
