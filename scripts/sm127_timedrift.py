#!/usr/bin/env python
"""Time-drift diagnostic for fluency vs desired retention in the simulator with answer-time feedback (SM127).
Pre-registered in artifacts/small-model/sm127/design.md section 12.2 (before this ran).

Question: is the higher seconds per review of the feedback lanes (desired retention 0.90: 19.3 s vs 15.7 s per later-day
review) (a) a self-feeding loop of sampled answer times, or (b) a real effect of the policy reviewing different cards?

The pilot's engine (scripts/sm127_fluency.py, Engine127) on days 0-149 of the same 365-day set-up (same budget, same
draws: the lanes also in the pilot reproduce its 15-day buckets), 8 lanes, per-review log kept in memory; only
aggregates are written (per-user sums, no review rows):
  real schedule: tf (real grades and times), fr-shared1#0 (sampled, fed back, daily re-anchoring), fr#0 (no
  re-anchoring), fr-shared1-nofb#0 (re-anchored, the user's median time of the grade in the trunk's duration input);
  twins: ihr-0.90#0 / ihr-0.90-nofb#0, ihr-0.98#0 / ihr-0.98-nofb#0.

  run     --user U   one user -> artifacts/small-model/sm127/timedrift/rows/U.json (resumable: exists = done)
  report  rows -> timedrift-tables.md / timedrift-result.json
"""
from argparse import ArgumentParser
import datetime as dt
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import sm127_fluency as F   # noqa: E402  (patches sm41_engine.vstep, as in the pilot)

E = F.E
TD = F.HERE / "timedrift"
F.STOP = TD / "STOP"                      # quiet_wait reads this module global
LANES = ["tf", "fr-shared1#0", "fr#0", "fr-shared1-nofb#0", "ihr-0.90#0", "ihr-0.90-nofb#0", "ihr-0.98#0",
         "ihr-0.98-nofb#0"]
TWINS = [("ihr-0.90#0", "ihr-0.90-nofb#0"), ("ihr-0.98#0", "ihr-0.98-nofb#0"), ("fr-shared1#0", "tf"),
         ("fr-shared1-nofb#0", "tf"), ("fr#0", "tf"), ("fr-shared1#0", "fr-shared1-nofb#0")]
DAYS = 150
MONTH = 30
PHASE = 75                                 # persistence: days 0-74 / 75-149 (the pair's second review)
ED_EDGES = [2, 4, 8, 16, 32, 64, 128]      # interval buckets of the ed x grade table: 1, 2-3, 4-7, ..., >= 128
ED4 = [4, 16, 64]                          # decomposition cells: <= 3, 4-15, 16-63, >= 64 days
COL = {c: k for k, c in enumerate(E.LOG_COLS)}


class EngineD(F.Engine127):
    NOFB_REPLAY = True


def _later(L, b):
    m = (L[:, COL["lane"]] == b) & (L[:, COL["kind"]] == 2)
    X = L[m]
    return (X[:, COL["day"]].astype(np.int64), X[:, COL["card"]].astype(np.int64), X[:, COL["ed"]].astype(np.int64),
            X[:, COL["rating"]].astype(np.int64), X[:, COL["ms"]])


def lane_aggs(L, b, C):
    day, card, ed, g, ms = _later(L, b)
    s = ms / 1000.0
    ls = np.log(np.maximum(ms, 1.0) / 1000.0)
    nm = (DAYS + MONTH - 1) // MONTH
    mo = np.minimum(day // MONTH, nm - 1)
    mon = np.zeros((nm, 7))
    for k, v in enumerate((np.ones_like(s), s, ls, (ms >= 60000).astype(float), (g > 1).astype(float),
                           (ms <= 0).astype(float), (g == 1).astype(float) * s)):
        np.add.at(mon[:, k], mo, v)
    eb = np.searchsorted(ED_EDGES, ed, side="right")
    cells = np.zeros((len(ED_EDGES) + 1, 4, 3))
    for k, v in enumerate((np.ones_like(s), s, ls)):
        np.add.at(cells[:, :, k], (eb, g - 1), v)
    # persistence: consecutive later-day reviews of the same card in this lane (the log is in time order)
    o = np.argsort(card, kind="stable")
    c2, d2, l2, g2, t2 = card[o], day[o], ls[o], g[o], ms[o] > 0
    same = (c2[1:] == c2[:-1]) & t2[1:] & t2[:-1]
    x, y = l2[:-1][same], l2[1:][same]
    ph = (d2[1:][same] >= PHASE).astype(np.int64)
    bp = ((g2[:-1] > 1) & (g2[1:] > 1))[same]
    pers = np.zeros((2, 2, 6))        # phase x (all pairs, both passed) x (n, sx, sy, sxx, syy, sxy)
    for j, msk in ((0, np.ones_like(bp)), (1, bp)):
        for k, v in enumerate((np.ones_like(x), x, y, x * x, y * y, x * y)):
            np.add.at(pers[:, j, k], ph[msk], v[msk])
    n_c = np.bincount(card, minlength=C).astype(np.float64)
    s_c = np.bincount(card, weights=s, minlength=C)
    unit = card * 8 + np.searchsorted(ED4, ed, side="right") * 2 + (g > 1)
    n_u = np.bincount(unit, minlength=C * 8).astype(np.float64)
    s_u = np.bincount(unit, weights=s, minlength=C * 8)
    return {"month": mon, "cells": cells, "pers": pers}, (n_c, s_c, n_u, s_u)


def decomp(fa, na):
    """Sums for the split of the feedback / reference ratio of seconds per later-day review: per unit (card, or card x
    interval bucket x pass/fail) n and seconds of each lane. Returns [n_f, s_f, n_n, s_n (all units), n_f, s_f, n_n, s_n
    (units in both), sum_O n_f * m_n, sum_O n_n * m_f]."""
    nf, sf = fa
    nn, sn = na
    O = (nf > 0) & (nn > 0)
    mf = np.where(O, sf / np.maximum(nf, 1), 0.0)
    mn = np.where(O, sn / np.maximum(nn, 1), 0.0)
    return [nf.sum(), sf.sum(), nn.sum(), sn.sum(), nf[O].sum(), sf[O].sum(), nn[O].sum(), sn[O].sum(),
            float((nf[O] * mn[O]).sum()), float((nn[O] * mf[O]).sum())]


def run(args):
    F.S.low_priority()
    out = TD / "rows" / f"{args.user}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"[sm127-td] user {args.user}: done already", flush=True)
        return 0
    tag = f"user {args.user}"
    try:
        F.quiet_wait(tag, margin_min=args.margin)
        E.wait_memory(tag, args.min_free_gb)
        model = F.Model127(threads=1)
        D = int(F.user_entry(args.user)["split_day"])
        t0 = time.perf_counter()
        # the 365-day engine (budget, replay draws and keys as in the pilot), run for the first args.days days
        warm = None
        rev = F.S.load_user(args.user)[0]
        warm = E.Warm(model, rev, args.user)
        end = int(np.searchsorted(rev["day_offset"], D, side="left"))
        warm.advance(end, E.FSRS7(E.FSRS7_DEFAULT))
        stop = int(np.searchsorted(rev["day_offset"], D + F.HORIZON, side="left"))
        budget = float(rev["duration"][end:stop].sum()) / 1000.0 / F.HORIZON

        def pause(d):
            F.quiet_wait(f"{tag} day {d}", margin_min=0)
        eng = EngineD(model, warm, D, F.HORIZON, LANES, budget, F.medians_later(rev, end), pause=pause)
        eng.log_on = True
        tw = time.perf_counter() - t0
        print(f"[sm127-td] {tag} split {D}: warm-up {warm.i} reviews {tw:.0f} s; {eng.C} cards, {eng.B} lanes",
              flush=True)
        t1 = time.perf_counter()
        for d in range(args.days):
            eng.day(d)
        wall = time.perf_counter() - t1
    except F.Stopped as ex:
        print(f"[sm127-td] {ex}: exiting without output", flush=True)
        return 3
    L = np.asarray(eng.logs, np.float64).reshape(-1, len(E.LOG_COLS))
    C = eng.C
    lanes, per = {}, {}
    for b, nm in enumerate(LANES):
        a, per[nm] = lane_aggs(L, b, C)
        a = {k: v.round(6).tolist() for k, v in a.items()}
        nb = args.days // E.BUCKET
        a["time_later"] = eng.acc["time_later"][b, :nb].round(6).tolist()
        a["load"] = eng.acc["load"][b, :nb].round(6).tolist()
        lanes[nm] = a
    dec = {}
    for f_, n_ in TWINS:
        dec[f"{f_} vs {n_}"] = {"card": decomp(per[f_][:2], per[n_][:2]), "cell": decomp(per[f_][2:], per[n_][2:])}
    # identity with the pilot row (complete 15-day buckets)
    ident = None
    pr = F.HERE / "rows" / f"{args.user}.json"
    if pr.exists():
        P = json.loads(pr.read_text())
        nb = args.days // E.BUCKET
        ident = {}
        for nm in LANES:
            if nm in P["lanes"]:
                k = P["lanes"].index(nm)
                mx = 0.0
                for key in ("time_later", "load"):
                    a_ = np.asarray(lanes[nm][key]); p_ = np.asarray(P["acc"][key])[k, :nb]
                    mx = max(mx, float(np.max(np.abs(a_ - p_) / np.maximum(np.abs(p_), 1.0))))
                ident[nm] = mx
    res = {"user": args.user, "split_day": D, "days": args.days, "lanes": LANES, "cards": C, "aggs": lanes,
           "decomp": dec, "identity_max_rel": ident, "lane_reviews": int(eng.lane_reviews),
           "warmup_seconds": round(tw, 1), "wall_seconds": round(wall, 1),
           "finished": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"}
    F.atomic_json(out, res)
    print(f"[sm127-td] {tag}: {eng.lane_reviews} lane reviews in {wall:.0f} s; identity {ident}", flush=True)
    return 0


# ------------------------------------------------------------------ report
def _pool_r(p):
    """Pooled within-user correlation from per-user sums [users, 6]."""
    n, sx, sy, sxx, syy, sxy = (p[:, k] for k in range(6))
    nn = np.maximum(n, 1)
    cov = (sxy - sx * sy / nn).sum(); vx = (sxx - sx * sx / nn).sum(); vy = (syy - sy * sy / nn).sum()
    return float(cov / math.sqrt(max(vx * vy, 1e-12)))


def _shares(d):
    """d: summed decomp vector. Returns the total ratio, within-unit ratios (both weightings), within share."""
    nf, sf, nn, sn, nfO, sfO, nnO, snO, xf, xn = d
    R = (sf / nf) / (sn / nn)
    mf, mn = sfO / nfO, snO / nnO
    Wf = mf / (xf / nfO)                 # feedback's unit weights, per-unit means swapped
    Wn = (xn / nnO) / mn                 # the reference's unit weights
    lr = math.log(R)
    share = 0.5 * (math.log(Wf) + math.log(Wn)) / lr if abs(lr) > 1e-9 else float("nan")
    return {"ratio": R, "within_fbw": Wf, "within_refw": Wn, "within_share": share,
            "overlap_review_share_f": nfO / nf, "overlap_review_share_ref": nnO / nn}


def report(args):
    rows = [json.loads(p.read_text()) for p in sorted((TD / "rows").glob("*.json"))]
    U = len(rows)
    idx = F.boot_idx(U)
    L = [f"# Time-drift diagnostic for fluency vs desired retention (SM127): {U} users, days 0-{rows[0]['days'] - 1}",
         "", "Pre-registered in design.md 12.2. Later-day reviews only. Answer time = the whole time on the card "
         "(answered within S s means reading the back included). Ratios are ratios of totals over users; CIs paired "
         f"user bootstrap ({F.BOOT:,} resamples, seed {F.SEED}).", ""]
    ident = [max(r["identity_max_rel"].values()) for r in rows if r.get("identity_max_rel")]
    L.append(f"Identity with the pilot rows (lanes also in the pilot, 15-day buckets of time and load): max relative "
             f"difference {max(ident) if ident else float('nan'):.2e} over {len(ident)} users "
             f"({'OK' if ident and max(ident) <= 1e-4 else 'FAILED: do not read'}).")
    out = {"users": [r["user"] for r in rows], "identity_max_rel": max(ident) if ident else None}
    names = {"tf": "real schedule, real times", "fr-shared1#0": "real schedule, sampled times fed back (re-anchored)",
             "fr#0": "real schedule, sampled times fed back, no re-anchoring",
             "fr-shared1-nofb#0": "real schedule, sampled times, median time in the trunk (re-anchored)",
             "ihr-0.90#0": "desired retention 0.90, feedback", "ihr-0.90-nofb#0": "desired retention 0.90, no feedback",
             "ihr-0.98#0": "desired retention 0.98, feedback", "ihr-0.98-nofb#0": "desired retention 0.98, no feedback"}
    M = {nm: np.array([r["aggs"][nm]["month"] for r in rows]) for nm in LANES}      # [U, months, 7]
    TL = {nm: np.array([r["aggs"][nm]["time_later"] for r in rows]) for nm in LANES}  # [U, buckets, 4]
    nmo = M["tf"].shape[1]
    L += ["", "## Seconds per later-day review by 30-day month, and the model's expected seconds (bin midpoints)", "",
          "| lane | " + " | ".join(f"month {k + 1}" for k in range(nmo)) + " | all | model expected | share >= 60 s |",
          "|" + "---|" * (nmo + 4)]
    out["months"] = {}
    for nm in LANES:
        m = M[nm].sum(0)
        tl = TL[nm].sum((0, 1))
        sp = [m[k, 1] / max(m[k, 0], 1) for k in range(nmo)]
        out["months"][nm] = {"sec": sp, "all": m[:, 1].sum() / m[:, 0].sum(), "expected": tl[3] / tl[0],
                             "cap_share": m[:, 3].sum() / m[:, 0].sum()}
        L.append(f"| {names[nm]} | " + " | ".join(f"{v:.1f}" for v in sp) +
                 f" | {out['months'][nm]['all']:.2f} | {tl[3] / tl[0]:.2f} | {out['months'][nm]['cap_share']:.3f} |")

    def ratio_ci(a, b, sel=slice(None)):
        """seconds per review of lane a over lane b: ratio of totals over users, paired bootstrap."""
        sa, na = M[a][:, sel, 1].sum(1), M[a][:, sel, 0].sum(1)
        sb, nb = M[b][:, sel, 1].sum(1), M[b][:, sel, 0].sum(1)
        v = (sa.sum() / na.sum()) / (sb.sum() / nb.sum())
        bs = (sa[idx].sum(1) / na[idx].sum(1)) / (sb[idx].sum(1) / nb[idx].sum(1))
        return v, [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    L += ["", "## Ratios of seconds per later-day review (months 1, 5 and all)", "",
          "| comparison | month 1 | month 5 | all [CI] |", "|---|---|---|---|"]
    out["ratios"] = {}
    for a, b in [("fr-shared1#0", "tf"), ("fr-shared1-nofb#0", "tf"), ("fr#0", "tf"), ("fr#0", "fr-shared1#0"),
                 ("ihr-0.90#0", "ihr-0.90-nofb#0"), ("ihr-0.98#0", "ihr-0.98-nofb#0")]:
        r1, _ = ratio_ci(a, b, slice(0, 1)); r5, _ = ratio_ci(a, b, slice(nmo - 1, nmo)); ra, ci = ratio_ci(a, b)
        out["ratios"][f"{a} vs {b}"] = {"month1": r1, "month_last": r5, "all": ra, "ci": ci}
        L.append(f"| {names[a]} vs {names[b]} | {r1:.3f} | {r5:.3f} | {ra:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] |")
    # persistence
    L += ["", "## Persistence: within-user correlation of log answer time, consecutive later-day reviews of a card", "",
          "| lane | days 0-74 | days 75-149 | both passed, 0-74 | both passed, 75-149 | pairs |", "|---|---|---|---|---|---|"]
    out["persistence"] = {}
    PR = {nm: np.array([r["aggs"][nm]["pers"] for r in rows]) for nm in LANES}    # [U, phase, kind, 6]
    for nm in LANES:
        p = PR[nm]
        v = [_pool_r(p[:, ph, j]) for j in (0, 1) for ph in (0, 1)]
        out["persistence"][nm] = {"all": v[:2], "both_passed": v[2:], "pairs": int(p[:, :, 0, 0].sum())}
        L.append(f"| {names[nm]} | " + " | ".join(f"{x:.3f}" for x in v) + f" | {int(p[:, :, 0, 0].sum()):,} |")

    def pers_all(nm, ids=None):
        p = PR[nm] if ids is None else PR[nm][ids]
        return _pool_r(p[:, :, 0].sum(1))
    for a, b in [("fr-shared1#0", "tf"), ("ihr-0.90#0", "ihr-0.90-nofb#0"), ("fr-shared1-nofb#0", "tf")]:
        v = pers_all(a) - pers_all(b)
        bs = np.array([pers_all(a, i) - pers_all(b, i) for i in idx[:2000]])
        out["persistence"][f"{a} minus {b}"] = {"diff": v, "ci": [float(np.percentile(bs, 2.5)),
                                                                   float(np.percentile(bs, 97.5))]}
        L.append(f"\nPersistence {names[a]} minus {names[b]} (whole run): {v:+.3f} "
                 f"[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] (2,000 resamples).")
    # decomposition
    L += ["", "## Twin decomposition of the seconds-per-review ratio", "",
          "Within = the same unit reviewed in both lanes, slower or faster; unit = card, or card x interval bucket "
          "(<= 3, 4-15, 16-63, >= 64 days) x pass/fail (the cell, the pre-registered measure). Within share = mean of "
          "the two weightings' log within-ratio / log total ratio; the rest is selection (which cards, intervals and "
          "grades are reviewed, including units reviewed in one lane only).", "",
          "| comparison | ratio | within (card) | within share (card) | within (cell, fb / ref weights) | "
          "within share (cell) [CI] | reviews in shared cells (fb / ref) |", "|---|---|---|---|---|---|---|"]
    out["decomp"] = {}
    for f_, n_ in TWINS:
        key = f"{f_} vs {n_}"
        Dc = np.array([r["decomp"][key]["card"] for r in rows]); Dl = np.array([r["decomp"][key]["cell"] for r in rows])
        sc, sl = _shares(Dc.sum(0)), _shares(Dl.sum(0))
        bs = np.array([_shares(Dl[i].sum(0))["within_share"] for i in idx[:2000]])
        ci = [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
        out["decomp"][key] = {"card": sc, "cell": sl, "cell_share_ci": ci}
        L.append(f"| {names[f_]} vs {names[n_]} | {sc['ratio']:.3f} | {0.5 * (sc['within_fbw'] + sc['within_refw']):.3f} | "
                 f"{sc['within_share']:.2f} | {sl['within_fbw']:.3f} / {sl['within_refw']:.3f} | "
                 f"{sl['within_share']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] | {sl['overlap_review_share_f']:.2f} / "
                 f"{sl['overlap_review_share_ref']:.2f} |")
    # interval x grade table for the 0.90 twins
    CE = {nm: np.array([r["aggs"][nm]["cells"] for r in rows]).sum(0) for nm in LANES}   # [8, 4, 3]
    lab = ["1", "2-3", "4-7", "8-15", "16-31", "32-63", "64-127", ">=128"]
    for f_, n_ in [("ihr-0.90#0", "ihr-0.90-nofb#0"), ("fr-shared1#0", "tf")]:
        a, b = CE[f_], CE[n_]
        L += ["", f"## {names[f_]} vs {names[n_]}: by interval (days) and grade", "",
              "| interval | share of reviews (fb / ref) | Again s (fb / ref) | Good s (fb / ref) | all s (fb / ref) |",
              "|---|---|---|---|---|"]
        for k in range(len(lab)):
            na_, nb_ = a[k, :, 0].sum(), b[k, :, 0].sum()
            def q(x, g):
                return x[k, g, 1] / x[k, g, 0] if x[k, g, 0] else float("nan")
            L.append(f"| {lab[k]} | {na_ / a[:, :, 0].sum():.3f} / {nb_ / b[:, :, 0].sum():.3f} | {q(a, 0):.1f} / "
                     f"{q(b, 0):.1f} | {q(a, 2):.1f} / {q(b, 2):.1f} | {a[k, :, 1].sum() / max(na_, 1):.1f} / "
                     f"{b[k, :, 1].sum() / max(nb_, 1):.1f} |")
        # standardised: feedback's cell means at the reference's cell mix
        w = b[:, :, 0]; mfa = np.where(a[:, :, 0] > 0, a[:, :, 1] / np.maximum(a[:, :, 0], 1), np.nan)
        ok = np.isfinite(mfa) & (w > 0)
        std = (mfa[ok] * w[ok]).sum() / w[ok].sum() / (b[:, :, 1][ok].sum() / w[ok].sum())
        raw = (a[:, :, 1].sum() / a[:, :, 0].sum()) / (b[:, :, 1].sum() / b[:, :, 0].sum())
        out.setdefault("standardised", {})[f"{f_} vs {n_}"] = {"raw": raw, "at_ref_mix": std}
        L.append(f"\nRatio raw {raw:.3f}; at the reference's interval x grade mix {std:.3f}.")
    # pre-registered reading
    sh = out["decomp"]["ihr-0.90#0 vs ihr-0.90-nofb#0"]["cell"]["within_share"]
    px = out["persistence"]["fr-shared1#0 minus tf"]["diff"]
    pt = out["persistence"]["ihr-0.90#0 minus ihr-0.90-nofb#0"]["diff"]
    if sh >= 0.5 and (px >= 0.05 or pt >= 0.10):
        reading = "(a) drift: a self-feeding loop of sampled answer times"
    elif (1 - sh) >= 0.5 and px < 0.05:
        reading = "(b) a real effect of the policy reviewing different cards"
    elif sh >= 0.5 and px < 0.05:
        reading = "within-card but realistic: the same cards are slower with feedback, fed-back times persist no more than real ones"
    else:
        reading = "mixed"
    rn = out["ratios"]["fr#0 vs fr-shared1#0"]["month_last"]
    out["reading"] = {"within_share_090": sh, "persistence_excess_real_schedule": px,
                      "persistence_excess_twins_090": pt, "verdict": reading,
                      "no_reanchoring_last_month_ratio": rn,
                      "user_level_drift_flag": bool(rn >= 1.05)}
    L += ["", f"**Reading (design.md 12.2): {reading}.** Within-cell share of the desired retention 0.90 gap "
              f"{sh:.2f}; persistence excess of fed-back sampled times over real on the real schedule {px:+.3f}; "
              f"desired retention 0.90 feedback minus no feedback {pt:+.3f}. No re-anchoring vs re-anchoring, last "
              f"month: {rn:.3f} ({'user-level drift flag' if rn >= 1.05 else 'no user-level drift flag'})."]
    lr = sum(r["lane_reviews"] for r in rows); ws = sum(r["wall_seconds"] + r["warmup_seconds"] for r in rows)
    out["cost"] = {"lane_reviews": lr, "process_hours": round(ws / 3600, 2)}
    L += ["", f"Cost: {lr:,} lane reviews, {ws / 3600:.2f} CPU process-hours."]
    (TD / "timedrift-tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    F.atomic_json(TD / "timedrift-result.json", out)
    print("\n".join(L))
    return 0


def main():
    ap = ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--user", type=int, required=True)
    r.add_argument("--days", type=int, default=DAYS); r.add_argument("--margin", type=float, default=30.0)
    r.add_argument("--min-free-gb", type=float, default=6.0)
    sub.add_parser("report")
    a = ap.parse_args()
    sys.exit({"run": run, "report": report}[a.cmd](a) or 0)


if __name__ == "__main__":
    main()
