#!/usr/bin/env python
"""The show-time error sensitivity test (SM100): pre-registered rule and read-out (fixed 2026-11-09 before any GPU run; CPU).

Question (outside research, user 11-09 "let's try it"): the show-time query interval is rebuilt as the card's previous
answer -> this review's answer minus this review's recorded duration (e2s-query). When the recorded duration is too short
(undo + regrade, a timeout cap, a timer that does not tick) the rebuilt show time is too late and the interval a few
seconds to minutes too long; undo and timeouts go with failing, so the error could in principle carry the outcome into
the evaluation. The model has no seconds-level collection gap input, so it can only use the error through its sensitivity
to the query interval. We measure that sensitivity.

Model: the show-time instant-only small model (SM96), `sm96-inst-e31st-st-dec1ep32k/w8+s8`, checkpoint
wt/sm96/.../sm96-inst-e31st-st-dec-1ep32k-muon (sha256 pinned in launch-sm100.sh), scored as SM96 was (same scorer path,
w8+s8, --deleted-cards neighbour, --split-existing, cap 0.8), plus --dump (per-review instant predictions).
Inputs (one scorer pass each, 723 users of the 732 set; rows gpu/sm100-rows.jsonl, dumps gpu/dump/):
  e2s-query (shift 0, re-scored for the dumps), e2s-query+10s, e2s-query+60s, e2s-query+300s, normal.
  (scripts/sm31_inputs.py: every show-time interval + N s, with its cumulative / sin / cos / scaled columns and the curve
  query label, exactly as show_time_query_rows derives them; shift 0 == e2s-query bit for bit, sm100_check.py.)

PRIMARY (verdict), 722 users (the users the big model (1 leak-free epoch) normal rows have), instant head:
  shifted - unshifted (both from this run's rows), review-weighted, paired user bootstrap 2,000, seed 0, 95% CI (the show-time
  report's `compare`), unit points = % of the big model (1 leak-free epoch) normal-input instant LogLoss (0.275757).
  negligible      = for BOTH +10 s and +60 s the 95% CI lies inside (-0.05, +0.05) points
  not negligible  = otherwise: report the size and look closer (by interval bucket, timeout subset)
SECONDARY (reported, no verdict):
  +300 s the same way; mean |dR| of the instant predictions vs shift 0 (review-weighted over all benchmark reviews of the
  722 users), its 99th percentile and the share of reviews with |dR| > 0.01; |dR| and the LogLoss change by show-time
  interval bucket (< 10 min, 10 min - 1 day, >= 1 day; unshifted show-time interval from the revlog);
  reproduction: the shift-0 and normal rows vs SM96's own rows (per-user instant LogLoss, max abs difference);
  the model's own show-time cost (e2s-query - normal), for scale (SM96: +0.055 [+0.042, +0.072]).
TIMEOUT SUBSET (reported, no verdict), from the dumps + the revlogs: per user, cap = the user's max recorded duration; the
  user is "capped" if cap >= 10,000 ms and the cap value occurs >= 20 times and on >= 0.5% of the user's reviews; timeout
  reviews = the capped users' benchmark reviews with duration == cap. Reported: capped users, cap values, the timeout share
  of all benchmark reviews, the failure rate on timeout vs other reviews, and the instant LogLoss on timeout reviews under
  normal vs show-time (shift 0) inputs (paired user bootstrap over the capped users, weights = their timeout reviews), plus
  the same on the capped users' other reviews, and the +60 s shift's change on timeout reviews.
Curve head untrained (curve loss weight 0): never reported.

  python sm100_rule.py selftest  -> SM96's own rows through this code: e2s - e2s == 0 and SM96's show-time cost
                                    (+0.055 [+0.042, +0.072]) reproduced; exit 0 if equal
  python sm100_rule.py decide    -> sm100-decision.{json,md}; exit 2 while rows are missing
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
G = HERE / "gpu"
DUMP = G / "dump"
ROWS = G / "sm100-rows.jsonl"
SM96_ROWS = Path("C:/Users/jonat/workspace/wt/sm96/artifacts/small-model/sm96/gpu/sm96-rows.jsonl")   # read only
REPORT = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/small-model/showtime/showtime_report.py")
REPORT_SHA = "3cdfed03afd40734d4b1931c244e8c1e73453d281a40c3843e80cd310d70dd9d"   # main 9031ed51 (as sm96_rule.py)
REVLOGS = Path("C:/Users/jonat/workspace/anki-rwkv/data/anki-revlogs-10k-original-20261011/revlogs")
TAG = "sm96-inst-e31st-st-dec1ep32k/w8+s8"
L0 = TAG + "@e2s-query"
SHIFTS = {"+10s": TAG + "@e2s-query+10s", "+60s": TAG + "@e2s-query+60s", "+300s": TAG + "@e2s-query+300s"}
PRIMARY = ("+10s", "+60s")
LNORMAL = TAG
BOUND = 0.05
BUCKETS = (("< 10 min", 0, 600), ("10 min - 1 day", 600, 86400), (">= 1 day", 86400, float("inf")))
CAP_MIN_MS, CAP_MIN_COUNT, CAP_MIN_SHARE = 10_000, 20, 0.005
NAME = "the show-time instant-only small model (SM96)"


def report():
    got = hashlib.sha256(REPORT.read_bytes()).hexdigest()
    if got != REPORT_SHA:
        raise SystemExit(f"[sm100-rule] showtime_report.py changed (sha256 {got[:12]} != {REPORT_SHA[:12]}): refusing")
    spec = importlib.util.spec_from_file_location("showtime_report", REPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fp(x):
    return f"{x['points']:+.3f} [{x['points_ci'][0]:+.3f}, {x['points_ci'][1]:+.3f}]"


def inside(x):
    return -BOUND < x["points_ci"][0] and x["points_ci"][1] < BOUND


def selftest():
    r = report()
    users, big = r.pools()
    e = r.load([(SM96_ROWS, L0, None)], set(users))
    n = r.load([(SM96_ROWS, LNORMAL, None)], set(users))
    us = sorted(set(users) & set(e) & set(n))
    zero = r.compare(e, e, big, us)["instant"]
    cost = r.compare(e, n, big, us)["instant"]
    ok = len(us) == 722 and zero["points"] == 0 and zero["points_ci"] == [0.0, 0.0] and inside(zero) \
        and cost["points"] == 0.055 and cost["points_ci"] == [0.042, 0.072]
    print(f"[sm100-rule] selftest: {len(us)} users; SM96 e2s - e2s {fp(zero)} (inside +/-{BOUND}: {inside(zero)}); "
          f"SM96 show-time cost {fp(cost)} (sm96-decision +0.055 [+0.042, +0.072]) -> {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def dump_file(user, label):
    return DUMP / f"{user}_{label.replace('/', '_')}.npz"


def read_dump(user, label):
    f = dump_file(user, label)
    if not f.exists():
        return None
    z = np.load(f)
    return z["review_th"], z["y"].astype("float64"), z["p_instant"].astype("float64")


def ll_terms(y, p):
    eps = np.finfo("float64").eps
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def revlog(user):
    import pandas as pd
    df = pd.read_parquet(REVLOGS / f"user_id={user}", columns=["duration", "elapsed_seconds"])
    dur = df["duration"].to_numpy().astype("float64")
    es = df["elapsed_seconds"].to_numpy().astype("float64")
    show = np.floor(np.maximum(es - dur / 1000.0, 0.0))
    show[es == -1] = -1.0
    return dur, show


def cap_of(dur):
    m = float(dur.max()) if len(dur) else 0.0
    k = int((dur == m).sum())
    capped = m >= CAP_MIN_MS and k >= CAP_MIN_COUNT and k >= CAP_MIN_SHARE * len(dur)
    return (m if capped else None), k


def boot(a_sum, b_sum, w, ref_sum=None):
    """Paired user bootstrap (2,000, seed 0) of (sum a - sum b) / sum w (per-review LogLoss delta) and in % of b."""
    a_sum, b_sum, w = map(lambda x: np.asarray(x, float), (a_sum, b_sum, w))
    idx = np.random.default_rng(0).integers(0, len(w), size=(2000, len(w)))
    d = (a_sum - b_sum)[idx].sum(1) / w[idx].sum(1)
    pct = (a_sum - b_sum)[idx].sum(1) / b_sum[idx].sum(1) * 100
    return {"a": round(float(a_sum.sum() / w.sum()), 6), "b": round(float(b_sum.sum() / w.sum()), 6),
            "delta": round(float((a_sum - b_sum).sum() / w.sum()), 6),
            "delta_ci": [round(float(np.percentile(d, q)), 6) for q in (2.5, 97.5)],
            "pct_of_b": round(float((a_sum - b_sum).sum() / b_sum.sum() * 100), 3),
            "pct_of_b_ci": [round(float(np.percentile(pct, q)), 3) for q in (2.5, 97.5)],
            "users": int(len(w)), "reviews": int(w.sum())}


def dump_analysis(users, big_instant):
    """|dR| per shift (+ by bucket) and the timeout subset; per user, numpy only."""
    sh = {k: {"abs_sum": 0.0, "n": 0, "over_001": 0, "hist": np.zeros(2001, np.int64),
              "b_abs": np.zeros(3), "b_n": np.zeros(3, np.int64), "b_d": np.zeros(3), "b_ll0": np.zeros(3)} for k in SHIFTS}
    t = {"capped_users": 0, "caps": {}, "all_reviews": 0, "timeout_reviews": 0, "fail_timeout": 0, "fail_other": 0,
         "other_reviews_capped_users": 0}
    per_user = []   # capped users: (timeout n, LL normal sum, LL e2s sum, LL +60 sum, other n, other normal, other e2s)
    missing, mismatch = [], 0
    for u in users:
        d0 = read_dump(u, L0)
        if d0 is None:
            missing.append((u, L0)); continue
        th, y, p0 = d0
        dur, show = revlog(u)
        i = th.astype(np.int64) - 1
        sbin = np.digitize(show[i], [600, 86400])        # 0: < 10 min (incl. -1), 1: 10 min - 1 day, 2: >= 1 day
        l0 = ll_terms(y, p0)
        for k, lab in SHIFTS.items():
            ds = read_dump(u, lab)
            if ds is None:
                missing.append((u, lab)); continue
            if not np.array_equal(ds[0], th):
                mismatch += 1; continue
            a = np.abs(ds[2] - p0)
            s = sh[k]
            s["abs_sum"] += a.sum(); s["n"] += len(a); s["over_001"] += int((a > 0.01).sum())
            s["hist"] += np.bincount(np.minimum((a * 2000).astype(np.int64), 2000), minlength=2001)
            ls = ll_terms(y, ds[2])
            for b in range(3):
                m = sbin == b
                s["b_abs"][b] += a[m].sum(); s["b_n"][b] += int(m.sum()); s["b_d"][b] += (ls - l0)[m].sum(); s["b_ll0"][b] += l0[m].sum()
        dn = read_dump(u, LNORMAL)
        if dn is None:
            missing.append((u, LNORMAL)); continue
        t["all_reviews"] += len(th)
        cap, _ = cap_of(dur)
        if cap is None or not np.array_equal(dn[0], th):
            continue
        t["capped_users"] += 1
        t["caps"][str(int(cap))] = t["caps"].get(str(int(cap)), 0) + 1
        to = dur[i] == cap
        t["timeout_reviews"] += int(to.sum()); t["other_reviews_capped_users"] += int((~to).sum())
        t["fail_timeout"] += int((y[to] == 0).sum()); t["fail_other"] += int((y[~to] == 0).sum())
        ln = ll_terms(y, dn[2])
        d60 = read_dump(u, SHIFTS["+60s"])
        l60 = ll_terms(y, d60[2]) if d60 is not None and np.array_equal(d60[0], th) else l0
        per_user.append((int(to.sum()), ln[to].sum(), l0[to].sum(), l60[to].sum(), int((~to).sum()), ln[~to].sum(), l0[~to].sum()))
    out = {"missing": len(missing), "missing_examples": missing[:5], "review_th_mismatch": mismatch, "shifts": {}}
    for k, s in sh.items():
        if not s["n"]:
            continue
        c = np.cumsum(s["hist"])
        p99 = float(np.searchsorted(c, 0.99 * c[-1]) / 2000)
        out["shifts"][k] = {"reviews": s["n"], "mean_abs_dR": round(s["abs_sum"] / s["n"], 6), "p99_abs_dR_upper": round(p99 + 0.0005, 4),
                            "share_abs_dR_over_0.01": round(s["over_001"] / s["n"], 5),
                            "by_bucket": {BUCKETS[b][0]: {"share": round(int(s["b_n"][b]) / s["n"], 4),
                                                          "mean_abs_dR": round(float(s["b_abs"][b] / max(s["b_n"][b], 1)), 6),
                                                          "dLogLoss_per_review": round(float(s["b_d"][b] / max(s["b_n"][b], 1)), 6),
                                                          "pct_of_bucket_logloss": round(float(s["b_d"][b] / max(s["b_ll0"][b], 1e-12) * 100), 3),
                                                          "points_contribution": round(float(s["b_d"][b] / s["n"] / big_instant * 100), 3)}
                                           for b in range(3)}}
    if per_user:
        a = np.array(per_user, float)
        has = a[:, 0] > 0
        t["timeout_share"] = round(t["timeout_reviews"] / max(t["all_reviews"], 1), 5)
        t["fail_rate_timeout"] = round(t["fail_timeout"] / max(t["timeout_reviews"], 1), 4)
        t["fail_rate_other_capped_users"] = round(t["fail_other"] / max(t["other_reviews_capped_users"], 1), 4)
        t["timeout_e2s_minus_normal"] = boot(a[has, 2], a[has, 1], a[has, 0])
        t["timeout_plus60_minus_e2s"] = boot(a[has, 3], a[has, 2], a[has, 0])
        t["other_e2s_minus_normal"] = boot(a[:, 6], a[:, 5], a[:, 4])
        for key in ("timeout_e2s_minus_normal", "timeout_plus60_minus_e2s", "other_e2s_minus_normal"):
            x = t[key]
            x["points_overall_denominator"] = round(x["delta"] / big_instant * 100, 3)
    out["timeout"] = t
    return out


def decide():
    r = report()
    users, big = r.pools()
    have = set(users)
    base = r.load([(ROWS, L0, None)], have)
    sets = {k: r.load([(ROWS, lab, None)], have) for k, lab in SHIFTS.items()}
    norm = r.load([(ROWS, LNORMAL, None)], have)
    need = [("shift 0", base)] + [(k, sets[k]) for k in PRIMARY]
    for k, x in need:
        if not have <= set(x):
            print(f"[sm100-rule] pending: {k} rows on {len(have & set(x))} of {len(users)} rule users")
            return 2
    big_i = r.big_denominator(big, users, "instant")
    out = {"rule": f"shifted - unshifted, instant, {NAME}, 722 users: negligible if the 95% CI of BOTH +10 s and +60 s lies "
                   f"inside (-{BOUND}, +{BOUND}) points", "label": TAG, "users": len(users), "big_instant": round(big_i, 6), "pairs": {}}
    for k, x in sets.items():
        us = sorted(have & set(x) & set(base))
        out["pairs"][k] = r.compare(x, base, big, us)["instant"] if us else None
    prim = [out["pairs"][k] for k in PRIMARY]
    out["verdict"] = "negligible" if all(p is not None and inside(p) for p in prim) else "not negligible"
    if norm and have <= set(norm):
        out["own_show_time_cost"] = r.compare(base, norm, big, users)["instant"]
    # reproduction vs SM96's rows (same checkpoint, same scorer path; dumps on)
    rep = {}
    for kind, lab, mine in (("e2s-query", L0, base), ("normal", LNORMAL, norm)):
        old = r.load([(SM96_ROWS, lab, None)], have)
        us = sorted(set(old) & set(mine))
        rep[kind] = {"users": len(us), "max_abs_instant_logloss_diff": max((abs(old[u]["instant"][0] - mine[u]["instant"][0]) for u in us), default=None)}
    out["reproduction_vs_sm96_rows"] = rep
    out["dumps"] = dump_analysis(users, big_i)
    (HERE / "sm100-decision.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = ["# The show-time error sensitivity test (SM100): decision", "",
          f"Verdict: **{out['verdict']}**. {NAME}, instant head, {len(users)} users of the 732 set, points = % of the big model "
          f"(1 leak-free epoch) normal-input instant LogLoss {out['big_instant']}; negligible needs the 95% CI of both +10 s "
          f"and +60 s inside (-{BOUND}, +{BOUND}).", "",
          "| show-time interval shift | instant LogLoss shifted | unshifted | points [95% CI] | users better / worse | mean abs dR | share abs dR > 0.01 |",
          "|---|---|---|---|---|---|---|"]
    for k, p in out["pairs"].items():
        if p:
            d = out["dumps"]["shifts"].get(k, {})
            md.append(f"| {k}{' (primary)' if k in PRIMARY else ''} | {p['new']} | {p['ref']} | {fp(p)} | {p['users_better']} / {p['users_worse']} | "
                      f"{d.get('mean_abs_dR', 'n/a')} | {d.get('share_abs_dR_over_0.01', 'n/a')} |")
    if "own_show_time_cost" in out:
        md += ["", f"For scale, the model's own show-time cost (show-time - normal inputs): {fp(out['own_show_time_cost'])} points."]
    md += ["", "## By show-time interval bucket (dumps)", "", "| shift | bucket | share of reviews | mean abs dR | LogLoss change, % of bucket | points contribution |", "|---|---|---|---|---|---|"]
    for k, d in out["dumps"]["shifts"].items():
        for b, v in d["by_bucket"].items():
            md.append(f"| {k} | {b} | {v['share']} | {v['mean_abs_dR']} | {v['pct_of_bucket_logloss']:+.3f} | {v['points_contribution']:+.3f} |")
    t = out["dumps"].get("timeout", {})
    if "timeout_e2s_minus_normal" in t:
        a, b, c = t["timeout_e2s_minus_normal"], t["timeout_plus60_minus_e2s"], t["other_e2s_minus_normal"]
        md += ["", "## Timeout subset (dumps + revlogs)", "",
               f"{t['capped_users']} capped users (cap values: {t['caps']}); timeout reviews {t['timeout_reviews']:,} = "
               f"{t['timeout_share'] * 100:.2f}% of {t['all_reviews']:,} benchmark reviews; failure rate {t['fail_rate_timeout']} on timeout "
               f"reviews vs {t['fail_rate_other_capped_users']} on the capped users' other reviews.", "",
               "| reviews | inputs a - b | a | b | LogLoss delta [95% CI] | % of b [95% CI] |", "|---|---|---|---|---|---|"]
        for name, lab, x in (("timeout", "show-time - normal", a), ("timeout", "+60 s - show-time", b), ("capped users' other", "show-time - normal", c)):
            md.append(f"| {name} | {lab} | {x['a']} | {x['b']} | {x['delta']:+.6f} [{x['delta_ci'][0]:+.6f}, {x['delta_ci'][1]:+.6f}] | "
                      f"{x['pct_of_b']:+.3f} [{x['pct_of_b_ci'][0]:+.3f}, {x['pct_of_b_ci'][1]:+.3f}] |")
    md += ["", f"Reproduction vs the rows of {NAME}: {json.dumps(rep)}; dumps missing {out['dumps']['missing']}, "
               f"review_th mismatches {out['dumps']['review_th_mismatch']}."]
    (HERE / "sm100-decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    pr = "; ".join(f"{k} {fp(out['pairs'][k])}" for k in out["pairs"] if out["pairs"][k])
    print(f"[sm100-rule] verdict {out['verdict']}: {pr} points ({len(users)} users); mean |dR| "
          + ", ".join(f"{k} {v['mean_abs_dR']}" for k, v in out["dumps"]["shifts"].items()))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "decide"
    raise SystemExit(selftest() if cmd == "selftest" else decide())
