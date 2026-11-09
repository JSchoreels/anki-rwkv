#!/usr/bin/env python
"""SM100b (the show-time error sensitivity test (SM100), follow-up): pre-registered rule (fixed 2026-11-09 before the GPU ticket; CPU).

Question: does the decision pair "the show-time instant-only small model (SM96) minus the show-time retrain of the release
(SM92), instant head, show-time inputs (e2s-query), 722 users" change when the timeout reviews are excluded?

Timeout reviews = SM100's exact definition (sm100_rule.cap_of / revlog, imported): per user cap = max recorded duration; the
user is "capped" if cap >= 10,000 ms, the cap value occurs >= 20 times and on >= 0.5% of the user's reviews; timeout reviews =
the capped users' benchmark reviews with duration == cap (benchmark review i = dump review_th - 1). Users that are not capped
have no timeout reviews.

Estimator (per subset S of benchmark reviews, from the per-review dumps): per user u, n_u = reviews in S, d_u = sum over S of
(LogLoss SM96 - LogLoss SM92); points = sum d_u / (0.275757 * sum n_u) * 100 = % of the big model (1 leak-free epoch)
normal-input instant LogLoss (0.275757, 722 users); review-weighted; paired user bootstrap 2,000 (seed 0, the report's index
matrix), 95% CI. Subsets: (a) all reviews, (b) excluding timeout reviews, (c) timeout reviews only.
Check: (a) must reproduce -0.214 [-0.233, -0.195] (the show-time report's `compare` on the rows, exact estimator, and this
dump estimator, both printed).
VERDICT: "unchanged" if |points(b) - points(a)| < 0.05 points, else "changed". Also reported: (b)'s CI, the shares, and SM96's
own absolute instant LogLoss (show-time and normal inputs) with and without timeouts, and the reproduction of SM92's existing
e2s rows (dumped per-user instant LogLoss and the new rows vs rows of the SM92 run).

  python sm100b_rule.py selftest   CPU, before launch: the SM96 side from the existing SM100 dumps + the estimator on
                                   (SM96 show-time - SM96 normal) vs the SM100 rows (+0.055 [+0.042, +0.072]) + (a) on rows
  python sm100b_rule.py decide     after the ticket: sm100b-decision.{json,md}; exit 2 while rows / dumps are missing
"""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
S100 = HERE.parent / "sm100"
G = HERE / "gpu"
ROWS = G / "sm100b-rows.jsonl"
DUMP = G / "dump"
DUMP96 = S100 / "gpu" / "dump"
SM92_OLD_ROWS = Path("C:/Users/jonat/workspace/wt/sm92/artifacts/small-model/sm92/gpu/sm92-inputs.jsonl")   # read only
TAG96, TAG92 = "sm96-inst-e31st-st-dec1ep32k/w8+s8", "sm92-e26-dec1ep32k/w8+s8"
BIG_INSTANT = 0.275757
BOUND = 0.05
TARGET_A = (-0.214, -0.233, -0.195)
NAME96, NAME92 = "the show-time instant-only small model (SM96)", "the show-time retrain of the release (SM92)"


def load_rule():
    spec = importlib.util.spec_from_file_location("sm100_rule", S100 / "sm100_rule.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dump(folder, user, label):
    f = folder / f"{user}_{label.replace('/', '_')}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return z["review_th"], z["y"].astype("float64"), z["p_instant"].astype("float64")


def per_user(R, users, folder_a, lab_a, folder_b, lab_b):
    """rows [user, n_all, n_to, sum_a_all, sum_b_all, sum_a_to, sum_b_to] + missing list; a, b = LogLoss sums."""
    out, missing = [], []
    for u in users:
        da, db = dump(folder_a, u, lab_a), dump(folder_b, u, lab_b)
        if da is None or db is None:
            missing.append(u); continue
        if not np.array_equal(da[0], db[0]) or not np.array_equal(da[1], db[1]):
            raise SystemExit(f"[sm100b-rule] user {u}: review_th / y differ between the two dumps")
        dur, _ = R.revlog(u)
        cap, _ = R.cap_of(dur)
        to = (dur[da[0].astype(np.int64) - 1] == cap) if cap is not None else np.zeros(len(da[0]), bool)
        la, lb = R.ll_terms(da[1], da[2]), R.ll_terms(db[1], db[2])
        out.append([u, len(to), int(to.sum()), la.sum(), lb.sum(), la[to].sum(), lb[to].sum()])
    return np.array(out, float) if out else np.zeros((0, 7)), missing


def subsets(a):
    n, nt = a[:, 1], a[:, 2]
    return {"all": (n, a[:, 3], a[:, 4]), "no_timeout": (n - nt, a[:, 3] - a[:, 5], a[:, 4] - a[:, 6]),
            "timeout_only": (nt, a[:, 5], a[:, 6])}


def est(n, sa, sb):
    """points of (a - b) per review in % of BIG_INSTANT, paired user bootstrap 2,000 seed 0; a, b also as per-review LogLoss."""
    idx = np.random.default_rng(0).integers(0, len(n), size=(2000, len(n)))
    d = sa - sb
    boot = d[idx].sum(1) / (BIG_INSTANT * n[idx].sum(1)) * 100
    return {"reviews": int(n.sum()), "users": int((n > 0).sum()), "a": round(float(sa.sum() / n.sum()), 6),
            "b": round(float(sb.sum() / n.sum()), 6), "delta": round(float(d.sum() / n.sum()), 6),
            "points": round(float(d.sum() / (BIG_INSTANT * n.sum()) * 100), 3),
            "points_ci": [round(float(np.percentile(boot, q)), 3) for q in (2.5, 97.5)],
            "users_better": int((sa < sb)[n > 0].sum()), "users_worse": int((sa > sb)[n > 0].sum())}


def fp(x):
    return f"{x['points']:+.3f} [{x['points_ci'][0]:+.3f}, {x['points_ci'][1]:+.3f}]"


def selftest():
    R = load_rule()
    rep = R.report()
    users, big = rep.pools()
    ok = True
    # (1) rows: (a) on the existing SM96 and SM92 rows through the show-time report's compare
    s96 = rep.load([(R.SM96_ROWS, R.L0, None)], set(users))
    s92 = rep.load([(SM92_OLD_ROWS, TAG92 + "@e2s-query", None)], set(users))
    us = sorted(set(users) & set(s96) & set(s92))
    pa = rep.compare(s96, s92, big, us)["instant"]
    a_ok = len(us) == 722 and (pa["points"], *pa["points_ci"]) == TARGET_A
    print(f"[sm100b-rule] selftest rows: {len(us)} users; SM96 - SM92 e2s instant (a, exact report estimator) {fp(pa)} -> {'OK' if a_ok else 'FAILED'}")
    ok &= a_ok
    # (2) dumps: the SM96 side. Estimator on (SM96 show-time - SM96 normal) vs the SM100 result (+0.055 [+0.042, +0.072])
    a, missing = per_user(R, us, DUMP96, R.L0, DUMP96, R.LNORMAL)
    sub = subsets(a)
    cost = est(*sub["all"])
    e_ok = not missing and len(a) == 722 and cost["points"] == 0.055
    print(f"[sm100b-rule] selftest dumps: {len(a)} users ({len(missing)} missing); estimator, SM96 show-time - normal, all reviews: {fp(cost)} "
          f"(rows: +0.055 [+0.042, +0.072]) -> {'OK' if e_ok else 'FAILED'}")
    ok &= e_ok
    # (3) SM96's dumped per-user LogLoss equals its rows, and the SM96 absolute side with / without timeouts
    worst = 0.0
    for r_, u in zip(a, [int(x) for x in a[:, 0]]):
        worst = max(worst, abs(r_[3] / r_[1] - s96[u]["instant"][0]))
    r_ok = worst < 1e-6
    print(f"[sm100b-rule] selftest: SM96 dumped vs row instant LogLoss, max abs diff over users {worst:.2e} -> {'OK' if r_ok else 'FAILED'}")
    ok &= r_ok
    n, nt = a[:, 1].sum(), a[:, 2].sum()
    ll = {"all": a[:, 3].sum() / n, "no_timeout": (a[:, 3].sum() - a[:, 5].sum()) / (n - nt), "timeout_only": a[:, 5].sum() / nt}
    print(f"[sm100b-rule] SM96 absolute instant LogLoss (show-time inputs): all {ll['all']:.6f}, without timeouts {ll['no_timeout']:.6f}, "
          f"timeouts only {ll['timeout_only']:.6f}; timeout reviews {int(nt):,} of {int(n):,} ({nt / n * 100:.2f}%)")
    ok &= int(nt) > 0
    print(f"[sm100b-rule] selftest: {'ALL OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def decide():
    R = load_rule()
    rep = R.report()
    users, big = rep.pools()
    new = rep.load([(ROWS, TAG92 + "@e2s-query", None)], set(users))
    if not set(users) <= set(new):
        print(f"[sm100b-rule] pending: SM92 e2s rows on {len(set(users) & set(new))} of {len(users)} rule users")
        return 2
    s96 = rep.load([(R.SM96_ROWS, R.L0, None)], set(users))
    old = rep.load([(SM92_OLD_ROWS, TAG92 + "@e2s-query", None)], set(users))
    us = sorted(users)
    a, missing = per_user(R, us, DUMP96, R.L0, DUMP, TAG92 + "@e2s-query")
    if missing:
        print(f"[sm100b-rule] pending: dumps missing for {len(missing)} users, e.g. {missing[:5]}")
        return 2
    res = {"users": len(a), "big_instant": BIG_INSTANT, "rows_pair_a": rep.compare(s96, new, big, us)["instant"]}
    # reproduction of SM92's existing e2s rows
    diffs_rows = [abs(new[u]["instant"][0] - old[u]["instant"][0]) for u in us if u in old]
    diffs_dump = [abs(a[i, 4] / a[i, 1] - old[u]["instant"][0]) for i, u in enumerate(us) if u in old]
    res["reproduction_sm92_e2s"] = {"users_compared": len(diffs_rows), "max_abs_diff_new_rows_vs_old_rows": max(diffs_rows, default=None),
                                    "rows_bit_for_bit": bool(diffs_rows) and max(diffs_rows) == 0.0,
                                    "max_abs_diff_dumped_vs_old_rows": max(diffs_dump, default=None)}
    sub = subsets(a)
    res["pair"] = {k: est(*v) for k, v in sub.items()}
    pa, pb, pc = res["pair"]["all"], res["pair"]["no_timeout"], res["pair"]["timeout_only"]
    res["a_reproduces_target"] = [res["rows_pair_a"]["points"], *res["rows_pair_a"]["points_ci"]] == list(TARGET_A)
    res["dump_a_within_0.002_of_target"] = all(abs(x - t) <= 0.002 for x, t in zip([pa["points"], *pa["points_ci"]], TARGET_A))
    res["b_minus_a_points"] = round(pb["points"] - pa["points"], 3)
    res["verdict"] = "unchanged" if abs(pb["points"] - pa["points"]) < BOUND else "changed"
    n, nt = a[:, 1].sum(), a[:, 2].sum()
    res["timeout_share"] = round(float(nt / n), 5)
    # SM96 / SM92 absolute instant LogLoss, show-time inputs, with and without timeouts; SM96 normal inputs too
    def absolute(sa_all, sa_to):
        return {"all": round(float(sa_all / n), 6), "without_timeouts": round(float((sa_all - sa_to) / (n - nt)), 6),
                "timeouts_only": round(float(sa_to / nt), 6)}
    res["sm96_instant_logloss_showtime"] = absolute(a[:, 3].sum(), a[:, 5].sum())
    res["sm92_instant_logloss_showtime"] = absolute(a[:, 4].sum(), a[:, 6].sum())
    an, miss = per_user(R, us, DUMP96, R.LNORMAL, DUMP96, R.LNORMAL)
    if not miss:
        res["sm96_instant_logloss_normal"] = absolute(an[:, 3].sum(), an[:, 5].sum())
    (HERE / "sm100b-decision.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    md = ["# SM100b: model comparison without timeout reviews (decision)", "",
          f"Verdict: **{res['verdict']}** (|(b) - (a)| = {abs(res['b_minus_a_points']):.3f} points; unchanged needs < {BOUND}). "
          f"Pair: {NAME96} minus {NAME92}, instant head, show-time inputs (e2s-query), {res['users']} users of the 732 set, review-weighted, "
          f"paired user bootstrap 2,000 (seed 0), 95% CI; points = % of the big model (1 leak-free epoch) normal-input instant LogLoss {BIG_INSTANT}. "
          f"Negative = {NAME96} better.", "",
          "| reviews | review count | users | LogLoss SM96 | LogLoss SM92 | delta | points [95% CI] | users SM96 better / worse |", "|---|---|---|---|---|---|---|---|"]
    for k, lab in (("all", "(a) all reviews"), ("no_timeout", "(b) without timeout reviews"), ("timeout_only", "(c) timeout reviews only")):
        p = res["pair"][k]
        md.append(f"| {lab} | {p['reviews']:,} | {p['users']} | {p['a']} | {p['b']} | {p['delta']:+.6f} | {fp(p)} | {p['users_better']} / {p['users_worse']} |")
    md += ["", f"(a) reproduces the target {list(TARGET_A)}: {res['a_reproduces_target']} (report estimator on the rows: {fp(res['rows_pair_a'])}); "
               f"the dump estimator's (a) is within 0.002 of it: {res['dump_a_within_0.002_of_target']}. "
               f"Timeout reviews {int(nt):,} = {res['timeout_share'] * 100:.2f}% of {int(n):,}.", "",
           f"{NAME96}, instant LogLoss, show-time inputs: {json.dumps(res['sm96_instant_logloss_showtime'])}; normal inputs: "
           f"{json.dumps(res.get('sm96_instant_logloss_normal'))}.",
           f"{NAME92}, show-time inputs: {json.dumps(res['sm92_instant_logloss_showtime'])}.", "",
           f"Reproduction of {NAME92}'s existing e2s rows: {json.dumps(res['reproduction_sm92_e2s'])}."]
    (HERE / "sm100b-decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[sm100b-rule] verdict {res['verdict']}: (a) {fp(pa)}; (b) {fp(pb)}; (c) {fp(pc)} points ({res['users']} users); "
          f"reproduces target {res['a_reproduces_target']} (dump (a) within 0.002: {res['dump_a_within_0.002_of_target']}); SM92 rows bit-for-bit {res['reproduction_sm92_e2s']['rows_bit_for_bit']}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "decide"
    raise SystemExit(selftest() if cmd == "selftest" else decide())
