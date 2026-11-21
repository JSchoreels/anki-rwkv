#!/usr/bin/env python
"""Trained smaller state for Tiny (SM137): pre-registered rules (fixed 2026-11-21 before any GPU ticket; CPU, seconds).

All 732-set numbers: show-time inputs (`e2s-query`), leak-free (neighbour), w8 weights, the 722 rule users of the 732 set
(showtime_report.py pools(), sha256 pinned), review-weighted LogLoss, paired user bootstrap 2,000, seed 0, 95% CI;
"%" = 100 x (new / reference - 1) of the reference's own LogLoss (pct_of_ref). Trained heads: io instant; th curve +
instant. The lane shape of a run is `v<V>k<K>` (value x key lanes kept per head in the card and note state).

PILOT (`pilot --lanes v16k8`, Instant-Only chain, after the one constant-LR lane epoch 42):
  x = 100 x (lane epoch 42 / the six-layer Tiny with longer recovery (SM135)'s epoch 42 - 1), instant head, both w8+s8.
  SM135's epoch 42 is the matched control: same start (the six-layer Tiny (SM130)'s epoch 41) except the cut lanes,
  same data order, same options. Decision, in this order:
    any non-finite rule user (lane epoch)  -> stop
    x > +5.00%                             -> stop (far outside any plausible cost: investigate before anything else)
    x <= +0.30%                            -> go (the full plan with these lanes)
    lanes = v16k8                          -> pilot_v16k12 (one more pilot epoch with the milder cut, same rule)
    otherwise                              -> stop
  Reported, no decision: the lane epoch at the release format w8+s8hx8 vs the control's w8+s8; the zero-shot cut (the
  lane cut of epoch 41, no training) vs SM130's epoch 41.
EARLY TWO-HEAD CHECK (`early-th --lanes L`, after the two-head lane epoch 33, before its cool-down):
  y_h = 100 x (lane epoch 33 / SM130's epoch 32 - 1), w8+s8, curve and instant. stop if non-finite or any y_h > +1.00%
  (point); otherwise cooldown. (No matched control exists for the two-head chain; one more constant-LR epoch did not
  help the Instant-Only chain (SM135: +0.082%), so the offset epoch is not a hidden advantage.)
MAIN RULE (`decide --chain io|th --lanes L`, after the two-pass cool-down):
  d_h = 100 x (lane student / Small - 1) per trained head; the student at the release-candidate state format w8+s8hx8
  (int8 rows with float16 row scales, int8 shifts, for cards and notes), Small = the longer Small Instant-Only model
  (SM113) / the show-time two-head cool-down for Small (SM118) at their published w8+s8 rows (LogLoss 0.275933 /
  curve 0.300016, instant 0.276599 on these users). go iff for EVERY trained head the 95% CI upper bound of d_h <= +1.00%
  (the Tiny budget, user 11-18), every rule user finite, AND the long-card check passed. Non-inferiority to a budget: the
  repo's 0.05% minimum effect applies to "better than" claims only and is not part of this rule.
  Reported: the same at w8+s8; vs the six-layer Tiny (SM130) final at w8+s8 and at w8+s8hx8 (rescored, same users).
LONG-CARD CHECK (`longcard --chain io|th --lanes L`, CPU rows with --prior-buckets 0,32,128,300,1000, both models at
  w8+s8hx8, the users of users-longcard.json = the 50 Tiny users the long-card drift test (SM136) completed):
  per trained head, cost_B = 100 x (sum LogLoss of the student / sum LogLoss of the six-layer Tiny (SM130) final - 1)
  over every predicted review in bucket B (the scorer's "all" sums; 128+ = buckets from 128 prior answers on), and
  excess = cost_128+ - cost_all, paired user bootstrap 2,000, seed 0. pass iff no student user is non-finite and, per
  trained head, the 95% CI upper bound of excess <= +0.25% (SM136's ceiling for a format change, applied to what the
  long cards lose beyond the student's own average cost, so a uniform capacity cost is not counted twice).

  python sm137_rule.py selftest
  python sm137_rule.py users                       -> users-longcard.json
  python sm137_rule.py smoke --gpu G --cpu C       GPU vs CPU rows of the lane cut, |difference| <= 1e-5 (exit 1 otherwise)
  python sm137_rule.py pilot --lanes v16k8         -> sm137-pilot-io-v16k8.{json,md}; exit 2 while rows are missing
  python sm137_rule.py early-th --lanes v16k8      -> sm137-early-th-v16k8.json
  python sm137_rule.py longcard --chain io --lanes v16k8 -> sm137-longcard-io-v16k8.json
  python sm137_rule.py decide --chain io --lanes v16k8   -> sm137-decision-io-v16k8.{json,md}
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
G, CPU = HERE / "gpu", HERE / "cpu"
WT = HERE.parents[2]
SMG = WT / "artifacts/small-model"
W = Path("C:/Users/jonat/workspace/wt")
REPORT = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/small-model/showtime/showtime_report.py")
REPORT_SHA = "3cdfed03afd40734d4b1931c244e8c1e73453d281a40c3843e80cd310d70dd9d"
EARLY_ROWS, ROWS, LONG_ROWS = G / "sm137-early-rows.jsonl", G / "sm137-rows.jsonl", CPU / "sm137-longcard-rows.jsonl"
R135E, R130E, R130 = SMG / "sm135/gpu/sm135-early-rows.jsonl", SMG / "sm130/gpu/sm130-early-rows.jsonl", SMG / "sm130/gpu/sm130-rows.jsonl"
R118, R113, R136 = SMG / "sm118/gpu/sm118-rows.jsonl", W / "sm113/artifacts/small-model/sm113/gpu/sm113-rows.jsonl", SMG / "sm136/gpu/sm136-rows.jsonl"
PINS = {R135E: "3bfa250f8df69f909354ab988da9f73eeea0486635436771591afa838294f598",
        R130E: "f22c2f85ecab321880cee0cb6d4d8ff05b9771effabc4a0bede29d29095f6cf6",
        R130: "f4b38086f872762442d754752542e9d26606fc11fee10bd586c8e2e739f0a5b6",
        R118: "430ffe8a809c27079862acf0bfb57b12b279a8472aa5eff19dfb232db378befb",
        R113: "af104a0774b1aa7da7b26bc48ac667a9991d86efd8dda4ac8de7bb51a3fe281a",
        R136: "03bef4ef8b8cd43ac936c43f814da6293e7946c7d137c95ce2f5cddc1c79731c"}
E2S = "@e2s-query"
PUBLISHED, RC = "w8+s8", "w8+s8hx8"
GO_PILOT, STOP_PILOT, CEILING, EARLY_TH, LONG_CEILING = 0.30, 5.00, 1.00, 1.00, 0.25
LONG_EDGES = [0, 32, 128, 300, 1000]
FALLBACK = {"v16k8": "v16k12"}   # the milder second pilot after a v16k8 pilot above the go bar
CHAINS = {
    "io": {"heads": ("instant",), "final": "sm137-io-{lanes}-dec2ep32k-t99", "sm130_final": "sm130-io-dec2ep32k-t99",
           "small": (R113, "sm113-inst-e35st-st-dec2ep32k-t99"), "small_name": "the longer Small Instant-Only model (SM113)"},
    "th": {"heads": ("curve", "instant"), "final": "sm137-th-{lanes}-dec2ep32k-t91s2", "sm130_final": "sm130-th-dec2ep32k-t91s2",
           "small": (R118, "sm118-e26-dec2ep32k-t91s2"), "small_name": "the show-time two-head cool-down for Small (SM118)"},
}
KEY = {"curve": "ahead", "instant": "instant"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def report():
    if sha(REPORT) != REPORT_SHA:
        raise SystemExit("[sm137-rule] showtime_report.py changed: refusing")
    spec = importlib.util.spec_from_file_location("showtime_report", REPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_pins():
    for path, want in PINS.items():
        if sha(path) != want:
            raise SystemExit(f"[sm137-rule] reference rows {path.name} changed: refusing")


def pct(x, h):
    return {"value": x[h]["pct_of_ref"], "ci": x[h]["pct_of_ref_ci"], "new": x[h]["new"], "ref": x[h]["ref"],
            "better": x[h]["users_better"], "worse": x[h]["users_worse"]}


def f(x):
    return f"{x['value']:+.3f}% [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"


def nonfinite_users(path, label):
    out = set()
    if Path(path).exists():
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("label") == label and "nonfinite" in r and "benchmark" not in r:
                out.add(r.get("user_id"))
    return out


def finite(rows, users, heads):
    return all(math.isfinite(rows[u][KEY[h]][0]) for u in users for h in heads)


def pilot_decision(x, ok_finite, lanes):
    if not ok_finite:
        return "stop"
    if x > STOP_PILOT:
        return "stop"
    if x <= GO_PILOT:
        return "go"
    return f"pilot_{FALLBACK[lanes]}" if lanes in FALLBACK else "stop"


def pilot(lanes, rows=EARLY_ROWS, out_dir=HERE, ctl=(R135E, "sm135-io-e42-st-const")):
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    lab = f"sm137-io-{lanes}-e42-st-const"
    a = r.load([(rows, f"{lab}/{PUBLISHED}{E2S}", None)], have)
    c = r.load([(ctl[0], f"{ctl[1]}/{PUBLISHED}{E2S}", None)], have)
    bad = nonfinite_users(rows, f"{lab}/{PUBLISHED}{E2S}") & have
    if not (have <= set(a) | bad and have <= set(c)):
        print(f"[sm137-rule] pilot {lanes} pending: lane epoch {len(have & set(a))}, control {len(have & set(c))} of {len(users)}")
        return 2
    ok = not bad and finite(a, users, ("instant",))
    x = pct(r.compare(a, c, big, users), "instant") if ok else None
    decision = pilot_decision(x["value"] if x else float("inf"), ok, lanes)
    res = {"lanes": lanes, "users": len(users), "finite": ok, "nonfinite_users": sorted(bad), "vs_control": x,
           "control": ctl[1], "decision": decision, "go_at_most_pct": GO_PILOT, "stop_above_pct": STOP_PILOT}
    rc = r.load([(rows, f"{lab}/{RC}{E2S}", None)], have)
    if ok and have <= set(rc) and finite(rc, users, ("instant",)):
        res["release_format_vs_control"] = pct(r.compare(rc, c, big, users), "instant")
    z = r.load([(rows, f"sm137-io-{lanes}-e41-cut/{PUBLISHED}{E2S}", None)], have)
    z0 = r.load([(R130E, f"sm130-io-e41-st-const/{PUBLISHED}{E2S}", None)], have)
    if have <= set(z) and have <= set(z0) and finite(z, users, ("instant",)):
        res["zero_shot_cut_vs_sm130_e41"] = pct(r.compare(z, z0, big, users), "instant")
    name = f"sm137-pilot-io-{lanes}"
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    L = [f"# Trained smaller state for Tiny (SM137): pilot ({lanes}, Instant-Only epoch 42)", "",
         f"Decision: **{decision}**. Pre-registered: go if the lane epoch costs at most +{GO_PILOT:.2f}% instant vs the "
         f"matched control (the six-layer Tiny with longer recovery (SM135)'s epoch 42), w8+s8, show-time inputs, "
         f"{len(users)} users; above that a v16k8 pilot is followed by a v16k12 pilot, any other lane shape stops; "
         f"stop above +{STOP_PILOT:.2f}% or on a non-finite user.", "",
         "| comparison | instant % [95% CI] | LogLoss new / reference | better / worse users |", "|---|---|---|---|"]
    if x:
        L.append(f"| lane epoch 42 (w8+s8) vs SM135 epoch 42 (w8+s8) | {f(x)} | {x['new']} / {x['ref']} | {x['better']} / {x['worse']} |")
    for key, what in (("release_format_vs_control", "lane epoch 42 (w8+s8hx8) vs SM135 epoch 42 (w8+s8)"),
                      ("zero_shot_cut_vs_sm130_e41", "zero-shot lane cut of SM130 epoch 41 vs SM130 epoch 41 (w8+s8)")):
        if key in res:
            y = res[key]
            L.append(f"| {what} | {f(y)} | {y['new']} / {y['ref']} | {y['better']} / {y['worse']} |")
    (out_dir / f"{name}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[sm137-rule] pilot {lanes} {decision}: " + (f"instant vs SM135 e42 {f(x)}" if x else "non-finite"))
    return 0


def early_th(lanes, rows=EARLY_ROWS, out_dir=HERE):
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    lab = f"sm137-th-{lanes}-e33-st-const/{PUBLISHED}{E2S}"
    a = r.load([(rows, lab, None)], have)
    p = r.load([(R130E, f"sm130-th-e32-st-const/{PUBLISHED}{E2S}", None)], have)
    bad = nonfinite_users(rows, lab) & have
    if not (have <= set(a) | bad and have <= set(p)):
        print(f"[sm137-rule] early-th {lanes} pending: {len(have & set(a))} of {len(users)}")
        return 2
    ok = not bad and finite(a, users, ("curve", "instant"))
    y = {h: pct(r.compare(a, p, big, users), h) for h in ("curve", "instant")} if ok else {}
    decision = "cooldown" if ok and all(y[h]["value"] <= EARLY_TH for h in y) else "stop"
    res = {"lanes": lanes, "users": len(users), "finite": ok, "vs_sm130_e32": y, "decision": decision, "stop_above_pct": EARLY_TH}
    (out_dir / f"sm137-early-th-{lanes}.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(f"[sm137-rule] early-th {lanes} {decision}: " + ", ".join(f"{h} {f(y[h])}" for h in y))
    return 0


# ----------------------------------------------------------------------------------------------- long cards

def long_sums(path, label, heads, users):
    """{user: {head: (ll_long, ll_all, n_long)}} from --prior-buckets rows ("all" = every predicted review); non-finite
    users (a 'nonfinite' row or a non-finite sum) in the second value."""
    out, bad = {}, set()
    long_from = LONG_EDGES.index(128)
    if Path(path).exists():
        for line in open(path, encoding="utf-8"):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("label") != label or row.get("user_id") not in users:
                continue
            if "nonfinite" in row and "benchmark" not in row:
                bad.add(row["user_id"])
                continue
            pb = row.get("prior_buckets")
            if not pb or pb.get("edges") != LONG_EDGES:
                continue
            got = {}
            for h in heads:
                b = pb["all"].get(KEY[h])
                if b is None:
                    break
                ll, n = b["ll"], b["n"]
                got[h] = (sum(ll[long_from:]), sum(ll), sum(n[long_from:]))
            else:
                if all(math.isfinite(v[0]) and math.isfinite(v[1]) for v in got.values()):
                    out[row["user_id"]] = got
                else:
                    bad.add(row["user_id"])
    return out, bad


def excess_stats(new, ref, users, head, boot=2000, seed=0):
    """cost_128+, cost_all and excess = cost_128+ - cost_all (%), point and paired user bootstrap 95% CI."""
    a = np.array([[new[u][head][0], new[u][head][1]] for u in users])
    b = np.array([[ref[u][head][0], ref[u][head][1]] for u in users])

    def stat(idx):
        sa, sb = a[idx].sum(0), b[idx].sum(0)
        long_cost, all_cost = 100 * (sa[0] / sb[0] - 1), 100 * (sa[1] / sb[1] - 1)
        return long_cost, all_cost, long_cost - all_cost

    point = stat(np.arange(len(users)))
    idx = np.random.default_rng(seed).integers(0, len(users), size=(boot, len(users)))
    draws = np.array([stat(i) for i in idx])
    ci = [[round(float(np.percentile(draws[:, j], q)), 3) for q in (2.5, 97.5)] for j in range(3)]
    return {"cost_128plus": round(point[0], 3), "cost_128plus_ci": ci[0], "cost_all": round(point[1], 3),
            "cost_all_ci": ci[1], "excess": round(point[2], 3), "excess_ci": ci[2],
            "long_reviews": int(sum(new[u][head][2] for u in users))}


def longcard(chain, lanes, rows=LONG_ROWS, out_dir=HERE):
    c = CHAINS[chain]
    users = json.loads((HERE / "users-longcard.json").read_text())["users"]
    new_label = c["final"].format(lanes=lanes) + f"/{RC}{E2S}"
    ref_label = c["sm130_final"] + f"/{RC}{E2S}"
    new, bad = long_sums(rows, new_label, c["heads"], set(users))
    ref, _ = long_sums(rows, ref_label, c["heads"], set(users))
    if not (set(users) <= set(new) | bad and set(users) <= set(ref)):
        print(f"[sm137-rule] longcard {chain} {lanes} pending: student {len(set(new) | bad)}, SM130 {len(ref)} of {len(users)}")
        return 2
    per = {h: excess_stats(new, ref, users, h) for h in c["heads"]} if not bad else {}
    ok = not bad and all(per[h]["excess_ci"][1] <= LONG_CEILING for h in per)
    res = {"chain": chain, "lanes": lanes, "users": len(users), "nonfinite_users": sorted(bad), "heads": per,
           "ceiling_pct": LONG_CEILING, "verdict": "pass" if ok else "fail"}
    (out_dir / f"sm137-longcard-{chain}-{lanes}.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(f"[sm137-rule] longcard {chain} {lanes} {res['verdict']}: " +
          ", ".join(f"{h} excess {per[h]['excess']:+.3f}% [{per[h]['excess_ci'][0]:+.3f}, {per[h]['excess_ci'][1]:+.3f}]" for h in per))
    return 0


def decide(chain, lanes, rows=ROWS, out_dir=HERE):
    c = CHAINS[chain]
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    tag = c["final"].format(lanes=lanes)
    s_rc = r.load([(rows, f"{tag}/{RC}{E2S}", None)], have)
    s_pub = r.load([(rows, f"{tag}/{PUBLISHED}{E2S}", None)], have)
    small = r.load([(c["small"][0], f"{c['small'][1]}/{PUBLISHED}{E2S}", None)], have)
    bad = nonfinite_users(rows, f"{tag}/{RC}{E2S}") & have
    lc = out_dir / f"sm137-longcard-{chain}-{lanes}.json"
    if not (have <= set(s_rc) | bad and have <= set(small)) or not lc.exists():
        print(f"[sm137-rule] decide {chain} {lanes} pending: student {len(have & set(s_rc))} of {len(users)}, long-card file "
              f"{'present' if lc.exists() else 'missing'}")
        return 2
    ok = not bad and finite(s_rc, users, c["heads"])
    d = {h: pct(r.compare(s_rc, small, big, users), h) for h in c["heads"]} if ok else {}
    long_ok = json.loads(lc.read_text())["verdict"] == "pass"
    go = ok and long_ok and all(d[h]["ci"][1] <= CEILING for h in d)
    res = {"chain": chain, "lanes": lanes, "users": len(users), "finite": ok, "nonfinite_users": sorted(bad),
           "vs_small_release_format": d, "long_card": long_ok, "verdict": "go" if go else "no go", "ceiling_pct": CEILING}
    if have <= set(s_pub) and ok:
        res["vs_small_published"] = {h: pct(r.compare(s_pub, small, big, users), h) for h in c["heads"]}
        s130 = r.load([(R130, f"{c['sm130_final']}/{PUBLISHED}{E2S}", None)], have)
        if have <= set(s130):
            res["vs_sm130_published"] = {h: pct(r.compare(s_pub, s130, big, users), h) for h in c["heads"]}
    s130rc = r.load([(rows, f"{c['sm130_final']}/{RC}{E2S}", None)], have)
    if ok and have <= set(s130rc):
        res["vs_sm130_release_format"] = {h: pct(r.compare(s_rc, s130rc, big, users), h) for h in c["heads"]}
    (out_dir / f"sm137-decision-{chain}-{lanes}.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    L = [f"# Trained smaller state for Tiny (SM137): decision ({chain}, {lanes})", "",
         f"Verdict: **{res['verdict']}**. Rule: every trained head's 95% CI upper bound of 100 x (student at w8+s8hx8 / "
         f"{c['small_name']} at w8+s8 - 1) <= +{CEILING:.2f}%, every rule user finite, long-card check pass ({long_ok}).", "",
         "| head | vs Small, release format % [95% CI] | LogLoss student / Small |", "|---|---|---|"]
    for h in d:
        L.append(f"| {h} | {f(d[h])} | {d[h]['new']} / {d[h]['ref']} |")
    (out_dir / f"sm137-decision-{chain}-{lanes}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[sm137-rule] decide {chain} {lanes} {res['verdict']}: " + ", ".join(f"{h} {f(d[h])}" for h in d))
    return 0


def smoke(gpu, cpu, tol=1e-5):
    """GPU smoke of the lane cut: every (user, label, head) benchmark LogLoss of the GPU rows within tol of the CPU rows
    (AGENT.md strict rule for the int8 variants w8+s8 / w8+s8hx8), and the same row set."""
    def rows(path):
        out = {}
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if "benchmark" in r:
                for h in ("ahead", "instant"):
                    out[(r["user_id"], r["label"], h)] = r["benchmark"][h]["LogLoss"]
        return out
    g, c = rows(gpu), rows(cpu)
    worst = max((abs(g[k] - c[k]) for k in c if k in g), default=float("inf"))
    ok = set(g) == set(c) and len(c) > 0 and worst <= tol
    print(f"[sm137-rule] smoke {'OK' if ok else 'FAILED'}: {len(c)} (user, label, head) values, max |GPU - CPU| {worst:.2e} "
          f"(tol {tol:g})")
    return 0 if ok else 1


def users_longcard():
    got = {}
    for line in open(R136, encoding="utf-8"):
        r = json.loads(line)
        if r.get("label") in ("sm124-c1-e29st-dec2ep32k-t91s2/w8+s8hx8@e2s-query", "sm115-c1-e38st-st-dec2ep32k-t99/w8+s8hx8@e2s-query") \
                and "benchmark" in r:
            got.setdefault(r["label"], set()).add(r["user_id"])
    users = sorted(set.intersection(*got.values()))
    doc = {"note": "Trained smaller state for Tiny (SM137): long-card check users = the users of the 732 set with a card of "
                   ">= 128 prior answers that the long-card drift test (SM136) scored completely for both Tiny models "
                   "(the show-time two-head cool-down for Tiny (SM124) and the longer cool-down for Tiny (SM115)) at "
                   "w8+s8hx8 on the CPU under the 5.9 GB memory guard. Public benchmark user ids only.",
           "source": str(R136.relative_to(WT)), "source_sha256": PINS[R136], "users": users}
    (HERE / "users-longcard.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"[sm137-rule] users-longcard.json: {len(users)} users")
    return 0


def selftest():
    cases = [((0.10, True, "v16k8"), "go"), ((0.30, True, "v16k8"), "go"), ((0.31, True, "v16k8"), "pilot_v16k12"),
             ((0.31, True, "v16k12"), "stop"), ((6.0, True, "v16k8"), "stop"), ((0.0, False, "v16k8"), "stop"),
             ((-0.2, True, "v16k12"), "go")]
    ok = all(pilot_decision(*a) == want for a, want in cases)
    rng = np.random.default_rng(1)
    users = list(range(60))
    ref = {u: {"instant": (float(rng.uniform(5, 50)), float(rng.uniform(100, 900)), 100)} for u in users}
    same = excess_stats(ref, ref, users, "instant")
    uniform = {u: {"instant": (v["instant"][0] * 1.01, v["instant"][1] * 1.01, 100)} for u, v in ref.items()}
    longbad = {u: {"instant": (v["instant"][0] * 1.02, v["instant"][1] + v["instant"][0] * 0.02, 100)} for u, v in ref.items()}
    e_uniform, e_long = excess_stats(uniform, ref, users, "instant"), excess_stats(longbad, ref, users, "instant")
    ok &= same["excess"] == 0 and same["excess_ci"] == [0.0, 0.0]
    ok &= abs(e_uniform["cost_all"] - 1.0) < 1e-6 and abs(e_uniform["excess"]) < 1e-6      # uniform cost: no excess
    ok &= e_long["cost_128plus"] > 1.99 and e_long["excess_ci"][0] > LONG_CEILING            # long-card-only cost: fails
    # the pilot loader on real rows: the control against itself is x = 0 -> go
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "rows.jsonl"
        with open(src, "w", encoding="utf-8") as out:
            for line in open(R135E, encoding="utf-8"):
                out.write(line.replace("sm135-io-e42-st-const/", "sm137-io-v16k8-e42-st-const/"))
        rc = pilot("v16k8", rows=src, out_dir=Path(tmp))
        res = json.loads((Path(tmp) / "sm137-pilot-io-v16k8.json").read_text())
        ok &= rc == 0 and res["decision"] == "go" and res["vs_control"]["value"] == 0.0
    print(f"[sm137-rule] selftest {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    sub.add_parser("users")
    s = sub.add_parser("smoke")
    s.add_argument("--gpu", required=True)
    s.add_argument("--cpu", required=True)
    for name in ("pilot", "early-th"):
        s = sub.add_parser(name)
        s.add_argument("--lanes", default="v16k8")
    for name in ("longcard", "decide"):
        s = sub.add_parser(name)
        s.add_argument("--chain", choices=CHAINS, required=True)
        s.add_argument("--lanes", default="v16k8")
    a = p.parse_args()
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "users":
        return users_longcard()
    if a.cmd == "smoke":
        return smoke(a.gpu, a.cpu)
    if a.cmd == "pilot":
        return pilot(a.lanes)
    if a.cmd == "early-th":
        return early_th(a.lanes)
    if a.cmd == "longcard":
        return longcard(a.chain, a.lanes)
    return decide(a.chain, a.lanes)


if __name__ == "__main__":
    raise SystemExit(main())
