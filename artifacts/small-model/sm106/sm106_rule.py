#!/usr/bin/env python
"""The Tiny two-head model (SM106): pre-registered rules (fixed 2026-11-11 before any GPU run; CPU, seconds).

New model: the show-time retrain (SM92)'s constant-LR epoch 26 with the card stream cut to 1 layer (w64-c1) + 3
constant-LR two-head epochs 27..29 on show-time inputs (SM92 epoch 26's recipe, both heads) + the release's one-epoch 32k
Muon cool-down on show-time inputs from epoch 29 (SM92's cool-down except the start and the architecture).
Label `sm106-c1-e29st-st-dec1ep32k/w8+s8@e2s-query`.

Everything is on show-time inputs (`e2s-query`), leak-free w8+s8 rows (`--split-existing --deleted-cards neighbour`).
Loaders and the paired bootstrap are the show-time report's (`artifacts/small-model/showtime/showtime_report.py` on main,
sha256 pinned, the same pin as the Tiny Instant-Only model (SM101)): 2,000 resamples, seed 0, 95% CI, review-weighted
LogLoss, 722 users of the 732 set (the users the big model (1 leak-free epoch) rows have). Unit: % = % of the big model
(1 leak-free epoch)'s show-time LogLoss on the same users (curve 0.29958, instant 0.277157), per head. References are
read from their own rows (sha256 pinned where they live in this branch), never rescored.

EARLY CHECK (`early`, after epoch 27; stop rule):
  arm  = epoch 27 (constant LR, w64-c1), label sm106-c1-e27-st-const/w8+s8@e2s-query
  ref  = SM92's epoch 26 (constant LR, w64, the cut's source), label sm92-e26-const/w8+s8@e2s-query
  both heads, show-time inputs, paired, 722 users.
  stop      if the point estimate arm - ref > +0.25% on the curve head OR on the instant head (the Tiny Instant-Only
            model (SM101)'s early bar, applied to each trained head)
  continue  otherwise.
FINAL (`decide`, after the cool-down):
  A  the pruning bar (the verdict): new - the big model (1 leak-free epoch), show-time inputs: curve 95% CI upper end
     <= +0.5% AND instant 95% CI upper end <= +0.4%.   pass = "go", otherwise "no go".
  B  report only (no verdict): new - SM92 (curve and instant), new - SM101 (instant; SM101's curve is untrained).
  Also reported: new - the release (SM63); the 1,500 pool (% of the release's own LogLoss: new - the release, new - SM101
  instant); the row check on the 6 check users.

  python sm106_rule.py selftest   -> the rules' code on known rows (SM92 / the release / SM101 as "new"; early rule on
                                     SM92 vs the release both ways); exit 0 if all equal the recorded values
  python sm106_rule.py early      -> sm106-early.{json,md}; exit 2 while rows are missing
  python sm106_rule.py decide     -> sm106-decision.{json,md}; exit 2 while the new model's 722-user rows are missing
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
G = HERE / "gpu"
WT = HERE.parents[2]
REPORT = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/small-model/showtime/showtime_report.py")
REPORT_SHA = "3cdfed03afd40734d4b1931c244e8c1e73453d281a40c3843e80cd310d70dd9d"   # main b07b6a13 (same pin as SM96 / SM98 / SM101)
ROWS, ROWS1500, EARLY_ROWS = G / "sm106-rows.jsonl", G / "sm106-rows-1500.jsonl", G / "sm106-early-rows.jsonl"
LBL = "sm106-c1-e29st-st-dec1ep32k/w8+s8"
EARLY_ARM, EARLY_REF = "sm106-c1-e27-st-const/w8+s8", "sm92-e26-const/w8+s8"
S101G = WT / "artifacts/small-model/sm101/gpu"
PINS = {S101G / "sm101-rows.jsonl": "ea52895c4f57e63dd8c8ad13a882cfde997205d69aeee8700d3d016a1ed1e823",
        S101G / "sm101-rows-1500.jsonl": "c80c3dee26fdade604abe896204d4f1bfced63764441ec567d7665491b3affb8"}
LBL101 = "sm101-c1-e38st-st-dec1ep32k/w8+s8"
BAR = {"curve": 0.5, "instant": 0.4}
EARLY_STOP = 0.25
HEADS = ("curve", "instant")


def report():
    got = hashlib.sha256(REPORT.read_bytes()).hexdigest()
    if got != REPORT_SHA:
        raise SystemExit(f"[sm106-rule] showtime_report.py changed (sha256 {got[:12]} != {REPORT_SHA[:12]}): refusing")
    spec = importlib.util.spec_from_file_location("showtime_report", REPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_pins():
    for path, want in PINS.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"[sm106-rule] reference rows {path.name} changed (sha256 {got[:12]} != {want[:12]}): refusing")


def base(r):
    """722 rule users and the big model (1 leak-free epoch)'s show-time rows (the % denominator and weights)."""
    users, _big_normal = r.pools()
    bige = r.load(r.MODELS["big"]["e2s"], set(users))
    missing = set(users) - set(bige)
    if missing:
        raise SystemExit(f"[sm106-rule] the big model's show-time rows miss {len(missing)} rule users")
    return users, bige


def fp(x):
    return f"{x['points']:+.3f}% [{x['points_ci'][0]:+.3f}, {x['points_ci'][1]:+.3f}]"


def fpct(x):
    return f"{x['pct_of_ref']:+.3f}% [{x['pct_of_ref_ci'][0]:+.3f}, {x['pct_of_ref_ci'][1]:+.3f}]"


def bar_ok(x, head):
    return x["points_ci"][1] <= BAR[head]


def early_decision(c):
    return "stop" if any(c[h]["points"] > EARLY_STOP for h in HEADS) else "continue"


def final(r, new, users, bige):
    """Rule A on `new` (show-time rows): new - big, both on show-time inputs, per head."""
    us = sorted(set(users) & set(new))
    a = r.compare(new, bige, bige, us)
    ok = {h: bar_ok(a[h], h) for h in HEADS}
    return {"users": len(us), "A_new_minus_big_e2s": {h: a[h] for h in HEADS}, "A_pass": ok,
            "verdict": "go" if all(ok.values()) and len(us) == len(users) else "no go"}


def selftest():
    r = report()
    check_pins()
    users, bige = base(r)
    have = set(users)
    n92 = r.load(r.MODELS["sm92"]["e2s"], have)
    rel = r.load(r.MODELS["release"]["e2s"], have)
    n101 = r.load([(S101G / "sm101-rows.jsonl", LBL101 + r.E2S, None)], have)
    f92, frel, f101 = final(r, n92, users, bige), final(r, rel, users, bige), final(r, n101, users, bige)
    us = sorted(have & set(n92) & set(rel))
    e1 = r.compare(n92, rel, bige, us)
    e2 = r.compare(rel, n92, bige, us)
    a92, arel = f92["A_new_minus_big_e2s"], frel["A_new_minus_big_e2s"]
    got = {"sm92_A": [a92[h]["points"] for h in HEADS] + [a92[h]["points_ci"][1] for h in HEADS],
           "release_A": [arel[h]["points"] for h in HEADS] + [arel[h]["points_ci"][1] for h in HEADS],
           "early_sm92_vs_release": [e1[h]["points"] for h in HEADS], "early_release_vs_sm92": [e2[h]["points"] for h in HEADS]}
    checks = {
        "users_722": f92["users"] == 722 == len(users),
        "sm92_as_new": got["sm92_A"] == SELF["sm92_A"] and f92["verdict"] == SELF["sm92_verdict"],
        "release_as_new": got["release_A"] == SELF["release_A"] and frel["verdict"] == SELF["release_verdict"],
        "sm101_as_new_no_go_curve_untrained": f101["verdict"] == "no go" and not f101["A_pass"]["curve"],
        "early_sm92_vs_release": got["early_sm92_vs_release"] == SELF["early_sm92_vs_release"] and early_decision(e1) == "continue",
        "early_release_vs_sm92": got["early_release_vs_sm92"] == SELF["early_release_vs_sm92"] and early_decision(e2) == "stop",
        "early_edges": early_decision({"curve": {"points": 0.25}, "instant": {"points": 0.25}}) == "continue"
        and early_decision({"curve": {"points": 0.251}, "instant": {"points": -1}}) == "stop"
        and early_decision({"curve": {"points": -1}, "instant": {"points": 0.251}}) == "stop",
        "bar_edges": bar_ok({"points_ci": [0, 0.5]}, "curve") and not bar_ok({"points_ci": [0, 0.501]}, "curve")
        and bar_ok({"points_ci": [0, 0.4]}, "instant") and not bar_ok({"points_ci": [0, 0.401]}, "instant")}
    ok = all(checks.values())
    print(f"[sm106-rule] selftest: SM92 as new: A curve {fp(a92['curve'])}, instant {fp(a92['instant'])} -> {f92['verdict']}; "
          f"the release as new: A curve {fp(arel['curve'])}, instant {fp(arel['instant'])} -> {frel['verdict']}; SM101 as new "
          f"-> {f101['verdict']} (curve untrained); early SM92 - release curve {fp(e1['curve'])} instant {fp(e1['instant'])} -> "
          f"{early_decision(e1)}; early release - SM92 -> {early_decision(e2)}; got {json.dumps(got)}; {json.dumps(checks)} -> "
          f"{'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


# recorded by the first selftest run on 2026-11-11 (before any SM106 GPU run); the rule code must reproduce them
# [curve point, instant point, curve CI upper end, instant CI upper end], % of the big model's show-time LogLoss
SELF = {"sm92_A": [0.259, -0.101, 0.463, -0.015], "sm92_verdict": "go", "release_A": [0.278, 0.202, 0.461, 0.235],
        "release_verdict": "go", "early_sm92_vs_release": [-0.019, -0.304], "early_release_vs_sm92": [0.019, 0.304]}


def early(out_dir=HERE, rows=EARLY_ROWS):
    r = report()
    users, bige = base(r)
    have = set(users)
    arm = r.load([(rows, EARLY_ARM + r.E2S, None)], have)
    ref = r.load([(rows, EARLY_REF + r.E2S, None)], have)
    if not (have <= set(arm) and have <= set(ref)):
        print(f"[sm106-rule] early pending: arm {len(have & set(arm))}, ref {len(have & set(ref))} of {len(users)} rule users")
        return 2
    c = r.compare(arm, ref, bige, users)
    out = {"rule": f"stop if epoch 27 (w64-c1) - SM92 epoch 26 (w64), both constant LR, show-time inputs, {len(users)} users, "
                   f"point estimate > +{EARLY_STOP}% (of the big model (1 leak-free epoch)'s show-time LogLoss) on the curve "
                   f"OR the instant head", "arm_label": EARLY_ARM, "ref_label": EARLY_REF, "users": len(users),
           "curve": c["curve"], "instant": c["instant"], "decision": early_decision(c)}
    (Path(out_dir) / "sm106-early.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = ["# The Tiny two-head model (SM106): early check", "",
          f"Decision: **{out['decision']}**. Epoch 27 of the Tiny two-head model (SM106) (card stream 1 layer, constant LR) "
          f"minus the show-time retrain (SM92)'s epoch 26 (constant LR, the cut's source), show-time inputs, {len(users)} "
          f"users, % of the big model (1 leak-free epoch)'s show-time LogLoss:", "",
          "| head | arm | ref | % [95% CI] | better / worse |", "|---|---|---|---|---|"]
    for h in HEADS:
        x = c[h]
        md.append(f"| {h} | {x['new']} | {x['ref']} | {fp(x)} | {x['users_better']} / {x['users_worse']} |")
    md += ["", f"Rule (pre-registered): stop if the point estimate is above +{EARLY_STOP}% on either head; otherwise continue "
           f"to epochs 28-29 and the cool-down.", ""]
    (Path(out_dir) / "sm106-early.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[sm106-rule] early {out['decision']}: epoch 27 - SM92 epoch 26 show-time curve {fp(c['curve'])}, instant "
          f"{fp(c['instant'])} ({len(users)} users)")
    return 0


def decide(out_dir=HERE, rows=ROWS, rows1500=ROWS1500, label=LBL):
    r = report()
    check_pins()
    users, bige = base(r)
    have = set(users)
    new = r.load([(rows, label + r.E2S, None)], have)
    if not have <= set(new):
        print(f"[sm106-rule] pending: new show-time rows {len(have & set(new))} of {len(users)} rule users")
        return 2
    f = final(r, new, users, bige)
    out = {"rule": "go = A: new - the big model (1 leak-free epoch), show-time inputs: curve 95% CI upper end <= +0.5% AND "
                   "instant 95% CI upper end <= +0.4%; B (new - SM92, new - SM101) report only", "label": label,
           "users": len(users), "big_e2s": {h: round(r.big_denominator(bige, users, k), 6) for k, h in r.HEADS.items()}} | f
    refs = {"sm92": r.load(r.MODELS["sm92"]["e2s"], have), "sm101": r.load([(S101G / "sm101-rows.jsonl", LBL101 + r.E2S, None)], have),
            "release": r.load(r.MODELS["release"]["e2s"], have)}

    def pair(a, b, denom=bige):
        us = sorted(set(users) & set(a) & set(b))
        return r.compare(a, b, denom, us) if us else None

    out["B_report"] = {"new_minus_sm92_e2s": pair(new, refs["sm92"]), "new_minus_sm101_e2s": pair(new, refs["sm101"]),
                       "new_minus_release_e2s": pair(new, refs["release"])}
    q = {}
    if r.POOL1500.exists() and Path(rows1500).exists():
        pool = set(json.loads(r.POOL1500.read_text())["users"])
        rel = r.load(r.MODELS["release"]["e2s1500"], pool)
        s101 = r.load([(S101G / "sm101-rows.jsonl", LBL101 + r.E2S, None), (S101G / "sm101-rows-1500.jsonl", LBL101 + r.E2S, None)], pool)
        nw = r.load([(rows, label + r.E2S, None), (rows1500, label + r.E2S, None)], pool)
        us = sorted(set(nw) & set(rel) & set(s101))
        q["users"] = len(us)
        if us:
            q["new_minus_release"] = r.compare(nw, rel, rel, us)
            q["new_minus_sm101"] = r.compare(nw, s101, rel, us)
            spec = json.loads(r.USERS1500.read_text())
            u732 = set(json.loads(r.U732.read_text())["users"]) | set(json.loads(r.ULONG.read_text())["users"])
            check = sorted(set(spec["users"] if isinstance(spec["users"], list) else []) & u732)
            a = r.load([(rows, label + r.E2S, None)], set(check)); b = r.load([(rows1500, label + r.E2S, None)], set(check))
            d = [max(abs(a[u][h][0] - b[u][h][0]) for h in r.HEADS) for u in check if u in a and u in b]
            q["row_check"] = {"users": len(d), "max_abs_logloss_diff": max(d) if d else None}
    out["pool_1500"] = q
    (Path(out_dir) / "sm106-decision.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    a = f["A_new_minus_big_e2s"]
    md = ["# The Tiny two-head model (SM106): decision", "",
          f"Verdict: **{f['verdict']}** ({f['users']} users of the 732 set, show-time inputs; % = % of the big model (1 "
          f"leak-free epoch)'s show-time LogLoss, curve {out['big_e2s']['curve']} / instant {out['big_e2s']['instant']}; "
          f"paired user bootstrap 2,000, seed 0, 95% CI).", "",
          "## A, the pruning bar: the Tiny two-head model (SM106) minus the big model (1 leak-free epoch)", "",
          "| head | new | big | % [95% CI] | bar (CI upper end) | pass |", "|---|---|---|---|---|---|"]
    for h in HEADS:
        md.append(f"| {h} | {a[h]['new']} | {a[h]['ref']} | {fp(a[h])} | <= +{BAR[h]}% | {'pass' if f['A_pass'][h] else 'fail'} |")
    md += ["", "## B, report only (show-time inputs, % of the big model (1 leak-free epoch))", "",
           "| new minus | head | new | ref | % [95% CI] | better / worse |", "|---|---|---|---|---|---|"]
    names = {"new_minus_sm92_e2s": ("the show-time retrain (SM92)", HEADS),
             "new_minus_sm101_e2s": ("the Tiny Instant-Only model (SM101)", ("instant",)),
             "new_minus_release_e2s": ("the release (SM63)", HEADS)}
    for k, (nm, hs) in names.items():
        x = out["B_report"][k]
        for h in hs if x else ():
            y = x[h]
            md.append(f"| {nm} | {h} | {y['new']} | {y['ref']} | {fp(y)} | {y['users_better']} / {y['users_worse']} |")
    md.append("")
    if q.get("new_minus_release"):
        md += [f"## 1,500 validation users, show-time inputs ({q['users']} users), % of the release's own LogLoss", "",
               "| comparison | head | new | ref | % [95% CI] | better / worse |", "|---|---|---|---|---|---|"]
        for k, nm, hs in (("new_minus_release", "new minus the release (SM63)", HEADS),
                          ("new_minus_sm101", "new minus the Tiny Instant-Only model (SM101)", ("instant",))):
            for h in hs:
                y = q[k][h]
                md.append(f"| {nm} | {h} | {y['new']} | {y['ref']} | {fpct(y)} | {y['users_better']} / {y['users_worse']} |")
        md += ["", f"Row check: {json.dumps(q['row_check'])}", ""]
    else:
        md += ["## 1,500 validation users", "", "Not scored yet.", ""]
    (Path(out_dir) / "sm106-decision.md").write_text("\n".join(md), encoding="utf-8")
    b92 = out["B_report"]["new_minus_sm92_e2s"]
    print(f"[sm106-rule] verdict {f['verdict']}: A new - big show-time curve {fp(a['curve'])} "
          f"({'pass' if f['A_pass']['curve'] else 'fail'}), instant {fp(a['instant'])} ({'pass' if f['A_pass']['instant'] else 'fail'}); "
          f"B new - SM92 curve {fp(b92['curve'])}, instant {fp(b92['instant'])}; 1,500 pool users {q.get('users', 0)}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "decide"
    raise SystemExit({"selftest": selftest, "early": early, "decide": decide}[cmd]())
