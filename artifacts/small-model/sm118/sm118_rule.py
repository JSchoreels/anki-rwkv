#!/usr/bin/env python
"""The show-time two-head cool-down for Small (SM118): pre-registered rules (fixed 2026-11-16 before any GPU run; CPU,
seconds).

New model: the longer cool-down recipe that passed for the Small Instant-Only and the Tiny Instant-Only models (the
longer big instant-only teacher (SM113), the longer cool-down for Tiny (SM115)) applied to the Small two-head model.
Start = the show-time retrain of the release (SM92)'s constant-LR epoch 26 (w64, read only; SM92's own cool-down
started there); one one_minus_sin decay 7e-4 -> 0 over 2 x 581.9M reviews, fresh Muon, show-time inputs, BOTH heads
trained (no curve loss weight), the show-time big model (SM91s2)'s show-time KD replay targets (curve and grade
columns, alpha 0.9, decay alpha 0.5), split into pass 1 (== SM92's cool-down except lr_resume and the targets;
== SM113's pass 1 except the start, the curve loss weight and the targets, sm118_pair.py p1) and pass 2 (pass 1's
end, the same rows in a new order, sm118_pair.py p2).
Labels: final `sm118-e26-dec2ep32k-t91s2/w8+s8` (gpu/sm118-rows.jsonl, show-time rows `@e2s-query`); early stop
`sm118-dec2ep-p1/w8+s8@e2s-query` (pass 1's final model) and `sm92-dec-u4056/w8+s8@e2s-query` (SM92's own cool-down
snapshot after 4,056 of 8,000 users, scored by this run's launcher; master sha256 pinned there) in
gpu/sm118-mid-rows.jsonl; SM92 on the 1,500 validation users `sm92-e26-dec1ep32k/w8+s8@e2s-query` in
gpu/sm92-rows-1500.jsonl (scored by this run: SM92 has no 1,500-pool rows). References are read from their own
committed rows (sha256 pinned), never rescored: SM92 and SM91s2 (this branch's copies), SM98 and SM113 (instant
read-outs), SM91s2's 1,500-pool rows (sm99-rows-1500.jsonl), the release from the show-time report.

Loaders and the paired bootstrap are the show-time report's (`artifacts/small-model/showtime/showtime_report.py` on
main, sha256 pinned, the same pin as SM96 ... SM115): 2,000 resamples, seed 0, 95% CI, review-weighted LogLoss, 722
users of the 732 set. Unit ("%"): % of the big model (1 leak-free epoch)'s normal-input LogLoss of the same head on
the same users (curve 0.298966, instant 0.275757).

Per head h (curve, instant), x_h = new - SM92 on show-time inputs:
  gain(h)       = the 95% CI of x_h entirely below 0 AND x_h <= -0.05%
  regression(h) = x_h >= +0.05% OR the 95% CI of x_h entirely above 0

EARLY STOP (`mid`, after pass 1 = half of the reviews, learning rate 29.29% of its peak):
  y_h = SM118's pass-1 final model - SM92's own cool-down snapshot after 4,056 of 8,000 users (288,456,043 of
  581,942,724 reviews, learning rate 29.77% of its peak at its last update: the nearest snapshot to the matching point
  of SM92's one-epoch schedule), both heads, show-time inputs, 722 users.
  continue  if (y_curve < 0 OR y_instant < 0) AND y_curve < +0.05% AND y_instant < +0.05% (point estimates)
  stop      otherwise: verdict "no go (stopped after pass 1)"; saves ~8 GPU h.
  Read-out (no verdict): pass 1 - SM92's final model.
FINAL (`decide`):
  go     = gain on at least one head AND regression on neither head;
  no go  = otherwise.
  Read-out with it (no verdict), both heads: the share of the teacher gap closed, (SM118 - SM92) / (SM91s2 - SM92);
  SM118 - SM91s2 (the teacher), - the release (SM63) (show-time); SM118 - SM92 on normal inputs; SM118 - the big
  model (1 leak-free epoch) on normal inputs and on show-time inputs (the smaller-model bar: curve <= +0.5%, instant
  <= +0.4%, point estimates as the 10-29 pruning bar, CIs reported); the show-time cost; the early-stop pairs; instant
  only: SM118 - the Small Instant-Only model (SM98), - the longer big instant-only teacher (SM113). The 1,500
  validation users on show-time inputs (% of the release's own LogLoss): SM118 - SM92 (by id part), - SM91s2, - the
  release, and the row checks.

  python sm118_rule.py selftest   -> the rules' code on known rows (exit 0 if all equal)
  python sm118_rule.py mid        -> sm118-mid.{json,md}; exit 2 while rows are missing
  python sm118_rule.py decide     -> sm118-decision.{json,md}; exit 2 while the final model's 722-user rows are missing
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
REPORT_SHA = "3cdfed03afd40734d4b1931c244e8c1e73453d281a40c3843e80cd310d70dd9d"   # main 9031ed51 (same pin as SM96 ... SM115)
ROWS, ROWS1500, MIDROWS, ROWS92_1500 = (G / "sm118-rows.jsonl", G / "sm118-rows-1500.jsonl", G / "sm118-mid-rows.jsonl",
                                        G / "sm92-rows-1500.jsonl")
LBL = "sm118-e26-dec2ep32k-t91s2/w8+s8"
MID_NEW, MID_REF = "sm118-dec2ep-p1/w8+s8", "sm92-dec-u4056/w8+s8"
SMG = WT / "artifacts/small-model"
R92, R91 = SMG / "sm92/gpu/sm92-inputs.jsonl", SMG / "sm91s2/gpu/sm91s2-inputs.jsonl"
R91_1500 = SMG / "sm99/gpu/sm99-rows-1500.jsonl"
R98, R113 = SMG / "sm98/gpu/sm98-rows.jsonl", SMG / "sm113/gpu/sm113-rows.jsonl"
PINS = {R92: "5c714ea0f1c328a053e21c6ac0cbabfb0397ef68bbda6d861201d61ad475adff",      # == wt/sm92's own copy
        R91: "e156b6fd907857b22f96d64035c06417b063c88d416a959bc91231ed85075c80",
        R91_1500: "81e7bab8e0100a417b42e64f4fd2ea00fefdd16fce877dec8dc7977807e70b3d",  # same pin as SM115
        R98: "32c7cecdbc3e1729505cd7f4c61a1484180be8a6067e6475af604e9a165f2330",       # same pin as SM115
        R113: "af104a0774b1aa7da7b26bc48ac667a9991d86efd8dda4ac8de7bb51a3fe281a"}      # same pin as SM115
LBL92, LBL91 = "sm92-e26-dec1ep32k/w8+s8", "sm91s2-big-showtime/w8+s8"
LBL98, LBL113 = "sm98-inst-e35st-st-dec1ep32k/w8+s8", "sm113-inst-e35st-st-dec2ep32k-t99/w8+s8"
MIN_EFFECT = 0.05
HEADS = ("curve", "instant")
# the show-time retrain of the release (SM92)'s published rule pair (sm92-decision.md): SM92 - the release, show-time
SM92_MINUS_RELEASE = {"curve": (-0.019, [-0.092, 0.085]), "instant": (-0.305, [-0.405, -0.232])}
# sm92-decision.md: SM92 - the show-time big model (SM91s2), show-time: +0.315 [+0.149, +0.491] / +0.251 [+0.222, +0.280]
SM91_MINUS_SM92 = {"curve": (-0.315, [-0.491, -0.149]), "instant": (-0.251, [-0.28, -0.222])}
NAMES = {"sm92": "the show-time retrain of the release (SM92)", "sm91s2": "the show-time big model (SM91s2)",
         "release": "the release (SM63)", "big": "the big model (1 leak-free epoch)",
         "sm98": "the Small Instant-Only model (SM98)", "sm113": "the longer big instant-only teacher (SM113)",
         "sm118": "the show-time two-head cool-down for Small (SM118)"}


def report():
    got = hashlib.sha256(REPORT.read_bytes()).hexdigest()
    if got != REPORT_SHA:
        raise SystemExit(f"[sm118-rule] showtime_report.py changed (sha256 {got[:12]} != {REPORT_SHA[:12]}): refusing")
    spec = importlib.util.spec_from_file_location("showtime_report", REPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_pins():
    for path, want in PINS.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"[sm118-rule] reference rows {path.name} changed (sha256 {got[:12]} != {str(want)[:12]}): refusing")


def refs(r, rows92_1500=ROWS92_1500):
    return {"sm92": {"e2s": [(R92, LBL92 + r.E2S, None)], "normal": [(R92, LBL92, None)],
                     "e2s1500": [(R92, LBL92 + r.E2S, None), (rows92_1500, LBL92 + r.E2S, None)]},
            "sm91s2": {"e2s": [(R91, LBL91 + r.E2S, None)], "normal": [(R91, LBL91, None)],
                       "e2s1500": [(R91, LBL91 + r.E2S, None), (R91_1500, LBL91 + r.E2S, None)]},
            "sm98": {"e2s": [(R98, LBL98 + r.E2S, None)]}, "sm113": {"e2s": [(R113, LBL113 + r.E2S, None)]},
            "release": r.MODELS["release"]}


def own(rows, rows1500, label, r):
    return {"e2s": [(rows, label + r.E2S, None)], "normal": [(rows, label, None)],
            "e2s1500": [(rows, label + r.E2S, None), (rows1500, label + r.E2S, None)]}


def fp(x):
    return f"{x['points']:+.3f}% [{x['points_ci'][0]:+.3f}, {x['points_ci'][1]:+.3f}]"


def fpct(x):
    return f"{x['pct_of_ref']:+.3f}% [{x['pct_of_ref_ci'][0]:+.3f}, {x['pct_of_ref_ci'][1]:+.3f}]"


def gain(x):
    return x["points_ci"][1] < 0 and x["points"] <= -MIN_EFFECT


def regression(x):
    return x["points"] >= MIN_EFFECT or x["points_ci"][0] > 0


def verdict(c):
    """c = compare() output (keys 'curve', 'instant'): go if one head gains and no head regresses."""
    gains = [h for h in HEADS if gain(c[h])]
    regs = [h for h in HEADS if regression(c[h])]
    return {"verdict": "go" if gains and not regs else "no go", "gain_heads": gains, "regression_heads": regs}


def mid_decision(c):
    y = {h: c[h]["points"] for h in HEADS}
    return "continue" if (y["curve"] < 0 or y["instant"] < 0) and all(v < MIN_EFFECT for v in y.values()) else "stop"


def primary(r, new_e2s, users, big):
    s92 = r.load(refs(r)["sm92"]["e2s"], set(users))
    us = sorted(set(users) & set(new_e2s) & set(s92))
    c = r.compare(new_e2s, s92, big, us)
    return {"users": len(us), "new_minus_sm92_e2s": c, **verdict(c)}


def gap_closed(x_new, x_teacher):
    """per head: (new - SM92) / (SM91s2 - SM92), points on the same users."""
    return {h: (round(x_new[h]["points"] / x_teacher[h]["points"], 3) if x_teacher[h]["points"] else None) for h in HEADS}


def scaled(rows, factor):
    return {u: {h: (v[h][0] * factor.get(h, 1.0), v[h][1]) for h in v} for u, v in rows.items()}


def selftest():
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    src = refs(r)
    n92, n91, nrel = (r.load(src[k]["e2s"], have) for k in ("sm92", "sm91s2", "release"))
    p92, p91, prel = (primary(r, x, users, big) for x in (n92, n91, nrel))
    x92rel = r.compare(n92, nrel, big, sorted(have & set(n92) & set(nrel)))   # SM92's own rule pair through this code
    # synthetic: 0.04% better instant only; 0.1% better instant with the curve 0.06% worse; 0.1% better curve only
    syn = {"inst_0p04": primary(r, scaled(n92, {"ahead": 1.0, "instant": 0.9996}), users, big),
           "inst_0p1_curve_worse_0p06": primary(r, scaled(n92, {"ahead": 1.0006, "instant": 0.999}), users, big),
           "curve_0p1_only": primary(r, scaled(n92, {"ahead": 0.999, "instant": 1.0}), users, big)}
    pin = lambda x: {h: (x[h]["points"], x[h]["points_ci"]) for h in HEADS}  # noqa: E731
    checks = {
        "sm92_against_itself_zero_no_go_mid_stop": p92["users"] == 722 and all(p92["new_minus_sm92_e2s"][h]["points"] == 0 for h in HEADS)
        and p92["verdict"] == "no go" and mid_decision(p92["new_minus_sm92_e2s"]) == "stop",
        "sm92_minus_release_published_pair": pin(x92rel) == SM92_MINUS_RELEASE and x92rel["users"] == 722
        and verdict(x92rel) == {"verdict": "go", "gain_heads": ["instant"], "regression_heads": []},
        "release_minus_sm92_no_go_instant_regression": prel["verdict"] == "no go" and prel["regression_heads"] == ["instant"]
        and mid_decision(prel["new_minus_sm92_e2s"]) == "stop",
        "sm91s2_minus_sm92_published_go_both_gap_one": pin(p91["new_minus_sm92_e2s"]) == SM91_MINUS_SM92 and p91["verdict"] == "go"
        and p91["gain_heads"] == ["curve", "instant"] and gap_closed(p91["new_minus_sm92_e2s"], p91["new_minus_sm92_e2s"]) == {"curve": 1.0, "instant": 1.0}
        and mid_decision(p91["new_minus_sm92_e2s"]) == "continue",
        "synthetic_instant_0p04_no_go_continue": syn["inst_0p04"]["verdict"] == "no go"
        and -0.045 < syn["inst_0p04"]["new_minus_sm92_e2s"]["instant"]["points"] < -0.035
        and syn["inst_0p04"]["new_minus_sm92_e2s"]["instant"]["points_ci"][1] < 0
        and syn["inst_0p04"]["new_minus_sm92_e2s"]["curve"]["points"] == 0 and mid_decision(syn["inst_0p04"]["new_minus_sm92_e2s"]) == "continue",
        "synthetic_instant_gain_curve_0p06_worse_no_go_stop": syn["inst_0p1_curve_worse_0p06"]["verdict"] == "no go"
        and syn["inst_0p1_curve_worse_0p06"]["gain_heads"] == ["instant"] and "curve" in syn["inst_0p1_curve_worse_0p06"]["regression_heads"]
        and mid_decision(syn["inst_0p1_curve_worse_0p06"]["new_minus_sm92_e2s"]) == "stop",
        "synthetic_curve_gain_only_go": syn["curve_0p1_only"]["verdict"] == "go" and syn["curve_0p1_only"]["gain_heads"] == ["curve"],
        "edges": verdict({"curve": {"points": 0.0, "points_ci": [-0.1, 0.1]}, "instant": {"points": -0.05, "points_ci": [-0.07, -0.001]}})["verdict"] == "go"
        and verdict({"curve": {"points": 0.0, "points_ci": [-0.1, 0.1]}, "instant": {"points": -0.049, "points_ci": [-0.07, -0.03]}})["verdict"] == "no go"
        and verdict({"curve": {"points": 0.0, "points_ci": [-0.1, 0.1]}, "instant": {"points": -0.06, "points_ci": [-0.12, 0.0]}})["verdict"] == "no go"
        and verdict({"curve": {"points": 0.049, "points_ci": [-0.05, 0.15]}, "instant": {"points": -0.1, "points_ci": [-0.12, -0.08]}})["verdict"] == "go"
        and verdict({"curve": {"points": 0.05, "points_ci": [-0.05, 0.15]}, "instant": {"points": -0.1, "points_ci": [-0.12, -0.08]}})["verdict"] == "no go"
        and verdict({"curve": {"points": 0.02, "points_ci": [0.001, 0.04]}, "instant": {"points": -0.1, "points_ci": [-0.12, -0.08]}})["verdict"] == "no go"
        and mid_decision({"curve": {"points": 0.049}, "instant": {"points": -0.001}}) == "continue"
        and mid_decision({"curve": {"points": 0.05}, "instant": {"points": -0.2}}) == "stop"
        and mid_decision({"curve": {"points": 0.0}, "instant": {"points": 0.0}}) == "stop"
        and mid_decision({"curve": {"points": -0.01}, "instant": {"points": 0.049}}) == "continue"}
    checks["end_to_end_mid_and_decide"] = end_to_end(r)
    ok = all(checks.values())
    print(f"[sm118-rule] selftest: SM92 - SM92 {fp(p92['new_minus_sm92_e2s']['curve'])} / {fp(p92['new_minus_sm92_e2s']['instant'])} -> "
          f"{p92['verdict']}; SM92 - release {fp(x92rel['curve'])} / {fp(x92rel['instant'])} -> {verdict(x92rel)['verdict']}; "
          f"SM91s2 - SM92 {fp(p91['new_minus_sm92_e2s']['curve'])} / {fp(p91['new_minus_sm92_e2s']['instant'])} -> {p91['verdict']}; "
          f"{json.dumps(checks)} -> {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def end_to_end(r):
    """mid + decide on SM92's own rows relabelled (as pass 1, as the early-stop reference, as the final model, and SM92's
    732 rows as its 1,500-pool rows): mid must say stop (0 is not < 0); decide must read 0 -> no go, gap closed 0, with
    every table filled."""
    import io
    import tempfile
    from contextlib import redirect_stdout

    def relabel(src, dst, mapping):
        with open(src, encoding="utf-8") as f, open(dst, "a", encoding="utf-8") as g:
            for line in f:
                x = json.loads(line)
                lb = x.get("label", "")
                for a, b in mapping.items():
                    if lb == a or lb == a + r.E2S:
                        x["label"] = lb.replace(a, b)
                        g.write(json.dumps(x) + "\n")
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        rows, rows1500, mrows, r92 = t / "rows.jsonl", t / "rows-1500.jsonl", t / "mid.jsonl", t / "sm92-1500.jsonl"
        relabel(R92, rows, {LBL92: LBL})
        relabel(R92, rows1500, {LBL92: LBL})
        relabel(R92, r92, {LBL92: LBL92})
        relabel(R92, mrows, {LBL92: MID_NEW})
        relabel(R92, mrows, {LBL92: MID_REF})
        with redirect_stdout(io.StringIO()):
            a = mid(t, mrows)
            b = decide(t, rows, rows1500, mrows, r92)
        m = json.loads((t / "sm118-mid.json").read_text())
        d = json.loads((t / "sm118-decision.json").read_text())
        q = d["pool_1500"]
        return (a == 0 and b == 0 and m["decision"] == "stop" and all(m["vs_u4056"][h]["points"] == 0 for h in HEADS)
                and all(m["vs_sm92_final"][h]["points"] == 0 for h in HEADS)
                and d["verdict"] == "no go" and all(d["new_minus_sm92_e2s"][h]["points"] == 0 for h in HEADS)
                and d["gap_closed"] == {"curve": 0.0, "instant": 0.0}
                and all(v is not None for v in d["secondary_732"].values())
                and {"new_minus_sm91s2_e2s", "new_minus_release_e2s", "new_minus_sm92_normal", "new_normal_minus_big_normal",
                     "new_e2s_minus_big_e2s", "new_e2s_minus_new_normal", "new_minus_sm98_e2s", "new_minus_sm113_e2s",
                     "mid_p1_minus_sm92_u4056_e2s", "mid_p1_minus_sm92_final_e2s"} <= set(d["secondary_732"])
                and q.get("users", 0) >= 700 and q["new_minus_sm92"]["instant"]["pct_of_ref"] == 0
                and q.get("new_minus_sm91s2") is not None and q.get("new_minus_release") is not None
                and (t / "sm118-decision.md").exists() and (t / "sm118-mid.md").exists())


def mid(out_dir=HERE, mrows=MIDROWS):
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    a = r.load([(mrows, MID_NEW + r.E2S, None)], have)
    b = r.load([(mrows, MID_REF + r.E2S, None)], have)
    c = r.load(refs(r)["sm92"]["e2s"], have)
    if not (have <= set(a) and have <= set(b) and have <= set(c)):
        print(f"[sm118-rule] mid pending: rows pass 1 {len(have & set(a))}, SM92 u4056 {len(have & set(b))}, "
              f"SM92 final {len(have & set(c))} of {len(users)} rule users")
        return 2
    x = r.compare(a, b, big, sorted(have))
    y = r.compare(a, c, big, sorted(have))
    out = {"rule": f"continue (pass 2) if SM118's pass-1 final model - SM92's cool-down snapshot after 4,056 users (show-time "
                   f"inputs, {len(users)} users) has a point estimate < 0 on at least one head and < +0.05% on both; stop otherwise",
           "users": len(users), "labels": [MID_NEW, MID_REF, LBL92], "vs_u4056": x, "vs_sm92_final": y, "decision": mid_decision(x)}
    (Path(out_dir) / "sm118-mid.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = ["# The show-time two-head cool-down for Small (SM118): early stop after pass 1", "",
          f"Decision: **{out['decision']}**. The show-time two-head cool-down for Small (SM118)'s model after pass 1 of 2 "
          f"(learning rate 29.29% of its peak) minus the show-time retrain of the release (SM92)'s own cool-down snapshot after "
          f"4,056 of 8,000 users (learning rate 29.77% of its peak), show-time inputs, {len(users)} users (% of the big model "
          f"(1 leak-free epoch) normal-input LogLoss of each head):", "",
          "| head | pass 1 | SM92 snapshot | % [95% CI] | better / worse |", "|---|---|---|---|---|"]
    for h in HEADS:
        md.append(f"| {h} | {x[h]['new']} | {x[h]['ref']} | {fp(x[h])} | {x[h]['users_better']} / {x[h]['users_worse']} |")
    md += ["", f"Read-out (no verdict): pass 1 minus the show-time retrain of the release (SM92)'s final model: curve {fp(y['curve'])}, "
           f"instant {fp(y['instant'])}.", "",
           "Rule (pre-registered): continue with pass 2 if at least one head's point estimate is below 0 and neither is +0.05% "
           "or more; otherwise stop (verdict no go).", ""]
    (Path(out_dir) / "sm118-mid.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[sm118-rule] mid {out['decision']}: SM118 pass 1 - SM92 u4056 show-time curve {fp(x['curve'])}, instant {fp(x['instant'])}; "
          f"- SM92 final curve {fp(y['curve'])}, instant {fp(y['instant'])} ({len(users)} users)")
    return 0


def decide(out_dir=HERE, rows=ROWS, rows1500=ROWS1500, mrows=MIDROWS, rows92_1500=ROWS92_1500):
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    mine = own(rows, rows1500, LBL, r)
    new = {v: r.load(mine[v], have) for v in ("e2s", "normal")}
    if not have <= set(new["e2s"]):
        print(f"[sm118-rule] pending: {LBL} show-time rows {len(have & set(new['e2s']))} of {len(users)} rule users")
        return 2
    out = {"rule": "SM118 - SM92, show-time inputs, 722 users, per head: gain = 95% CI below 0 and <= -0.05%; regression = "
                   ">= +0.05% or 95% CI above 0; go = a gain on at least one head and a regression on neither",
           "label": LBL, "users": len(users), "big_denominator": {h: round(r.big_denominator(big, users, k), 6) for k, h in r.HEADS.items()}}
    p = primary(r, new["e2s"], users, big)
    out |= p
    src = refs(r, rows92_1500)
    ref = {k: {v: r.load(s[v], have) for v in ("e2s", "normal") if v in s} for k, s in src.items()}

    def pair(a, b):
        us = sorted(set(users) & set(a) & set(b))
        return r.compare(a, b, big, us) if us else None

    teacher = pair(ref["sm91s2"]["e2s"], ref["sm92"]["e2s"])
    out["sm91s2_minus_sm92_e2s"] = teacher
    out["gap_closed"] = gap_closed(p["new_minus_sm92_e2s"], teacher)
    big_e2s = r.load(r.MODELS["big"]["e2s"], have)
    sec = {"new_minus_sm91s2_e2s": pair(new["e2s"], ref["sm91s2"]["e2s"]),
           "new_minus_release_e2s": pair(new["e2s"], ref["release"]["e2s"]),
           "new_e2s_minus_big_e2s": pair(new["e2s"], big_e2s),
           "new_minus_sm98_e2s": pair(new["e2s"], ref["sm98"]["e2s"]),
           "new_minus_sm113_e2s": pair(new["e2s"], ref["sm113"]["e2s"])}
    if have <= set(new["normal"]):
        sec |= {"new_minus_sm92_normal": pair(new["normal"], ref["sm92"]["normal"]),
                "new_normal_minus_big_normal": pair(new["normal"], big),
                "new_e2s_minus_new_normal": pair(new["e2s"], new["normal"])}
    ma = r.load([(mrows, MID_NEW + r.E2S, None)], have)
    mb = r.load([(mrows, MID_REF + r.E2S, None)], have)
    if have <= set(ma) and have <= set(mb):
        sec["mid_p1_minus_sm92_u4056_e2s"] = pair(ma, mb)
    if have <= set(ma):
        sec["mid_p1_minus_sm92_final_e2s"] = pair(ma, ref["sm92"]["e2s"])
    out["secondary_732"] = sec
    q = {}
    if r.POOL1500.exists():
        pool = set(json.loads(r.POOL1500.read_text())["users"])
        rel = r.load(r.MODELS["release"]["e2s1500"], pool)
        o92, o91 = r.load(src["sm92"]["e2s1500"], pool), r.load(src["sm91s2"]["e2s1500"], pool)
        nw = r.load(mine["e2s1500"], pool)
        us = sorted(set(nw) & set(rel) & set(o92))
        q["users"] = len(us)
        if us and Path(rows1500).exists() and Path(rows92_1500).exists():
            q["new_minus_sm92"] = r.compare(nw, o92, rel, us)
            q["new_minus_release"] = r.compare(nw, rel, rel, us)
            u91 = sorted(set(us) & set(o91))
            q["users_with_sm91s2"] = len(u91)
            q["new_minus_sm91s2"] = r.compare(nw, o91, rel, u91) if u91 else None
            lo, hi = [u for u in us if u < 5000], [u for u in us if u >= 5000]
            q["by_part_new_minus_sm92"] = {"ids < 5000 (the 732 set)": r.compare(nw, o92, rel, lo) if lo else None,
                                           "ids >= 5000 (training users of the big models)": r.compare(nw, o92, rel, hi) if hi else None}
            spec = json.loads(r.USERS1500.read_text())
            u732 = set(json.loads(r.U732.read_text())["users"]) | set(json.loads(r.ULONG.read_text())["users"])
            check = sorted(set(spec["users"] if isinstance(spec["users"], list) else []) & u732)
            rc = {}
            for name, (fa, fb, lb) in {"new": (rows, rows1500, LBL), "sm92": (R92, rows92_1500, LBL92)}.items():
                a = r.load([(fa, lb + r.E2S, None)], set(check)); b = r.load([(fb, lb + r.E2S, None)], set(check))
                d = [max(abs(a[u][h][0] - b[u][h][0]) for h in r.HEADS) for u in check if u in a and u in b]
                rc[name] = {"users": len(d), "max_abs_logloss_diff": max(d) if d else None}
            q["row_check"] = rc
    out["pool_1500"] = q
    (Path(out_dir) / "sm118-decision.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    names = {"new_minus_sm91s2_e2s": (NAMES["sm91s2"] + " (the teacher)", "show-time"),
             "new_minus_release_e2s": (NAMES["release"], "show-time"),
             "new_e2s_minus_big_e2s": (NAMES["big"] + " (smaller-model bar, show-time: curve <= +0.5%, instant <= +0.4%)", "show-time"),
             "new_minus_sm98_e2s": (NAMES["sm98"] + " (instant only)", "show-time"),
             "new_minus_sm113_e2s": (NAMES["sm113"] + " (instant only)", "show-time"),
             "new_minus_sm92_normal": (NAMES["sm92"], "normal"),
             "new_normal_minus_big_normal": (NAMES["big"] + " (the smaller-model bar: curve <= +0.5%, instant <= +0.4%)", "normal"),
             "new_e2s_minus_new_normal": ("the new model on normal inputs (its show-time cost)", "show-time vs normal"),
             "mid_p1_minus_sm92_u4056_e2s": ("early stop: SM118 after pass 1 minus SM92's snapshot after 4,056 users", "show-time"),
             "mid_p1_minus_sm92_final_e2s": ("early stop read-out: SM118 after pass 1 minus SM92's final model", "show-time")}
    x, t = p["new_minus_sm92_e2s"], teacher
    md = ["# The show-time two-head cool-down for Small (SM118): decision", "",
          f"Verdict: **{p['verdict']}** (gain on: {', '.join(p['gain_heads']) or 'no head'}; regression on: "
          f"{', '.join(p['regression_heads']) or 'no head'}; {len(users)} users of the 732 set; % of the big model (1 leak-free "
          f"epoch) normal-input LogLoss of each head, curve {out['big_denominator']['curve']}, instant "
          f"{out['big_denominator']['instant']}; paired user bootstrap 2,000, seed 0, 95% CI).", "",
          "Rule (pre-registered): per head, gain = CI below 0 and at most -0.05%; regression = +0.05% or more, or CI above 0; "
          "go = a gain on at least one head and a regression on neither.", "",
          "| the show-time two-head cool-down for Small (SM118) minus the show-time retrain of the release (SM92), show-time | new | ref | % [95% CI] | better / worse | teacher gap closed |",
          "|---|---|---|---|---|---|"]
    for h in HEADS:
        md.append(f"| {h} | {x[h]['new']} | {x[h]['ref']} | {fp(x[h])} | {x[h]['users_better']} / {x[h]['users_worse']} | "
                  f"{out['gap_closed'][h]} (the show-time big model (SM91s2) minus SM92: {fp(t[h])}) |")
    md += ["", "## Read-out (732 set, %)", "", "| minus | inputs | curve % [95% CI] | instant % [95% CI] | instant better / worse |",
           "|---|---|---|---|---|"]
    for k, v in sec.items():
        if v:
            cv = "(not trained)" if k in ("new_minus_sm98_e2s", "new_minus_sm113_e2s") else fp(v["curve"])
            md.append(f"| {names[k][0]} | {names[k][1]} | {cv} | {fp(v['instant'])} | {v['instant']['users_better']} / {v['instant']['users_worse']} |")
    md.append("")
    if q.get("new_minus_sm92"):
        md += [f"## 1,500 validation users, show-time inputs ({q['users']} users), % of the release's own LogLoss", "",
               "| comparison | users | curve % [95% CI] | instant % [95% CI] |", "|---|---|---|---|"]
        rowsq = [("new minus " + NAMES["sm92"], q["new_minus_sm92"]), ("new minus " + NAMES["sm91s2"], q.get("new_minus_sm91s2")),
                 ("new minus " + NAMES["release"], q["new_minus_release"])]
        rowsq += [(f"new minus {NAMES['sm92']}, {part}", v) for part, v in q["by_part_new_minus_sm92"].items()]
        for nm, v in rowsq:
            if v:
                md.append(f"| {nm} | {v['users']} | {fpct(v['curve'])} | {fpct(v['instant'])} |")
        md += ["", f"Users with rows of the show-time big model (SM91s2): {q.get('users_with_sm91s2')}. Its ids >= 5000 are its own "
               f"training users, so new minus SM91s2 there is not a held-out comparison. Row checks (732 rows vs 1,500-pool rows "
               f"of the same users): {json.dumps(q['row_check'])}", ""]
    else:
        md += ["## 1,500 validation users", "", "Not scored yet.", ""]
    (Path(out_dir) / "sm118-decision.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[sm118-rule] verdict {p['verdict']} (gain {p['gain_heads']}, regression {p['regression_heads']}): SM118 - SM92 show-time "
          f"curve {fp(x['curve'])}, instant {fp(x['instant'])}; gap closed {out['gap_closed']}; SM118 - SM91s2 curve "
          f"{fp(sec['new_minus_sm91s2_e2s']['curve'])}, instant {fp(sec['new_minus_sm91s2_e2s']['instant'])}; 1,500 pool users {q.get('users', 0)}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "decide"
    raise SystemExit({"selftest": selftest, "mid": mid, "decide": decide}[cmd]())
