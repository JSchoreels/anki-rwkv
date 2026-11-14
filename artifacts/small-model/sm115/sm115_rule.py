#!/usr/bin/env python
"""The longer cool-down for Tiny (SM115): pre-registered rules (fixed 2026-11-14 before any GPU run; CPU, seconds).

New model: the longer big instant-only teacher (SM113)'s recipe applied to Tiny. Start = the Tiny Instant-Only model
(SM101)'s constant-LR epoch 38 (w64-c1, read only; SM101's own cool-down started there); one one_minus_sin decay
7e-4 -> 0 over 2 x 581.9M reviews, fresh Muon, show-time inputs, curve loss weight 0, the show-time instant-only big
model (SM99)'s show-time KD replay targets (decay alpha 0.5), split into pass 1 (== SM101's cool-down except lr_resume
and the targets; == SM113's pass 1 except the start and the architecture, sm115_pair.py p1) and pass 2 (pass 1's end,
the same rows in a new order, sm115_pair.py p2).
Labels: final `sm115-c1-e38st-st-dec2ep32k-t99/w8+s8` (show-time rows `@e2s-query`, gpu/sm115-rows.jsonl); early stop
`sm115-dec2ep-p1/w8+s8@e2s-query` (pass 1's final model) and `sm101-dec-u4094/w8+s8@e2s-query` (SM101's own cool-down
snapshot after 4,094 of 8,000 users, scored by this run's launcher, snapshot master sha256 pinned there) in
gpu/sm115-mid-rows.jsonl. References read from their own committed rows (sha256 pinned), never rescored: SM101, SM98,
SM99 rows of this branch; SM113's final and pass-1 rows (branch sm113, dc4658e9).

Loaders and the paired bootstrap are the show-time report's (`artifacts/small-model/showtime/showtime_report.py` on main,
sha256 pinned, the same pin as SM96 / SM98 / SM99 / SM101 / SM110 / SM113): 2,000 resamples, seed 0, 95% CI,
review-weighted LogLoss, the instant head, 722 users of the 732 set. Unit: % of the big model (1 leak-free epoch)'s
normal-input instant LogLoss on the same users (0.275757). The curve head is untrained (weight 0): never in a verdict.

EARLY STOP (`mid`, after pass 1 = half of the reviews, learning rate 29.29% of its peak):
  x = SM115's pass-1 final model - SM101's own cool-down snapshot after 4,094 of 8,000 users (290,914,508 of 581,942,724
  reviews, learning rate 29.31% of its peak at its last update: the matching point of SM101's one-epoch schedule),
  instant, show-time inputs, 722 users.
  continue  if the point estimate x < 0: pass 2 runs.
  stop      otherwise (x >= 0): verdict "no go (stopped after pass 1)"; saves ~8 GPU h.
  Read-out with it (no verdict): pass 1 - SM113's pass 1 (Small, the same recipe at the same point).
FINAL (`decide`):
  Primary (the verdict): SM115 - SM101, instant, show-time inputs, 722 users:
    go     = the 95% CI entirely below 0 AND points <= -0.05;
    no go  = otherwise.
  Read-out with it (no verdict): SM115 - SM113 (Tiny vs Small, the same recipe); the share of the teacher gap closed,
  (SM115 - SM101) / (SM99 - SM101); SM115 - SM99, - SM98, - SM92, - the release (SM63) (show-time); SM115 - SM101
  (normal); SM115 normal - the big model (1 leak-free epoch) normal (the smaller-model instant bar: CI upper end <=
  +0.40%); the show-time cost; the early-stop pairs. The 1,500 validation users on show-time inputs (% of the release's
  own LogLoss): SM115 - SM101, - SM113, - SM99, - the release, by id part, and the row check.

  python sm115_rule.py selftest   -> the rules' code on known rows (exit 0 if all equal)
  python sm115_rule.py mid        -> sm115-mid.{json,md}; exit 2 while rows are missing
  python sm115_rule.py decide     -> sm115-decision.{json,md}; exit 2 while the final model's 722-user rows are missing
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
REPORT_SHA = "3cdfed03afd40734d4b1931c244e8c1e73453d281a40c3843e80cd310d70dd9d"   # main 9031ed51 (same pin as SM96 ... SM113)
ROWS, ROWS1500, MIDROWS = G / "sm115-rows.jsonl", G / "sm115-rows-1500.jsonl", G / "sm115-mid-rows.jsonl"
LBL = "sm115-c1-e38st-st-dec2ep32k-t99/w8+s8"
MID_NEW, MID_REF, MID_113 = "sm115-dec2ep-p1/w8+s8", "sm101-dec-u4094/w8+s8", "sm113-dec2ep-p1/w8+s8"
S101G, S98G, S99G, S113G = (WT / f"artifacts/small-model/{x}/gpu" for x in ("sm101", "sm98", "sm99", "sm113"))
R113_MID = S113G / "sm113-mid-rows.jsonl"
PINS = {S101G / "sm101-rows.jsonl": "19b6dc1e85e8a673f17533e5422655580072562b7c6925123560421412a9bca2",      # branch sm101 after its rewrite (== wt/sm101)
        S101G / "sm101-rows-1500.jsonl": "7d4df7753fe2f685e08b948740c686f4a0ea31063620634e68024838bdc6af76",
        S98G / "sm98-rows.jsonl": "32c7cecdbc3e1729505cd7f4c61a1484180be8a6067e6475af604e9a165f2330",       # after f561e1c0 (provenance_commit only)
        S98G / "sm98-rows-1500.jsonl": "8a1dba01d40139948822735578df22c04b3400fbed1d44bf088d3d3439e702af",
        S99G / "sm99-rows.jsonl": "df4c48e0a0c3e95b99a64629fb262ccb0805194fd0702c1e46d6adf7737dacea",       # after f561e1c0 (provenance_commit only)
        S99G / "sm99-rows-1500.jsonl": "81e7bab8e0100a417b42e64f4fd2ea00fefdd16fce877dec8dc7977807e70b3d",
        S113G / "sm113-rows.jsonl": "af104a0774b1aa7da7b26bc48ac667a9991d86efd8dda4ac8de7bb51a3fe281a",        # branch sm113 dc4658e9
        S113G / "sm113-rows-1500.jsonl": "3bdab98320112d6ceaf5c23fb5dfbdd96ec510a0d74231d304bc187d7bc16b4c",
        R113_MID: "cb82ff46598f262c518881f955b2020ad5002dcb28f121f1416b37530cacee11"}
LBL101, LBL98, LBL99 = "sm101-c1-e38st-st-dec1ep32k/w8+s8", "sm98-inst-e35st-st-dec1ep32k/w8+s8", "sm99-big-inst-e2/w8+s8"
LBL113 = "sm113-inst-e35st-st-dec2ep32k-t99/w8+s8"
MIN_EFFECT = 0.05
SM98_MINUS_SM101 = (-0.192, [-0.215, -0.171])    # SM101's decision: SM101 - SM98 show-time +0.192 [+0.171, +0.215] (selftest pin)
SM113_MINUS_SM98 = (-0.094, [-0.103, -0.085])    # SM113's decision (selftest pin)
NAMES = {"sm101": "the Tiny Instant-Only model (SM101)", "sm98": "the Small Instant-Only model (SM98)",
         "sm99": "the show-time instant-only big model (SM99)", "sm92": "the show-time retrain of the release (SM92)",
         "release": "the release (SM63)", "big": "the big model (1 leak-free epoch)",
         "sm113": "the longer big instant-only teacher (SM113)", "sm115": "the longer cool-down for Tiny (SM115)"}


def report():
    got = hashlib.sha256(REPORT.read_bytes()).hexdigest()
    if got != REPORT_SHA:
        raise SystemExit(f"[sm115-rule] showtime_report.py changed (sha256 {got[:12]} != {REPORT_SHA[:12]}): refusing")
    spec = importlib.util.spec_from_file_location("showtime_report", REPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_pins():
    for path, want in PINS.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"[sm115-rule] reference rows {path.name} changed (sha256 {got[:12]} != {str(want)[:12]}): refusing")


def own(rows, rows1500, label, r):
    return {"e2s": [(rows, label + r.E2S, None)], "normal": [(rows, label, None)],
            "e2s1500": [(rows, label + r.E2S, None), (rows1500, label + r.E2S, None)]}


def refs(r):
    return {"sm101": own(S101G / "sm101-rows.jsonl", S101G / "sm101-rows-1500.jsonl", LBL101, r),
            "sm98": own(S98G / "sm98-rows.jsonl", S98G / "sm98-rows-1500.jsonl", LBL98, r),
            "sm99": own(S99G / "sm99-rows.jsonl", S99G / "sm99-rows-1500.jsonl", LBL99, r),
            "sm113": own(S113G / "sm113-rows.jsonl", S113G / "sm113-rows-1500.jsonl", LBL113, r),
            "sm92": r.MODELS["sm92"], "release": r.MODELS["release"]}


def fp(x):
    return f"{x['points']:+.3f}% [{x['points_ci'][0]:+.3f}, {x['points_ci'][1]:+.3f}]"


def fpct(x):
    return f"{x['pct_of_ref']:+.3f}% [{x['pct_of_ref_ci'][0]:+.3f}, {x['pct_of_ref_ci'][1]:+.3f}]"


def mid_decision(x):
    return "continue" if x["points"] < 0 else "stop"


def verdict(x):
    return "go" if x["points_ci"][1] < 0 and x["points"] <= -MIN_EFFECT else "no go"


def primary(r, new_e2s, users, big):
    s101 = r.load(refs(r)["sm101"]["e2s"], set(users))
    us = sorted(set(users) & set(new_e2s) & set(s101))
    x = r.compare(new_e2s, s101, big, us)["instant"]
    return {"users": len(us), "new_minus_sm101_e2s": x, "verdict": verdict(x)}


def gap_closed(x_new, x_teacher):
    """share of the teacher gap closed: (new - SM101) / (SM99 - SM101), on points of the same users."""
    return round(x_new["points"] / x_teacher["points"], 3) if x_teacher["points"] else None


def selftest():
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    src = refs(r)
    n = {k: r.load(src[k]["e2s"], have) for k in ("sm101", "sm98", "sm99", "sm113")}
    p101, p98, p99, p113 = (primary(r, n[k], users, big) for k in ("sm101", "sm98", "sm99", "sm113"))
    x98, x99, x113 = p98["new_minus_sm101_e2s"], p99["new_minus_sm101_e2s"], p113["new_minus_sm101_e2s"]
    small = {u: {h: (v[h][0] * 0.9996, v[h][1]) for h in v} for u, v in n["sm101"].items()}   # 0.04% better everywhere
    ps = primary(r, small, users, big)
    us = sorted(have)
    y113 = r.compare(n["sm113"], n["sm98"], big, us)["instant"]   # SM113's own primary, through this code
    checks = {
        "sm101_against_itself_zero_no_go_mid_stop": p101["users"] == 722 and p101["new_minus_sm101_e2s"]["points"] == 0
        and p101["verdict"] == "no go" and mid_decision(p101["new_minus_sm101_e2s"]) == "stop",
        "sm98_minus_sm101_known_go_continue": p98["users"] == 722 and (x98["points"], x98["points_ci"]) == SM98_MINUS_SM101
        and x98["new"] == 0.276191 and x98["ref"] == 0.27672 and p98["verdict"] == "go" and mid_decision(x98) == "continue",
        "sm113_minus_sm98_known_through_this_code": (y113["points"], y113["points_ci"]) == SM113_MINUS_SM98,
        "sm99_minus_sm101_go_gap_one": p99["users"] == 722 and p99["verdict"] == "go" and x99["points"] < -0.25
        and gap_closed(x99, x99) == 1.0 and x99["new"] == 0.275894,
        "sm113_minus_sm101_go": p113["users"] == 722 and p113["verdict"] == "go" and x113["points"] < -0.25,
        "synthetic_0p04pct_better_is_no_go_but_continue": ps["verdict"] == "no go" and -0.045 < ps["new_minus_sm101_e2s"]["points"] < -0.035
        and ps["new_minus_sm101_e2s"]["points_ci"][1] < 0 and mid_decision(ps["new_minus_sm101_e2s"]) == "continue",
        "mid_edges": mid_decision({"points": -0.001}) == "continue" and mid_decision({"points": 0.0}) == "stop"
        and mid_decision({"points": 0.001}) == "stop",
        "go_edges": verdict({"points": -0.05, "points_ci": [-0.07, -0.001]}) == "go"
        and verdict({"points": -0.049, "points_ci": [-0.07, -0.03]}) == "no go"
        and verdict({"points": -0.06, "points_ci": [-0.12, 0.0]}) == "no go"}
    checks["end_to_end_mid_and_decide"] = end_to_end(r)
    ok = all(checks.values())
    print(f"[sm115-rule] selftest: SM101 - SM101 {fp(p101['new_minus_sm101_e2s'])} -> {p101['verdict']}; SM98 - SM101 {fp(x98)} -> "
          f"{p98['verdict']}; SM99 - SM101 {fp(x99)} -> {p99['verdict']}; SM113 - SM101 {fp(x113)} -> {p113['verdict']} "
          f"(gap closed {gap_closed(x113, x99)}); SM113 - SM98 {fp(y113)}; synthetic 0.04% better "
          f"{fp(ps['new_minus_sm101_e2s'])} -> {ps['verdict']}; {json.dumps(checks)} -> {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def end_to_end(r):
    """mid + decide on SM101's own rows relabelled (as pass 1, as the early-stop reference and as the final model): mid
    must say stop (0 is not < 0); decide must read 0 -> no go, gap closed 0, with every table filled."""
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
        rows, rows1500, mrows = t / "rows.jsonl", t / "rows-1500.jsonl", t / "mid.jsonl"
        relabel(S101G / "sm101-rows.jsonl", rows, {LBL101: LBL})
        relabel(S101G / "sm101-rows-1500.jsonl", rows1500, {LBL101: LBL})
        relabel(S101G / "sm101-rows.jsonl", mrows, {LBL101: MID_NEW})
        relabel(S101G / "sm101-rows.jsonl", mrows, {LBL101: MID_REF})
        with redirect_stdout(io.StringIO()):
            a = mid(t, mrows)
            b = decide(t, rows, rows1500, mrows)
        m = json.loads((t / "sm115-mid.json").read_text())
        d = json.loads((t / "sm115-decision.json").read_text())
        q = d["pool_1500"]
        return (a == 0 and b == 0 and m["decision"] == "stop" and m["instant"]["points"] == 0
                and m["vs_sm113_p1"]["points"] > 0.1
                and d["verdict"] == "no go" and d["new_minus_sm101_e2s"]["points"] == 0 and d["gap_closed"] == 0.0
                and all(v is not None for v in d["secondary_732"].values()) and "new_minus_sm113_e2s" in d["secondary_732"]
                and "mid_p1_minus_sm101_u4094_e2s" in d["secondary_732"] and "new_normal_minus_big_normal" in d["secondary_732"]
                and q.get("users", 0) > 1400 and q["new_minus_sm101"]["instant"]["pct_of_ref"] == 0
                and q.get("new_minus_sm113") is not None
                and q.get("users_with_sm99", 0) == q["users"] and q["row_check"]["max_abs_logloss_diff"] == 0.0
                and (t / "sm115-decision.md").exists() and (t / "sm115-mid.md").exists())


def mid(out_dir=HERE, mrows=MIDROWS):
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    a = r.load([(mrows, MID_NEW + r.E2S, None)], have)
    b = r.load([(mrows, MID_REF + r.E2S, None)], have)
    c = r.load([(R113_MID, MID_113 + r.E2S, None)], have)
    if not (have <= set(a) and have <= set(b) and have <= set(c)):
        print(f"[sm115-rule] mid pending: rows pass 1 {len(have & set(a))}, SM101 u4094 {len(have & set(b))}, "
              f"SM113 pass 1 {len(have & set(c))} of {len(users)} rule users")
        return 2
    x = r.compare(a, b, big, sorted(have))["instant"]
    y = r.compare(a, c, big, sorted(have))["instant"]
    out = {"rule": f"continue (pass 2) if SM115's pass-1 final model - SM101's cool-down snapshot after 4,094 users (instant, "
                   f"show-time inputs, {len(users)} users) has a point estimate < 0; stop otherwise", "users": len(users),
           "labels": [MID_NEW, MID_REF, MID_113], "instant": x, "vs_sm113_p1": y, "decision": mid_decision(x)}
    (Path(out_dir) / "sm115-mid.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = ["# The longer cool-down for Tiny (SM115): early stop after pass 1", "",
          f"Decision: **{out['decision']}**. The longer cool-down for Tiny (SM115)'s model after pass 1 of 2 (learning rate "
          f"29.29% of its peak) minus the Tiny Instant-Only model (SM101)'s own cool-down snapshot after 4,094 of 8,000 users "
          f"(learning rate 29.31% of its peak), instant head, show-time inputs, {len(users)} users: {fp(x)} (% of the big model "
          f"(1 leak-free epoch) normal-input instant LogLoss), {x['users_better']} better / {x['users_worse']} worse; "
          f"{x['new']} vs {x['ref']}.", "",
          f"Read-out (no verdict): minus the longer big instant-only teacher (SM113)'s model after its pass 1 (the Small "
          f"Instant-Only architecture, the same recipe at the same point): {fp(y)}, {y['users_better']} better / "
          f"{y['users_worse']} worse.", "",
          "Rule (pre-registered): continue with pass 2 if the point estimate is below 0; otherwise stop (verdict no go).", ""]
    (Path(out_dir) / "sm115-mid.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[sm115-rule] mid {out['decision']}: SM115 pass 1 - SM101 u4094 show-time instant {fp(x)}; - SM113 pass 1 {fp(y)} "
          f"({len(users)} users)")
    return 0


def decide(out_dir=HERE, rows=ROWS, rows1500=ROWS1500, mrows=MIDROWS):
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    mine = own(rows, rows1500, LBL, r)
    new = {v: r.load(mine[v], have) for v in ("e2s", "normal")}
    if not have <= set(new["e2s"]):
        print(f"[sm115-rule] pending: {LBL} show-time rows {len(have & set(new['e2s']))} of {len(users)} rule users")
        return 2
    out = {"rule": "SM115 - SM101, instant, show-time inputs, 722 users: go = 95% CI below 0 and points <= -0.05",
           "label": LBL, "users": len(users), "big_instant": round(r.big_denominator(big, users, "instant"), 6)}
    p = primary(r, new["e2s"], users, big)
    out |= p
    src = refs(r)
    ref = {k: {v: r.load(s[v], have) for v in ("e2s", "normal")} for k, s in src.items()}

    def pair(a, b):
        us = sorted(set(users) & set(a) & set(b))
        return r.compare(a, b, big, us) if us else None

    teacher = pair(ref["sm99"]["e2s"], ref["sm101"]["e2s"])
    out["sm99_minus_sm101_e2s"] = teacher["instant"]
    out["gap_closed"] = gap_closed(p["new_minus_sm101_e2s"], teacher["instant"])
    sec = {"new_minus_sm113_e2s": pair(new["e2s"], ref["sm113"]["e2s"]),
           "new_minus_sm99_e2s": pair(new["e2s"], ref["sm99"]["e2s"]),
           "new_minus_sm98_e2s": pair(new["e2s"], ref["sm98"]["e2s"]),
           "new_minus_sm92_e2s": pair(new["e2s"], ref["sm92"]["e2s"]),
           "new_minus_release_e2s": pair(new["e2s"], ref["release"]["e2s"])}
    if have <= set(new["normal"]):
        sec |= {"new_minus_sm101_normal": pair(new["normal"], ref["sm101"]["normal"]),
                "new_normal_minus_big_normal": pair(new["normal"], big),
                "new_e2s_minus_new_normal": pair(new["e2s"], new["normal"])}
    ma = r.load([(mrows, MID_NEW + r.E2S, None)], have)
    mb, mc = r.load([(mrows, MID_REF + r.E2S, None)], have), r.load([(R113_MID, MID_113 + r.E2S, None)], have)
    if have <= set(ma) and have <= set(mb):
        sec["mid_p1_minus_sm101_u4094_e2s"] = pair(ma, mb)
    if have <= set(ma) and have <= set(mc):
        sec["mid_p1_minus_sm113_p1_e2s"] = pair(ma, mc)
    out["secondary_732"] = sec
    q = {}
    if r.POOL1500.exists():
        pool = set(json.loads(r.POOL1500.read_text())["users"])
        rel = r.load(r.MODELS["release"]["e2s1500"], pool)
        other = {k: r.load(src[k]["e2s1500"], pool) for k in ("sm101", "sm99", "sm113")}
        nw = r.load(mine["e2s1500"], pool)
        us = sorted(set(nw) & set(rel) & set(other["sm101"]))
        q["users"] = len(us)
        if us and Path(rows1500).exists():
            q["new_minus_release"] = r.compare(nw, rel, rel, us)
            q["new_minus_sm101"] = r.compare(nw, other["sm101"], rel, us)
            u113 = sorted(set(us) & set(other["sm113"]))
            q["new_minus_sm113"] = r.compare(nw, other["sm113"], rel, u113) if u113 else None
            u99 = sorted(set(us) & set(other["sm99"]))
            q["users_with_sm99"] = len(u99)
            if u99:
                q["new_minus_sm99"] = r.compare(nw, other["sm99"], rel, u99)
            lo, hi = [u for u in us if u < 5000], [u for u in us if u >= 5000]
            q["by_part_new_minus_sm101"] = {"ids < 5000 (the 732 set)": r.compare(nw, other["sm101"], rel, lo) if lo else None,
                                            "ids >= 5000 (SM99's training users)": r.compare(nw, other["sm101"], rel, hi) if hi else None}
            spec = json.loads(r.USERS1500.read_text())
            u732 = set(json.loads(r.U732.read_text())["users"]) | set(json.loads(r.ULONG.read_text())["users"])
            check = sorted(set(spec["users"] if isinstance(spec["users"], list) else []) & u732)
            a = r.load([(rows, LBL + r.E2S, None)], set(check)); b = r.load([(rows1500, LBL + r.E2S, None)], set(check))
            d = [max(abs(a[u][h][0] - b[u][h][0]) for h in r.HEADS) for u in check if u in a and u in b]
            q["row_check"] = {"users": len(d), "max_abs_logloss_diff": max(d) if d else None}
    out["pool_1500"] = q
    (Path(out_dir) / "sm115-decision.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    names = {"new_minus_sm113_e2s": (NAMES["sm113"] + " (Small, the same recipe)", "show-time"),
             "new_minus_sm99_e2s": (NAMES["sm99"] + " (the teacher)", "show-time"),
             "new_minus_sm98_e2s": (NAMES["sm98"], "show-time"), "new_minus_sm92_e2s": (NAMES["sm92"], "show-time"),
             "new_minus_release_e2s": (NAMES["release"], "show-time"), "new_minus_sm101_normal": (NAMES["sm101"], "normal"),
             "new_normal_minus_big_normal": (NAMES["big"] + " (the smaller-model instant bar: CI upper end <= +0.40%)", "normal"),
             "new_e2s_minus_new_normal": ("the new model on normal inputs (its show-time cost)", "show-time vs normal"),
             "mid_p1_minus_sm101_u4094_e2s": ("early stop: SM115 after pass 1 minus SM101's snapshot after 4,094 users", "show-time"),
             "mid_p1_minus_sm113_p1_e2s": ("early stop read-out: SM115 after pass 1 minus SM113 after its pass 1", "show-time")}
    x, t = p["new_minus_sm101_e2s"], teacher["instant"]
    md = ["# The longer cool-down for Tiny (SM115): decision", "",
          f"Verdict: **{p['verdict']}** ({len(users)} users of the 732 set; % of the big model (1 leak-free epoch) "
          f"normal-input instant LogLoss {out['big_instant']}; paired user bootstrap 2,000, seed 0, 95% CI).", "",
          f"- Primary: the longer cool-down for Tiny (SM115) minus the Tiny Instant-Only model (SM101), instant, "
          f"show-time inputs: {fp(x)} ({x['new']} vs {x['ref']}), {x['users_better']} better / {x['users_worse']} worse. Go "
          f"needs the CI below 0 and <= -0.05%.", "",
          f"- Share of the teacher gap closed: {out['gap_closed']} (the show-time instant-only big model (SM99) minus the Tiny "
          f"Instant-Only model (SM101) is {fp(t)} on the same users).", "",
          "## Read-out (732 set, instant head, %)", "", "| minus | inputs | new | ref | % [95% CI] | better / worse |",
          "|---|---|---|---|---|---|"]
    for k, v in sec.items():
        if v:
            y = v["instant"]
            md.append(f"| {names[k][0]} | {names[k][1]} | {y['new']} | {y['ref']} | {fp(y)} | {y['users_better']} / {y['users_worse']} |")
    md.append("")
    if q.get("new_minus_release"):
        md += [f"## 1,500 validation users, show-time inputs ({q['users']} users), % of the release's own LogLoss", "",
               "| comparison | new | ref | % [95% CI] | better / worse |", "|---|---|---|---|---|"]
        for k, nm in (("new_minus_sm101", "new minus " + NAMES["sm101"]), ("new_minus_sm113", "new minus " + NAMES["sm113"]),
                      ("new_minus_sm99", "new minus " + NAMES["sm99"]), ("new_minus_release", "new minus " + NAMES["release"])):
            if q.get(k):
                y = q[k]["instant"]
                md.append(f"| {nm} | {y['new']} | {y['ref']} | {fpct(y)} | {y['users_better']} / {y['users_worse']} |")
        for part, v in q["by_part_new_minus_sm101"].items():
            if v:
                y = v["instant"]
                md.append(f"| new minus {NAMES['sm101']}, {part} ({v['users']} users) | {y['new']} | {y['ref']} | {fpct(y)} | {y['users_better']} / {y['users_worse']} |")
        md += ["", f"Users with rows of the show-time instant-only big model (SM99): {q.get('users_with_sm99')}. Its ids >= 5000 "
               f"are its own training users, so new minus SM99 there is not a held-out comparison. Row check: "
               f"{json.dumps(q['row_check'])}", ""]
    else:
        md += ["## 1,500 validation users", "", "Not scored yet.", ""]
    md += ["The curve head is untrained (curve loss weight 0); curve numbers are in the JSON only and not part of the verdict.", ""]
    (Path(out_dir) / "sm115-decision.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[sm115-rule] verdict {p['verdict']}: SM115 - SM101 show-time instant {fp(x)}; SM115 - SM113 "
          f"{fp(sec['new_minus_sm113_e2s']['instant'])}; gap closed {out['gap_closed']}; SM115 - SM99 "
          f"{fp(sec['new_minus_sm99_e2s']['instant'])}; 1,500 pool users {q.get('users', 0)}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "decide"
    raise SystemExit({"selftest": selftest, "mid": mid, "decide": decide}[cmd]())
