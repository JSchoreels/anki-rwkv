#!/usr/bin/env python
"""The longer big instant-only teacher (SM113): pre-registered rules (fixed 2026-11-13 before any GPU run; CPU, seconds).

New model: the big instant-only teacher (SM110)'s cool-down stretched over two passes of the 8,000 users. Start = the
Small Instant-Only model (SM98)'s constant-LR epoch 35 (read only); one one_minus_sin decay 7e-4 -> 0 over 2 x 581.9M
reviews, fresh Muon, show-time inputs, curve loss weight 0, the show-time instant-only big model (SM99)'s show-time KD
replay targets (decay alpha 0.5), split into pass 1 (== SM110's cool-down except lr_resume, sm113_pair.py p1) and pass 2
(pass 1's end, the same rows in a new order, sm113_pair.py p2).
Labels: final `sm113-inst-e35st-st-dec2ep32k-t99/w8+s8` (show-time rows `@e2s-query`, gpu/sm113-rows.jsonl); early stop
`sm113-dec2ep-p1/w8+s8@e2s-query` (pass 1's final model) in gpu/sm113-mid-rows.jsonl. References: SM98, SM96, SM99 rows of
this branch; SM110's final and mid-check rows, copied from wt/sm110 to ref/ (sha256 pinned): `sm110-...-t99`,
`sm110-dec-u4000` and `sm98-dec-u4000` (SM98's own cool-down snapshot after 4,000 of 8,000 users).

Loaders and the paired bootstrap are the show-time report's (`artifacts/small-model/showtime/showtime_report.py` on main,
sha256 pinned, the same pin as SM96 / SM98 / SM99 / SM101 / SM110): 2,000 resamples, seed 0, 95% CI, review-weighted
LogLoss, the instant head, 722 users of the 732 set. Unit: % of the big model (1 leak-free epoch) normal-input instant
LogLoss on the same users (0.275757). Reference rows are read from their own files (sha256 pinned), never rescored.

EARLY STOP (`mid`, after pass 1 = half of the reviews, learning rate 29.3% of its peak):
  x = SM113's pass-1 final model - SM98's cool-down snapshot users-04000 (learning rate 29.9% of its peak, the matching
  point of SM98's one-epoch schedule), instant, show-time inputs, 722 users.
  continue  if the point estimate x < 0: pass 2 runs.
  stop      otherwise (x >= 0): verdict "no go (stopped after pass 1)"; saves ~8.8 GPU h.
  Read-out with it (no verdict): pass 1 - SM110's snapshot users-04000 (the same targets at the same learning rate).
FINAL (`decide`):
  Primary (the verdict): SM113 - SM98, instant, show-time inputs, 722 users:
    go     = the 95% CI entirely below 0 AND points <= -0.05;
    no go  = otherwise.
  Read-out with it (no verdict): SM113 - SM110 (the longer cool-down with the same targets); the share of the teacher gap
  closed, (SM113 - SM98) / (SM99 - SM98); SM113 - SM99, - SM96, - SM92, - the release (SM63) (show-time); SM113 - SM98
  (normal); SM113 normal - the big model (1 leak-free epoch) normal (the smaller-model bar: CI upper end <= +0.40%); the
  show-time cost; the early-stop pairs. The 1,500 validation users on show-time inputs (% of the release's own LogLoss):
  SM113 - SM98, - SM110, - SM99, - the release, by id part, and the row check.
  The curve head is untrained (weight 0): never part of a verdict.

  python sm113_rule.py selftest   -> the rules' code on known rows (exit 0 if all equal)
  python sm113_rule.py mid        -> sm113-mid.{json,md}; exit 2 while rows are missing
  python sm113_rule.py decide     -> sm113-decision.{json,md}; exit 2 while the final model's 722-user rows are missing
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
G = HERE / "gpu"
REF = HERE / "ref"
WT = HERE.parents[2]
REPORT = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/small-model/showtime/showtime_report.py")
REPORT_SHA = "3cdfed03afd40734d4b1931c244e8c1e73453d281a40c3843e80cd310d70dd9d"   # main 9031ed51 (same pin as SM96 / SM98 / SM99 / SM101 / SM110)
ROWS, ROWS1500, MIDROWS = G / "sm113-rows.jsonl", G / "sm113-rows-1500.jsonl", G / "sm113-mid-rows.jsonl"
LBL = "sm113-inst-e35st-st-dec2ep32k-t99/w8+s8"
MID_NEW = "sm113-dec2ep-p1/w8+s8"
MID_REF, MID_110 = "sm98-dec-u4000/w8+s8", "sm110-dec-u4000/w8+s8"
S96G, S98G, S99G = (WT / f"artifacts/small-model/{x}/gpu" for x in ("sm96", "sm98", "sm99"))
R110, R110_1500, R110_MID = REF / "sm110-rows.jsonl", REF / "sm110-rows-1500.jsonl", REF / "sm110-mid-rows.jsonl"
PINS = {S96G / "sm96-rows.jsonl": "cc93a32ac9f8ab522d028db342c2f07019874cddf50d7a728b117b3e418dd4f9",
        S96G / "sm96-rows-1500.jsonl": "7bc69afbf90f81ef88f275d2d6911f19c7bdfafc9091aca69ef2b62134f5c8f8",
        S98G / "sm98-rows.jsonl": "d4cc71b8a1ecc3d3250c19c58952b3607608cbb7d8ebc312109a9e77fc16234f",
        S98G / "sm98-rows-1500.jsonl": "288b2d1d9390233748d6372e50d1f77c688b33d1b80844101e8bf3b2698fc5fa",
        S99G / "sm99-rows.jsonl": "bc8c08b294968aa4a58a38241400714957eccd3071a70e5b85665d2fd46060c5",       # branch sm99 df9eab05
        S99G / "sm99-rows-1500.jsonl": "f87fb3d85691fc22754915b422323a033a437319a2a8ce534cc28e137a303004",
        R110: "cc873698aa33807a81ef06e1e2cd971b8363d59ee7a58184a4399012ff1cb8a0",                            # wt/sm110 gpu/, 11-13
        R110_1500: "44c4916a91be3820addb4fd6c1031f14fa78d82854716aa2e42a5e35c2425cf5",
        R110_MID: "bca7022a0451a8bb3b321dbd54801428788e4d5bde2d7190e13caf5d33987665"}
LBL96, LBL98, LBL99 = "sm96-inst-e31st-st-dec1ep32k/w8+s8", "sm98-inst-e35st-st-dec1ep32k/w8+s8", "sm99-big-inst-e2/w8+s8"
LBL110 = "sm110-inst-e35st-st-dec1ep32k-t99/w8+s8"
MIN_EFFECT = 0.05
SM99_MINUS_SM98 = (-0.108, [-0.133, -0.084])   # SM99's decision (selftest pin)
SM96_MINUS_SM98 = (0.035, [0.029, 0.041])
SM110_MINUS_SM98 = (-0.023, [-0.028, -0.019])  # SM110's decision (selftest pin)
SM110_MID = (-0.001, [-0.010, 0.007])          # SM110's mid-check (selftest pin)
NAMES = {"sm98": "the Small Instant-Only model (SM98)", "sm99": "the show-time instant-only big model (SM99)",
         "sm96": "the show-time instant-only small model (SM96)", "sm92": "the show-time retrain of the release (SM92)",
         "release": "the release (SM63)", "big": "the big model (1 leak-free epoch)",
         "sm110": "the big instant-only teacher (SM110)", "sm113": "the longer big instant-only teacher (SM113)"}


def report():
    got = hashlib.sha256(REPORT.read_bytes()).hexdigest()
    if got != REPORT_SHA:
        raise SystemExit(f"[sm113-rule] showtime_report.py changed (sha256 {got[:12]} != {REPORT_SHA[:12]}): refusing")
    spec = importlib.util.spec_from_file_location("showtime_report", REPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_pins():
    for path, want in PINS.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"[sm113-rule] reference rows {path.name} changed (sha256 {got[:12]} != {str(want)[:12]}): refusing")


def own(rows, rows1500, label, r):
    return {"e2s": [(rows, label + r.E2S, None)], "normal": [(rows, label, None)],
            "e2s1500": [(rows, label + r.E2S, None), (rows1500, label + r.E2S, None)]}


def refs(r):
    return {"sm96": own(S96G / "sm96-rows.jsonl", S96G / "sm96-rows-1500.jsonl", LBL96, r),
            "sm98": own(S98G / "sm98-rows.jsonl", S98G / "sm98-rows-1500.jsonl", LBL98, r),
            "sm99": own(S99G / "sm99-rows.jsonl", S99G / "sm99-rows-1500.jsonl", LBL99, r),
            "sm110": own(R110, R110_1500, LBL110, r),
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
    s98 = r.load(refs(r)["sm98"]["e2s"], set(users))
    us = sorted(set(users) & set(new_e2s) & set(s98))
    x = r.compare(new_e2s, s98, big, us)["instant"]
    return {"users": len(us), "new_minus_sm98_e2s": x, "verdict": verdict(x)}


def gap_closed(x_new, x_teacher):
    """share of the teacher gap closed: (new - SM98) / (SM99 - SM98), on points of the same users."""
    return round(x_new["points"] / x_teacher["points"], 3) if x_teacher["points"] else None


def selftest():
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    src = refs(r)
    n = {k: r.load(src[k]["e2s"], have) for k in ("sm98", "sm96", "sm99", "sm110")}
    p98, p96, p99, p110 = (primary(r, n[k], users, big) for k in ("sm98", "sm96", "sm99", "sm110"))
    x96, x99, x110 = p96["new_minus_sm98_e2s"], p99["new_minus_sm98_e2s"], p110["new_minus_sm98_e2s"]
    small = {u: {h: (v[h][0] * 0.9996, v[h][1]) for h in v} for u, v in n["sm98"].items()}   # 0.04% better everywhere
    ps = primary(r, small, users, big)
    m110 = r.compare(r.load([(R110_MID, MID_110 + r.E2S, None)], have), r.load([(R110_MID, MID_REF + r.E2S, None)], have),
                     big, sorted(have))["instant"]
    checks = {
        "sm98_against_itself_zero_no_go_mid_stop": p98["users"] == 722 and p98["new_minus_sm98_e2s"]["points"] == 0
        and p98["verdict"] == "no go" and mid_decision(p98["new_minus_sm98_e2s"]) == "stop",
        "sm96_minus_sm98_known_no_go_stop": (x96["points"], x96["points_ci"]) == SM96_MINUS_SM98 and p96["verdict"] == "no go"
        and mid_decision(x96) == "stop",
        "sm99_minus_sm98_known_go_continue": p99["users"] == 722 and (x99["points"], x99["points_ci"]) == SM99_MINUS_SM98
        and x99["new"] == 0.275894 and x99["ref"] == 0.276191 and p99["verdict"] == "go" and mid_decision(x99) == "continue"
        and gap_closed(x99, x99) == 1.0,
        "sm110_minus_sm98_known_no_go": p110["users"] == 722 and (x110["points"], x110["points_ci"]) == SM110_MINUS_SM98
        and p110["verdict"] == "no go" and gap_closed(x110, x99) == 0.213,
        "sm110_mid_pair_known_continue": (m110["points"], m110["points_ci"]) == SM110_MID and mid_decision(m110) == "continue",
        "synthetic_0p04pct_better_is_no_go_but_continue": ps["verdict"] == "no go" and -0.045 < ps["new_minus_sm98_e2s"]["points"] < -0.035
        and ps["new_minus_sm98_e2s"]["points_ci"][1] < 0 and mid_decision(ps["new_minus_sm98_e2s"]) == "continue",
        "mid_edges": mid_decision({"points": -0.001}) == "continue" and mid_decision({"points": 0.0}) == "stop"
        and mid_decision({"points": 0.001}) == "stop",
        "go_edges": verdict({"points": -0.05, "points_ci": [-0.07, -0.001]}) == "go"
        and verdict({"points": -0.049, "points_ci": [-0.07, -0.03]}) == "no go"
        and verdict({"points": -0.06, "points_ci": [-0.12, 0.0]}) == "no go"}
    checks["end_to_end_mid_and_decide"] = end_to_end(r)
    ok = all(checks.values())
    print(f"[sm113-rule] selftest: SM98 - SM98 {fp(p98['new_minus_sm98_e2s'])} -> {p98['verdict']}; SM96 - SM98 {fp(x96)} -> "
          f"{p96['verdict']}; SM99 - SM98 {fp(x99)} -> {p99['verdict']}; SM110 - SM98 {fp(x110)} -> {p110['verdict']} "
          f"(gap closed {gap_closed(x110, x99)}); SM110 mid pair {fp(m110)} -> {mid_decision(m110)}; synthetic 0.04% better "
          f"{fp(ps['new_minus_sm98_e2s'])} -> {ps['verdict']}; {json.dumps(checks)} -> {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def end_to_end(r):
    """mid + decide on SM98's own rows relabelled (as pass 1, as the early-stop references and as the final model): mid
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
        rows, rows1500, mrows, refmid = t / "rows.jsonl", t / "rows-1500.jsonl", t / "mid.jsonl", t / "refmid.jsonl"
        relabel(S98G / "sm98-rows.jsonl", rows, {LBL98: LBL})
        relabel(S98G / "sm98-rows-1500.jsonl", rows1500, {LBL98: LBL})
        relabel(S98G / "sm98-rows.jsonl", mrows, {LBL98: MID_NEW})
        relabel(S98G / "sm98-rows.jsonl", refmid, {LBL98: MID_REF})
        relabel(S98G / "sm98-rows.jsonl", refmid, {LBL98: MID_110})
        with redirect_stdout(io.StringIO()):
            a = mid(t, mrows, refmid)
            b = decide(t, rows, rows1500, mrows, refmid)
        m = json.loads((t / "sm113-mid.json").read_text())
        d = json.loads((t / "sm113-decision.json").read_text())
        q = d["pool_1500"]
        return (a == 0 and b == 0 and m["decision"] == "stop" and m["vs_sm110_u4000"]["points"] == 0
                and d["verdict"] == "no go" and d["new_minus_sm98_e2s"]["points"] == 0 and d["gap_closed"] == 0.0
                and all(v is not None for v in d["secondary_732"].values()) and "new_minus_sm110_e2s" in d["secondary_732"]
                and q.get("users", 0) > 1400 and q["new_minus_sm98"]["instant"]["pct_of_ref"] == 0
                and q.get("new_minus_sm110") is not None
                and q.get("users_with_sm99", 0) == q["users"] and q["row_check"]["max_abs_logloss_diff"] == 0.0
                and (t / "sm113-decision.md").exists() and (t / "sm113-mid.md").exists())


def mid(out_dir=HERE, mrows=MIDROWS, refmid=R110_MID):
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    a = r.load([(mrows, MID_NEW + r.E2S, None)], have)
    b = r.load([(refmid, MID_REF + r.E2S, None)], have)
    c = r.load([(refmid, MID_110 + r.E2S, None)], have)
    if not (have <= set(a) and have <= set(b) and have <= set(c)):
        print(f"[sm113-rule] mid pending: rows pass 1 {len(have & set(a))}, SM98 u4000 {len(have & set(b))}, "
              f"SM110 u4000 {len(have & set(c))} of {len(users)} rule users")
        return 2
    x = r.compare(a, b, big, sorted(have))["instant"]
    y = r.compare(a, c, big, sorted(have))["instant"]
    out = {"rule": f"continue (pass 2) if SM113's pass-1 final model - SM98's cool-down snapshot users-04000 (instant, "
                   f"show-time inputs, {len(users)} users) has a point estimate < 0; stop otherwise", "users": len(users),
           "labels": [MID_NEW, MID_REF, MID_110], "instant": x, "vs_sm110_u4000": y, "decision": mid_decision(x)}
    (Path(out_dir) / "sm113-mid.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = ["# The longer big instant-only teacher (SM113): early stop after pass 1", "",
          f"Decision: **{out['decision']}**. The longer big instant-only teacher (SM113)'s model after pass 1 of 2 (learning "
          f"rate 29.3% of its peak) minus the Small Instant-Only model (SM98)'s cool-down snapshot after 4,000 of 8,000 users "
          f"(learning rate 29.9% of its peak), instant head, show-time inputs, {len(users)} users: {fp(x)} (% of the big model "
          f"(1 leak-free epoch) normal-input instant LogLoss), {x['users_better']} better / {x['users_worse']} worse; "
          f"{x['new']} vs {x['ref']}.", "",
          f"Read-out (no verdict): minus the big instant-only teacher (SM110)'s snapshot after 4,000 users (the same targets at "
          f"the same learning rate): {fp(y)}, {y['users_better']} better / {y['users_worse']} worse.", "",
          "Rule (pre-registered): continue with pass 2 if the point estimate is below 0; otherwise stop (verdict no go).", ""]
    (Path(out_dir) / "sm113-mid.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[sm113-rule] mid {out['decision']}: SM113 pass 1 - SM98 u4000 show-time instant {fp(x)}; - SM110 u4000 {fp(y)} "
          f"({len(users)} users)")
    return 0


def decide(out_dir=HERE, rows=ROWS, rows1500=ROWS1500, mrows=MIDROWS, refmid=R110_MID):
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    mine = own(rows, rows1500, LBL, r)
    new = {v: r.load(mine[v], have) for v in ("e2s", "normal")}
    if not have <= set(new["e2s"]):
        print(f"[sm113-rule] pending: {LBL} show-time rows {len(have & set(new['e2s']))} of {len(users)} rule users")
        return 2
    out = {"rule": "SM113 - SM98, instant, show-time inputs, 722 users: go = 95% CI below 0 and points <= -0.05",
           "label": LBL, "users": len(users), "big_instant": round(r.big_denominator(big, users, "instant"), 6)}
    p = primary(r, new["e2s"], users, big)
    out |= p
    src = refs(r)
    ref = {k: {v: r.load(s[v], have) for v in ("e2s", "normal")} for k, s in src.items()}

    def pair(a, b):
        us = sorted(set(users) & set(a) & set(b))
        return r.compare(a, b, big, us) if us else None

    teacher = pair(ref["sm99"]["e2s"], ref["sm98"]["e2s"])
    out["sm99_minus_sm98_e2s"] = teacher["instant"]
    out["gap_closed"] = gap_closed(p["new_minus_sm98_e2s"], teacher["instant"])
    sec = {"new_minus_sm110_e2s": pair(new["e2s"], ref["sm110"]["e2s"]),
           "new_minus_sm99_e2s": pair(new["e2s"], ref["sm99"]["e2s"]),
           "new_minus_sm96_e2s": pair(new["e2s"], ref["sm96"]["e2s"]),
           "new_minus_sm92_e2s": pair(new["e2s"], ref["sm92"]["e2s"]),
           "new_minus_release_e2s": pair(new["e2s"], ref["release"]["e2s"])}
    if have <= set(new["normal"]):
        sec |= {"new_minus_sm98_normal": pair(new["normal"], ref["sm98"]["normal"]),
                "new_normal_minus_big_normal": pair(new["normal"], big),
                "new_e2s_minus_new_normal": pair(new["e2s"], new["normal"])}
    ma = r.load([(mrows, MID_NEW + r.E2S, None)], have)
    mb, mc = r.load([(refmid, MID_REF + r.E2S, None)], have), r.load([(refmid, MID_110 + r.E2S, None)], have)
    if have <= set(ma) and have <= set(mb):
        sec["mid_p1_minus_sm98_u4000_e2s"] = pair(ma, mb)
    if have <= set(ma) and have <= set(mc):
        sec["mid_p1_minus_sm110_u4000_e2s"] = pair(ma, mc)
    out["secondary_732"] = sec
    q = {}
    if r.POOL1500.exists():
        pool = set(json.loads(r.POOL1500.read_text())["users"])
        rel = r.load(r.MODELS["release"]["e2s1500"], pool)
        other = {k: r.load(src[k]["e2s1500"], pool) for k in ("sm98", "sm99", "sm110")}
        nw = r.load(mine["e2s1500"], pool)
        us = sorted(set(nw) & set(rel) & set(other["sm98"]))
        q["users"] = len(us)
        if us and Path(rows1500).exists():
            q["new_minus_release"] = r.compare(nw, rel, rel, us)
            q["new_minus_sm98"] = r.compare(nw, other["sm98"], rel, us)
            u110 = sorted(set(us) & set(other["sm110"]))
            q["new_minus_sm110"] = r.compare(nw, other["sm110"], rel, u110) if u110 else None
            u99 = sorted(set(us) & set(other["sm99"]))
            q["users_with_sm99"] = len(u99)
            if u99:
                q["new_minus_sm99"] = r.compare(nw, other["sm99"], rel, u99)
            lo, hi = [u for u in us if u < 5000], [u for u in us if u >= 5000]
            q["by_part_new_minus_sm98"] = {"ids < 5000 (the 732 set)": r.compare(nw, other["sm98"], rel, lo) if lo else None,
                                           "ids >= 5000 (SM99's training users)": r.compare(nw, other["sm98"], rel, hi) if hi else None}
            spec = json.loads(r.USERS1500.read_text())
            u732 = set(json.loads(r.U732.read_text())["users"]) | set(json.loads(r.ULONG.read_text())["users"])
            check = sorted(set(spec["users"] if isinstance(spec["users"], list) else []) & u732)
            a = r.load([(rows, LBL + r.E2S, None)], set(check)); b = r.load([(rows1500, LBL + r.E2S, None)], set(check))
            d = [max(abs(a[u][h][0] - b[u][h][0]) for h in r.HEADS) for u in check if u in a and u in b]
            q["row_check"] = {"users": len(d), "max_abs_logloss_diff": max(d) if d else None}
    out["pool_1500"] = q
    (Path(out_dir) / "sm113-decision.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    names = {"new_minus_sm110_e2s": (NAMES["sm110"] + " (the same targets, one-epoch cool-down)", "show-time"),
             "new_minus_sm99_e2s": (NAMES["sm99"] + " (the teacher)", "show-time"),
             "new_minus_sm96_e2s": (NAMES["sm96"], "show-time"), "new_minus_sm92_e2s": (NAMES["sm92"], "show-time"),
             "new_minus_release_e2s": (NAMES["release"], "show-time"), "new_minus_sm98_normal": (NAMES["sm98"], "normal"),
             "new_normal_minus_big_normal": (NAMES["big"] + " (the smaller-model bar: CI upper end <= +0.40%)", "normal"),
             "new_e2s_minus_new_normal": ("the new model on normal inputs (its show-time cost)", "show-time vs normal"),
             "mid_p1_minus_sm98_u4000_e2s": ("early stop: SM113 after pass 1 minus SM98's snapshot after 4,000 users", "show-time"),
             "mid_p1_minus_sm110_u4000_e2s": ("early stop read-out: SM113 after pass 1 minus SM110's snapshot after 4,000 users", "show-time")}
    x, t = p["new_minus_sm98_e2s"], teacher["instant"]
    md = ["# The longer big instant-only teacher (SM113): decision", "",
          f"Verdict: **{p['verdict']}** ({len(users)} users of the 732 set; % of the big model (1 leak-free epoch) "
          f"normal-input instant LogLoss {out['big_instant']}; paired user bootstrap 2,000, seed 0, 95% CI).", "",
          f"- Primary: the longer big instant-only teacher (SM113) minus the Small Instant-Only model (SM98), instant, "
          f"show-time inputs: {fp(x)} ({x['new']} vs {x['ref']}), {x['users_better']} better / {x['users_worse']} worse. Go "
          f"needs the CI below 0 and <= -0.05%.", "",
          f"- Share of the teacher gap closed: {out['gap_closed']} (the show-time instant-only big model (SM99) minus the Small "
          f"Instant-Only model (SM98) is {fp(t)} on the same users).", "",
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
        for k, nm in (("new_minus_sm98", "new minus " + NAMES["sm98"]), ("new_minus_sm110", "new minus " + NAMES["sm110"]),
                      ("new_minus_sm99", "new minus " + NAMES["sm99"]), ("new_minus_release", "new minus " + NAMES["release"])):
            if q.get(k):
                y = q[k]["instant"]
                md.append(f"| {nm} | {y['new']} | {y['ref']} | {fpct(y)} | {y['users_better']} / {y['users_worse']} |")
        for part, v in q["by_part_new_minus_sm98"].items():
            if v:
                y = v["instant"]
                md.append(f"| new minus {NAMES['sm98']}, {part} ({v['users']} users) | {y['new']} | {y['ref']} | {fpct(y)} | {y['users_better']} / {y['users_worse']} |")
        md += ["", f"Users with rows of the show-time instant-only big model (SM99): {q.get('users_with_sm99')}. Its ids >= 5000 "
               f"are its own training users, so new minus SM99 there is not a held-out comparison. Row check: "
               f"{json.dumps(q['row_check'])}", ""]
    else:
        md += ["## 1,500 validation users", "", "Not scored yet.", ""]
    md += ["The curve head is untrained (curve loss weight 0); curve numbers are in the JSON only and not part of the verdict.", ""]
    (Path(out_dir) / "sm113-decision.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[sm113-rule] verdict {p['verdict']}: SM113 - SM98 show-time instant {fp(x)}; SM113 - SM110 "
          f"{fp(sec['new_minus_sm110_e2s']['instant'])}; gap closed {out['gap_closed']}; SM113 - SM99 "
          f"{fp(sec['new_minus_sm99_e2s']['instant'])}; 1,500 pool users {q.get('users', 0)}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "decide"
    raise SystemExit({"selftest": selftest, "mid": mid, "decide": decide}[cmd]())
