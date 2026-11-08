#!/usr/bin/env python
"""The show-time instant-only small model (SM96): pre-registered rule (fixed 2026-11-08 before any GPU run; CPU, seconds).

New model: the instant-only small model (SM57) stage 4's constant-LR epoch 30 + one constant-LR instant-only epoch 31 on
show-time inputs + the release's one-epoch 32k Muon cool-down from epoch 31 on show-time inputs ("show_time_intervals":
true = the scorer's e2s-query inputs, the switch of the show-time retrain of the release (SM92), applied as in its epoch 26
and cool-down), curve loss weight 0. Label `sm96-inst-e31st-st-dec1ep32k/w8+s8`.
Redesign 2026-11-09 00:20 (user, before any GPU run): epoch 31 added before the cool-down; the rule itself (primary,
threshold, pools, bootstrap) is unchanged, only the label moved from `sm96-inst-e30-st-dec1ep32k/w8+s8`. The secondary
new - stage 4 pair now reads epoch 31 + the switch together, not the switch alone.

Loaders and the paired bootstrap are the show-time report's (`artifacts/small-model/showtime/showtime_report.py` on main,
sha256 pinned below; `load`, `compare`, `pools`): 2,000 resamples, seed 0, 95% CI, review-weighted LogLoss.

PRIMARY (the verdict), 722 users of the 732 set (the users the big model (1 leak-free epoch) normal rows have), instant
head, both on show-time inputs: new - the show-time retrain of the release (SM92). Unit: points = % of the big model
(1 leak-free epoch) normal-input instant LogLoss on the same users (0.275757).
  go     = the 95% CI entirely below 0 AND the gain >= 0.05 points (points <= -0.05)
  no go  = otherwise
SECONDARY (reported, no verdict):
  new - stage 4 of the instant-only small model (SM57), show-time inputs (epoch 31 + the cool-down on show-time inputs);
  new - stage 4 and new - SM92 on normal inputs; the new model's own show-time cost (show-time - normal);
  new - the release (SM63) on show-time inputs;
  1,500 pool, show-time inputs, unit % of the release's own LogLoss (the `pool1500` convention): new - the release,
  new - stage 4, by part (ids < 5000 / >= 5000), row check (the check users of the 1,500 ticket vs their 732-set rows).
The curve head is untrained (curve loss weight 0): curve numbers are printed and flagged, never part of the verdict.

  python sm96_rule.py selftest   -> reproduces the show-time report's stage 4 - SM92 pair (+0.130 [+0.059, +0.216]) with
                                    this rule's code; exit 0 if equal
  python sm96_rule.py decide     -> sm96-decision.{json,md}; exit 2 while the new model's 722-user rows are missing
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
G = HERE / "gpu"
REPORT = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/small-model/showtime/showtime_report.py")
REPORT_SHA = "3cdfed03afd40734d4b1931c244e8c1e73453d281a40c3843e80cd310d70dd9d"   # main b07b6a13
ROWS, ROWS1500 = G / "sm96-rows.jsonl", G / "sm96-rows-1500.jsonl"
LBL = "sm96-inst-e31st-st-dec1ep32k/w8+s8"
MIN_EFFECT = 0.05


def report():
    got = hashlib.sha256(REPORT.read_bytes()).hexdigest()
    if got != REPORT_SHA:
        raise SystemExit(f"[sm96-rule] showtime_report.py changed (sha256 {got[:12]} != {REPORT_SHA[:12]}): refusing")
    spec = importlib.util.spec_from_file_location("showtime_report", REPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fp(x):
    return f"{x['points']:+.3f} [{x['points_ci'][0]:+.3f}, {x['points_ci'][1]:+.3f}]"


def fpct(x):
    return f"{x['pct_of_ref']:+.3f} [{x['pct_of_ref_ci'][0]:+.3f}, {x['pct_of_ref_ci'][1]:+.3f}]"


def verdict(x):
    return "go" if x["points_ci"][1] < 0 and x["points"] <= -MIN_EFFECT else "no go"


def selftest():
    r = report()
    users, big = r.pools()
    s4 = r.load(r.MODELS["s4"]["e2s"], set(users))
    sm92 = r.load(r.MODELS["sm92"]["e2s"], set(users))
    us = sorted(set(users) & set(s4) & set(sm92))
    x = r.compare(s4, sm92, big, us)["instant"]
    want = (0.130, [0.059, 0.216])
    ok = len(us) == 722 and abs(x["points"] - want[0]) < 1e-9 and x["points_ci"] == want[1] and x["new"] == 0.277236 \
        and x["ref"] == 0.276876
    print(f"[sm96-rule] selftest: {len(us)} users, stage 4 - SM92 show-time instant {fp(x)} (report +0.130 [+0.059, +0.216]); "
          f"stage 4 {x['new']} SM92 {x['ref']}; rule on it: {verdict(x)} -> {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def decide():
    r = report()
    users, big = r.pools()
    have = set(users)
    E2S = r.E2S
    new_e = r.load([(ROWS, LBL + E2S, None)], have)
    new_n = r.load([(ROWS, LBL, None)], have)
    if not set(users) <= set(new_e):
        print(f"[sm96-rule] pending: new show-time rows on {len(set(users) & set(new_e))} of {len(users)} rule users")
        return 2
    ref = {k: {"e2s": r.load(r.MODELS[k]["e2s"], have), "normal": r.load(r.MODELS[k]["normal"], have)} for k in ("sm92", "s4", "release")}
    out = {"rule": "new - SM92, instant, show-time inputs, 722 users: go = 95% CI below 0 and points <= -0.05",
           "label": LBL, "users": len(users), "big_instant": round(r.big_denominator(big, users, "instant"), 6)}

    def pair(a, b):
        us = sorted(set(users) & set(a) & set(b))
        return r.compare(a, b, big, us) if us else None

    out["primary_new_minus_sm92_e2s"] = p = pair(new_e, ref["sm92"]["e2s"])
    if p is None or p["users"] != len(users):
        print("[sm96-rule] pending: SM92 rows incomplete"); return 2
    out["verdict"] = verdict(p["instant"])
    sec = {"new_minus_s4_e2s": pair(new_e, ref["s4"]["e2s"]), "new_minus_release_e2s": pair(new_e, ref["release"]["e2s"])}
    if new_n:
        sec |= {"new_minus_s4_normal": pair(new_n, ref["s4"]["normal"]), "new_minus_sm92_normal": pair(new_n, ref["sm92"]["normal"]),
                "new_e2s_minus_new_normal": pair(new_e, new_n)}
    out["secondary_732"] = sec
    # 1,500 pool, show-time inputs, % of the release's own LogLoss (the report's pool1500 convention)
    q = {}
    if r.POOL1500.exists():
        pool = set(json.loads(r.POOL1500.read_text())["users"])
        rel = r.load(r.MODELS["release"]["e2s1500"], pool)
        s4 = r.load(r.MODELS["s4"]["e2s1500"], pool)
        new = r.load([(ROWS, LBL + E2S, None), (ROWS1500, LBL + E2S, None)], pool)
        us = sorted(set(new) & set(s4) & set(rel))
        q["users"] = len(us)
        if us and ROWS1500.exists():
            q["new_minus_release"] = r.compare(new, rel, rel, us)
            q["new_minus_s4"] = r.compare(new, s4, rel, us)
            old, nw = [u for u in us if u < 5000], [u for u in us if u >= 5000]
            q["by_part"] = {"ids < 5000 (the 732 set)": r.compare(new, rel, rel, old) if old else None,
                            "ids >= 5000": r.compare(new, rel, rel, nw) if nw else None}
            spec = json.loads(r.USERS1500.read_text())
            u732 = set(json.loads(r.U732.read_text())["users"]) | set(json.loads(r.ULONG.read_text())["users"])
            check = sorted(set(spec["users"] if isinstance(spec["users"], list) else []) & u732)
            a = r.load([(ROWS, LBL + E2S, None)], set(check)); b = r.load([(ROWS1500, LBL + E2S, None)], set(check))
            d = [max(abs(a[u][h][0] - b[u][h][0]) for h in r.HEADS) for u in check if u in a and u in b]
            q["row_check"] = {"users": len(d), "max_abs_logloss_diff": max(d) if d else None}
    out["pool_1500"] = q
    (HERE / "sm96-decision.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    names = {"new_minus_s4_e2s": ("stage 4 of the instant-only small model (SM57)", "show-time"),
             "new_minus_release_e2s": ("the release (SM63)", "show-time"),
             "new_minus_s4_normal": ("stage 4 of the instant-only small model (SM57)", "normal"),
             "new_minus_sm92_normal": ("the show-time retrain of the release (SM92)", "normal"),
             "new_e2s_minus_new_normal": ("the new model on normal inputs (its show-time cost)", "show-time vs normal")}
    pi = p["instant"]
    md = ["# The show-time instant-only small model (SM96): decision", "",
          f"Verdict: **{out['verdict']}**. New - the show-time retrain of the release (SM92), instant head, show-time inputs, "
          f"{p['users']} users: {fp(pi)} points (% of the big model (1 leak-free epoch) normal-input instant LogLoss "
          f"{out['big_instant']}), {pi['users_better']} better / {pi['users_worse']} worse; go needs the CI below 0 and <= -0.05.", "",
          f"New {pi['new']} vs SM92 {pi['ref']} (instant LogLoss, show-time inputs).", "",
          "## Secondary (732 set, instant head, points)", "", "| new minus | inputs | new | ref | points [95% CI] | better / worse |",
          "|---|---|---|---|---|---|"]
    for k, x in sec.items():
        if x:
            y = x["instant"]
            md.append(f"| {names[k][0]} | {names[k][1]} | {y['new']} | {y['ref']} | {fp(y)} | {y['users_better']} / {y['users_worse']} |")
    md.append("")
    if q.get("new_minus_release"):
        md += [f"## 1,500 validation users, show-time inputs ({q['users']} users), % of the release's own LogLoss", "",
               "| comparison | new | ref | % [95% CI] | better / worse |", "|---|---|---|---|---|"]
        for k, nm in (("new_minus_release", "new - the release (SM63)"), ("new_minus_s4", "new - stage 4 of the instant-only small model (SM57)")):
            y = q[k]["instant"]
            md.append(f"| {nm} | {y['new']} | {y['ref']} | {fpct(y)} | {y['users_better']} / {y['users_worse']} |")
        for part, x in q["by_part"].items():
            if x:
                y = x["instant"]
                md.append(f"| new - the release, {part} ({x['users']} users) | {y['new']} | {y['ref']} | {fpct(y)} | {y['users_better']} / {y['users_worse']} |")
        md += ["", f"Row check: {json.dumps(q['row_check'])}", ""]
    else:
        md += ["## 1,500 validation users", "", "Not scored yet.", ""]
    md += ["The curve head is untrained (curve loss weight 0); curve numbers are in the JSON only and not part of the verdict.", ""]
    (HERE / "sm96-decision.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[sm96-rule] verdict {out['verdict']}: new - SM92 show-time instant {fp(pi)} points ({p['users']} users); "
          f"new - stage 4 show-time {fp(sec['new_minus_s4_e2s']['instant']) if sec['new_minus_s4_e2s'] else '-'}; "
          f"1,500 pool users {q.get('users', 0)}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "decide"
    raise SystemExit(selftest() if cmd == "selftest" else decide())
