#!/usr/bin/env python
"""The smaller instant-only model (SM101): pre-registered rules (fixed 2026-11-10 before any GPU run; CPU, seconds).

New model: the show-time instant-only model, stage 2 (SM98)'s constant-LR epoch 35 with the card stream cut to 1 layer
(w64-c1) + 3 constant-LR instant-only epochs 36..38 on show-time inputs (SM98's epoch recipe) + the release's one-epoch
32k Muon cool-down on show-time inputs from epoch 38 (SM98's cool-down except the start and the architecture).
Label `sm101-c1-e38st-st-dec1ep32k/w8+s8`.

Loaders and the paired bootstrap are the show-time report's (`artifacts/small-model/showtime/showtime_report.py` on main,
sha256 pinned, the same pin as SM96 / SM98): 2,000 resamples, seed 0, 95% CI, review-weighted LogLoss, 722 users of the
732 set (the users the big model (1 leak-free epoch) normal rows have). Unit: points = % of the big model (1 leak-free
epoch) normal-input instant LogLoss on the same users (0.275757). References are read from their own rows (sha256
pinned), never rescored: the show-time instant-only small model (SM96) and SM98 (rows carried in this branch), the
show-time retrain of the release (SM92) and the release (SM63) (the report's sources).

EARLY CHECK (`early`, after epoch 36; stop rule):
  arm  = epoch 36 (constant LR, w64-c1), label sm101-c1-e36-st-const/w8+s8@e2s-query
  ref  = SM98's epoch 35 (constant LR, w64, the cut's source), label sm98-inst-e35-st-const/w8+s8@e2s-query
  instant head, show-time inputs, paired, 722 users.
  stop      if the point estimate arm - ref > +0.25 points (the final rule below leaves ~0.20 points of room for the
            cut on show-time inputs; 2 more epochs + the cool-down are not expected to win back more than ~0.05)
  continue  otherwise.
FINAL (`decide`, after the cool-down):
  A  the bar: new - the big model (1 leak-free epoch), both on normal inputs, instant head: the 95% CI upper end
     <= +0.40 points (the smaller model (SM73) instant bar, the whole CI inside it);
  B  what the fork ships: new - SM92, both on show-time inputs (`e2s-query`), instant head: the 95% CI entirely below 0
     AND the gain >= 0.05 points (points <= -0.05).
  go = A and B; no go otherwise.
  Reported, no verdict: new - SM96 and new - SM98 (show-time; normal); new - SM92 normal; new - the release show-time;
  new - big on show-time inputs; the new model's show-time cost; the 1,500 pool on show-time inputs (% of the release's
  own LogLoss: new - the release, new - SM96, new - SM98); the row check. The curve head is untrained (weight 0): never
  part of a verdict.

  python sm101_rule.py selftest   -> the rules' code on known rows: SM98 as "new" (A +0.110 [+0.081, +0.138], B -0.248
                                     [-0.270, -0.228] -> go; SM98 - SM96 show-time -0.035 [-0.041, -0.029] as in SM98's
                                     decision); SM92 as "new" (B = 0 -> no go); early rule on SM98 vs SM96 (-0.035 ->
                                     continue) and the release vs SM98 (+0.554 -> stop); exit 0 if all equal
  python sm101_rule.py early      -> sm101-early.{json,md}; exit 2 while rows are missing
  python sm101_rule.py decide     -> sm101-decision.{json,md}; exit 2 while the new model's 722-user rows are missing
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
REPORT_SHA = "3cdfed03afd40734d4b1931c244e8c1e73453d281a40c3843e80cd310d70dd9d"   # main 9031ed51 (same pin as SM96 / SM98)
ROWS, ROWS1500, EARLY_ROWS = G / "sm101-rows.jsonl", G / "sm101-rows-1500.jsonl", G / "sm101-early-rows.jsonl"
LBL = "sm101-c1-e38st-st-dec1ep32k/w8+s8"
EARLY_ARM, EARLY_REF = "sm101-c1-e36-st-const/w8+s8", "sm98-inst-e35-st-const/w8+s8"
S96G, S98G = WT / "artifacts/small-model/sm96/gpu", WT / "artifacts/small-model/sm98/gpu"
PINS = {S96G / "sm96-rows.jsonl": "cc93a32ac9f8ab522d028db342c2f07019874cddf50d7a728b117b3e418dd4f9",
        S96G / "sm96-rows-1500.jsonl": "7bc69afbf90f81ef88f275d2d6911f19c7bdfafc9091aca69ef2b62134f5c8f8",
        S98G / "sm98-rows.jsonl": "d4cc71b8a1ecc3d3250c19c58952b3607608cbb7d8ebc312109a9e77fc16234f",
        S98G / "sm98-rows-1500.jsonl": "288b2d1d9390233748d6372e50d1f77c688b33d1b80844101e8bf3b2698fc5fa"}
LBL96, LBL98 = "sm96-inst-e31st-st-dec1ep32k/w8+s8", "sm98-inst-e35st-st-dec1ep32k/w8+s8"
BAR, MIN_EFFECT, EARLY_STOP = 0.40, 0.05, 0.25


def report():
    got = hashlib.sha256(REPORT.read_bytes()).hexdigest()
    if got != REPORT_SHA:
        raise SystemExit(f"[sm101-rule] showtime_report.py changed (sha256 {got[:12]} != {REPORT_SHA[:12]}): refusing")
    spec = importlib.util.spec_from_file_location("showtime_report", REPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_pins():
    for path, want in PINS.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"[sm101-rule] reference rows {path.name} changed (sha256 {got[:12]} != {want[:12]}): refusing")


def own(rows, rows1500, label, r):
    return {"e2s": [(rows, label + r.E2S, None)], "normal": [(rows, label, None)],
            "e2s1500": [(rows, label + r.E2S, None), (rows1500, label + r.E2S, None)]}


def fp(x):
    return f"{x['points']:+.3f} [{x['points_ci'][0]:+.3f}, {x['points_ci'][1]:+.3f}]"


def fpct(x):
    return f"{x['pct_of_ref']:+.3f} [{x['pct_of_ref_ci'][0]:+.3f}, {x['pct_of_ref_ci'][1]:+.3f}]"


def bar_ok(x):
    return x["points_ci"][1] <= BAR


def gain_ok(x):
    return x["points_ci"][1] < 0 and x["points"] <= -MIN_EFFECT


def early_decision(x):
    return "stop" if x["points"] > EARLY_STOP else "continue"


def final(r, new, users, big):
    """The final rule on `new` = {"e2s": rows, "normal": rows}: A (bar, normal inputs) and B (vs SM92, show-time)."""
    sm92e = r.load(r.MODELS["sm92"]["e2s"], set(users))
    us = sorted(set(users) & set(new["e2s"]) & set(new["normal"]) & set(sm92e))
    a = r.compare(new["normal"], big, big, us)["instant"]
    b = r.compare(new["e2s"], sm92e, big, us)["instant"]
    verdict = "go" if bar_ok(a) and gain_ok(b) else "no go"
    return {"users": len(us), "A_bar_new_minus_big_normal": a, "A_pass": bar_ok(a), "B_new_minus_sm92_e2s": b,
            "B_pass": gain_ok(b), "verdict": verdict}


def selftest():
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    s98 = own(S98G / "sm98-rows.jsonl", S98G / "sm98-rows-1500.jsonl", LBL98, r)
    s96 = own(S96G / "sm96-rows.jsonl", S96G / "sm96-rows-1500.jsonl", LBL96, r)
    n98 = {v: r.load(s98[v], have) for v in ("e2s", "normal")}
    n96 = {v: r.load(s96[v], have) for v in ("e2s", "normal")}
    n92 = {v: r.load(r.MODELS["sm92"][v], have) for v in ("e2s", "normal")}
    rel_e = r.load(r.MODELS["release"]["e2s"], have)
    f98, f92 = final(r, n98, users, big), final(r, n92, users, big)
    a98, b98 = f98["A_bar_new_minus_big_normal"], f98["B_new_minus_sm92_e2s"]
    us = sorted(have & set(n98["e2s"]) & set(n96["e2s"]) & set(rel_e))
    e1 = r.compare(n98["e2s"], n96["e2s"], big, us)["instant"]
    e2 = r.compare(rel_e, n98["e2s"], big, us)["instant"]
    checks = {
        "sm98_as_new_go": f98["users"] == 722 and a98["points"] == 0.11 and a98["points_ci"] == [0.081, 0.138]
        and b98["points"] == -0.248 and b98["points_ci"] == [-0.270, -0.228] and f98["verdict"] == "go",
        "sm92_as_new_no_go": f92["B_new_minus_sm92_e2s"]["points"] == 0 and f92["verdict"] == "no go"
        and f92["A_bar_new_minus_big_normal"]["points"] == 0.362,
        "early_sm98_vs_sm96_continue": len(us) == 722 and e1["points"] == -0.035 and e1["points_ci"] == [-0.041, -0.029]
        and early_decision(e1) == "continue",
        "early_release_vs_sm98_stop": e2["points"] == 0.554 and early_decision(e2) == "stop",
        "early_threshold_edges": early_decision({"points": 0.25}) == "continue" and early_decision({"points": 0.251}) == "stop",
        "bar_edges": bar_ok({"points_ci": [0.3, 0.40]}) and not bar_ok({"points_ci": [0.3, 0.401]}),
        "gain_edges": gain_ok({"points": -0.05, "points_ci": [-0.07, -0.001]}) and not gain_ok({"points": -0.049, "points_ci": [-0.07, -0.03]})
        and not gain_ok({"points": -0.06, "points_ci": [-0.12, 0.0]})}
    ok = all(checks.values())
    print(f"[sm101-rule] selftest: SM98 as new: A {fp(a98)} (bar <= +0.40), B {fp(b98)} -> {f98['verdict']}; SM92 as new: "
          f"A {fp(f92['A_bar_new_minus_big_normal'])}, B {fp(f92['B_new_minus_sm92_e2s'])} -> {f92['verdict']}; early SM98 - "
          f"SM96 {fp(e1)} -> {early_decision(e1)}; early release - SM98 {fp(e2)} -> {early_decision(e2)}; "
          f"{json.dumps(checks)} -> {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def early(out_dir=HERE, rows=EARLY_ROWS):
    r = report()
    users, big = r.pools()
    have = set(users)
    arm = r.load([(rows, EARLY_ARM + r.E2S, None)], have)
    ref = r.load([(rows, EARLY_REF + r.E2S, None)], have)
    if not (have <= set(arm) and have <= set(ref)):
        print(f"[sm101-rule] early pending: arm {len(have & set(arm))}, ref {len(have & set(ref))} of {len(users)} rule users")
        return 2
    x = r.compare(arm, ref, big, users)["instant"]
    out = {"rule": f"stop if epoch 36 (w64-c1) - SM98 epoch 35 (w64), both constant LR, instant, show-time inputs, "
                   f"{len(users)} users, point estimate > +{EARLY_STOP} points", "arm_label": EARLY_ARM, "ref_label": EARLY_REF,
           "users": len(users), "instant": x, "curve_untrained": r.compare(arm, ref, big, users)["curve"],
           "decision": early_decision(x)}
    (Path(out_dir) / "sm101-early.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = ["# The smaller instant-only model (SM101): early check", "",
          f"Decision: **{out['decision']}**. Epoch 36 of the smaller instant-only model (SM101) (card stream 1 layer, constant "
          f"LR) minus the show-time instant-only model, stage 2 (SM98)'s epoch 35 (constant LR, the cut's source), instant "
          f"head, show-time inputs, {len(users)} users: {fp(x)} points (% of the big model (1 leak-free epoch) normal-input "
          f"instant LogLoss), {x['users_better']} better / {x['users_worse']} worse; arm {x['new']} vs ref {x['ref']}.", "",
          f"Rule (pre-registered): stop if the point estimate is above +{EARLY_STOP} points; otherwise continue to epochs "
          f"37-38 and the cool-down.", ""]
    (Path(out_dir) / "sm101-early.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[sm101-rule] early {out['decision']}: epoch 36 - SM98 epoch 35 show-time instant {fp(x)} points ({len(users)} users)")
    return 0


def decide(out_dir=HERE, rows=ROWS, rows1500=ROWS1500, label=LBL):
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    mine = own(rows, rows1500, label, r)
    new = {v: r.load(mine[v], have) for v in ("e2s", "normal")}
    if not (have <= set(new["e2s"]) and have <= set(new["normal"])):
        print(f"[sm101-rule] pending: new rows show-time {len(have & set(new['e2s']))}, normal {len(have & set(new['normal']))} "
              f"of {len(users)} rule users")
        return 2
    out = {"rule": "go = A (new - big, normal inputs, instant: 95% CI upper end <= +0.40 points) AND B (new - SM92, "
                   "show-time inputs, instant: 95% CI below 0 and points <= -0.05)", "label": label, "users": len(users),
           "big_instant": round(r.big_denominator(big, users, "instant"), 6)}
    f = final(r, new, users, big)
    if f["users"] != len(users):
        print("[sm101-rule] pending: SM92 rows incomplete"); return 2
    out |= f
    srcs = {"sm96": own(S96G / "sm96-rows.jsonl", S96G / "sm96-rows-1500.jsonl", LBL96, r),
            "sm98": own(S98G / "sm98-rows.jsonl", S98G / "sm98-rows-1500.jsonl", LBL98, r),
            "sm92": r.MODELS["sm92"], "release": r.MODELS["release"]}
    ref = {k: {v: r.load(s[v], have) for v in ("e2s", "normal")} for k, s in srcs.items()}

    def pair(a, b):
        us = sorted(set(users) & set(a) & set(b))
        return r.compare(a, b, big, us) if us else None

    sec = {"new_minus_sm96_e2s": pair(new["e2s"], ref["sm96"]["e2s"]), "new_minus_sm98_e2s": pair(new["e2s"], ref["sm98"]["e2s"]),
           "new_minus_release_e2s": pair(new["e2s"], ref["release"]["e2s"]),
           "new_minus_sm96_normal": pair(new["normal"], ref["sm96"]["normal"]),
           "new_minus_sm98_normal": pair(new["normal"], ref["sm98"]["normal"]),
           "new_minus_sm92_normal": pair(new["normal"], ref["sm92"]["normal"]),
           "new_e2s_minus_big_normal": pair(new["e2s"], big),
           "new_e2s_minus_new_normal": pair(new["e2s"], new["normal"])}
    out["secondary_732"] = sec
    q = {}
    if r.POOL1500.exists():
        pool = set(json.loads(r.POOL1500.read_text())["users"])
        rel = r.load(r.MODELS["release"]["e2s1500"], pool)
        s96 = r.load(srcs["sm96"]["e2s1500"], pool)
        s98 = r.load(srcs["sm98"]["e2s1500"], pool)
        nw = r.load(mine["e2s1500"], pool)
        us = sorted(set(nw) & set(rel) & set(s96) & set(s98))
        q["users"] = len(us)
        if us and Path(rows1500).exists():
            q["new_minus_release"] = r.compare(nw, rel, rel, us)
            q["new_minus_sm96"] = r.compare(nw, s96, rel, us)
            q["new_minus_sm98"] = r.compare(nw, s98, rel, us)
            spec = json.loads(r.USERS1500.read_text())
            u732 = set(json.loads(r.U732.read_text())["users"]) | set(json.loads(r.ULONG.read_text())["users"])
            check = sorted(set(spec["users"] if isinstance(spec["users"], list) else []) & u732)
            a = r.load([(rows, label + r.E2S, None)], set(check)); b = r.load([(rows1500, label + r.E2S, None)], set(check))
            d = [max(abs(a[u][h][0] - b[u][h][0]) for h in r.HEADS) for u in check if u in a and u in b]
            q["row_check"] = {"users": len(d), "max_abs_logloss_diff": max(d) if d else None}
    out["pool_1500"] = q
    (Path(out_dir) / "sm101-decision.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    names = {"new_minus_sm96_e2s": ("the show-time instant-only small model (SM96)", "show-time"),
             "new_minus_sm98_e2s": ("the show-time instant-only model, stage 2 (SM98)", "show-time"),
             "new_minus_release_e2s": ("the release (SM63)", "show-time"),
             "new_minus_sm96_normal": ("the show-time instant-only small model (SM96)", "normal"),
             "new_minus_sm98_normal": ("the show-time instant-only model, stage 2 (SM98)", "normal"),
             "new_minus_sm92_normal": ("the show-time retrain of the release (SM92)", "normal"),
             "new_e2s_minus_big_normal": ("the big model (1 leak-free epoch) on normal inputs", "show-time vs normal"),
             "new_e2s_minus_new_normal": ("the new model on normal inputs (its show-time cost)", "show-time vs normal")}
    a, b = f["A_bar_new_minus_big_normal"], f["B_new_minus_sm92_e2s"]
    md = ["# The smaller instant-only model (SM101): decision", "",
          f"Verdict: **{f['verdict']}** ({len(users)} users of the 732 set; points = % of the big model (1 leak-free epoch) "
          f"normal-input instant LogLoss {out['big_instant']}; paired user bootstrap 2,000, seed 0, 95% CI).", "",
          f"- A, the bar: the smaller instant-only model (SM101) minus the big model (1 leak-free epoch), instant, normal "
          f"inputs: {fp(a)} points ({a['new']} vs {a['ref']}); pass needs the CI upper end <= +0.40: "
          f"**{'pass' if f['A_pass'] else 'fail'}**.",
          f"- B, what the fork ships for RWKV-Instant: the smaller instant-only model (SM101) minus the show-time retrain of "
          f"the release (SM92), instant, "
          f"show-time inputs: {fp(b)} points ({b['new']} vs {b['ref']}), {b['users_better']} better / {b['users_worse']} "
          f"worse; pass needs the CI below 0 and <= -0.05: **{'pass' if f['B_pass'] else 'fail'}**.", "",
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
        for k, nm in (("new_minus_release", "new minus the release (SM63)"),
                      ("new_minus_sm96", "new minus the show-time instant-only small model (SM96)"),
                      ("new_minus_sm98", "new minus the show-time instant-only model, stage 2 (SM98)")):
            y = q[k]["instant"]
            md.append(f"| {nm} | {y['new']} | {y['ref']} | {fpct(y)} | {y['users_better']} / {y['users_worse']} |")
        md += ["", f"Row check: {json.dumps(q['row_check'])}", ""]
    else:
        md += ["## 1,500 validation users", "", "Not scored yet.", ""]
    md += ["The curve head is untrained (curve loss weight 0); curve numbers are in the JSON only and not part of the verdict.", ""]
    (Path(out_dir) / "sm101-decision.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[sm101-rule] verdict {f['verdict']}: A new - big normal instant {fp(a)} ({'pass' if f['A_pass'] else 'fail'}); "
          f"B new - SM92 show-time instant {fp(b)} ({'pass' if f['B_pass'] else 'fail'}); new - SM96 show-time "
          f"{fp(sec['new_minus_sm96_e2s']['instant']) if sec['new_minus_sm96_e2s'] else '-'}; 1,500 pool users {q.get('users', 0)}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "decide"
    raise SystemExit({"selftest": selftest, "early": early, "decide": decide}[cmd]())
