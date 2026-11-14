#!/usr/bin/env python
"""The show-time instant-only big model (SM99): pre-registered rules (fixed 2026-11-11 before any GPU run; CPU, seconds).

New model: the show-time big model (SM91 step 2) + big-model epoch 1 (+ epoch 2 unless the early rule stops), each the
big model's own recipe (fresh AdamW, warmup 200, 3.5e-4 -> 0 one_minus_sin over the epoch, users 5000-10000, 2 x 8,192
carried state, no teacher) on show-time inputs with curve loss weight 0 (instant-only). Labels
`sm99-big-inst-e1/w8+s8` and `sm99-big-inst-e2/w8+s8` (show-time rows `@e2s-query`), rows gpu/sm99-rows.jsonl.

Loaders and the paired bootstrap are the show-time report's (`artifacts/small-model/showtime/showtime_report.py` on main,
sha256 pinned, the same pin as SM96 / SM98 / SM101): 2,000 resamples, seed 0, 95% CI, review-weighted LogLoss, the
instant head, 722 users of the 732 set (the users the big model (1 leak-free epoch) normal rows have). Unit: % of the big
model (1 leak-free epoch) normal-input instant LogLoss on the same users (0.275757). Reference rows are read from their
own files (sha256 pinned), never rescored: the show-time instant-only model, stage 2 (SM98), the show-time instant-only
small model (SM96) and the show-time big model (SM91 step 2) (rows carried in this branch); the show-time retrain of the
release (SM92) and the release (SM63) (the report's sources).

EARLY RULE (`early`, after epoch 1; epoch 1 is a finished model: its learning rate has decayed to 0):
  x = epoch 1 - SM98, instant head, show-time inputs, 722 users.
  continue  if the point estimate x < 0 (epoch 1 is better than SM98): run epoch 2.
  stop      otherwise (x >= 0): no epoch 2; the final read-out runs on epoch 1.
  Why: a second epoch of the same recipe on the same users is expected to add at most ~0.05% (the better teacher (SM65)'s
  added big-model epoch: -0.03% instant on normal inputs; SM98's 4 added small-model epochs: -0.035%), so from x >= 0 it
  cannot plausibly reach the -0.05% line with a CI below 0; stopping saves ~9.3 GPU h.
FINAL (`decide`): the final model = epoch 2 if the early rule said continue, else epoch 1 (fixed, not the better of the two).
  Primary (the verdict, a measurement): final - SM98, instant, show-time inputs, 722 users:
    "ceiling above SM98" (points <= -0.05 AND the 95% CI below 0): worth a teacher experiment next;
    otherwise "at the ceiling": the small instant-only model is at the big-model ceiling.
  Read-out with it (no verdict): final - the show-time big model (SM91 step 2) (show-time; normal); final - SM96, final
  - SM92, final - the release (SM63) (show-time); final - SM98 (normal); final normal - the big model (1 leak-free epoch)
  normal; the final model's show-time cost; epoch 1 - SM98 and epoch 2 - epoch 1 (show-time) when epoch 2 ran; the
  1,500 validation users on show-time inputs (% of the release's own LogLoss: final - SM98, final - SM91 step 2, final
  - SM96, final - the release; row check). The curve head is untrained (weight 0): never part of a verdict.

  python sm99_rule.py selftest   -> the rules' code on known rows (exit 0 if all equal)
  python sm99_rule.py early      -> sm99-early.{json,md}; exit 2 while rows are missing
  python sm99_rule.py decide     -> sm99-decision.{json,md}; exit 2 while the final model's 722-user rows are missing
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
REPORT_SHA = "3cdfed03afd40734d4b1931c244e8c1e73453d281a40c3843e80cd310d70dd9d"   # main 9031ed51 (same pin as SM96 / SM98 / SM101)
ROWS, ROWS1500 = G / "sm99-rows.jsonl", G / "sm99-rows-1500.jsonl"
LBL1, LBL2 = "sm99-big-inst-e1/w8+s8", "sm99-big-inst-e2/w8+s8"
S96G, S98G = WT / "artifacts/small-model/sm96/gpu", WT / "artifacts/small-model/sm98/gpu"
S91 = WT / "artifacts/small-model/sm91s2/gpu/sm91s2-inputs.jsonl"
PINS = {S96G / "sm96-rows.jsonl": "cc93a32ac9f8ab522d028db342c2f07019874cddf50d7a728b117b3e418dd4f9",
        S96G / "sm96-rows-1500.jsonl": "7bc69afbf90f81ef88f275d2d6911f19c7bdfafc9091aca69ef2b62134f5c8f8",
        S98G / "sm98-rows.jsonl": "d4cc71b8a1ecc3d3250c19c58952b3607608cbb7d8ebc312109a9e77fc16234f",
        S98G / "sm98-rows-1500.jsonl": "288b2d1d9390233748d6372e50d1f77c688b33d1b80844101e8bf3b2698fc5fa",
        S91: "72912af5e88a79464b0179abcfc75582e69f00caba4e216f1a7eae34ba6b8017"}
LBL96, LBL98, LBL91 = "sm96-inst-e31st-st-dec1ep32k/w8+s8", "sm98-inst-e35st-st-dec1ep32k/w8+s8", "sm91s2-big-showtime/w8+s8"
MIN_EFFECT = 0.05
NAMES = {"sm98": "the show-time instant-only model, stage 2 (SM98)", "sm91s2": "the show-time big model (SM91 step 2)",
         "sm96": "the show-time instant-only small model (SM96)", "sm92": "the show-time retrain of the release (SM92)",
         "release": "the release (SM63)", "big": "the big model (1 leak-free epoch)"}


def report():
    got = hashlib.sha256(REPORT.read_bytes()).hexdigest()
    if got != REPORT_SHA:
        raise SystemExit(f"[sm99-rule] showtime_report.py changed (sha256 {got[:12]} != {REPORT_SHA[:12]}): refusing")
    spec = importlib.util.spec_from_file_location("showtime_report", REPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_pins():
    for path, want in PINS.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"[sm99-rule] reference rows {path.name} changed (sha256 {got[:12]} != {want[:12]}): refusing")


def own(rows, rows1500, label, r):
    return {"e2s": [(rows, label + r.E2S, None)], "normal": [(rows, label, None)],
            "e2s1500": [(rows, label + r.E2S, None), (rows1500, label + r.E2S, None)]}


def refs(r, rows1500=ROWS1500):
    return {"sm96": own(S96G / "sm96-rows.jsonl", S96G / "sm96-rows-1500.jsonl", LBL96, r),
            "sm98": own(S98G / "sm98-rows.jsonl", S98G / "sm98-rows-1500.jsonl", LBL98, r),
            "sm91s2": own(S91, rows1500, LBL91, r), "sm92": r.MODELS["sm92"], "release": r.MODELS["release"]}


def fp(x):
    return f"{x['points']:+.3f}% [{x['points_ci'][0]:+.3f}, {x['points_ci'][1]:+.3f}]"


def fpct(x):
    return f"{x['pct_of_ref']:+.3f}% [{x['pct_of_ref_ci'][0]:+.3f}, {x['pct_of_ref_ci'][1]:+.3f}]"


def early_decision(x):
    return "continue" if x["points"] < 0 else "stop"


def ceiling(x):
    return "ceiling above SM98" if x["points"] <= -MIN_EFFECT and x["points_ci"][1] < 0 else "at the ceiling"


def primary(r, new_e2s, users, big):
    s98 = r.load(refs(r)["sm98"]["e2s"], set(users))
    us = sorted(set(users) & set(new_e2s) & set(s98))
    x = r.compare(new_e2s, s98, big, us)["instant"]
    return {"users": len(us), "new_minus_sm98_e2s": x, "verdict": ceiling(x)}


def selftest():
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    src = refs(r)
    n = {k: {v: r.load(src[k][v], have) for v in ("e2s", "normal")} for k in ("sm98", "sm96", "sm91s2")}
    p98 = primary(r, n["sm98"]["e2s"], users, big)
    p96 = primary(r, n["sm96"]["e2s"], users, big)
    p91 = primary(r, n["sm91s2"]["e2s"], users, big)
    better = {u: {h: (v[h][0] * 0.997, v[h][1]) for h in v} for u, v in n["sm98"]["e2s"].items()}   # 0.3% better everywhere
    pb = primary(r, better, users, big)
    x91 = p91["new_minus_sm98_e2s"]
    checks = {
        "sm98_against_itself_zero_at_ceiling_stop": p98["users"] == 722 and p98["new_minus_sm98_e2s"]["points"] == 0
        and p98["verdict"] == "at the ceiling" and early_decision(p98["new_minus_sm98_e2s"]) == "stop",
        "sm96_minus_sm98_plus0035_at_ceiling_stop": p96["new_minus_sm98_e2s"]["points"] == 0.035
        and p96["new_minus_sm98_e2s"]["points_ci"] == [0.029, 0.041] and p96["verdict"] == "at the ceiling"
        and early_decision(p96["new_minus_sm98_e2s"]) == "stop",
        "sm91s2_minus_sm98_known": p91["users"] == 722 and (x91["points"], x91["points_ci"]) == SM91S2_MINUS_SM98 and p91["verdict"] == "at the ceiling",
        "synthetic_0p3pct_better_is_above": pb["verdict"] == "ceiling above SM98" and pb["new_minus_sm98_e2s"]["points"] < -0.25
        and early_decision(pb["new_minus_sm98_e2s"]) == "continue",
        "early_edges": early_decision({"points": -0.001}) == "continue" and early_decision({"points": 0.0}) == "stop"
        and early_decision({"points": 0.001}) == "stop",
        "ceiling_edges": ceiling({"points": -0.05, "points_ci": [-0.07, -0.001]}) == "ceiling above SM98"
        and ceiling({"points": -0.049, "points_ci": [-0.07, -0.03]}) == "at the ceiling"
        and ceiling({"points": -0.06, "points_ci": [-0.12, 0.0]}) == "at the ceiling"}
    checks["end_to_end_early_and_decide"] = end_to_end(r)
    ok = all(checks.values())
    print(f"[sm99-rule] selftest: SM98 - SM98 {fp(p98['new_minus_sm98_e2s'])} -> {p98['verdict']}; SM96 - SM98 "
          f"{fp(p96['new_minus_sm98_e2s'])} -> {p96['verdict']} / early {early_decision(p96['new_minus_sm98_e2s'])}; the show-time "
          f"big model (SM91 step 2) - SM98 {fp(x91)} -> {p91['verdict']} / early {early_decision(x91)}; synthetic 0.3% better "
          f"{fp(pb['new_minus_sm98_e2s'])} -> {pb['verdict']}; {json.dumps(checks)} -> {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def end_to_end(r):
    """early + decide on SM98's own rows relabelled as epoch 1 (and SM96's 1,500 rows as the show-time big model's):
    early must say stop (0 is not < 0), decide must pick epoch 1 and read 0 -> at the ceiling, with every table filled."""
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
        rows, rows1500 = t / "rows.jsonl", t / "rows-1500.jsonl"
        relabel(S98G / "sm98-rows.jsonl", rows, {LBL98: LBL1})
        relabel(S98G / "sm98-rows-1500.jsonl", rows1500, {LBL98: LBL1})
        relabel(S96G / "sm96-rows-1500.jsonl", rows1500, {LBL96: LBL91})
        with redirect_stdout(io.StringIO()):
            a = early(t, rows)
            b = decide(t, rows, rows1500)
        e = json.loads((t / "sm99-early.json").read_text())
        d = json.loads((t / "sm99-decision.json").read_text())
        q = d["pool_1500"]
        return (a == 0 and b == 0 and e["decision"] == "stop" and d["final_label"] == LBL1 and d["verdict"] == "at the ceiling"
                and d["new_minus_sm98_e2s"]["points"] == 0 and all(v is not None for v in d["secondary_732"].values())
                and q.get("users", 0) > 1400 and q["new_minus_sm98"]["instant"]["pct_of_ref"] == 0
                and q.get("users_with_sm91s2", 0) == q["users"] and q["row_check"]["new"]["max_abs_logloss_diff"] == 0.0
                and (t / "sm99-decision.md").exists())


SM91S2_MINUS_SM98 = (-0.003, [-0.026, 0.019])   # the show-time big model (SM91 step 2) - SM98, show-time instant, 722 users (selftest pin)


def early(out_dir=HERE, rows=ROWS):
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    arm = r.load([(rows, LBL1 + r.E2S, None)], have)
    if not have <= set(arm):
        print(f"[sm99-rule] early pending: epoch 1 show-time rows {len(have & set(arm))} of {len(users)} rule users")
        return 2
    p = primary(r, arm, users, big)
    x = p["new_minus_sm98_e2s"]
    s91 = r.load(refs(r)["sm91s2"]["e2s"], have)
    y = r.compare(arm, s91, big, sorted(have & set(s91)))["instant"]
    out = {"rule": f"continue to epoch 2 if epoch 1 - SM98 (instant, show-time inputs, {p['users']} users) has a point "
                   f"estimate < 0; stop otherwise (the final read-out then runs on epoch 1)", "arm_label": LBL1,
           "users": p["users"], "instant": x, "decision": early_decision(x), "epoch1_minus_sm91s2_e2s": y,
           "epoch1_verdict_if_final": p["verdict"]}
    (Path(out_dir) / "sm99-early.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = ["# The show-time instant-only big model (SM99): early rule", "",
          f"Decision: **{out['decision']}**. Epoch 1 of the show-time instant-only big model (SM99) minus the show-time "
          f"instant-only model, stage 2 (SM98), instant head, show-time inputs, {p['users']} users: {fp(x)} (% of the big "
          f"model (1 leak-free epoch) normal-input instant LogLoss), {x['users_better']} better / {x['users_worse']} worse; "
          f"{x['new']} vs {x['ref']}.", "",
          f"For reference: epoch 1 minus the show-time big model (SM91 step 2), its start, show-time instant: {fp(y)}.", "",
          "Rule (pre-registered): continue to epoch 2 if the point estimate is below 0; otherwise stop, and the final "
          "read-out runs on epoch 1.", ""]
    (Path(out_dir) / "sm99-early.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[sm99-rule] early {out['decision']}: epoch 1 - SM98 show-time instant {fp(x)}; epoch 1 - SM91 step 2 {fp(y)} ({p['users']} users)")
    return 0


def final_label(out_dir=HERE):
    e = Path(out_dir) / "sm99-early.json"
    if not e.exists():
        return None
    return LBL2 if json.loads(e.read_text())["decision"] == "continue" else LBL1


def decide(out_dir=HERE, rows=ROWS, rows1500=ROWS1500):
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    label = final_label(out_dir)
    if label is None:
        print("[sm99-rule] pending: no early decision (sm99-early.json)"); return 2
    mine = own(rows, rows1500, label, r)
    new = {v: r.load(mine[v], have) for v in ("e2s", "normal")}
    if not (have <= set(new["e2s"]) and have <= set(new["normal"])):
        print(f"[sm99-rule] pending: {label} rows show-time {len(have & set(new['e2s']))}, normal "
              f"{len(have & set(new['normal']))} of {len(users)} rule users")
        return 2
    out = {"rule": "final = epoch 2 if the early rule said continue, else epoch 1; verdict on final - SM98 (instant, "
                   "show-time inputs): 'ceiling above SM98' if points <= -0.05 and the 95% CI below 0, else 'at the ceiling'",
           "final_label": label, "users": len(users), "big_instant": round(r.big_denominator(big, users, "instant"), 6)}
    p = primary(r, new["e2s"], users, big)
    out |= p
    src = refs(r, rows1500)
    ref = {k: {v: r.load(s[v], have) for v in ("e2s", "normal")} for k, s in src.items()}

    def pair(a, b):
        us = sorted(set(users) & set(a) & set(b))
        return r.compare(a, b, big, us) if us else None

    sec = {"new_minus_sm91s2_e2s": pair(new["e2s"], ref["sm91s2"]["e2s"]),
           "new_minus_sm91s2_normal": pair(new["normal"], ref["sm91s2"]["normal"]),
           "new_minus_sm96_e2s": pair(new["e2s"], ref["sm96"]["e2s"]),
           "new_minus_sm92_e2s": pair(new["e2s"], ref["sm92"]["e2s"]),
           "new_minus_release_e2s": pair(new["e2s"], ref["release"]["e2s"]),
           "new_minus_sm98_normal": pair(new["normal"], ref["sm98"]["normal"]),
           "new_normal_minus_big_normal": pair(new["normal"], big),
           "new_e2s_minus_new_normal": pair(new["e2s"], new["normal"])}
    if label == LBL2:
        e1 = r.load([(rows, LBL1 + r.E2S, None)], have)
        sec["epoch1_minus_sm98_e2s"] = pair(e1, ref["sm98"]["e2s"])
        sec["epoch2_minus_epoch1_e2s"] = pair(new["e2s"], e1)
    out["secondary_732"] = sec
    q = {}
    if r.POOL1500.exists():
        pool = set(json.loads(r.POOL1500.read_text())["users"])
        rel = r.load(r.MODELS["release"]["e2s1500"], pool)
        other = {k: r.load(src[k]["e2s1500"], pool) for k in ("sm98", "sm91s2", "sm96")}
        nw = r.load(mine["e2s1500"], pool)
        us = sorted(set(nw) & set(rel) & set(other["sm98"]) & set(other["sm96"]))
        q["users"] = len(us)
        if us and Path(rows1500).exists():
            q["new_minus_release"] = r.compare(nw, rel, rel, us)
            q["new_minus_sm98"] = r.compare(nw, other["sm98"], rel, us)
            q["new_minus_sm96"] = r.compare(nw, other["sm96"], rel, us)
            u91 = sorted(set(us) & set(other["sm91s2"]))
            q["users_with_sm91s2"] = len(u91)
            if u91:
                q["new_minus_sm91s2"] = r.compare(nw, other["sm91s2"], rel, u91)
            spec = json.loads(r.USERS1500.read_text())
            u732 = set(json.loads(r.U732.read_text())["users"]) | set(json.loads(r.ULONG.read_text())["users"])
            check = sorted(set(spec["users"] if isinstance(spec["users"], list) else []) & u732)
            rc = {}
            for name, (a_rows, lb) in {"new": (rows, label), "sm91s2": (S91, LBL91)}.items():
                a = r.load([(a_rows, lb + r.E2S, None)], set(check)); b = r.load([(rows1500, lb + r.E2S, None)], set(check))
                d = [max(abs(a[u][h][0] - b[u][h][0]) for h in r.HEADS) for u in check if u in a and u in b]
                rc[name] = {"users": len(d), "max_abs_logloss_diff": max(d) if d else None}
            q["row_check"] = rc
    out["pool_1500"] = q
    (Path(out_dir) / "sm99-decision.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    names = {"new_minus_sm91s2_e2s": (NAMES["sm91s2"], "show-time"), "new_minus_sm91s2_normal": (NAMES["sm91s2"], "normal"),
             "new_minus_sm96_e2s": (NAMES["sm96"], "show-time"), "new_minus_sm92_e2s": (NAMES["sm92"], "show-time"),
             "new_minus_release_e2s": (NAMES["release"], "show-time"), "new_minus_sm98_normal": (NAMES["sm98"], "normal"),
             "new_normal_minus_big_normal": (NAMES["big"], "normal"),
             "new_e2s_minus_new_normal": ("the final model on normal inputs (its show-time cost)", "show-time vs normal"),
             "epoch1_minus_sm98_e2s": ("epoch 1 minus " + NAMES["sm98"], "show-time"),
             "epoch2_minus_epoch1_e2s": ("epoch 2 minus epoch 1", "show-time")}
    x = p["new_minus_sm98_e2s"]
    which = "epoch 2" if label == LBL2 else "epoch 1 (the early rule stopped before epoch 2)"
    md = ["# The show-time instant-only big model (SM99): decision", "",
          f"Verdict: **{p['verdict']}** (final model: {which}; {len(users)} users of the 732 set; % of the big model (1 "
          f"leak-free epoch) normal-input instant LogLoss {out['big_instant']}; paired user bootstrap 2,000, seed 0, 95% CI).", "",
          f"- Primary: the show-time instant-only big model (SM99) minus the show-time instant-only model, stage 2 (SM98), "
          f"instant, show-time inputs: {fp(x)} ({x['new']} vs {x['ref']}), {x['users_better']} better / {x['users_worse']} "
          f"worse. \"Ceiling above SM98\" needs <= -0.05% with the CI below 0 (then a teacher experiment is worth it); "
          f"otherwise the small instant-only model is at the big-model ceiling.", ""]
    s91 = sec["new_minus_sm91s2_e2s"]
    if s91:
        md += [f"- With it: the show-time instant-only big model (SM99) minus the show-time big model (SM91 step 2), instant, "
               f"show-time inputs: {fp(s91['instant'])} ({s91['instant']['new']} vs {s91['instant']['ref']}).", ""]
    md += ["## Read-out (732 set, instant head, %)", "", "| minus | inputs | new | ref | % [95% CI] | better / worse |",
           "|---|---|---|---|---|---|"]
    for k, v in sec.items():
        if v:
            y = v["instant"]
            md.append(f"| {names[k][0]} | {names[k][1]} | {y['new']} | {y['ref']} | {fp(y)} | {y['users_better']} / {y['users_worse']} |")
    md.append("")
    if q.get("new_minus_release"):
        md += [f"## 1,500 validation users, show-time inputs ({q['users']} users), % of the release's own LogLoss", "",
               "| comparison | new | ref | % [95% CI] | better / worse |", "|---|---|---|---|---|"]
        for k, nm in (("new_minus_sm98", "new minus " + NAMES["sm98"]), ("new_minus_sm91s2", "new minus " + NAMES["sm91s2"]),
                      ("new_minus_sm96", "new minus " + NAMES["sm96"]), ("new_minus_release", "new minus " + NAMES["release"])):
            if q.get(k):
                y = q[k]["instant"]
                md.append(f"| {nm} | {y['new']} | {y['ref']} | {fpct(y)} | {y['users_better']} / {y['users_worse']} |")
        md += ["", f"Users with rows of the show-time big model (SM91 step 2): {q.get('users_with_sm91s2')}. Row check: "
               f"{json.dumps(q['row_check'])}", ""]
    else:
        md += ["## 1,500 validation users", "", "Not scored yet.", ""]
    md += ["The curve head is untrained (curve loss weight 0); curve numbers are in the JSON only and not part of the verdict.", ""]
    (Path(out_dir) / "sm99-decision.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[sm99-rule] verdict {p['verdict']} ({which}): new - SM98 show-time instant {fp(x)}; new - SM91 step 2 "
          f"{fp(s91['instant']) if s91 else '-'}; 1,500 pool users {q.get('users', 0)}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "decide"
    raise SystemExit({"selftest": selftest, "early": early, "decide": decide}[cmd]())
