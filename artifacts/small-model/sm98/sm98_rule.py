#!/usr/bin/env python
"""The show-time instant-only model, stage 2 (SM98): pre-registered rule (fixed 2026-11-09 before any GPU run; CPU, seconds).

New model: the show-time instant-only small model (SM96)'s constant-LR epoch 31 + 4 more constant-LR instant-only epochs
32..35 on show-time inputs (epoch 31's recipe, the epoch index advanced) + the release's one-epoch 32k Muon cool-down on
show-time inputs from epoch 35 ("show_time_intervals": true, curve loss weight 0; SM96's cool-down except the start).
Label `sm98-inst-e35st-st-dec1ep32k/w8+s8`. Question: do 4 more instant-only epochs on show-time inputs help, as 1 -> 4
more epochs did in the instant-only small model (SM57) (stage 1 -> stage 4: -0.071 points of instant)?

Loaders and the paired bootstrap are the show-time report's (`artifacts/small-model/showtime/showtime_report.py` on main,
sha256 pinned below; `load`, `compare`, `pools`): 2,000 resamples, seed 0, 95% CI, review-weighted LogLoss. The reference
SM96 is read from its own rows (committed on branch sm96, carried into this branch; sha256 pinned, equal to the files in
wt/sm96), never rescored.

PRIMARY (the verdict), 722 users of the 732 set (the users the big model (1 leak-free epoch) normal rows have), instant
head, both on show-time inputs (`e2s-query`): new - the show-time instant-only small model (SM96). Unit: points = % of the
big model (1 leak-free epoch) normal-input instant LogLoss on the same users (0.275757).
  go     = the 95% CI entirely below 0 AND the gain >= 0.05 points (points <= -0.05)
  no go  = otherwise
SECONDARY (reported, no verdict):
  new - the show-time retrain of the release (SM92), show-time inputs;
  new - the instant-only small model (SM57) stage 4, show-time inputs; new - the release (SM63), show-time inputs;
  new - SM96, new - SM92 and new - stage 4 on normal inputs; the new model's own show-time cost (show-time - normal);
  1,500 pool, show-time inputs, unit % of the release's own LogLoss (the `pool1500` convention): new - the release,
  new - SM96, new - stage 4, new - the release by part (ids < 5000 / >= 5000), row check (the check users of the 1,500
  ticket vs their 732-set rows).
The curve head is untrained (curve loss weight 0): curve numbers are in the JSON only, never part of the verdict.

  python sm98_rule.py selftest   -> reproduces SM96's decision from its rows with this rule's code (SM96 - SM92 show-time
                                    instant -0.214 [-0.233, -0.195], 722 users, go; SM96 - SM96 = 0; the 1,500 pool SM96 -
                                    the release -0.481 [-0.534, -0.438], 1,487 users); exit 0 if equal
  python sm98_rule.py decide     -> sm98-decision.{json,md}; exit 2 while the new model's 722-user rows are missing
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
REPORT_SHA = "3cdfed03afd40734d4b1931c244e8c1e73453d281a40c3843e80cd310d70dd9d"   # main 9031ed51 (same pin as SM96)
ROWS, ROWS1500 = G / "sm98-rows.jsonl", G / "sm98-rows-1500.jsonl"
LBL = "sm98-inst-e35st-st-dec1ep32k/w8+s8"
# the reference: the show-time instant-only small model (SM96), its own rows (branch sm96 e54fdb93)
S96G = WT / "artifacts/small-model/sm96/gpu"
S96ROWS, S96ROWS1500 = S96G / "sm96-rows.jsonl", S96G / "sm96-rows-1500.jsonl"
S96ORIG = Path("C:/Users/jonat/workspace/wt/sm96/artifacts/small-model/sm96/gpu")
S96SHA = {"sm96-rows.jsonl": "cc93a32ac9f8ab522d028db342c2f07019874cddf50d7a728b117b3e418dd4f9",
          "sm96-rows-1500.jsonl": "7bc69afbf90f81ef88f275d2d6911f19c7bdfafc9091aca69ef2b62134f5c8f8"}
LBL96 = "sm96-inst-e31st-st-dec1ep32k/w8+s8"
MIN_EFFECT = 0.05


def report():
    got = hashlib.sha256(REPORT.read_bytes()).hexdigest()
    if got != REPORT_SHA:
        raise SystemExit(f"[sm98-rule] showtime_report.py changed (sha256 {got[:12]} != {REPORT_SHA[:12]}): refusing")
    spec = importlib.util.spec_from_file_location("showtime_report", REPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_sm96_rows():
    for name, want in S96SHA.items():
        got = hashlib.sha256((S96G / name).read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"[sm98-rule] SM96 rows {name} changed (sha256 {got[:12]} != {want[:12]}): refusing")


def sm96_sources(r):
    return {"e2s": [(S96ROWS, LBL96 + r.E2S, None)], "normal": [(S96ROWS, LBL96, None)],
            "e2s1500": [(S96ROWS, LBL96 + r.E2S, None), (S96ROWS1500, LBL96 + r.E2S, None)]}


def fp(x):
    return f"{x['points']:+.3f} [{x['points_ci'][0]:+.3f}, {x['points_ci'][1]:+.3f}]"


def fpct(x):
    return f"{x['pct_of_ref']:+.3f} [{x['pct_of_ref_ci'][0]:+.3f}, {x['pct_of_ref_ci'][1]:+.3f}]"


def verdict(x):
    return "go" if x["points_ci"][1] < 0 and x["points"] <= -MIN_EFFECT else "no go"


def selftest():
    r = report()
    check_sm96_rows()
    same_files = {n: hashlib.sha256((S96ORIG / n).read_bytes()).hexdigest() == s for n, s in S96SHA.items()}
    users, big = r.pools()
    have = set(users)
    s96 = r.load(sm96_sources(r)["e2s"], have)
    sm92 = r.load(r.MODELS["sm92"]["e2s"], have)
    us = sorted(have & set(s96) & set(sm92))
    x = r.compare(s96, sm92, big, us)["instant"]
    same = r.compare(s96, s96, big, us)["instant"]
    ok1 = len(us) == 722 and x["points"] == -0.214 and x["points_ci"] == [-0.233, -0.195] and x["new"] == 0.276287 \
        and x["ref"] == 0.276876 and verdict(x) == "go"
    ok2 = same["points"] == 0 and same["points_ci"] == [0, 0] and verdict(same) == "no go"
    pool = set(json.loads(r.POOL1500.read_text())["users"])
    rel = r.load(r.MODELS["release"]["e2s1500"], pool)
    s96p = r.load(sm96_sources(r)["e2s1500"], pool)
    s4p = r.load(r.MODELS["s4"]["e2s1500"], pool)
    up = sorted(set(s96p) & set(s4p) & set(rel))
    y = r.compare(s96p, rel, rel, up)["instant"]
    ok3 = len(up) == 1487 and y["pct_of_ref"] == -0.481 and y["pct_of_ref_ci"] == [-0.534, -0.438]
    ok = ok1 and ok2 and ok3 and all(same_files.values())
    print(f"[sm98-rule] selftest: {len(us)} users, SM96 - SM92 show-time instant {fp(x)} (SM96 decision -0.214 [-0.233, -0.195]), "
          f"SM96 {x['new']} SM92 {x['ref']}, rule on it: {verdict(x)}; SM96 - SM96 {fp(same)} -> {verdict(same)}; 1,500 pool "
          f"{len(up)} users SM96 - the release {fpct(y)} % (decision -0.481 [-0.534, -0.438]); SM96 rows == wt/sm96 files "
          f"{same_files} -> {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def decide():
    r = report()
    check_sm96_rows()
    users, big = r.pools()
    have = set(users)
    E2S = r.E2S
    new_e = r.load([(ROWS, LBL + E2S, None)], have)
    new_n = r.load([(ROWS, LBL, None)], have)
    if not set(users) <= set(new_e):
        print(f"[sm98-rule] pending: new show-time rows on {len(set(users) & set(new_e))} of {len(users)} rule users")
        return 2
    srcs = {k: {"e2s": r.MODELS[k]["e2s"], "normal": r.MODELS[k]["normal"]} for k in ("sm92", "s4", "release")}
    srcs["sm96"] = sm96_sources(r)
    ref = {k: {v: r.load(s[v], have) for v in ("e2s", "normal")} for k, s in srcs.items()}
    out = {"rule": "new - SM96, instant, show-time inputs, 722 users: go = 95% CI below 0 and points <= -0.05",
           "label": LBL, "reference_label": LBL96, "users": len(users),
           "big_instant": round(r.big_denominator(big, users, "instant"), 6)}

    def pair(a, b):
        us = sorted(set(users) & set(a) & set(b))
        return r.compare(a, b, big, us) if us else None

    out["primary_new_minus_sm96_e2s"] = p = pair(new_e, ref["sm96"]["e2s"])
    if p is None or p["users"] != len(users):
        print("[sm98-rule] pending: SM96 rows incomplete"); return 2
    out["verdict"] = verdict(p["instant"])
    sec = {"new_minus_sm92_e2s": pair(new_e, ref["sm92"]["e2s"]), "new_minus_s4_e2s": pair(new_e, ref["s4"]["e2s"]),
           "new_minus_release_e2s": pair(new_e, ref["release"]["e2s"])}
    if new_n:
        sec |= {"new_minus_sm96_normal": pair(new_n, ref["sm96"]["normal"]),
                "new_minus_sm92_normal": pair(new_n, ref["sm92"]["normal"]),
                "new_minus_s4_normal": pair(new_n, ref["s4"]["normal"]),
                "new_e2s_minus_new_normal": pair(new_e, new_n)}
    out["secondary_732"] = sec
    # 1,500 pool, show-time inputs, % of the release's own LogLoss (the report's pool1500 convention)
    q = {}
    if r.POOL1500.exists():
        pool = set(json.loads(r.POOL1500.read_text())["users"])
        rel = r.load(r.MODELS["release"]["e2s1500"], pool)
        s4 = r.load(r.MODELS["s4"]["e2s1500"], pool)
        s96 = r.load(sm96_sources(r)["e2s1500"], pool)
        new = r.load([(ROWS, LBL + E2S, None), (ROWS1500, LBL + E2S, None)], pool)
        us = sorted(set(new) & set(s4) & set(rel) & set(s96))
        q["users"] = len(us)
        if us and ROWS1500.exists():
            q["new_minus_release"] = r.compare(new, rel, rel, us)
            q["new_minus_sm96"] = r.compare(new, s96, rel, us)
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
    (HERE / "sm98-decision.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    names = {"new_minus_sm92_e2s": ("the show-time retrain of the release (SM92)", "show-time"),
             "new_minus_s4_e2s": ("the instant-only small model (SM57) stage 4", "show-time"),
             "new_minus_release_e2s": ("the release (SM63)", "show-time"),
             "new_minus_sm96_normal": ("the show-time instant-only small model (SM96)", "normal"),
             "new_minus_sm92_normal": ("the show-time retrain of the release (SM92)", "normal"),
             "new_minus_s4_normal": ("the instant-only small model (SM57) stage 4", "normal"),
             "new_e2s_minus_new_normal": ("the new model on normal inputs (its show-time cost)", "show-time vs normal")}
    pi = p["instant"]
    md = ["# The show-time instant-only model, stage 2 (SM98): decision", "",
          f"Verdict: **{out['verdict']}**. The show-time instant-only model, stage 2 (SM98) minus the show-time instant-only "
          f"small model (SM96), instant head, show-time inputs, {p['users']} users: {fp(pi)} points (% of the big model "
          f"(1 leak-free epoch) normal-input instant LogLoss {out['big_instant']}), {pi['users_better']} better / "
          f"{pi['users_worse']} worse; go needs the CI below 0 and <= -0.05.", "",
          f"New {pi['new']} vs the show-time instant-only small model (SM96) {pi['ref']} (instant LogLoss, show-time inputs).", "",
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
                      ("new_minus_s4", "new minus the instant-only small model (SM57) stage 4")):
            y = q[k]["instant"]
            md.append(f"| {nm} | {y['new']} | {y['ref']} | {fpct(y)} | {y['users_better']} / {y['users_worse']} |")
        for part, x in q["by_part"].items():
            if x:
                y = x["instant"]
                md.append(f"| new minus the release (SM63), {part} ({x['users']} users) | {y['new']} | {y['ref']} | {fpct(y)} | {y['users_better']} / {y['users_worse']} |")
        md += ["", f"Row check: {json.dumps(q['row_check'])}", ""]
    else:
        md += ["## 1,500 validation users", "", "Not scored yet.", ""]
    md += ["The curve head is untrained (curve loss weight 0); curve numbers are in the JSON only and not part of the verdict.", ""]
    (HERE / "sm98-decision.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[sm98-rule] verdict {out['verdict']}: new - SM96 show-time instant {fp(pi)} points ({p['users']} users); "
          f"new - SM92 show-time {fp(sec['new_minus_sm92_e2s']['instant']) if sec['new_minus_sm92_e2s'] else '-'}; "
          f"1,500 pool users {q.get('users', 0)}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "decide"
    raise SystemExit(selftest() if cmd == "selftest" else decide())
