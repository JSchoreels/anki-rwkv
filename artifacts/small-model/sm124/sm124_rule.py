#!/usr/bin/env python
"""The show-time two-head cool-down for Tiny (SM124): pre-registered rules (fixed 2026-11-17 before any GPU run of the
training; CPU, seconds). The code is the show-time two-head cool-down for Small (SM118)'s sm118_rule.py with the
reference changed from the show-time retrain of the release (SM92) to the Tiny two-head model (SM106).

New model: the recipe of the show-time two-head cool-down for Small (SM118) (go: curve -0.114%, instant -0.100% vs SM92)
applied to the Tiny two-head model. Start = the Tiny two-head model (SM106)'s constant-LR epoch 29 (w64-c1, read only;
SM106's own one-epoch cool-down started there); one one_minus_sin decay 7e-4 -> 0 over 2 x 581.9M reviews, fresh Muon,
show-time inputs, BOTH heads trained (no curve loss weight), the show-time big model (SM91s2)'s show-time KD replay
targets (curve and grade columns, alpha 0.9, decay alpha 0.5; the 8,000-user folder SM118 exported, reused), split into
pass 1 (== SM106's cool-down except lr_resume and the targets; == SM118's pass 1 except the start and the architecture,
sm124_pair.py p1) and pass 2 (pass 1's end, the same rows in a new order, sm124_pair.py p2).
Labels: final `sm124-c1-e29st-dec2ep32k-t91s2/w8+s8` (gpu/sm124-rows.jsonl, show-time rows `@e2s-query`); early stop
`sm124-dec2ep-p1/w8+s8@e2s-query` (pass 1's final model) and `sm106-dec-u4000/w8+s8@e2s-query` (SM106's own cool-down
snapshot after 4,000 of 8,000 users, scored by this run's launcher; master sha256 pinned there) in
gpu/sm124-mid-rows.jsonl. References are read from their own committed rows (sha256 pinned), never rescored: SM106
(show-time 732 set and 1,500 pool, copied byte for byte from wt/sm106), SM92, SM91s2, SM118 (Small, the same recipe),
SM101 and SM115 (instant read-outs), SM91s2's 1,500-pool rows (sm99-rows-1500.jsonl), the release from the show-time
report.

Loaders and the paired bootstrap are the show-time report's (`artifacts/small-model/showtime/showtime_report.py` on
main, sha256 pinned, the same pin as SM96 ... SM118): 2,000 resamples, seed 0, 95% CI, review-weighted LogLoss, 722
users of the 732 set. Unit ("%"): % of the big model (1 leak-free epoch)'s normal-input LogLoss of the same head on
the same users (curve 0.298966, instant 0.275757), as SM113 ... SM118. (SM106's own decision used the big model's
show-time LogLoss as the unit; its numbers are ~0.2-0.5% larger in that unit.)

Per head h (curve, instant), x_h = new - SM106 on show-time inputs:
  gain(h)       = the 95% CI of x_h entirely below 0 AND x_h <= -0.05%
  regression(h) = x_h >= +0.05% OR the 95% CI of x_h entirely above 0

EARLY STOP (`mid`, after pass 1 = half of the reviews, learning rate 29.29% of its peak):
  y_h = SM124's pass-1 final model - SM106's own cool-down snapshot after 4,000 of 8,000 users (284,944,410 of
  581,942,724 reviews, learning rate 30.45% of its peak at its last update: the nearest snapshot to the matching point
  of SM106's one-epoch schedule; SM106's cool-down ran in one attempt), both heads, show-time inputs, 722 users.
  continue  if (y_curve < 0 OR y_instant < 0) AND y_curve < +0.05% AND y_instant < +0.05% (point estimates)
  stop      otherwise: verdict "no go (stopped after pass 1)"; saves ~6-8 GPU h.
  Read-out (no verdict): pass 1 - SM106's final model.
FINAL (`decide`):
  go     = gain on at least one head AND regression on neither head;
  no go  = otherwise.
  Read-out with it (no verdict), both heads: the share of the teacher gap closed, (SM124 - SM106) / (SM91s2 - SM106);
  SM124 - SM91s2 (the teacher), - the release (SM63), - SM92, - SM118 (show-time); SM124 - the big model (1 leak-free
  epoch) on show-time and on normal inputs (the smaller-model bar: curve <= +0.5%, instant <= +0.4%, point estimates
  as the 10-29 pruning bar, CIs reported); the show-time cost; the early-stop pairs; instant only: SM124 - the Tiny
  Instant-Only model (SM101), - the longer cool-down for Tiny (SM115). SM106 has no normal-input rows, so there is no
  normal-input pair against it. The 1,500 validation users on show-time inputs (% of the release's own LogLoss):
  SM124 - SM106 (by id part), - SM91s2, - SM118, - the release, and the row checks.

  python sm124_rule.py selftest   -> the rules' code on known rows (exit 0 if all equal)
  python sm124_rule.py mid        -> sm124-mid.{json,md}; exit 2 while rows are missing
  python sm124_rule.py decide     -> sm124-decision.{json,md}; exit 2 while the final model's 722-user rows are missing
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
REPORT_SHA = "3cdfed03afd40734d4b1931c244e8c1e73453d281a40c3843e80cd310d70dd9d"   # main 9031ed51 (same pin as SM96 ... SM118)
ROWS, ROWS1500, MIDROWS = G / "sm124-rows.jsonl", G / "sm124-rows-1500.jsonl", G / "sm124-mid-rows.jsonl"
LBL = "sm124-c1-e29st-dec2ep32k-t91s2/w8+s8"
MID_NEW, MID_REF = "sm124-dec2ep-p1/w8+s8", "sm106-dec-u4000/w8+s8"
SMG = WT / "artifacts/small-model"
R106, R106_1500 = SMG / "sm106/gpu/sm106-rows.jsonl", SMG / "sm106/gpu/sm106-rows-1500.jsonl"
R92, R91 = SMG / "sm92/gpu/sm92-inputs.jsonl", SMG / "sm91s2/gpu/sm91s2-inputs.jsonl"
R91_1500 = SMG / "sm99/gpu/sm99-rows-1500.jsonl"
R118, R118_1500 = SMG / "sm118/gpu/sm118-rows.jsonl", SMG / "sm118/gpu/sm118-rows-1500.jsonl"
R101, R115 = SMG / "sm101/gpu/sm101-rows.jsonl", SMG / "sm115/gpu/sm115-rows.jsonl"
PINS = {R106: "5e38b4ebae16545be4b05f262a7d25849517d43a91910c576e97db0d4abb1f27",       # == wt/sm106's own copy
        R106_1500: "77c4df4afdc1eaa8953cc7def38b8ce38c07d57dc55accf4d3b032b37d2da3e8",  # == wt/sm106's own copy
        R92: "5c714ea0f1c328a053e21c6ac0cbabfb0397ef68bbda6d861201d61ad475adff",       # same pin as SM118
        R91: "e156b6fd907857b22f96d64035c06417b063c88d416a959bc91231ed85075c80",       # same pin as SM118
        R91_1500: "81e7bab8e0100a417b42e64f4fd2ea00fefdd16fce877dec8dc7977807e70b3d",  # same pin as SM115 / SM118
        R118: "430ffe8a809c27079862acf0bfb57b12b279a8472aa5eff19dfb232db378befb",
        R118_1500: "741ad45c94165073b35024f5cc4f4499db3a837dc4694ea551dd388bc07fb896",
        R101: "19b6dc1e85e8a673f17533e5422655580072562b7c6925123560421412a9bca2",
        R115: "992fc0b6a6c3c77d232067c9234592f7262d1a48ebb35c7eb07568524da3db0c"}
LBL106, LBL92, LBL91 = "sm106-c1-e29st-st-dec1ep32k/w8+s8", "sm92-e26-dec1ep32k/w8+s8", "sm91s2-big-showtime/w8+s8"
LBL118, LBL101, LBL115 = ("sm118-e26-dec2ep32k-t91s2/w8+s8", "sm101-c1-e38st-st-dec1ep32k/w8+s8",
                          "sm115-c1-e38st-st-dec2ep32k-t99/w8+s8")
MIN_EFFECT = 0.05
HEADS = ("curve", "instant")
# sm118-decision.json (the same code and unit): SM118 - SM92, show-time
SM118_MINUS_SM92 = {"curve": (-0.114, [-0.177, -0.062]), "instant": (-0.1, [-0.11, -0.091])}
# sm106-decision.md, in the big model's SHOW-TIME LogLoss unit (curve 0.29958, instant 0.277157): SM106 - SM92 and
# SM106 - the release, show-time; reproduced here after rescaling to this file's unit (tolerance 0.003%)
SM106_PUBLISHED_E2S_UNIT = {"sm92": {"curve": (0.107, [0.074, 0.137]), "instant": (0.178, [0.158, 0.199])},
                            "release": {"curve": (0.088, [0.001, 0.210]), "instant": (-0.126, [-0.223, -0.052])}}
NAMES = {"sm106": "the Tiny two-head model (SM106)", "sm91s2": "the show-time big model (SM91s2)",
         "sm92": "the show-time retrain of the release (SM92)", "release": "the release (SM63)",
         "big": "the big model (1 leak-free epoch)", "sm118": "the show-time two-head cool-down for Small (SM118)",
         "sm101": "the Tiny Instant-Only model (SM101)", "sm115": "the longer cool-down for Tiny (SM115)",
         "sm124": "the show-time two-head cool-down for Tiny (SM124)"}


def report():
    got = hashlib.sha256(REPORT.read_bytes()).hexdigest()
    if got != REPORT_SHA:
        raise SystemExit(f"[sm124-rule] showtime_report.py changed (sha256 {got[:12]} != {REPORT_SHA[:12]}): refusing")
    spec = importlib.util.spec_from_file_location("showtime_report", REPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_pins():
    for path, want in PINS.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"[sm124-rule] reference rows {path.name} changed (sha256 {got[:12]} != {str(want)[:12]}): refusing")


def refs(r):
    return {"sm106": {"e2s": [(R106, LBL106 + r.E2S, None)],
                      "e2s1500": [(R106, LBL106 + r.E2S, None), (R106_1500, LBL106 + r.E2S, None)]},
            "sm92": {"e2s": [(R92, LBL92 + r.E2S, None)], "normal": [(R92, LBL92, None)]},
            "sm91s2": {"e2s": [(R91, LBL91 + r.E2S, None)], "normal": [(R91, LBL91, None)],
                       "e2s1500": [(R91, LBL91 + r.E2S, None), (R91_1500, LBL91 + r.E2S, None)]},
            "sm118": {"e2s": [(R118, LBL118 + r.E2S, None)], "normal": [(R118, LBL118, None)],
                      "e2s1500": [(R118, LBL118 + r.E2S, None), (R118_1500, LBL118 + r.E2S, None)]},
            "sm101": {"e2s": [(R101, LBL101 + r.E2S, None)]}, "sm115": {"e2s": [(R115, LBL115 + r.E2S, None)]},
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
    s106 = r.load(refs(r)["sm106"]["e2s"], set(users))
    us = sorted(set(users) & set(new_e2s) & set(s106))
    c = r.compare(new_e2s, s106, big, us)
    return {"users": len(us), "new_minus_sm106_e2s": c, **verdict(c)}


def gap_closed(x_new, x_teacher):
    """per head: (new - SM106) / (SM91s2 - SM106), points on the same users."""
    return {h: (round(x_new[h]["points"] / x_teacher[h]["points"], 3) if x_teacher[h]["points"] else None) for h in HEADS}


def scaled(rows, factor):
    return {u: {h: (v[h][0] * factor.get(h, 1.0), v[h][1]) for h in v} for u, v in rows.items()}


def published_match(x, pub, ratio, tol=0.003):
    """x in this file's unit (big normal LogLoss); pub in the big model's show-time unit: x * ratio ~= pub."""
    return all(abs(x[h]["points"] * ratio[h] - pub[h][0]) <= tol and
               all(abs(a * ratio[h] - b) <= tol for a, b in zip(x[h]["points_ci"], pub[h][1])) for h in HEADS)


def selftest():
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    src = refs(r)
    n106, n91, nrel, n92, n118 = (r.load(src[k]["e2s"], have) for k in ("sm106", "sm91s2", "release", "sm92", "sm118"))
    p106, p91, prel, p92 = (primary(r, x, users, big) for x in (n106, n91, nrel, n92))
    x118 = r.compare(n118, n92, big, sorted(have & set(n118) & set(n92)))   # SM118's own rule pair through this code
    x106_92 = r.compare(n106, n92, big, sorted(have & set(n106) & set(n92)))
    x106_rel = r.compare(n106, nrel, big, sorted(have & set(n106) & set(nrel)))
    big_e2s = r.load(r.MODELS["big"]["e2s"], have)
    ratio = {h: r.big_denominator(big, users, k) / r.big_denominator(big_e2s, users, k) for k, h in r.HEADS.items()}
    syn = {"inst_0p04": primary(r, scaled(n106, {"ahead": 1.0, "instant": 0.9996}), users, big),
           "inst_0p1_curve_worse_0p06": primary(r, scaled(n106, {"ahead": 1.0006, "instant": 0.999}), users, big),
           "curve_0p1_only": primary(r, scaled(n106, {"ahead": 0.999, "instant": 1.0}), users, big)}
    pin = lambda x: {h: (x[h]["points"], x[h]["points_ci"]) for h in HEADS}  # noqa: E731
    checks = {
        "sm106_against_itself_zero_no_go_mid_stop": p106["users"] == 722 and all(p106["new_minus_sm106_e2s"][h]["points"] == 0 for h in HEADS)
        and p106["verdict"] == "no go" and mid_decision(p106["new_minus_sm106_e2s"]) == "stop",
        "sm118_minus_sm92_published_pair_go_both": pin(x118) == SM118_MINUS_SM92 and x118["users"] == 722
        and verdict(x118) == {"verdict": "go", "gain_heads": ["curve", "instant"], "regression_heads": []},
        "sm106_published_pairs_rescaled": x106_92["users"] == 722 == x106_rel["users"]
        and published_match(x106_92, SM106_PUBLISHED_E2S_UNIT["sm92"], ratio)
        and published_match(x106_rel, SM106_PUBLISHED_E2S_UNIT["release"], ratio),
        "release_minus_sm106_no_go_instant_regression_mid_stop": prel["verdict"] == "no go" and "instant" in prel["regression_heads"]
        and mid_decision(prel["new_minus_sm106_e2s"]) == "stop",
        "sm92_minus_sm106_go_both": p92["verdict"] == "go" and p92["gain_heads"] == ["curve", "instant"]
        and mid_decision(p92["new_minus_sm106_e2s"]) == "continue",
        "sm91s2_minus_sm106_go_both_gap_one": p91["verdict"] == "go" and p91["gain_heads"] == ["curve", "instant"]
        and gap_closed(p91["new_minus_sm106_e2s"], p91["new_minus_sm106_e2s"]) == {"curve": 1.0, "instant": 1.0}
        and mid_decision(p91["new_minus_sm106_e2s"]) == "continue",
        "synthetic_instant_0p04_no_go_continue": syn["inst_0p04"]["verdict"] == "no go"
        and -0.045 < syn["inst_0p04"]["new_minus_sm106_e2s"]["instant"]["points"] < -0.035
        and syn["inst_0p04"]["new_minus_sm106_e2s"]["instant"]["points_ci"][1] < 0
        and syn["inst_0p04"]["new_minus_sm106_e2s"]["curve"]["points"] == 0 and mid_decision(syn["inst_0p04"]["new_minus_sm106_e2s"]) == "continue",
        "synthetic_instant_gain_curve_0p06_worse_no_go_stop": syn["inst_0p1_curve_worse_0p06"]["verdict"] == "no go"
        and syn["inst_0p1_curve_worse_0p06"]["gain_heads"] == ["instant"] and "curve" in syn["inst_0p1_curve_worse_0p06"]["regression_heads"]
        and mid_decision(syn["inst_0p1_curve_worse_0p06"]["new_minus_sm106_e2s"]) == "stop",
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
    print(f"[sm124-rule] selftest: SM106 - SM106 {fp(p106['new_minus_sm106_e2s']['curve'])} / {fp(p106['new_minus_sm106_e2s']['instant'])} -> "
          f"{p106['verdict']}; SM118 - SM92 {fp(x118['curve'])} / {fp(x118['instant'])} -> {verdict(x118)['verdict']}; "
          f"SM106 - SM92 {fp(x106_92['curve'])} / {fp(x106_92['instant'])}; SM106 - release {fp(x106_rel['curve'])} / {fp(x106_rel['instant'])}; "
          f"SM91s2 - SM106 {fp(p91['new_minus_sm106_e2s']['curve'])} / {fp(p91['new_minus_sm106_e2s']['instant'])} -> {p91['verdict']}; "
          f"release - SM106 -> {prel['verdict']} {prel['regression_heads']}; {json.dumps(checks)} -> {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def end_to_end(r):
    """mid + decide on SM106's own rows relabelled (as pass 1, as the early-stop reference, as the final model; SM106's
    1,500-pool rows as the new model's; SM118's normal rows stand in for the new model's normal rows): mid must say stop
    (0 is not < 0); decide must read 0 -> no go, gap closed 0, with every table filled."""
    import io
    import tempfile
    from contextlib import redirect_stdout

    def relabel(src, dst, mapping, only=None):
        with open(src, encoding="utf-8") as f, open(dst, "a", encoding="utf-8") as g:
            for line in f:
                x = json.loads(line)
                lb = x.get("label", "")
                for a, b in mapping.items():
                    if (only in (None, "e2s") and lb == a + r.E2S) or (only in (None, "normal") and lb == a):
                        x["label"] = lb.replace(a, b)
                        g.write(json.dumps(x) + "\n")
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        rows, rows1500, mrows = t / "rows.jsonl", t / "rows-1500.jsonl", t / "mid.jsonl"
        relabel(R106, rows, {LBL106: LBL})
        relabel(R118, rows, {LBL118: LBL}, only="normal")
        relabel(R106_1500, rows1500, {LBL106: LBL})
        relabel(R106, mrows, {LBL106: MID_NEW})
        relabel(R106, mrows, {LBL106: MID_REF})
        with redirect_stdout(io.StringIO()):
            a = mid(t, mrows)
            b = decide(t, rows, rows1500, mrows)
        m = json.loads((t / "sm124-mid.json").read_text())
        d = json.loads((t / "sm124-decision.json").read_text())
        q = d["pool_1500"]
        return (a == 0 and b == 0 and m["decision"] == "stop" and all(m["vs_u4000"][h]["points"] == 0 for h in HEADS)
                and all(m["vs_sm106_final"][h]["points"] == 0 for h in HEADS)
                and d["verdict"] == "no go" and all(d["new_minus_sm106_e2s"][h]["points"] == 0 for h in HEADS)
                and d["gap_closed"] == {"curve": 0.0, "instant": 0.0}
                and all(v is not None for v in d["secondary_732"].values())
                and {"new_minus_sm91s2_e2s", "new_minus_release_e2s", "new_minus_sm92_e2s", "new_minus_sm118_e2s",
                     "new_e2s_minus_big_e2s", "new_normal_minus_big_normal", "new_e2s_minus_new_normal", "new_minus_sm101_e2s",
                     "new_minus_sm115_e2s", "mid_p1_minus_sm106_u4000_e2s", "mid_p1_minus_sm106_final_e2s"} <= set(d["secondary_732"])
                and q.get("users", 0) >= 700 and q["new_minus_sm106"]["instant"]["pct_of_ref"] == 0
                and q.get("new_minus_sm91s2") is not None and q.get("new_minus_release") is not None and q.get("new_minus_sm118") is not None
                and (t / "sm124-decision.md").exists() and (t / "sm124-mid.md").exists())


def mid(out_dir=HERE, mrows=MIDROWS):
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    a = r.load([(mrows, MID_NEW + r.E2S, None)], have)
    b = r.load([(mrows, MID_REF + r.E2S, None)], have)
    c = r.load(refs(r)["sm106"]["e2s"], have)
    if not (have <= set(a) and have <= set(b) and have <= set(c)):
        print(f"[sm124-rule] mid pending: rows pass 1 {len(have & set(a))}, SM106 u4000 {len(have & set(b))}, "
              f"SM106 final {len(have & set(c))} of {len(users)} rule users")
        return 2
    x = r.compare(a, b, big, sorted(have))
    y = r.compare(a, c, big, sorted(have))
    out = {"rule": f"continue (pass 2) if SM124's pass-1 final model - SM106's cool-down snapshot after 4,000 users (show-time "
                   f"inputs, {len(users)} users) has a point estimate < 0 on at least one head and < +0.05% on both; stop otherwise",
           "users": len(users), "labels": [MID_NEW, MID_REF, LBL106], "vs_u4000": x, "vs_sm106_final": y, "decision": mid_decision(x)}
    (Path(out_dir) / "sm124-mid.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = ["# The show-time two-head cool-down for Tiny (SM124): early stop after pass 1", "",
          f"Decision: **{out['decision']}**. The show-time two-head cool-down for Tiny (SM124)'s model after pass 1 of 2 "
          f"(learning rate 29.29% of its peak) minus the Tiny two-head model (SM106)'s own cool-down snapshot after "
          f"4,000 of 8,000 users (learning rate 30.45% of its peak), show-time inputs, {len(users)} users (% of the big model "
          f"(1 leak-free epoch) normal-input LogLoss of each head):", "",
          "| head | pass 1 | SM106 snapshot | % [95% CI] | better / worse |", "|---|---|---|---|---|"]
    for h in HEADS:
        md.append(f"| {h} | {x[h]['new']} | {x[h]['ref']} | {fp(x[h])} | {x[h]['users_better']} / {x[h]['users_worse']} |")
    md += ["", f"Read-out (no verdict): pass 1 minus the Tiny two-head model (SM106)'s final model: curve {fp(y['curve'])}, "
           f"instant {fp(y['instant'])}.", "",
           "Rule (pre-registered): continue with pass 2 if at least one head's point estimate is below 0 and neither is +0.05% "
           "or more; otherwise stop (verdict no go).", ""]
    (Path(out_dir) / "sm124-mid.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[sm124-rule] mid {out['decision']}: SM124 pass 1 - SM106 u4000 show-time curve {fp(x['curve'])}, instant {fp(x['instant'])}; "
          f"- SM106 final curve {fp(y['curve'])}, instant {fp(y['instant'])} ({len(users)} users)")
    return 0


def decide(out_dir=HERE, rows=ROWS, rows1500=ROWS1500, mrows=MIDROWS):
    r = report()
    check_pins()
    users, big = r.pools()
    have = set(users)
    mine = own(rows, rows1500, LBL, r)
    new = {v: r.load(mine[v], have) for v in ("e2s", "normal")}
    if not have <= set(new["e2s"]):
        print(f"[sm124-rule] pending: {LBL} show-time rows {len(have & set(new['e2s']))} of {len(users)} rule users")
        return 2
    out = {"rule": "SM124 - SM106, show-time inputs, 722 users, per head: gain = 95% CI below 0 and <= -0.05%; regression = "
                   ">= +0.05% or 95% CI above 0; go = a gain on at least one head and a regression on neither",
           "label": LBL, "users": len(users), "big_denominator": {h: round(r.big_denominator(big, users, k), 6) for k, h in r.HEADS.items()}}
    p = primary(r, new["e2s"], users, big)
    out |= p
    src = refs(r)
    ref = {k: {v: r.load(s[v], have) for v in ("e2s", "normal") if v in s} for k, s in src.items()}

    def pair(a, b):
        us = sorted(set(users) & set(a) & set(b))
        return r.compare(a, b, big, us) if us else None

    teacher = pair(ref["sm91s2"]["e2s"], ref["sm106"]["e2s"])
    out["sm91s2_minus_sm106_e2s"] = teacher
    out["gap_closed"] = gap_closed(p["new_minus_sm106_e2s"], teacher)
    big_e2s = r.load(r.MODELS["big"]["e2s"], have)
    sec = {"new_minus_sm91s2_e2s": pair(new["e2s"], ref["sm91s2"]["e2s"]),
           "new_minus_release_e2s": pair(new["e2s"], ref["release"]["e2s"]),
           "new_minus_sm92_e2s": pair(new["e2s"], ref["sm92"]["e2s"]),
           "new_minus_sm118_e2s": pair(new["e2s"], ref["sm118"]["e2s"]),
           "new_e2s_minus_big_e2s": pair(new["e2s"], big_e2s),
           "new_minus_sm101_e2s": pair(new["e2s"], ref["sm101"]["e2s"]),
           "new_minus_sm115_e2s": pair(new["e2s"], ref["sm115"]["e2s"])}
    if have <= set(new["normal"]):
        sec |= {"new_normal_minus_big_normal": pair(new["normal"], big),
                "new_minus_sm118_normal": pair(new["normal"], ref["sm118"]["normal"]),
                "new_e2s_minus_new_normal": pair(new["e2s"], new["normal"])}
    ma = r.load([(mrows, MID_NEW + r.E2S, None)], have)
    mb = r.load([(mrows, MID_REF + r.E2S, None)], have)
    if have <= set(ma) and have <= set(mb):
        sec["mid_p1_minus_sm106_u4000_e2s"] = pair(ma, mb)
    if have <= set(ma):
        sec["mid_p1_minus_sm106_final_e2s"] = pair(ma, ref["sm106"]["e2s"])
    out["secondary_732"] = sec
    q = {}
    if r.POOL1500.exists():
        pool = set(json.loads(r.POOL1500.read_text())["users"])
        rel = r.load(r.MODELS["release"]["e2s1500"], pool)
        o106, o91, o118 = (r.load(src[k]["e2s1500"], pool) for k in ("sm106", "sm91s2", "sm118"))
        nw = r.load(mine["e2s1500"], pool)
        us = sorted(set(nw) & set(rel) & set(o106))
        q["users"] = len(us)
        if us and Path(rows1500).exists():
            q["new_minus_sm106"] = r.compare(nw, o106, rel, us)
            q["new_minus_release"] = r.compare(nw, rel, rel, us)
            u118 = sorted(set(us) & set(o118))
            q["new_minus_sm118"] = r.compare(nw, o118, rel, u118) if u118 else None
            u91 = sorted(set(us) & set(o91))
            q["users_with_sm91s2"] = len(u91)
            q["new_minus_sm91s2"] = r.compare(nw, o91, rel, u91) if u91 else None
            lo, hi = [u for u in us if u < 5000], [u for u in us if u >= 5000]
            q["by_part_new_minus_sm106"] = {"ids < 5000 (the 732 set)": r.compare(nw, o106, rel, lo) if lo else None,
                                            "ids >= 5000 (training users of the big models)": r.compare(nw, o106, rel, hi) if hi else None}
            spec = json.loads(r.USERS1500.read_text())
            u732 = set(json.loads(r.U732.read_text())["users"]) | set(json.loads(r.ULONG.read_text())["users"])
            check = sorted(set(spec["users"] if isinstance(spec["users"], list) else []) & u732)
            rc = {}
            for name, (fa, fb, lb) in {"new": (rows, rows1500, LBL), "sm106": (R106, R106_1500, LBL106)}.items():
                a = r.load([(fa, lb + r.E2S, None)], set(check)); b = r.load([(fb, lb + r.E2S, None)], set(check))
                d = [max(abs(a[u][h][0] - b[u][h][0]) for h in r.HEADS) for u in check if u in a and u in b]
                rc[name] = {"users": len(d), "max_abs_logloss_diff": max(d) if d else None}
            q["row_check"] = rc
    out["pool_1500"] = q
    (Path(out_dir) / "sm124-decision.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    names = {"new_minus_sm91s2_e2s": (NAMES["sm91s2"] + " (the teacher)", "show-time"),
             "new_minus_release_e2s": (NAMES["release"], "show-time"),
             "new_minus_sm92_e2s": (NAMES["sm92"] + " (Small)", "show-time"),
             "new_minus_sm118_e2s": (NAMES["sm118"] + " (Small, the same recipe)", "show-time"),
             "new_e2s_minus_big_e2s": (NAMES["big"] + " (smaller-model bar, show-time: curve <= +0.5%, instant <= +0.4%)", "show-time"),
             "new_minus_sm101_e2s": (NAMES["sm101"] + " (instant only)", "show-time"),
             "new_minus_sm115_e2s": (NAMES["sm115"] + " (instant only)", "show-time"),
             "new_normal_minus_big_normal": (NAMES["big"] + " (the smaller-model bar: curve <= +0.5%, instant <= +0.4%)", "normal"),
             "new_minus_sm118_normal": (NAMES["sm118"] + " (Small, the same recipe)", "normal"),
             "new_e2s_minus_new_normal": ("the new model on normal inputs (its show-time cost)", "show-time vs normal"),
             "mid_p1_minus_sm106_u4000_e2s": ("early stop: SM124 after pass 1 minus SM106's snapshot after 4,000 users", "show-time"),
             "mid_p1_minus_sm106_final_e2s": ("early stop read-out: SM124 after pass 1 minus SM106's final model", "show-time")}
    x, t = p["new_minus_sm106_e2s"], teacher
    md = ["# The show-time two-head cool-down for Tiny (SM124): decision", "",
          f"Verdict: **{p['verdict']}** (gain on: {', '.join(p['gain_heads']) or 'no head'}; regression on: "
          f"{', '.join(p['regression_heads']) or 'no head'}; {len(users)} users of the 732 set; % of the big model (1 leak-free "
          f"epoch) normal-input LogLoss of each head, curve {out['big_denominator']['curve']}, instant "
          f"{out['big_denominator']['instant']}; paired user bootstrap 2,000, seed 0, 95% CI).", "",
          "Rule (pre-registered): per head, gain = CI below 0 and at most -0.05%; regression = +0.05% or more, or CI above 0; "
          "go = a gain on at least one head and a regression on neither.", "",
          "| the show-time two-head cool-down for Tiny (SM124) minus the Tiny two-head model (SM106), show-time | new | ref | % [95% CI] | better / worse | teacher gap closed |",
          "|---|---|---|---|---|---|"]
    for h in HEADS:
        md.append(f"| {h} | {x[h]['new']} | {x[h]['ref']} | {fp(x[h])} | {x[h]['users_better']} / {x[h]['users_worse']} | "
                  f"{out['gap_closed'][h]} (the show-time big model (SM91s2) minus SM106: {fp(t[h])}) |")
    md += ["", "## Read-out (732 set, %)", "", "| minus | inputs | curve % [95% CI] | instant % [95% CI] | instant better / worse |",
           "|---|---|---|---|---|"]
    for k, v in sec.items():
        if v:
            cv = "(not trained)" if k in ("new_minus_sm101_e2s", "new_minus_sm115_e2s") else fp(v["curve"])
            md.append(f"| {names[k][0]} | {names[k][1]} | {cv} | {fp(v['instant'])} | {v['instant']['users_better']} / {v['instant']['users_worse']} |")
    md.append("")
    if q.get("new_minus_sm106"):
        md += [f"## 1,500 validation users, show-time inputs ({q['users']} users), % of the release's own LogLoss", "",
               "| comparison | users | curve % [95% CI] | instant % [95% CI] |", "|---|---|---|---|"]
        rowsq = [("new minus " + NAMES["sm106"], q["new_minus_sm106"]), ("new minus " + NAMES["sm91s2"], q.get("new_minus_sm91s2")),
                 ("new minus " + NAMES["sm118"], q.get("new_minus_sm118")), ("new minus " + NAMES["release"], q["new_minus_release"])]
        rowsq += [(f"new minus {NAMES['sm106']}, {part}", v) for part, v in q["by_part_new_minus_sm106"].items()]
        for nm, v in rowsq:
            if v:
                md.append(f"| {nm} | {v['users']} | {fpct(v['curve'])} | {fpct(v['instant'])} |")
        md += ["", f"Users with rows of the show-time big model (SM91s2): {q.get('users_with_sm91s2')}. Its ids >= 5000 are its own "
               f"training users, so new minus SM91s2 there is not a held-out comparison. Row checks (732 rows vs 1,500-pool rows "
               f"of the same users): {json.dumps(q['row_check'])}", ""]
    else:
        md += ["## 1,500 validation users", "", "Not scored yet.", ""]
    (Path(out_dir) / "sm124-decision.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[sm124-rule] verdict {p['verdict']} (gain {p['gain_heads']}, regression {p['regression_heads']}): SM124 - SM106 show-time "
          f"curve {fp(x['curve'])}, instant {fp(x['instant'])}; gap closed {out['gap_closed']}; SM124 - SM91s2 curve "
          f"{fp(sec['new_minus_sm91s2_e2s']['curve'])}, instant {fp(sec['new_minus_sm91s2_e2s']['instant'])}; 1,500 pool users {q.get('users', 0)}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "decide"
    raise SystemExit({"selftest": selftest, "mid": mid, "decide": decide}[cmd]())
