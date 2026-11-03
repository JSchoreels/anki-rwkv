#!/usr/bin/env python
"""Show-time interval retrain (SM91) step 3: pre-registered rule (CPU, seconds). DESIGN.md in this folder.

New        step 1's arm epoch 21 (the leak-free small model (SM72s) epoch 20 + one constant-LR epoch on show-time inputs,
           old targets) + the quarter-epoch 32k Muon cool-down on show-time inputs with NEW KD replay targets: the
           show-time big model (step 2) replaying the cool-down users' full histories on show-time inputs.
           Label sm91s3-arm-e21-q32k-stt.
Reference  step 1's arm: the same epoch 21 + the same cool-down with the OLD targets (the big model (1 leak-free
           epoch) on normal inputs). Rows reused, read only: wt/sm91/artifacts/small-model/sm91/gpu/sm91-inputs.jsonl,
           sm91-arm-e21-q32k/w8+s8@e2s-query (and its normal-input rows sm91-arm-e21-q32k/w8+s8).
Scoring: leak-free (small_model_heldout.py --split-existing --deleted-cards neighbour), w8+s8, --inputs e2s-query
(and normal), the 732 set (+ the 3 long users). Rule pool: the 722 users of the 732 set that the big model (1 leak-free
epoch) normal-input rows have. Review-weighted benchmark LogLoss; points = % of the big model (1 leak-free epoch)
normal-input LogLoss on the same users; paired user bootstrap 2,000 resamples, seed 0, 95% CI.

Rule (fixed 2026-11-03 before launch):
  adopt the show-time targets  the new model beats step 1's arm on e2s-query inputs by >= 0.05 points on at least one
                               head (curve or instant) with that head's 95% CI entirely below zero, and no head is
                               worse by >= 0.05 points with its CI entirely above zero.
  no gain                      otherwise (keep step 1's targets; a show-time teacher does not pay at cool-down scale).

Also reported (not the rule), each as new and as step 1's arm: vs the big model (1 leak-free epoch) on e2s-query (SM31
rows), vs the show-time big model on e2s-query (step 2 rows), vs the released small model on e2s-query (SM31 rows), vs
the big model (1 leak-free epoch) on normal inputs (cost left); new normal - step 1's arm normal. The smaller-model bar
(<= +0.5% curve / <= +0.4% instant, % of the reference's own LogLoss): (a) normal inputs vs the big model (1 leak-free
epoch) normal (the bar as defined), (b) e2s-query vs the big model (1 leak-free epoch) e2s-query, (c) e2s-query vs the
show-time big model e2s-query (DESIGN.md: which one should apply to show-time models). The 725-user 732+long pool.

  python artifacts/small-model/sm91s3/sm91s3_rule.py decide [--rows FILE]  -> sm91s3-decision.{json,md}; exit 2 if rows missing
  python artifacts/small-model/sm91s3/sm91s3_rule.py self-test             -> exit 0 if step 1's rule numbers are reproduced
"""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from scripts.sm71_report import load as _load  # noqa: E402

ROWS = HERE / "gpu" / "sm91s3-inputs.jsonl"
NEW = "sm91s3-arm-e21-q32k-stt/w8+s8"
E2S = "@e2s-query"
SM91 = Path("C:/Users/jonat/workspace/wt/sm91/artifacts/small-model/sm91")                       # read only
STEP1 = (SM91 / "gpu" / "sm91-inputs.jsonl", "sm91-arm-e21-q32k/w8+s8")                       # step 1's arm
SM77REF = (SM91 / "gpu" / "sm91-inputs.jsonl", "sm77-ref-e21-q32k/w8+s8@e2s-query")             # self-test only
SM31 = Path("C:/Users/jonat/workspace/wt/sm31/artifacts/small-model/sm31/gpu/sm31-inputs.jsonl")  # read only
BIG_E2S = (SM31, "big-leakfree-1ep/w8+s8@e2s-query")
SMALL_E2S = (SM31, "s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/w8+s8@e2s-query")          # the released small model
SHOWBIG_E2S = (Path("C:/Users/jonat/workspace/wt/sm91s2/artifacts/small-model/sm91s2/gpu/sm91s2-inputs.jsonl"),
               "sm91s2-big-showtime/w8+s8@e2s-query")
BIG = (Path("C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/gpu-big/bigleak-neighbour.jsonl"),
       "big-leakfree-1ep/w8+s8")  # normal inputs: the points denominator
S72 = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s")  # read only
U732, ULONG = S72 / "eval-users-732-fit.json", S72 / "users-732-long.json"
HEADS = {"ahead": "curve", "instant": "instant"}
GO_POINTS = 0.05
BAR = {"curve": 0.5, "instant": 0.4}   # smaller-model bar, % of the reference's LogLoss


def load(source, label, users):
    return {u: r for u, r in _load(source, label, users).items() if u in users}


def compare(new, ref, big, users):
    """new - ref per head over `users`: raw review-weighted delta, points of the big model (1 leak-free epoch) normal
    LogLoss, bootstrap CIs (the step 1 / step 2 rules' compare), plus pct = % of the reference's own LogLoss (the bar)."""
    idx = np.random.default_rng(0).integers(0, len(users), size=(2000, len(users)))
    out = {"users": len(users)}
    for head, h in HEADS.items():
        a = np.array([new[u]["all"][head][0] for u in users]); b = np.array([ref[u]["all"][head][0] for u in users])
        c = np.array([big[u]["all"][head][0] for u in users])
        w = np.array([big[u]["all"][head][1] for u in users], float)
        a, b, c = (np.where(w > 0, x, 0.0) for x in (a, b, c))
        boot_raw = ((a - b)[idx] * w[idx]).sum(1) / w[idx].sum(1)
        boot = ((a - b)[idx] * w[idx]).sum(1) / (c[idx] * w[idx]).sum(1) * 100
        boot_pct = ((a - b)[idx] * w[idx]).sum(1) / (b[idx] * w[idx]).sum(1) * 100
        big_loss = (c * w).sum()
        out[h] = {"new": round(float((a * w).sum() / w.sum()), 6), "ref": round(float((b * w).sum() / w.sum()), 6),
                  "big": round(float(big_loss / w.sum()), 6), "delta": round(float(((a - b) * w).sum() / w.sum()), 6),
                  "delta_ci": [round(float(np.percentile(boot_raw, q)), 6) for q in (2.5, 97.5)],
                  "points": round(float(((a - b) * w).sum() / big_loss * 100), 3),
                  "points_ci": [round(float(np.percentile(boot, q)), 3) for q in (2.5, 97.5)],
                  "pct_of_ref": round(float(((a - b) * w).sum() / (b * w).sum() * 100), 3),
                  "pct_of_ref_ci": [round(float(np.percentile(boot_pct, q)), 3) for q in (2.5, 97.5)],
                  "users_better": int((a < b).sum()), "users_worse": int((a > b).sum())}
    return out


def verdict(pair):
    go = [f"{h} {pair[h]['points']:+.3f} points [{pair[h]['points_ci'][0]:+.3f}, {pair[h]['points_ci'][1]:+.3f}]"
          for h in HEADS.values() if pair[h]["points"] <= -GO_POINTS and pair[h]["points_ci"][1] < 0]
    bad = [f"{h} {pair[h]['points']:+.3f} points [{pair[h]['points_ci'][0]:+.3f}, {pair[h]['points_ci'][1]:+.3f}]"
           for h in HEADS.values() if pair[h]["points"] >= GO_POINTS and pair[h]["points_ci"][0] > 0]
    if go and not bad:
        return "adopt the show-time targets", "gain >= 0.05 points with the CI below zero: " + "; ".join(go)
    why = ", ".join(f"{h} {pair[h]['points']:+.3f} [{pair[h]['points_ci'][0]:+.3f}, {pair[h]['points_ci'][1]:+.3f}]" for h in HEADS.values())
    if bad:
        return "no gain", f"a head is worse by >= 0.05 points with the CI above zero ({'; '.join(bad)}); {why}"
    return "no gain", f"no head gains >= 0.05 points with the CI below zero ({why})"


def pools():
    users732 = set(json.loads(U732.read_text())["users"])
    long_users = set(json.loads(ULONG.read_text())["users"])
    big = load(*BIG, users732 | long_users)
    return sorted(users732 & set(big)), big


def bar(x):
    return {h: {"pct": x[h]["pct_of_ref"], "ci": x[h]["pct_of_ref_ci"], "limit": BAR[h], "passes": x[h]["pct_of_ref"] <= BAR[h]}
            for h in HEADS.values()}


def decide(args):
    users, big = pools()
    rows = Path(args.rows)
    have = set(big)
    sets = {"new_e2s": load(rows, NEW + E2S, have) if rows.exists() else {},
            "step1_e2s": load(STEP1[0], STEP1[1] + E2S, have),
            "new_normal": load(rows, NEW, have) if rows.exists() else {},
            "step1_normal": load(*STEP1, have),
            "big_e2s": load(*BIG_E2S, have), "showbig_e2s": load(*SHOWBIG_E2S, have), "small_e2s": load(*SMALL_E2S, have),
            "big_normal": big}
    missing = {k: [u for u in users if u not in v] for k, v in sets.items()}
    if missing["new_e2s"] or missing["step1_e2s"]:
        print(f"[sm91s3-rule] rule rows missing: new e2s-query {len(missing['new_e2s'])}, step 1's arm e2s-query "
              f"{len(missing['step1_e2s'])} of {len(users)} users")
        return 2
    pair = compare(sets["new_e2s"], sets["step1_e2s"], big, users)
    res = {"rule_pool": {"users": len(users), "new_minus_step1_e2s": pair}}
    res["verdict"], res["why"] = verdict(pair)
    info = res["rule_pool"]
    for who in ("new", "step1"):
        for ref in ("big_e2s", "showbig_e2s", "small_e2s", "big_normal"):
            if not missing[ref]:
                info[f"{who}_e2s_minus_{ref}"] = compare(sets[f"{who}_e2s"], sets[ref], big, users)
    bars = {}
    for who in ("new", "step1"):
        b = {}
        if not missing[f"{who}_normal"]:
            b["a_normal_vs_big_normal"] = bar(compare(sets[f"{who}_normal"], big, big, users))
        if f"{who}_e2s_minus_big_e2s" in info:
            b["b_e2s_vs_big_e2s"] = bar(info[f"{who}_e2s_minus_big_e2s"])
        if f"{who}_e2s_minus_showbig_e2s" in info:
            b["c_e2s_vs_showbig_e2s"] = bar(info[f"{who}_e2s_minus_showbig_e2s"])
        bars[who] = b
    info["smaller_model_bar"] = bars
    if not missing["new_normal"]:
        info["new_normal_minus_step1_normal"] = compare(sets["new_normal"], sets["step1_normal"], big, users)
        info["new_e2s_minus_new_normal"] = compare(sets["new_e2s"], sets["new_normal"], big, users)
    both = sorted(set(big) & set(sets["new_e2s"]) & set(sets["step1_e2s"]))
    if len(both) > len(users):
        res["732+long"] = {"users": len(both), "new_minus_step1_e2s": compare(sets["new_e2s"], sets["step1_e2s"], big, both)}
    res.update(rows=str(rows), labels={"new_e2s": NEW + E2S, "new_normal": NEW, "step1": f"{STEP1[0]} {STEP1[1]}[{E2S}]",
                                       "big_e2s": f"{BIG_E2S[0]} {BIG_E2S[1]}", "showbig_e2s": f"{SHOWBIG_E2S[0]} {SHOWBIG_E2S[1]}",
                                       "small_e2s": f"{SMALL_E2S[0]} {SMALL_E2S[1]}", "big_normal": f"{BIG[0]} {BIG[1]}"},
               missing_informational={k: len(v) for k, v in missing.items()},
               rule={"go_if": f"delta (new - step 1's arm, e2s-query) <= -{GO_POINTS} points with 95% CI upper < 0 on curve or "
                              f"instant, and no head >= +{GO_POINTS} points with CI lower > 0",
                     "unit": "points = % of the big model (1 leak-free epoch) normal-input LogLoss, review-weighted, 722 users"})
    (HERE / "sm91s3-decision.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    md = ["# Show-time interval retrain (SM91) step 3: cool-down with show-time teacher targets vs step 1's arm (old targets), both scored on show-time inputs", "",
          "Leak-free scoring (deleted cards: neighbour), w8+s8, review-weighted LogLoss, user bootstrap 2,000 seed 0, 95% CI; points = % of "
          "the big model (1 leak-free epoch) with normal inputs. New: step 1's arm epoch 21 + the quarter-epoch 32k Muon cool-down on "
          "show-time inputs with KD targets from the show-time big model on show-time inputs. Reference: step 1's arm (the same epoch 21 + "
          "the same cool-down with the old targets: the big model (1 leak-free epoch) on normal inputs).", "",
          "| pool | users | comparison | head | new | ref | delta [95% CI] | points [95% CI] | users better / worse |",
          "|---|---|---|---|---|---|---|---|---|"]
    names = {"new_minus_step1_e2s": "new e2s - step 1's arm e2s (the rule)",
             "new_e2s_minus_big_e2s": "new e2s - the big model (1 leak-free epoch) e2s",
             "step1_e2s_minus_big_e2s": "step 1's arm e2s - the big model (1 leak-free epoch) e2s",
             "new_e2s_minus_showbig_e2s": "new e2s - the show-time big model e2s",
             "step1_e2s_minus_showbig_e2s": "step 1's arm e2s - the show-time big model e2s",
             "new_e2s_minus_small_e2s": "new e2s - the released small model e2s",
             "step1_e2s_minus_small_e2s": "step 1's arm e2s - the released small model e2s",
             "new_e2s_minus_big_normal": "new e2s - the big model (1 leak-free epoch) normal (cost left)",
             "step1_e2s_minus_big_normal": "step 1's arm e2s - the big model (1 leak-free epoch) normal",
             "new_normal_minus_step1_normal": "new normal - step 1's arm normal",
             "new_e2s_minus_new_normal": "new e2s - new normal (its own show-time cost)"}
    for pool in ("rule_pool", "732+long"):
        if pool not in res:
            continue
        for key, name in names.items():
            if key not in res[pool]:
                continue
            for h in HEADS.values():
                x = res[pool][key][h]
                md.append(f"| {'732 (rule)' if pool == 'rule_pool' else pool} | {res[pool]['users']} | {name} | {h} | {x['new']:.5f} | "
                          f"{x['ref']:.5f} | {x['delta']:+.6f} [{x['delta_ci'][0]:+.6f}, {x['delta_ci'][1]:+.6f}] | "
                          f"{x['points']:+.3f} [{x['points_ci'][0]:+.3f}, {x['points_ci'][1]:+.3f}] | {x['users_better']} / {x['users_worse']} |")
    md += ["", "Smaller-model bar (<= +0.5% curve / <= +0.4% instant, % of the reference's own LogLoss; not the rule):", "",
           "| model | reference | curve % [95% CI] | instant % [95% CI] | passes |", "|---|---|---|---|---|"]
    refname = {"a_normal_vs_big_normal": "(a) normal inputs vs the big model (1 leak-free epoch) normal",
               "b_e2s_vs_big_e2s": "(b) e2s-query vs the big model (1 leak-free epoch) e2s-query",
               "c_e2s_vs_showbig_e2s": "(c) e2s-query vs the show-time big model e2s-query"}
    for who, label in (("new", "new"), ("step1", "step 1's arm")):
        for key, b in bars[who].items():
            md.append(f"| {label} | {refname[key]} | {b['curve']['pct']:+.3f} [{b['curve']['ci'][0]:+.3f}, {b['curve']['ci'][1]:+.3f}] | "
                      f"{b['instant']['pct']:+.3f} [{b['instant']['ci'][0]:+.3f}, {b['instant']['ci'][1]:+.3f}] | "
                      f"{'yes' if b['curve']['passes'] and b['instant']['passes'] else 'no'} |")
    md += ["", f"Rule: adopt the show-time targets if the new model beats step 1's arm on e2s-query by >= {GO_POINTS:.2f} points on curve "
           "or instant with the 95% CI below zero, and no head is worse by >= 0.05 points with the CI above zero; else no gain.", "",
           f"Verdict: **{res['verdict']}** ({res['why']})."]
    (HERE / "sm91s3-decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    c, i = pair["curve"], pair["instant"]
    print(f"[sm91s3-rule] {len(users)} users, new - step 1's arm on e2s-query: curve {c['points']:+.3f} points "
          f"{c['points_ci']}, instant {i['points']:+.3f} points {i['points_ci']}: {res['verdict'].upper()} ({res['why']})")
    return 0


def self_test(_args):
    """Step 1's rule pair (step 1's arm e2s - the tiny architecture additions (SM77) reference e2s) through compare():
    must reproduce sm91-decision.json (points, CIs, deltas, 722 users); step 1's arm against itself: 0 / no gain; a
    synthetic model 0.06 points better on instant only: adopt; 0.04 points better: no gain; 0.06 better on instant but
    0.06 worse on curve: no gain."""
    users, big = pools()
    want = json.loads((SM91 / "sm91-decision.json").read_text())["rule_pool"]["arm_minus_reference_e2s"]
    arm = load(STEP1[0], STEP1[1] + E2S, set(big))
    ref = load(*SM77REF, set(big))
    got = compare(arm, ref, big, users)
    fails = []
    if len(users) != 722:
        fails.append(f"rule pool has {len(users)} users, not 722")
    for h in HEADS.values():
        for k in ("points", "points_ci", "delta", "delta_ci", "users_better", "users_worse"):
            if got[h][k] != want[h][k]:
                fails.append(f"{h} {k}: got {got[h][k]}, step 1 {want[h][k]}")
    zero = compare(arm, arm, big, users)
    if any(zero[h]["delta"] != 0 or zero[h]["points_ci"] != [0.0, 0.0] for h in HEADS.values()) or verdict(zero)[0] != "no gain":
        fails.append("step 1's arm vs itself is not 0 / no gain")

    def fake(curve_points, instant_points):
        return {u: {"all": {"ahead": (arm[u]["all"]["ahead"][0] + curve_points / 100 * big[u]["all"]["ahead"][0], arm[u]["all"]["ahead"][1]),
                            "instant": (arm[u]["all"]["instant"][0] + instant_points / 100 * big[u]["all"]["instant"][0], arm[u]["all"]["instant"][1])}}
                for u in users}
    checks = {"instant -0.06": (fake(0, -0.06), "adopt the show-time targets"), "instant -0.04": (fake(0, -0.04), "no gain"),
              "instant -0.06 curve +0.06": (fake(0.06, -0.06), "no gain")}
    synth = {}
    for name, (model, expect) in checks.items():
        v = verdict(compare(model, arm, big, users))
        synth[name] = v[0]
        if v[0] != expect:
            fails.append(f"synthetic {name}: {v}")
    print(json.dumps({"users": len(users), "step1_rule_reproduced": {h: [got[h]["points"], got[h]["points_ci"]] for h in HEADS.values()},
                      "step1_published": {h: [want[h]["points"], want[h]["points_ci"]] for h in HEADS.values()},
                      "self_pair": {h: zero[h]["points"] for h in HEADS.values()}, "synthetic": synth,
                      "ok": not fails, "failures": fails}))
    return 0 if not fails else 1


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("decide", "self-test"))
    parser.add_argument("--rows", type=Path, default=ROWS)
    a = parser.parse_args()
    raise SystemExit(decide(a) if a.command == "decide" else self_test(a))
