#!/usr/bin/env python
"""Show-time interval retrain (SM91) step 1: pre-registered rule (CPU, seconds). DESIGN.md in this folder.

Arm       the leak-free small model (SM72s) epoch 20 + 1 constant-LR epoch with SM72s's epoch-21 protocol + the
          quarter-epoch 32k Muon cool-down (the reference's users, order, schedule, options), trained on show-time
          interval inputs (track option "show_time_intervals": the duration-leak scoring (SM31) `e2s-query` transform).
Reference SM72s's own epoch 21 + the same cool-down (the tiny architecture additions (SM77) reference,
          wt/sm77/artifacts/small-model/sm77-ref-e21-dec-q32k-muon), trained on normal inputs; re-scored here.
Both: same architecture (w64), start weights, users, order, seeds and KD replay targets; only the inputs differ.
Scoring: leak-free (small_model_heldout.py --split-existing --deleted-cards neighbour), w8+s8, --inputs e2s-query
(show-time intervals, deployment-realistic), the 732 set. Rule pool: the 722 users that the big model (1 leak-free
epoch) rows have. Review-weighted benchmark LogLoss; points = % of the big model's normal-input LogLoss on the same
users (as SM31); paired user bootstrap 2,000 resamples, seed 0, 95% CI.

Rule (fixed 2026-11-02 before launch):
  go to step 2  the arm beats the reference on e2s-query inputs by >= 0.10 points on at least one head (curve or
                instant), with that head's 95% CI entirely below zero (delta = arm - reference <= -0.10 points and CI
                upper bound < 0). 0.10 points = ~0.00030 curve / ~0.00028 instant LogLoss.
  stop          otherwise.

Also reported (not the rule): the reference's own show-time cost at this checkpoint (reference e2s-query - reference
normal; SM31 measured +0.13 / +0.43 on the released small model); the share of that cost the arm wins back
((reference e2s - arm e2s) / (reference e2s - reference normal)) and the gain as a share of SM31's +0.13 / +0.43; the
arm on normal inputs vs the reference on normal inputs; each vs the big model; the 725-user 732+long pool.

  python artifacts/small-model/sm91/sm91_rule.py decide [--rows FILE]  -> sm91-decision.{json,md}; exit 2 if users missing
  python artifacts/small-model/sm91/sm91_rule.py self-test             -> exit 0 if SM31's numbers are reproduced
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

ROWS = HERE / "gpu" / "sm91-inputs.jsonl"
ARM = "sm91-arm-e21-q32k/w8+s8"
REF = "sm77-ref-e21-q32k/w8+s8"
E2S = "@e2s-query"
REF_NORMAL = Path("C:/Users/jonat/workspace/wt/sm77/artifacts/small-model/sm77/gpu/sm77-neighbour.jsonl")  # read only
BIG = (Path("C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/gpu-big/bigleak-neighbour.jsonl"),
       "big-leakfree-1ep/w8+s8")  # read only, normal inputs: the points denominator
S72 = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s")  # read only
U732, ULONG = S72 / "eval-users-732-fit.json", S72 / "users-732-long.json"
HEADS = {"ahead": "curve", "instant": "instant"}
GO_POINTS = 0.10                                     # gain needed on one head, points (% of the big model)
SM31_COST = {"curve": 0.128, "instant": 0.428}       # released small model, e2s-query - normal, 722 users (sm31-result)
SM31 = ROOT / "artifacts/small-model/sm31"


def load(source, label, users):
    return {u: r for u, r in _load(source, label, users).items() if u in users}


def compare(new, ref, big, users):
    """new - ref per head over `users`: raw review-weighted delta, points of the big model, bootstrap CIs."""
    idx = np.random.default_rng(0).integers(0, len(users), size=(2000, len(users)))
    out = {"users": len(users)}
    for head, h in HEADS.items():
        a = np.array([new[u]["all"][head][0] for u in users]); b = np.array([ref[u]["all"][head][0] for u in users])
        c = np.array([big[u]["all"][head][0] for u in users])
        w = np.array([big[u]["all"][head][1] for u in users], float)
        a, b, c = (np.where(w > 0, x, 0.0) for x in (a, b, c))
        boot_raw = ((a - b)[idx] * w[idx]).sum(1) / w[idx].sum(1)
        boot = ((a - b)[idx] * w[idx]).sum(1) / (c[idx] * w[idx]).sum(1) * 100
        big_loss = (c * w).sum()
        out[h] = {"new": round(float((a * w).sum() / w.sum()), 6), "ref": round(float((b * w).sum() / w.sum()), 6),
                  "big": round(float(big_loss / w.sum()), 6), "delta": round(float(((a - b) * w).sum() / w.sum()), 6),
                  "delta_ci": [round(float(np.percentile(boot_raw, q)), 6) for q in (2.5, 97.5)],
                  "points": round(float(((a - b) * w).sum() / big_loss * 100), 3),
                  "points_ci": [round(float(np.percentile(boot, q)), 3) for q in (2.5, 97.5)],
                  "users_better": int((a < b).sum()), "users_worse": int((a > b).sum()),
                  "new_vs_big_pct": round(float(((a - c) * w).sum() / big_loss * 100), 3),
                  "ref_vs_big_pct": round(float(((b - c) * w).sum() / big_loss * 100), 3)}
    return out


def verdict(pair):
    go = [f"{h} {pair[h]['points']:+.3f} points [{pair[h]['points_ci'][0]:+.3f}, {pair[h]['points_ci'][1]:+.3f}]"
          for h in HEADS.values() if pair[h]["points"] <= -GO_POINTS and pair[h]["points_ci"][1] < 0]
    if go:
        return "go to step 2", "gain >= 0.10 points with the CI below zero: " + "; ".join(go)
    return "stop", "no head gains >= 0.10 points with the CI below zero (" + ", ".join(
        f"{h} {pair[h]['points']:+.3f} [{pair[h]['points_ci'][0]:+.3f}, {pair[h]['points_ci'][1]:+.3f}]" for h in HEADS.values()) + ")"


def pools():
    users732 = set(json.loads(U732.read_text())["users"])
    long_users = set(json.loads(ULONG.read_text())["users"])
    big = load(*BIG, users732 | long_users)
    return sorted(users732 & set(big)), big


def decide(args):
    users, big = pools()
    rows = Path(args.rows)
    have = set(big)
    sets = {"arm_e2s": load(rows, ARM + E2S, have) if rows.exists() else {},
            "ref_e2s": load(rows, REF + E2S, have) if rows.exists() else {},
            "arm_normal": load(rows, ARM, have) if rows.exists() else {},
            "ref_normal": load(REF_NORMAL, REF, have)}
    missing = {k: [u for u in users if u not in v] for k, v in sets.items()}
    if missing["arm_e2s"] or missing["ref_e2s"]:
        print(f"[sm91-rule] rule rows missing: arm e2s-query {len(missing['arm_e2s'])}, reference e2s-query "
              f"{len(missing['ref_e2s'])} of {len(users)} users")
        return 2
    res = {"rule_pool": {"users": len(users), "arm_minus_reference_e2s": compare(sets["arm_e2s"], sets["ref_e2s"], big, users)}}
    pair = res["rule_pool"]["arm_minus_reference_e2s"]
    res["verdict"], res["why"] = verdict(pair)
    info = {}
    if not missing["ref_normal"]:
        cost = compare(sets["ref_e2s"], sets["ref_normal"], big, users)
        resid = compare(sets["arm_e2s"], sets["ref_normal"], big, users)
        info["reference_show_time_cost"] = cost
        info["arm_e2s_minus_reference_normal"] = resid
        info["win_back"] = {h: {"reference_cost_points": cost[h]["points"], "arm_gain_points": round(-pair[h]["points"], 3),
                                "share_of_reference_cost_pct": round(-pair[h]["points"] / cost[h]["points"] * 100, 1) if cost[h]["points"] else None,
                                "share_of_sm31_cost_pct": round(-pair[h]["points"] / SM31_COST[h] * 100, 1)} for h in HEADS.values()}
    if not missing["arm_normal"] and not missing["ref_normal"]:
        info["arm_normal_minus_reference_normal"] = compare(sets["arm_normal"], sets["ref_normal"], big, users)
    if not missing["arm_normal"]:
        info["arm_e2s_minus_arm_normal"] = compare(sets["arm_e2s"], sets["arm_normal"], big, users)
    res["rule_pool"].update(info)
    both = sorted(set(big) & set(sets["arm_e2s"]) & set(sets["ref_e2s"]))
    if len(both) > len(users):
        res["732+long"] = {"users": len(both), "arm_minus_reference_e2s": compare(sets["arm_e2s"], sets["ref_e2s"], big, both)}
    res.update(rows=str(rows), labels={"arm_e2s": ARM + E2S, "ref_e2s": REF + E2S, "arm_normal": ARM,
                                       "ref_normal": f"{REF_NORMAL} {REF}"}, big=f"{BIG[0]} {BIG[1]}",
               missing_informational={k: len(v) for k, v in missing.items()},
               rule={"go_if": f"delta (arm - reference, e2s-query) <= -{GO_POINTS} points with 95% CI upper < 0 on curve or instant",
                     "unit": "points = % of the big model (1 leak-free epoch) normal-input LogLoss, review-weighted, 722 users"})
    (HERE / "sm91-decision.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    md = ["# Show-time interval retrain (SM91) step 1: show-time-trained arm vs normal-trained reference, both scored on show-time inputs", "",
          "Leak-free scoring (deleted cards: neighbour), w8+s8, `--inputs e2s-query`, review-weighted LogLoss, user bootstrap 2,000 "
          "seed 0, 95% CI; points = % of the big model (1 leak-free epoch) with normal inputs. Arm: the leak-free small model (SM72s) "
          "epoch 20 + 1 constant-LR epoch + quarter-epoch 32k Muon cool-down, trained on show-time intervals. Reference: SM72s epoch 21 "
          "+ the same cool-down, trained on normal intervals (the tiny architecture additions (SM77) reference). Same start, users, "
          "order, seeds, KD targets.", "",
          "| pool | users | comparison | head | new | ref | delta [95% CI] | points [95% CI] | users better / worse |",
          "|---|---|---|---|---|---|---|---|---|"]
    names = {"arm_minus_reference_e2s": "arm e2s - reference e2s (the rule)",
             "reference_show_time_cost": "reference e2s - reference normal (its show-time cost)",
             "arm_e2s_minus_reference_normal": "arm e2s - reference normal (cost left)",
             "arm_normal_minus_reference_normal": "arm normal - reference normal",
             "arm_e2s_minus_arm_normal": "arm e2s - arm normal"}
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
    if "win_back" in res["rule_pool"]:
        wb = res["rule_pool"]["win_back"]
        md += ["", "Win-back (not the rule): " + "; ".join(
            f"{h}: gain {wb[h]['arm_gain_points']:+.3f} points = {wb[h]['share_of_reference_cost_pct']}% of the reference's own show-time "
            f"cost ({wb[h]['reference_cost_points']:+.3f}) and {wb[h]['share_of_sm31_cost_pct']}% of SM31's released-model cost "
            f"(+{SM31_COST[h]:.3f})" for h in HEADS.values()) + "."]
    md += ["", f"Rule: go to step 2 if the arm beats the reference by >= {GO_POINTS:.2f} points on curve or instant with the 95% CI "
           "below zero; else stop.", "", f"Verdict: **{res['verdict']}** ({res['why']})."]
    (HERE / "sm91-decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    c, i = pair["curve"], pair["instant"]
    print(f"[sm91-rule] {len(users)} users, arm - reference on e2s-query: curve {c['points']:+.3f} points {c['points_ci']}, "
          f"instant {i['points']:+.3f} points {i['points_ci']}: {res['verdict'].upper()} ({res['why']})")
    return 0


def self_test(_args):
    """SM31's released small model, e2s-query rows vs its normal rows, through compare(): must reproduce sm31-result.json
    (points and CIs, 722 users); the reference's normal rows against themselves: 0 and verdict stop."""
    users, big = pools()
    result = json.loads((SM31 / "sm31-result.json").read_text())
    small = result["labels"]["small"]
    e2s = load(SM31 / "gpu" / "sm31-inputs.jsonl", small + E2S, set(big))
    normal = load(Path(result["normal_rows"][small]), small, set(big))
    got = compare(e2s, normal, big, users)
    want = result["pools"]["732 (rule)"][small]["e2s-query"]
    fails = []
    for h in HEADS.values():
        if got[h]["points"] != want[h]["points"] or got[h]["points_ci"] != want[h]["ci"] or got[h]["delta"] != want[h]["delta"]:
            fails.append(f"{h}: got {got[h]['points']} {got[h]['points_ci']} delta {got[h]['delta']}, "
                         f"SM31 {want[h]['points']} {want[h]['ci']} delta {want[h]['delta']}")
    ref = load(REF_NORMAL, REF, set(big))
    zero = compare(ref, ref, big, users)
    if any(zero[h]["delta"] != 0 or zero[h]["points_ci"] != [0.0, 0.0] for h in HEADS.values()) or verdict(zero)[0] != "stop":
        fails.append("reference vs itself is not 0 / stop")
    # a synthetic arm 0.2 points better on instant only (each user's instant LogLoss lowered by 0.2% of the big's): go
    fake = {u: {"all": {"ahead": ref[u]["all"]["ahead"],
                        "instant": (ref[u]["all"]["instant"][0] - 0.002 * big[u]["all"]["instant"][0], ref[u]["all"]["instant"][1])}}
            for u in users}
    fv = verdict(compare(fake, ref, big, users))
    if fv[0] != "go to step 2" or not fv[1].startswith("gain") or "instant" not in fv[1] or "curve" in fv[1]:
        fails.append(f"synthetic -0.2 points instant arm: {fv}")
    print(json.dumps({"users": len(users), "sm31_e2s_query_reproduced": {h: [got[h]["points"], got[h]["points_ci"]] for h in HEADS.values()},
                      "sm31_published": {h: [want[h]["points"], want[h]["ci"]] for h in HEADS.values()},
                      "self_pair": {h: zero[h]["points"] for h in HEADS.values()}, "synthetic_instant_gain_verdict": fv[0],
                      "ok": not fails, "failures": fails}))
    return 0 if not fails else 1


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("decide", "self-test"))
    parser.add_argument("--rows", type=Path, default=ROWS)
    a = parser.parse_args()
    raise SystemExit(decide(a) if a.command == "decide" else self_test(a))
