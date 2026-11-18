#!/usr/bin/env python
"""Show-time interval retrain (SM91) step 2: pre-registered rule (CPU, seconds). DESIGN.md in this folder.

New model  the big model (fork RWKV) + one added epoch from the big model (1 leak-free epoch) with the leak-free retrain
           (SM72) recipe (users 5000-10000, shuffle seed 1, fresh AdamW, warmup 200, 3.5e-4 -> 0, 2 lanes x 8,192
           carried state, deleted-card inputs removed), trained on show-time intervals (track option
           "show_time_intervals": the duration-leak scoring (SM31) `e2s-query` transform). Label sm91s2-big-showtime.
Reference  the big model (1 leak-free epoch), scored on show-time inputs by SM31 (reused rows, read only:
           wt/sm31/artifacts/small-model/sm31/gpu/sm31-inputs.jsonl, big-leakfree-1ep/w8+s8@e2s-query).
Scoring: leak-free (small_model_heldout.py --split-existing --deleted-cards neighbour), w8+s8, --inputs e2s-query, the
732 set (+ the 3 long users). Rule pool: the 722 users of the 732 set that the big model (1 leak-free epoch) normal-input
rows have. Review-weighted benchmark LogLoss; points = % of the big model (1 leak-free epoch) normal-input LogLoss on the
same users (the SM31 unit); paired user bootstrap 2,000 resamples, seed 0, 95% CI.

Rule (fixed 2026-11-02 before launch):
  go to step 3  the new model beats the big model (1 leak-free epoch) on e2s-query inputs by >= 0.10 points on at
                least one head (curve or instant), with that head's 95% CI entirely below zero (delta = new - reference
                <= -0.10 points and CI upper bound < 0).
  stop          otherwise.

Also reported (not the rule): how much of SM31's big-model show-time cost (+0.205 curve / +0.508 instant = reference
e2s-query - reference normal) the new model wins back (gain / cost); the cost left (new e2s-query - reference normal);
the new model on normal inputs vs the reference on normal inputs (the added epoch on answer-to-answer inputs: an added
epoch alone can move LogLoss, so this row is the confound check); the new model's own show-time cost (new e2s - new
normal); the 725-user 732+long pool.

  python artifacts/small-model/sm91s2/sm91s2_rule.py decide [--rows FILE]  -> sm91s2-decision.{json,md}; exit 2 if rows missing
  python artifacts/small-model/sm91s2/sm91s2_rule.py self-test             -> exit 0 if SM31's big-model cost is reproduced
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

ROWS = HERE / "gpu" / "sm91s2-inputs.jsonl"
NEW = "sm91s2-big-showtime/w8+s8"
E2S = "@e2s-query"
SM31 = Path("C:/Users/jonat/workspace/wt/sm31/artifacts/small-model/sm31")                 # read only
REF_E2S = (SM31 / "gpu" / "sm31-inputs.jsonl", "big-leakfree-1ep/w8+s8@e2s-query")       # read only, SM31's rows
BIG = (Path("C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/gpu-big/bigleak-neighbour.jsonl"),
       "big-leakfree-1ep/w8+s8")  # read only, normal inputs: the reference on normal inputs and the points denominator
S72 = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s")  # read only
U732, ULONG = S72 / "eval-users-732-fit.json", S72 / "users-732-long.json"
HEADS = {"ahead": "curve", "instant": "instant"}
GO_POINTS = 0.10                                    # gain needed on one head, points (% of the big model)
SM31_COST = {"curve": 0.205, "instant": 0.508}      # big model (1 leak-free epoch), e2s-query - normal, 722 users (sm31-result)


def load(source, label, users):
    return {u: r for u, r in _load(source, label, users).items() if u in users}


def compare(new, ref, big, users):
    """new - ref per head over `users`: raw review-weighted delta, points of the big model, bootstrap CIs
    (the show-time interval retrain (SM91) step 1 rule's compare, unchanged)."""
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
        return "go to step 3", "gain >= 0.10 points with the CI below zero: " + "; ".join(go)
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
    sets = {"new_e2s": load(rows, NEW + E2S, have) if rows.exists() else {},
            "ref_e2s": load(*REF_E2S, have),
            "new_normal": load(rows, NEW, have) if rows.exists() else {},
            "ref_normal": big}
    missing = {k: [u for u in users if u not in v] for k, v in sets.items()}
    if missing["new_e2s"] or missing["ref_e2s"]:
        print(f"[sm91s2-rule] rule rows missing: new e2s-query {len(missing['new_e2s'])}, reference e2s-query "
              f"{len(missing['ref_e2s'])} of {len(users)} users")
        return 2
    res = {"rule_pool": {"users": len(users), "new_minus_reference_e2s": compare(sets["new_e2s"], sets["ref_e2s"], big, users)}}
    pair = res["rule_pool"]["new_minus_reference_e2s"]
    res["verdict"], res["why"] = verdict(pair)
    cost = compare(sets["ref_e2s"], sets["ref_normal"], big, users)
    info = {"reference_show_time_cost": cost,
            "new_e2s_minus_reference_normal": compare(sets["new_e2s"], sets["ref_normal"], big, users),
            "win_back": {h: {"reference_cost_points": cost[h]["points"], "new_gain_points": round(-pair[h]["points"], 3),
                             "share_of_sm31_cost_pct": round(-pair[h]["points"] / SM31_COST[h] * 100, 1),
                             "share_of_reference_cost_pct": round(-pair[h]["points"] / cost[h]["points"] * 100, 1) if cost[h]["points"] else None}
                         for h in HEADS.values()}}
    if not missing["new_normal"]:
        info["new_normal_minus_reference_normal"] = compare(sets["new_normal"], sets["ref_normal"], big, users)
        info["new_e2s_minus_new_normal"] = compare(sets["new_e2s"], sets["new_normal"], big, users)
    res["rule_pool"].update(info)
    both = sorted(set(big) & set(sets["new_e2s"]) & set(sets["ref_e2s"]))
    if len(both) > len(users):
        res["732+long"] = {"users": len(both), "new_minus_reference_e2s": compare(sets["new_e2s"], sets["ref_e2s"], big, both)}
        bn = sorted(set(both) & set(sets["new_normal"]))
        if len(bn) == len(both):
            res["732+long"]["new_normal_minus_reference_normal"] = compare(sets["new_normal"], sets["ref_normal"], big, both)
    res.update(rows=str(rows), labels={"new_e2s": NEW + E2S, "ref_e2s": f"{REF_E2S[0]} {REF_E2S[1]}", "new_normal": NEW,
                                       "ref_normal": f"{BIG[0]} {BIG[1]}"}, big=f"{BIG[0]} {BIG[1]}",
               missing_informational={k: len(v) for k, v in missing.items()},
               rule={"go_if": f"delta (new - big model (1 leak-free epoch), e2s-query) <= -{GO_POINTS} points with 95% CI upper < 0 on curve or instant",
                     "unit": "points = % of the big model (1 leak-free epoch) normal-input LogLoss, review-weighted, 722 users"})
    (HERE / "sm91s2-decision.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    md = ["# Show-time interval retrain (SM91) step 2: the show-time big model vs the big model (1 leak-free epoch), both scored on show-time inputs", "",
          "Leak-free scoring (deleted cards: neighbour), w8+s8, `--inputs e2s-query`, review-weighted LogLoss, user bootstrap 2,000 "
          "seed 0, 95% CI; points = % of the big model (1 leak-free epoch) with normal inputs. New: the big model (1 leak-free epoch) "
          "+ one added epoch with the leak-free retrain (SM72) recipe (users 5000-10000, 3.5e-4 -> 0), trained on show-time "
          "intervals. Reference: the big model (1 leak-free epoch), rows from the duration-leak scoring (SM31).", "",
          "| pool | users | comparison | head | new | ref | delta [95% CI] | points [95% CI] | users better / worse |",
          "|---|---|---|---|---|---|---|---|---|"]
    names = {"new_minus_reference_e2s": "new e2s - reference e2s (the rule)",
             "reference_show_time_cost": "reference e2s - reference normal (SM31's show-time cost)",
             "new_e2s_minus_reference_normal": "new e2s - reference normal (cost left)",
             "new_normal_minus_reference_normal": "new normal - reference normal (the added epoch on normal inputs)",
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
    wb = res["rule_pool"]["win_back"]
    md += ["", "Win-back (not the rule): " + "; ".join(
        f"{h}: gain {wb[h]['new_gain_points']:+.3f} points = {wb[h]['share_of_sm31_cost_pct']}% of SM31's big-model show-time "
        f"cost (+{SM31_COST[h]:.3f})" for h in HEADS.values()) + "."]
    md += ["", f"Rule: go to step 3 if the new model beats the big model (1 leak-free epoch) by >= {GO_POINTS:.2f} points on curve "
           "or instant with the 95% CI below zero; else stop.", "", f"Verdict: **{res['verdict']}** ({res['why']})."]
    (HERE / "sm91s2-decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    c, i = pair["curve"], pair["instant"]
    print(f"[sm91s2-rule] {len(users)} users, new - big model (1 leak-free epoch) on e2s-query: curve {c['points']:+.3f} points "
          f"{c['points_ci']}, instant {i['points']:+.3f} points {i['points_ci']}: {res['verdict'].upper()} ({res['why']})")
    return 0


def self_test(_args):
    """SM31's big model, e2s-query rows vs its normal rows, through compare(): must reproduce sm31-result.json (points,
    CIs and deltas, 722 users); the reference against itself: 0 and stop; a synthetic model 0.2 points better on
    instant only: go."""
    users, big = pools()
    result = json.loads((SM31 / "sm31-result.json").read_text())
    label = result["labels"]["big"]
    if label + E2S != REF_E2S[1] or Path(result["normal_rows"][label]).resolve() != BIG[0].resolve() or label != BIG[1]:
        print(json.dumps({"ok": False, "failures": [f"sm31-result labels / rows {label} {result['normal_rows'].get(label)} "
                                                    f"do not match {REF_E2S} {BIG}"]}))
        return 1
    e2s = load(*REF_E2S, set(big))
    got = compare(e2s, big, big, users)
    want = result["pools"]["732 (rule)"][label]["e2s-query"]
    fails = []
    if len(users) != 722:
        fails.append(f"rule pool has {len(users)} users, not 722")
    for h in HEADS.values():
        if got[h]["points"] != want[h]["points"] or got[h]["points_ci"] != want[h]["ci"] or got[h]["delta"] != want[h]["delta"]:
            fails.append(f"{h}: got {got[h]['points']} {got[h]['points_ci']} delta {got[h]['delta']}, "
                         f"SM31 {want[h]['points']} {want[h]['ci']} delta {want[h]['delta']}")
        if round(got[h]["points"], 3) != SM31_COST[h]:
            fails.append(f"{h}: SM31_COST {SM31_COST[h]} != {got[h]['points']}")
    zero = compare(e2s, e2s, big, users)
    if any(zero[h]["delta"] != 0 or zero[h]["points_ci"] != [0.0, 0.0] for h in HEADS.values()) or verdict(zero)[0] != "stop":
        fails.append("reference vs itself is not 0 / stop")
    fake = {u: {"all": {"ahead": e2s[u]["all"]["ahead"],
                        "instant": (e2s[u]["all"]["instant"][0] - 0.002 * big[u]["all"]["instant"][0], e2s[u]["all"]["instant"][1])}}
            for u in users}
    fv = verdict(compare(fake, e2s, big, users))
    if fv[0] != "go to step 3" or not fv[1].startswith("gain") or "instant" not in fv[1] or "curve" in fv[1]:
        fails.append(f"synthetic -0.2 points instant model: {fv}")
    print(json.dumps({"users": len(users), "sm31_big_e2s_query_cost_reproduced": {h: [got[h]["points"], got[h]["points_ci"]] for h in HEADS.values()},
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
