#!/usr/bin/env python
"""Show-time retrain of the release (SM92): pre-registered rule (CPU, seconds). DESIGN.md in this folder.

New        the leak-free small model (SM72s) constant-LR epoch 25 (the release's start) + epoch 26 on show-time inputs
           (the release's epoch 21-25 recipe, flag "show_time_intervals": true) + the release's exact cool-down (one
           epoch, 32k windows, fresh Muon, LR 7e-4 -> 0, same users / order / seeds / KD replay targets) on show-time
           inputs. Label sm92-e26-dec1ep32k/w8+s8 (rows gpu/sm92-inputs.jsonl).
Release    the released small model (cool-down from epoch 25, SM63): e2s-query rows from the duration-leak scoring
           (SM31), normal rows from the leak-free small model (SM72s) scoring. Read only.
Scoring: leak-free (small_model_heldout.py --split-existing --deleted-cards neighbour), w8+s8, --inputs e2s-query and
normal, the 732 set (+ the 3 long users). Rule pool: the 722 users of the 732 set that the big model (1 leak-free epoch)
normal-input rows have. Review-weighted benchmark LogLoss; points = % of the big model (1 leak-free epoch)
normal-input LogLoss on the same users; paired user bootstrap 2,000 resamples, seed 0, 95% CI.

Rule (fixed 2026-11-03 before launch):
  ship candidate  the new model beats the release, both scored on e2s-query inputs, by >= 0.05 points on at least one
                  head (curve or instant) with that head's 95% CI entirely below zero, and no head is worse by >= 0.05
                  points with its CI entirely above zero.
  no ship         otherwise.
Confound: the new model has one more epoch (26 vs 25) than the release; "new normal - release normal" shows its size.

Also reported (not the rule): new e2s vs the big model (1 leak-free epoch) e2s-query (SM31 rows) and new normal vs
its normal rows; new e2s vs the show-time big model e2s-query (step 2 rows, wt/sm91s2); the new model's own show-time
cost (new e2s - new normal); new normal - release normal; the release's own rows against the same references; the
smaller-model bar (<= +0.5% curve / <= +0.4% instant, % of the reference's own LogLoss) three ways: (a) normal vs the
big model normal, (b) e2s-query vs the big model e2s-query, (c) e2s-query vs the show-time big model e2s-query; the
725-user 732+long pool.

  python artifacts/small-model/sm92/sm92_rule.py decide [--rows FILE]  -> sm92-decision.{json,md}; exit 2 if rule rows missing
  python artifacts/small-model/sm92/sm92_rule.py self-test             -> exit 0 if step 3's published numbers are reproduced
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

ROWS = HERE / "gpu" / "sm92-inputs.jsonl"
NEW = "sm92-e26-dec1ep32k/w8+s8"
E2S = "@e2s-query"
RELEASE_LABEL = "s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/w8+s8"
SM31 = Path("C:/Users/jonat/workspace/wt/sm31/artifacts/small-model/sm31/gpu/sm31-inputs.jsonl")  # read only
REL_E2S = (SM31, RELEASE_LABEL + E2S)
REL_NORMAL = (Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/gpu/small-lf-neighbour.jsonl"), RELEASE_LABEL)
BIG_E2S = (SM31, "big-leakfree-1ep/w8+s8@e2s-query")
SHOWBIG_E2S = (Path("C:/Users/jonat/workspace/wt/sm91s2/artifacts/small-model/sm91s2/gpu/sm91s2-inputs.jsonl"),
               "sm91s2-big-showtime/w8+s8@e2s-query")
BIG = (Path("C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/gpu-big/bigleak-neighbour.jsonl"),
       "big-leakfree-1ep/w8+s8")  # normal inputs: the points denominator
STEP1 = (Path("C:/Users/jonat/workspace/wt/sm91/artifacts/small-model/sm91/gpu/sm91-inputs.jsonl"),
         "sm91-arm-e21-q32k/w8+s8@e2s-query")   # self-test only
STEP3 = ROOT / "artifacts/small-model/sm91s3/sm91s3-decision.json"   # self-test only
S72 = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s")  # read only
U732, ULONG = S72 / "eval-users-732-fit.json", S72 / "users-732-long.json"
HEADS = {"ahead": "curve", "instant": "instant"}
GO_POINTS = 0.05
BAR = {"curve": 0.5, "instant": 0.4}   # smaller-model bar, % of the reference's LogLoss


def load(source, label, users):
    return {u: r for u, r in _load(source, label, users).items() if u in users}


def compare(new, ref, big, users):
    """new - ref per head over `users` (the SM91 steps' compare): raw review-weighted delta, points of the big model
    (1 leak-free epoch) normal LogLoss, bootstrap CIs, plus pct = % of the reference's own LogLoss (the bar)."""
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


def fmt(x):
    return f"{x['points']:+.3f} [{x['points_ci'][0]:+.3f}, {x['points_ci'][1]:+.3f}]"


def verdict(pair):
    go = [f"{h} {fmt(pair[h])}" for h in HEADS.values() if pair[h]["points"] <= -GO_POINTS and pair[h]["points_ci"][1] < 0]
    bad = [f"{h} {fmt(pair[h])}" for h in HEADS.values() if pair[h]["points"] >= GO_POINTS and pair[h]["points_ci"][0] > 0]
    why = ", ".join(f"{h} {fmt(pair[h])}" for h in HEADS.values())
    if go and not bad:
        return "ship candidate", "gain >= 0.05 points with the CI below zero: " + "; ".join(go)
    if bad:
        return "no ship", f"a head is worse by >= 0.05 points with the CI above zero ({'; '.join(bad)}); {why}"
    return "no ship", f"no head gains >= 0.05 points with the CI below zero ({why})"


def pools():
    users732 = set(json.loads(U732.read_text())["users"])
    long_users = set(json.loads(ULONG.read_text())["users"])
    big = load(*BIG, users732 | long_users)
    return sorted(users732 & set(big)), big


def bar(x):
    return {h: {"pct": x[h]["pct_of_ref"], "ci": x[h]["pct_of_ref_ci"], "limit": BAR[h], "passes": x[h]["pct_of_ref"] <= BAR[h]}
            for h in HEADS.values()}


NAMES = {"new_minus_release_e2s": "new e2s - the release e2s (the rule)",
         "new_normal_minus_release_normal": "new normal - the release normal (the extra-epoch confound)",
         "new_e2s_minus_new_normal": "new e2s - new normal (its own show-time cost)",
         "release_e2s_minus_release_normal": "the release e2s - the release normal (its show-time cost)",
         "new_e2s_minus_big_e2s": "new e2s - the big model (1 leak-free epoch) e2s",
         "release_e2s_minus_big_e2s": "the release e2s - the big model (1 leak-free epoch) e2s",
         "new_normal_minus_big_normal": "new normal - the big model (1 leak-free epoch) normal",
         "release_normal_minus_big_normal": "the release normal - the big model (1 leak-free epoch) normal",
         "new_e2s_minus_showbig_e2s": "new e2s - the show-time big model e2s",
         "release_e2s_minus_showbig_e2s": "the release e2s - the show-time big model e2s",
         "new_e2s_minus_big_normal": "new e2s - the big model (1 leak-free epoch) normal (cost left)",
         "release_e2s_minus_big_normal": "the release e2s - the big model (1 leak-free epoch) normal"}


def decide(args):
    users, big = pools()
    rows = Path(args.rows)
    have = set(big)
    s = {"new_e2s": load(rows, NEW + E2S, have) if rows.exists() else {},
         "new_normal": load(rows, NEW, have) if rows.exists() else {},
         "release_e2s": load(*REL_E2S, have), "release_normal": load(*REL_NORMAL, have),
         "big_e2s": load(*BIG_E2S, have), "showbig_e2s": load(*SHOWBIG_E2S, have), "big_normal": big}
    missing = {k: [u for u in users if u not in v] for k, v in s.items()}
    if missing["new_e2s"] or missing["release_e2s"]:
        print(f"[sm92-rule] rule rows missing: new e2s-query {len(missing['new_e2s'])}, the release e2s-query "
              f"{len(missing['release_e2s'])} of {len(users)} users")
        return 2
    pair = compare(s["new_e2s"], s["release_e2s"], big, users)
    res = {"rule_pool": {"users": len(users), "new_minus_release_e2s": pair}}
    res["verdict"], res["why"] = verdict(pair)
    info = res["rule_pool"]
    pairs = [("new_normal", "release_normal"), ("new_e2s", "new_normal"), ("release_e2s", "release_normal"),
             ("new_e2s", "big_e2s"), ("release_e2s", "big_e2s"), ("new_normal", "big_normal"), ("release_normal", "big_normal"),
             ("new_e2s", "showbig_e2s"), ("release_e2s", "showbig_e2s"), ("new_e2s", "big_normal"), ("release_e2s", "big_normal")]
    for x, y in pairs:
        if not missing[x] and not missing[y]:
            info[f"{x}_minus_{y}"] = compare(s[x], s[y], big, users)
    bars = {}
    for who in ("new", "release"):
        b = {}
        for key, src in (("a_normal_vs_big_normal", f"{who}_normal_minus_big_normal"), ("b_e2s_vs_big_e2s", f"{who}_e2s_minus_big_e2s"),
                         ("c_e2s_vs_showbig_e2s", f"{who}_e2s_minus_showbig_e2s")):
            if src in info:
                b[key] = bar(info[src])
        bars[who] = b
    info["smaller_model_bar"] = bars
    both = sorted(set(big) & set(s["new_e2s"]) & set(s["release_e2s"]))
    if len(both) > len(users):
        res["732+long"] = {"users": len(both), "new_minus_release_e2s": compare(s["new_e2s"], s["release_e2s"], big, both)}
        bothn = sorted(set(both) & set(s["new_normal"]) & set(s["release_normal"]))
        if len(bothn) == len(both):
            res["732+long"]["new_normal_minus_release_normal"] = compare(s["new_normal"], s["release_normal"], big, both)
    res.update(rows=str(rows),
               labels={"new_e2s": NEW + E2S, "new_normal": NEW, "release_e2s": f"{REL_E2S[0]} {REL_E2S[1]}",
                       "release_normal": f"{REL_NORMAL[0]} {REL_NORMAL[1]}", "big_e2s": f"{BIG_E2S[0]} {BIG_E2S[1]}",
                       "showbig_e2s": f"{SHOWBIG_E2S[0]} {SHOWBIG_E2S[1]}", "big_normal": f"{BIG[0]} {BIG[1]}"},
               missing_informational={k: len(v) for k, v in missing.items()},
               confound="the new model has one more epoch (26) than the release (25); new normal - release normal shows its size",
               rule={"ship_candidate_if": f"delta (new - the release, both e2s-query) <= -{GO_POINTS} points with 95% CI upper < 0 "
                                         f"on curve or instant, and no head >= +{GO_POINTS} points with CI lower > 0",
                     "unit": "points = % of the big model (1 leak-free epoch) normal-input LogLoss, review-weighted, 722 users"})
    (HERE / "sm92-decision.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    md = ["# Show-time retrain of the release (SM92): the new model vs the released small model, both scored on show-time inputs", "",
          "Leak-free scoring (deleted cards: neighbour), w8+s8, review-weighted LogLoss, user bootstrap 2,000 seed 0, 95% CI; points = % of "
          "the big model (1 leak-free epoch) with normal inputs. New: the leak-free small model (SM72s) epoch 25 + epoch 26 on show-time "
          "inputs + the release's one-epoch 32k Muon cool-down on show-time inputs (same users, order, seeds, KD targets). Release: the "
          "released small model (cool-down from epoch 25). Confound: the new model has one more epoch than the release (the normal-vs-normal "
          "row shows its size).", "",
          "| pool | users | comparison | head | model | ref | delta [95% CI] | points [95% CI] | users better / worse |",
          "|---|---|---|---|---|---|---|---|---|"]
    for pool in ("rule_pool", "732+long"):
        if pool not in res:
            continue
        for key, name in NAMES.items():
            if key not in res[pool]:
                continue
            for h in HEADS.values():
                x = res[pool][key][h]
                md.append(f"| {'732 (rule)' if pool == 'rule_pool' else pool} | {res[pool]['users']} | {name} | {h} | {x['new']:.5f} | "
                          f"{x['ref']:.5f} | {x['delta']:+.6f} [{x['delta_ci'][0]:+.6f}, {x['delta_ci'][1]:+.6f}] | "
                          f"{fmt(x)} | {x['users_better']} / {x['users_worse']} |")
    md += ["", "Smaller-model bar (<= +0.5% curve / <= +0.4% instant, % of the reference's own LogLoss; not the rule):", "",
           "| model | reference | curve % [95% CI] | instant % [95% CI] | passes |", "|---|---|---|---|---|"]
    refname = {"a_normal_vs_big_normal": "(a) normal inputs vs the big model (1 leak-free epoch) normal",
               "b_e2s_vs_big_e2s": "(b) e2s-query vs the big model (1 leak-free epoch) e2s-query",
               "c_e2s_vs_showbig_e2s": "(c) e2s-query vs the show-time big model e2s-query"}
    for who, label in (("new", "new"), ("release", "the release")):
        for key, b in bars[who].items():
            md.append(f"| {label} | {refname[key]} | {b['curve']['pct']:+.3f} [{b['curve']['ci'][0]:+.3f}, {b['curve']['ci'][1]:+.3f}] | "
                      f"{b['instant']['pct']:+.3f} [{b['instant']['ci'][0]:+.3f}, {b['instant']['ci'][1]:+.3f}] | "
                      f"{'yes' if b['curve']['passes'] and b['instant']['passes'] else 'no'} |")
    md += ["", f"Rule: ship candidate if the new model beats the release on e2s-query by >= {GO_POINTS:.2f} points on curve or instant "
           "with the 95% CI below zero, and no head is worse by >= 0.05 points with the CI above zero; else no ship.", "",
           f"Verdict: **{res['verdict']}** ({res['why']})."]
    (HERE / "sm92-decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    c, i = pair["curve"], pair["instant"]
    print(f"[sm92-rule] {len(users)} users, new - the release on e2s-query: curve {fmt(c)}, instant {fmt(i)}: "
          f"{res['verdict'].upper()} ({res['why']})")
    return 0


def self_test(_args):
    """Step 3's published "step 1's arm e2s - the released small model e2s" pair (sm91s3-decision.json) through
    compare() with this script's release rows: points, CIs, deltas, users better / worse equal, 722 users; the release
    against itself: 0 / no ship; synthetic models: 0.06 points better on instant only -> ship candidate; 0.04 better ->
    no ship; 0.06 better on instant but 0.06 worse on curve -> no ship; 0.06 better on curve only -> ship candidate."""
    users, big = pools()
    want = json.loads(STEP3.read_text())["rule_pool"]["step1_e2s_minus_small_e2s"]
    rel = load(*REL_E2S, set(big))
    got = compare(load(*STEP1, set(big)), rel, big, users)
    fails = []
    if len(users) != 722:
        fails.append(f"rule pool has {len(users)} users, not 722")
    for h in HEADS.values():
        for k in ("new", "ref", "points", "points_ci", "delta", "delta_ci", "users_better", "users_worse"):
            if got[h][k] != want[h][k]:
                fails.append(f"{h} {k}: got {got[h][k]}, step 3 {want[h][k]}")
    zero = compare(rel, rel, big, users)
    if any(zero[h]["delta"] != 0 or zero[h]["points_ci"] != [0.0, 0.0] for h in HEADS.values()) or verdict(zero)[0] != "no ship":
        fails.append("the release vs itself is not 0 / no ship")

    def fake(curve_points, instant_points):
        return {u: {"all": {"ahead": (rel[u]["all"]["ahead"][0] + curve_points / 100 * big[u]["all"]["ahead"][0], rel[u]["all"]["ahead"][1]),
                            "instant": (rel[u]["all"]["instant"][0] + instant_points / 100 * big[u]["all"]["instant"][0], rel[u]["all"]["instant"][1])}}
                for u in users}
    checks = {"instant -0.06": (fake(0, -0.06), "ship candidate"), "instant -0.04": (fake(0, -0.04), "no ship"),
              "instant -0.06 curve +0.06": (fake(0.06, -0.06), "no ship"), "curve -0.06": (fake(-0.06, 0), "ship candidate")}
    synth = {}
    for name, (model, expect) in checks.items():
        v = verdict(compare(model, rel, big, users))
        synth[name] = v[0]
        if v[0] != expect:
            fails.append(f"synthetic {name}: {v}")
    print(json.dumps({"users": len(users), "step3_pair_reproduced": {h: [got[h]["points"], got[h]["points_ci"]] for h in HEADS.values()},
                      "step3_published": {h: [want[h]["points"], want[h]["points_ci"]] for h in HEADS.values()},
                      "self_pair": {h: zero[h]["points"] for h in HEADS.values()}, "synthetic": synth,
                      "ok": not fails, "failures": fails}))
    return 0 if not fails else 1


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("decide", "self-test"))
    parser.add_argument("--rows", type=Path, default=ROWS)
    a = parser.parse_args()
    raise SystemExit(decide(a) if a.command == "decide" else self_test(a))
