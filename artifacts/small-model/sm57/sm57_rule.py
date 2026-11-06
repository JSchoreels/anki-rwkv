#!/usr/bin/env python
"""The instant-only small model (SM57): pre-registered rule (CPU, seconds). DESIGN.md in this folder.

Instant-only  the leak-free small model (SM72s) constant-LR epoch 25 (the release's start) + epoch 26 (the release's
              epoch 21-25 recipe) + the release's exact cool-down (one epoch, 32k windows, fresh Muon, LR 7e-4 -> 0,
              same users / order / seeds / KD replay targets), both trained with "curve_loss_weight": 0.0 (every curve
              term of the loss, curve KD included, at weight 0). Label sm57-inst-dec1ep32k/w8+s8.
Joint         the same epoch 26 + cool-down with the normal joint loss (no extra option). Label sm57-joint-dec1ep32k/w8+s8.
Release       the released small model (cool-down from epoch 25, SM63): normal-input rows of the leak-free small model
              (SM72s) scoring. Read only.
All normal (answer-time) inputs. Scoring: leak-free (small_model_heldout.py --split-existing --deleted-cards neighbour),
w8+s8, the 732 set (+ the 3 long users). Rule pool: the 722 users of the 732 set that the big model (1 leak-free epoch)
rows have. Review-weighted LogLoss; points = % of the big model (1 leak-free epoch) normal-input LogLoss on the same
users; paired user bootstrap 2,000 resamples, seed 0, 95% CI (sm92_rule.compare, unchanged).

Rule (fixed 2026-11-06 before launch; minimum size added by the user 2026-11-07 ~00:40, before the joint arm was
scored: "we would need at least .05 better result"), instant head only, instant-only arm - joint arm:
  go    the 95% CI is entirely below zero AND the gain is at least 0.05 points (points <= -0.05; the release-candidate
        size used by the show-time retrain of the release (SM92)). Then stage 2 is proposed (not run): a from-scratch
        comparison at small scale (DESIGN.md).
  stop  otherwise.
Also reported (not the rule): instant-only - the release (instant); joint - the release (both heads: the one-more-epoch
confound); the joint arm's curve; both arms vs the big model (1 leak-free epoch); the smaller-model bar (<= +0.5% curve /
<= +0.4% instant) for the joint arm and, instant only, for the instant-only arm; the 725-user 732+long pool; training
minutes, reviews/s and GPU memory per arm and stage (gpu/segments.jsonl + result.json). The instant-only arm's curve
head is not trained (zero gradients; its AdamW momentum and the weight decay still move it), so its curve is listed for
completeness only.

  python artifacts/small-model/sm57/sm57_rule.py decide [--rows FILE]  -> sm57-decision.{json,md}; exit 2 if rule rows missing
  python artifacts/small-model/sm57/sm57_rule.py self-test             -> exit 0 if the SM92 normal pair is reproduced
"""
from argparse import ArgumentParser
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("sm92_rule", ROOT / "artifacts/small-model/sm92/sm92_rule.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)
load, compare, fmt, pools, bar, HEADS, REL_NORMAL, BIG = R.load, R.compare, R.fmt, R.pools, R.bar, R.HEADS, R.REL_NORMAL, R.BIG

ROWS = HERE / "gpu" / "sm57-rows.jsonl"
INST = "sm57-inst-dec1ep32k/w8+s8"
JOINT = "sm57-joint-dec1ep32k/w8+s8"
SIZE_POINTS = 0.05
TRACK = "E89-F65e-F77a"
SM = ROOT / "artifacts/small-model"
FOLDERS = {"instant-only": {"epoch 26": SM / "sm57-inst-e26", "cool-down": SM / "sm57-inst-e26-dec-1ep32k-muon"},
           "joint": {"epoch 26": SM / "sm57-joint-e26", "cool-down": SM / "sm57-joint-e26-dec-1ep32k-muon"}}
SM92_ROWS = Path("C:/Users/jonat/workspace/wt/sm92/artifacts/small-model/sm92/gpu/sm92-inputs.jsonl")   # self-test only
SM92_DECISION = Path("C:/Users/jonat/workspace/wt/sm92/artifacts/small-model/sm92/sm92-decision.json")  # self-test only

NAMES = {"inst_minus_joint": "instant-only - joint (the rule: instant)",
         "inst_minus_release": "instant-only - the release",
         "joint_minus_release": "joint - the release (the one-more-epoch confound)",
         "inst_minus_big": "instant-only - the big model (1 leak-free epoch)",
         "joint_minus_big": "joint - the big model (1 leak-free epoch)",
         "release_minus_big": "the release - the big model (1 leak-free epoch)"}


def verdict(pair):
    x = pair["instant"]
    if x["points_ci"][1] >= 0:
        return "stop", f"instant {fmt(x)}: CI not below zero"
    if x["points"] > -SIZE_POINTS:
        return "stop", f"instant {fmt(x)}: CI below zero but the gain is below the {SIZE_POINTS:.2f}-point minimum"
    return "go", f"instant {fmt(x)}: CI below zero and at least the {SIZE_POINTS:.2f}-point minimum"


def training_stats():
    """Per arm and stage: segments (gpu/segments.jsonl, written by the launcher), the final segment's result.json."""
    seg_path = HERE / "gpu" / "segments.jsonl"
    segs = [json.loads(line) for line in seg_path.read_text().splitlines() if line.strip()] if seg_path.exists() else []
    out = {}
    for arm, stages in FOLDERS.items():
        for stage, folder in stages.items():
            res_path = folder / "expertium" / TRACK / "result.json"
            r = json.loads(res_path.read_text()) if res_path.exists() else {}
            mine = [s for s in segs if Path(s["folder"]).name == folder.name]
            out[f"{arm} / {stage}"] = {
                "status": r.get("status"), "segments": len(mine),
                "wall_minutes": round(sum(s.get("minutes", 0) for s in mine), 1),
                "nvidia_smi_max_used_mib": max((s.get("nvidia_smi_max_used_mib") or 0 for s in mine), default=None),
                "nvidia_smi_baseline_mib": min((s.get("nvidia_smi_baseline_mib") or 1e9 for s in mine), default=None),
                "last_segment": {"reviews": r.get("review_presentations"), "training_minutes": round(r.get("training_seconds", 0) / 60, 1),
                                 "reviews_per_second": round(r.get("reviews_per_second", 0)), "peak_allocated_mib": r.get("peak_allocated_mib"),
                                 "peak_reserved_mib": r.get("peak_reserved_mib"), "curve_loss_weight": r.get("curve_loss_weight")}}
    return out


def decide(args):
    users, big = pools()
    rows = Path(args.rows)
    have = set(big)
    s = {"inst": load(rows, INST, have) if rows.exists() else {}, "joint": load(rows, JOINT, have) if rows.exists() else {},
         "release": load(*REL_NORMAL, have), "big": big}
    missing = {k: [u for u in users if u not in v] for k, v in s.items()}
    res = {"rule_pool": {"users": len(users)}}
    info = res["rule_pool"]
    for x, y in (("inst", "joint"), ("inst", "release"), ("joint", "release"), ("inst", "big"), ("joint", "big"), ("release", "big")):
        if not missing[x] and not missing[y]:
            info[f"{x}_minus_{y}"] = compare(s[x], s[y], big, users)
    if "inst_minus_joint" in info:
        res["verdict"], res["why"] = verdict(info["inst_minus_joint"])
    else:
        res["verdict"], res["why"] = "pending", (f"rule rows missing: instant-only {len(missing['inst'])}, joint "
                                                 f"{len(missing['joint'])} of {len(users)} users")
    bars = {}
    for who, key in (("joint", "joint_minus_big"), ("instant-only", "inst_minus_big"), ("release", "release_minus_big")):
        if key in info:
            bars[who] = bar(info[key])
    info["smaller_model_bar"] = bars
    present = [k for k in ("inst", "joint") if s[k]]
    both = sorted(set(big).intersection(*(set(s[k]) for k in present + ["release"]))) if present else []
    if len(both) > len(users):
        res["732+long"] = {"users": len(both)}
        for x, y in (("inst", "joint"), ("inst", "release"), ("joint", "release")):
            if s[x] and s[y]:
                res["732+long"][f"{x}_minus_{y}"] = compare(s[x], s[y], big, both)
    res["training"] = training_stats()
    res.update(rows=str(rows), labels={"inst": INST, "joint": JOINT, "release": f"{REL_NORMAL[0]} {REL_NORMAL[1]}",
                                       "big": f"{BIG[0]} {BIG[1]}"},
               missing={k: len(v) for k, v in missing.items()},
               rule={"go_if": "instant-only - joint, instant head, 95% CI upper < 0 and points <= -0.05 (normal inputs, 722 users; minimum added by the user 11-07)",
                     "unit": "points = % of the big model (1 leak-free epoch) normal-input LogLoss, review-weighted"})
    (HERE / "sm57-decision.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    md = ["# The instant-only small model (SM57): instant-only vs the matched joint arm", "",
          "Leak-free scoring (deleted cards: neighbour), normal inputs, w8+s8, review-weighted LogLoss, user bootstrap 2,000 seed 0, "
          "95% CI; points = % of the big model (1 leak-free epoch). Both arms: the leak-free small model (SM72s) epoch 25 + epoch 26 + "
          "the release's one-epoch 32k Muon cool-down (same users, order, seeds, KD targets); the instant-only arm trains both with "
          "every curve loss term at weight 0. Release: the released small model (cool-down from epoch 25, SM63). The instant-only "
          "arm's curve head is untrained (curve rows for completeness only).", "",
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
                tag = " (untrained curve head)" if h == "curve" and key.startswith("inst") else ""
                md.append(f"| {'732 (rule)' if pool == 'rule_pool' else pool} | {res[pool]['users']} | {name}{tag} | {h} | "
                          f"{x['new']:.5f} | {x['ref']:.5f} | {x['delta']:+.6f} [{x['delta_ci'][0]:+.6f}, {x['delta_ci'][1]:+.6f}] | "
                          f"{fmt(x)} | {x['users_better']} / {x['users_worse']} |")
    md += ["", "Smaller-model bar (<= +0.5% curve / <= +0.4% instant vs the big model (1 leak-free epoch), % of its LogLoss; not the rule):", "",
           "| model | curve % [95% CI] | instant % [95% CI] | passes |", "|---|---|---|---|"]
    for who, b in bars.items():
        passes = b["instant"]["passes"] if who == "instant-only" else b["curve"]["passes"] and b["instant"]["passes"]
        md.append(f"| {who} | {b['curve']['pct']:+.3f} [{b['curve']['ci'][0]:+.3f}, {b['curve']['ci'][1]:+.3f}] | "
                  f"{b['instant']['pct']:+.3f} [{b['instant']['ci'][0]:+.3f}, {b['instant']['ci'][1]:+.3f}] | "
                  f"{'yes' if passes else 'no'}{' (instant only)' if who == 'instant-only' else ''} |")
    md += ["", "Training (wall minutes over all segments; reviews/s, training minutes and peak memory of the last segment; nvidia-smi max "
           "includes the desktop's baseline):", "",
           "| arm / stage | status | segments | wall min | last segment: reviews | training min | reviews/s | peak allocated MiB | "
           "peak reserved MiB | nvidia-smi max / baseline MiB |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, t in res["training"].items():
        ls = t["last_segment"]
        md.append(f"| {k} | {t['status']} | {t['segments']} | {t['wall_minutes']} | {ls['reviews'] or 0:,} | {ls['training_minutes']} | "
                  f"{ls['reviews_per_second']:,} | {round(ls['peak_allocated_mib'] or 0):,} | {round(ls['peak_reserved_mib'] or 0):,} | "
                  f"{t['nvidia_smi_max_used_mib']} / {t['nvidia_smi_baseline_mib']} |")
    md += ["", "Rule: go if instant-only - joint on the instant head has its 95% CI entirely below zero and a gain of at least "
           f"{SIZE_POINTS:.2f} points (minimum added by the user 11-07, before the joint arm was scored); else stop. Go proposes stage 2 "
           "(from-scratch comparison at small scale, DESIGN.md), not run.", "", f"Verdict: **{res['verdict']}** ({res['why']})."]
    (HERE / "sm57-decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    line = f"[sm57-rule] {len(users)} users: {res['verdict'].upper()} ({res['why']})"
    if "inst_minus_release" in info:
        line += f"; instant-only - release instant {fmt(info['inst_minus_release']['instant'])}"
    if "joint_minus_release" in info:
        line += f"; joint - release curve {fmt(info['joint_minus_release']['curve'])}, instant {fmt(info['joint_minus_release']['instant'])}"
    print(line)
    return 0 if res["verdict"] != "pending" else 2


def self_test(_args):
    """The show-time retrain of the release (SM92)'s published "new normal - the release normal" pair (sm92-decision.json)
    through this script's loading + sm92_rule.compare: points, CIs, users better / worse equal, 722 users; the release vs
    itself: stop; synthetic arms vs the release: instant -0.08 points -> go, -0.03 (below the minimum) -> stop, +0.03 -> stop, 0 -> stop."""
    users, big = pools()
    want = json.loads(SM92_DECISION.read_text())["rule_pool"]["new_normal_minus_release_normal"]
    rel = load(*REL_NORMAL, set(big))
    got = compare(load(SM92_ROWS, "sm92-e26-dec1ep32k/w8+s8", set(big)), rel, big, users)
    fails = []
    if len(users) != 722:
        fails.append(f"rule pool has {len(users)} users, not 722")
    for h in HEADS.values():
        for k in ("new", "ref", "points", "points_ci", "delta", "delta_ci", "users_better", "users_worse"):
            if got[h][k] != want[h][k]:
                fails.append(f"{h} {k}: got {got[h][k]}, SM92 {want[h][k]}")
    zero = compare(rel, rel, big, users)
    if verdict(zero)[0] != "stop":
        fails.append("the release vs itself is not stop")

    def fake(instant_points):
        return {u: {"all": {"ahead": rel[u]["all"]["ahead"],
                            "instant": (rel[u]["all"]["instant"][0] + instant_points / 100 * big[u]["all"]["instant"][0],
                                        rel[u]["all"]["instant"][1])}} for u in users}
    synth = {}
    for pts, expect in ((-0.08, "go"), (-0.03, "stop"), (0.03, "stop"), (0.0, "stop")):
        v = verdict(compare(fake(pts), rel, big, users))
        synth[str(pts)] = v[0]
        if v[0] != expect:
            fails.append(f"synthetic {pts}: {v}")
    print(json.dumps({"users": len(users), "sm92_pair_reproduced": {h: [got[h]["points"], got[h]["points_ci"]] for h in HEADS.values()},
                      "sm92_published": {h: [want[h]["points"], want[h]["points_ci"]] for h in HEADS.values()},
                      "synthetic": synth, "ok": not fails, "failures": fails}))
    return 0 if not fails else 1


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("decide", "self-test"))
    parser.add_argument("--rows", type=Path, default=ROWS)
    a = parser.parse_args()
    raise SystemExit(decide(a) if a.command == "decide" else self_test(a))
