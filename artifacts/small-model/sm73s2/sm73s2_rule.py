#!/usr/bin/env python
"""Smaller model, card stream 1 layer (SM73 stage 2) early check: pre-registered rule (CPU, seconds).

Arm       the leak-free small model (SM72s) epoch 20 cut to 1 card layer (w64-c1, scripts/sm73s2.py cut), + 1
          constant-LR epoch (SM72s's epoch-21 protocol) + the quarter-epoch 32k Muon cool-down (the reference's users,
          order, schedule and options; sm73s2_pair.py).
Reference SM72s's own epoch 21 + the same quarter-epoch 32k Muon cool-down (the tiny architecture additions (SM77)
          reference, wt/sm77/artifacts/small-model/sm77-ref-e21-dec-q32k-muon, label sm77-ref-e21-q32k/w8+s8, read only).
Both have 21 epochs + a quarter cool-down. Scoring: leak-free (small_model_heldout.py --split-existing --deleted-cards
neighbour), w8+s8, the 732 set: the rule pool is the 722 users that the big model (1 leak-free epoch) rows have.
Review-weighted benchmark LogLoss; paired user bootstrap 2,000 resamples, seed 0, 95% CI.

Rule (DESIGN.md section 4, fixed 2026-11-02 before launch; DESIGN-ADDENDUM.md):
  stop          the paired delta (arm - reference, point estimate) > +0.00020 curve OR > +0.00015 instant
  go to finish  otherwise (the finish -- more epochs + the full cool-down -- is decided by the main session)

Also reported: each head vs the big model (1 leak-free epoch), the 725-user 732+long pool, the delta in points of %
of the big model, a projection "release + delta" (the released small model (SM72s 25 epochs + 1-epoch 32k Muon
cool-down) is +0.36% / +0.28% on the 722 users: an estimate, not part of the rule), and the bytes per card + note in
the fork's serialized layout (rslib/src/rwkv/precision.rs).

  python artifacts/small-model/sm73s2/sm73s2_rule.py decide [--rows FILE]   -> sm73s2-decision.{json,md}; exit 2 if users missing
  python artifacts/small-model/sm73s2/sm73s2_rule.py bytes                  -> the bytes table only
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

ROWS = HERE / "gpu" / "sm73s2-neighbour.jsonl"
ARM_LABEL = "sm73s2-arm-e21-q32k/w8+s8"
REF = (Path("C:/Users/jonat/workspace/wt/sm77/artifacts/small-model/sm77/gpu/sm77-neighbour.jsonl"),
       "sm77-ref-e21-q32k/w8+s8")  # read only
BIG = (Path("C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/gpu-big/bigleak-neighbour.jsonl"),
       "big-leakfree-1ep/w8+s8")  # read only
S72 = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s")  # read only
U732, ULONG = S72 / "eval-users-732-fit.json", S72 / "users-732-long.json"
HEADS = {"ahead": "curve", "instant": "instant"}
STOP = {"curve": 0.00020, "instant": 0.00015}          # raw LogLoss, arm - reference
RELEASE_VS_BIG = {"curve": 0.36, "instant": 0.28}       # % vs big, 722 users (released small model, main session 11-02)
BAR = {"curve": 0.50, "instant": 0.40}                  # % vs big, the smaller model's (SM73) bar (user 10-29)


def fork_bytes(layers, precision):
    """Serialized bytes of one entity's state (rslib/src/rwkv/precision.rs ModuleState::compact): magic (14) + precision
    byte + u32 layer count; per layer u32 + f16 time shift (64), u32 + matrix rows (64 rows of a 4-byte f32 scale + 16
    int8 / 8 int4-pair bytes), u32 + f16 channel shift (64, only for layers with a channel mixer)."""
    row = 4 + (16 if precision == "int8" else 8)
    total = 14 + 1 + 4
    for has_channel_mixer in layers:
        total += 4 + 64 * 2 + 4 + 64 * row + 4 + (64 * 2 if has_channel_mixer else 0)
    return total


def bytes_table():
    # channel mixers: w64 card layer 0 yes, card layer 1 stripped, note layer 0 yes (rslib architecture.rs)
    out = {}
    for name, card in (("w64 (2 card layers)", [True, False]), ("w64-c1 (1 card layer)", [True])):
        out[name] = {p: {"card": fork_bytes(card, p), "note": fork_bytes([True], p),
                         "card_and_note": fork_bytes(card, p) + fork_bytes([True], p)} for p in ("int8", "int4")}
    assert out["w64 (2 card layers)"]["int8"]["card_and_note"] == 4554   # the published small-model numbers
    assert out["w64 (2 card layers)"]["int4"]["card_and_note"] == 3018
    return out


def load(source, label, users):
    return {u: r for u, r in _load(source, label, users).items() if u in users}


def compare(new, ref, big, users):
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
        out[h] = {"arm": round(float((a * w).sum() / w.sum()), 6), "reference": round(float((b * w).sum() / w.sum()), 6),
                  "big": round(float(big_loss / w.sum()), 6), "delta": round(float(((a - b) * w).sum() / w.sum()), 6),
                  "delta_ci": [round(float(np.percentile(boot_raw, q)), 6) for q in (2.5, 97.5)],
                  "points": round(float(((a - b) * w).sum() / big_loss * 100), 3),
                  "points_ci": [round(float(np.percentile(boot, q)), 3) for q in (2.5, 97.5)],
                  "users_better": int((a < b).sum()),
                  "arm_vs_big_pct": round(float(((a - c) * w).sum() / big_loss * 100), 3),
                  "reference_vs_big_pct": round(float(((b - c) * w).sum() / big_loss * 100), 3)}
    return out


def verdict(res):
    over = [f"{h} {res[h]['delta']:+.6f} > +{STOP[h]:.5f}" for h in HEADS.values() if res[h]["delta"] > STOP[h]]
    if over:
        return "stop", "; ".join(over)
    return "go to finish", ", ".join(f"{h} {res[h]['delta']:+.6f} <= +{STOP[h]:.5f}" for h in HEADS.values())


def first_row(rows, label):
    for line in Path(rows).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("label") == label and "parameters" in r:
            return r
    return {}


def decide(args):
    users732 = set(json.loads(U732.read_text())["users"])
    long_users = set(json.loads(ULONG.read_text())["users"])
    big = load(*BIG, users732 | long_users)
    users = sorted(users732 & set(big))
    rows = Path(args.rows)
    arm = load(rows, args.arm_label, set(big)) if rows.exists() else {}
    ref = load(*REF, set(big))
    missing = [u for u in users if u not in arm or u not in ref]
    if missing:
        print(f"[sm73s2-rule] {len(users) - len(missing)} of {len(users)} users scored in both arms (missing {missing[:10]})")
        return 2
    res = {"rule_pool": compare(arm, ref, big, users)}
    both = sorted(set(big) & set(arm) & set(ref))
    if len(both) > len(users):
        res["732+long"] = compare(arm, ref, big, both)
    res["verdict"], res["why"] = verdict(res["rule_pool"])
    p = res["rule_pool"]
    res["projection"] = {h: {"release_vs_big_pct": RELEASE_VS_BIG[h], "projected_vs_big_pct": round(RELEASE_VS_BIG[h] + p[h]["points"], 3),
                             "bar_pct": BAR[h]} for h in HEADS.values()}
    a, r = first_row(rows, args.arm_label), first_row(REF[0], REF[1])
    res["size"] = {"parameters_arm": a.get("parameters"), "parameters_reference": r.get("parameters"),
                   "weight_bytes_arm": a.get("weight_bytes"), "weight_bytes_reference": r.get("weight_bytes"),
                   "fork_state_bytes": bytes_table()}
    res.update(rows=str(rows), arm_label=args.arm_label, reference=f"{REF[0]} {REF[1]}", big=f"{BIG[0]} {BIG[1]}",
               rule={"stop_if_delta_above": STOP, "unit": "raw review-weighted LogLoss, arm - reference, point estimate"})
    (HERE / "sm73s2-decision.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    md = ["# Smaller model, card stream 1 layer (SM73 stage 2) early check: 1 card layer vs 2, paired cool-downs", "",
          "Leak-free scoring, w8+s8, review-weighted LogLoss, user bootstrap 2,000 seed 0, 95% CI. Arm: the leak-free small "
          "model (SM72s) epoch 20 cut to 1 card layer + 1 constant-LR epoch + quarter-epoch 32k Muon cool-down. Reference: "
          "SM72s epoch 21 + the same cool-down. Points = % of the big model (1 leak-free epoch).", "",
          "| pool | users | head | arm | reference | delta [95% CI] | points [95% CI] | users better | arm vs big | reference vs big |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for pool in ("rule_pool", "732+long"):
        if pool in res:
            for h in HEADS.values():
                x = res[pool][h]
                md.append(f"| {'732 (rule)' if pool == 'rule_pool' else pool} | {res[pool]['users']} | {h} | {x['arm']:.5f} | "
                          f"{x['reference']:.5f} | {x['delta']:+.6f} [{x['delta_ci'][0]:+.6f}, {x['delta_ci'][1]:+.6f}] | "
                          f"{x['points']:+.2f} [{x['points_ci'][0]:+.2f}, {x['points_ci'][1]:+.2f}] | {x['users_better']} | "
                          f"{x['arm_vs_big_pct']:+.2f}% | {x['reference_vs_big_pct']:+.2f}% |")
    s, fb = res["size"], res["size"]["fork_state_bytes"]
    md += ["", f"Rule: stop if the delta exceeds +{STOP['curve']:.5f} curve or +{STOP['instant']:.5f} instant (point estimate).",
           "", f"Projection (estimate, not the rule): the released small model is +{RELEASE_VS_BIG['curve']:.2f}% / "
           f"+{RELEASE_VS_BIG['instant']:.2f}% vs the big model (1 leak-free epoch); + this delta = "
           f"{res['projection']['curve']['projected_vs_big_pct']:+.2f}% curve / {res['projection']['instant']['projected_vs_big_pct']:+.2f}% "
           f"instant (bar +{BAR['curve']:.2f}% / +{BAR['instant']:.2f}%).", "",
           f"Size: {s['parameters_reference']:,} -> {s['parameters_arm']:,} parameters; scorer w8 weight bytes "
           f"{s['weight_bytes_reference']:,} -> {s['weight_bytes_arm']:,}. Bytes per card + note (fork layout): int8 "
           f"{fb['w64 (2 card layers)']['int8']['card_and_note']:,} -> {fb['w64-c1 (1 card layer)']['int8']['card_and_note']:,}, int4 "
           f"{fb['w64 (2 card layers)']['int4']['card_and_note']:,} -> {fb['w64-c1 (1 card layer)']['int4']['card_and_note']:,}.",
           "", f"Verdict: **{res['verdict']}** ({res['why']})."]
    (HERE / "sm73s2-decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[sm73s2-rule] {p['users']} users: curve {p['curve']['delta']:+.6f} ({p['curve']['points']:+.2f} points {p['curve']['points_ci']}) "
          f"instant {p['instant']['delta']:+.6f} ({p['instant']['points']:+.2f} points {p['instant']['points_ci']}) (arm - reference); "
          f"arm vs big {p['curve']['arm_vs_big_pct']:+.2f}% / {p['instant']['arm_vs_big_pct']:+.2f}%: {res['verdict'].upper()} ({res['why']})")
    return 0


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("decide", "bytes"))
    parser.add_argument("--rows", type=Path, default=ROWS)
    parser.add_argument("--arm-label", default=ARM_LABEL)
    args = parser.parse_args()
    if args.command == "bytes":
        print(json.dumps(bytes_table()))
        raise SystemExit(0)
    raise SystemExit(decide(args))
