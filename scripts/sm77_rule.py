#!/usr/bin/env python
"""Tiny architecture additions (SM77) stage 1 decision (CPU, seconds; artifacts/small-model/sm77/DESIGN.md on main).

Arm = the leak-free small model (SM72s) from epoch K + one constant-LR epoch + a quarter-epoch 32k Muon cool-down with
the additions on (w64-sm77: key extras (A) + learned card / note initial state (B)); reference = the same with the
additions off. Both scored leak-free (small_model_heldout.py --split-existing --deleted-cards neighbour, w8+s8) on the
722 users of the 732 set that the big model (1 leak-free epoch) rows use. Review-weighted benchmark LogLoss over all
test reviews; deltas in points of % of the big model (1 leak-free epoch) on the same users (0.20 points ~ 0.00060
curve, 0.15 ~ 0.00041 instant); paired user bootstrap 2,000 resamples, seed 0, 95% CI. Pre-registered rule:

  go    curve <= -0.20 points or instant <= -0.15 points, that head's CI clear of zero, other head's point <= +0.03
  stop  both heads gain less than 0.10 points (both > -0.10)
  grey  otherwise: report and ask

  decide --arm-label L --ref-label L [--rows FILE]   writes artifacts/small-model/sm77/sm77-decision.{json,md}; prints
         one line; exit 2 if users are missing. Also reports the 725-user 732+long pool, the parameter and export-size
         change (the scorer's w8 weight bytes) and both arms vs the big model (1 leak-free epoch).
"""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sm71_report import load as _load  # noqa: E402

HERE = ROOT / "artifacts/small-model/sm77"
ROWS = HERE / "gpu" / "sm77-neighbour.jsonl"
BIG = (Path("C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/gpu-big/bigleak-neighbour.jsonl"),
       "big-leakfree-1ep/w8+s8")  # read only
S72 = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s")  # read only
U732, ULONG = S72 / "eval-users-732-fit.json", S72 / "users-732-long.json"
HEADS = {"ahead": "curve", "instant": "instant"}
GO = {"curve": -0.20, "instant": -0.15}
OTHER_MAX, STOP = 0.03, -0.10


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
        boot = ((a - b)[idx] * w[idx]).sum(1) / (c[idx] * w[idx]).sum(1) * 100
        big_loss = (c * w).sum()
        out[h] = {"arm": round(float((a * w).sum() / w.sum()), 6), "reference": round(float((b * w).sum() / w.sum()), 6),
                  "big": round(float(big_loss / w.sum()), 6), "delta": round(float(((a - b) * w).sum() / w.sum()), 6),
                  "points": round(float(((a - b) * w).sum() / big_loss * 100), 3),
                  "ci": [round(float(np.percentile(boot, q)), 3) for q in (2.5, 97.5)], "users_better": int((a < b).sum()),
                  "arm_vs_big_pct": round(float(((a - c) * w).sum() / big_loss * 100), 3),
                  "reference_vs_big_pct": round(float(((b - c) * w).sum() / big_loss * 100), 3)}
    return out


def verdict(res):
    for h, other in (("curve", "instant"), ("instant", "curve")):
        if res[h]["points"] <= GO[h] and res[h]["ci"][1] < 0 and res[other]["points"] <= OTHER_MAX:
            return "go", f"{h} {res[h]['points']:+.2f} points (CI clear of zero), {other} {res[other]['points']:+.2f}"
    if all(res[h]["points"] > STOP for h in HEADS.values()):
        return "stop", "both heads gain under 0.10 points"
    return "grey", "between the go and stop bars: report and ask"


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
    ref = load(rows, args.ref_label, set(big)) if rows.exists() else {}
    missing = [u for u in users if u not in arm or u not in ref]
    if missing:
        print(f"[sm77-rule] {len(users) - len(missing)} of {len(users)} users scored in both arms (missing {missing[:10]})")
        return 2
    res = {"rule_pool": compare(arm, ref, big, users)}
    both = sorted(set(big) & set(arm) & set(ref))
    if len(both) > len(users):
        res["732+long"] = compare(arm, ref, big, both)
    res["verdict"], res["why"] = verdict(res["rule_pool"])
    a, r = first_row(rows, args.arm_label), first_row(rows, args.ref_label)
    res["size"] = {"parameters_arm": a.get("parameters"), "parameters_reference": r.get("parameters"),
                   "weight_bytes_arm": a.get("weight_bytes"), "weight_bytes_reference": r.get("weight_bytes"),
                   "weight_bytes_change": (a.get("weight_bytes") or 0) - (r.get("weight_bytes") or 0),
                   "state_bytes_change": 0}
    res.update(rows=str(rows), arm_label=args.arm_label, ref_label=args.ref_label, big=f"{BIG[0]} {BIG[1]}",
               rule={"go": GO, "other_max": OTHER_MAX, "stop": STOP, "unit": "points of % of the big model (1 leak-free epoch)"})
    (HERE / "sm77-decision.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    md = ["# Tiny architecture additions (SM77) stage 1: key extras + learned initial state vs the same recipe without them",
          "", "Leak-free scoring, w8+s8, review-weighted LogLoss, user bootstrap 2,000 seed 0, 95% CI; deltas in points of % "
          "of the big model (1 leak-free epoch).", "",
          "| pool | users | head | arm | reference | delta | points [95% CI] | users better | arm vs big | reference vs big |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for pool in ("rule_pool", "732+long"):
        if pool in res:
            for h in HEADS.values():
                x = res[pool][h]
                md.append(f"| {'732 (rule)' if pool == 'rule_pool' else pool} | {res[pool]['users']} | {h} | {x['arm']:.5f} | "
                          f"{x['reference']:.5f} | {x['delta']:+.6f} | {x['points']:+.2f} [{x['ci'][0]:+.2f}, {x['ci'][1]:+.2f}] | "
                          f"{x['users_better']} | {x['arm_vs_big_pct']:+.2f}% | {x['reference_vs_big_pct']:+.2f}% |")
    s = res["size"]
    md += ["", f"Size: {s['parameters_reference']:,} -> {s['parameters_arm']:,} parameters; scorer w8 weight bytes "
           f"{s['weight_bytes_reference']:,} -> {s['weight_bytes_arm']:,} ({s['weight_bytes_change']:+,} B); card / note state "
           "bytes unchanged.", "", f"Verdict: **{res['verdict']}** ({res['why']})."]
    (HERE / "sm77-decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    p = res["rule_pool"]
    print(f"[sm77-rule] {p['users']} users: curve {p['curve']['points']:+.2f} points {p['curve']['ci']} instant "
          f"{p['instant']['points']:+.2f} points {p['instant']['ci']} (arm - reference): {res['verdict'].upper()} ({res['why']}); "
          f"weights {s['weight_bytes_change']:+,} B")
    return 0


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("decide",))
    parser.add_argument("--rows", type=Path, default=ROWS)
    parser.add_argument("--arm-label", required=True)
    parser.add_argument("--ref-label", required=True)
    raise SystemExit(decide(parser.parse_args()))
