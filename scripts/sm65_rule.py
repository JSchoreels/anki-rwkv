#!/usr/bin/env python
"""Better teacher (SM65) stage 1 decision (CPU, seconds): the big model after one more leak-free epoch on the 3,969 new
users (label big-moreusers/w8+s8) vs the big model (1 leak-free epoch) (SM72 step 1), both scored leak-free
(scripts/small_model_heldout.py --split-existing --deleted-cards neighbour), on the 722 users of the 732 set that the
reference rows use (reference 0.29897 / 0.27576). Review-weighted benchmark LogLoss over all test reviews, % vs the
reference, paired user bootstrap 2,000 resamples, seed 0, 95% CI. Pre-registered rule (artifacts/small-model/sm65/DESIGN.md):

  go    curve <= -0.20% or instant <= -0.15%, with that head's CI clear of zero and the other head's point <= +0.03%
  stop  both heads gain less than 0.10 points (both > -0.10%)
  grey  otherwise: report and ask

  decide [--rows FILE]   writes artifacts/small-model/sm65/sm65-decision.{json,md}; prints one line; exit 2 if users
                         are missing (the 732+long pool, 725 users, is reported too, not used by the rule)
"""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sm71_report import load as _load  # noqa: E402

HERE = ROOT / "artifacts/small-model/sm65"
ROWS = HERE / "gpu" / "bigmore-neighbour.jsonl"
LABEL = "big-moreusers/w8+s8"
REF = (Path("C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/gpu-big/bigleak-neighbour.jsonl"),
       "big-leakfree-1ep/w8+s8")  # read only
S56 = Path("C:/Users/jonat/workspace/wt/sm56/artifacts/small-model/sm56")  # read only
U732, ULONG = S56 / "eval-users-732-fit.json", S56 / "smalljobs/users-732-retry3.json"
HEADS = {"ahead": "curve", "instant": "instant"}
GO = {"curve": -0.20, "instant": -0.15}
OTHER_MAX, STOP = 0.03, -0.10


def load(source, label, users):
    return {u: r for u, r in _load(source, label, users).items() if u in users}


def compare(new, ref, users):
    idx = np.random.default_rng(0).integers(0, len(users), size=(2000, len(users)))
    out = {"users": len(users)}
    for head, h in HEADS.items():
        a = np.array([new[u]["all"][head][0] for u in users]); b = np.array([ref[u]["all"][head][0] for u in users])
        w = np.array([ref[u]["all"][head][1] for u in users], float)
        a, b = np.where(w > 0, a, 0.0), np.where(w > 0, b, 0.0)
        boot = ((a - b)[idx] * w[idx]).sum(1) / (b[idx] * w[idx]).sum(1) * 100
        out[h] = {"new": round(float((a * w).sum() / w.sum()), 6), "reference": round(float((b * w).sum() / w.sum()), 6),
                  "delta": round(float(((a - b) * w).sum() / w.sum()), 6),
                  "pct": round(float(((a - b) * w).sum() / (b * w).sum() * 100), 3),
                  "ci": [round(float(np.percentile(boot, q)), 3) for q in (2.5, 97.5)], "users_better": int((a < b).sum())}
    return out


def verdict(res):
    for h, other in (("curve", "instant"), ("instant", "curve")):
        if res[h]["pct"] <= GO[h] and res[h]["ci"][1] < 0 and res[other]["pct"] <= OTHER_MAX:
            return "go", f"{h} {res[h]['pct']:+.2f}% (CI clear of zero), {other} {res[other]['pct']:+.2f}%"
    if all(res[h]["pct"] > STOP for h in HEADS.values()):
        return "stop", "both heads gain under 0.10 points"
    return "grey", "between the go and stop bars: report and ask"


def decide(rows):
    users732 = set(json.loads(U732.read_text())["users"])
    ref = load(*REF, users732 | set(json.loads(ULONG.read_text())["users"]))
    users = sorted(users732 & set(ref))
    new = load(rows, LABEL, set(ref)) if Path(rows).exists() else {}
    missing = [u for u in users if u not in new]
    if missing:
        print(f"[sm65-rule] {LABEL}: {len(users) - len(missing)} of {len(users)} users scored (missing {missing[:10]})")
        return 2
    res = {"rule_pool": compare(new, ref, users)}
    both = sorted(set(ref) & set(new))
    if len(both) > len(users):
        res["732+long"] = compare(new, ref, both)
    res["verdict"], res["why"] = verdict(res["rule_pool"])
    res.update(rows=str(rows), label=LABEL, reference=f"{REF[0]} {REF[1]}", rule={"go": GO, "other_max": OTHER_MAX, "stop": STOP})
    (HERE / "sm65-decision.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    md = ["# Better teacher (SM65) stage 1: big model + 1 leak-free epoch on 3,969 new users vs the big model (1 leak-free epoch)",
          "", "Leak-free scoring, w8+s8, review-weighted LogLoss, user bootstrap 2,000 seed 0, 95% CI.", "",
          "| pool | users | head | new | reference | delta | % vs reference [95% CI] | users better |", "|---|---|---|---|---|---|---|---|"]
    for pool in ("rule_pool", "732+long"):
        if pool in res:
            for h in HEADS.values():
                r = res[pool][h]
                md.append(f"| {'732 (rule)' if pool == 'rule_pool' else pool} | {res[pool]['users']} | {h} | {r['new']:.5f} | "
                          f"{r['reference']:.5f} | {r['delta']:+.6f} | {r['pct']:+.2f}% [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] | {r['users_better']} |")
    md += ["", f"Verdict: **{res['verdict']}** ({res['why']})."]
    (HERE / "sm65-decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    p = res["rule_pool"]
    print(f"[sm65-rule] {p['users']} users: curve {p['curve']['pct']:+.2f}% {p['curve']['ci']} instant {p['instant']['pct']:+.2f}% "
          f"{p['instant']['ci']} vs the big model (1 leak-free epoch): {res['verdict'].upper()} ({res['why']})")
    return 0


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("decide",))
    parser.add_argument("--rows", type=Path, default=ROWS)
    args = parser.parse_args()
    raise SystemExit(decide(args.rows))
