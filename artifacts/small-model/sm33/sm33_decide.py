"""Grade-order sanity (SM33) stage 2 decision (CPU, seconds). Pre-registered before the cool-down ran.

Arm = the leak-free small model (SM72s) epoch 21 + the quarter-epoch 32k Muon cool-down with the grade-order hinge;
reference = the same cool-down without it (the tiny architecture additions (SM77) reference). Accuracy: both scored
leak-free (w8+s8) on the 722 users of the 732 set that the big model (1 leak-free epoch) rows use; review-weighted
benchmark LogLoss, deltas in points of % of the big model (scripts/sm77_rule.compare: paired user bootstrap 2,000,
seed 0, 95% CI). Order: scripts/sm33_grade_order.py measure folders of both (the same 3,000 sampled reviews per user),
displayed intervals at 0.90 before the fork's PAVA (ceil, 1..36,500 days); share of reviews with any inverted pair,
review-weighted (each sampled review stands for n_reviews / n_sampled), paired user bootstrap 2,000, seed 0.

  go         both heads <= +0.03 points and the break share drops with its CI clear of zero
  cost       the break share drops (CI clear of zero) but a head is > +0.03 points
  no effect  the break share does not drop clearly

  python artifacts/small-model/sm33/sm33_decide.py --arm-rows F --arm-label L --ref-rows F --ref-label L
         --arm-breaks DIR --ref-breaks DIR      -> artifacts/small-model/sm33/sm33-decision.{json,md}
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from scripts.sm77_rule import BIG, U732, ULONG, compare, load  # noqa: E402

OTHER_MAX = 0.03
PAIRS = ((0, 1, "Again>Hard"), (1, 2, "Hard>Good"), (2, 3, "Good>Easy"))


def displayed(days):
    return np.clip(np.ceil(np.asarray(days, dtype=np.float64) - 1e-9), 1, 36500)


def breaks(folder):
    out = {}
    for f in sorted((Path(folder) / "users").glob("*.npz")):
        z = np.load(f)
        d = displayed(z["intervals"][:, 1, :])
        raw = z["intervals"][:, 1, :].astype(np.float64)
        fk = displayed(z["fork_pava_09"])
        out[int(f.stem)] = {"n_reviews": int(z["n_reviews"]), "review_th": z["review_th"],
                            "any": np.any(np.diff(d, axis=1) < 0, axis=1),
                            **{name: d[:, a] > d[:, b] for a, b, name in PAIRS},
                            "good_pava_changed": fk[:, 2] != d[:, 2],
                            "raw_any": np.any(np.diff(raw, axis=1) < 0, axis=1)}
    return out


def paired(arm, ref, key, users):
    a = np.array([arm[u][key].mean() for u in users])
    b = np.array([ref[u][key].mean() for u in users])
    w = np.array([arm[u]["n_reviews"] for u in users], dtype=np.float64)
    idx = np.random.default_rng(0).integers(0, len(users), size=(2000, len(users)))
    boot = ((a - b)[idx] * w[idx]).sum(1) / w[idx].sum(1)
    return {"arm": float((a * w).sum() / w.sum()), "reference": float((b * w).sum() / w.sum()),
            "delta": float(((a - b) * w).sum() / w.sum()), "ci": [float(np.percentile(boot, q)) for q in (2.5, 97.5)],
            "users_better": int((a < b).sum()), "users_worse": int((a > b).sum())}


def main():
    p = argparse.ArgumentParser()
    for k in ("--arm-rows", "--arm-label", "--ref-rows", "--ref-label", "--arm-breaks", "--ref-breaks"):
        p.add_argument(k, required=True)
    args = p.parse_args()
    users732 = set(json.loads(U732.read_text())["users"])
    long_users = set(json.loads(ULONG.read_text())["users"])
    big = load(*BIG, users732 | long_users)
    users = sorted(users732 & set(big))
    arm = load(Path(args.arm_rows), args.arm_label, set(big))
    ref = load(Path(args.ref_rows), args.ref_label, set(big))
    missing = [u for u in users if u not in arm or u not in ref]
    if missing:
        print(f"[sm33-rule] {len(users) - len(missing)} of {len(users)} users scored in both arms (missing {missing[:10]})")
        return 2
    acc = compare(arm, ref, big, users)
    ba, br = breaks(args.arm_breaks), breaks(args.ref_breaks)
    bu = sorted(set(ba) & set(br))
    for u in bu:
        if not np.array_equal(ba[u]["review_th"], br[u]["review_th"]):
            raise SystemExit(f"user {u}: the two measures sampled different reviews")
    order = {k: paired(ba, br, k, bu) for k in ("any", "Again>Hard", "Hard>Good", "Good>Easy", "good_pava_changed", "raw_any")}
    drop = order["any"]["ci"][1] < 0
    ok_acc = all(acc[h]["points"] <= OTHER_MAX for h in ("curve", "instant"))
    verdict = "go" if drop and ok_acc else ("cost" if drop else "no effect")
    res = {"accuracy_722": acc, "order_0.90_displayed": order, "order_users": len(bu), "verdict": verdict,
           "rule": {"other_max_points": OTHER_MAX, "drop": "review-weighted any-pair break share, CI upper < 0"},
           "arm_label": args.arm_label, "ref_label": args.ref_label, "arm_rows": args.arm_rows, "ref_rows": args.ref_rows,
           "arm_breaks": args.arm_breaks, "ref_breaks": args.ref_breaks, "big": f"{BIG[0]} {BIG[1]}"}
    (HERE / "sm33-decision.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    md = ["# Grade-order sanity (SM33) stage 2: the grade-order hinge vs the same cool-down without it", "",
          "Accuracy: leak-free scoring, w8+s8, review-weighted LogLoss on 722 users, points of % of the big model (1 leak-free "
          "epoch), user bootstrap 95% CI. Order: displayed intervals at desired retention 0.90 before the fork's PAVA, share of "
          f"reviews with an inverted pair, review-weighted, {len(bu)} users x 3,000 sampled reviews (the same reviews in both).", "",
          "| head | arm | reference | delta | points [95% CI] | users better |", "|---|---|---|---|---|---|"]
    for h in ("curve", "instant"):
        x = acc[h]
        md.append(f"| {h} | {x['arm']:.5f} | {x['reference']:.5f} | {x['delta']:+.6f} | {x['points']:+.2f} [{x['ci'][0]:+.2f}, "
                  f"{x['ci'][1]:+.2f}] | {x['users_better']} |")
    md += ["", "| order measure (0.90) | arm | reference | delta [95% CI] | users better / worse |", "|---|---|---|---|---|"]
    for k, x in order.items():
        md.append(f"| {k} | {x['arm']:.2%} | {x['reference']:.2%} | {x['delta']:+.2%} [{x['ci'][0]:+.2%}, {x['ci'][1]:+.2%}] | "
                  f"{x['users_better']} / {x['users_worse']} |")
    md += ["", f"Verdict: **{verdict}**."]
    (HERE / "sm33-decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[sm33-rule] 722 users: curve {acc['curve']['points']:+.2f} {acc['curve']['ci']} instant {acc['instant']['points']:+.2f} "
          f"{acc['instant']['ci']} points; break share {order['any']['reference']:.1%} -> {order['any']['arm']:.1%} "
          f"(CI {order['any']['ci'][0]:+.2%}..{order['any']['ci'][1]:+.2%}): {verdict.upper()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
