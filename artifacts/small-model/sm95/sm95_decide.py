"""The grade order by construction (SM95) stage 1 decision (CPU, about a minute). Pre-registered 2026-11-07 before any
GPU run (artifacts/small-model/sm95/HANDOFF.md).

Arm = the leak-free small model (SM72s) epoch 21 + the quarter-epoch 32k Muon cool-down with "grade_rectify" (density
0.08): on a seeded 8% of the labelled reviews the curve is trained on its grade's value in the rectified (isotonic)
4-vector of the four what-if curves. Its scored probability is the rectified one (scorer --grade-rectify: three what-if
rows before every labelled review, the four curves pooled at the label time), because that is the curve the scheduler
uses: the four buttons are the four what-if curves pooled to be ordered, i.e. exactly what the fork's PAVA computes
today. Reference = the same cool-down without the option (the tiny architecture additions (SM77) reference), scored the
usual way (raw) and, because the rectifier is defined for any model, also rectified. Third comparator: the grade-order
sanity check (SM33) hinge arm (raw).

Accuracy: leak-free (w8+s8) review-weighted benchmark LogLoss on the 722 users of the 732 set that the big model (1
leak-free epoch) rows use (minus users missing in any of the four arm / reference labels; >= 700 required), points of %
of the big model's LogLoss (curve 0.298966, instant 0.275757 on 722 users), paired user bootstrap 2,000, seed 0
(scripts/sm77_rule.compare). Order: scripts/sm33_grade_order.py measure folders (the same 3,000 sampled reviews per
user), re-derived by scripts/sm95_rectify.py derive (raw or rectified curves).

  order ok   the arm's rectified curves: zero inverted pairs on every sampled review, raw (unrounded) and displayed
             (ceil, 1..36,500 d) at desired retention 0.80, 0.90 and 0.95, and the fork's PAVA changes no displayed
             interval at 0.90 (by construction; anything else is a bug)
  go         order ok AND both heads (arm rectified - reference raw) <= +0.05 points (point estimates; CIs reported)
  cost       order ok but a head > +0.05 points
  fail       order not ok
  incomplete fewer than 700 users in all four labels, or a measure folder missing

  python artifacts/small-model/sm95/sm95_decide.py --rows F --ref-rows F [--sm33-rows F]
         --arm-measure DIR --ref-measure DIR [--sm33-measure DIR] --derived DIR
         -> artifacts/small-model/sm95/sm95-decision.{json,md}; one line on stdout
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

MAX_POINTS = 0.05
MIN_USERS = 700
PAIRS = ((0, 1, "Again>Hard"), (1, 2, "Hard>Good"), (2, 3, "Good>Easy"))
TARGETS = ("0.80", "0.90", "0.95")
LABELS = {"arm_rect": "sm95-arm-e21-q32k-rect/w8+s8", "arm_raw": "sm95-arm-e21-q32k/w8+s8",
          "ref_rect": "sm77-ref-e21-q32k-rect/w8+s8", "ref_raw": "sm77-ref-e21-q32k/w8+s8",
          "sm33_raw": "sm33-arm-e21-q32k/w8+s8"}


def displayed(days):
    return np.clip(np.ceil(np.asarray(days, dtype=np.float64) - 1e-9), 1, 36500)


def derive(measure, out, rectify):
    from scripts.sm95_rectify import cmd_derive
    cmd_derive(argparse.Namespace(measure=str(measure), out=str(out), rectify=rectify))
    return out


def breaks(folder):
    res = {}
    for f in sorted((Path(folder) / "users").glob("*.npz")):
        z = np.load(f)
        iv = z["intervals"].astype(np.float64)               # [n, 3 targets, 4]
        row = {"n_reviews": int(z["n_reviews"]), "review_th": z["review_th"]}
        for ti, t in enumerate(TARGETS):
            raw, d = iv[:, ti, :], displayed(iv[:, ti, :])
            row[f"raw_any@{t}"] = np.any(np.diff(raw, axis=1) < 0, axis=1)
            row[f"displayed_any@{t}"] = np.any(np.diff(d, axis=1) < 0, axis=1)
            if t == "0.90":
                for a, b, name in PAIRS:
                    row[f"{name}@0.90"] = d[:, a] > d[:, b]
        fk, fk0 = displayed(z["fork_pava_09"]), displayed(z["fork_nopava_09"])
        row["good_changed_by_fork_pava@0.90"] = fk[:, 2] != fk0[:, 2]
        row["any_changed_by_fork_pava@0.90"] = np.any(fk != fk0, axis=1)
        res[int(f.stem)] = row
    return res


def rate(b, users, key):
    a = np.array([b[u][key].mean() for u in users])
    w = np.array([b[u]["n_reviews"] for u in users], dtype=np.float64)
    return a, w


def paired(x, y, key, users):
    a, w = rate(x, users, key)
    b, _ = rate(y, users, key)
    idx = np.random.default_rng(0).integers(0, len(users), size=(2000, len(users)))
    boot = ((a - b)[idx] * w[idx]).sum(1) / w[idx].sum(1)
    return {"a": float((a * w).sum() / w.sum()), "b": float((b * w).sum() / w.sum()), "delta": float(((a - b) * w).sum() / w.sum()),
            "ci": [float(np.percentile(boot, q)) for q in (2.5, 97.5)], "users_better": int((a < b).sum()),
            "users_worse": int((a > b).sum())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", required=True, help="sm95 scoring rows (arm raw / rectified, reference rectified)")
    p.add_argument("--ref-rows", required=True, help="the reference's raw scoring rows (SM77)")
    p.add_argument("--sm33-rows", default="")
    p.add_argument("--arm-measure", required=True)
    p.add_argument("--ref-measure", required=True)
    p.add_argument("--sm33-measure", default="")
    p.add_argument("--derived", required=True)
    args = p.parse_args()
    users732 = set(json.loads(U732.read_text())["users"])
    long_users = set(json.loads(ULONG.read_text())["users"])
    big = load(*BIG, users732 | long_users)
    base_users = sorted(users732 & set(big))
    src = {"arm_rect": args.rows, "arm_raw": args.rows, "ref_rect": args.rows, "ref_raw": args.ref_rows,
           **({"sm33_raw": args.sm33_rows} if args.sm33_rows else {})}
    rows = {k: load(Path(f), LABELS[k], set(big)) for k, f in src.items()}
    users = [u for u in base_users if all(u in rows[k] for k in ("arm_rect", "arm_raw", "ref_rect", "ref_raw"))]
    missing = [u for u in base_users if u not in users]
    derived = Path(args.derived)
    meas = {"arm_rect": (args.arm_measure, True), "arm_raw": (args.arm_measure, False),
            "ref_rect": (args.ref_measure, True), "ref_raw": (args.ref_measure, False),
            **({"sm33_raw": (args.sm33_measure, False)} if args.sm33_measure else {})}
    if len(users) < MIN_USERS or any(not (Path(m) / "users").is_dir() for m, _ in meas.values()):
        print(f"[sm95-rule] INCOMPLETE: {len(users)} of {len(base_users)} users in all four labels (missing {missing[:10]}); "
              f"measures {[m for m, _ in meas.values()]}")
        return 2
    acc = {"primary: arm rectified vs reference raw": compare(rows["arm_rect"], rows["ref_raw"], big, users),
           "arm rectified vs reference rectified (training effect, both scored as the scheduler uses them)":
               compare(rows["arm_rect"], rows["ref_rect"], big, users),
           "reference rectified vs reference raw (rectifying today's model without training)":
               compare(rows["ref_rect"], rows["ref_raw"], big, users),
           "arm raw vs reference raw (the arm's unpooled curves)": compare(rows["arm_raw"], rows["ref_raw"], big, users)}
    if "sm33_raw" in rows and all(u in rows["sm33_raw"] for u in users):
        acc["grade-order hinge (SM33) arm vs reference raw"] = compare(rows["sm33_raw"], rows["ref_raw"], big, users)
    br = {k: breaks(derive(m, derived / k, r)) for k, (m, r) in meas.items()}
    bu = sorted(set.intersection(*(set(b) for b in br.values())))
    for u in bu:
        for k in br:
            if not np.array_equal(br[k][u]["review_th"], br["ref_raw"][u]["review_th"]):
                raise SystemExit(f"user {u}: measure {k} sampled other reviews than the reference's")
    keys = [f"{kind}_any@{t}" for t in TARGETS for kind in ("raw", "displayed")] + [f"{n}@0.90" for _, _, n in PAIRS] + \
           ["good_changed_by_fork_pava@0.90", "any_changed_by_fork_pava@0.90"]
    order = {k: {key: float(sum(br[k][u][key].sum() for u in bu)) for key in keys} for k in br}           # counts
    shares = {k: {key: float(rate(br[k], bu, key)[0].dot(rate(br[k], bu, key)[1]) / rate(br[k], bu, key)[1].sum()) for key in keys}
              for k in br}                                                                               # review-weighted
    order_vs_ref = {key: paired(br["arm_rect"], br["ref_raw"], key, bu) for key in keys}
    sampled = int(sum(len(br["arm_rect"][u]["review_th"]) for u in bu))
    order_ok = all(order["arm_rect"][key] == 0 for key in keys)
    prim = acc["primary: arm rectified vs reference raw"]
    heads_ok = all(prim[h]["points"] <= MAX_POINTS for h in ("curve", "instant"))
    verdict = ("go" if heads_ok else "cost") if order_ok else "fail"
    res = {"verdict": verdict, "users": len(users), "missing_users": missing, "order_users": len(bu), "sampled_reviews": sampled,
           "accuracy": acc, "order_break_counts": order, "order_shares_review_weighted": shares,
           "order_arm_rect_vs_ref_raw": order_vs_ref, "labels": {k: LABELS[k] for k in src}, "rows": src,
           "measures": {k: m for k, (m, _) in meas.items()},
           "rule": {"max_points": MAX_POINTS, "min_users": MIN_USERS, "order": "zero breaks of the arm's rectified curves",
                    "primary": "arm rectified vs reference raw"}, "big": f"{BIG[0]} {BIG[1]}"}
    (HERE / "sm95-decision.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    md = ["# The grade order by construction (SM95) stage 1: rectified cool-down vs the same cool-down without it", "",
          f"Accuracy: leak-free scoring, w8+s8, review-weighted LogLoss on {len(users)} users, points of % of the big model "
          "(1 leak-free epoch), paired user bootstrap 95% CI. \"Rectified\" = the curve of each review pooled with the other "
          "three grades' what-if curves at the same time so that Again <= Hard <= Good <= Easy (the scheduler's curve).", "",
          "| comparison | head | a | b | delta | points [95% CI] | users better |", "|---|---|---|---|---|---|---|"]
    for name, x in acc.items():
        for h in ("curve", "instant"):
            y = x[h]
            md.append(f"| {name} | {h} | {y['arm']:.5f} | {y['reference']:.5f} | {y['delta']:+.6f} | {y['points']:+.2f} "
                      f"[{y['ci'][0]:+.2f}, {y['ci'][1]:+.2f}] | {y['users_better']} |")
    md += ["", f"Order: {len(bu)} users x 3,000 sampled reviews ({sampled:,} reviews, the same in every measure); share of "
           "reviews with an inverted pair (review-weighted) and, for the arm, the count.", "",
           "| measure | arm rectified (count) | arm raw | reference rectified | reference raw | grade-order hinge (SM33) raw |",
           "|---|---|---|---|---|---|"]
    for key in keys:
        md.append(f"| {key} | {shares['arm_rect'][key]:.2%} ({int(order['arm_rect'][key])}) | {shares['arm_raw'][key]:.2%} | "
                  f"{shares['ref_rect'][key]:.2%} | {shares['ref_raw'][key]:.2%} | "
                  f"{(shares['sm33_raw'][key] if 'sm33_raw' in shares else float('nan')):.2%} |")
    md += ["", f"Verdict: **{verdict}** (go = zero breaks and both heads <= +{MAX_POINTS} points vs the reference raw)."]
    (HERE / "sm95-decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[sm95-rule] {len(users)} users: curve {prim['curve']['points']:+.2f} {prim['curve']['ci']} instant "
          f"{prim['instant']['points']:+.2f} {prim['instant']['ci']} points (arm rectified vs reference raw); arm breaks "
          f"{int(sum(order['arm_rect'].values()))} on {sampled:,} sampled reviews (reference raw any pair displayed at 0.90: "
          f"{shares['ref_raw']['displayed_any@0.90']:.1%}): {verdict.upper()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
