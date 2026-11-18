#!/usr/bin/env python
"""Small-model epoch gate (docs/SMALL_MODEL.md section 5.6): go / stop after epoch K of a multi-epoch run, on CPU.

The check is the real run's own epoch: its checkpoint is scored on the fixed 100-user subset
(artifacts/small-model/gate-users.json, rows in heldout.jsonl under the run's normal label), next to the fork and to
tonight's reference run scale-w64kd-gate (epochs 1-3 and its 0.25-epoch decay) on the same users. Paired,
review-weighted benchmark LogLoss per head (curve, instant), w8+s8 labels.

Rule (both heads must pass every test; the thresholds are module constants and are written into the decision):
  A  matching point   Δ = cand_eK - ref_eK < 0, its 95% user-bootstrap upper bound < 0, and on at least one head
                      |Δ| >= BAND (0.0062 curve / 0.0027 instant: the k = 1 whole-training noise band, A15).
                      --same-recipe (more users/epochs of the reference recipe, which cannot beat itself): A becomes
                      no regression, Δ <= +BAND on both heads; B decides.
  B  projection       final ≈ cand_eK + b · (ln N - ln K) + (ref_dec - ref_e3), with b the log-epoch slope: the
                      reference's least-squares fit L = a + b ln e over its epochs 1-3 when K = 1; the candidate's own
                      (cand_e2 - cand_e1) / ln 2 when K = 2. Go only if final <= ref_dec - MARGIN · ref_dec
                      (MARGIN 1%: about 1 pp of the gap to the fork, the smallest step worth a multi-hour run).
  C  (K = 2 only)     not flattening: cand_e1 - cand_e2 >= 0.5 · (ref_e1 - ref_e2).
Missing rows (a user of the subset without a row for any label) -> verdict "stop" with reason "incomplete".

usage: small_model_gate.py --candidate s-w64kd-gate-e1/w8+s8 [--previous s-w64kd-gate-e1/w8+s8] --epoch K --epochs N
           --out artifacts/small-model/gate-<run>-e<K>.json [--jsonl ...] [--users-file ...]
       small_model_gate.py --self-test
exit 0 = go, 10 = stop, 2 = usage error.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HEADS = {"ahead": "curve", "instant": "instant"}
BAND = {"ahead": 0.0062, "instant": 0.0027}
MARGIN = 0.01
FLATTEN = 0.5
REFERENCE = {"fork": "fork/w8+s8", 1: "scale-w64kd-e1/w8+s8", 2: "scale-w64kd-e2/w8+s8", 3: "scale-w64kd-e3/w8+s8",
             "dec": "scale-w64kd-dec/w8+s8"}
GO, STOP = 0, 10


def load(path, labels, users):
    per = {label: {} for label in labels}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row.get("label") in per and row["user_id"] in users:
            bench = row.get("benchmark", {})
            if "ahead" in bench and "instant" in bench:
                per[row["label"]][row["user_id"]] = {h: (bench[h]["LogLoss"], bench[h]["size"]) for h in HEADS}
    return per


def weighted(per, label, users, head):
    values = np.array([per[label][u][head][0] for u in users])
    sizes = np.array([per[label][u][head][1] for u in users], dtype=float)
    return values, sizes


def rw(values, sizes):
    return float((values * sizes).sum() / sizes.sum())


def boot_upper(cand, ref, sizes, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sizes), size=(n, len(sizes)))
    w = sizes[idx]
    deltas = ((cand[idx] - ref[idx]) * w).sum(1) / w.sum(1)
    return float(np.percentile(deltas, 97.5))


def decide(per, users, candidate, previous, epoch, epochs, same_recipe=False):
    labels = [candidate] + ([previous] if previous else []) + list(REFERENCE.values())
    missing = {label: len([u for u in users if u not in per[label]]) for label in labels}
    if any(missing.values()):
        return STOP, {"verdict": "stop", "reason": "incomplete", "missing_users": missing}
    out = {"users": len(users), "candidate": candidate, "previous": previous, "epoch": epoch, "epochs": epochs,
           "rule": {"band": BAND, "margin_relative": MARGIN, "flatten_fraction": FLATTEN, "same_recipe": same_recipe,
                    "reference": REFERENCE}, "heads": {}}
    passed = ({"A_no_regression": True} if same_recipe else {"A_matching_point": True, "A_beyond_band_one_head": False})
    passed["B_projection"] = True
    if epoch == 2:
        passed["C_not_flattening"] = True
    for head, name in HEADS.items():
        cand, sizes = weighted(per, candidate, users, head)
        ref = {k: rw(*weighted(per, v, users, head)) for k, v in REFERENCE.items()}
        c = rw(cand, sizes)
        matching = weighted(per, REFERENCE[epoch], users, head)[0]
        delta = c - ref[epoch]
        upper = boot_upper(cand, matching, sizes)
        x = np.log([1.0, 2.0, 3.0])
        b_ref = float(np.polyfit(x, [ref[1], ref[2], ref[3]], 1)[0])
        h = {"cand": c, "reference": ref, "delta_matching": delta, "delta_upper95": upper,
             "delta_pct_vs_fork": (c / ref["fork"] - 1) * 100, "slope_reference": b_ref}
        if same_recipe:
            passed["A_no_regression"] &= delta <= BAND[head]
        else:
            passed["A_matching_point"] &= delta < 0 and upper < 0
            passed["A_beyond_band_one_head"] |= -delta >= BAND[head]
        slope = b_ref
        if epoch == 2:
            prev = rw(*weighted(per, previous, users, head))
            slope = (c - prev) / math.log(2)
            h.update({"cand_previous": prev, "slope_candidate": slope,
                      "improvement_candidate": prev - c, "improvement_reference": ref[1] - ref[2]})
            passed["C_not_flattening"] &= (prev - c) >= FLATTEN * (ref[1] - ref[2])
        projected = c + slope * (math.log(epochs) - math.log(epoch)) + (ref["dec"] - ref[3])
        target = ref["dec"] * (1 - MARGIN)
        h.update({"projected_final": projected, "projected_pct_vs_fork": (projected / ref["fork"] - 1) * 100,
                  "best_dec": ref["dec"], "best_dec_pct_vs_fork": (ref["dec"] / ref["fork"] - 1) * 100,
                  "go_target": target})
        passed["B_projection"] &= projected <= target
        out["heads"][name] = h
    out["passed"] = passed
    go = all(passed.values())
    out["verdict"] = "go" if go else "stop"
    out["reason"] = "all tests passed" if go else "failed: " + ", ".join(k for k, v in passed.items() if not v)
    return (GO if go else STOP), out


def self_test():
    rng = np.random.default_rng(1)
    users = list(range(1000, 1100))
    sizes = rng.integers(500, 50000, size=len(users))
    base = {"fork": 0.290, 1: 0.320, 2: 0.314, 3: 0.311, "dec": 0.307}

    def rows(level, noise=0.002):
        return {u: {h: (level + (0.02 if h == "ahead" else 0) + rng.normal(0, noise), int(s)) for h in HEADS}
                for u, s in zip(users, sizes)}
    per = {REFERENCE[k]: rows(v) for k, v in base.items()}
    cases = []
    per["good-e1"] = rows(0.305)       # 0.015 better than ref e1: should go
    code, d = decide(per, users, "good-e1", None, 1, 6)
    cases.append(("clearly better epoch 1 -> go", code == GO, d["reason"]))
    per["same-e1"] = rows(0.3195)      # inside the band: stop
    code, d = decide(per, users, "same-e1", None, 1, 6)
    cases.append(("epoch 1 inside the band -> stop", code == STOP and not d["passed"]["A_beyond_band_one_head"], d["reason"]))
    per["good-e2"] = rows(0.3049)      # e2 barely better than e1: flattening -> stop
    code, d = decide(per, users, "good-e2", "good-e1", 2, 6)
    cases.append(("epoch 2 flat vs epoch 1 -> stop", code == STOP and not d["passed"]["C_not_flattening"], d["reason"]))
    per["steep-e2"] = rows(0.296)
    code, d = decide(per, users, "steep-e2", "good-e1", 2, 6)
    cases.append(("epoch 2 keeps improving -> go", code == GO, d["reason"]))
    # Beats ref e1 by 0.0029: past the instant band (0.0027), but with the reference slope it projects only 0.0029
    # under ref dec, short of the 1% margin (0.0031) -> stop on B.
    per["small-e1"] = rows(0.3171, noise=0.0003)
    code, d = decide(per, users, "small-e1", None, 1, 3)
    cases.append(("beats epoch 1 but projection short of the margin -> stop", code == STOP
                  and d["passed"]["A_matching_point"] and not d["passed"]["B_projection"], d["reason"]))
    per["same-e1"] = rows(0.3200)    # the reference recipe itself, trained for 8 epochs: projection decides -> go
    code, d = decide(per, users, "same-e1", None, 1, 8, same_recipe=True)
    cases.append(("same recipe, more epochs: no regression + projection -> go", code == GO, d["reason"]))
    code, d = decide(per, users, "same-e1", None, 1, 3, same_recipe=True)
    cases.append(("same recipe, same 3 epochs: projection = reference -> stop", code == STOP
                  and not d["passed"]["B_projection"], d["reason"]))
    del per["good-e1"][users[0]]
    code, d = decide(per, users, "good-e1", None, 1, 6)
    cases.append(("missing row -> stop (incomplete)", code == STOP and d["reason"] == "incomplete", d["missing_users"]["good-e1"]))
    for name, ok, detail in cases:
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")
    return 0 if all(ok for _, ok, _ in cases) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate")
    parser.add_argument("--previous", default=None, help="the candidate's epoch K-1 label (needed for K = 2)")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--jsonl", default="artifacts/small-model/heldout.jsonl")
    parser.add_argument("--users-file", default="artifacts/small-model/gate-users.json")
    parser.add_argument("--out")
    parser.add_argument("--same-recipe", action="store_true", help="the run repeats the reference recipe at more "
                        "users/epochs: test A is no-regression instead of beating the reference")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not (args.candidate and args.epoch in (1, 2) and args.epochs and args.epochs > args.epoch and args.out
            and (args.epoch == 1 or args.previous)):
        parser.error("needs --candidate, --epoch 1|2 (2 also --previous), --epochs > epoch, --out")
    users = json.loads((ROOT / args.users_file).read_text())["users"]
    labels = [args.candidate] + ([args.previous] if args.previous else []) + list(REFERENCE.values())
    per = load(ROOT / args.jsonl, labels, set(users))
    code, decision = decide(per, users, args.candidate, args.previous, args.epoch, args.epochs, args.same_recipe)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=1) + "\n", encoding="utf-8")
    print(f"[gate] {args.candidate} epoch {args.epoch}/{args.epochs}: {decision['verdict'].upper()} ({decision['reason']})")
    for name, h in decision.get("heads", {}).items():
        print(f"[gate]   {name}: cand {h['cand']:.5f} ({h['delta_pct_vs_fork']:+.2f}% vs fork), Δ vs ref e{args.epoch} "
              f"{h['delta_matching']:+.5f} (upper95 {h['delta_upper95']:+.5f}), projected final {h['projected_final']:.5f} "
              f"({h['projected_pct_vs_fork']:+.2f}%) vs target {h['go_target']:.5f} (best dec {h['best_dec_pct_vs_fork']:+.2f}%)")
    return code


if __name__ == "__main__":
    sys.exit(main())
