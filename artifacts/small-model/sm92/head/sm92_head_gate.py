#!/usr/bin/env python
"""Show-time retrain of the release (SM92): gate of the answer-time head retrained on the frozen SM92 model.

Runs SM82c's own rule (scripts/sm82c_head.py decide of the sm82c worktree, unchanged) with three names redirected:
the output folder (this folder, files renamed to sm92-head-decision.{json,md}), the recorded grade rows (SM92's
normal-input scores) and their label. Then the gate fixed before launch:

  PASS if the head's time LogLoss (censored 16 bins, given the true grade) on the same evaluation (722 users of the
  732 pool, the same time-scored reviews as SM82c: decide refuses rows whose reviews differ from the baselines') is
  <= SM82c's 1.9568 x 1.01, AND SM82c's rule passes (grade unchanged bitwise, head beats both baselines).

usage (CPU): $PY -X utf8 sm92_head_gate.py --time-rows T --ablation-rows A --grade-check G --train-json J
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = Path("C:/Users/jonat/workspace/wt/sm82c")
SM92 = Path("C:/Users/jonat/workspace/wt/sm92/artifacts/small-model/sm92")
REF = WT / "artifacts/small-model/sm82c/sm82c-decision.json"
LABEL = "sm92-e26-dec1ep32k"
TOLERANCE = 0.01

sys.path.insert(0, str(WT))
from scripts import sm82c_head as h  # noqa: E402

h.HERE = HERE
h.SOLD = SM92 / "gpu/sm92-inputs.jsonl"
h.SLBL = LABEL


def main():
    sys.argv = [sys.argv[0], "decide", *sys.argv[1:]]
    rc = h.main()
    if rc != 0:
        print(f"[sm92-head-gate] decide exit {rc}: no gate")
        return rc
    for ext in ("json", "md"):
        src, dst = HERE / f"sm82c-decision.{ext}", HERE / f"sm92-head-decision.{ext}"
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        src.unlink()
    res = json.loads((HERE / "sm92-head-decision.json").read_text(encoding="utf-8"))
    ref = json.loads(REF.read_text(encoding="utf-8"))
    new_ll = res["time"]["predictors"]["model"]["time_logloss"]
    ref_ll = ref["time"]["predictors"]["model"]["time_logloss"]
    same_eval = res["time"]["users"] == ref["time"]["users"] and res["time"]["reviews"] == ref["time"]["reviews"]
    limit = ref_ll * (1 + TOLERANCE)
    ok = bool(same_eval and new_ll <= limit and res["verdict"] == "pass")
    gate = {"pass": ok, "time_logloss": new_ll, "sm82c_time_logloss": ref_ll, "limit": round(limit, 5),
            "relative_pct": round(100 * (new_ll / ref_ll - 1), 3), "same_evaluation": same_eval,
            "users": res["time"]["users"], "reviews": res["time"]["reviews"], "sm82c_rule_verdict": res["verdict"],
            "mae_ln_s": res["time"]["predictors"]["model"]["mae_ln_s"],
            "sm82c_mae_ln_s": ref["time"]["predictors"]["model"]["mae_ln_s"]}
    res["sm92_gate"] = gate
    (HERE / "sm92-head-decision.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    with open(HERE / "sm92-head-decision.md", "a", encoding="utf-8") as f:
        f.write(f"\nSM92 gate (fixed before launch): time LogLoss {new_ll:.4f} vs SM82c {ref_ll:.4f} "
                f"({gate['relative_pct']:+.2f}%, limit +1% = {limit:.4f}), same evaluation {same_eval}, "
                f"SM82c rule {res['verdict']}: **{'PASS' if ok else 'FAIL'}**\n")
    print(f"[sm92-head-gate] {gate['users']} users: time LogLoss {new_ll:.4f} vs SM82c {ref_ll:.4f} "
          f"({gate['relative_pct']:+.2f}%), rule {res['verdict']}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
