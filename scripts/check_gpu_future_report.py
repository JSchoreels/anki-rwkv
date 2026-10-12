"""Regression checks for report acceptance; fabricated fixtures are not measurements."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance
from scripts import report_gpu_future as report


def checks():
    # These 500 repeated fixture rows test decision logic only. They do not
    # represent users, model training, predictive quality or measured speed.
    rows = [{"user_id": user, "train_range": [0, 2], "validation_range": [2, 4]} for user in range(1, 501)]
    case = {"name": "expertium/F10-fp8", "config": {"model": "expertium", "seed": 2025, "task": "joint", "fp8": True}}
    plan = {"authorization": {"status": "approved", "user_request": "fabricated regression fixture; not real run authorization"},
            "acceptance": {"status": "approved", "max_binary_logloss_regression": 0.001},
            "cohort": {"rows": rows, "evaluation_users": 500, "independent_final_test_users": [501],
                       "dataset_path": "fixture", "dataset_revision": "fixture"}}
    def fixture(loss=0.4, seconds=10):
        per_user = [{"user_id": row["user_id"], "input_sha256": "fixture", "raw_reviews": 2,
                     "curve_labels": 1, "instant_labels": 1, "curve_log_loss": loss, "instant_log_loss": loss} for row in rows]
        metrics = {task + "_" + suffix: value for task in ("curve", "instant")
                   for suffix, value in (("labels", 500), ("evaluated_users", 500), ("log_loss_label_weighted", loss), ("log_loss_user_average", loss))}
        history = [{"update": 1, "raw_review_presentations": 1000, "training_seconds": seconds,
                    "elapsed_seconds": seconds + 2, "metrics": metrics, "per_user": per_user}]
        return {"status": "complete", "config": {**case["config"], "dataset_path": "fixture", "dataset_revision": "fixture",
                                                     "evaluation_users": 500},
                "model": {"initial_parameter_sha256": "fixture", "initial_backbone_sha256": "fixture"},
                "updates": 1, "raw_review_presentations": 1000, "history": history, "evaluation": metrics,
                "training_seconds": seconds, "elapsed_seconds": seconds + 2}
    passed = []
    candidate, control = fixture(seconds=5), fixture()
    synthetic = deepcopy(candidate)
    synthetic["config"]["synthetic_only"] = True
    assert report.compare_case(plan, case, synthetic, [control, control])["status"] == "synthetic only; not validation"
    pending = deepcopy(plan)
    pending["authorization"]["status"] = "pending"
    assert report.compare_case(pending, case, candidate, [control, control])["status"] == "not validated: broader run approval missing"
    passed += ["synthetic results never validated", "pending plan does not grant approval"]
    saved_fingerprint = report.source_fingerprint
    # File integrity is covered separately below using an actual recorded archive.
    report.source_fingerprint = lambda value: "fixture"
    try:
        result = report.compare_case(plan, case, candidate, [control, control])
        assert result["quality_pass"] and result["speed_pass"] and result["controls_stable"]
        passed.append("same-score decision with both measured cost scopes")
        worse, repeated = fixture(0.4005, 5), fixture(0.3991)
        result = report.compare_case(plan, case, worse, [control, repeated])
        assert result["controls_stable"] and not result["quality_pass"]
        assert result["status"] == "quality non-inferiority not established"
        passed.append("candidate must meet the quality bound against both controls")
        assert report.compare_case(plan, case, candidate, [control, fixture(0.402)])["status"] == "inconclusive: controls disagree"
        passed.append("unstable controls cannot validate a candidate")
        for kind in ("input", "aggregate", "cohort_size"):
            bad, bad_plan = deepcopy(candidate), deepcopy(plan)
            if kind == "input":
                bad["history"][-1]["per_user"][0]["input_sha256"] = "different"
            elif kind == "aggregate":
                bad["evaluation"]["curve_log_loss_label_weighted"] += 0.01
            else:
                bad_plan["cohort"]["rows"] = rows[:-1]
                bad_plan["cohort"]["evaluation_users"] = 499
            try:
                report.compare_case(bad_plan, case, bad, [control, control])
            except ValueError:
                passed.append(kind + " mismatch rejected")
            else:
                raise AssertionError(kind + " mismatch accepted")
        early = deepcopy(candidate)
        early.update(raw_review_presentations=500, stop_reason="fixture early stop")
        assert report.compare_case(plan, case, early, [control, control])["status"] == "reduced exposure; observed checkpoints/target times only"
        passed.append("early stopping does not inherit full-exposure acceptance")
        pretrained = deepcopy(candidate)
        pretrained["model"]["pretraining"] = {"manifest": {"training_reviews": 123}}
        assert report.compare_case(plan, case, pretrained, [control, control])["status"] == "inconclusive: prior pretraining time not recorded"
        passed.append("missing prior cost is inconclusive")
        specialty_comparisons = {"expertium/F13-" + task: {"quality_pass": True, "full_exposure": True, "controls_stable": True}
                                 for task in ("curve", "instant")}
        cases = {"expertium/control": fixture(seconds=7), "expertium/control-repeat": fixture(seconds=7),
                 "expertium/F13-curve": fixture(seconds=4), "expertium/F13-instant": fixture(seconds=4)}
        combined = report.specialist_costs(cases, specialty_comparisons)
        assert combined["quality_pass"] and not combined["speed_pass"] and combined["summed_cost"]["training_seconds"] == 8
        passed.append("both specialist costs counted, even when each is individually faster")
    finally:
        report.source_fingerprint = saved_fingerprint
    actual = json.loads((ROOT / "artifacts/benchmarks/gpu-future/integration-final-joint-v2/result.json").read_text())
    assert len(report.source_fingerprint(actual)) == 64
    passed.append("actual saved source manifest verified")
    return passed


def main():
    provenance = process_provenance()
    result = {"analysis_provenance": provenance, "scope": "fabricated acceptance/statistics regression fixtures and one actual archive verification; no GPU launches or predictive-quality/speed measurements"}
    target = ROOT / "artifacts/benchmarks/gpu-future/report-check-v1.json"
    if target.exists():
        raise FileExistsError("Preserve earlier checks; choose a new evidence filename")
    try:
        result.update(status="passed", checks=checks())
        print(f"Passed {len(result['checks'])} reporting regression checks")
    except Exception as exc:
        result.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
