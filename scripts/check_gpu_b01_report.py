"""Synthetic checks of B01 decision gates and percentage rendering; no CUDA."""

from copy import deepcopy
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance
from scripts.report_gpu_b01 import assess, write_report
from scripts.run_gpu_batch1 import aggregate


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmarks/gpu-baseline-b01/report-checks-v1"))
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Output must remain within artifacts")
    output.mkdir(parents=True, exist_ok=False)
    provenance = process_provenance()
    cases = []
    for model in ("baseline", "expertium"):
        for track, seconds in (("control", 100), ("control-repeat", 105), ("B01", 70)):
            users = [{"user_id": user, "curve_labels": 10 + user, "instant_labels": 20 + user,
                      "curve_log_loss": 0.4, "instant_log_loss": 0.38} for user in range(1, 501)]
            cases.append({"model": model, "track": track, "status": "complete", "smoke_only": False,
                          "evaluation_users": 500, "per_user": users, "evaluation": aggregate(users),
                          "training_seconds": seconds, "cold_training_seconds": seconds, "reviews_per_second": 10000 / seconds,
                          "peak_reserved_mib": 1, "implementation_sha256": "synthetic-source", "initial_parameter_sha256": "synthetic-init",
                          "validation_input_sha256": "synthetic-inputs", "updates": 500, "review_presentations": 10000,
                          "training_curve_labels": 2000, "training_instant_labels": 3000,
                          "execution_mode": {"torch_graph_executor_optimize": False}, "protocol_sha256": "synthetic-protocol"})
    mode = {"torch_graph_executor_optimize": False, "deterministic_algorithms": True, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    for row in cases:
        row["execution_mode"] = dict(mode)
    suite = {"status": "synthetic checks only", "results": cases, "protocol": {"execution_mode": mode}}
    outcomes = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        outcomes.append(name)

    check("stable equal-quality faster bundle accepted for both models", assess(suite)["B01_accepted_both_models"])
    changed = deepcopy(suite)
    changed["results"][1]["evaluation"]["curve_log_loss_user_average"] += 0.002
    check("unstable controls are inconclusive", not assess(changed)["B01_accepted_both_models"])
    changed = deepcopy(suite)
    changed["results"][2]["cold_training_seconds"] = 110
    check("candidate must be faster than both controls", not assess(changed)["B01_accepted_both_models"])
    changed = deepcopy(suite)
    for user in changed["results"][2]["per_user"]:
        user["instant_log_loss"] += 0.002
    changed["results"][2]["evaluation"] = aggregate(changed["results"][2]["per_user"])
    check("one failing binary task prevents acceptance", not assess(changed)["B01_accepted_both_models"])
    changed = deepcopy(suite)
    changed["results"] = changed["results"][:1]
    check("missing repeated controls stay pending", not assess(changed)["B01_accepted_both_models"])
    for key in ("execution_mode", "protocol_sha256", "initial_parameter_sha256", "validation_input_sha256"):
        changed = deepcopy(suite)
        changed["results"][2][key] = "changed"
        try:
            assess(changed)
        except ValueError:
            check(f"unmatched {key} rejected", True)
        else:
            raise AssertionError(f"unmatched {key} accepted")
    write_report(output, suite, update_document=False)
    rendered = (output / "summary.md").read_text()
    check("percentage delta time and speed rendered", "-30.00%" in rendered and "+42.86%" in rendered)
    check("both-control uncertainty rows rendered", "vs control-repeat" in rendered and "Adjusted upper LL bounds" in rendered)
    check("report reflects frozen deterministic execution mode", "deterministic algorithms=True" in rendered and "CUBLAS workspace=:4096:8" in rendered)
    from scripts.run_gpu_b01 import case
    previous_workspace = os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    try:
        try:
            case(None, {"execution_mode": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}})
        except RuntimeError as exc:
            check("missing frozen workspace rejected before CUDA case", "CUBLAS workspace" in str(exc))
        else:
            raise AssertionError("Unmatched workspace was accepted")
    finally:
        if previous_workspace is not None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_workspace
    (output / "result.json").write_text(json.dumps({"status": "passed", "synthetic_only": True, "checks": outcomes, "provenance": provenance}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "checks": len(outcomes), "synthetic_only": True}))


if __name__ == "__main__":
    main()
