"""Freeze the authorized B01 deterministic retry without changing its scope."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance
from scripts.run_gpu_b01 import B01SourceWatch
from scripts.run_gpu_batch1 import atomic_json


def main():
    output = ROOT / "artifacts/benchmarks/gpu-baseline-b01-deterministic-v3"
    if output.exists():
        raise FileExistsError("Preserve prior protocols/results; no implicit retry overwrite")
    old = ROOT / "artifacts/benchmarks/gpu-baseline-b01"
    prior_suite = json.loads((old / "suite.json").read_text())
    if prior_suite["status"] != "inconclusive: controls disagree" or len(prior_suite["results"]) != 2:
        raise ValueError("Retry only the recorded terminal failed repeated-control comparison")
    prior_bytes = (old / "protocol.json").read_bytes()
    original = json.loads(prior_bytes)
    if original["protocol_status"] != "frozen before comparison":
        raise ValueError("Unexpected original frozen-protocol schema")
    pair_path = ROOT / "artifacts/retraining/gpu-retrain-r01/b01-prefix-deterministic-pair-v1.json"
    pair = json.loads(pair_path.read_text())
    if not pair["all_prefix_hashes_and_losses_match"] or pair["paired_updates"] != 93:
        raise ValueError("Require the completed exact deterministic prefix pair")
    evidence = []
    for model in ("baseline", "expertium"):
        path = ROOT / f"artifacts/retraining/gpu-retrain-r01/b01-deterministic-checks-cuda-{model}-v1/result.json"
        check = json.loads(path.read_text())
        if (check["status"] != "passed" or not check["all_staging_checks_exact"] or not check["all_graph_checks_exact"]
                or not check["deterministic_algorithms"] or check["torch_graph_executor_optimize"]
                or check["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8"):
            raise ValueError("Require both exact deterministic staging/graph checks")
        evidence.append({"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "provenance": check["provenance"], "scope": check["scope"]})
    reporting_path = ROOT / "artifacts/retraining/gpu-retrain-r01/b01-deterministic-report-checks-v1/result.json"
    reporting = json.loads(reporting_path.read_text())
    if reporting["status"] != "passed" or len(reporting["checks"]) != 13 or not reporting["synthetic_only"]:
        raise ValueError("Require passed decision/render/workspace guard checks")
    provenance = process_provenance()
    watch = B01SourceWatch(provenance)
    protocol = deepcopy(original)
    protocol.update(protocol_status=original["protocol_status"],
                    frozen_utc=datetime.now(timezone.utc).isoformat(), freeze_provenance=provenance,
                    implementation_sha256=watch.implementation_sha256,
                    execution_mode={"torch_graph_executor_optimize": False, "deterministic_algorithms": True,
                                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                                    "reason": "Ordinary controls failed; deterministic execution configuration matched all 93 prefix hashes and both frozen staging/graph checks. Full-run stability remains gated."},
                    component_checks=evidence,
                    retry={"prior_suite": "artifacts/benchmarks/gpu-baseline-b01/suite.json", "prior_protocol": "artifacts/benchmarks/gpu-baseline-b01/protocol.json",
                           "prior_protocol_sha256": hashlib.sha256(prior_bytes).hexdigest(), "prior_frozen_provenance": original["freeze_provenance"],
                           "reason": "Retry repeatability calibration before B01/R01 acceptance; preserve failed controls and require fresh matched controls",
                           "prefix_pair": pair_path.relative_to(ROOT).as_posix(), "reporting_checks": reporting_path.relative_to(ROOT).as_posix(),
                           "preserved_preflight_failure": "artifacts/benchmarks/gpu-baseline-b01-deterministic-v2",
                           "changed": ["deterministic CUDA algorithms on", "CUBLAS workspace :4096:8", "explicit runtime environment enforcement/metadata and mode-aware report"],
                           "new_optimization_candidates": False, "additional_training_seeds": False})
    preserved = ("rows", "independent_final_test_users", "training", "training_reviews", "validation_reviews", "seeds", "holdout",
                 "acceptance", "acceptance_details", "dataset_revision", "dataset_path", "selected_components", "options",
                 "graph_policy", "models", "ordered_cases", "fixed_runtime", "evaluation_arithmetic", "comparison_scope")
    if any(protocol[key] != original[key] for key in preserved):
        raise ValueError("Authorized comparison scope or gate changed")
    output.mkdir()
    atomic_json(output / "protocol.json", protocol)
    print(json.dumps({"frozen": str(output.relative_to(ROOT)), "implementation_sha256": watch.implementation_sha256,
                      "users": 500, "seed": protocol["seeds"], "cases": len(protocol["ordered_cases"]), "execution_mode": protocol["execution_mode"]}))


if __name__ == "__main__":
    main()
