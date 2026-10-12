"""Audit existing B01 functional runner results; does not launch training/CUDA."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance


def main():
    destination = ROOT / "artifacts/benchmarks/gpu-baseline-b01/runner-smoke-audit.json"
    if destination.exists():
        raise FileExistsError("Preserve the existing audit")
    provenance = process_provenance()
    outcomes = []
    keys = ("implementation_sha256", "initial_parameter_sha256", "trained_parameter_sha256", "validation_input_sha256",
            "updates", "review_presentations", "evaluation", "execution_mode", "protocol_sha256")
    for model, version in (("baseline", "v2"), ("expertium", "v3")):
        paths = [ROOT / "artifacts/benchmarks/gpu-baseline-b01" / f"runner-smoke-{version}" / model / track / "result.json" for track in ("control", "B01")]
        cases = [json.loads(path.read_text()) for path in paths]
        if not all(c["status"] == "complete" and c["smoke_only"] and c["updates"] == 2 and c["review_presentations"] == 1024 for c in cases):
            raise AssertionError("Functional runner coverage/result mismatch")
        mismatches = [key for key in keys if cases[0].get(key) != cases[1].get(key)]
        if mismatches:
            raise AssertionError(f"Functional paired outputs differ: {mismatches}")
        outcomes.append({"model": model, "cases": [path.relative_to(ROOT).as_posix() for path in paths],
                         "matched_fields": list(keys), "all_equal": True,
                         "experiment_provenance": [case["provenance"] for case in cases]})
    telemetry = []
    for model, track in (("baseline", "B01"), ("expertium", "control"), ("expertium", "B01")):
        folder = ROOT / "artifacts/benchmarks/gpu-baseline-b01/runner-smoke-v3" / model / track
        result = json.loads((folder / "result.json").read_text())
        summary = json.loads((folder / "resources-summary.json").read_text())
        if result["status"] != "complete" or not summary["samples"] or summary["sampling_errors"]:
            raise AssertionError("Revised telemetry did not retain samples")
        samples = [json.loads(line) for line in (folder / "resources.jsonl").read_text().splitlines()]
        if not all(sample["cpu_per_core_percent"] and sample["ram_available_bytes"] > 0 for sample in samples):
            raise AssertionError("CPU/RAM telemetry incomplete")
        if not any(sample["processes"] for sample in samples):
            raise AssertionError("Process/worker telemetry missing")
        telemetry.append({"model": model, "track": track, "summary": summary})
    report = {"status": "passed", "analysis_provenance": provenance, "paired_cases": outcomes, "telemetry": telemetry,
              "scope": "existing two-user functional cases only, two optimizer updates and 1024 training/1024 held-out reviews per case; neither broad validation nor full-history resource saturation"}
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "paired_models": 2, "telemetry_cases": 3, "smoke_only": True}))


if __name__ == "__main__":
    main()
