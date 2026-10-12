"""Read saved paired logs without loading models or launching CUDA work."""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance
from scripts.profile_gpu_resources import snapshot


def audit(left, right):
    reports, logs, inputs = [], [], []
    for folder in (left, right):
        report, metadata = snapshot(folder / "result.json")
        log, log_metadata = snapshot(folder / "updates.jsonl", lines=True)
        if any(row.get("step") != index + 1 for index, row in enumerate(log)):
            raise ValueError("Saved updates must be a contiguous prefix starting at step 1")
        reports.append(report)
        logs.append(log)
        inputs.extend((metadata, log_metadata))
    keys = ("step", "user_id", "offset", "reviews", "seed", "curve_labels", "instant_labels", "lr")
    mismatches = [a["step"] for a, b in zip(*logs) if any(a[key] != b[key] for key in keys)]
    differences = [{"step": a["step"], "user_id": a["user_id"], "offset": a["offset"],
                    "control_loss": a["loss"], "candidate_loss": b["loss"],
                    "control_gradient_norm": a["gradient_norm"], "candidate_gradient_norm": b["gradient_norm"]}
                   for a, b in zip(*logs) if any(a[key] != b[key] for key in ("loss", "gradient_norm"))]
    manifests = []
    for report in reports:
        with ZipFile(ROOT / report["provenance"]["source_archive"]) as archive:
            manifests.append(json.loads(archive.read("source-manifest.json"))["files"])
    # Compare only code used by these running cases. Other chats can commit
    # unrelated files and future prototypes between child-process starts.
    consumed = {name for name in set(manifests[0]) | set(manifests[1])
                if name.startswith(("rwkv/", "expertium/")) or name in (
                    "pyproject.toml", "uv.lock", "scripts/run_gpu_batch1.py", "scripts/gpu_batch1.py",
                    "scripts/benchmark_gpu_trial.py", "scripts/train_gpu_trial.py", "scripts/gpu_optimizations.py",
                    "scripts/gpu_packed_encodings.py", "scripts/trial_batches.py", "scripts/compare.py",
                    "scripts/experiment_provenance.py", "scripts/gpu_power.py",
                    "scripts/run_gpu_b01.py", "scripts/gpu_b01.py", "scripts/report_gpu_b01.py",
                    "scripts/gpu_resource_monitor.py", "scripts/run_gpu_b01.ps1",
                    "scripts/report_gpu_batch1.py", "scripts/report_gpu_batch1_relative.py")}
    complete = all(report.get("status") == "complete" for report in reports)
    def hashes_match(key):
        values = [report.get(key) for report in reports]
        return values[0] == values[1] if all(values) else None
    return {"scope": "complete saved logs" if complete else "saved update prefixes; full comparison pending",
            "limitations": "read-only saved-log/source audit; not a rerun, input-tensor parity test, full quality gate or diagnosis of the cause",
            "input_snapshots": inputs, "case_statuses": [report.get("status") for report in reports],
            "cases": [folder.relative_to(ROOT).as_posix() for folder in (left, right)],
            "experiment_provenance": [report["provenance"] for report in reports],
            "update_counts": [len(log) for log in logs], "common_prefix_updates": min(map(len, logs)),
            "complete_log_counts_match": len(logs[0]) == len(logs[1]) if complete else None,
            "metadata_fields": list(keys),
            "metadata_mismatch_steps": mismatches,
            "consumed_source_differences": [name for name in sorted(consumed) if manifests[0].get(name) != manifests[1].get(name)],
            "initial_weights_match": hashes_match("initial_parameter_sha256"),
            "validation_inputs_match": hashes_match("validation_input_sha256"),
            "implementation_fingerprint_match": hashes_match("implementation_sha256"),
            "protocol_hash_match": hashes_match("protocol_sha256"),
            "differing_logged_numerical_updates": len(differences), "first_numerical_differences": differences[:3],
            "interpretation": "Equal logged batch metadata and equal initial weights do not establish identical gradients. Tiny initial numerical differences can grow during training; repeated controls and a focused GPU reproducibility check are still required. The origin of any difference remains unverified."}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    if not target.is_relative_to(ROOT / "artifacts") or target.exists():
        parser.error("Use a new artifact JSON; preserve earlier evidence")
    provenance = process_provenance()
    result = {"analysis_provenance": provenance, **audit(args.control.resolve(), args.candidate.resolve())}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("update_counts", "metadata_mismatch_steps", "consumed_source_differences",
                                                   "differing_logged_numerical_updates", "first_numerical_differences")}, indent=2))


if __name__ == "__main__":
    main()
