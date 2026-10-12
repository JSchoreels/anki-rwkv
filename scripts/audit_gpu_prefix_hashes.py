"""Audit two completed instrumented prefix diagnostics; no CUDA initialization."""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance
from scripts.profile_gpu_resources import snapshot

HASHES = ("cpu_input_sha256", "device_input_sha256", "post_forward_cuda_rng_sha256",
          "pre_update_master_sha256", "gradient_sha256")


def audit(left, right):
    reports, logs, inputs = [], [], []
    for folder in (left, right):
        report, metadata = snapshot(folder / "diagnostic.json")
        rows, row_metadata = snapshot(folder / "prefix-hashes.jsonl", lines=True)
        if report["status"] != "diagnostic completed; not quality validation" or not report["hash_prefix"]:
            raise ValueError("Use completed instrumented diagnostic prefixes")
        if len(rows) != report["target_update"] or any(row["step"] != index + 1 for index, row in enumerate(rows)):
            raise ValueError("Complete contiguous prefix hashes required")
        if row_metadata["ignored_unfinished_tail"]:
            raise ValueError("Completed diagnostic has an unfinished hash-log tail")
        for row in rows:
            if any(len(row[key]) != 64 or any(char not in "0123456789abcdef" for char in row[key]) for key in HASHES):
                raise ValueError("Invalid recorded hash")
        reports.append(report)
        logs.append(rows)
        inputs.extend((metadata, row_metadata))
    matched_keys = ("target_update", "user_id", "offset", "deterministic_algorithms", "hash_prefix",
                    "diagnostic_source_sha256", "frozen_protocol_sha256", "CUBLAS_WORKSPACE_CONFIG",
                    "prefix_initial_weights_sha256", "prefix_implementation_sha256", "torch", "cuda")
    if any(key not in reports[0] or key not in reports[1] or reports[0][key] != reports[1][key] for key in matched_keys):
        raise ValueError("Diagnostic settings/source/runtime/initial weights do not match")
    metadata_keys = ("step", "user_id", "offset", "seed")
    if any(any(a[key] != b[key] for key in metadata_keys) for a, b in zip(*logs)):
        raise ValueError("Paired prefix metadata differs")
    fields = {}
    for key in (*HASHES, "loss"):
        differing = [a["step"] for a, b in zip(*logs) if a[key] != b[key]]
        fields[key] = {"differing_steps": differing, "first_difference": differing[0] if differing else None}
    first = fields["gradient_sha256"]["first_difference"]
    boundary = None
    if first:
        a, b = (rows[first - 1] for rows in logs)
        boundary = {"step": first, "user_id": a["user_id"], "offset": a["offset"],
                    "same_cpu_input": a[HASHES[0]] == b[HASHES[0]],
                    "same_device_input": a[HASHES[1]] == b[HASHES[1]],
                    "same_post_forward_rng": a[HASHES[2]] == b[HASHES[2]],
                    "same_pre_update_master": a[HASHES[3]] == b[HASHES[3]]}
    return {"scope": "completed instrumented prefix pair; not predictive-quality validation or exact historical replay",
            "paired_updates": len(logs[0]), "matched_fields": list(matched_keys),
            "case_provenance": [report["provenance"] for report in reports], "input_snapshots": inputs,
            "fields": fields, "first_gradient_difference_boundary": boundary,
            "all_prefix_hashes_and_losses_match": all(not value["differing_steps"] for value in fields.values()),
            "interpretation": "Inputs/RNG/master matches at a differing gradient localize a numerical/backward boundary, not a specific operation. Trace/hash synchronization can hide scheduling effects. Full control and 500-user gates remain required."}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = (ROOT / args.output).resolve()
    if not target.is_relative_to(ROOT / "artifacts") or target.exists():
        parser.error("Preserve previous evidence; choose a new artifact JSON")
    result = {"analysis_provenance": process_provenance(), **audit((ROOT / args.left).resolve(), (ROOT / args.right).resolve())}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"all_prefix_hashes_and_losses_match": result["all_prefix_hashes_and_losses_match"],
                      "first_differences": {key: value["first_difference"] for key, value in result["fields"].items()},
                      "first_gradient_difference_boundary": result["first_gradient_difference_boundary"]}))


if __name__ == "__main__":
    main()
