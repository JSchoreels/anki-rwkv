"""F74 T2: block pairing equals the committed rule on every recorded iteration; first-block pairing only when opted in."""
from argparse import ArgumentParser
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-commit", default="d834844", help="Commit holding the pre-F74 pairing rule")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    from scripts.experiment_provenance import process_provenance
    from scripts.gpu_r01_policy import replicate_blocks
    old_source = subprocess.run(["git", "show", f"{args.baseline_commit}:scripts/gpu_r01_policy.py"], cwd=ROOT,
                                capture_output=True, text=True, check=True, encoding="utf-8").stdout
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "old_policy.py"
        path.write_text(old_source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("old_policy", path)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
    recorded = []
    for iteration in sorted((ROOT / "artifacts/retraining/gpu-retrain-r01").glob("iteration-*")):
        if not (iteration / "replicates").exists() or not (iteration / "protocol.json").exists():
            continue
        protocol = json.loads((iteration / "protocol.json").read_text(encoding="utf-8"))
        control = protocol.get("control_tracks", {}).get("expertium")
        candidate = protocol.get("candidate_tracks", {}).get("expertium")
        if not control or not candidate:
            continue
        new_pairs = replicate_blocks(iteration, "expertium", control, candidate)
        old_pairs = old.replicate_blocks(iteration, "expertium", control, candidate)
        recorded.append({"iteration": iteration.name, "blocks": len(new_pairs), "turnaround": "turnaround" in protocol,
                         "equal": [list(map(str, pair)) for pair in new_pairs] == [list(map(str, pair)) for pair in old_pairs]})
    # Synthetic layout: r1 R, full C, r2 R, r3 C. Off: the full run is ignored and r2 replaces the waiting r1 (committed rule). On: (r1, full), (r2, r3).
    synthetic = {}
    with tempfile.TemporaryDirectory() as folder:
        base = Path(folder) / "iteration-9999"
        for case in ("replicates/r1/expertium/R", "expertium/C", "replicates/r2/expertium/R", "replicates/r3/expertium/C"):
            (base / case).mkdir(parents=True)
            (base / case / "result.json").write_text("{}", encoding="utf-8")
        for setting in ("separate", "first_block", None):
            protocol = {} if setting is None else {"turnaround": {"candidate_full_run": setting}}
            (base / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
            synthetic[str(setting)] = [[p.relative_to(base).as_posix() for p in pair] for pair in replicate_blocks(base, "expertium", "R", "C")]
    expected = {"None": [["replicates/r2/expertium/R", "replicates/r3/expertium/C"]],
                "separate": [["replicates/r2/expertium/R", "replicates/r3/expertium/C"]],
                "first_block": [["replicates/r1/expertium/R", "expertium/C"], ["replicates/r2/expertium/R", "replicates/r3/expertium/C"]]}
    passed = bool(recorded) and all(row["equal"] for row in recorded) and synthetic == expected
    record = {"status": "passed" if passed else "failed", "recorded_iterations": recorded, "synthetic": synthetic, "expected": expected,
              "scope": "pairing only; recorded iterations use the committed rule unchanged, first-block pairing applies only to opted-in protocols",
              "provenance": process_provenance()}
    (output / "result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "iterations": len(recorded), "unequal": [r["iteration"] for r in recorded if not r["equal"]],
                      "synthetic_ok": synthetic == expected}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
