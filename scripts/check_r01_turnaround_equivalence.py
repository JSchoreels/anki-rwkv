"""F74 T1/T2 equivalence on recorded iterations, and the measured cost of the T1 input proof.

Replay: within every recorded iteration, exact replicas of one track (same options, implementation, config,
final-parameter and held-out-input hashes) must carry byte-identical per-user and aggregate held-out scores, so
copying the first evaluated run's scores (T1) leaves every loss comparison, report and decision unchanged.
It also totals the evaluation and separate full-run seconds T1 and T2 would have avoided (historical, not new).

--measure-heldout (CPU, needs a quiet machine): prepares and hashes the held-out inputs exactly like the training
process's reuse path (same rows, seeds, worker pool and hash order), times it and compares the digest with the
recorded `validation_input_sha256`.
"""
from argparse import ArgumentParser
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
R = ROOT / "artifacts/retraining/gpu-retrain-r01"


def canonical(value):
    return json.dumps(value, sort_keys=True)


def replay():
    rows, totals = [], defaultdict(float)
    for iteration in sorted(R.glob("iteration-*")):
        paths = [*iteration.glob("expertium/*/result.json"), *iteration.glob("replicates/*/expertium/*/result.json")]
        if not (iteration / "replicates").exists() or not paths:
            continue
        groups = defaultdict(list)
        for path in paths:
            case = json.loads(path.read_text(encoding="utf-8"))
            if case.get("status") != "complete" or case.get("smoke_only") is not False:
                continue
            key = canonical([case["options"], case.get("implementation_sha256"), case.get("config"),
                             case.get("trained_parameter_sha256"), case.get("validation_input_sha256")])
            groups[key].append((case.get("started_utc") or "", path, case))
        reusable = mismatched = 0
        seconds = 0.0
        for members in groups.values():
            members.sort(key=lambda item: item[0])
            origin = members[0][2]
            for _, path, case in members[1:]:
                if case["per_user"] == origin["per_user"] and case["evaluation"] == origin["evaluation"]:
                    reusable += 1
                    seconds += case["validation_seconds"]
                else:
                    mismatched += 1
        protocol = json.loads((iteration / "protocol.json").read_text(encoding="utf-8"))
        candidate = (protocol.get("candidate_tracks") or {}).get("expertium")
        full = iteration / "expertium" / str(candidate) / "result.json"
        separate = json.loads(full.read_text(encoding="utf-8")).get("process_wall_seconds", 0.0) if full.exists() else 0.0
        evaluations = sum(len(members) for members in groups.values())
        rows.append({"iteration": iteration.name, "full_runs": evaluations, "reusable_evaluations": reusable,
                     "score_mismatches": mismatched, "evaluation_seconds_avoidable": round(seconds, 1),
                     "separate_full_run_wall_seconds": round(separate, 1)})
        totals["full_runs"] += evaluations
        totals["reusable_evaluations"] += reusable
        totals["score_mismatches"] += mismatched
        totals["evaluation_seconds_avoidable"] += seconds
        totals["separate_full_run_wall_seconds"] += separate
    return rows, {key: round(value, 1) for key, value in totals.items()}


def measure_heldout(result_path, workers):
    """Prepare and hash held-out inputs as the T1 reuse path does; no GPU work."""
    result_path = Path(result_path)
    case = json.loads(result_path.read_text(encoding="utf-8"))
    protocol = json.loads((result_path.parents[2] / "protocol.json").read_text(encoding="utf-8"))
    os.environ["GPU_BATCH1_DATA_PATH"] = str(ROOT / protocol["dataset_path"])
    from scripts.r01_workflow import configure_runtime
    configure_runtime(ROOT, 4)
    from rwkv.data_processing import get_rwkv_data
    from rwkv.prepare_batch import prepare  # noqa: F401 - imported as training does before spawning workers
    from scripts.gpu_batch1 import DATA, split_frame
    from scripts.gpu_r01_training import evaluation_job
    from scripts.trial_batches import preparation_pool
    rows = protocol.get("evaluation_rows", protocol["rows"])
    digest = hashlib.sha256()
    started = time.perf_counter()
    with preparation_pool(workers) as pool:
        pool_ready = time.perf_counter() - started
        for row in rows:
            user = row["user_id"]
            section = split_frame(get_rwkv_data(DATA, user), row, "validation")
            offsets = range(0, len(section), 1024)
            seeds = [2025 + user + row["validation_range"][0] + offset for offset in offsets]
            futures = [pool.submit(evaluation_job, section.iloc[offset:offset + 1024], user, seed) for offset, seed in zip(offsets, seeds)]
            for future in futures:
                batch = future.result()
                for tensor in (batch.start, batch.labels, batch.label_review_th):
                    digest.update(tensor.numpy().tobytes())
    elapsed = time.perf_counter() - started
    return {"case": result_path.relative_to(ROOT).as_posix(), "workers": workers, "evaluation_users": len(rows),
            "prepare_and_hash_seconds": round(elapsed, 3), "pool_start_seconds_within": round(pool_ready, 3),
            "recorded_validation_seconds": case["validation_seconds"], "digest": digest.hexdigest(),
            "digest_equals_recorded": digest.hexdigest() == case["validation_input_sha256"]}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measure-heldout", type=Path, help="Recorded full case result.json whose held-out inputs are re-prepared")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    from scripts.experiment_provenance import process_provenance
    rows, totals = replay()
    record = {"replay": {"iterations": rows, "totals": totals,
                         "conclusion": "exact replicas carry byte-identical held-out scores; T1 copies leave reports and decisions unchanged"
                         if totals.get("score_mismatches") == 0 else "score mismatch found; T1 must not reuse these groups"}}
    passed = totals.get("score_mismatches") == 0 and totals.get("reusable_evaluations", 0) > 0
    if args.measure_heldout:
        case = json.loads((ROOT / args.measure_heldout).read_text(encoding="utf-8"))
        record["heldout_proof_cost"] = measure_heldout(ROOT / args.measure_heldout, case["options"].get("preparation_workers", 6))
        passed = passed and record["heldout_proof_cost"]["digest_equals_recorded"]
    record.update(status="passed" if passed else "failed", provenance=process_provenance(),
                  scope="recorded artifacts only; avoidable seconds are historical costs, not new measurements of a changed loop")
    (output / "result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    brief = {"status": record["status"], **totals}
    if "heldout_proof_cost" in record:
        brief.update({key: record["heldout_proof_cost"][key] for key in ("prepare_and_hash_seconds", "digest_equals_recorded")})
    print(json.dumps(brief))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
