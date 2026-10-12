"""Preserve the approved cohort while drafting the explicitly requested B01 run."""

from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance
from scripts.run_gpu_batch1 import atomic_json


def freeze(output):
    destination = output / "protocol.json"
    if destination.exists():
        raise FileExistsError("The protocol is already frozen; preserve it")
    draft = json.loads((output / "protocol-draft.json").read_text())
    original = json.loads((ROOT / draft["original_protocol"]).read_text())
    for key in ("rows", "independent_final_test_users", "training", "training_reviews", "validation_reviews", "seeds", "holdout", "acceptance", "evaluation_arithmetic", "dataset_revision", "dataset_path"):
        if draft[key] != original[key]:
            raise ValueError(f"Authorized scope changed: {key}")
    evidence = ["checks-cpu-v3", "checks-cuda-baseline-v3", "checks-cuda-expertium-v2"]
    records = []
    for folder in evidence:
        path = output / folder / "result.json"
        result = json.loads(path.read_text())
        if result["status"] != "passed":
            raise ValueError(f"Component check did not pass: {folder}")
        if "CUDA" in result["mode"]:
            if result["torch_graph_executor_optimize"] or result["deterministic_algorithms"] or not result["all_staging_checks_exact"] or not result["all_graph_checks_exact"]:
                raise ValueError("Component evidence differs from the intended execution mode")
        records.append({"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "provenance": result["provenance"]})
    from scripts.run_gpu_b01 import B01SourceWatch
    provenance = process_provenance()
    watch = B01SourceWatch(provenance)
    draft.update(protocol_status="frozen before comparison", frozen_utc=datetime.now(timezone.utc).isoformat(),
                 freeze_provenance=provenance, implementation_sha256=watch.implementation_sha256,
                 selected_components=["F14", "F01", "F03", "F04", "F05 exact shapes", "F06"],
                 options={"gpu_packed_features": True, "prefetch_users": True, "worker_local": True,
                          "packed_ipc": True, "async_transfer": True, "dynamic_graphs": True, "reduced_sync": True},
                 execution_mode={"torch_graph_executor_optimize": False, "deterministic_algorithms": False,
                                 "reason": "Both frozen real-input cases changed during initial optimized-executor calls; optimization disabled yielded exact repeated gradients with ordinary or deterministic algorithms. Apply the same compatible execution mode to both fresh controls and B01; historical causes still need full-run evidence."},
                 component_checks=records,
                 graph_policy={"shape": "exact, no buckets", "slots": 1, "admission": "second use",
                               "baseline_miss": "release retained graph before unmatched eager update", "expertium_miss": "retain until new capture admission/eviction",
                               "rng": "reset declared update seed after admission and after warmup/capture before replay"},
                 fixed_runtime={"cpu_threads": 4, "gpu_memory_fraction": 0.72, "prefetch_depth": "2 × model workers",
                                "wkv_tile": 32, "training_dropout": "enabled", "validation_dropout": "disabled"},
                 ordered_cases=[{"model": model, "track": track} for model in ("baseline", "expertium") for track in ("control", "control-repeat", "B01")],
                 comparison_scope="same 500 users and budgets for six fresh isolated cases; no old checkpoint reuse; confirm controls before each model's B01 run",
                 historical_reference="Old optimized-executor F01–F04/control remain reference evidence, not matched B01 controls",
                 telemetry="all matched cases: one-second per-core/process-tree CPU, RAM/available headroom, I/O/paging, allocated/reserved/peak VRAM; existing training GPU power/compute utilization/memory monitor; sampler wall overhead recorded; unavailable bandwidth/occupancy explicit",
                 timing_details="training time includes complete histories, loads/workers/preparation, metadata materialization, transfers, updates, captures and logs; setup/primer/checkpoint saves/validation and full subprocess wall separate",
                 acceptance_details="retain the original four-metric absolute 0.001 bound/control stability and Bonferroni divisor64, 10000 paired-user resamples seed20251012; test B01 against both fresh controls for each model; cold training faster than both; single-seed uncertainty disclosed",
                 later_R01="Frozen B01 quality reference only after both models pass; total <1% relative loss allowance and complete time-to-quality ranking apply later, not to B01")
    atomic_json(destination, draft)
    print(json.dumps({"frozen": destination.relative_to(ROOT).as_posix(), "implementation_sha256": watch.implementation_sha256,
                      "members": draft["selected_components"], "cases": len(draft["ordered_cases"])}))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmarks/gpu-baseline-b01"))
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Output must remain below workspace artifacts")
    if args.freeze:
        return freeze(output)
    destination = output / "protocol-draft.json"
    if destination.exists():
        raise FileExistsError("Preserve the existing draft; revise it explicitly")
    old = ROOT / "artifacts/benchmarks/gpu-batch1-500"
    original = json.loads((old / "protocol.json").read_text())
    transition = json.loads((old / "B01-priority-transition-request.json").read_text())
    users = [row["user_id"] for row in original["rows"]]
    reserved = original["independent_final_test_users"]
    if len(users) != 500 or len(set(users)) != 500 or any(not 1 <= user <= 5000 for user in users):
        raise ValueError("The authorized 500-user cohort changed")
    if len(reserved) != 100 or set(users) & set(reserved):
        raise ValueError("The reserved final cohort changed")
    protocol = deepcopy(original)
    protocol.update(
        bundle_id="B01",
        protocol_status="draft; membership and execution mode are not frozen; no B01 training authorized by this draft alone",
        user_authorization=transition["authorization"],
        prepared_utc=datetime.now(timezone.utc).isoformat(),
        preparation_provenance=process_provenance(),
        original_protocol="artifacts/benchmarks/gpu-batch1-500/protocol.json",
        original_protocol_sha256=hashlib.sha256((old / "protocol.json").read_bytes()).hexdigest(),
        priority_transition=transition,
        models=["baseline", "expertium"],
        selected_components=None,
        execution_mode=None,
        candidates=["F14 packed GPU features", "F01 ordered prefetch", "F03 worker-local/packed IPC",
                    "F04 transfer overlap", "F05 exact-shape graphs", "F06 reduced synchronization/logging"],
        deferred=["F02 cache reuse", "bucketed F05", "F07", "F08", "F09", "F10", "F11", "F13"],
        comparisons="two fresh matched controls and one fixed B01 per model; all execution-mode choices fixed equally before the comparisons",
        unchanged_scope="identical cohort/reserved users, temporal ranges, review budgets, seed, model batching, optimizer, precision and update order",
        membership_gate="seeded real-input, full-gradient and dropout/RNG checks; admission/capture/replay/fallback checks before including exact-shape graphs",
        acceptance_notes="preserve all four held-out binary scores using the existing paired-user bound and repeated-control stability criteria; complete full-history training faster than both controls",
    )
    atomic_json(destination, protocol)
    print(json.dumps({"draft": destination.relative_to(ROOT).as_posix(), "users": len(users),
                      "training_reviews": protocol["training_reviews"], "validation_reviews": protocol["validation_reviews"],
                      "selected_components": None, "execution_mode": None}))


if __name__ == "__main__":
    main()
