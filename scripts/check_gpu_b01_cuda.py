"""Check B01 CUDA staging, full gradients, seeded dropout and exact graphs.

Frozen paired weights and real inputs, no optimizer updates or held-out scores.
Each executor/determinism mode runs in a fresh process with preserved evidence.
"""

from argparse import ArgumentParser
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance
from scripts.run_gpu_batch1 import SourceWatch, atomic_json, tensor_hash


def checks(args, output, provenance):
    state = json.loads((ROOT / "artifacts/benchmarks/gpu-batch1-500/process.json").read_text())
    if state.get("status") == "running" or (ROOT / "artifacts/gpu-batch1.lock").exists():
        raise RuntimeError("The old benchmark must exit before CUDA checks")
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["CUPY_CACHE_DIR"] = str(ROOT / "artifacts/cache/cupy-cache")
    os.environ["TEMP"] = os.environ["TMP"] = str(ROOT / "artifacts/diagnostics/check-temp")
    protocol = json.loads((ROOT / "artifacts/benchmarks/gpu-baseline-b01/protocol-draft.json").read_text())
    os.environ["GPU_BATCH1_DATA_PATH"] = protocol["dataset_path"]
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("Use a fresh CUDA diagnostic process")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(args.deterministic, warn_only=False)
    torch._C._set_graph_executor_optimize(not args.jit_unoptimized)
    torch.cuda.set_per_process_memory_fraction(0.72, 0)
    from scripts.check_gpu_b01 import fingerprint, load_batch
    from scripts.gpu_b01 import DATA, materialize, map_batch, prepare_job, transfer_batches
    from scripts.gpu_batch1 import DynamicGraphs, ReducedSyncLoss
    from scripts.gpu_optimizations import GraphTrainingAdapter
    from scripts.check_gpu_reproducibility import differences, tensor_hash as gradient_hash
    from scripts.benchmark_gpu_trial import benchmark_models, fresh_checkpoint
    from scripts.trial_batches import TrialBatch
    from rwkv.model import nvrtc
    watch = SourceWatch(provenance)
    with ZipFile(ROOT / provenance["source_archive"]) as archive:
        manifest = json.loads(archive.read("source-manifest.json"))["files"]
    for name in ("scripts/gpu_b01.py", "scripts/check_gpu_b01.py", "scripts/check_gpu_b01_cuda.py", "scripts/check_gpu_reproducibility.py"):
        watch.hashes[name] = manifest[name]
    inputs = json.loads((args.inputs / "result.json").read_text())
    if inputs["status"] != "passed" or inputs["mode"] != "CPU staging":
        raise ValueError("Use passed real-input CPU staging evidence")
    input_watch = SourceWatch(inputs["provenance"])
    if input_watch.implementation_sha256 != watch.implementation_sha256:
        raise ValueError("Original training/preparation implementation changed")
    with ZipFile(ROOT / inputs["provenance"]["source_archive"]) as archive:
        if json.loads(archive.read("source-manifest.json"))["files"]["scripts/gpu_b01.py"] != manifest["scripts/gpu_b01.py"]:
            raise ValueError("B01 staging changed since CPU inputs")
    selected = [case for case in inputs["cases"] if case["model"] == args.model]
    # Prepare a real, small graph fixture before GPU model creation. Its smaller
    # footprint permits eager/captured gradients to be compared in one process.
    row = next(row for row in protocol["rows"] if row["user_id"] == args.graph_user)
    graph_reviews = min(args.graph_reviews, row["train_range"][1])
    graph_seed = 2025 + row["user_id"]
    graph_cpu = prepare_job(row, 0, graph_reviews, graph_seed, {}).batch if args.graph else None
    nvrtc.register()
    nvrtc.set_checkpoint_length(32)
    nvrtc.set_raw_pointer_launch(args.graph)
    device = torch.device("cuda:0")
    config = {"MODEL": args.model, "DTYPE": "bfloat16", "SEED": 2025, "WKV_BACKEND": "nvrtc"}
    checkpoint = fresh_checkpoint(args.model)["model"]
    checkpoint_source = "fresh initialization seed 2025; no optimizer updates"
    if args.model == "baseline":
        path = ROOT / "artifacts/benchmarks/gpu-batch1-500/baseline/control/master.pth"
        saved = json.loads(path.with_name("result.json").read_text())
        if saved["status"] != "complete" or saved["implementation_sha256"] != watch.implementation_sha256:
            raise ValueError("Baseline frozen checkpoint is not matched")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if tensor_hash(checkpoint) != saved["trained_parameter_sha256"]:
            raise ValueError("Frozen checkpoint weights changed")
        checkpoint_source = {"path": path.relative_to(ROOT).as_posix(), "provenance": saved["provenance"]}
    master, child, dtype = benchmark_models(config, device)
    master.load_state_dict(checkpoint)
    checkpoint = None
    parameter_hash = tensor_hash(master.state_dict())
    reduced = ReducedSyncLoss(child)
    report = {"model": args.model, "mode": "CUDA staging/gradients", "input_provenance": inputs["provenance"],
              "torch": str(torch.__version__), "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
              "deterministic_algorithms": args.deterministic, "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
              "torch_graph_executor_optimize": torch._C._get_graph_executor_optimize(),
              "dropout": "enabled", "optimizer_updates": 0, "checkpoint": checkpoint_source,
              "parameter_sha256": parameter_hash, "cases": [], "graph_checks": None,
              "scope": "three real batches/model, frozen weights and a separate recorded graph fixture; not full-history quality or speed validation"}

    def step(active, batch, seed, loss_adapter=None):
        child.zero_grad(set_to_none=True)
        active.zero_grad(set_to_none=True)
        active.copy_downcast_(master, dtype)
        active.train()
        torch.manual_seed(seed)
        before = hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()
        if isinstance(active, GraphTrainingAdapter):
            stats = active.graph_update(batch, seed)
        else:
            stats = loss_adapter.get_loss(batch) if loss_adapter else active.get_loss(batch)
            if stats is None or not torch.isfinite(stats.average_loss):
                raise RuntimeError("Nonfinite functional loss")
            stats.average_loss.backward()
        torch.cuda.synchronize()
        after = hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()
        gradients = {name: p.grad.detach().cpu().clone() for name, p in active.named_parameters() if p.grad is not None}
        if not all(torch.isfinite(value).all() for value in gradients.values()):
            raise RuntimeError("Nonfinite functional gradient")
        values = {"loss": float(stats.average_loss.detach()), "gradient_sha256": gradient_hash(gradients),
                  "rng_before_sha256": before, "rng_after_sha256": after,
                  "curve_prediction_sha256": gradient_hash({"curve": stats.p_curve.detach().cpu()}),
                  "instant_prediction_sha256": gradient_hash({"instant": stats.p_imm.detach().cpu()})}
        return values, gradients

    def compare(reference, gradients, actual, actual_gradients):
        return {"loss_equal": reference["loss"] == actual["loss"],
                "predictions_equal": all(reference[key] == actual[key] for key in ("curve_prediction_sha256", "instant_prediction_sha256")),
                "rng_equal": all(reference[key] == actual[key] for key in ("rng_before_sha256", "rng_after_sha256")),
                **differences(gradients, actual_gradients)}

    for case in selected:
        watch.check()
        batches = {}
        for kind, entry in case["files"].items():
            path = args.inputs / entry["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("Input file changed")
            batches[kind] = load_batch(path)
            if fingerprint(batches[kind]) != entry["tensor_sha256"]:
                raise ValueError("Input tensors changed")
        original = batches["original"].to(device)
        original.start = original.start.to(dtype)
        reference, reference_gradients = step(child, original, case["seed"])
        outcomes = []
        repeated, gradients = step(child, original, case["seed"])
        outcomes.append({"path": "eager repeat", "values": repeated, **compare(reference, reference_gradients, repeated, gradients)})
        del gradients
        for asynchronous in (False, True):
            item = TrialBatch(batches["packed"], case["offset"], case["raw_reviews"], batches["packed"].num_data, case["seed"], 0)
            source = iter([(row, item)])
            pipeline = transfer_batches(source, dtype, asynchronous)
            _, _, moved = next(pipeline)
            torch.cuda.synchronize()
            input_equal = fingerprint(original) == fingerprint(moved)
            actual, gradients = step(child, moved, case["seed"])
            outcomes.append({"path": "packed asynchronous" if asynchronous else "packed synchronous", "inputs_equal": input_equal,
                             "values": actual, **compare(reference, reference_gradients, actual, gradients)})
            del gradients, moved
            pipeline.close()
        actual, gradients = step(child, original, case["seed"], reduced)
        outcomes.append({"path": "reduced synchronization loss", "values": actual, **compare(reference, reference_gradients, actual, gradients)})
        del gradients, reference_gradients, original, batches
        child.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        report["cases"].append({"input": case, "reference": reference, "paths": outcomes})
        print(f"{args.model} user {case['user_id']}: " + str([(v['path'], v.get('inputs_equal', True), v['bitwise_equal'], v['rng_equal']) for v in outcomes]), flush=True)
        atomic_json(output / "progress.json", {"provenance": provenance, **report})

    if args.graph:
        graph_batch = graph_cpu.to(device)
        graph_batch.start = graph_batch.start.to(dtype)
        graph_rows = []
        references = []
        seeds = (graph_seed, graph_seed, graph_seed + 1, graph_seed)
        # A full Default eager update cannot coexist with a retained full-size
        # graph pool under the fixed memory fraction. Preserve reference results
        # on CPU before capture, rather than enlarging the memory budget.
        for index, seed in enumerate(seeds):
            current_batch = map_batch(graph_cpu, lambda x: x.to(device).clone())
            current_batch.start = current_batch.start.to(dtype)
            if index == 2:
                current_batch.start[:, 0] += 0.25
            references.append(step(child, current_batch, seed))
            del current_batch
        child.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        graphs = DynamicGraphs(child, lambda: benchmark_models(config, device)[1], retain_on_miss=args.model == "expertium")
        for index, seed in enumerate(seeds):
            # Distinct source tensors exercise static input refresh. The third
            # iteration alters a feature, then the fourth restores real inputs.
            current_batch = map_batch(graph_cpu, lambda x: x.to(device).clone())
            current_batch.start = current_batch.start.to(dtype)
            if index == 2:
                current_batch.start[:, 0] += 0.25
            reference, reference_gradients = references[index]
            torch.manual_seed(seed)
            active = graphs.select(current_batch)
            actual, gradients = step(active, current_batch, seed)
            outcome = {"index": index, "seed": seed, "stage": "eager admission" if active is child else "capture" if index == 1 else "replay",
                       "reference": reference, "values": actual, **compare(reference, reference_gradients, actual, gradients)}
            graph_rows.append(outcome)
            report["graph_checks"] = {"fixture_user": row["user_id"], "raw_reviews": graph_reviews, "rows": graph_rows, "metrics": graphs.metrics}
            atomic_json(output / "progress.json", {"provenance": provenance, **report})
            print(f"{args.model} graph {outcome['stage']} seed {seed}: gradient_equal={outcome['bitwise_equal']} RNG_equal={outcome['rng_equal']}", flush=True)
            del gradients, reference_gradients, active, current_batch
        # A second real slice checks shape-miss eager fallback.
        fallback_cpu = prepare_job(row, 0, 512, graph_seed, {}).batch
        fallback = fallback_cpu.to(device)
        fallback.start = fallback.start.to(dtype)
        reference, reference_gradients = step(child, fallback, graph_seed)
        torch.manual_seed(graph_seed)
        active = graphs.select(fallback)
        if active is not child:
            raise AssertionError("First use of a new exact shape did not fall back to eager")
        actual, gradients = step(active, fallback, graph_seed)
        graph_rows.append({"stage": "shape fallback", "reference": reference, "values": actual,
                           **compare(reference, reference_gradients, actual, gradients)})
        report["graph_checks"] = {"fixture_user": row["user_id"], "raw_reviews": graph_reviews,
                                  "source_refresh": "new source tensors each iteration; feature changed on third call, original restored on fourth",
                                  "rows": graph_rows, "metrics": graphs.metrics}
        del gradients, reference_gradients, active, graphs, fallback, fallback_cpu, graph_batch, graph_cpu
        del references
        gc.collect()
        torch.cuda.empty_cache()
    if tensor_hash(master.state_dict()) != parameter_hash:
        raise AssertionError("Diagnostic changed master weights")
    watch.check()
    report["all_staging_checks_exact"] = all(all(p["bitwise_equal"] and p["loss_equal"] and p["predictions_equal"] and p["rng_equal"] and p.get("inputs_equal", True)
                                                        for p in case["paths"]) for case in report["cases"])
    report["all_graph_checks_exact"] = None if not args.graph else all(p["bitwise_equal"] and p["loss_equal"] and p["predictions_equal"] and p["rng_equal"] for p in report["graph_checks"]["rows"])
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=Path("artifacts/benchmarks/gpu-baseline-b01/checks-cpu-v3"))
    parser.add_argument("--model", choices=("baseline", "expertium"), required=True)
    parser.add_argument("--jit-unoptimized", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--graph-user", type=int, default=157)
    parser.add_argument("--graph-reviews", type=int, default=1024)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        parser.error("Use a new workspace artifact directory")
    output.mkdir(parents=True)
    provenance = process_provenance()
    started = time.perf_counter()
    report = {"status": "running", "provenance": provenance}
    try:
        report.update(checks(args, output, provenance))
        report["status"] = "passed" if report["all_staging_checks_exact"] and report["all_graph_checks_exact"] is not False else "parity failed"
    except Exception as exc:
        report.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        report["diagnostic_wall_seconds"] = time.perf_counter() - started
        atomic_json(output / "result.json", report)
    print(json.dumps({key: report[key] for key in ("status", "all_staging_checks_exact", "all_graph_checks_exact")}), flush=True)


if __name__ == "__main__":
    main()
