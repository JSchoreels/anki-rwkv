"""Replay B01 control up to its first drifting batch, then freeze/repeat backward.

Diagnostic only: 92 optimizer updates followed by repeated step-93 gradients.
No holdout evaluation, candidate timing comparison or quality acceptance.
The shared frozen training loop is observed with a main-thread trace hook;
instrumentation can change scheduling and is not an exact historical replay.
"""

from argparse import ArgumentParser, Namespace
import ast
from dataclasses import fields
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance


class DiagnosticComplete(InterruptedError):
    pass


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--hash-prefix", action="store_true", help="Instrument full inputs/parameters/gradients/RNG at every prefix update")
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists() or args.repetitions < 2:
        parser.error("Preserve previous evidence; use a new artifact directory and at least two repeats")
    lock = ROOT / "artifacts/gpu-batch1.lock"
    if lock.exists():
        raise RuntimeError("GPU workload lock exists; do not launch a competing diagnostic")
    output.mkdir(parents=True)
    with lock.open("x") as file:
        json.dump({"pid": os.getpid(), "output": output.relative_to(ROOT).as_posix(), "scope": "B01 prefix diagnostic"}, file)
    result = {"status": "running", "provenance": process_provenance(), "target_update": 93,
              "user_id": 180, "offset": 245760, "evaluation_users": 0,
              "scope": __doc__, "deterministic_algorithms": args.deterministic, "hash_prefix": args.hash_prefix,
              "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "repetitions": []}
    started = time.perf_counter()
    try:
        frozen_path = ROOT / "artifacts/benchmarks/gpu-baseline-b01/protocol.json"
        frozen_bytes = frozen_path.read_bytes()
        protocol = json.loads(frozen_bytes)
        result["frozen_protocol_sha256"] = hashlib.sha256(frozen_bytes).hexdigest()
        result["original_execution_mode"] = protocol["execution_mode"]
        protocol["execution_mode"] = {**protocol["execution_mode"], "deterministic_algorithms": args.deterministic,
                                      "reason": "Isolated prefix reproducibility diagnostic; not a comparison protocol"}
        (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
        os.environ["CUPY_CACHE_DIR"] = str(ROOT / "artifacts/cache/cupy-cache")
        os.environ["TEMP"] = str(ROOT / "artifacts/diagnostics/check-temp")
        os.environ["TMP"] = os.environ["TEMP"]
        if args.deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError("Use a fresh CUDA process")
        from scripts import run_gpu_b01 as runner
        original = runner.original
        source = inspect.getsource(original.run_case)
        tree = ast.parse(source)
        calls = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "transfer_child_grad_to_master"]
        if len(calls) != 1:
            raise ValueError("Cannot uniquely locate post-backward observation point")
        hook_line = original.run_case.__code__.co_firstlineno + calls[0] - 1
        result.update(hook_line=hook_line, CUBLAS_WORKSPACE_CONFIG=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                      torch=str(torch.__version__), cuda=torch.version.cuda,
                      trace_scope="main-thread shared run_case only; no production source edits")

        def digest_tensors(mapping):
            digest = hashlib.sha256()
            for name, tensor in sorted(mapping.items()):
                value = tensor.detach().cpu().contiguous()
                digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
                digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
            return digest.hexdigest()

        def gradients(model):
            return {name: parameter.grad.detach().cpu().clone() for name, parameter in model.named_parameters()
                    if parameter.grad is not None}

        def digest_batch(batch):
            tensors = {}
            metadata = {}
            def visit(name, value):
                if isinstance(value, torch.Tensor):
                    tensors[name] = value
                elif isinstance(value, (list, tuple)):
                    for index, entry in enumerate(value):
                        visit(name + "/" + str(index), entry)
                else:
                    metadata[name] = int(value)
            for field in fields(batch):
                visit(field.name, getattr(batch, field.name))
            return hashlib.sha256((digest_tensors(tensors) + json.dumps(metadata, sort_keys=True)).encode()).hexdigest()

        def trace(frame, event, arg):
            if frame.f_code is not original.run_case.__code__:
                return None
            if event != "line" or frame.f_lineno != hook_line:
                return trace
            local = frame.f_locals
            item, batch, model, master, dtype = (local[key] for key in ("item", "batch", "active", "master", "dtype"))
            if args.hash_prefix:
                record = {"step": local["report"]["updates"] + 1, "user_id": local["user"], "offset": item.offset,
                          "seed": item.seed, "cpu_input_sha256": digest_batch(item.batch),
                          "device_input_sha256": digest_batch(batch), "pre_update_master_sha256": digest_tensors(master.state_dict()),
                          "gradient_sha256": digest_tensors(gradients(model)),
                          "post_forward_cuda_rng_sha256": digest_tensors({"state": torch.cuda.get_rng_state()}),
                          "loss": float(local["stats"].average_loss.detach())}
                with (output / "prefix-hashes.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(record, allow_nan=False) + "\n")
            if local["report"]["updates"] != 92:
                return trace
            if item.offset != result["offset"] or local["user"] != result["user_id"]:
                raise ValueError("Prefix observation does not match the actual first drifting batch")
            frozen_master = {name: value.detach().cpu().clone() for name, value in master.state_dict().items()}
            result.update(prefix_optimizer_updates=92, prefix_review_presentations=local["report"]["review_presentations"],
                          prefix_initial_weights_sha256=local["report"]["initial_parameter_sha256"],
                          prefix_implementation_sha256=local["report"]["implementation_sha256"],
                          seed=item.seed, raw_reviews=item.reviews,
                          frozen_master_sha256=digest_tensors(frozen_master),
                          device_input_sha256=digest_tensors({"start": batch.start, "labels": batch.labels}),
                          observation="First gradient is from the shared loop; subsequent repetitions use those same pre-update master weights/input and declared seed")
            torch.save(frozen_master, output / "master-before-step93.pth")
            payload = {field.name: getattr(item.batch, field.name) for field in fields(item.batch)}
            payload["num_data"] = int(payload["num_data"])
            payload["sub_gather_lens"] = [[int(value) for value in group] for group in payload["sub_gather_lens"]]
            torch.save(payload, output / "input-step93.pt")
            reference = gradients(model)
            reference_hash = digest_tensors(reference)
            first_loss = float(local["stats"].average_loss.detach())
            first_rng = digest_tensors({"cuda_rng_state": torch.cuda.get_rng_state()})
            result["repetitions"].append({"repetition": 1, "loss": first_loss, "gradient_sha256": reference_hash,
                                          "post_forward_cuda_rng_sha256": first_rng, "bitwise_equal": True, "max_absolute_gradient_difference": 0.0})
            for repetition in range(2, args.repetitions + 1):
                model.zero_grad(set_to_none=True)
                model.copy_downcast_(master, dtype)
                model.train()
                torch.manual_seed(item.seed)
                stats = model.get_loss(batch)
                if stats is None or not torch.isfinite(stats.average_loss):
                    raise RuntimeError("Nonfinite frozen-prefix loss")
                stats.average_loss.backward()
                current = gradients(model)
                if current.keys() != reference.keys() or any(not torch.isfinite(value).all() for value in current.values()):
                    raise RuntimeError("Missing/nonfinite frozen-prefix gradients")
                sha = digest_tensors(current)
                maximum = max(float((value.float() - reference[name].float()).abs().max()) for name, value in current.items())
                result["repetitions"].append({"repetition": repetition, "loss": float(stats.average_loss.detach()),
                                              "gradient_sha256": sha, "post_forward_cuda_rng_sha256": digest_tensors({"cuda_rng_state": torch.cuda.get_rng_state()}),
                                              "bitwise_equal": sha == reference_hash, "max_absolute_gradient_difference": maximum})
                original.atomic_json(output / "diagnostic.json", result)
                print(f"Step93 frozen repetition {repetition}: gradient_equal={sha == reference_hash}; maximum={maximum}", flush=True)
            if digest_tensors(master.state_dict()) != result["frozen_master_sha256"]:
                raise AssertionError("Frozen diagnostic changed pre-update master weights")
            result["all_repeated_gradients_bitwise_equal"] = all(row["bitwise_equal"] for row in result["repetitions"])
            result["all_post_forward_rng_states_match"] = all(row["post_forward_cuda_rng_sha256"] == first_rng for row in result["repetitions"])
            raise DiagnosticComplete("Intentional cutoff after step-93 frozen gradients; no step-93 optimizer update or evaluation")

        case_args = Namespace(model="baseline", track="control", output=output.relative_to(ROOT).as_posix(),
                              expected_source=protocol["implementation_sha256"], smoke=False)
        previous_trace = sys.gettrace()
        try:
            sys.settrace(trace)
            code = runner.case(case_args, protocol)
        finally:
            sys.settrace(previous_trace)
        raw = json.loads((output / "baseline/control/result.json").read_text())
        if code != 1 or raw.get("error_type") != "DiagnosticComplete" or len(result["repetitions"]) != args.repetitions:
            raise RuntimeError(f"Prefix diagnostic did not reach its intentional cutoff: {raw.get('error')}")
        result.update(status="diagnostic completed; not quality validation", raw_case_status=raw["status"],
                      execution_mode=raw["execution_mode"], raw_case_provenance=raw["provenance"])
    except Exception as exc:
        result.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        result["diagnostic_wall_seconds"] = time.perf_counter() - started
        (output / "diagnostic.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        lock.unlink(missing_ok=True)
    print(json.dumps({key: result[key] for key in ("status", "all_repeated_gradients_bitwise_equal", "all_post_forward_rng_states_match")}))


if __name__ == "__main__":
    main()
