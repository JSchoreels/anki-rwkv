"""Diagnose saved first-batch numerical drift; not quality/speed validation.

CPU mode prepares the first differing F01/F02 inputs by serial, loader-thread
and sample-cache paths. CUDA mode is a separate, fresh process after the suite
exits: frozen weights and inputs are repeated without optimizer updates.
"""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
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


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def batch_hash(batch):
    from scripts.train_gpu_future import batch_fingerprint
    digest = hashlib.sha256()
    batch_fingerprint(batch, digest)
    return digest.hexdigest()


def cpu_inputs(output):
    import torch
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU diagnostic must not initialize CUDA")
    suite = ROOT / "artifacts/benchmarks/gpu-batch1-500"
    protocol = json.loads((suite / "protocol.json").read_text())
    os.environ["GPU_BATCH1_DATA_PATH"] = protocol["dataset_path"]
    from scripts.gpu_batch1 import DATA, prepare_job, sample_key, split_frame
    from rwkv.data_processing import get_rwkv_data
    if json.loads((DATA / "download_manifest.json").read_text())["revision"] != protocol["dataset_revision"]:
        raise ValueError("Dataset revision differs from the active comparison")
    cases = []
    for user, offset, historical in ((189, 32768, "F02 first differing gradient norm, step 100"),
                                      (237, 16384, "F01 first differing logged loss/norm, step 132")):
        row = next(row for row in protocol["rows"] if row["user_id"] == user)
        end = min(offset + 16384, row["train_range"][1])
        seed = 2025 + user + offset
        serial = split_frame(get_rwkv_data(DATA, user), row, "train").iloc[offset:end]
        first = prepare_job(row, offset, end, seed, {}, section=serial)
        with ThreadPoolExecutor(max_workers=1) as loader:
            threaded = loader.submit(get_rwkv_data, DATA, user).result()
        section = split_frame(threaded, row, "train").iloc[offset:end]
        second = prepare_job(row, offset, end, seed, {}, section=section)
        hashes = {"serial_source_load": batch_hash(first.batch), "loader_thread_source_load": batch_hash(second.batch)}
        cache_path = suite / "sample-cache" / (sample_key(row, offset, end, 128) + ".pt")
        if cache_path.exists():
            cached = prepare_job(row, offset, end, seed, {"sample_cache": True}, cache_path=cache_path)
            hashes["sample_cache"] = batch_hash(cached.batch)
        if len(set(hashes.values())) != 1:
            raise AssertionError(f"Focused CPU input mismatch: {hashes}")
        payload = {field.name: getattr(first.batch, field.name) for field in fields(first.batch)}
        # Greedy split endpoints can be NumPy scalar integers. Preserve their
        # values as built-in primitives so weights_only loading needs no pickle
        # class allowlist or relaxed loader.
        payload["num_data"] = int(payload["num_data"])
        payload["sub_gather_lens"] = [[int(value) for value in group] for group in payload["sub_gather_lens"]]
        path = output / f"input-{user}-{offset}.pt"
        torch.save(payload, path)
        # Check the serialized diagnostic input too; no dataclass pickle needed.
        from rwkv.model.srs_model import PreparedBatch
        restored = PreparedBatch(**torch.load(path, weights_only=True, map_location="cpu"))
        if batch_hash(restored) != hashes["serial_source_load"]:
            raise AssertionError("Diagnostic input serialization changed tensors")
        cases.append({"user_id": user, "offset": offset, "end": end, "raw_reviews": first.reviews,
                      "chunks": first.chunks, "seed": seed, "historical_location": historical,
                      "hashes": hashes, "input_file": path.name,
                      "input_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU diagnostic unexpectedly initialized CUDA")
    return {"mode": "CPU inputs", "dataset_path": protocol["dataset_path"], "dataset_revision": protocol["dataset_revision"],
            "cases": cases, "preparation": "serial prepare_job; source load on main/loader thread and existing sample-cache path",
            "scope": "two historical input locations only; does not reproduce the live four-worker pipeline or all 1881 batches"}


def tensor_hash(values):
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def differences(reference, actual):
    import torch
    if set(reference) != set(actual):
        raise ValueError("Gradient parameter sets differ")
    changed = [name for name in reference if not torch.equal(reference[name], actual[name])]
    return {"bitwise_equal": not changed, "different_tensors": len(changed), "first_different_names": changed[:8],
            "max_abs": max((float((reference[name].float() - actual[name].float()).abs().max()) for name in changed), default=0.0)}


def isolate_gather(batch, dtype, device, repetitions, width=128):
    """Actual time-shift indices; arbitrary fixed gradients on unpadded positions."""
    import torch
    # Choose the largest sequence group to retain realistic repeated indices.
    _, stream, index = max((group.numel(), stream, index) for stream, groups in enumerate(batch.sub_gather) for index, group in enumerate(groups))
    length = batch.sub_gather_lens[stream][index]
    indices = batch.time_shift_selects[stream][index].reshape(-1, length).to(device)
    real = (batch.sub_gather[stream][index].reshape(-1, length) >= 0).to(device)
    torch.manual_seed(2025)
    value = torch.randn(*indices.shape, width, dtype=dtype, device=device, requires_grad=True)
    gradient = torch.randn_like(value) * real.unsqueeze(-1)
    expanded = indices.unsqueeze(-1).expand_as(value)
    rows, reference = [], None
    for _ in range(repetitions):
        result = torch.gather(value, 1, expanded)
        grad = torch.autograd.grad(result, value, gradient)[0].detach().cpu()
        if reference is None:
            reference = {"gradient": grad}
        rows.append({"gradient_sha256": tensor_hash({"gradient": grad}), **differences(reference, {"gradient": grad})})
    return {"stream": stream, "split": index, "length": length, "width": width, "dtype": str(dtype), "repetitions": rows,
            "scope": "isolated gather backward, fixed arbitrary gradient masked to nonpadding positions; not a model gradient or historical replay"}


def cuda_repeats(args, output, provenance):
    import psutil
    state, lock = ROOT / "artifacts/benchmarks/gpu-batch1-500/process.json", ROOT / "artifacts/gpu-batch1.lock"
    if (state.exists() and json.loads(state.read_text()).get("status") == "running") or (lock.exists() and psutil.pid_exists(json.loads(lock.read_text())["pid"])):
        raise RuntimeError("Verify the active suite has exited before launching GPU diagnostics")
    # Set before importing/building any CUDA model. Deterministic and ordinary
    # modes must use separate fresh processes; report their actual environment.
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["CUPY_CACHE_DIR"] = str(ROOT / "artifacts/cache/cupy-cache")
    temporary = ROOT / "artifacts/benchmarks/gpu-future/temp"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = str(temporary)
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("GPU diagnostic modes require a fresh CUDA process")
    torch.set_num_threads(4)
    if args.jit_unoptimized:
        torch._C._set_graph_executor_optimize(False)
    torch.use_deterministic_algorithms(args.deterministic, warn_only=False)
    torch.cuda.set_per_process_memory_fraction(0.72, 0)
    from rwkv.model import nvrtc
    from rwkv.model.srs_model import PreparedBatch
    from rwkv.train_rwkv import CLIP, transfer_child_grad_to_master
    from scripts.benchmark_gpu_trial import benchmark_models
    from scripts.run_gpu_batch1 import SourceWatch, tensor_hash as model_hash
    watch = SourceWatch(provenance)
    with ZipFile(ROOT / provenance["source_archive"]) as archive:
        own_manifest = json.loads(archive.read("source-manifest.json"))["files"]
    for name in ("scripts/check_gpu_reproducibility.py", "scripts/train_gpu_future.py"):
        watch.hashes[name] = own_manifest[name]
    watch.check()
    inputs = json.loads((args.inputs / "result.json").read_text())
    if inputs["status"] != "passed" or inputs.get("mode") != "CPU inputs":
        raise ValueError("Use a successful CPU input diagnostic")
    input_watch = SourceWatch(inputs["provenance"])
    if input_watch.implementation_sha256 != watch.implementation_sha256:
        raise ValueError("Training/preparation source differs from recorded diagnostic input")
    case_folder = ROOT / "artifacts/benchmarks/gpu-batch1-500" / args.model / "control"
    recorded = json.loads((case_folder / "result.json").read_text())
    if recorded["status"] != "complete" or recorded["evaluation_users"] != 500:
        raise ValueError("Use a completed matched-control checkpoint")
    if recorded["implementation_sha256"] != watch.implementation_sha256:
        raise ValueError("Source changed since the frozen control checkpoint")
    checkpoint_path = case_folder / "master.pth"
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    if model_hash(checkpoint) != recorded["trained_parameter_sha256"]:
        raise ValueError("Frozen checkpoint differs from the recorded trained weights")
    nvrtc.register()
    nvrtc.set_checkpoint_length(32)
    nvrtc.set_raw_pointer_launch(False)
    device = torch.device("cuda:0")
    master, child, dtype = benchmark_models({"MODEL": args.model, "DTYPE": "bfloat16", "SEED": 2025, "WKV_BACKEND": "nvrtc"}, device)
    master.load_state_dict(checkpoint)
    checkpoint = None
    report = {"mode": "CUDA repeats", "model": args.model, "deterministic_algorithms": args.deterministic,
              "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "torch": str(torch.__version__),
              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
              "torch_graph_executor_optimize": torch._C._get_graph_executor_optimize(),
              "torch_jit_profiled_runs": torch._C._jit_get_num_profiled_runs(),
              "frozen_checkpoint": str(checkpoint_path.relative_to(ROOT)), "frozen_checkpoint_provenance": recorded["provenance"],
              "checkpoint_file_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
              "checkpoint_parameter_sha256": recorded["trained_parameter_sha256"], "input_provenance": inputs["provenance"],
              "optimizer_updates": 0, "cases": [],
              "scope": "repeated frozen final-control weights/inputs; not historical prefix replay, predictive quality or speed validation"}
    for case in inputs["cases"]:
        path = args.inputs / case["input_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["input_file_sha256"]:
            raise ValueError("Diagnostic input file changed")
        batch = PreparedBatch(**torch.load(path, weights_only=True, map_location="cpu"))
        if batch_hash(batch) != case["hashes"]["serial_source_load"]:
            raise ValueError("Diagnostic input tensors changed")
        rows, reference = [], None
        for repetition in range(args.repetitions):
            watch.check()
            master.zero_grad(set_to_none=True)
            child.zero_grad(set_to_none=True)
            child.copy_downcast_(master, dtype)
            child.train()
            # Vary a small live allocation before copying identical input data.
            # This changes allocator history without changing the frozen tensors.
            scratch = torch.empty((repetition + 1) * 262144, device=device)
            moved = batch.to(device)
            moved.start = moved.start.to(dtype)
            torch.manual_seed(case["seed"])
            stats = child.get_loss(moved)
            if stats is None or not torch.isfinite(stats.average_loss):
                raise RuntimeError("Nonfinite frozen-input objective")
            stats.average_loss.backward()
            gradient = {name: parameter.grad.detach().cpu().clone() for name, parameter in child.named_parameters() if parameter.grad is not None}
            if not all(torch.isfinite(value).all() for value in gradient.values()):
                raise RuntimeError("Nonfinite repeated gradient")
            if reference is None:
                reference = gradient
            transfer_child_grad_to_master(master, child)
            norm = torch.nn.utils.clip_grad_norm_(master.parameters(), CLIP, error_if_nonfinite=True)
            rows.append({"repetition": repetition, "loss": float(stats.average_loss.detach()), "gradient_norm_before_clip": float(norm),
                         "gradient_sha256": tensor_hash(gradient), **differences(reference, gradient)})
            moved = scratch = stats = gradient = None
            print(f"{args.model} user {case['user_id']} repetition {repetition}: gradient_equal={rows[-1]['bitwise_equal']}", flush=True)
        child.zero_grad(set_to_none=True)
        master.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        isolated = [isolate_gather(batch, dtype, device, args.repetitions, 128 if args.model == "baseline" else 32)
                    for dtype in (torch.float32, torch.bfloat16)]
        report["cases"].append({"input": case, "full_model_repetitions": rows, "isolated_gather": isolated})
        atomic_json(output / "progress.json", {"provenance": provenance, **report})
    if model_hash(master.state_dict()) != report["checkpoint_parameter_sha256"]:
        raise AssertionError("Frozen diagnostic unexpectedly changed model weights")
    watch.check()
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--jit-unoptimized", action="store_true", help="Separate fresh-process diagnostic of TorchScript executor optimization")
    parser.add_argument("--model", choices=("baseline", "expertium"), default="baseline")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists() or args.repetitions < 2:
        parser.error("Use a new artifact directory and at least two repetitions")
    if args.cuda and not args.inputs:
        parser.error("CUDA mode requires --inputs from a successful CPU diagnostic")
    if args.deterministic and not args.cuda:
        parser.error("Deterministic CUDA mode is a separate GPU process")
    if args.inputs:
        args.inputs = args.inputs.resolve()
        if not args.inputs.is_relative_to(ROOT / "artifacts"):
            parser.error("Inputs must be recorded workspace artifacts")
    output.mkdir(parents=True)
    provenance = process_provenance()
    started = time.perf_counter()
    result = {"status": "running", "provenance": provenance}
    try:
        result.update(cuda_repeats(args, output, provenance) if args.cuda else cpu_inputs(output))
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        result["diagnostic_wall_seconds"] = time.perf_counter() - started
        atomic_json(output / "result.json", result)
    print(f"Diagnostic passed: {output / 'result.json'}; not broader validation")


if __name__ == "__main__":
    main()
