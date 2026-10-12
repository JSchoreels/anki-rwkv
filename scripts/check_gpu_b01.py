"""Real-input B01 staging and seeded gradient/RNG diagnostics; not validation."""

from argparse import ArgumentParser
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance
from scripts.run_gpu_batch1 import SourceWatch, atomic_json


def fingerprint(batch):
    from scripts.gpu_b01 import batch_tensors
    digest = hashlib.sha256()
    digest.update(json.dumps([int(batch.num_data), [[int(n) for n in g] for g in batch.sub_gather_lens]]).encode())
    for value in batch_tensors(batch):
        value = value.detach().cpu().contiguous()
        digest.update(f"{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.reshape(-1).view(__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def save_batch(batch, path):
    import torch
    from scripts.gpu_b01 import map_batch
    # Torch serialization forbids heterogeneous dtype views of one storage.
    # The IPC sharing assertion happens before this diagnostic-only cloning.
    independent = map_batch(batch, lambda value: value.clone())
    payload = {field.name: getattr(independent, field.name) for field in fields(independent)}
    payload["num_data"] = int(payload["num_data"])
    payload["sub_gather_lens"] = [[int(n) for n in g] for g in payload["sub_gather_lens"]]
    torch.save(payload, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_batch(path):
    import torch
    from rwkv.model.srs_model import PreparedBatch
    from scripts.gpu_packed_encodings import PackedPreparedBatch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return (PackedPreparedBatch if "packed" in payload else PreparedBatch)(**payload)


def cpu_checks(output, protocol):
    import torch
    torch.set_num_threads(1)
    from scripts.gpu_b01 import DATA, batch_tensors, materialize, prepare_job, training_batches
    from scripts.gpu_batch1 import prepare_job as original_job, split_frame
    from scripts.trial_batches import preparation_pool
    from rwkv.data_processing import get_rwkv_data
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU checks must not initialize CUDA")
    cases = []
    with preparation_pool(4) as pool:
        for model, span in (("baseline", 16384), ("expertium", 32768)):
            for user, offset in ((157, 0), (189, 32768), (237, 16384)):
                row = next(r for r in protocol["rows"] if r["user_id"] == user)
                end, seed = min(offset + span, row["train_range"][1]), 2025 + user + offset
                section = split_frame(get_rwkv_data(DATA, user), row, "train").iloc[offset:end]
                torch.manual_seed(9001)
                original = original_job(row, offset, end, seed, {}, section=section).batch
                original_rng_after = torch.get_rng_state().clone()
                packed = prepare_job(row, offset, end, seed, {"gpu_packed_features": True}, section=section).batch
                packed_rng_after = torch.get_rng_state().clone()
                if not torch.equal(original_rng_after, packed_rng_after):
                    raise AssertionError("Original and deferred CPU preparation consumed different seeded RNG draws")
                worker_original = pool.submit(prepare_job, row, offset, end, seed, {"worker_local": True, "packed_ipc": True}).result().batch
                worker_packed = pool.submit(prepare_job, row, offset, end, seed, {"worker_local": True, "packed_ipc": True, "gpu_packed_features": True}).result().batch
                if fingerprint(original) != fingerprint(worker_original):
                    raise AssertionError("Worker-local original preparation/IPC changed inputs")
                if fingerprint(packed) != fingerprint(worker_packed):
                    raise AssertionError("Packed metadata changed across worker-local/IPC staging")
                stores = {value.untyped_storage().data_ptr() for value in batch_tensors(worker_packed)}
                if len(stores) != 1:
                    raise AssertionError("F14 metadata did not share the packed IPC storage")
                reference = materialize(original, torch.bfloat16)
                expanded = materialize(worker_packed, torch.bfloat16)
                if fingerprint(reference) != fingerprint(expanded):
                    raise AssertionError("CPU BF16 expansion differs from original preparation")
                files = {}
                for name, batch in (("original", original), ("packed", worker_packed)):
                    path = output / f"{model}-{user}-{offset}-{name}.pt"
                    files[name] = {"file": path.name, "sha256": save_batch(batch, path), "tensor_sha256": fingerprint(batch)}
                    if fingerprint(load_batch(path)) != files[name]["tensor_sha256"]:
                        raise AssertionError("Serialization changed diagnostic inputs")
                cases.append({"model": model, "user_id": user, "offset": offset, "end": end, "seed": seed,
                              "raw_reviews": end - offset, "files": files, "ipc_storages": len(stores),
                              "cpu_bf16_materialized_sha256": fingerprint(reference),
                              "seeded_preparation_rng_after_sha256": hashlib.sha256(original_rng_after.numpy().tobytes()).hexdigest()})
                print(f"{model} user {user}: original/worker/packed IPC/CPU BF16 inputs match", flush=True)
        # A boundary fixture checks the actual spawned-worker ordered queue,
        # including worker-local loading and F01 prefetch. It intentionally has
        # two small histories and is not a full-cohort parity claim.
        selected = [dict(next(r for r in protocol["rows"] if r["user_id"] == user), train_range=[0, 256]) for user in (157, 237)]
        options = {"prefetch_users": True, "worker_local": True, "packed_ipc": True, "gpu_packed_features": True}
        order = []
        for row, item in training_batches(selected, 128, 2025, pool, options, 8):
            control = prepare_job(row, item.offset, item.offset + item.reviews, item.seed, {}).batch
            if fingerprint(materialize(item.batch, torch.bfloat16)) != fingerprint(materialize(control, torch.bfloat16)):
                raise AssertionError("Boundary queue changed model inputs")
            order.append([row["user_id"], item.offset, item.reviews, item.seed])
        expected = [[user, offset, 128, 2025 + user + offset] for user in (157, 237) for offset in (0, 128)]
        if order != expected:
            raise AssertionError("Prefetch changed review/update/seed order")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU checks unexpectedly initialized CUDA")
    return {"mode": "CPU staging", "cases": cases, "boundary_queue": order,
            "scope": "six real model-specific batch locations, four spawned workers and a two-user boundary fixture; not all full-history inputs or quality/speed validation"}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        parser.error("Use a new workspace artifact directory")
    output.mkdir(parents=True)
    provenance = process_provenance()
    protocol = json.loads((ROOT / "artifacts/benchmarks/gpu-baseline-b01/protocol-draft.json").read_text())
    os.environ["GPU_BATCH1_DATA_PATH"] = protocol["dataset_path"]
    watch = SourceWatch(provenance)
    from zipfile import ZipFile
    with ZipFile(ROOT / provenance["source_archive"]) as archive:
        manifest = json.loads(archive.read("source-manifest.json"))["files"]
    for name in ("scripts/gpu_b01.py", "scripts/check_gpu_b01.py"):
        watch.hashes[name] = manifest[name]
    started = time.perf_counter()
    report = {"status": "running", "provenance": provenance}
    try:
        report.update(cpu_checks(output, protocol))
        watch.check()
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        report["diagnostic_wall_seconds"] = time.perf_counter() - started
        atomic_json(output / "result.json", report)
    print(f"B01 staging checks passed: {output / 'result.json'}; not broader validation")


if __name__ == "__main__":
    main()
