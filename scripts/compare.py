"""Compare the baseline and pinned Expertium iter36 on identical CPU or CUDA batches."""

from argparse import ArgumentParser, Namespace
from dataclasses import asdict, replace
import hashlib
import os
import json
from pathlib import Path
import sys
import time
from tempfile import NamedTemporaryFile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance

import torch
import torch.nn.functional as F

from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG as BASELINE_CONFIG
from rwkv.data_processing import create_sample, get_rwkv_data
from rwkv.model.srs_model import PreparedBatch, SrsRWKV as BaselineModel
from rwkv.model.srs_model_rnn import SrsRWKVRnn as BaselineRNN
from rwkv.prepare_batch import prepare
from rwkv.train_rwkv import CLIP, get_optimizer, transfer_child_grad_to_master
from expertium.architecture import DEFAULT_ANKI_RWKV_CONFIG as SMALL_CONFIG
from expertium.model.srs_model import SrsRWKV as SmallModel
from expertium.model.srs_model_rnn import SrsRWKVRnn as SmallRNN


def build_batches(config):
    train, validation, coverage = [], [], {}
    fingerprint = hashlib.sha256()
    for user in range(config["USER_START"], config["USER_END"] + 1):
        frame = get_rwkv_data(Path(config["DATA_PATH"]), user)
        train_end = config["TRAIN_REVIEWS_PER_USER"]
        validation_end = train_end + config["VALIDATION_REVIEWS_PER_USER"]
        if len(frame) < validation_end:
            raise ValueError(f"User {user} has fewer than {validation_end} reviews")
        coverage[str(user)] = {"available_reviews": len(frame)}
        for split, begin, end, destination in [
            ("train", 0, train_end, train),
            ("validation", train_end, validation_end, validation),
        ]:
            for offset in range(begin, end, config["CHUNK_REVIEWS"]):
                section = frame.iloc[offset:min(end, offset + config["CHUNK_REVIEWS"])].copy()
                # Next-review targets must stay inside their chronological split.
                section["has_label"] = (section["has_label"].astype(bool) & section["label_review_th"].between(begin + 1, end)).astype(int)
                sample = create_sample(user, section, [], torch.float32, torch.device("cpu"))
                mask = sample.global_labels[:, 4].bool()
                if mask.any():
                    assert sample.label_review_ths[mask].min().item() >= begin + 1
                    assert sample.label_review_ths[mask].max().item() <= end
                destination.append({"user_id": user, "review_start": offset + 1,
                                    "review_end": offset + len(section), "sample": sample})
        print(f"Prepared user {user}: train 1-{train_end}, validation {train_end + 1}-{validation_end}", flush=True)
    def pack(split, chunks):
        batches = []
        for offset in range(0, len(chunks), config["BATCH_CHUNKS"]):
            group = chunks[offset:offset + config["BATCH_CHUNKS"]]
            first = group[0]
            batch = prepare([item["sample"] for item in group],
                            seed=config["SEED"] + first["user_id"] + first["review_start"] - 1)
            batch = replace(batch, sub_gather_lens=[[int(length) for length in group]
                                                   for group in batch.sub_gather_lens])
            batches.append({"chunks": [{key: value for key, value in item.items() if key != "sample"}
                                       for item in group], "batch": batch})
        return batches

    train = pack("train", train)
    validation = pack("validation", validation)
    ordered = [(split, item) for split, items in [("train", train), ("validation", validation)] for item in items]
    if config["BATCH_CHUNKS"] == 1:
        # Preserve the original CPU trial's per-user fingerprint order.
        ordered.sort(key=lambda entry: (entry[1]["chunks"][0]["user_id"], entry[0] == "validation",
                                        entry[1]["chunks"][0]["review_start"]))
    for split, item in ordered:
        fingerprint.update(split.encode())
        for tensor in [item["batch"].start, item["batch"].labels, item["batch"].label_review_th]:
            fingerprint.update(tensor.numpy().tobytes())
    return train, validation, coverage, fingerprint.hexdigest()


def cache_key(config):
    """Invalidate prepared inputs when their sources, data, or selection change."""
    keys = ["USER_START", "USER_END", "TRAIN_REVIEWS_PER_USER", "VALIDATION_REVIEWS_PER_USER",
            "CHUNK_REVIEWS", "BATCH_CHUNKS", "SEED"]
    signature = {key: config[key] for key in keys}
    data = Path(config["DATA_PATH"]).resolve()
    signature["data_path"] = str(data)
    signature["torch"] = str(torch.__version__)
    files = {}
    for table in ["revlogs", "cards", "decks"]:
        for user in range(config["USER_START"], config["USER_END"] + 1):
            partition = data / table / f"user_id={user}"
            # A missing metadata partition uses the root schema/filter fallback.
            source = partition if partition.is_dir() else data / table
            for path in sorted(source.rglob("*.parquet")):
                stat = path.stat()
                files[str(path.relative_to(data))] = [stat.st_size, stat.st_mtime_ns]
    signature["dataset_files"] = files
    signature["source_hashes"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ["scripts/compare.py", "rwkv/data_processing.py", "rwkv/prepare_batch.py",
                     "rwkv/config.py", "rwkv/architecture.py", "rwkv/model/srs_model.py"]
    }
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()


def load_batches(config):
    cache = Path(config["CACHE_PATH"]) if config.get("CACHE_PATH") else None
    key = cache_key(config) if cache else None
    if cache and cache.exists():
        saved = torch.load(cache, map_location="cpu", weights_only=True)
        if saved["key"] == key:
            def unpack(items):
                return [{"chunks": item["chunks"], "batch": PreparedBatch(**item["batch"])}
                        for item in items]
            print(f"Reusing prepared batches from {cache}", flush=True)
            return unpack(saved["train"]), unpack(saved["validation"]), saved["coverage"], saved["fingerprint"], True
    train, validation, coverage, fingerprint = build_batches(config)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        def pack(items):
            return [{"chunks": item["chunks"], "batch": asdict(item["batch"])} for item in items]
        with NamedTemporaryFile(dir=cache.parent, suffix=".pt", delete=False) as temporary:
            pending = Path(temporary.name)
        try:
            torch.save({"key": key, "train": pack(train), "validation": pack(validation),
                        "coverage": coverage, "fingerprint": fingerprint}, pending)
            pending.replace(cache)
        finally:
            pending.unlink(missing_ok=True)
    return train, validation, coverage, fingerprint, False


def require_device(config):
    device = torch.device(config["DEVICE"])
    if device.type not in ["cpu", "cuda"]:
        raise ValueError("DEVICE must be cpu or cuda")
    if config["DTYPE"] not in ["float32", "bfloat16"]:
        raise ValueError("DTYPE must be float32 or bfloat16")
    if device.type == "cpu" and config["DTYPE"] != "float32":
        raise ValueError("CPU comparisons require float32")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Install the cu128 PyTorch build and check your NVIDIA driver.")
        from rwkv.model import RWKV_CUDA
        if RWKV_CUDA is None:
            raise RuntimeError("RWKV CUDA extension is missing. Run just build-cuda after just setup.")
        torch.cuda.set_device(device)
        if config["DTYPE"] == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support BF16 training")
    return device


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def move_batches(items, device, dtype):
    moved = []
    for item in items:
        batch = item["batch"].to(device)
        moved.append({"chunks": item["chunks"], "batch": replace(batch, start=batch.start.to(dtype))})
    return moved


def training_models(model_class, architecture, device, dtype, seed):
    torch.manual_seed(seed)
    master = model_class(architecture).float().to(device)
    if dtype == torch.float32:
        model = master
    else:
        model = model_class(architecture).selective_cast(dtype).to(device)
        model.copy_downcast_(master, dtype)
    # F91 eager_model sets RWKV_NO_JIT before the model modules are imported; the comparison then uses plain nn.Modules.
    assert isinstance(model, torch.jit.ScriptModule) or os.environ.get("RWKV_NO_JIT"), "Comparison requires both models' default JIT path"
    return master, model


def evaluate(model, batches, deferred_metric_reads=False):
    model.eval()
    totals = {"curve_labels": 0, "instant_labels": 0, "curve_log_loss": 0.0,
              "curve_raw_log_loss": 0.0, "instant_log_loss": 0.0,
              "instant_rating_cross_entropy": 0.0}
    pending = []
    def add_metrics(values, ahead_n, imm_n):
        totals["curve_log_loss"] += values[0] * ahead_n
        totals["curve_raw_log_loss"] += values[1] * ahead_n
        totals["instant_rating_cross_entropy"] += values[2] * imm_n
        totals["instant_log_loss"] += values[3]
    with torch.no_grad():
        for item in batches:
            batch = item["batch"]
            stats = model.get_loss(batch)
            if stats is None or not torch.isfinite(stats.average_loss):
                raise RuntimeError("Non-finite evaluation loss")
            mask = stats.is_query.bool() & stats.has_label.bool()
            losses = F.binary_cross_entropy(
                stats.p_imm.clamp(1e-5, 1 - 1e-5), batch.labels[:, :, 2], reduction="none"
            )
            totals["curve_labels"] += stats.ahead_n
            totals["instant_labels"] += stats.imm_n
            if deferred_metric_reads:
                # Keep the original scalar reductions, scalar precision and batch order.
                pending.append((torch.stack((stats.ahead_avg, stats.ahead_raw_avg, stats.imm_avg,
                                             losses[mask].sum())), stats.ahead_n, stats.imm_n))
            else:
                add_metrics((stats.ahead_avg.item(), stats.ahead_raw_avg.item(), stats.imm_avg.item(),
                             losses[mask].sum().item()), stats.ahead_n, stats.imm_n)
        if pending:
            for values, (_, ahead_n, imm_n) in zip(torch.stack([row[0] for row in pending]).tolist(), pending):
                add_metrics(values, ahead_n, imm_n)
    assert totals["curve_labels"] > 0 and totals["instant_labels"] > 0
    for key in ["curve_log_loss", "curve_raw_log_loss"]:
        totals[key] /= totals["curve_labels"]
    for key in ["instant_log_loss", "instant_rating_cross_entropy"]:
        totals[key] /= totals["instant_labels"]
    return totals


def state_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    items = value.values() if isinstance(value, dict) else value
    return sum(state_bytes(item) for item in items)


def run_model(name, model_class, rnn_class, architecture, train, validation, config):
    device = torch.device(config["DEVICE"])
    dtype = getattr(torch, config["DTYPE"])
    master, model = training_models(model_class, architecture, device, dtype, config["SEED"])
    optimizer = get_optimizer(Namespace(PEAK_LR=config["PEAK_LR"]), master)
    initial_head = master.w_linear.weight.detach().clone()
    result = {"parameters": sum(p.numel() for p in model.parameters()),
              "parameter_bytes_float32": sum(p.numel() * p.element_size() for p in master.parameters()),
              "validation_before": evaluate(model, validation), "steps": 0,
              "review_presentations": 0, "skipped_unlabelled_chunks": 0,
              "training_seconds": 0.0, "trained_users": []}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    run_started = time.perf_counter()
    trained_users = set()
    output = Path(config["OUTPUT_DIR"])
    with (output / f"{name}_metrics.jsonl").open("w") as metrics:
        for epoch in range(config["EPOCHS"]):
            for index, item in enumerate(train):
                batch = item["batch"]
                # Dropout RNG is reset separately from common feature-encoding RNG.
                torch.manual_seed(config["SEED"] + epoch * len(train) + index)
                model.train()
                synchronize(device)
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                if model is not master:
                    model.zero_grad(set_to_none=True)
                    model.copy_downcast_(master, dtype)
                stats = model.get_loss(batch)
                if stats is None or not torch.isfinite(stats.average_loss):
                    raise RuntimeError(f"Non-finite {name} training loss")
                if stats.ahead_n + stats.imm_n == 0:
                    result["skipped_unlabelled_chunks"] += len(item["chunks"])
                    continue
                stats.average_loss.backward()
                if model is not master:
                    transfer_child_grad_to_master(master, model)
                norm = torch.nn.utils.clip_grad_norm_(master.parameters(), CLIP, error_if_nonfinite=True)
                lr = config["PEAK_LR"] * min(1.0, (result["steps"] + 1) / config["WARMUP_STEPS"])
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
                synchronize(device)
                result["training_seconds"] += time.perf_counter() - started
                result["steps"] += 1
                result["review_presentations"] += sum(chunk["review_end"] - chunk["review_start"] + 1
                                                      for chunk in item["chunks"])
                labelled = batch.labels[:, :, 4].bool().any(dim=1).tolist()
                trained_users.update(chunk["user_id"] for chunk, has_labels in zip(item["chunks"], labelled) if has_labels)
                row = {"chunks": item["chunks"]}
                row.update(step=result["steps"], epoch=epoch + 1, loss=stats.average_loss.item(),
                           gradient_norm=norm.item(), curve_labels=stats.ahead_n,
                           instant_labels=stats.imm_n, lr=lr)
                metrics.write(json.dumps(row) + "\n")
                if result["steps"] % 20 == 0:
                    print(f"{name}: {result['steps']} steps, {result['training_seconds']:.2f}s optimization time", flush=True)
    synchronize(device)
    result["training_loop_seconds"] = time.perf_counter() - run_started
    if device.type == "cuda":
        result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    assert trained_users == set(range(config["USER_START"], config["USER_END"] + 1))
    result["trained_users"] = sorted(trained_users)
    result["head_max_weight_change"] = (master.w_linear.weight.detach() - initial_head).abs().max().item()
    assert result["head_max_weight_change"] > 0
    if model is not master:
        model.copy_downcast_(master, dtype)
    result["validation_after"] = evaluate(model, validation)
    checkpoint = Path(config["CHECKPOINT_DIR"]) / f"{name}.pth"
    torch.save(master.state_dict(), checkpoint)
    torch.save(optimizer.state_dict(), checkpoint.with_name(f"{name}_optim.pth"))
    result["checkpoint"] = str(checkpoint)
    result["checkpoint_bytes"] = checkpoint.stat().st_size
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert all(torch.isfinite(tensor).all() for tensor in state.values())
    recurrent = rnn_class(architecture).eval()
    recurrent.load_state_dict(state, strict=True)
    states = [None] * 5
    with torch.inference_mode():
        for _ in range(3):
            ahead, weights, ratings, *states = recurrent.review(torch.zeros(1, 92), *states)
            assert torch.isfinite(ahead).all() and torch.isfinite(ratings).all()
            assert torch.allclose(weights.sum(-1), torch.ones(1))
    # review() returns states in card, note, deck, preset, user order.
    result["recurrent_state_bytes_float32"] = dict(zip(
        ["card", "note", "deck", "preset", "user"], map(state_bytes, states)
    ))
    result["reviews_per_training_second"] = result["review_presentations"] / result["training_seconds"]
    print(name, json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/compare-cpu.toml")
    args = parser.parse_args()
    provenance = process_provenance()
    with open(args.config, "rb") as source:
        config = tomllib.load(source)
    config.setdefault("DEVICE", "cpu")
    config.setdefault("DTYPE", "float32")
    config.setdefault("BATCH_CHUNKS", 1)
    assert config["USER_START"] <= config["USER_END"]
    assert all(config[key] > 0 for key in ["TRAIN_REVIEWS_PER_USER", "VALIDATION_REVIEWS_PER_USER",
                                         "CHUNK_REVIEWS", "BATCH_CHUNKS", "EPOCHS", "CPU_THREADS", "WARMUP_STEPS"])
    assert all(layer.state_qmax == float("inf") and layer.state_lowrank_rank == 0
               for _, layer in SMALL_CONFIG.modules), "This comparison does not enable QAT"
    device = require_device(config)
    torch.set_num_threads(config["CPU_THREADS"])
    for key in ["OUTPUT_DIR", "CHECKPOINT_DIR"]:
        Path(config[key]).mkdir(parents=True, exist_ok=True)
    (Path(config["OUTPUT_DIR"]) / "comparison.json").unlink(missing_ok=True)
    start = time.perf_counter()
    train, validation, coverage, fingerprint, cache_hit = load_batches(config)
    preparation_seconds = time.perf_counter() - start
    start = time.perf_counter()
    dtype = getattr(torch, config["DTYPE"])
    train, validation = move_batches(train, device, dtype), move_batches(validation, device, dtype)
    synchronize(device)
    transfer_seconds = time.perf_counter() - start
    results = {}
    for name, model_class, rnn_class, architecture in [
        ("baseline", BaselineModel, BaselineRNN, BASELINE_CONFIG),
        ("expertium_iter36", SmallModel, SmallRNN, SMALL_CONFIG),
    ]:
        results[name] = run_model(name, model_class, rnn_class, architecture, train, validation, config)
    baseline, small = results.values()
    assert baseline["parameters"] == 2_762_884 and small["parameters"] == 192_800
    assert baseline["steps"] == small["steps"]
    assert baseline["review_presentations"] == small["review_presentations"]
    for phase in ["validation_before", "validation_after"]:
        for key in ["curve_labels", "instant_labels"]:
            assert baseline[phase][key] == small[phase][key]
    summary = {"status": "complete", "config": config, "torch": str(torch.__version__),
               "provenance": provenance,
               "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
               "cuda_runtime": torch.version.cuda, "dtype": config["DTYPE"], "cache_hit": cache_hit,
               "batch_transfer_seconds": transfer_seconds, "coverage": coverage, "batch_sha256": fingerprint,
               "preparation_seconds": preparation_seconds, "models": results,
               "parameter_reduction_factor": baseline["parameters"] / small["parameters"],
               "training_speedup_factor": baseline["training_seconds"] / small["training_seconds"],
               "small_minus_baseline_validation_log_loss": {
                   key: small["validation_after"][key] - baseline["validation_after"][key]
                   for key in ["curve_log_loss", "instant_log_loss"]}}
    (Path(config["OUTPUT_DIR"]) / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Comparison complete", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
