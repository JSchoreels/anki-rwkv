"""Train the baseline on real user histories with CUDA and power monitoring."""

from argparse import ArgumentParser, Namespace
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import numpy as np

from gpu_power import limited_gpu
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.data_processing import get_rwkv_data
from rwkv.model import RWKV_CUDA
from rwkv.model.srs_model import PreparedBatch, SrsRWKV
from rwkv.train_rwkv import CLIP, get_optimizer, transfer_child_grad_to_master
from scripts.trial_batches import batches, preparation_pool
from scripts.experiment_provenance import process_provenance


def checkpoint_containers(value):
    """Keep checkpoint metadata compatible with PyTorch's safe tensor loader."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: checkpoint_containers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [checkpoint_containers(item) for item in value]
    return value


def build_model(backend, device):
    if backend not in ("cuda", "nvrtc", "reference"):
        raise ValueError("WKV_BACKEND must be 'cuda', 'nvrtc', or 'reference'")
    if backend == "nvrtc":
        from rwkv.model.nvrtc import register
        register()
    if backend == "cuda" and RWKV_CUDA is None:
        raise RuntimeError("RWKV_CUDA is not compiled; build it or explicitly select WKV_BACKEND='reference'")
    architecture = replace(DEFAULT_ANKI_RWKV_CONFIG, modules=[
        (name, replace(config, use_reference_wkv=backend == "reference"))
        for name, config in DEFAULT_ANKI_RWKV_CONFIG.modules
    ])
    return SrsRWKV(architecture).float().to(device)


def training_models(config, device):
    dtype = getattr(torch, config.get("DTYPE", "float32"))
    master = build_model(config["WKV_BACKEND"], device)
    if dtype == torch.float32:
        model = master
    else:
        model = build_model(config["WKV_BACKEND"], device).selective_cast(dtype)
        model.copy_downcast_(master, dtype)
    return master, model, dtype


def preflight(config):
    if not 1 <= config["USER_START"] <= config["USER_END"]:
        raise ValueError("Invalid user range")
    for name in ("CHUNK_REVIEWS", "EPOCHS", "CPU_THREADS", "WARMUP_STEPS", "CHECKPOINT_EVERY_STEPS"):
        if config[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if config["REVIEWS_PER_USER"] < 0:
        raise ValueError("REVIEWS_PER_USER must be nonnegative")
    if config.get("BATCH_CHUNKS", 1) <= 0:
        raise ValueError("BATCH_CHUNKS must be positive")
    if config.get("PREFETCH_WORKERS", 0) < 0 or config.get("PREFETCH_BATCHES", 4) <= 0:
        raise ValueError("Invalid prefetch configuration")
    if config.get("DTYPE", "float32") not in ("float32", "bfloat16"):
        raise ValueError("DTYPE must be float32 or bfloat16")
    if not 0 < config.get("GPU_MEMORY_FRACTION", 1.0) <= 1:
        raise ValueError("GPU_MEMORY_FRACTION must be in (0, 1]")
    device = torch.device(config["DEVICE"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This trial requires CUDA-enabled PyTorch and an NVIDIA GPU")
    if config["WKV_BACKEND"] not in ("reference", "nvrtc", "cuda"):
        raise ValueError("WKV_BACKEND must be 'cuda', 'nvrtc', or 'reference'")
    if config["WKV_BACKEND"] == "nvrtc" and config["CHUNK_REVIEWS"] > 191:
        raise ValueError("NVRTC chunks must have at most 191 reviews (383 timesteps including queries)")
    data = Path(config["DATA_PATH"])
    missing = [str(data / "revlogs" / f"user_id={user}")
               for user in range(config["USER_START"], config["USER_END"] + 1)
               if not list((data / "revlogs" / f"user_id={user}").glob("*.parquet"))]
    missing.extend(str(data / table) for table in ("cards", "decks")
                   if not list((data / table).glob("**/*.parquet")))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} user partitions; first: {missing[0]}. Download the selected dataset users first.")
    return device


def run(config, resume=False):
    provenance = process_provenance()
    device = preflight(config)
    torch.set_num_threads(config["CPU_THREADS"])
    torch.manual_seed(config["SEED"])
    torch.cuda.set_device(device)
    # Bound the Torch allocation cache as well as live tensors. Dynamic history
    # shapes otherwise retain enough idle allocations to trigger WDDM paging.
    torch.cuda.set_per_process_memory_fraction(config.get("GPU_MEMORY_FRACTION", 1.0), device)
    output, checkpoint = Path(config["OUTPUT_DIR"]), Path(config["CHECKPOINT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir(parents=True, exist_ok=True)
    master, model, dtype = training_models(config, device)
    optimizer = get_optimizer(Namespace(PEAK_LR=config["PEAK_LR"]), master)
    initial_head = master.w_linear.weight.detach().clone()
    summary = {"config": config, "torch": str(torch.__version__), "device": str(device),
               "provenance": provenance,
               "gpu": torch.cuda.get_device_name(device), "dtype": str(dtype), "master_dtype": "float32",
               "parameters": sum(p.numel() for p in master.parameters()), "users": {},
               "steps": 0, "review_presentations": 0, "selected_review_presentations": 0,
               "skipped_unlabelled_chunks": 0, "elapsed_seconds": 0.0, "status": "running"}
    position = [0, config["USER_START"], 0]
    probe = None
    latest = checkpoint / "latest.pth"
    if resume:
        saved = torch.load(latest, weights_only=True, map_location="cpu")
        previous_config = saved["summary"]["config"]
        # Allow explicitly recorded batching/backend/power transitions. Data,
        # architecture and optimizer settings must remain the same.
        operational = {"WKV_BACKEND", "POWER_LIMIT_WATTS", "BATCH_CHUNKS", "DTYPE", "PREFETCH_WORKERS", "PREFETCH_BATCHES", "GPU_MEMORY_FRACTION"}
        if {k: v for k, v in previous_config.items() if k not in operational} != {
                k: v for k, v in config.items() if k not in operational}:
            raise ValueError("Resume configuration must match except batching, backend, precision, prefetch, and power")
        master.load_state_dict(saved["model"])
        if model is not master:
            model.copy_downcast_(master, dtype)
        optimizer.load_state_dict(saved["optimizer"])
        summary, position = saved["summary"], saved["position"]
        if "error" in summary:
            summary.setdefault("recovered_errors", []).append({"step": summary["steps"],
                "position": list(position), "error": summary.pop("error")})
        if previous_config != config:
            summary.setdefault("configuration_changes", []).append({"step": summary["steps"],
                "previous": previous_config, "current": config})
        summary["config"] = config
        summary["dtype"], summary["master_dtype"] = str(dtype), "float32"
        summary["status"] = "running"
        initial_head = saved["initial_head"].to(device)
        if saved["probe"] is not None:
            probe = PreparedBatch(**saved["probe"]).to(device)
            probe.start = probe.start.to(dtype)
        if "rng_cpu" in saved:
            torch.set_rng_state(saved["rng_cpu"])
            torch.cuda.set_rng_state(saved["rng_cuda"], device)
    elif latest.exists():
        raise FileExistsError(f"{latest} already exists; use --resume or choose new output directories")
    summary.setdefault("source_sessions", []).append({"start_step": summary["steps"],
        "resume": resume, "provenance": provenance})
    previous_elapsed = summary["elapsed_seconds"]
    started = time.monotonic()

    def save(status):
        summary.update(status=status, elapsed_seconds=previous_elapsed + time.monotonic() - started,
                       peak_gpu_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                       next_position=position)
        payload = {"model": master.state_dict(), "optimizer": optimizer.state_dict(),
                   "summary": summary, "position": position, "initial_head": initial_head.cpu(),
                   "probe": checkpoint_containers(asdict(probe.to("cpu"))) if probe is not None else None,
                   "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(device)}
        temporary = latest.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(latest)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"GPU training: {summary['parameters']:,} parameters; users {config['USER_START']}-{config['USER_END']}; "
          f"{config['WKV_BACKEND']} recurrence; full histories={config['REVIEWS_PER_USER'] == 0}", flush=True)
    try:
        with limited_gpu(device.index or 0, config["POWER_LIMIT_WATTS"], output), preparation_pool(config.get("PREFETCH_WORKERS", 0)) as pool:
            save("running")
            with (output / "metrics.jsonl").open("a" if resume else "w", encoding="utf-8") as metrics:
                for epoch in range(position[0], config["EPOCHS"]):
                    first_user = position[1] if epoch == position[0] else config["USER_START"]
                    for user in range(first_user, config["USER_END"] + 1):
                        offset_start = position[2] if [epoch, user] == position[:2] else 0
                        frame = get_rwkv_data(Path(config["DATA_PATH"]), user)
                        cap = config["REVIEWS_PER_USER"]
                        used = min(len(frame), cap) if cap else len(frame)
                        if not used:
                            raise ValueError(f"User {user} has no reviews")
                        summary["users"].setdefault(str(user), {"available_reviews": len(frame),
                            "selected_reviews_per_epoch": used, "optimizer_steps": 0})
                        print(f"User {user}: {used:,} reviews; starting at {offset_start:,}", flush=True)
                        for item in batches(frame, user, epoch, offset_start, config, pool):
                            offset = item.offset
                            position = [epoch, user, offset]
                            if (output / "STOP").exists():
                                save("stopped")
                                print("STOP file found; checkpoint saved", flush=True)
                                return summary
                            training_started = time.perf_counter()
                            # Worker preparation must not alter the parent's dropout RNG.
                            # This is the CUDA seed the original synchronous prepare set.
                            torch.manual_seed(item.seed)
                            batch = item.batch.to(device)
                            batch.start = batch.start.to(dtype)
                            model.train()
                            optimizer.zero_grad(set_to_none=True)
                            if model is not master:
                                model.zero_grad(set_to_none=True)
                                model.copy_downcast_(master, dtype)
                            stats = model.get_loss(batch)
                            if stats is None or not torch.isfinite(stats.average_loss):
                                raise RuntimeError(f"Non-finite loss for user {user}, offset {offset}")
                            if stats.ahead_n + stats.imm_n == 0:
                                summary["skipped_unlabelled_chunks"] += item.chunks
                            else:
                                if probe is None:
                                    probe = batch
                                    model.eval()
                                    with torch.no_grad():
                                        summary["fixed_probe_loss_before"] = model.get_loss(probe).average_loss.item()
                                    model.train()
                                stats.average_loss.backward()
                                if model is not master:
                                    transfer_child_grad_to_master(master, model)
                                norm = torch.nn.utils.clip_grad_norm_(master.parameters(), CLIP, error_if_nonfinite=True)
                                lr = config["PEAK_LR"] * min(1.0, (summary["steps"] + 1) / config["WARMUP_STEPS"])
                                for group in optimizer.param_groups:
                                    group["lr"] = lr
                                optimizer.step()
                                summary["steps"] += 1
                                summary["users"][str(user)]["optimizer_steps"] += 1
                                summary["review_presentations"] += item.reviews
                                row = {"step": summary["steps"], "epoch": epoch + 1, "user_id": user,
                                    "review_start": offset + 1, "review_end": offset + item.reviews,
                                    "loss": stats.average_loss.item(), "curve_loss": stats.ahead_avg.item(),
                                    "instant_loss": stats.imm_avg.item(), "curve_labels": stats.ahead_n,
                                    "instant_labels": stats.imm_n, "gradient_norm": norm.item(), "lr": lr,
                                    "batch_chunks": item.chunks, "dtype": str(dtype),
                                    "preparation_seconds": item.preparation_seconds,
                                    "preparation_wait_seconds": item.wait_seconds,
                                    "elapsed_seconds": previous_elapsed + time.monotonic() - started}
                                row["training_seconds"] = time.perf_counter() - training_started
                                metrics.write(json.dumps(row) + "\n")
                                metrics.flush()
                                if summary["steps"] == 1 or summary["steps"] % 10 == 0:
                                    print(f"Step {row['step']}: user={user}, loss={row['loss']:.5f}, elapsed={row['elapsed_seconds']:.1f}s", flush=True)
                            summary["selected_review_presentations"] += item.reviews
                            position = [epoch, user, offset + item.reviews]
                            del stats, batch
                            if summary["steps"] % config["CHECKPOINT_EVERY_STEPS"] == 0:
                                save("running")
                        position = [epoch, user + 1, 0]
                        save("running")
                        del frame
                        torch.cuda.empty_cache()
                    position = [epoch + 1, config["USER_START"], 0]
                if probe is None or any(item["optimizer_steps"] == 0 for item in summary["users"].values()):
                    raise RuntimeError("A selected user contributed no supervised chunks")
                model.eval()
                if model is not master:
                    model.copy_downcast_(master, dtype)
                with torch.no_grad():
                    summary["fixed_probe_loss_after"] = model.get_loss(probe).average_loss.item()
                summary["head_max_weight_change"] = (master.w_linear.weight.detach() - initial_head).abs().max().item()
                if not summary["head_max_weight_change"] > 0:
                    raise RuntimeError("Optimizer did not change the curve-head weights")
                model_path = checkpoint / "rwkv_gpu_trial.pth"
                torch.save(master.state_dict(), model_path)
                torch.save(optimizer.state_dict(), checkpoint / "rwkv_gpu_trial_optim.pth")
                summary["checkpoint"] = str(model_path)
                save("complete")
    except BaseException as exc:
        summary["error"] = str(exc)
        save("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
        raise
    print(json.dumps({key: value for key, value in summary.items() if key != "users"}, indent=2), flush=True)
    return summary


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train-gpu-trial.toml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    with open(args.config, "rb") as source:
        config = tomllib.load(source)
    run(config, resume=args.resume)


if __name__ == "__main__":
    main()
