"""Measured F09/F10/F11/F13 runner; proposed plans never authorize validation."""

from argparse import ArgumentParser, Namespace
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import gc
import hashlib
import importlib
import json
import math
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


def architecture_metadata(value):
    """Encode disabled infinite QAT configuration, never sanitize measurements."""
    if isinstance(value, dict):
        return {key: architecture_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [architecture_metadata(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def batch_fingerprint(batch, digest):
    digest.update(str(batch.num_data).encode())
    digest.update(json.dumps([[int(length) for length in group] for group in batch.sub_gather_lens]).encode())
    tensors = [batch.start, batch.labels, batch.label_review_th,
               *(tensor for name in ("sub_gather", "time_shift_selects", "skips")
                 for group in getattr(batch, name) for tensor in group)]
    for tensor in tensors:
        digest.update(f"{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.contiguous().numpy().tobytes())


class FutureSourceWatch:
    def __init__(self, provenance):
        from scripts.run_gpu_batch1 import SourceWatch
        self.base = SourceWatch(provenance)
        with ZipFile(ROOT / provenance["source_archive"]) as archive:
            manifest = json.loads(archive.read("source-manifest.json"))["files"]
        names = {"scripts/train_gpu_future.py", "scripts/gpu_future_models.py", "scripts/gpu_training_targets.py",
                 "scripts/gpu_fp8.py", "scripts/gpu_row_tiles.py", "scripts/kernels/rwkv_row_tiles.cu"}
        self.hashes = {name: manifest[name] for name in names}
        self.stats = {}

    def check(self):
        self.base.check()
        for name, digest in self.hashes.items():
            path = ROOT / name
            stat = path.stat()
            identity = stat.st_mtime_ns, stat.st_size
            if self.stats.get(name) != identity:
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise RuntimeError(f"Experiment source changed: {name}; restart from preserved initial state")
                self.stats[name] = identity


def build_models(config, device):
    import torch
    from scripts.gpu_future_models import TaskModel, capacity_variant, load_pretrained, python_model_class
    from scripts.gpu_fp8 import install_fp8, probe_fp8
    from rwkv.model.srs_model import is_excluded
    from rwkv.train_rwkv import get_optimizer
    package = "rwkv" if config["model"] == "baseline" else "expertium"
    architecture = importlib.import_module(package + ".architecture").DEFAULT_ANKI_RWKV_CONFIG
    architecture = capacity_variant(architecture, config.get("width_heads"), config.get("remove_layers", 0))
    model_class = python_model_class(config["model"])
    task = config.get("task", "joint")
    weights = dict(curve_weight=config.get("curve_weight", 1.0), instant_weight=config.get("instant_weight", 1.0))
    torch.manual_seed(config.get("seed", 2025))
    master = TaskModel(model_class(architecture).float(), task, **weights).to(device)
    metadata = {"architecture": architecture_metadata(asdict(architecture)), "task": task, "objective_weights": weights,
                "control_requirement": "same isolated Python model implementation", "master_dtype": "float32"}
    if config.get("pretrained_checkpoint"):
        metadata["pretraining"] = load_pretrained(master, config["pretrained_checkpoint"], config["pretrained_manifest"],
                                                  config["forbidden_pretraining_users"])
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    child = TaskModel(model_class(architecture).float(), task, **weights).to(device)
    for name, param in child.named_parameters():
        if not is_excluded(name):
            param.data = param.data.to(dtype)
    copy_parameters(master, child)
    if config.get("fp8"):
        if device.type != "cuda":
            raise ValueError("CPU runs cannot claim FP8 execution")
        metadata["fp8_probe"] = probe_fp8(device)
        if metadata["fp8_probe"]["status"] != "supported":
            raise RuntimeError(metadata["fp8_probe"].get("reason", "Installed backend does not support FP8"))
        metadata["fp8"] = install_fp8(child)
    optimizer = get_optimizer(Namespace(PEAK_LR=config.get("peak_lr", 7e-4)), master)
    metadata.update(parameters=sum(p.numel() for p in master.parameters()), child_dtype=str(dtype))
    return master, child, optimizer, dtype, metadata


def copy_parameters(master, child):
    import torch
    with torch.no_grad():
        sources = dict(master.named_parameters())
        for name, param in child.named_parameters():
            param.copy_(sources[name])


def transfer_gradients(master, child):
    sources = dict(child.named_parameters())
    for name, param in master.named_parameters():
        grad = sources[name].grad
        param.grad = grad.detach().to(param.dtype).clone() if grad is not None else None


def evaluate_batches(model, batches):
    import torch
    import torch.nn.functional as F
    model.eval()
    counts = {"curve": 0, "instant": 0}
    sums = {"curve": 0.0, "instant": 0.0}
    ratings_sum = 0.0
    with torch.no_grad():
        for batch in batches:
            stats = model.get_loss(batch)
            if not torch.isfinite(stats.average_loss):
                raise RuntimeError("Nonfinite held-out objective")
            for task, prediction in (("curve", stats.p_curve), ("instant", stats.p_imm)):
                if prediction is None:
                    continue
                mask = stats.has_label.bool() & (stats.is_query.bool() if task == "instant" else ~stats.is_query.bool())
                loss = (F.binary_cross_entropy_with_logits(stats.curve_logits, batch.labels[..., 2], reduction="none")
                        if task == "curve" else F.binary_cross_entropy(prediction, batch.labels[..., 2], reduction="none"))
                sums[task] += float(loss[mask].sum())
                counts[task] += int(mask.sum())
                if task == "instant":
                    ratings_sum += float(stats.instant_rating_loss) * int(mask.sum())
    result = {task + "_labels": counts[task] for task in counts}
    for task in counts:
        result[task + "_log_loss"] = sums[task] / counts[task] if counts[task] else None
    result["instant_rating_cross_entropy"] = ratings_sum / counts["instant"] if counts["instant"] else None
    return result


def aggregate_users(rows):
    result = {}
    for task in ("curve", "instant"):
        supported = [row for row in rows if row[task + "_log_loss"] is not None]
        count = sum(row[task + "_labels"] for row in supported)
        result[task + "_labels"] = count
        result[task + "_evaluated_users"] = len(supported)
        result[task + "_log_loss_label_weighted"] = (sum(row[task + "_log_loss"] * row[task + "_labels"] for row in supported) / count if count else None)
        result[task + "_log_loss_user_average"] = (sum(row[task + "_log_loss"] for row in supported) / len(supported) if supported else None)
    return result


def train(config, output, provider, evaluator, provenance, total_steps):
    """Provider includes loading/preparation shutdown; checkpoint validation is explicit."""
    import torch
    from rwkv.train_rwkv import CLIP
    from scripts.gpu_training_targets import AccuracyStopper, learning_rate
    from scripts.gpu_fp8 import projection_counts
    from scripts.gpu_row_tiles import row_tiled_backend
    started = time.perf_counter()
    watch = FutureSourceWatch(provenance)
    watch.check()
    device = torch.device(config.get("device", "cuda:0"))
    torch.set_num_threads(config.get("cpu_threads", 1 if device.type == "cpu" else 4))
    if device.type == "cuda":
        from rwkv.model import nvrtc
        torch.cuda.set_per_process_memory_fraction(0.72, device)
        nvrtc.register()
        nvrtc.set_checkpoint_length(config.get("checkpoint_tile", 32))
        nvrtc.set_raw_pointer_launch(config.get("raw_pointers", True))
    if config.get("row_tiles") and (config.get("checkpoint_every", 0) or config.get("checkpoint_steps")):
        raise ValueError("F09 uses common-backend final evaluation; intermediate checkpoints are for F11")
    master, child, optimizer, dtype, metadata = build_models(config, device)
    from scripts.run_gpu_batch1 import tensor_hash
    metadata["initial_parameter_sha256"] = tensor_hash(master.state_dict())
    backbone = {name: value for name, value in master.state_dict().items()
                if not name.startswith(("head_", "w_linear.", "ahead_linear.", "p_linear."))}
    metadata["initial_backbone_sha256"] = tensor_hash(backbone)
    if device.type == "cuda":
        compilation_start = time.perf_counter()
        if config.get("row_tiles"):
            from scripts.gpu_row_tiles import compile_module
            compile_module(device, dtype, config["row_tiles"], config.get("checkpoint_tile", 8))
        else:
            nvrtc.module(device, dtype)
        nvrtc.set_checkpoint_length(32)
        nvrtc.module(device, torch.float32)
        nvrtc.set_checkpoint_length(config.get("checkpoint_tile", 32))
        metadata["backend_compilation_seconds"] = time.perf_counter() - compilation_start
    initial = {name: parameter.detach().clone() for name, parameter in master.named_parameters() if name in ("w_linear.weight", "p_linear.weight")}
    report = {"status": "running", "provenance": provenance, "config": config, "model": metadata,
              "torch": str(torch.__version__), "cuda": torch.version.cuda, "updates": 0,
              "raw_review_presentations": 0, "curve_training_labels": 0, "instant_training_labels": 0,
              "unlabelled_batches": 0, "history": [], "coverage": {}, "validation_seconds": 0.0,
              "float32_matmul_precision": torch.get_float32_matmul_precision(),
              "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "timing_scope": "training includes loading, worker startup/shutdown, preparation, copies, gradients, updates and logging; validation excluded from training but included in elapsed_seconds"}
    if device.type == "cuda":
        report.update(gpu=torch.cuda.get_device_name(device), capability=list(torch.cuda.get_device_capability(device)))
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    report["model_setup_seconds"] = time.perf_counter() - started
    stopper = AccuracyStopper(config["targets"], config.get("target_confirmations", 2)) if config.get("targets") else None
    if stopper and config.get("task", "joint") != "joint":
        raise ValueError("F11 target stopping requires both supported tasks; specialist comparisons use fixed exposure")
    trained_start = time.perf_counter()
    paused = 0.0

    def checkpoint(final=False):
        nonlocal paused
        watch.check()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_seconds = time.perf_counter() - trained_start - paused
        validation_start = time.perf_counter()
        per_user = evaluator(master, device)
        torch.save(master.state_dict(), output / f"checkpoint-{report['updates']:06d}.pth")
        validation_cost = time.perf_counter() - validation_start
        paused += validation_cost
        report["validation_seconds"] += validation_cost
        metrics = aggregate_users(per_user)
        expected_users = config.get("evaluation_users", len(per_user))
        for task in ("curve", "instant"):
            if config.get("task", "joint") in ("joint", task) and metrics[task + "_evaluated_users"] != expected_users:
                raise RuntimeError("A supported task is missing held-out labels for a selected evaluation user")
        entry = {"update": report["updates"], "raw_review_presentations": report["raw_review_presentations"],
                 "training_seconds": training_seconds, "elapsed_seconds": time.perf_counter() - started,
                 "validation_seconds": validation_cost, "metrics": metrics, "per_user": per_user,
                 "gradient_training_users": sum(row["updates"] > 0 for row in report["coverage"].values())}
        report["history"].append(entry)
        atomic_json(output / "result.json", report)
        reached = stopper.observe(metrics, entry["elapsed_seconds"]) if stopper else False
        print(f"Checkpoint {entry['update']}: {report['raw_review_presentations']} reviews; target={reached}", flush=True)
        return reached

    try:
        backend = row_tiled_backend(config["row_tiles"], config.get("checkpoint_tile", 8)) if config.get("row_tiles") else nullcontext()
        with backend, provider() as items, (output / "updates.jsonl").open("w", encoding="utf-8") as log:
            for user, item in items:
                watch.check()
                if (output / "STOP").exists():
                    raise InterruptedError("STOP requested; preserve this attempt and restart fresh")
                batch = item.batch.to(device)
                batch.start = batch.start.to(dtype)
                labels = item.batch.labels
                curve_n = int(((1 - labels[..., 6]) * labels[..., 4]).sum())
                instant_n = int((labels[..., 6] * labels[..., 4]).sum())
                coverage = report["coverage"].setdefault(str(user), {"reviews": 0, "batches": 0, "updates": 0})
                if item.offset != coverage["reviews"]:
                    raise RuntimeError("Review coverage gap or repeated offset")
                coverage["reviews"] += item.reviews
                coverage["batches"] += 1
                report["raw_review_presentations"] += item.reviews
                report["curve_training_labels"] += curve_n
                report["instant_training_labels"] += instant_n
                supported_n = (curve_n if config.get("task") == "curve" else instant_n if config.get("task") == "instant" else curve_n + instant_n)
                if not supported_n:
                    report["unlabelled_batches"] += 1
                    continue
                torch.manual_seed(item.seed)
                child.train()
                child.zero_grad(set_to_none=True)
                optimizer.zero_grad(set_to_none=True)
                copy_parameters(master, child)
                stats = child.get_loss(batch)
                if not torch.isfinite(stats.average_loss):
                    raise RuntimeError("Nonfinite training objective")
                stats.average_loss.backward()
                transfer_gradients(master, child)
                norm = torch.nn.utils.clip_grad_norm_(master.parameters(), CLIP, error_if_nonfinite=True)
                step = report["updates"] + 1
                lr = learning_rate(step, total_steps, config.get("peak_lr", 7e-4), config.get("warmup", 5), config.get("schedule", "constant"))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
                report["updates"] = step
                coverage["updates"] += 1
                log.write(json.dumps({"update": step, "user_id": user, "offset": item.offset, "reviews": item.reviews,
                                      "seed": item.seed, "loss": float(stats.average_loss.detach()), "gradient_norm": float(norm),
                                      "lr": lr, "curve_labels": curve_n, "instant_labels": instant_n}) + "\n")
                log.flush()
                if ((config.get("checkpoint_every") and step % config["checkpoint_every"] == 0)
                        or step in config.get("checkpoint_steps", [])):
                    if checkpoint():
                        report["stop_reason"] = ("synthetic target confirmation" if config.get("synthetic_only") else
                                                 "predeclared target confirmed on development reviews")
                        break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        report["training_seconds"] = time.perf_counter() - trained_start - paused
        report["reviews_per_second"] = report["raw_review_presentations"] / report["training_seconds"]
        report["fp8_counts"] = projection_counts(child)
        if device.type == "cuda":
            report.update(peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                          peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20)
            nvrtc.set_checkpoint_length(32)
        report["maximum_supported_head_weight_change"] = max(float((dict(master.named_parameters())[name].detach() - value).abs().max()) for name, value in initial.items())
        child = optimizer = batch = item = stats = None
        master.zero_grad(set_to_none=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if not report["history"] or report["history"][-1]["update"] != report["updates"]:
            checkpoint(final=True)
        report["evaluation"] = report["history"][-1]["metrics"]
        report["gradient_training_users"] = report["history"][-1]["gradient_training_users"]
        report["first_target_observation_seconds"] = stopper.first_hit_seconds if stopper else None
        report["trained_parameter_sha256"] = tensor_hash(master.state_dict())
        if not report.get("stop_reason") and config.get("expected_training_reviews") is not None:
            if report["raw_review_presentations"] != config["expected_training_reviews"]:
                raise RuntimeError("Incomplete selected training exposure")
        report["status"] = "complete"
        watch.check()
    except Exception as exc:
        report.update(status="stopped" if isinstance(exc, InterruptedError) else "failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(output / "result.json", report)
    return report


def proposed_plan(cohort, output):
    cases = []
    for model in ("baseline", "expertium"):
        chunks = 128 if model == "baseline" else 256
        total_updates = sum(math.ceil((row["train_range"][1] - row["train_range"][0]) / (128 * chunks)) for row in cohort["rows"])
        checkpoints = sorted({math.ceil(total_updates * fraction / 4) for fraction in (1, 2, 3, 4)})
        common = {"model": model, "seed": 2025, "schedule": "constant", "checkpoint_every": 0, "checkpoint_steps": checkpoints,
                  "peak_lr": 7e-4, "warmup": 5, "task": "joint", "checkpoint_tile": 32, "raw_pointers": True}
        for name, settings in (("control", {}), ("F09-control-tile8", {"checkpoint_tile": 8, "checkpoint_steps": []}),
                               ("F09-rows2", {"row_tiles": 2, "checkpoint_tile": 8, "checkpoint_steps": []}),
                               ("F09-rows4", {"row_tiles": 4, "checkpoint_tile": 8, "checkpoint_steps": []}),
                               ("F09-control-tile8-repeat", {"checkpoint_tile": 8, "checkpoint_steps": []}),
                               ("F10-fp8", {"fp8": True}), ("F11-depth", {"remove_layers": 1}),
                               ("F11-cosine", {"schedule": "cosine"}), ("F11-wsd", {"schedule": "wsd"}),
                               ("control-repeat", {})):
            cases.append({"name": model + "/" + name, "config": {**common, **settings}})
        if model == "baseline":
            cases.append({"name": model + "/F11-width64", "config": {**common, "width_heads": 2}})
        if model == "expertium":
            for name, settings in (("F13-curve", {"task": "curve"}), ("F13-instant", {"task": "instant"}),
                                   ("F13-joint-instant-weight", {"curve_weight": 0.5, "instant_weight": 1.5}),
                                   ("F13-joint-curve-weight", {"curve_weight": 1.5, "instant_weight": 0.5})):
                cases.append({"name": model + "/" + name, "config": {**common, **settings}})
    plan = {"authorization": {"status": "pending explicit user approval; first-batch approval applies only to F01–F08"},
            "cohort": cohort, "cases": cases,
            "evaluation": "common original preparation and FP32 master with original recurrence; later-review development holdout; record how many users have received gradient training at each checkpoint; reserved final test untouched",
            "acceptance": {"status": "proposed; requires user approval with this batch", "max_binary_logloss_regression": 0.001,
                           "quality": "paired-user bootstrap upper bounds for supported tasks, weighted and user-averaged; 10000 resamples; Bonferroni 64 comparisons",
                           "controls": "F09 uses tile8 and tile8-repeat with identical raw pointer launches; others use control and control-repeat with identical checkpoint cadence",
                           "speed": "complete training and end-to-end checkpoint cost faster than both matched controls; prior pretraining cost included where available; unknown cost remains inconclusive",
                           "scope": "one seed; 500 later-review evaluation users; all selected training reviews; independent final cohort unscored",
                           "F11_equal_time": "actual checkpoints at common time/review limits, with gaps and missing comparisons explicit",
                           "F13_cost": "individual costs and summed Curve+Instant specialist training/validation cost versus joint control"},
            "deferred_cases": [{"track": "F11-pretrained", "requires": "checkpoint and manifest with prior exposure/provenance; no overlap with selected or reserved users"},
                               {"track": "F11-target-stop", "requires": "predeclared four-score targets and monitoring cadence; matched cadence control"}],
            "minimum_cuda_checks": "scripts/check_gpu_future.py --cuda after first-batch GPU worker exits"}
    atomic_json(output / "proposed-plan.json", plan)
    return plan


def synthetic_run(config, output, provenance):
    import torch
    from rwkv.model.srs_model import PreparedBatch
    from scripts.trial_batches import TrialBatch
    torch.manual_seed(2025)
    labels = torch.tensor([[[86400, 0, 0, 1, 1, 0, 0], [86400, 0, 1, 3, 1, 0, 1],
                            [172800, 0, 1, 4, 1, 0, 0], [172800, 0, 1, 2, 1, 0, 1]]], dtype=torch.float32)
    batch = PreparedBatch(1, torch.randn(4, 92), [[torch.arange(4)] for _ in range(5)], [[4] for _ in range(5)],
                          [[torch.tensor([0, 0, 1, 2])] for _ in range(5)],
                          [[torch.tensor([False, True, False, True])] for _ in range(5)], labels, torch.arange(4).view(1, -1))
    @contextmanager
    def provider():
        yield iter([(17, TrialBatch(batch, 0, 2, 1, 2042, 0)), (17, TrialBatch(batch, 2, 2, 1, 2044, 0))])
    def evaluator(model, device):
        return [{"user_id": 17, **evaluate_batches(model, [batch.to(device)])}]
    config = {**config, "device": "cpu", "synthetic_only": True, "checkpoint_every": 1, "warmup": 1}
    return train(config, output, provider, evaluator, provenance, 2)


def real_run(plan, case, output, provenance):
    import psutil
    state = ROOT / "artifacts/benchmarks/gpu-batch1-500/process.json"
    lock = ROOT / "artifacts/gpu-batch1.lock"
    if (state.exists() and json.loads(state.read_text()).get("status") == "running") or (lock.exists() and psutil.pid_exists(json.loads(lock.read_text())["pid"])):
        raise RuntimeError("First-batch worker must be verified terminal before a new GPU workload")
    if plan["authorization"].get("status") != "approved" or not plan["authorization"].get("user_request"):
        raise ValueError("This proposed plan has no explicit broader-run authorization")
    if plan["acceptance"].get("status") != "approved":
        raise ValueError("Acceptance criteria must be approved before the broader run")
    import torch
    cohort, config = plan["cohort"], case["config"]
    rows = cohort["rows"]
    users = {row["user_id"] for row in rows}
    if len(users) != cohort["evaluation_users"] or len(users) != len(rows) or len(users) < 500:
        raise ValueError("Broader validation requires at least 500 distinct users")
    if users & set(cohort["independent_final_test_users"]) or any(not 1 <= user <= 5000 for user in users):
        raise ValueError("Invalid cohort or overlap with the reserved final test")
    os.environ["GPU_BATCH1_DATA_PATH"] = cohort["dataset_path"]
    os.environ["CUPY_CACHE_DIR"] = str(ROOT / "artifacts/cache/cupy-cache")
    from scripts.gpu_batch1 import DATA, make_samples, split_frame
    if json.loads((DATA / "download_manifest.json").read_text())["revision"] != cohort["dataset_revision"]:
        raise ValueError("Dataset revision differs from the approved plan")
    from rwkv.data_processing import get_rwkv_data
    from rwkv.prepare_batch import prepare
    from scripts.trial_batches import batches, preparation_pool
    chunks, workers = (128, 4) if config["model"] == "baseline" else (256, 6)
    batch_config = {"CHUNK_REVIEWS": 128, "BATCH_CHUNKS": chunks, "REVIEWS_PER_USER": 0,
                    "PREFETCH_WORKERS": workers, "PREFETCH_BATCHES": workers * 2, "SEED": config["seed"]}
    config = {**config, "dataset_path": cohort["dataset_path"], "dataset_revision": cohort["dataset_revision"],
              "evaluation_users": len(rows), "epochs": 1, "batch_chunks": chunks, "preparation_workers": workers}
    config["expected_training_reviews"] = sum(row["train_range"][1] - row["train_range"][0] for row in rows)
    if config.get("pretrained_checkpoint"):
        config["forbidden_pretraining_users"] = [row["user_id"] for row in rows] + cohort["independent_final_test_users"]
    @contextmanager
    def provider():
        with preparation_pool(workers) as pool:
            def items():
                for row in rows:
                    frame = split_frame(get_rwkv_data(DATA, row["user_id"]), row, "train")
                    prepared = iter(batches(frame, row["user_id"], 0, 0, batch_config, pool))
                    try:
                        for item in prepared:
                            yield row["user_id"], item
                    finally:
                        prepared.close()
            iterator = items()
            try:
                yield iterator
            finally:
                iterator.close()
    def evaluator(model, device):
        per_user = []
        for row in rows:
            frame = split_frame(get_rwkv_data(DATA, row["user_id"]), row, "validation")
            digest = hashlib.sha256()
            def heldout():
                for offset in range(0, len(frame), 1024):
                    batch = prepare(make_samples(frame.iloc[offset:offset + 1024], row["user_id"], 128),
                                    seed=config["seed"] + row["user_id"] + row["validation_range"][0] + offset)
                    batch_fingerprint(batch, digest)
                    yield batch.to(device)
            scores = evaluate_batches(model, heldout())
            per_user.append({"user_id": row["user_id"], "input_sha256": digest.hexdigest(), "raw_reviews": len(frame), **scores})
        return per_user
    total = sum(math.ceil((row["train_range"][1] - row["train_range"][0]) / (128 * chunks)) for row in rows)
    return train(config, output, provider, evaluator, provenance, total)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--cohort", type=Path, default=ROOT / "artifacts/benchmarks/gpu-batch1-500/protocol.json")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--model", choices=("baseline", "expertium"), default="expertium")
    parser.add_argument("--task", choices=("joint", "curve", "instant"), default="joint")
    parser.add_argument("--schedule", choices=("constant", "cosine", "wsd"), default="constant")
    parser.add_argument("--width-heads", type=int)
    parser.add_argument("--remove-layers", type=int, default=0)
    parser.add_argument("--target-file", type=Path, help="Predeclared four-score targets; synthetic fixtures are not real quality targets")
    parser.add_argument("--target-confirmations", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts") or (output / "result.json").exists() or (output / "proposed-plan.json").exists():
        parser.error("Use a new output below workspace artifacts")
    output.mkdir(parents=True, exist_ok=True)
    temporary = ROOT / "artifacts/benchmarks/gpu-future/temp"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = str(temporary)
    provenance = process_provenance()
    atomic_json(output / "attempt.json", {"provenance": provenance, "command": sys.argv})
    original_exception_hook = sys.excepthook
    def retain_failure(kind, value, trace):
        if not (output / "result.json").exists():
            atomic_json(output / "result.json", {"status": "failed", "provenance": provenance,
                                                "error": str(value), "traceback": "".join(traceback.format_exception(kind, value, trace)),
                                                "synthetic_only": args.synthetic})
        original_exception_hook(kind, value, trace)
    sys.excepthook = retain_failure
    if args.prepare:
        plan = proposed_plan(json.loads(args.cohort.read_text()), output)
        print(f"Prepared {len(plan['cases'])} proposed cases; authorization remains pending")
    elif args.synthetic:
        config = {"model": args.model, "task": args.task, "schedule": args.schedule, "seed": 2025,
                  "width_heads": args.width_heads, "remove_layers": args.remove_layers}
        if args.target_file:
            config.update(targets=json.loads(args.target_file.read_text()), target_confirmations=args.target_confirmations)
        report = synthetic_run(config, output, provenance)
        print(f"Synthetic integration: {report['status']}; {report['updates']} updates; not validation")
    else:
        if not args.plan or not args.case:
            parser.error("Real runs require an approved --plan and --case")
        plan = json.loads(args.plan.read_text())
        case = next(case for case in plan["cases"] if case["name"] == args.case)
        real_run(plan, case, output, provenance)


if __name__ == "__main__":
    main()
