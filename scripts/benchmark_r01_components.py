"""Fixed-window component screen for R01 candidate selection; not a training run.

Each variant runs in a fresh process on identical review windows from fixed R01
training users, repeating the R01 update sequence (seeded dropout, master/child
copies, forward/backward, clipping, AdamW, host loss/norm reads). Preparation and
transfers happen before timing, so this isolates the per-update path. Results
select a candidate; they do not replace full-history timing or quality.
"""

from argparse import ArgumentParser
from argparse import Namespace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
COHORT = ROOT / "artifacts/retraining/gpu-retrain-r01/cohorts-user-holdout-v1.json"
ACCEPTED = {"baseline": {"prefetch_users": True},
            "expertium": {"prefetch_users": True, "preparation_workers": 8, "gpu_packed_features": True,
                          "packed_ipc": True, "async_transfer": True}}


def windows(model):
    """Fixed training-user windows: large histories dominate reviews; the median user gives partial batches."""
    cohort = json.loads(COHORT.read_text(encoding="utf-8"))
    rows = sorted(cohort["stages"]["100"]["rows"], key=lambda row: (row["available_reviews"], row["user_id"]))
    median = rows[len(rows) // 2]
    # Every variant trains on exactly these reviews, whatever its batch size.
    chosen = [(rows[-1], 196608), (rows[-5], 65536), (rows[-8], 65536), (median, median["available_reviews"])]
    return cohort["dataset_path"], [(row, min(size, row["available_reviews"])) for row, size in chosen]


def run_variant(model, name, options, output):
    data_path, selected = windows(model)
    os.environ["GPU_BATCH1_DATA_PATH"] = str(ROOT / data_path)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["CUPY_CACHE_DIR"] = str(ROOT / "artifacts/cache/cupy-cache")
    import torch
    from scripts.experiment_provenance import process_provenance
    provenance = process_provenance()
    torch._C._set_graph_executor_optimize(bool(options.get("jit_optimize", False)))
    torch.use_deterministic_algorithms(not options.get("nondeterministic", False), warn_only=False)
    from rwkv.model import nvrtc
    from rwkv.train_rwkv import CLIP, get_optimizer, transfer_child_grad_to_master
    from rwkv.data_processing import get_rwkv_data
    from scripts.benchmark_gpu_trial import benchmark_models, fresh_checkpoint
    from scripts.gpu_batch1 import split_frame
    from scripts.gpu_b01 import prepare_job, transfer_batches
    from scripts.gpu_r01_sync import ReducedSyncLoss
    from scripts.gpu_r01_update import MasterChildSync

    chunks = options.get("batch_chunks", 128 if model == "baseline" else 256)
    span = chunks * 128
    fraction = options.get("gpu_memory_fraction", 0.72)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    nvrtc.register()
    nvrtc.set_checkpoint_length(32)
    device = torch.device("cuda:0")
    config = {"MODEL": model, "DTYPE": "bfloat16", "SEED": 2025, "WKV_BACKEND": "nvrtc"}
    prepared, reviews = [], 0
    started = time.perf_counter()
    for row, length in selected:
        frame = split_frame(get_rwkv_data(Path(os.environ["GPU_BATCH1_DATA_PATH"]), row["user_id"]), row, "train")
        for offset in range(0, length, span):
            end = min(length, offset + span)
            item = prepare_job(row, offset, end, 2025 + row["user_id"] + offset, {**ACCEPTED[model], **options},
                               frame.iloc[offset:end])
            prepared.append((row, item))
            reviews += item.reviews
    preparation_seconds = time.perf_counter() - started
    checkpoint = fresh_checkpoint(model, 2025)
    master, base, dtype = benchmark_models(config, device)
    master.load_state_dict(checkpoint["model"])
    base.copy_downcast_(master, dtype)
    for module in base.modules():
        if hasattr(module, "exact_shift"):
            module.exact_shift = options.get("exact_shift", False)
        if hasattr(module, "exact_index"):
            module.exact_index = options.get("exact_index", False)
    batches = [(item, batch) for _, item, batch in transfer_batches(iter(prepared), dtype, False)]
    optimizer = get_optimizer(Namespace(PEAK_LR=7e-4), master)
    grouped = MasterChildSync(master, base) if options.get("grouped_sync") else None
    reduced = ReducedSyncLoss(base) if options.get("reduced_sync") else None
    updates = 0

    def update(item, batch):
        nonlocal updates
        torch.manual_seed(item.seed)
        optimizer.zero_grad(set_to_none=True)
        base.zero_grad(set_to_none=True)
        base.train()
        grouped.downcast() if grouped else base.copy_downcast_(master, dtype)
        stats = reduced.get_loss(batch) if reduced else base.get_loss(batch)
        if stats is None or not torch.isfinite(stats.average_loss):
            raise RuntimeError("Non-finite benchmark loss")
        stats.average_loss.backward()
        grouped.transfer_grads() if grouped else transfer_child_grad_to_master(master, base)
        norm = torch.nn.utils.clip_grad_norm_(master.parameters(), CLIP, error_if_nonfinite=True)
        updates += 1
        for group in optimizer.param_groups:
            group["lr"] = 7e-4 * min(1.0, updates / 5)
        optimizer.step()
        if reduced:
            return torch.stack((stats.average_loss.detach(), norm)).cpu().tolist()
        return stats.average_loss.item(), norm.item()

    result = {"model": model, "variant": name, "options": options, "provenance": provenance,
              "windows": [{"user_id": row["user_id"], "reviews": length} for row, length in selected],
              "reviews_per_pass": reviews, "batches_per_pass": len(batches), "preparation_seconds": preparation_seconds,
              "scope": "fixed windows, preprepared batches on GPU; per-update path only; four passes; first includes JIT warmup; warm value is the median of passes 2-4",
              "passes": []}
    try:
        torch.cuda.reset_peak_memory_stats()
        losses = []
        for _ in range(4):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            for item, batch in batches:
                losses.append(update(item, batch)[0])
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            result["passes"].append({"seconds": elapsed, "ms_per_update": 1000 * elapsed / len(batches),
                                     "reviews_per_second": reviews / elapsed})
        result.update(status="complete", final_loss=losses[-1], first_loss=losses[0],
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                      warm_reviews_per_second=sorted(p["reviews_per_second"] for p in result["passes"][1:])[1])
    except torch.OutOfMemoryError as error:
        result.update(status="out of memory", error=str(error)[:300])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result.get(key) for key in ("model", "variant", "status", "warm_reviews_per_second", "peak_reserved_gib")}), flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, help="JSON {model: {variant: options}} run in separate processes")
    parser.add_argument("--model")
    parser.add_argument("--variant")
    parser.add_argument("--options", default="{}")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Output must remain under workspace artifacts")
    if args.plan:
        plan = json.loads((ROOT / args.plan).read_text(encoding="utf-8"))
        for model, variants in plan.items():
            for name, options in variants.items():
                target = output / model / f"{name}.json"
                if target.exists():
                    continue  # Preserve completed variants; rerun only missing ones.
                subprocess.run([sys.executable, __file__, "--model", model, "--variant", name,
                                "--options", json.dumps(options), "--output", str(target)], check=False)
        return
    run_variant(args.model, args.variant, json.loads(args.options), output)


if __name__ == "__main__":
    main()
