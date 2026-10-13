"""CUDA check: two R01 option sets give bitwise-identical Expertium loss and gradients.

Builds the same five real Expertium batches as check_gpu_exact_index.py, runs a
BF16 forward/backward from the fresh seed-2025 checkpoint under option set A and
option set B, and compares the loss and every parameter gradient bit for bit.
Use it before freezing a protocol for any change meant to keep exact parity:
pass the accepted reference options as --a and the candidate options as --b
(JSON objects, or a path to a plan/selection JSON plus --a-track/--b-track).

Options that change arithmetic are applied: exact_shift, exact_index,
exact_group_norm, fused_shift, row_tiles_exact, shared_shift_orders,
stream_launches, prefetch_rows, sync_free_loss, jit_optimize and
nondeterministic. Worker, transfer and scheduling options do not enter one
forward/backward and are listed as not exercised.
"""

from argparse import ArgumentParser
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import time
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
APPLIED = {"native_wkv", "native_group_norm", "native_index", "native_glue", "partial_shift_lerp", "explicit_shift_lerp", "exact_shift_lerp", "exact_bonus_products", "batched_split_gather", "fused_padded_gather", "exact_shift", "exact_index", "exact_group_norm", "fused_shift", "row_tiles_exact", "shared_shift_orders",
           "stream_launches", "driver_launch", "prefetch_rows", "prefetch_shift", "sync_free_loss", "jit_optimize", "nondeterministic", "skip_uninitialized_fill", "skip_padding_sources", "concurrent_splits", "concurrent_split_limit", "exact_gate_glue", "exact_channel_glue", "worker_scatter_orders", "recurrence_prefetch", "recurrence_load_pipeline", "wave_aware_backward", "flat_buffers", "compact_previous_state"}


def load_options(value, track):
    """A JSON object, or a plan/selection file with the track's options."""
    if value.lstrip().startswith("{"):
        return json.loads(value)
    data = json.loads((ROOT / value).read_text(encoding="utf-8"))
    if "case_options_by_model" in data:
        data = data["case_options_by_model"]
    data = data.get("expertium", data)
    if track:
        data = data[track]
    return data.get("options", data)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="Reference options: JSON object or plan/selection path")
    parser.add_argument("--b", required=True, help="Candidate options: JSON object or plan/selection path")
    parser.add_argument("--a-track")
    parser.add_argument("--b-track")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-diagnostics", action="store_true")
    parser.add_argument("--twin-pairs", type=int, default=0, help="P3b diagnostic pairs per batch, alternating AB/BA; timed forward/backward after warmup, identical restored RNG")
    parser.add_argument("--recurrence-only-stream", action="store_true")
    parser.add_argument("--self-test", action="store_true",
                        help="Negative control: flip the lowest bit of one BF16 weight in B; the check must then fail")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        parser.error("Use a new workspace artifact directory")
    variants = {"a": load_options(args.a, args.a_track), "b": load_options(args.b, args.b_track)}
    from scripts.r01_workflow import configure_runtime
    configure_runtime(ROOT, int(variants["a"].get("main_threads", 1)))
    cohort = json.loads((ROOT / "artifacts/retraining/gpu-retrain-r01/cohorts-user-holdout-v1.json").read_text(encoding="utf-8"))
    os.environ["GPU_BATCH1_DATA_PATH"] = str(ROOT / cohort["dataset_path"])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    torch.set_num_threads(int(variants["a"].get("main_threads", 1)))
    from rwkv.data_processing import get_rwkv_data
    from rwkv.model import exact_index, nvrtc
    from scripts.benchmark_gpu_trial import benchmark_models, fresh_checkpoint
    from scripts.experiment_provenance import process_provenance
    from scripts.gpu_batch1 import split_frame
    from scripts.gpu_b01 import prepare_job, transfer_batches
    from scripts.gpu_row_tiles import row_tiled_backend
    nvrtc.register()
    nvrtc.set_checkpoint_length(32)
    preparation = {"prefetch_users": True, "preparation_workers": 8, "gpu_packed_features": True, "packed_ipc": True}
    # Use the accepted byte-exact CPU path when checking an accepted R01 bundle.
    # Preparation is common to both variants; changed worker options still need
    # a separate CPU batch-byte comparison, as disclosed by not_exercised.
    for key in ("fast_samples", "batched_id_encodings", "columnar_samples"):
        if key in variants["a"]:
            preparation[key] = variants["a"][key]
    if any(o.get("worker_scatter_orders") for o in variants.values()):
        preparation.update(worker_scatter_orders=True, skip_padding_sources=True, shared_shift_orders=True)
    if preparation.get("batched_id_encodings"):
        from scripts.gpu_preprocessing_stage import prepare_job
    rows = sorted(cohort["stages"]["100"]["rows"], key=lambda row: row["available_reviews"])
    picks = [(rows[-1], 0, 32768), (rows[-3], 32768, 65536), (rows[len(rows) // 2], 0, 32768), (rows[0], 0, 32768), (rows[10], 0, 32768)]
    jobs = []
    for row, offset, end in picks:
        frame = split_frame(get_rwkv_data(Path(os.environ["GPU_BATCH1_DATA_PATH"]), row["user_id"]), row, "train")
        end = min(end, len(frame))
        jobs.append((row, prepare_job(row, offset, end, 2025 + row["user_id"] + offset, preparation, frame.iloc[offset:end])))
        print(f"[prepare] {len(jobs)}/{len(picks)} real batches", flush=True)
    batches = [(row, item, batch) for (row, _), (_, item, batch) in zip(jobs, transfer_batches(iter(jobs), torch.bfloat16, False))]
    result = {"provenance": process_provenance(), "torch": torch.__version__, "options": variants, "self_test": args.self_test,
              "not_exercised": {name: sorted(set(options) - APPLIED) for name, options in variants.items()},
              "scope": "five real Expertium batches; BF16 forward/backward from the fresh seed-2025 checkpoint; loss and every parameter gradient",
              "batches": [], "status": "running"}
    if not args.twin_pairs and any(o.get("skip_padding_sources") for o in variants.values()):
        original_rows_backward = exact_index.rows_backward
        result["padding_sources"] = {"calls": 0, "sources": 0, "nonzero": 0, "negative_zero": 0, "nonfinite": 0}

        def audited_rows_backward(grad, index, rows):
            padded = grad[index < 0]
            if padded.numel():
                audit = result["padding_sources"]
                audit["calls"] += 1
                audit["sources"] += padded.shape[0]
                audit["nonzero"] += int(torch.count_nonzero(padded).item())
                audit["negative_zero"] += int(torch.count_nonzero(torch.signbit(padded) & (padded == 0)).item())
                audit["nonfinite"] += int(torch.count_nonzero(~torch.isfinite(padded)).item())
                if audit["nonzero"] or audit["nonfinite"]:
                    raise RuntimeError("F50 padding gradient contract failed")
            return original_rows_backward(grad, index, rows)

        exact_index.rows_backward = audited_rows_backward

    def same(a, b):
        bits = {torch.bfloat16: torch.int16, torch.float32: torch.int32}[a.dtype]
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(bits), b.view(bits))

    def run(options, item, batch, checkpoint):
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        memory_start = torch.cuda.memory_stats()
        torch._C._set_graph_executor_optimize(bool(options.get("jit_optimize", False)))
        torch.use_deterministic_algorithms(not options.get("nondeterministic", False), warn_only=False)
        torch.utils.deterministic.fill_uninitialized_memory = not options.get("skip_uninitialized_fill", False)
        master, model, dtype = benchmark_models({"MODEL": "expertium", "DTYPE": "bfloat16", "SEED": 2025, "WKV_BACKEND": "nvrtc"},
                                                torch.device("cuda:0"))
        master.load_state_dict(checkpoint["model"])
        model.copy_downcast_(master, dtype)
        if options.get("native_wkv") or options.get("native_group_norm") or options.get("native_index") or options.get("native_glue"):
            from rwkv.model.native_launch import configure
            configure(model, options)
        if options.get("flat_buffers"):
            from argparse import Namespace
            from rwkv.train_rwkv import get_optimizer
            from scripts.gpu_r01_flat import FlatMasterChildSync
            flat = FlatMasterChildSync(master, model, get_optimizer(Namespace(PEAK_LR=7e-4), master))
            flat.downcast()
        if options is variants["b"] and args.self_test:
            with torch.no_grad():
                next(p for p in model.parameters() if p.dtype == torch.bfloat16).view(-1).view(torch.int16)[0] ^= 1
        for module in model.modules():
            for flag in ("fused_shift", "partial_shift_lerp", "explicit_shift_lerp", "exact_shift_lerp", "exact_shift", "exact_index", "exact_group_norm"):
                if hasattr(module, flag):
                    setattr(module, flag, bool(options.get(flag, False)))
        if options.get("sync_free_loss"):
            model.sync_free_loss = True
        if options.get("skip_padding_sources"):
            if not options.get("exact_index"):
                raise ValueError("F50 requires exact_index")
            model.skip_padding_sources = True
        for module in model.modules():
            if hasattr(module, "exact_bonus_products"):
                module.exact_bonus_products = bool(options.get("exact_bonus_products", False))
            if hasattr(module, "exact_gate_glue"):
                module.exact_gate_glue = int(options.get("exact_gate_glue", 0))
            if hasattr(module, "exact_channel_glue"):
                module.exact_channel_glue = bool(options.get("exact_channel_glue", False))
        model.batched_split_gather = bool(options.get("batched_split_gather", False))
        model.fused_padded_gather = bool(options.get("fused_padded_gather", False))
        model.concurrent_split_limit = int(options.get("concurrent_split_limit", 0))
        if options.get("concurrent_splits"):
            model.concurrent_splits = True
        model.train()
        torch.manual_seed(item.seed)
        with ExitStack() as stack:
            if options.get("row_tiles_exact"):
                stack.enter_context(row_tiled_backend(rows=options["row_tiles_exact"], tile=32, exact=True,
                                                     compact_previous=bool(options.get("compact_previous_state", False)),
                                  next_step_prefetch=bool(options.get("recurrence_prefetch", False)),
                                  load_pipeline=int(options.get("recurrence_load_pipeline", 0)),
                                  wave_aware=bool(options.get("wave_aware_backward", False))))
            nvrtc.set_stream_launch(options.get("stream_launches", False))
            nvrtc.set_driver_launch(options.get("driver_launch", False))
            exact_index.set_shared_orders(options.get("shared_shift_orders", False))
            if options.get("worker_scatter_orders"):
                from scripts.gpu_scatter_orders import install_orders
                install_orders(batch)
            exact_index.set_prefetch_rows(options.get("prefetch_rows", False))
            exact_index.set_prefetch_shift(options.get("prefetch_shift", False))
            stack.callback(nvrtc.set_stream_launch, False)
            stack.callback(nvrtc.set_driver_launch, False)
            stack.callback(exact_index.set_shared_orders, False)
            stack.callback(exact_index.set_prefetch_rows, False)
            stack.callback(exact_index.set_prefetch_shift, False)
            if args.recurrence_only_stream and options is variants["b"]:
                original_launch = nvrtc.launch
                side = torch.cuda.Stream()
                def recurrence_launch(kernel, device, shape, tensors):
                    parent = torch.cuda.current_stream(device)
                    side.wait_stream(parent)
                    with torch.cuda.stream(side):
                        original_launch(kernel, device, shape, tensors)
                    parent.wait_stream(side)
                nvrtc.launch = recurrence_launch
                stack.callback(setattr, nvrtc, "launch", original_launch)
            if args.twin_pairs:
                cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
                stats = model.get_loss(batch)
                stats.average_loss.backward()
                model.zero_grad(set_to_none=True)
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng)
                torch.cuda.synchronize()
            started = time.perf_counter()
            stats = model.get_loss(batch)
            stats.average_loss.backward()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
        torch.cuda.synchronize()
        return stats.average_loss.detach().clone(), {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}, {"peak_reserved_bytes": torch.cuda.max_memory_reserved(), "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "allocator_retries": torch.cuda.memory_stats()["num_alloc_retries"] - memory_start["num_alloc_retries"], "forward_backward_seconds": elapsed}

    try:
        checkpoint = fresh_checkpoint("expertium", 2025)
        for row, item, batch in batches:
            pairs = []
            for pair in range(args.twin_pairs):
                order = ("a", "b") if pair % 2 == 0 else ("b", "a")
                measured = {name: run(variants[name], item, batch, checkpoint) for name in order}
                equal = same(measured['a'][0], measured['b'][0]) and measured['a'][1].keys() == measured['b'][1].keys() and all(same(measured['a'][1][n], measured['b'][1][n]) for n in measured['a'][1])
                pairs.append({"order": list(order), "a_seconds": measured['a'][2]['forward_backward_seconds'], "b_seconds": measured['b'][2]['forward_backward_seconds'], "bitwise_equal": equal})
                print(f"[twin] batch {len(result['batches'])+1}/5 pair {pair+1}/{args.twin_pairs}; exact {equal}", flush=True)
            if pairs:
                result.setdefault('twin_pairs', []).append({"user_id": row['user_id'], "reviews": int(batch.num_data), "pairs": pairs})
            (loss_a, grads_a, memory_a), (loss_b, grads_b, memory_b) = (run(variants[name], item, batch, checkpoint) for name in ("a", "b"))
            names = sorted(grads_a)
            differing = [name for name in names if name not in grads_b or not same(grads_a[name], grads_b[name])]
            result["batches"].append({"user_id": row["user_id"], "offset": item.offset, "reviews": int(batch.num_data),
                                      "loss_bitwise_equal": same(loss_a, loss_b), "same_gradient_set": sorted(grads_b) == names,
                                      "parameters": len(names), "differing_gradients": differing[:10],
                                      "memory": {"a": memory_a, "b": memory_b}, "differing_gradient_count": len(differing)})
            print(f"[compare] {len(result['batches'])}/{len(batches)} batches; differing gradients {len(differing)}", flush=True)
        batches_equal = sum(b["loss_bitwise_equal"] and b["same_gradient_set"] and not b["differing_gradient_count"] for b in result["batches"])
        result["summary"] = {"batches": len(result["batches"]), "batches_bitwise_equal": batches_equal}
        result["status"] = "passed" if batches_equal == len(result["batches"]) > 0 else "failed"
        if args.twin_pairs:
            pairs = [p for batch in result['twin_pairs'] for p in batch['pairs']]
            effects = [100 * math.log(p['a_seconds'] / p['b_seconds']) for p in pairs]
            mean = statistics.mean(effects)
            se = statistics.stdev(effects) / math.sqrt(len(effects)) if len(effects) > 1 else None
            result['twin_timing'] = {"pairs": len(pairs), "mean_log_gain_percent": mean, "standard_error_percent": se, "descriptive_normal_95_interval_percent": [mean-1.96*se, mean+1.96*se] if se is not None else None, "scope": "diagnostic only; warm forward/backward wall time excluding model setup, GC, allocator clearing and gradient clones; alternating AB/BA within batch; seeded RNG restored after warmup; pooled heterogeneous batches, not frozen promotion gate"}
            if not all(p['bitwise_equal'] for p in pairs):
                result['status'] = 'failed'
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = True
        output.mkdir(parents=True)
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "summary": result.get("summary"), "not_exercised_b": result["not_exercised"]["b"][:6],
                          "error": result.get("error")}))


if __name__ == "__main__":
    main()
