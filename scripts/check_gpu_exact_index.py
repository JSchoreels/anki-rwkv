"""CUDA check: F26/F27 sort-free backward equals deterministic PyTorch bit for bit.

Covers the time-shift gather and hierarchy row selection on real Expertium
indices, then whole-model parameter gradients on real batches.
"""

from argparse import ArgumentParser
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--f38", action="store_true",
                        help="Also check F38 shared shift orders and stream launches, and the whole E27 kernel path")
    parser.add_argument("--f35", action="store_true",
                        help="Also check and time F35 prefetching row-selection backward (adds it to the --f38 model path)")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        parser.error("Use a new workspace artifact directory")
    cohort = json.loads((ROOT / "artifacts/retraining/gpu-retrain-r01/cohorts-user-holdout-v1.json").read_text(encoding="utf-8"))
    os.environ["GPU_BATCH1_DATA_PATH"] = str(ROOT / cohort["dataset_path"])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    torch._C._set_graph_executor_optimize(False)
    torch.use_deterministic_algorithms(True, warn_only=False)
    from rwkv.data_processing import get_rwkv_data
    from rwkv.model import nvrtc
    from rwkv.model import exact_index
    from rwkv.model.exact_index import rows_backward, shift_backward
    from scripts.benchmark_gpu_trial import benchmark_models, fresh_checkpoint
    from scripts.experiment_provenance import process_provenance
    from scripts.gpu_batch1 import split_frame
    from scripts.gpu_b01 import prepare_job, transfer_batches
    nvrtc.register()
    nvrtc.set_checkpoint_length(32)
    options = {"prefetch_users": True, "preparation_workers": 8, "gpu_packed_features": True, "packed_ipc": True}
    rows = sorted(cohort["stages"]["100"]["rows"], key=lambda row: row["available_reviews"])
    picks = [(rows[-1], 0, 32768), (rows[-3], 32768, 65536), (rows[len(rows) // 2], 0, 32768), (rows[0], 0, 32768), (rows[10], 0, 32768)]
    jobs = []
    for row, offset, end in picks:
        frame = split_frame(get_rwkv_data(Path(os.environ["GPU_BATCH1_DATA_PATH"]), row["user_id"]), row, "train")
        end = min(end, len(frame))
        jobs.append((row, prepare_job(row, offset, end, 2025 + row["user_id"] + offset, options, frame.iloc[offset:end])))
    batches = [(row, item, batch) for (row, _), (_, item, batch) in zip(jobs, transfer_batches(iter(jobs), torch.bfloat16, False))]
    result = {"provenance": process_provenance(), "torch": torch.__version__, "operations": [], "models": [], "status": "running",
              "scope": "real Expertium indices; random BF16/FP32 gradients with zeros and signed zeros; whole-model BF16 parameter gradients"}
    generator = torch.Generator(device="cuda").manual_seed(27)

    def noisy(shape, dtype):
        grad = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
        grad[torch.rand(shape, device="cuda", generator=generator) < 0.2] = 0.0
        grad[torch.rand(shape, device="cuda", generator=generator) < 0.05] = -0.0
        return grad

    def same(a, b):
        bits = {torch.bfloat16: torch.int16, torch.float32: torch.int32}[a.dtype]
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(bits), b.view(bits))

    try:
        for row, item, batch in batches:
            rows_in = batch.start.shape[0]
            for module, (gathers, selects, lengths) in enumerate(zip(batch.sub_gather, batch.time_shift_selects, batch.sub_gather_lens)):
                for gather, select, length in zip(gathers, selects, lengths):
                    index = torch.clamp(gather, min=0).long()
                    shift = select.view(-1, length).long()
                    for dtype in (torch.bfloat16, torch.float32):
                        x = torch.randn((max(rows_in, int(index.max()) + 1) + 3, 32), device="cuda", dtype=dtype, generator=generator).requires_grad_()
                        rows_grad = noisy((index.numel(), 32), dtype)
                        torch.index_select(x, 0, index).backward(rows_grad)
                        rows_equal = same(x.grad, rows_backward(rows_grad, index, x.shape[0]))
                        y = torch.randn((*shift.shape, 32), device="cuda", dtype=dtype, generator=generator).requires_grad_()
                        grad = noisy(y.shape, dtype)
                        torch.gather(y, 1, shift.unsqueeze(-1).expand(-1, -1, 32)).backward(grad)
                        shift_equal = same(y.grad, shift_backward(grad, shift))
                        f38 = {}
                        if args.f38:
                            # Cache miss, then a hit on the same split indices, both with stream launches.
                            nvrtc.set_stream_launch(True)
                            exact_index.set_shared_orders(True)
                            first = shift_backward(grad, shift)
                            entries = len(exact_index._orders)
                            second = shift_backward(grad, shift)
                            f38 = {"f38_shift_bitwise_equal": same(y.grad, first) and same(y.grad, second),
                                   "f38_cache_hit": len(exact_index._orders) == entries,
                                   "f38_rows_bitwise_equal": same(x.grad, rows_backward(rows_grad, index, x.shape[0]))}
                            nvrtc.set_stream_launch(False)
                            exact_index.set_shared_orders(False)
                        if args.f35:
                            def timed_rows(enabled, repeats=20):
                                exact_index.set_prefetch_rows(enabled)
                                rows_backward(rows_grad, index, x.shape[0])
                                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                begin.record()
                                for _ in range(repeats):
                                    value = rows_backward(rows_grad, index, x.shape[0])
                                end.record()
                                torch.cuda.synchronize()
                                exact_index.set_prefetch_rows(False)
                                return value, begin.elapsed_time(end) / repeats
                            prefetched, prefetch_ms = timed_rows(True)
                            _, sequential_ms = timed_rows(False)
                            f38.update(f35_rows_bitwise_equal=same(x.grad, prefetched),
                                       f35_sequential_ms=sequential_ms, f35_prefetch_ms=prefetch_ms)
                        result["operations"].append({**f38, "user_id": row["user_id"], "module": module, "dtype": str(dtype),
                                                     "rows": int(x.shape[0]), "selected": int(index.numel()),
                                                     "max_row_sources": int(torch.bincount(index.cpu()).max()),
                                                     "rows_bitwise_equal": rows_equal, "shift_bitwise_equal": shift_equal})
        device = torch.device("cuda:0")
        checkpoint = fresh_checkpoint("expertium", 2025)
        for row, item, batch in batches:
            grads = {}
            for exact in (False, True) + (("f38",) if args.f38 else ()):
                master, model, dtype = benchmark_models({"MODEL": "expertium", "DTYPE": "bfloat16", "SEED": 2025, "WKV_BACKEND": "nvrtc"}, device)
                master.load_state_dict(checkpoint["model"])
                model.copy_downcast_(master, dtype)
                for module in model.modules():
                    if hasattr(module, "exact_shift"):
                        module.exact_shift = bool(exact)
                    if hasattr(module, "exact_index"):
                        module.exact_index = bool(exact)
                    if hasattr(module, "exact_group_norm"):
                        module.exact_group_norm = exact == "f38"
                model.train()
                torch.manual_seed(item.seed)
                with ExitStack() as stack:
                    if exact == "f38":
                        # The whole E27 kernel path (F26/F27, F31 row tiles, F32 GroupNorm) plus F38.
                        from scripts.gpu_row_tiles import row_tiled_backend
                        stack.enter_context(row_tiled_backend(rows=4, tile=32, exact=True))
                        nvrtc.set_stream_launch(True)
                        exact_index.set_shared_orders(True)
                        exact_index.set_prefetch_rows(args.f35)
                        stack.callback(exact_index.set_prefetch_rows, False)
                        stack.callback(nvrtc.set_stream_launch, False)
                        stack.callback(exact_index.set_shared_orders, False)
                    stats = model.get_loss(batch)
                    stats.average_loss.backward()
                grads[exact] = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
                grads[str(exact) + "_loss"] = stats.average_loss.detach().clone()
            names = sorted(grads[False])
            f38 = {}
            if args.f38:
                f38 = {"f38_loss_bitwise_equal": same(grads["False_loss"], grads["f38_loss"]),
                       "f38_gradients_bitwise_equal": sorted(grads["f38"]) == names and all(same(grads[False][n], grads["f38"][n]) for n in names)}
            result["models"].append({**f38, "user_id": row["user_id"], "offset": item.offset, "parameters_with_gradients": len(names),
                                     "same_gradient_set": sorted(grads[True]) == names,
                                     "loss_bitwise_equal": same(grads["False_loss"], grads["True_loss"]),
                                     "all_gradients_bitwise_equal": sorted(grads[True]) == names and all(same(grads[False][n], grads[True][n]) for n in names)})
        ops, models = result["operations"], result["models"]
        result["summary"] = {"operation_cases": len(ops), "rows_equal": sum(o["rows_bitwise_equal"] for o in ops),
                             "shift_equal": sum(o["shift_bitwise_equal"] for o in ops),
                             "max_row_sources": max(o["max_row_sources"] for o in ops),
                             "model_batches": len(models), "model_gradients_equal": sum(m["all_gradients_bitwise_equal"] for m in models)}
        if args.f38:
            result["summary"].update(f38_shift_equal=sum(o["f38_shift_bitwise_equal"] for o in ops),
                                     f38_cache_hits=sum(o["f38_cache_hit"] for o in ops),
                                     f38_rows_equal=sum(o["f38_rows_bitwise_equal"] for o in ops),
                                     f38_model_gradients_equal=sum(m["f38_gradients_bitwise_equal"] and m["f38_loss_bitwise_equal"] for m in models))
        if args.f35:
            result["summary"].update(f35_rows_equal=sum(o["f35_rows_bitwise_equal"] for o in ops),
                                     f35_sequential_ms=sum(o["f35_sequential_ms"] for o in ops),
                                     f35_prefetch_ms=sum(o["f35_prefetch_ms"] for o in ops))
        f38_passed = not args.f38 or (all(o["f38_shift_bitwise_equal"] and o["f38_cache_hit"] and o["f38_rows_bitwise_equal"] for o in ops)
                                      and all(m["f38_gradients_bitwise_equal"] and m["f38_loss_bitwise_equal"] for m in models))
        f38_passed = f38_passed and (not args.f35 or all(o["f35_rows_bitwise_equal"] for o in ops))
        result["status"] = "passed" if (ops and models and f38_passed and all(o["rows_bitwise_equal"] and o["shift_bitwise_equal"] for o in ops)
                                        and all(m["all_gradients_bitwise_equal"] and m["loss_bitwise_equal"] for m in models)) else "failed"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        output.mkdir(parents=True)
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "summary": result.get("summary"), "error": result.get("error")}))


if __name__ == "__main__":
    main()
