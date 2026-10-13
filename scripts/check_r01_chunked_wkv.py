"""F69: chunked WKV7 prototype versus the exact sequential recurrence (outputs and all six input gradients).

Parts: (1) CPU float64 identity of the chunked algebra against the sequential recurrence; (2) CUDA synthetic
cases in the production ABI (BF16 payload, FP32 decay) against the exact row-tiled recurrence
(`RWKV7_WKV` under `row_tiled_backend(rows=4, tile=32, exact=True)`) and a float64 ground truth;
(3) real Expertium batches: every recurrence call of a real forward/backward is replayed with its real
inputs and real output gradient. Nothing is wired into training and no timing is taken.
Use --cpu-only for part 1 only (no GPU, no claim needed)."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NAMES = ("out", "dr", "dk", "dv", "dw", "da", "dkd")


def cpu_identity(torch, chunked_wkv, sequential_wkv):
    cases = []
    torch.manual_seed(7)
    for (b, t) in [(3, 40), (2, 16), (2, 7), (2, 65), (1, 1)]:
        for chunk in (8, 16, 32):
            for skip_mode in ("none", "random", "all"):
                n = lambda: torch.randn(b, t, 2, 32, dtype=torch.float64)
                r, k, v = n(), torch.nn.functional.normalize(n(), dim=-1), torch.nn.functional.normalize(n(), dim=-1)
                w = torch.exp(-torch.exp(-0.5 - torch.nn.functional.softplus(-n())))
                a, kd = torch.sigmoid(n()), torch.nn.functional.normalize(n(), dim=-1)
                skip = {"none": torch.zeros(b, t, dtype=torch.bool), "all": torch.ones(b, t, dtype=torch.bool),
                        "random": torch.rand(b, t) < 0.3}[skip_mode]
                ins = [x.requires_grad_() for x in (r, k, v, w, a, kd)]
                g = torch.randn(b, t, 2, 32, dtype=torch.float64)
                ref = sequential_wkv(*ins, skip)
                gref = torch.autograd.grad(ref, ins, g)
                out = chunked_wkv(*ins, skip, chunk=chunk, compute_dtype=torch.float64)
                gout = torch.autograd.grad(out, ins, g)
                err = max([float((out - ref).abs().max())] + [float((x - y).abs().max()) for x, y in zip(gref, gout)])
                cases.append({"b": b, "t": t, "chunk": chunk, "skip": skip_mode, "max_abs_error": err})
    return cases


def stats(candidate, exact, truth):
    """Per tensor: candidate versus exact and both versus float64 truth, all relative to the tensor's absmax."""
    scale = float(truth.abs().max()) or 1.0
    c, e, t = candidate.double(), exact.double(), truth.double()
    return {"cand_vs_exact_max_abs": float((c - e).abs().max()), "cand_vs_exact_rel": float((c - e).abs().max()) / scale,
            "cand_vs_exact_fraction_unequal": float((candidate != exact).double().mean()),
            "exact_vs_truth_rel": float((e - t).abs().max()) / scale, "cand_vs_truth_rel": float((c - t).abs().max()) / scale,
            "absmax": scale}


def evaluate(torch, exact_apply, chunk_fn, inputs, skip, grad, chunk, with_truth, sequential_wkv, chunk32=True):
    """inputs: r,k,v,w,a,kd (BF16 payload, FP32 w). Returns {tensor: stats} for one recurrence call."""
    def run(fn, dtype_cast=None):
        leaves = [x.detach().clone().requires_grad_() for x in inputs]
        out = fn(*leaves)
        out.backward(grad.to(out.dtype))
        return [out.detach()] + [x.grad.detach() for x in leaves]

    ex = run(lambda *x: exact_apply(*x, skip))
    ch = run(lambda *x: chunk_fn(*x, skip, chunk=chunk))
    if with_truth:
        leaves = [x.detach().double().requires_grad_() for x in inputs]
        out = sequential_wkv(*leaves, skip)
        out.backward(grad.double())
        tr = [out.detach()] + [x.grad.detach() for x in leaves]
    else:
        tr = [x.double() for x in ex]  # truth unavailable: exact stands in, flagged by with_truth=False
    result = {name: stats(c, e, t) for name, c, e, t in zip(NAMES, ch, ex, tr)}
    if not with_truth:
        for entry in result.values():
            del entry["exact_vs_truth_rel"], entry["cand_vs_truth_rel"]
    if chunk32 and with_truth:
        # compute-precision view: chunked in FP32 from the same BF16 values, no BF16 output rounding
        leaves = [x.detach().float().requires_grad_() for x in inputs]
        out = chunk_fn(*leaves, skip, chunk=chunk)
        out.backward(grad.float())
        c32 = [out.detach()] + [x.grad.detach() for x in leaves]
        for name, c, t in zip(NAMES, c32, tr):
            scale = float(t.abs().max()) or 1.0
            result[name]["cand_fp32_vs_truth_rel"] = float((c.double() - t.double()).abs().max()) / scale
    return result


def merge(target, new):
    for name, values in new.items():
        slot = target.setdefault(name, {})
        for key, value in values.items():
            slot[key] = max(slot.get(key, 0.0), value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--chunks", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--no-real", action="store_true")
    parser.add_argument("--real-only", action="store_true", help="skip the synthetic CUDA grid (already recorded)")
    parser.add_argument("--real-truth-batches", type=int, default=1, help="batches whose calls also get a float64 truth")
    args = parser.parse_args()
    out = ROOT / args.output
    if out.exists():
        parser.error("Use a new output directory")
    os.chdir(ROOT)
    from scripts.r01_workflow import configure_runtime
    configure_runtime(ROOT, 1)
    import torch
    from rwkv.model.chunked_wkv import chunked_wkv, sequential_wkv, check_decay_range
    result = {"torch": torch.__version__, "cpu_identity": cpu_identity(torch, chunked_wkv, sequential_wkv)}
    result["cpu_identity_max_abs_error"] = max(c["max_abs_error"] for c in result["cpu_identity"])
    result["status"] = "passed" if result["cpu_identity_max_abs_error"] < 1e-9 else "failed"
    if not args.cpu_only:
        from rwkv.model import nvrtc
        from expertium.model.rwkv_ops import RWKV7_WKV
        from scripts.check_gpu_row_tiles import synthetic
        from scripts.experiment_provenance import process_provenance
        from scripts.gpu_row_tiles import row_tiled_backend
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        nvrtc.register()
        nvrtc.set_stream_launch(True)
        result["provenance"] = process_provenance()
        result["synthetic_by_shape"] = {}
        generator = torch.Generator(device="cuda")
        with row_tiled_backend(rows=4, tile=32, exact=True):
            for index, (b, t) in enumerate([] if args.real_only else [(1, 1), (5, 31), (97, 33), (5, 64), (97, 256), (742, 256), (36177, 2), (318, 26), (4240, 12)]):
                for skip_mode in ("none", "every-third", "random"):
                    generator.manual_seed(20261013 + index)
                    values, skip, grad = synthetic(torch, b, t, skip_mode, generator)
                    for chunk in args.chunks:
                        key = f"B{b}xT{t}/chunk{chunk}/{skip_mode}"
                        decay = check_decay_range(values[3], chunk)
                        entry = evaluate(torch, RWKV7_WKV.apply, chunked_wkv, values, skip, grad, chunk,
                                         with_truth=b * t <= 30000, sequential_wkv=sequential_wkv)
                        entry["max_cumulative_neg_log_w"] = decay
                        entry["truth_available"] = b * t <= 30000
                        result["synthetic_by_shape"][key] = entry
                print(f"[synthetic] B{b}xT{t} done", flush=True)
            if not args.no_real:
                result["real"] = real_batches(torch, args, chunked_wkv, sequential_wkv, check_decay_range, RWKV7_WKV)
    out.mkdir(parents=True)
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "cpu_identity_max_abs_error": result["cpu_identity_max_abs_error"]}))
    if result["status"] != "passed":
        raise SystemExit(1)


def real_batches(torch, args, chunked_wkv, sequential_wkv, check_decay_range, RWKV7_WKV):
    """Replay every recurrence call of five real Expertium batches with its real inputs and real output gradient."""
    from rwkv.model import nvrtc
    from rwkv.data_processing import get_rwkv_data
    from scripts.benchmark_gpu_trial import benchmark_models, fresh_checkpoint
    from scripts.gpu_batch1 import split_frame
    from scripts.gpu_b01 import transfer_batches
    from scripts.gpu_preprocessing_stage import prepare_job
    cohort = json.loads((ROOT / "artifacts/retraining/gpu-retrain-r01/cohorts-user-holdout-v1.json").read_text(encoding="utf-8"))
    os.environ["GPU_BATCH1_DATA_PATH"] = str(ROOT / cohort["dataset_path"])
    import rwkv.data_processing  # noqa: F401  (path read at call time)
    preparation = {"prefetch_users": True, "preparation_workers": 1, "gpu_packed_features": True, "packed_ipc": True,
                   "fast_samples": True, "batched_id_encodings": True, "columnar_samples": True}
    rows = sorted(cohort["stages"]["100"]["rows"], key=lambda row: row["available_reviews"])
    picks = [(rows[-1], 0, 32768), (rows[-3], 32768, 65536), (rows[len(rows) // 2], 0, 32768), (rows[0], 0, 32768), (rows[10], 0, 32768)]
    jobs = []
    for row, offset, end in picks:
        frame = split_frame(get_rwkv_data(Path(os.environ["GPU_BATCH1_DATA_PATH"]), row["user_id"]), row, "train")
        end = min(end, len(frame))
        jobs.append((row, prepare_job(row, offset, end, 2025 + row["user_id"] + offset, preparation, frame.iloc[offset:end])))
        print(f"[real] prepared {len(jobs)}/5", flush=True)
    batches = list(transfer_batches(iter(jobs), torch.bfloat16, False))
    torch.use_deterministic_algorithms(True)
    checkpoint = fresh_checkpoint("expertium", 2025)
    master, model, dtype = benchmark_models({"MODEL": "expertium", "DTYPE": "bfloat16", "SEED": 2025, "WKV_BACKEND": "nvrtc"}, torch.device("cuda:0"))
    master.load_state_dict(checkpoint["model"])
    # A fresh checkpoint zero-initialises output projections, so every recurrence gradient would be exactly zero.
    perturbed = []
    generator = torch.Generator(device="cuda").manual_seed(20261013)
    with torch.no_grad():
        for name, parameter in master.named_parameters():
            if float(parameter.abs().max()) == 0.0:
                parameter.copy_(torch.randn(parameter.shape, generator=generator, device=parameter.device) * 0.02)
                perturbed.append(name)
    model.copy_downcast_(master, dtype)
    model.train()
    by_shape, calls_per_batch, decay_max, w_min = {}, [], 0.0, 1.0
    original = RWKV7_WKV.apply
    for bi, (row, item, batch) in enumerate(batches):
        records = []

        live = {}
        launcher = nvrtc.launch

        def recording_launch(kernel, device, shape, args):
            launcher(kernel, device, shape, args)
            if len(args) == 13:    # forward: b,t,h,r,k,v,w,a,kd,skip,out,checkpoints,states
                record = {"inputs": [x.detach().clone() for x in args[3:9]], "skip": args[9].detach().clone()}
                live[args[3].data_ptr()] = record
                records.append(record)
            elif len(args) == 19:  # backward: b,t,h,r,k,v,w,a,kd,skip,grad,checkpoints,states,grads...
                live[args[3].data_ptr()]["grad"] = args[10].detach().clone()
        nvrtc.launch = recording_launch
        try:
            torch.manual_seed(item.seed)
            model.zero_grad(set_to_none=True)
            stats_ = model.get_loss(batch)
            stats_.average_loss.backward()
        finally:
            nvrtc.launch = launcher
        records[:] = [x for x in records if "grad" in x]
        calls_per_batch.append(len(records))
        for record in records:
            inputs, skip, grad = record["inputs"], record["skip"], record["grad"]
            shape = tuple(inputs[0].shape[:2])
            decay_max = max(decay_max, check_decay_range(inputs[3], args.chunks[0]))
            w_min = min(w_min, float(inputs[3].min()))
            for chunk in args.chunks:
                entry = evaluate(torch, original, chunked_wkv, inputs, skip, grad, chunk,
                                 with_truth=bi < args.real_truth_batches and shape[0] * shape[1] <= 400000, sequential_wkv=sequential_wkv)
                merge(by_shape.setdefault(f"B{shape[0]}xT{shape[1]}/chunk{chunk}", {}), entry)
        del records
        torch.cuda.empty_cache()
        print(f"[real] batch {bi + 1}/5 replayed {calls_per_batch[-1]} recurrence calls", flush=True)
    return {"perturbed_zero_initialised_parameters": perturbed, "calls_per_batch": calls_per_batch, "min_decay_w": w_min, "max_cumulative_neg_log_w_chunk%d" % args.chunks[0]: decay_max,
            "by_shape": by_shape, "note": "stats are the maximum over all calls of that shape; truth fields exist only for the first --real-truth-batches batches"}


if __name__ == "__main__":
    main()
