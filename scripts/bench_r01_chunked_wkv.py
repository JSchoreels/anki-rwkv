"""F69: time the chunked WKV7 prototype against the native exact recurrence on the nine real E33 split shapes.

Device time from CUDA events around forward and backward (training-mode forward that saves for backward),
plus host launch time (CPU wall until the enqueue returns). Warm-up, repeated measurements, median/min/max/IQR.
Microbenchmark only: no training, no wiring. Needs the GPU claim."""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SHAPES = [(742, 256), (869, 256), (512, 256), (36177, 2), (7764, 4), (3359, 10), (318, 26), (695, 68), (4240, 12)]


def spread(values):
    values = sorted(values)
    q = statistics.quantiles(values, n=4) if len(values) >= 4 else [values[0], values[len(values) // 2], values[-1]]
    return {"median": statistics.median(values), "min": values[0], "max": values[-1], "iqr": q[2] - q[0], "n": len(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunks", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--shapes", type=int, default=len(SHAPES), help="use only the first N shapes (smoke)")
    args = parser.parse_args()
    out = ROOT / args.output
    if out.exists():
        parser.error("Use a new output directory")
    os.chdir(ROOT)
    from scripts.r01_workflow import configure_runtime
    configure_runtime(ROOT, 1)
    import torch
    from rwkv.model import nvrtc
    from rwkv.model.chunked_wkv import chunked_wkv
    from rwkv.model.native_launch import load
    from scripts.check_gpu_row_tiles import synthetic
    from scripts.experiment_provenance import process_provenance
    from scripts.gpu_row_tiles import compile_module
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    nvrtc.register()
    nvrtc.set_stream_launch(True)
    build = load()
    forward, backward = compile_module(torch.device("cuda", 0), torch.bfloat16, 4, 32, True)
    handles = (int(forward.kernel.ptr), int(backward.kernel.ptr))
    generator = torch.Generator(device="cuda")
    variants = {"native": lambda r, k, v, w, a, kd, s: torch.ops.rwkv_native.wkv(r, k, v, w, a, kd, s, handles[0], handles[1], 4, 32)}
    for chunk in args.chunks:
        variants[f"chunked{chunk}"] = lambda r, k, v, w, a, kd, s, c=chunk: chunked_wkv(r, k, v, w, a, kd, s, chunk=c)
    result = {"provenance": process_provenance(), "torch": torch.__version__, "native_build": build,
              "warmup": args.warmup, "repeats": args.repeats, "shapes": []}
    for index, (b, t) in enumerate(SHAPES[:args.shapes]):
        generator.manual_seed(20261013 + index)
        values, skip, grad = synthetic(torch, b, t, "random", generator)
        entry = {"B": b, "T": t, "variants": {}}
        torch.cuda.reset_peak_memory_stats()
        for name, fn in variants.items():
            fwd_ms, bwd_ms, host_fwd_ms, host_bwd_ms = [], [], [], []
            for step in range(args.warmup + args.repeats):
                leaves = [x.detach().clone().requires_grad_() for x in values]
                torch.cuda.synchronize()
                e = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                e[0].record()
                h0 = time.perf_counter()
                result_out = fn(*leaves, skip)
                h1 = time.perf_counter()
                e[1].record()
                result_out.backward(grad)
                h2 = time.perf_counter()
                e[2].record()
                torch.cuda.synchronize()
                if step >= args.warmup:
                    fwd_ms.append(e[0].elapsed_time(e[1]))
                    bwd_ms.append(e[1].elapsed_time(e[2]))
                    host_fwd_ms.append((h1 - h0) * 1000)
                    host_bwd_ms.append((h2 - h1) * 1000)
            entry["variants"][name] = {"fwd_ms": spread(fwd_ms), "bwd_ms": spread(bwd_ms),
                                       "fwd_bwd_median_ms": statistics.median(f + g for f, g in zip(fwd_ms, bwd_ms)),
                                       "host_fwd_ms": spread(host_fwd_ms), "host_bwd_ms": spread(host_bwd_ms),
                                       "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
            torch.cuda.reset_peak_memory_stats()
        native = entry["variants"]["native"]["fwd_bwd_median_ms"]
        for name, data in entry["variants"].items():
            data["fwd_bwd_vs_native"] = data["fwd_bwd_median_ms"] / native
        result["shapes"].append(entry)
        print(f"[bench] B{b}xT{t}: " + ", ".join(f"{n} {d['fwd_bwd_median_ms']:.3f} ms" for n, d in entry["variants"].items()), flush=True)
    totals = {name: sum(s["variants"][name]["fwd_bwd_median_ms"] for s in result["shapes"]) for name in variants}
    result["sum_fwd_bwd_median_ms_over_shapes"] = totals
    out.mkdir(parents=True)
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(totals))


if __name__ == "__main__":
    main()
