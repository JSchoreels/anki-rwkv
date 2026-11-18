#!/usr/bin/env python
"""GPU check for long-context training (docs/SMALL_MODEL.md section 5.3, P1): the NVRTC WKV-7 recurrence beyond
the validated 383 timesteps, and a memory/speed probe of a real update at 1,024-review chunks. A few minutes of GPU:
run it as one short queue ticket.

  parity  K = 16 and 32, T = 512, 1,024, 2,100 (a 1,024-review sample is about 2,050 rows with its queries):
          - generic NVRTC float32 forward and every input gradient against a float64 autograd reference
            (relative max error, threshold 1e-4 forward / 1e-3 gradients);
          - prefix: the first 383 outputs of a long run equal a 383-step run bit for bit (BF16 and float32);
          - row-tiled exact kernel (rows 4, tile 32; the training path of `row_tiles_exact`) equals the generic
            kernel bit for bit, forward and gradients, as scripts/check_wide_kernels.py checks up to 383.
  probe   one real 32,768-review update as 256 x 128 and as 32 x 1,024 chunks (same reviews): fork teacher targets
          (bf16) + w64 student forward/backward with KD, generic and row-tiled paths; median ms of 3 after warm-up,
          peak allocated MiB, finite loss. A probe, not a benchmark (no --timing claim needed for sizing).

  .venv/Scripts/python.exe scripts/check_long_wkv.py [--parts parity,probe] [--output artifacts/small-model/check-long-wkv.json]
"""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GPU_BATCH1_DATA_PATH", r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011")
os.environ.setdefault("CUPY_CACHE_DIR", str(ROOT / "artifacts/cache/cupy-cache"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
TEACHER = r"C:\Users\jonat\workspace\anki\qt\aqt\rwkv_inference\RWKV_trained_on_5000_10000.pth"
RESULTS = {}


def record(name, ok, detail=None):
    RESULTS[name] = {"ok": bool(ok), "detail": detail}
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def reference_wkv(values, skip, grad):
    """Float64 per-step recurrence with autograd (same update as rwkv_ops.single_timestep and the kernels)."""
    import torch
    from rwkv.model.rwkv_ops import single_timestep
    leaves = [v.detach().double().requires_grad_(True) for v in values]
    r, k, v, w, a, kd = leaves
    B, T, H, K = r.shape
    state = torch.zeros(B, H, K, K, dtype=torch.float64, device=r.device)
    outs = []
    for t in range(T):
        o, nxt = single_timestep(r[:, t], k[:, t], v[:, t], w[:, t], a[:, t], kd[:, t], state)
        outs.append(o)
        state = torch.where(skip[:, t, None, None, None], state, nxt)
    out = torch.stack(outs, 1)
    return [out.detach(), *torch.autograd.grad(out, leaves, grad.double())]


def rel(a, b):
    return float((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30))


def check_parity(device):
    import torch
    from rwkv.model import nvrtc
    from scripts import check_wide_kernels
    from scripts.check_wide_kernels import wkv_inputs, run_wkv, same
    check_wide_kernels.torch = torch  # that module imports torch inside its main()
    from scripts.gpu_row_tiles import row_tiled_backend
    nvrtc.set_checkpoint_length(32)
    names = ["out", "dr", "dk", "dv", "dw", "da", "dkd"]
    values, skip, grad = wkv_inputs(1, 384, 2, 16, torch.float32, device, 1)
    try:
        run_wkv(values, skip, grad)
        refused = False
    except ValueError:
        refused = True
    record("default guard still refuses T = 384", refused, refused)
    previous = nvrtc.set_max_timesteps(4096)
    try:
        for K in (16, 32):
            for T in (512, 1024, 2100):
                values, skip, grad = wkv_inputs(2, T, 2, K, torch.float32, device, K * 10000 + T)
                candidate = run_wkv(values, skip, grad)
                reference = reference_wkv(values, skip, grad)
                errors = {n: rel(c, r) for n, c, r in zip(names, candidate, reference)}
                ok = errors["out"] < 1e-4 and all(errors[n] < 1e-3 for n in names[1:])
                record(f"generic NVRTC float32 K={K} T={T} vs float64 reference (relative max error)", ok,
                       {n: f"{e:.1e}" for n, e in errors.items()})
                for dtype in (torch.bfloat16, torch.float32):
                    values, skip, grad = wkv_inputs(3, T, 2, K, dtype, device, K * 20000 + T)
                    long_out = run_wkv(values, skip, grad)[0]
                    short = [x[:, :383].contiguous() for x in values]
                    short_out = run_wkv(short, skip[:, :383].contiguous(), grad[:, :383].contiguous())[0]
                    record(f"prefix K={K} T={T} {str(dtype)[6:]}: first 383 outputs bit-identical",
                           same(long_out[:, :383].contiguous(), short_out), None)
                    reference = run_wkv(values, skip, grad)
                    with row_tiled_backend(rows=4, tile=32, exact=True):
                        tiled = run_wkv(values, skip, grad)
                    bad = [n for n, a, b in zip(names, tiled, reference) if not same(a, b)]
                    record(f"row-tiled exact K={K} T={T} {str(dtype)[6:]} == generic (forward + gradients)", not bad,
                           bad or None)
    finally:
        nvrtc.set_max_timesteps(previous)


def probe_batches(chunk_values):
    """The same 32,768 training reviews (users 101+, whole 1,024-review blocks) cut at each chunk length."""
    from rwkv.data_processing import get_rwkv_data
    from rwkv.prepare_batch import prepare
    from scripts.gpu_batch1 import DATA
    from scripts.gpu_r01_samples import samples_for
    pieces, total, user = [], 0, 101
    while total < 32768:
        try:
            frame = get_rwkv_data(DATA, user)
        except Exception:  # noqa: BLE001 - absent user ids are skipped
            user += 1
            continue
        take = min((len(frame) // 1024) * 1024, 32768 - total)
        if take:
            pieces.append((user, frame.iloc[:take]))
            total += take
        user += 1
    batches = {}
    for chunk in chunk_values:
        samples = []
        for user, section in pieces:
            samples += samples_for(section, user, {"chunk_reviews": chunk})
        batches[chunk] = prepare(samples, seed=2025, padding_factor=0.9, allowed_excess_in_one_step=20000)
    return batches, [user for user, _ in pieces]


def check_probe(device):
    import statistics
    import torch
    from contextlib import nullcontext
    from rwkv.model import nvrtc
    from scripts.compare import training_models
    from scripts.gpu_row_tiles import row_tiled_backend
    from scripts import expertium_options as xo
    from expertium.variants import variant
    from expertium.model.variant_model import VariantSrsRWKV
    nvrtc.set_checkpoint_length(32)
    previous = nvrtc.set_max_timesteps(4096)
    try:
        batches, users = probe_batches((128, 1024))
        teacher = xo.Teacher({"teacher": TEACHER, "alpha": 0.9, "decay_alpha": 0.5}, device, torch.bfloat16)
        _, model = training_models(VariantSrsRWKV, variant("w64"), device, torch.bfloat16, 2025)
        model.train()
        for chunk, batch in batches.items():
            moved = batch.to(device)
            moved.start = moved.start.to(torch.bfloat16)
            rows = {"samples": int(moved.start.shape[0]), "max_rows": int(moved.start.shape[1])}
            for path in ("generic", "row_tiles"):
                context = (lambda: row_tiled_backend(rows=4, tile=32, exact=True)) if path == "row_tiles" else nullcontext
                times, losses = [], []
                torch.cuda.reset_peak_memory_stats(device)
                for step in range(4):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    with context():
                        targets = teacher.targets(moved)
                        model.zero_grad(set_to_none=True)
                        torch.manual_seed(1234)
                        stats = model.get_loss(moved, kd_mix=targets)
                        stats.average_loss.backward()
                        torch.cuda.synchronize()
                    if step:
                        times.append((time.perf_counter() - started) * 1000)
                    losses.append(float(stats.average_loss))
                peak = torch.cuda.max_memory_allocated(device) / 2**20
                record(f"probe {path} chunks of {chunk} reviews (32,768 reviews, teacher + student fwd/bwd)",
                       all(map(lambda x: x == x and abs(x) < 1e6, losses)),
                       {**rows, "median_ms": round(statistics.median(times), 1),
                        "reviews_per_second_step_only": round(32768 / statistics.median(times) * 1000),
                        "peak_allocated_mib": round(peak), "loss": round(losses[-1], 5), "users": users})
    finally:
        nvrtc.set_max_timesteps(previous)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--parts", default="parity,probe")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/small-model/check-long-wkv.json")
    args = parser.parse_args()
    import torch
    torch.use_deterministic_algorithms(True)
    torch._C._set_graph_executor_optimize(False)
    from rwkv.model import nvrtc
    nvrtc.register()
    device = torch.device("cuda")
    started = time.perf_counter()
    for part in args.parts.split(","):
        globals()[f"check_{part}"](device)
    failed = [name for name, value in RESULTS.items() if not value["ok"]]
    summary = {"parts": args.parts, "cases": len(RESULTS), "failed": failed, "seconds": time.perf_counter() - started,
               "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "results": RESULTS}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} passed; {summary['seconds']:.1f} s; {args.output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
