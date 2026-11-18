#!/usr/bin/env python
"""Bitwise parity of the wide-kernels port (d=64/80, head size 16) against the generic PyTorch/NVRTC path.

Small synthetic tensors and a few real-user batches; seconds of GPU. Each case compares outputs and every input
gradient bit for bit (torch.equal on the raw bits).

  .venv/Scripts/python.exe scripts/check_wide_kernels.py [--parts rows,copy,group_norm,wkv,model] [--output ...]
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

RESULTS = {}


def bits(t):
    t = t.detach().contiguous()
    return t.view(torch.int16) if t.dtype in (torch.bfloat16, torch.float16) else t.view(torch.int32) if t.dtype == torch.float32 else t


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(bits(a), bits(b))


def record(name, ok, detail=None):
    RESULTS[name] = {"ok": bool(ok), "detail": detail}
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail is not None else ""), flush=True)


def check_rows(device):
    from rwkv.model import exact_index
    for dtype in (torch.bfloat16, torch.float32):
        for C in (16, 32, 33, 64, 80, 160):
            for rows, n, pad in ((50, 400, False), (1000, 3000, False), (200, 5000, True), (7, 9000, True)):
                g = torch.Generator().manual_seed(C * 7 + rows)
                x = torch.randn(rows, C, generator=g).to(device, dtype).requires_grad_(True)
                index = torch.randint(0, rows, (n,), generator=g).to(device)
                if pad:
                    index[torch.rand(n, generator=g).to(device) < 0.3] = -1
                grad = torch.randn(n, C, generator=g).to(device, dtype)
                if pad:
                    grad[index < 0] = 0  # F50 precondition: padding sources carry +0 gradients
                ref = torch.index_select(x, 0, index.clamp(min=0))
                (gx,) = torch.autograd.grad(ref, x, grad)
                variants = {"plain": lambda: exact_index.ExactIndexSelect.apply(x, index.clamp(min=0))}
                if pad:
                    variants["padded"] = lambda: exact_index.ExactPaddedIndexSelect.apply(x, index)
                if C > 32:
                    clamped = index.clamp(min=0)
                    order, starts = exact_index.row_order(clamped, rows)
                    variants["ordered"] = lambda: exact_index.OrderedIndexSelect.apply(x, clamped, order, starts)
                for name, fn in variants.items():
                    out = fn()
                    (gy,) = torch.autograd.grad(out, x, grad)
                    record(f"rows {name} {str(dtype)[6:]} C={C} rows={rows} n={n} pad={pad}", same(out, ref) and same(gy, gx))


def check_copy(device):
    from rwkv.model import exact_index
    for dtype in (torch.bfloat16, torch.float32):
        for C in (64, 80):
            g = torch.Generator().manual_seed(C)
            rows, n = 3000, 1700
            x = torch.randn(rows, C, generator=g).to(device, dtype).requires_grad_(True)
            src = torch.randn(n, C, generator=g).to(device, dtype).requires_grad_(True)
            target = torch.randperm(rows, generator=g)[:n].to(device)
            grad = torch.randn(rows, C, generator=g).to(device, dtype)
            ref = x.index_copy(0, target, src)
            rgx, rgs = torch.autograd.grad(ref, (x, src), grad)
            out = exact_index.ExactIndexCopy.apply(x, target, src)
            gx, gs = torch.autograd.grad(out, (x, src), grad)
            record(f"index_copy {str(dtype)[6:]} C={C}", same(out, ref) and same(gx, rgx) and same(gs, rgs))


def check_group_norm(device):
    from rwkv.model.exact_group_norm import ExactGroupNorm
    for dtype in (torch.bfloat16, torch.float32):
        for C, G in ((32, 1), (64, 4), (80, 5), (48, 3)):
            for N in (100, 129, 4096, 70001):
                g = torch.Generator().manual_seed(C + N)
                norm = torch.nn.GroupNorm(G, C, eps=64e-5).to(device, dtype)
                with torch.no_grad():
                    norm.weight.copy_(1 + 0.1 * torch.randn(C, generator=g))
                    norm.bias.copy_(0.1 * torch.randn(C, generator=g))
                x = torch.randn(N, C, generator=g).to(device, dtype).requires_grad_(True)
                grad = torch.randn(N, C, generator=g).to(device, dtype)
                ref = norm(x)
                rgs = torch.autograd.grad(ref, (x, norm.weight, norm.bias), grad)
                out = ExactGroupNorm.apply(x, norm.weight, norm.bias, G, norm.eps)
                gs = torch.autograd.grad(out, (x, norm.weight, norm.bias), grad)
                record(f"group_norm {str(dtype)[6:]} C={C} G={G} N={N}",
                       same(out, ref) and all(same(a, b) for a, b in zip(gs, rgs)))


def wkv_inputs(B, T, H, K, dtype, device, seed):
    g = torch.Generator().manual_seed(seed)
    shape = (B, T, H, K)
    make = lambda scale=1.0: (scale * torch.randn(shape, generator=g)).to(device)  # noqa: E731
    k = torch.nn.functional.normalize(make(), dim=-1)
    a = torch.sigmoid(make())
    inputs = [make(0.5), k * a, make(0.5), torch.exp(-torch.exp(make(0.5) - 0.5)), a, k]
    values = [x.to(dtype if i != 3 else torch.float32).contiguous() for i, x in enumerate(inputs)]
    skip = (torch.rand((B, T), generator=g) < 0.3).to(device)
    grad = torch.randn(shape, generator=g).to(device, dtype)
    return values, skip, grad


def run_wkv(values, skip, grad):
    from expertium.model.rwkv_ops import RWKV7_WKV
    leaves = [v.clone().requires_grad_(True) for v in values]
    out = RWKV7_WKV.apply(*leaves, skip)
    return [out, *torch.autograd.grad(out, leaves, grad)]


def check_wkv(device):
    from rwkv.model import nvrtc
    from scripts.gpu_row_tiles import row_tiled_backend
    nvrtc.set_checkpoint_length(32)
    shapes = [(1, 1, 5), (3, 31, 5), (4, 128, 4), (7, 129, 5), (2, 383, 4), (300, 64, 5), (33, 200, 4)]
    for dtype in (torch.bfloat16, torch.float32):
        for B, T, H in shapes:
            values, skip, grad = wkv_inputs(B, T, H, 16, dtype, device, B * 1000 + T)
            reference = run_wkv(values, skip, grad)
            for rows in (1, 2, 4, 8):
                for tile in (32, 16):
                    with row_tiled_backend(rows=rows, tile=tile, exact=True):
                        candidate = run_wkv(values, skip, grad)
                    names = ["out", "dr", "dk", "dv", "dw", "da", "dkd"]
                    bad = [n for n, a, b in zip(names, candidate, reference) if not same(a, b)]
                    record(f"wkv K=16 {str(dtype)[6:]} B={B} T={T} H={H} rows={rows} tile={tile}", not bad, bad or None)
    nvrtc.set_checkpoint_length(32)


def real_batches():
    from rwkv.data_processing import get_rwkv_data
    from rwkv.prepare_batch import prepare
    from scripts.gpu_batch1 import DATA, make_samples
    frames = {user: get_rwkv_data(DATA, user) for user in (7, 12, 30, 41)}
    specs = [((7, 0, 700),), ((7, 0, 1500), (12, 0, 1500), (30, 0, 1500)), ((41, 0, 6000), (12, 1500, 3000))]
    batches = []
    for seed, spec in enumerate(specs):
        samples = []
        for user, start, end in spec:
            samples += make_samples(frames[user].iloc[start:end], user, 128)
        batches.append(prepare(samples, seed=2025 + seed, padding_factor=0.9, allowed_excess_in_one_step=20000))
    return batches


def set_switches(model, wide, skip_padding):
    for module in model.modules():
        if hasattr(module, "exact_shift"):
            module.exact_shift = True  # accepted F26, already active on the generic path
        if hasattr(module, "exact_index"):
            module.exact_index = wide
        if hasattr(module, "exact_group_norm"):
            module.exact_group_norm = wide
    model.skip_padding_sources = wide and skip_padding


def model_step(model, batch, device):
    moved = batch.to(device)
    moved.start = moved.start.to(torch.bfloat16)
    model.zero_grad(set_to_none=True)
    torch.manual_seed(1234)
    stats = model.get_loss(moved)
    stats.average_loss.backward()
    return [stats.average_loss.detach(), stats.p_curve, stats.p_imm] + [p.grad for p in model.parameters()]


def check_model(device):
    from contextlib import nullcontext
    from scripts.compare import training_models
    from scripts.gpu_row_tiles import row_tiled_backend
    from rwkv.model import nvrtc
    from expertium.variants import variant
    from expertium.model.variant_model import VariantSrsRWKV
    nvrtc.set_checkpoint_length(32)
    batches = real_batches()
    for name in ("w80", "w64", "w80-sequential", "w64-sequential"):
        _, model = training_models(VariantSrsRWKV, variant(name), device, torch.bfloat16, 2025)
        # Zero-initialized heads (GRU, p_linear) would stop every trunk gradient at initialization; perturb all
        # weights so the stream kernels carry nonzero gradients.
        generator = torch.Generator().manual_seed(7)
        with torch.no_grad():
            for parameter in model.parameters():
                noise = 0.05 * torch.randn(parameter.shape, generator=generator)
                parameter.add_(noise.to(parameter.device, parameter.dtype))
        model.train()
        for b, batch in enumerate(batches):
            set_switches(model, False, False)
            reference = model_step(model, batch, device)
            repeat = model_step(model, batch, device)
            unstable = [i for i, (a, c) in enumerate(zip(reference, repeat)) if (a is None) != (c is None)
                        or (a is not None and not same(a, c))]
            record(f"model {name} batch {b} generic path repeatable", not unstable, {"mismatched_tensors": len(unstable)})
            for rows, skip_padding in ((None, False), (4, False), (4, True), (2, True)):
                set_switches(model, True, skip_padding)
                context = row_tiled_backend(rows=rows, tile=32, exact=True) if rows else nullcontext()
                with context:
                    candidate = model_step(model, batch, device)
                bad = [i for i, (a, c) in enumerate(zip(reference, candidate)) if (a is None) != (c is None)
                       or (a is not None and not same(a, c))]
                nonzero = sum(int(t is not None and t.dim() > 0 and bool(t.ne(0).any())) for t in reference[3:])
                record(f"model {name} batch {b} ({batch.start.shape[0]}x{batch.start.shape[1]}) row_tiles={rows} "
                       f"skip_padding={skip_padding}", not bad and nonzero > 0.75 * (len(reference) - 3),
                       {"mismatched_tensors": len(bad), "of": len(reference), "nonzero_gradients": nonzero,
                        "none_gradients": sum(t is None for t in reference[3:]),
                        "loss": reference[0].item()})
        del model
        torch.cuda.empty_cache()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--parts", default="rows,copy,group_norm,wkv,model")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/wide-kernels/check-parity.json")
    args = parser.parse_args()
    global torch
    import torch
    torch.use_deterministic_algorithms(True)
    # As the R01 trainer (scripts/expertium_speed.py, jit_optimize false): no profiling-executor fusion, which
    # otherwise changes the scripted GRU head's arithmetic after the first calls (first step != later steps).
    torch._C._set_graph_executor_optimize(False)
    from rwkv.model import nvrtc
    nvrtc.register()
    device = torch.device("cuda")
    started = time.perf_counter()
    parts = args.parts.split(",")
    for part in parts:
        globals()[f"check_{part}"](device)
    failed = [name for name, value in RESULTS.items() if not value["ok"]]
    summary = {"parts": parts, "cases": len(RESULTS), "failed": failed, "seconds": time.perf_counter() - started,
               "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "results": RESULTS}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} passed; {summary['seconds']:.1f} s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
