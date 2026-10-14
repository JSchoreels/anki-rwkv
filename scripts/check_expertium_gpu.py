#!/usr/bin/env python
"""GPU checks for the Expertium catch-up (docs/EXPERTIUM_CATCH_UP.md); run through scripts/r01_queue.py.

1. NVRTC recurrence at head size 16 (width-80 variant) against the PyTorch reference recurrence, float32 and
   BF16, forward outputs and all six input gradients; head size 32 alongside as the control.
2. The w80 model on CUDA (NVRTC, BF16 child of a float32 master, R01 dtype policy) against the same weights on
   CPU in float32: loss, curve and instant predictions on one real user's batch, finite gradients.

  .venv/Scripts/python.exe scripts/check_expertium_gpu.py [--output artifacts/expertium-catch-up/check-gpu.json]
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


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/expertium-catch-up/check-gpu.json"))
    args = parser.parse_args()
    import torch
    from rwkv.model import nvrtc
    nvrtc.register()
    from expertium.model.rwkv_ops import RWKV7_WKV, reference_rwkv7
    results, started = {}, time.perf_counter()

    def record(name, value, ok):
        results[name] = {"value": value, "ok": bool(ok)}
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {value}", flush=True)

    device = torch.device("cuda")
    for head, heads in ((32, 1), (16, 5)):
        for dtype, tolerance in ((torch.float32, 2e-4), (torch.bfloat16, 3e-2)):
            generator = torch.Generator(device="cpu").manual_seed(head)
            B, T = 4, 150
            shape = (B, T, heads, head)
            make = lambda scale=1.0: (scale * torch.randn(shape, generator=generator)).to(device)  # noqa: E731
            k = torch.nn.functional.normalize(make(), dim=-1)
            a = torch.sigmoid(make())
            inputs = [make(0.5), k * a, make(0.5), torch.exp(-torch.exp(make(0.5) - 0.5)), a, k]
            skip = (torch.rand((B, T), generator=generator) < 0.3).to(device)
            values = [x.to(dtype if i != 3 else torch.float32).contiguous().requires_grad_(True) for i, x in enumerate(inputs)]
            reference = [x.detach().float().requires_grad_(True) for x in values]
            grad = torch.randn(shape, generator=generator).to(device)
            out = RWKV7_WKV.apply(*values, skip)
            (out.float() * grad).sum().backward()
            expected = reference_rwkv7(*reference, skip)
            (expected * grad).sum().backward()
            forward = (out.float() - expected).abs().max().item() / max(1.0, expected.abs().max().item())
            backward = max(((v.grad.float() - r.grad).abs().max() / r.grad.abs().max().clamp(min=1.0)).item()
                           for v, r in zip(values, reference))
            record(f"NVRTC K={head} {str(dtype)[6:]} vs reference (forward, max rel grad)", [forward, backward],
                   forward < tolerance and backward < tolerance * 5)

    from rwkv.data_processing import get_rwkv_data
    from rwkv.prepare_batch import prepare
    from scripts.compare import training_models
    from scripts.gpu_batch1 import DATA, make_samples
    from expertium.variants import variant
    from expertium.model.variant_model import VariantSrsRWKV
    frame = get_rwkv_data(DATA, 7)
    batch = prepare(make_samples(frame.iloc[:300], 7, 128)[:1], seed=2025)
    for name in ("w80", "w80-sequential", "w64"):
        master, child = training_models(VariantSrsRWKV, variant(name), device, torch.bfloat16, 2025)
        cpu = VariantSrsRWKV(variant(name)).float()
        cpu.load_state_dict({k: v.cpu() for k, v in master.state_dict().items()})
        child.eval()
        cpu.eval()
        moved = batch.to(device)
        moved.start = moved.start.to(torch.bfloat16)
        with torch.no_grad():
            gpu_stats = child.get_loss(moved)
            cpu_stats = cpu.get_loss(batch)
        labelled = batch.labels[0, :, 4] > 0
        gap = max((gpu_stats.p_curve[0].float().cpu() - cpu_stats.p_curve[0])[labelled].abs().max().item(),
                  (gpu_stats.p_imm[0].float().cpu() - cpu_stats.p_imm[0])[labelled].abs().max().item())
        record(f"{name} CUDA BF16 vs CPU float32 predictions (max abs)", gap, gap < 2e-2)
        child.train()
        torch.manual_seed(0)
        stats = child.get_loss(moved)
        stats.average_loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in child.parameters())
        record(f"{name} CUDA training loss and gradients finite", stats.average_loss.item(),
               finite and torch.isfinite(stats.average_loss).item())
    results["seconds"] = time.perf_counter() - started
    results["gpu"] = torch.cuda.get_device_name(0)
    failed = [n for n, v in results.items() if isinstance(v, dict) and not v["ok"]]
    results["failed"] = failed
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    print(f"{len(failed)} failed; {results['seconds']:.1f} s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
