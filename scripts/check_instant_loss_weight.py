#!/usr/bin/env python
"""CPU check of the opt-in instant_loss_weight training option (docs/SMALL_MODEL.md, SM26).

On one real user's batch, a perturbed w64 student, with and without distillation targets:
  1. option absent vs weight 1.0 vs the loss code before SM26 (variant_model.py before the commit adding it): loss, per-review
     loss tensor and every gradient bit for bit;
  2. weight 2.0: loss = (loss at weight 1) + 1 x instant average, i.e. curve part + 2 x instant part with the
     regularizers unchanged; gradients differ; evaluation outputs (p_curve, p_imm) unchanged;
  3. the KD cache folder (digest of the batch options) is identical with and without the option.

  .venv/Scripts/python.exe scripts/check_instant_loss_weight.py [--output artifacts/small-model/check-instant-loss-weight.json]
"""
from argparse import ArgumentParser
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path(os.environ.get("GPU_BATCH1_DATA_PATH", r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011"))
os.environ["GPU_BATCH1_DATA_PATH"] = str(DATA)
os.environ["RWKV_NO_JIT"] = "1"  # eager mode: two identical passes are bit-identical
TEACHER = Path(r"C:\Users\jonat\workspace\anki\qt\aqt\rwkv_inference\RWKV_trained_on_5000_10000.pth")


def previous_variant_class(tmp):
    """VariantSrsRWKV exactly as before SM26 (variant_model.py in the parent of the first commit that adds the option)."""
    first = subprocess.run(["git", "log", "--reverse", "--format=%H", "-S", "instant_loss_weight", "HEAD", "--",
                            "expertium/model/variant_model.py"], cwd=ROOT, capture_output=True, text=True,
                           check=True).stdout.split()[0]
    base = subprocess.run(["git", "rev-parse", f"{first}^"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()
    source = subprocess.run(["git", "show", f"{base}:expertium/model/variant_model.py"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout
    assert "instant_loss_weight" not in source
    path = Path(tmp) / "variant_model_before_sm26.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("variant_model_before_sm26", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VariantSrsRWKV, base


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/small-model/check-instant-loss-weight.json"))
    parser.add_argument("--user", type=int, default=7)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    from rwkv.data_processing import get_rwkv_data
    from rwkv.prepare_batch import prepare
    from scripts.gpu_batch1 import make_samples
    from expertium import variants
    from scripts import expertium_options as xo
    started = time.perf_counter()
    results = {}

    def record(name, value, ok):
        results[name] = {"value": value, "ok": bool(ok)}
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {value}", flush=True)

    frame = get_rwkv_data(DATA, args.user)
    batch = prepare(make_samples(frame.iloc[:640], args.user, 128), seed=2025 + args.user)
    kd = xo.Teacher({"teacher": str(TEACHER), "alpha": 0.9, "teacher_dtype": "float32"}, torch.device("cpu"),
                    torch.float32).targets(batch)

    def run(cls, weight, kd_mix, sync_free=False):
        torch.manual_seed(2025)
        student = cls(variants.variant("w64"))
        generator = torch.Generator().manual_seed(7)
        with torch.no_grad():
            for parameter in student.parameters():
                parameter.add_(0.05 * torch.randn(parameter.shape, generator=generator))
        student.train()
        student.sync_free_loss = sync_free
        if weight is not False:
            student.instant_loss_weight = weight
        torch.manual_seed(11)
        stats = student.get_loss(batch, kd_mix=kd_mix)
        stats.average_loss.backward()
        grads = [p.grad.detach().clone() for p in student.parameters() if p.grad is not None]
        return stats, grads

    def same(a, b):
        return (torch.equal(a[0].average_loss, b[0].average_loss) and torch.equal(a[0].loss_tensor, b[0].loss_tensor)
                and len(a[1]) == len(b[1]) and all(torch.equal(x, y) for x, y in zip(a[1], b[1])))

    from expertium.model.variant_model import VariantSrsRWKV
    with tempfile.TemporaryDirectory() as tmp:
        Before, base = previous_variant_class(tmp)
        for label, kd_mix, sync_free in (("plain", None, False), ("kd + sync_free_loss", kd, True)):
            old = run(Before, False, kd_mix, sync_free)
            absent = run(VariantSrsRWKV, False, kd_mix, sync_free)
            one = run(VariantSrsRWKV, 1.0, kd_mix, sync_free)
            record(f"{label}: option absent == code before SM26 ({base[:8]}), loss + loss tensor + gradients bitwise",
                   old[0].average_loss.item(), same(old, absent))
            record(f"{label}: weight 1.0 == option absent bitwise", one[0].average_loss.item(), same(absent, one))
            two = run(VariantSrsRWKV, 2.0, kd_mix, sync_free)
            instant = absent[0].imm_avg
            expected = absent[0].average_loss + instant
            gap = (two[0].average_loss - expected).abs().item()
            record(f"{label}: weight 2.0 loss = loss(1.0) + instant average (curve + 2 x instant + regularizers)",
                   {"w1": absent[0].average_loss.item(), "w2": two[0].average_loss.item(), "instant": instant.item(),
                    "abs_gap": gap}, gap < 1e-6 * max(1.0, abs(expected.item())))
            differs = sum(not torch.equal(x, y) for x, y in zip(absent[1], two[1]))
            record(f"{label}: weight 2.0 gradients differ", f"{differs}/{len(two[1])} tensors", differs > 0)
            record(f"{label}: weight 2.0 reported predictions and instant average unchanged",
                   True, torch.equal(absent[0].p_curve, two[0].p_curve) and torch.equal(absent[0].p_imm, two[0].p_imm)
                   and torch.equal(absent[0].imm_avg, two[0].imm_avg))
        # 3. KD cache key
        spec = {"teacher": str(TEACHER), "alpha": 0.9, "decay_alpha": 0.5, "teacher_dtype": "float32", "cache": tmp}
        track = {"architecture": "w64", "batch_chunks": 256, "fast_samples": True, "training_seed": 2025,
                 "sync_free_loss": True}
        plain = xo.Teacher(spec, torch.device("cpu"), torch.float32, track)
        weighted = xo.Teacher(spec, torch.device("cpu"), torch.float32, {**track, "instant_loss_weight": 2.0})
        record("KD cache digest identical with and without instant_loss_weight",
               [plain.cache.name, weighted.cache.name], plain.cache == weighted.cache)
    ok = all(entry["ok"] for entry in results.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"ok": ok, "seconds": time.perf_counter() - started, "checks": results}, indent=2))
    print("ALL OK" if ok else "FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
