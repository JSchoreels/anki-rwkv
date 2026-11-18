#!/usr/bin/env python
"""CPU checks for the long-context training option `chunk_reviews` (docs/SMALL_MODEL.md section 5.3, P1).

  1. samples_for: unset = 128-review samples (unchanged); chunk_reviews 1,024 gives 1,024-review samples; the F30/F42
     fast columnar construction the w64 tracks use equals the original construction at 1,024 (every batch tensor).
  2. A 1,024-review batch runs through the w64 student with KD from the fork teacher (CPU reference recurrence):
     finite loss and gradients; the teacher cache round-trips it bit for bit.
  3. nvrtc.validate: 383 timesteps by default, set_max_timesteps raises and restores the limit.
  4. expertium_decay.prepare records chunk_reviews in its protocol; the R01 prefetch path accepts the option.

  .venv/Scripts/python.exe scripts/check_long_chunks.py [--output artifacts/small-model/check-long-chunks.json]
"""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path(os.environ.get("GPU_BATCH1_DATA_PATH", r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011"))
os.environ["GPU_BATCH1_DATA_PATH"] = str(DATA)
os.environ["RWKV_NO_JIT"] = "1"
TEACHER = Path(r"C:\Users\jonat\workspace\anki\qt\aqt\rwkv_inference\RWKV_trained_on_5000_10000.pth")
SOURCE_RUN = ROOT / "artifacts/accuracy/experiments/scale-w64kd-gate/run"


def tensors_equal(a, b, path="batch"):
    import torch
    if isinstance(a, torch.Tensor):
        return [] if isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b) else [path]
    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            return [path]
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in tensors_equal(x, y, f"{path}[{i}]")]
    if hasattr(a, "__dict__") and not isinstance(a, type):
        keys = sorted(set(vars(a)) | set(vars(b)))
        return [p for k in keys for p in tensors_equal(getattr(a, k, None), getattr(b, k, None), f"{path}.{k}")]
    return [] if a == b else [path]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/small-model/check-long-chunks.json"))
    parser.add_argument("--user", type=int, default=7)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    from rwkv.data_processing import get_rwkv_data
    from rwkv.prepare_batch import prepare
    from rwkv.model import nvrtc
    from expertium import variants
    from scripts.gpu_r01_samples import samples_for
    from scripts import expertium_options as xo
    started = time.perf_counter()
    results = {}

    def record(name, value, ok):
        results[name] = {"value": value, "ok": bool(ok)}
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {value}", flush=True)

    frame = get_rwkv_data(DATA, args.user)
    section = frame.iloc[:2048]
    default = samples_for(section, args.user, {})
    long = samples_for(section, args.user, {"chunk_reviews": 1024})
    record("samples_for: unset = 128-review samples, chunk_reviews 1024 = 1,024-review samples",
           [len(default), len(long)], len(default) == 16 and len(long) == 2)
    fast_options = {"fast_samples": True, "columnar_samples": True, "chunk_reviews": 1024}
    plain = prepare(long, seed=2025)
    fast = prepare(samples_for(section, args.user, fast_options), seed=2025)
    diff = tensors_equal(plain, fast)
    record("fast columnar samples == original construction at 1,024 (every batch field)",
           {"rows": list(plain.start.shape), "differing": diff[:5]}, not diff)

    student = variants.build("w64")
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.add_(0.05 * torch.randn(parameter.shape, generator=generator))
    student.train()
    with tempfile.TemporaryDirectory() as folder:
        spec = {"teacher": str(TEACHER), "alpha": 0.9, "decay_alpha": 0.5, "teacher_dtype": "float32", "cache": folder}
        teacher = xo.Teacher(spec, torch.device("cpu"), torch.float32, {"architecture": "w64", "chunk_reviews": 1024})
        key = (args.user, 0, 2048, 2025)
        written = teacher.targets(plain, key=key)
        read = teacher.targets(plain, key=key)
        labels = plain.labels
        ahead = ((1 - labels[:, :, 6]) * labels[:, :, 4]).bool()
        query = (labels[:, :, 6] * labels[:, :, 4]).bool()
        record("teacher cache round-trips a 1,024-review batch on the rows the loss reads",
               dict(teacher.cache_stats), torch.equal(written[0][ahead], read[0][ahead])
               and torch.equal(written[1][query], read[1][query]) and teacher.cache_stats["hits"] == 1)
        stats = student.get_loss(plain, kd_mix=read)
        stats.average_loss.backward()
        grads = [p.grad for p in student.parameters() if p.grad is not None]
        record("w64 student + KD on 1,024-review samples: finite loss and gradients",
               {"loss": round(stats.average_loss.item(), 5), "rows": list(plain.start.shape), "gradients": len(grads)},
               torch.isfinite(stats.average_loss).item() and all(torch.isfinite(g).all() for g in grads))

    def refused_at(steps):
        x = torch.zeros(1, steps, 1, 16)
        try:
            nvrtc.validate([x] * 6, torch.zeros(1, steps, dtype=torch.bool))
        except ValueError as error:
            return "timesteps" in str(error)
        return False
    default_refuses = refused_at(384) and not refused_at(383)
    previous = nvrtc.set_max_timesteps(4096)
    raised = not refused_at(2100) and refused_at(4097)
    nvrtc.set_max_timesteps(previous)
    record("nvrtc limit: 383 by default, raised by set_max_timesteps, restored",
           {"default": default_refuses, "raised": raised, "restored": nvrtc._max_timesteps},
           default_refuses and raised and nvrtc._max_timesteps == 383)

    import scripts.gpu_r01_prefetch as prefetch
    source = Path(prefetch.__file__).read_text(encoding="utf-8")
    record("R01 ordered prefetch accepts the chunk_reviews option", None, '"chunk_reviews"}' in source)
    if (SOURCE_RUN / "run.json").exists():
        out = ROOT / "artifacts/small-model/check-long-chunks-decay-tmp"
        shutil.rmtree(out, ignore_errors=True)
        from scripts.expertium_decay import prepare as decay_prepare
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            decay_prepare(SOURCE_RUN, 3, out, 0.25, "one_minus_sin", "none", 2,
                          {"chunk_reviews": 1024, "batch_chunks": 32}, 0.0)
        protocol = json.loads((out / "protocol.json").read_text())
        options = json.loads((out / "decay.json").read_text())["options"]
        shutil.rmtree(out, ignore_errors=True)
        record("expertium_decay.prepare: protocol and options carry chunk_reviews 1024 / batch_chunks 32",
               [protocol["training"]["chunk_reviews"], options.get("chunk_reviews"), options.get("batch_chunks")],
               protocol["training"]["chunk_reviews"] == 1024 and options.get("chunk_reviews") == 1024
               and options.get("batch_chunks") == 32)
    results["seconds"] = time.perf_counter() - started
    failed = [name for name, value in results.items() if isinstance(value, dict) and not value["ok"]]
    results["failed"] = failed
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"{len(failed)} failed; {results['seconds']:.1f} s; {output}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
