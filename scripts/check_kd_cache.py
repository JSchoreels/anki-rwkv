#!/usr/bin/env python
"""CPU check of the distillation teacher cache (scripts/expertium_options.Teacher, "cache"; docs/SMALL_MODEL.md).

On real batches of one user, with the fork teacher (float32 on CPU) and a perturbed w64 student:
  1. a cache miss returns the online targets bit for bit and writes a file;
  2. a cache hit gives the student the same loss and every gradient bit for bit as the online targets;
  3. decay_alpha is applied on a hit as on a miss;
  4. another batch under the same key, and a cache written by another teacher, are refused; batch-relevant options
     get their own cache folder, kernel switches (WIDTH32_ONLY) do not;
  5. stored bytes per labelled review (sizing for 5k / 8k users).

  .venv/Scripts/python.exe scripts/check_kd_cache.py [--output artifacts/small-model/check-kd-cache.json]
"""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path(os.environ.get("GPU_BATCH1_DATA_PATH", r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011"))
os.environ["GPU_BATCH1_DATA_PATH"] = str(DATA)
# TorchScript's profiling executor re-fuses after the first calls, so repeated passes differ in the last bit; eager
# mode makes two identical passes bit-identical (the control below), which a bitwise comparison needs.
os.environ["RWKV_NO_JIT"] = "1"
TEACHER = Path(r"C:\Users\jonat\workspace\anki\qt\aqt\rwkv_inference\RWKV_trained_on_5000_10000.pth")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/small-model/check-kd-cache.json"))
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
    other = prepare(make_samples(frame.iloc[640:1280], args.user, 128), seed=2025 + args.user + 640)

    def student_pass(targets):
        torch.manual_seed(2025)
        student = variants.build("w64")
        generator = torch.Generator().manual_seed(7)
        with torch.no_grad():
            for parameter in student.parameters():
                parameter.add_(0.05 * torch.randn(parameter.shape, generator=generator))
        student.train()
        torch.manual_seed(11)
        stats = student.get_loss(batch, kd_mix=targets)
        stats.average_loss.backward()
        return stats.average_loss.detach(), [p.grad.detach().clone() for p in student.parameters() if p.grad is not None]

    with tempfile.TemporaryDirectory() as folder:
        spec = {"teacher": str(TEACHER), "alpha": 0.9, "decay_alpha": 0.5, "teacher_dtype": "float32"}
        online = xo.Teacher(spec, torch.device("cpu"), torch.float32)
        track = {"architecture": "w64", "batch_chunks": 256, "fast_samples": True, "training_seed": 2025}
        cached = xo.Teacher({**spec, "cache": folder}, torch.device("cpu"), torch.float32, track)
        same_folder = xo.Teacher({**spec, "cache": folder}, torch.device("cpu"), torch.float32,
                                 {**track, "row_tiles_exact": True, "distillation": {"alpha": 0.1}}).cache
        other_folder = xo.Teacher({**spec, "cache": folder}, torch.device("cpu"), torch.float32,
                                  {**track, "batch_chunks": 32, "chunk_reviews": 1024}).cache
        record("cache folder: kernel switch and distillation spec share it, batch options do not",
               [cached.cache.name, other_folder.name], same_folder == cached.cache and other_folder != cached.cache)
        muon = xo.Teacher({**spec, "cache": folder}, torch.device("cpu"), torch.float32,
                          {**track, "optimizer": {"kind": "muon"}})
        iter53 = xo.Teacher({**spec, "cache": folder}, torch.device("cpu"), torch.float32,
                            {**track, "optimizer": "expertium-iter53"})
        record("cache folder: the optimizer does not change it (SM20), the manifest still matches",
               [cached.cache.name, muon.cache.name, iter53.cache.name],
               muon.cache == cached.cache == iter53.cache)
        key = (args.user, 0, 640, 2025 + args.user)
        reference = online.targets(batch)
        miss = cached.targets(batch, key=key, labels=batch.labels)
        record("miss returns the online targets bit for bit, one file written",
               cached.cache_stats, all(torch.equal(a, b) for a, b in zip(reference[:2], miss[:2]))
               and miss[2] == reference[2] and cached.cache_stats["writes"] == 1)
        hit = cached.targets(batch, key=key)
        loss_online, grads_online = student_pass(reference)
        loss_again, grads_again = student_pass(reference)
        record("control: two online passes are bit-identical", len(grads_again), torch.equal(loss_online, loss_again)
               and all(torch.equal(a, b) for a, b in zip(grads_online, grads_again)))
        loss_hit, grads_hit = student_pass(hit)
        same = torch.equal(loss_online, loss_hit) and len(grads_online) == len(grads_hit) and \
            all(torch.equal(a, b) for a, b in zip(grads_online, grads_hit))
        record("hit: student loss and every gradient bit-identical to online targets",
               {"loss": loss_online.item(), "gradients": len(grads_hit), "stats": dict(cached.cache_stats)},
               same and cached.cache_stats["hits"] == 1)
        decayed = cached.targets(batch, decaying=True, key=key)
        record("decay_alpha applied on a hit", decayed[2], decayed[2] == 0.5 and torch.equal(decayed[0], hit[0]))
        try:
            cached.targets(other, key=key)
            refused = False
        except RuntimeError:
            refused = True
        record("another batch under the same key is refused", refused, refused)
        manifest = cached.cache / "manifest.json"
        manifest.write_text(manifest.read_text().replace(cached.sha256, "0" * 64))
        try:
            xo.Teacher({**spec, "cache": folder}, torch.device("cpu"), torch.float32, track)
            refused = False
        except ValueError:
            refused = True
        record("a cache written by another teacher is refused", refused, refused)
        labels = batch.labels
        labelled = int((labels[:, :, 4] > 0).sum())
        reviews = 640
        size = sum(f.stat().st_size for f in cached.cache.glob("*.pt"))
        stored = torch.load(next(cached.cache.glob("*.pt")), weights_only=True)
        payload = 4 * (stored["curve"].numel() + stored["rating"].numel())
        per_review = payload / reviews
        record("payload bytes per training review (float32; a 32k-review training file adds ~1 KB of header)",
               {"file_bytes": size, "payload_bytes": payload, "labelled_rows": labelled, "reviews": reviews,
                "per_review": round(per_review, 2),
                "gate_5k_GB": round(per_review * 369_477_375 / 1e9, 2),
                "final_8k_GB": round(per_review * 581_942_738 / 1e9, 2)}, per_review < 64)
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
