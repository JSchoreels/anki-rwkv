#!/usr/bin/env python
"""CPU check of the `fork128` architecture preset and scripts/fork_checkpoint.py (docs/SMALL_MODEL.md).

On benchmark users' real batches:
  1. fork128 has 2,762,884 parameters and loads the fork's .pth strictly (also through fork_checkpoint.load_fork_state).
  2. Its eval-mode predictions (curve and instant) equal the upstream rwkv SrsRWKV loaded with the same weights (<= 1e-5).
  3. {"base": "fork128", "zero_features": [22]} equals the upstream fork with features2card.0.weight[:, 22] zeroed
     (<= 1e-5), and differs from the unzeroed fork (so the test is not vacuous).
  4. One CPU training step on a short sequence: finite loss, finite non-zero gradients, zero gradient on the dead
     input-22 column, finite parameters after an optimizer step.

  CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe scripts/check_fork_preset.py [--users 1000 2000] [--reviews 1500]
"""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path(os.environ.get("GPU_BATCH1_DATA_PATH", r"C:\Users\jonat\workspace\anki-rwkv\data\anki-revlogs-10k-original-20261011"))
os.environ["GPU_BATCH1_DATA_PATH"] = str(DATA)
TOLERANCE = 1e-5


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, nargs="+", default=[1000, 2000])
    parser.add_argument("--reviews", type=int, default=1500)
    parser.add_argument("--train-reviews", type=int, default=512)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    from argparse import Namespace
    from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG as UPSTREAM_CONFIG
    from rwkv.data_processing import get_rwkv_data
    from rwkv.model.srs_model import SrsRWKV as UpstreamSrsRWKV
    from rwkv.prepare_batch import prepare
    from rwkv.train_rwkv import get_optimizer
    from scripts.fork_checkpoint import FORK, load_fork_state
    from scripts.gpu_batch1 import make_samples
    from expertium import variants
    started = time.perf_counter()
    results = {}

    def record(name, value, ok):
        results[name] = {"value": value, "ok": bool(ok)}
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {value}", flush=True)

    fork_state = torch.load(FORK, map_location="cpu", weights_only=True)
    fork = UpstreamSrsRWKV(UPSTREAM_CONFIG)
    fork.load_state_dict(fork_state, strict=True)
    fork.eval()
    zeroed = UpstreamSrsRWKV(UPSTREAM_CONFIG)
    zeroed.load_state_dict(fork_state, strict=True)
    with torch.no_grad():
        zeroed.features2card[0].weight[:, 22] = 0.0
    zeroed.eval()

    preset = variants.build("fork128")
    count = sum(p.numel() for p in preset.parameters())
    record("fork128 parameters", count, count == variants.FORK128_PARAMETERS == 2_762_884)
    converted = load_fork_state(FORK)
    record("fork128 strict load of the converted fork state", preset.load_state_dict(converted, strict=True) is not None, True)
    preset.eval()
    label_free = variants.build({"base": "fork128", "zero_features": [22]})
    named = variants.build("fork128-nolabel")
    record("fork128-nolabel is fork128 + zero_features [22]", named.zero_features,
           named.zero_features == [22] and sum(p.numel() for p in named.parameters()) == variants.FORK128_PARAMETERS)
    label_free.load_state_dict(converted, strict=True)
    label_free.eval()
    dead = load_fork_state(FORK, zero_input=22)
    record("convert --zero-input 22 zeroes only that column",
           float(dead["features2card.0.weight"][:, 22].abs().sum()),
           float(dead["features2card.0.weight"][:, 22].abs().sum()) == 0.0
           and torch.equal(dead["features2card.0.weight"][:, :22], converted["features2card.0.weight"][:, :22]))

    def gap(a, b, labelled):
        return max((a.p_curve - b.p_curve)[0][labelled].abs().max().item(), (a.p_imm - b.p_imm)[0][labelled].abs().max().item())

    for user in args.users:
        frame = get_rwkv_data(DATA, user)
        sample = make_samples(frame.iloc[:args.reviews], user, 128)[0]
        batch = prepare([sample], seed=2025)
        labelled = batch.labels[0, :, 4] > 0
        label_types = batch.start[:, 22].float()
        with torch.no_grad():
            reference = fork.get_loss(batch)
            ours = preset.get_loss(batch)
            reference_zero = zeroed.get_loss(batch)
            ours_zero = label_free.get_loss(batch)
        record(f"user {user}: reviews / labelled rows / input-22 nonzero rows",
               [int(batch.start.size(0)), int(labelled.sum()), int((label_types != 0).sum())], labelled.sum() > 0
               and (label_types != 0).any())
        g = gap(reference, ours, labelled)
        record(f"user {user}: fork128 vs upstream fork, max |dp| curve+instant", g, g <= TOLERANCE)
        g = gap(reference_zero, ours_zero, labelled)
        record(f"user {user}: zero_features=[22] vs upstream with weight column zeroed, max |dp|", g, g <= TOLERANCE)
        g = gap(reference, reference_zero, labelled)
        record(f"user {user}: zeroing input 22 changes the fork (test is not vacuous), max |dp|", g, g > 1e-4)
        g = max(abs(float(reference.average_loss) - float(ours.average_loss)), 0.0)
        record(f"user {user}: eval loss fork vs fork128", [float(reference.average_loss), float(ours.average_loss)], g <= TOLERANCE)

    # One CPU training step on a short sequence (dropout on, as in training).
    user = args.users[0]
    sample = make_samples(get_rwkv_data(DATA, user).iloc[:args.train_reviews], user, 128)[0]
    batch = prepare([sample], seed=2025)
    for name, model in (("fork128", variants.build("fork128")),
                        ("fork128 zero_features=[22]", variants.build({"base": "fork128", "zero_features": [22]}))):
        model.load_state_dict(converted, strict=True)
        model.train()
        optimizer = get_optimizer(Namespace(PEAK_LR=7e-4), model)
        torch.manual_seed(0)
        stats = model.get_loss(batch)
        stats.average_loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        finite = torch.isfinite(stats.average_loss).item() and all(torch.isfinite(g).all() for g in grads)
        nonzero = sum(float(g.abs().sum()) > 0 for g in grads)
        column = model.features2card[0].weight.grad[:, 22].abs().sum().item()
        before = model.features2card[0].weight[:, 22].detach().clone()
        for group in optimizer.param_groups:
            group["lr"] = 7e-4
        optimizer.step()
        after_finite = all(torch.isfinite(p).all() for p in model.parameters())
        record(f"{name}: train step loss / grads with value / grad on input-22 column",
               [float(stats.average_loss.detach()), f"{nonzero}/{len(list(model.parameters()))}", column],
               finite and nonzero > 100 and after_finite and ("zero" not in name or column == 0.0))

    elapsed = time.perf_counter() - started
    failed = [k for k, v in results.items() if not v["ok"]]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed in {elapsed:.0f}s")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"results": results, "seconds": elapsed}, indent=1, default=str))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
