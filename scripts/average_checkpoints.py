"""Uniform weight average of several master.pth checkpoints of one run (SM74, epoch weight averaging).

usage: average_checkpoints.py --out OUT/master.pth IN1/master.pth IN2/master.pth ...

Every input must have the same keys, shapes and float dtypes; the average is taken in float64 and stored in the
inputs' dtype. A small JSON next to OUT records the inputs and their sha256.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()
    if len(args.inputs) < 2:
        raise SystemExit("need at least 2 checkpoints")
    states = [torch.load(p, map_location="cpu", weights_only=True) for p in args.inputs]
    keys = list(states[0])
    for path, state in zip(args.inputs[1:], states[1:]):
        if list(state) != keys:
            raise SystemExit(f"key mismatch: {path}")
    avg = type(states[0])()
    for k in keys:
        ref = states[0][k]
        if not ref.is_floating_point() or any(s[k].shape != ref.shape or s[k].dtype != ref.dtype for s in states):
            raise SystemExit(f"tensor {k}: not float or shape/dtype mismatch")
        avg[k] = (sum(s[k].double() for s in states) / len(states)).to(ref.dtype)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(avg, out)
    meta = {"inputs": [{"path": Path(p).as_posix(), "sha256": sha256(p)} for p in args.inputs],
            "out_sha256": sha256(out), "tensors": len(keys)}
    out.with_name("average.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(f"averaged {len(states)} checkpoints -> {out.as_posix()} ({len(keys)} tensors)")


if __name__ == "__main__":
    main()
