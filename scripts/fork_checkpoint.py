#!/usr/bin/env python
"""Fork checkpoint -> trainer-compatible initial weights for the `fork128` architecture preset.

The old fork's RWKV (upstream srs-benchmark class, d=128, 2,762,884 parameters) has exactly the parameter names and
shapes of `expertium.variants.build("fork128")`, so its state dict loads strictly with no key mapping. The trainer
starts a run from a plain state-dict file (`full_run.init_master`, i.e. `scripts/accuracy_loop.py prepare ...
--init-master P`, or a `master.pth` of an earlier run); this script writes that file from the fork's .pth.

  convert --out artifacts/fork-reference/fork128-init.pth [--fork PATH] [--zero-input 22]

--zero-input N additionally zeroes features2card.0.weight[:, N] (the input column of review-type feature N), so a
fine-tune that starts from the fork with input 22 removed also starts from weights that ignore it. It does not change
the outputs of a model built with zero_features=[22] (that model zeroes the input itself); it only keeps the dead
weight column from carrying the fork's trained values into the checkpoint.

Train with {"architecture": {"base": "fork128", "zero_features": [22]}} (label-free) or "fork128" (as the fork).
"""
from argparse import ArgumentParser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FORK = Path(r"C:\Users\jonat\workspace\anki\qt\aqt\rwkv_inference\RWKV_trained_on_5000_10000.pth")


def load_fork_state(path=FORK, zero_input=None):
    """The fork's state dict as float32 CPU tensors, strictly loadable into the fork128 variant."""
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {key: value.detach().clone().float() for key, value in state.items()}
    if zero_input is not None:
        state["features2card.0.weight"][:, int(zero_input)] = 0.0
    return state


def main():
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("convert")
    convert.add_argument("--fork", type=Path, default=FORK)
    convert.add_argument("--out", type=Path, required=True)
    convert.add_argument("--zero-input", type=int, default=None)
    args = parser.parse_args()
    import torch
    from expertium import variants
    state = load_fork_state(args.fork, args.zero_input)
    model = variants.build("fork128")
    model.load_state_dict(state, strict=True)
    count = sum(p.numel() for p in model.parameters())
    if count != variants.FORK128_PARAMETERS:
        raise SystemExit(f"fork128 has {count} parameters, expected {variants.FORK128_PARAMETERS}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"wrote {args.out} ({count:,} parameters, zero_input={args.zero_input})")


if __name__ == "__main__":
    main()
