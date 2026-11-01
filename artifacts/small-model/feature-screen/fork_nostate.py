"""Ablation: score the fork with its review-type input (dim 22, scaled_state) removed on every row.

Wraps scripts/small_model_heldout.py unchanged. The fork is a ScriptModule (no hooks), so the column of the first
input Linear that reads dim 22 (features2card.0.weight[:, 22]) is zeroed in the loaded state dict before quantization
and loading: identical to a 0 input there (query rows already carry 0). Same arguments as the scorer.
"""
import sys

import torch

WT = r"C:\Users\jonat\workspace\wt\small-model"
sys.path.insert(0, WT)

_torch_load = torch.load


def load(path, *args, **kwargs):
    state = _torch_load(path, *args, **kwargs)
    if str(path).endswith("RWKV_trained_on_5000_10000.pth") and "features2card.0.weight" in state:
        w = state["features2card.0.weight"].clone()
        w[:, 22] = 0.0
        state = dict(state, **{"features2card.0.weight": w})
        print("[fork_nostate] zeroed features2card.0.weight[:, 22] (review type)", flush=True)
    return state


torch.load = load
from scripts import small_model_heldout as sm  # noqa: E402

sm.main()
