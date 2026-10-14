"""F82: Muon for the 32x32 time-mixer and channel-mixer projections, AdamW for everything else.

`get_hybrid_optimizer` starts from the repository's AdamW groups (rwkv/train_rwkv.get_optimizer) and moves
group 0 (time-mixer projections) and group 1 (channel-mixer projections) to torch.optim.Muon with the same
weight decay. Heads, the features2card encoder, LoRA weights, norms and vectors stay on AdamW unchanged.

The R01 trainer writes one learning rate into every param group before each step. Muon groups carry an
`lr_scale`; `step` applies it for the Muon update only and restores the written value, so the trainer and
its warmup stay untouched. With adjust_lr_fn="match_rms_adamw" (Moonlight) a scale of 1 reuses AdamW's
learning rate.
"""
from argparse import Namespace

import torch

# Bound at import, so the reference groups stay AdamW's even after a runner patches train_rwkv.get_optimizer.
from rwkv.train_rwkv import ADAMW_BETAS, ADAMW_EPS, get_optimizer

MUON_GROUPS = (0, 1)


class HybridOptimizer:
    def __init__(self, adamw, muon):
        self.adamw, self.muon = adamw, muon
        self.param_groups = adamw.param_groups + muon.param_groups

    def zero_grad(self, set_to_none=True):
        self.adamw.zero_grad(set_to_none=set_to_none)
        self.muon.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self):
        self.adamw.step()
        written = [group["lr"] for group in self.muon.param_groups]
        for group in self.muon.param_groups:
            group["lr"] = group["lr"] * group["lr_scale"]
        try:
            self.muon.step()
        finally:
            for group, lr in zip(self.muon.param_groups, written):
                group["lr"] = lr

    def state_dict(self):
        return {"adamw": self.adamw.state_dict(), "muon": self.muon.state_dict()}

    def load_state_dict(self, state):
        self.adamw.load_state_dict(state["adamw"])
        self.muon.load_state_dict(state["muon"])


def get_hybrid_optimizer(model, peak_lr=7e-4, lr_scale=1.0, momentum=0.95, adjust_lr_fn="match_rms_adamw"):
    reference = get_optimizer(Namespace(PEAK_LR=peak_lr), model)
    adamw_groups, muon_groups = [], []
    for index, group in enumerate(reference.param_groups):
        settings = {key: value for key, value in group.items() if key in ("params", "weight_decay", "lr")}
        if index in MUON_GROUPS:
            if any(p.dim() != 2 for p in group["params"]):
                raise ValueError("Muon groups must hold 2-D matrices only")
            muon_groups.append(dict(settings, lr_scale=lr_scale))
        else:
            adamw_groups.append(settings)
    adamw = torch.optim.AdamW(adamw_groups, eps=ADAMW_EPS, betas=ADAMW_BETAS)
    muon = torch.optim.Muon(muon_groups, lr=peak_lr, momentum=momentum, adjust_lr_fn=adjust_lr_fn)
    return HybridOptimizer(adamw, muon)
