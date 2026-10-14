"""F82: Muon for the 32x32 time-mixer and channel-mixer projections, AdamW for everything else.

`get_hybrid_optimizer` starts from the repository's AdamW groups (rwkv/train_rwkv.get_optimizer) and moves
group 0 (time-mixer projections) and group 1 (channel-mixer projections) to torch.optim.Muon with the same
weight decay. Heads, the features2card encoder, LoRA weights, norms and vectors stay on AdamW unchanged.

The R01 trainer writes one learning rate into every param group before each step. Muon groups carry an
`lr_scale`; `step` applies it for the Muon update only and restores the written value, so the trainer and
its warmup stay untouched. With adjust_lr_fn="match_rms_adamw" (Moonlight) a scale of 1 reuses AdamW's
learning rate.

Expertium catch-up (docs/EXPERTIUM_CATCH_UP.md, item 3): `groups="matrices"` reproduces Expertium's iter 29/53
grouping (rwkv/train_rwkv.get_optimizer at Expertium/rwkv-anki-autoresearch@a9fcf466): every 2-D weight of the four
AdamW weight-decay groups (time mixer, channel mixer, heads, features2card encoder) on Muon, and with
`include_lora` the LoRA projections in their own Muon group at weight decay `lora_weight_decay` (iter 53: 0.0, the
value they had on AdamW). His Muon equals torch.optim.Muon up to float: same Newton-Schulz coefficients and bf16
iteration, Nesterov momentum in sum form instead of EMA form (identical after the normalisation), step scale
sqrt(max(1, rows/cols)) = adjust_lr_fn "original". He decays Muon weights at the AdamW rate (lr_adamw * wd);
`wd_at_adamw_rate` divides the group decay by lr_scale so torch's lr_muon * wd matches it.
The F82 default (`groups="projections"`) is unchanged.
"""
from argparse import Namespace

import torch

# Bound at import, so the reference groups stay AdamW's even after a runner patches train_rwkv.get_optimizer.
from rwkv.train_rwkv import ADAMW_BETAS, ADAMW_EPS, get_optimizer

MUON_GROUPS = (0, 1)
MATRIX_GROUPS = (0, 1, 2, 3)  # decay, channel mixer, head, features2card: Expertium's four Muon groups


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


def is_lora_matrix(name, param):
    # Expertium's RWKV_MUON_INCLUDE_LORA test (1x1 dummies squeeze below 2-D and stay on AdamW).
    return "weight" in name and "lora" in name and "scale" not in name and len(param.squeeze().shape) >= 2


def get_hybrid_optimizer(model, peak_lr=7e-4, lr_scale=1.0, momentum=0.95, adjust_lr_fn="match_rms_adamw",
                         groups="projections", include_lora=False, lora_weight_decay=0.0, wd_at_adamw_rate=False):
    reference = get_optimizer(Namespace(PEAK_LR=peak_lr), model)
    selected = {"projections": MUON_GROUPS, "matrices": MATRIX_GROUPS}[groups]
    names = {id(p): name for name, p in model.named_parameters()}
    adamw_groups, muon_groups, lora = [], [], []
    last = len(reference.param_groups) - 1
    for index, group in enumerate(reference.param_groups):
        settings = {key: value for key, value in group.items() if key in ("params", "weight_decay", "lr")}
        if index in selected:
            if any(p.dim() != 2 for p in group["params"]):
                raise ValueError("Muon groups must hold 2-D matrices only")
            if wd_at_adamw_rate:
                settings["weight_decay"] = settings["weight_decay"] / lr_scale
            muon_groups.append(dict(settings, lr_scale=lr_scale))
        else:
            if include_lora and index == last:
                lora = [p for p in settings["params"] if is_lora_matrix(names[id(p)], p)]
                moved = {id(p) for p in lora}
                settings["params"] = [p for p in settings["params"] if id(p) not in moved]
            adamw_groups.append(settings)
    if include_lora:
        if not lora or any(p.dim() != 2 for p in lora):
            raise ValueError("include_lora found no 2-D LoRA matrices")
        decay = lora_weight_decay / lr_scale if wd_at_adamw_rate else lora_weight_decay
        muon_groups.append({"params": lora, "weight_decay": decay, "lr": peak_lr, "lr_scale": lr_scale})
    adamw = torch.optim.AdamW(adamw_groups, eps=ADAMW_EPS, betas=ADAMW_BETAS)
    muon = torch.optim.Muon(muon_groups, lr=peak_lr, momentum=momentum, adjust_lr_fn=adjust_lr_fn)
    return HybridOptimizer(adamw, muon)


def expertium_iter53(model, peak_lr=7e-4):
    """Expertium's iter-53 optimizer on this trainer's AdamW peak: Muon peak 0.0025 absolute (his value; his AdamW
    peak was 1e-3), aspect-ratio step scale, AdamW-rate weight decay, LoRA matrices in a wd=0 Muon group."""
    return get_hybrid_optimizer(model, peak_lr, lr_scale=0.0025 / peak_lr, momentum=0.95, adjust_lr_fn="original",
                                groups="matrices", include_lora=True, lora_weight_decay=0.0, wd_at_adamw_rate=True)
