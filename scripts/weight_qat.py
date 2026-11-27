"""4-bit weights (SM180): quantization-aware training of the weights, straight-through estimator.

Installed by the trainer's `state_vq` track option with the recipe "weight-qat" (scripts/state_vq.install_training):

    "state_vq": {"recipe": "weight-qat", "bits": 4, "group": 32}

Every torch.nn.Linear of the training model (every 2-D `.weight` of these architectures, i.e. exactly the tensors the
scorer quantizes for w8 / w4g32 / w4g16) is switched to `QATLinear`: the forward uses
compress_quant.fake_quant(weight, bits, group) (symmetric abs-max, round-half-even, one float32 scale per `group` input
columns of a row), computed in float32 from the weight and cast back to the weight's dtype; the backward passes the
output gradient through to the weight unchanged (straight-through, no clipping mask). Biases stay as they are. The
parameters, their names and the state_dict are unchanged (only the module class is switched), so the optimizer, the
snapshots and master.pth are the same objects as without the option; the saved weights are the latent float weights,
which the scorer quantizes again with the same function (`w4g32` = compress_quant "w4-g32").

The teacher (another model object) is never touched. The recipe changes only the student's forward, never the batch or
the teacher input (state_vq is in expertium_options.TEACHER_BLIND).
"""
import torch

from scripts import compress_quant as cq


class _FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, bits, group):
        return cq.fake_quant(weight.detach().float(), bits, group).to(weight.dtype)

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None


def qat_weight(weight, bits, group):
    """Forward value fake_quant(weight) (float32 arithmetic, cast to weight.dtype); gradient identity."""
    return _FakeQuantSTE.apply(weight, bits, group)


class QATLinear(torch.nn.Linear):
    """nn.Linear whose forward uses the fake-quantized weight (attributes qat_bits / qat_group set by install)."""

    def forward(self, x):
        return torch.nn.functional.linear(x, qat_weight(self.weight, self.qat_bits, self.qat_group), self.bias)


def install(model, value, device=None):
    bits, group = int(value.get("bits", 4)), value.get("group", 32)
    group = None if group in (None, 0, "row") else int(group)
    if set(value) - {"recipe", "bits", "group"}:
        raise ValueError(f"weight-qat: unknown keys {sorted(set(value) - {'recipe', 'bits', 'group'})}")
    if bits not in (2, 3, 4, 8):
        raise ValueError(f"weight-qat: bits {bits}")
    names, params = [], 0
    for name, module in model.named_modules():
        if type(module) is torch.nn.Linear:
            module.__class__ = QATLinear
            module.qat_bits, module.qat_group = bits, group
            names.append(name)
            params += module.weight.numel()
        elif isinstance(module, torch.nn.Linear) and not isinstance(module, QATLinear):
            raise ValueError(f"weight-qat: {name} is a Linear subclass {type(module).__name__}")
    other2d = [n for n, p in model.named_parameters() if p.dim() == 2 and n.endswith(".weight")
               and n[:-len(".weight")] not in set(names)]
    if other2d:
        raise ValueError(f"weight-qat: 2-D weights outside nn.Linear: {other2d[:5]}")
    if not names:
        raise ValueError("weight-qat: no nn.Linear found")
    return {"recipe": "weight-qat", "bits": bits, "group": group, "linear_modules": len(names),
            "quantized_parameters": params, "device": str(device),
            "what": "SM180 straight-through fake quantization of every nn.Linear weight in the student's forward"}


def uninstall(model):
    for module in model.modules():
        if type(module) is QATLinear:
            module.__class__ = torch.nn.Linear
            del module.qat_bits, module.qat_group
