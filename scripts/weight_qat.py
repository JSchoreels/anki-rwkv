"""4-bit weights (SM180): quantization-aware training of the weights, straight-through estimator.

Installed by the trainer's `state_vq` track option with the recipe "weight-qat" (scripts/state_vq.install_training):

    "state_vq": {"recipe": "weight-qat", "bits": 4, "group": 32}

The trainer keeps float32 master weights (the optimizer's) and a child model (bfloat16 where selectively cast, the
TorchScript model that runs forward and backward); before every update the grouped master/child sync
(scripts/gpu_r01_update.MasterChildSync.downcast, release option "grouped_sync") copies master -> child. With this
recipe, right after that copy every 2-D `.weight` of the child (exactly the tensors the scorer quantizes for w8 / w4g32 /
w4g16, all nn.Linear weights) is overwritten with compress_quant.fake_quant(master weight in float32, bits, group) cast
to the child's dtype. The forward and the gradient therefore run at the quantized weights, and the gradient is applied to
the latent float32 master weights unchanged: the straight-through estimator. The saved master.pth holds the latent
weights, which the scorer quantizes again with the same function (`w4g32` = compress_quant "w4-g32").

Nothing else changes: no module, no parameter name, no state_dict key; the teacher (another model) is never touched;
state_vq is teacher-blind (expertium_options.TEACHER_BLIND). Only MasterChildSync is supported (the release options);
the flat-buffer sync refuses to start while the recipe is installed. Every patched downcast is counted in the returned
description (saved in result.json under "state_vq") and the first one prints "[weight-qat] first downcast: ...".
"""
import torch

from scripts import compress_quant as cq

QUANT = {}      # id(child parameter) -> (bits, group)
INFO = None     # the description returned by install (live counters)
_ORIGINAL = {}


@torch.no_grad()
def apply(pairs):
    """Overwrite the selected child weights with fake_quant(master); returns the number of tensors written."""
    n = 0
    for master, child in pairs:
        spec = QUANT.get(id(child))
        if spec is not None:
            if child is master or child.data_ptr() == master.data_ptr():
                raise ValueError("weight-qat: the child shares the master's storage (float32 training): no latent weights")
            child.copy_(cq.fake_quant(master.detach().float(), *spec).to(child.dtype))
            n += 1
    return n


def _patch():
    from scripts import gpu_r01_flat, gpu_r01_update
    if "downcast" in _ORIGINAL:
        return
    original = gpu_r01_update.MasterChildSync.downcast
    _ORIGINAL["downcast"] = original

    @torch.no_grad()
    def downcast(self):
        original(self)
        if QUANT:
            n = apply(self.pairs)
            if INFO is not None:
                if INFO["downcasts"] == 0:
                    print(f"[weight-qat] first downcast: {n} child weights fake-quantized "
                          f"({INFO['bits']} bits, group {INFO['group']})", flush=True)
                INFO["downcasts"] += 1
                INFO["tensors_last_downcast"] = n

    gpu_r01_update.MasterChildSync.downcast = downcast
    flat_init = gpu_r01_flat.FlatMasterChildSync.__init__
    _ORIGINAL["flat_init"] = flat_init

    def refuse(self, *args, **kwargs):
        if QUANT:
            raise ValueError("weight-qat: only the grouped MasterChildSync is supported (no flat_buffers)")
        flat_init(self, *args, **kwargs)

    gpu_r01_flat.FlatMasterChildSync.__init__ = refuse


def install(model, value, device=None):
    global INFO
    unknown = set(value) - {"recipe", "bits", "group"}
    if unknown:
        raise ValueError(f"weight-qat: unknown keys {sorted(unknown)}")
    bits, group = int(value.get("bits", 4)), value.get("group", 32)
    group = None if group in (None, 0, "row") else int(group)
    if bits not in (2, 3, 4, 8):
        raise ValueError(f"weight-qat: bits {bits}")
    names = [n for n, p in model.named_parameters() if n.endswith(".weight") and p.dim() == 2]
    if not names:
        raise ValueError("weight-qat: no 2-D weight found")
    QUANT.clear()
    params = dict(model.named_parameters())
    for n in names:
        QUANT[id(params[n])] = (bits, group)
    _patch()
    INFO = {"recipe": "weight-qat", "bits": bits, "group": group, "quantized_tensors": len(names),
            "quantized_parameters": sum(params[n].numel() for n in names), "device": str(device), "downcasts": 0,
            "what": "SM180 straight-through QAT: child 2-D weights = fake_quant(master) after every master/child sync"}
    return INFO


def uninstall():
    global INFO
    QUANT.clear()
    INFO = None
