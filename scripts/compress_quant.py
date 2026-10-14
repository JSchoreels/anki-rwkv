"""Weight and recurrent-state quantization for the fork's 2.76M RWKV (upstream srs-benchmark architecture, `rwkv/`).

Weight scheme (all variants): symmetric signed integers, absolute-maximum scale, round-half-to-even (torch.round),
codes in [-(2^(b-1)-1), 2^(b-1)-1] (int8: -127..127, int4: -7..7, as the fork's state codes). Every nn.Linear weight
is quantized (265 tensors, 2,719,744 of the 2,762,884 parameters); biases, LayerNorm/GroupNorm parameters, lerp/bonus
vectors stay float32. A group is a run of `group` consecutive input columns of one output row (the fork's state packs
32-element rows the same way); group=None means one scale per output row (per-channel).

State scheme (the fork's actual "int8"/"int4" option, rslib/src/rwkv/precision.rs): after every answer, the card and
note modules' WKV matrices are rounded per 32-element row (symmetric, abs-max f32 scale, round-half-even) and their
time/channel shift vectors are rounded to f16. Deck/preset/user state, weights and arithmetic stay float32. Emulated
here inside the CPU reference recurrence (`use_reference_wkv`), so it only applies on that path.
"""
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FORK_WEIGHTS = Path(r"C:\Users\jonat\workspace\anki\qt\aqt\rwkv_inference\RWKV_trained_on_5000_10000.pth")

# name -> (bits, group); bits None = float32
WEIGHT_SCHEMES = {
    "fp32": (None, None),
    "w8-pc": (8, None),
    "w8-g32": (8, 32),   # the fork's state encoding applied to weights
    "w4-pc": (4, None),
    "w4-g64": (4, 64),
    "w4-g32": (4, 32),
    "w3-pc": (3, None),  # codes -3..3
    "w3-g32": (3, 32),
    "w2-g32": (2, 32),   # codes -1..1 (ternary, abs-mean scale, stored in 2 bits)
    "w2-g16": (2, 16),
}


def linear_weight_names(model_or_state):
    """Every 2-D `.weight` (all are nn.Linear weights in this architecture; the modules are TorchScript wrappers)."""
    state = model_or_state if isinstance(model_or_state, dict) else model_or_state.state_dict()
    return [name for name, tensor in state.items() if name.endswith(".weight") and tensor.dim() == 2]


def fake_quant(weight, bits, group=None):
    """Quantize-dequantize a 2-D weight [out, in] (symmetric abs-max, round-half-even)."""
    if bits is None:
        return weight
    qmax = 2 ** (bits - 1) - 1
    out_features, in_features = weight.shape
    g = in_features if group is None else group
    groups = math.ceil(in_features / g)
    pad = groups * g - in_features
    w = torch.nn.functional.pad(weight, (0, pad)) if pad else weight
    w = w.reshape(out_features, groups, g)
    if bits == 2:  # ternary: BitNet b1.58 abs-mean scale (abs-max would round most weights to 0)
        scale = w.abs().mean(dim=-1, keepdim=True)
    else:
        scale = w.abs().amax(dim=-1, keepdim=True) / qmax
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    deq = (q * scale).reshape(out_features, groups * g)
    return deq[:, :in_features] if pad else deq


def ste_quant(weight, bits, group=None):
    """Straight-through estimator: forward = fake_quant, backward = identity."""
    if bits is None:
        return weight
    return weight + (fake_quant(weight.detach(), bits, group) - weight.detach())


def quantize_state_dict(state, names, bits, group):
    out = dict(state)
    if bits is not None:
        for name in names:
            out[name] = fake_quant(state[name].float(), bits, group).to(state[name].dtype)
    return out


def weight_bytes(state, names, bits, group, scale_bytes=4, other_bytes=4):
    """Storage bytes: packed codes + one scale per group for quantized tensors, `other_bytes` per other parameter."""
    names = set(names)
    total = {"codes": 0, "scales": 0, "other": 0}
    for name, tensor in state.items():
        if bits is not None and name in names:
            out_features, in_features = tensor.shape
            g = in_features if group is None else group
            total["codes"] += math.ceil(out_features * in_features * bits / 8)
            total["scales"] += out_features * math.ceil(in_features / g) * scale_bytes
        else:
            total["other"] += tensor.numel() * (4 if bits is None else other_bytes)
    total["total"] = sum(total.values())
    return total


def state_bytes(layers, d_model=128, heads=4, head_size=32, precision="f32"):
    """Per-entity (card or note) recurrent-state bytes as the fork stores them (headers excluded)."""
    matrix = heads * head_size * head_size
    if precision == "f32":
        per_layer = (matrix + 2 * d_model) * 4
    else:
        code = head_size if precision == "int8" else head_size // 2
        per_layer = (matrix // head_size) * (4 + code) + 2 * d_model * 2
    return layers * per_layer


def load_model(path=None, state=None, device="cpu", reference_wkv=True):
    from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
    from rwkv.model.srs_model import SrsRWKV
    model = SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG).float()
    if state is None:
        state = torch.load(path or FORK_WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    if reference_wkv:
        for module in model.modules():
            if hasattr(module, "use_reference_wkv"):
                module.use_reference_wkv = True
    return model.to(device).eval()


# ---------------------------------------------------------------- state quantization (fork precision.rs emulation)

STATE_MODULES = (0, 2)  # rwkv_modules order: card, deck, note, preset, user -> card and note are compacted

_PATCH_HELPERS = """

def _quantize_rows(x: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = 127.0 if bits == 8 else 7.0
    scale = x.abs().amax(dim=-1, keepdim=True) / qmax
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return torch.clamp(torch.round(x / scale), -qmax, qmax) * scale


def reference_rwkv7_q(r_BTHK: torch.Tensor, k_BTHK: torch.Tensor, v_BTHK: torch.Tensor, w_BTHK: torch.Tensor,
                      a_BTHK: torch.Tensor, k_deformed_BTHK: torch.Tensor, skip_BT: torch.Tensor, bits: int):
    if bits == 0:
        return reference_rwkv7(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT)
    out_dtype = k_BTHK.dtype
    r_BTHK = r_BTHK.float()
    k_BTHK = k_BTHK.float()
    v_BTHK = v_BTHK.float()
    w_BTHK = w_BTHK.float()
    a_BTHK = a_BTHK.float()
    k_deformed_BTHK = k_deformed_BTHK.float()
    skip_BT111 = skip_BT.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    B, T, H, K = r_BTHK.shape
    out_BTHK = torch.empty(B, T, H, K, dtype=torch.float32, device=r_BTHK.device)
    state_BHKK = torch.zeros(B, H, K, K, dtype=torch.float32, device=r_BTHK.device)
    for t in range(T):
        out_t, next_state_BHKK = single_timestep(
            r_BTHK[:, t], k_BTHK[:, t], v_BTHK[:, t], w_BTHK[:, t], a_BTHK[:, t], k_deformed_BTHK[:, t], state_BHKK)
        out_BTHK[:, t] = out_t
        # The answer's output used the f32 update; the stored state is rounded before the next answer reads it.
        next_state_BHKK = _quantize_rows(next_state_BHKK, bits)
        state_BHKK = torch.where(skip_BT111[:, t], state_BHKK, next_state_BHKK)
    return out_BTHK.to(out_dtype)

"""

_PATCHES = [
    ("from rwkv.model.rwkv_ops import RWKV7_WKV, reference_rwkv7\n",
     "from rwkv.model.rwkv_ops import RWKV7_WKV, reference_rwkv7, single_timestep\n" + _PATCH_HELPERS),
    ("            channel_ratio = torch.ones(1, 1, config.d_model)\n",
     "            self.state_bits = 0\n            channel_ratio = torch.ones(1, 1, config.d_model)\n"),
    ("        k_BTK = self.W_k(torch.lerp(x_BTC, x_shift_BTC, self.lerp_k))\n",
     "        if self.state_bits > 0:\n            x_shift_BTC = x_shift_BTC.half().float()\n"
     "        k_BTK = self.W_k(torch.lerp(x_BTC, x_shift_BTC, self.lerp_k))\n"),
    ("        self.use_reference_wkv = config.use_reference_wkv\n",
     "        self.use_reference_wkv = config.use_reference_wkv\n        self.state_bits = 0\n"),
    ("            rkvdag_8BTC = torch.lerp(\n                x_BTC.unsqueeze(0), x_shift_BTC.unsqueeze(0)",
     "            if self.state_bits > 0:\n                x_shift_BTC = x_shift_BTC.half().float()\n"
     "            rkvdag_8BTC = torch.lerp(\n                x_BTC.unsqueeze(0), x_shift_BTC.unsqueeze(0)"),
    ("            out_BTHK = reference_rwkv7(\n                r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT\n",
     "            out_BTHK = reference_rwkv7_q(\n"
     "                r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT, self.state_bits\n"),
]


def install_state_quant_model():
    """Replace rwkv.model.rwkv_model by a patched copy whose card/note modules can emulate the fork's compact state.
    Must run before rwkv.architecture / rwkv.model.srs_model are imported. With state_bits = 0 (the default) every
    module runs the original code path, so weight-only variants are unaffected."""
    import importlib.util
    name = "rwkv.model.rwkv_model"
    if name in sys.modules:
        if getattr(sys.modules[name], "STATE_QUANT_PATCHED", False):
            return sys.modules[name]
        raise RuntimeError("rwkv.model.rwkv_model was imported before install_state_quant_model()")
    path = ROOT / "rwkv" / "model" / "rwkv_model.py"
    source = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    for old, new in _PATCHES:
        if source.count(old) != 1:
            raise RuntimeError(f"patch anchor not found exactly once: {old[:60]!r}")
        source = source.replace(old, new)
    source += "\nSTATE_QUANT_PATCHED = True\n"
    # TorchScript reads method source through inspect, so the patched copy must exist as a real file.
    generated = ROOT / "artifacts" / "compression" / "_generated" / "rwkv_model_state_quant.py"
    generated.parent.mkdir(parents=True, exist_ok=True)
    if not generated.exists() or generated.read_text(encoding="utf-8") != source:
        generated.write_text(source, encoding="utf-8")
    import rwkv.model  # noqa: F401  (parent package)
    spec = importlib.util.spec_from_file_location(name, generated)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def set_state_bits(model, bits):
    """bits 8/4 = fork int8/int4 card+note state (CPU reference path only); 0/None = float32 state."""
    for index, module in enumerate(model.rwkv_modules):
        for block in module.blocks:
            value = int(bits or 0) if index in STATE_MODULES else 0
            block.time_mixer.state_bits = value
            block.channel_mixer.state_bits = value
    return model
