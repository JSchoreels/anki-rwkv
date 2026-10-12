"""F10 opt-in FP8 projection forward with BF16 backward and FP32 master weights.

Targets Torch 2.9's locally verified _scaled_mm signature. A successful device
probe is required; hardware capability alone does not establish backend support.
"""

import torch
import torch.nn.functional as F


def quantize(value):
    maximum = torch.finfo(torch.float8_e4m3fn).max
    amax = value.detach().float().abs().amax()
    scale = torch.where(amax > 0, amax / maximum, torch.ones_like(amax))
    scaled = (value.detach().float() / scale).clamp(-maximum, maximum)
    return scaled.to(torch.float8_e4m3fn), scale


class FP8Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias):
        flat = inputs.reshape(-1, inputs.shape[-1])
        # Dynamic history shapes need not have aligned token counts. Pad in
        # BF16 before quantizing, then crop output; retain original operands for
        # backward. Zero rows do not change the tensor scale.
        aligned_rows = (flat.shape[0] + 15) // 16 * 16
        padded = F.pad(flat, (0, 0, 0, aligned_rows - flat.shape[0]))
        qx, sx = quantize(padded)
        qw, sw = quantize(weight)
        output = torch._scaled_mm(qx.contiguous(), qw.t(), scale_a=sx, scale_b=sw,
                                  out_dtype=inputs.dtype, use_fast_accum=False)
        output = output[:flat.shape[0]]
        if bias is not None:
            output = output + bias
        ctx.save_for_backward(flat, weight)
        ctx.input_shape, ctx.has_bias = inputs.shape, bias is not None
        return output.view(*inputs.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, gradient):
        inputs, weight = ctx.saved_tensors
        flat = gradient.reshape(-1, weight.shape[0]).to(inputs.dtype)
        # Deliberate straight-through training: forward operands are quantized,
        # gradients use the original BF16 operands and BF16 GEMMs.
        dx = (flat @ weight).view(ctx.input_shape) if ctx.needs_input_grad[0] else None
        dw = flat.t() @ inputs if ctx.needs_input_grad[1] else None
        db = flat.float().sum(0).to(gradient.dtype) if ctx.has_bias and ctx.needs_input_grad[2] else None
        return dx, dw, db


class FP8Linear(torch.nn.Module):
    def __init__(self, linear, min_tokens=16):
        super().__init__()
        if linear.in_features % 16 or linear.out_features % 16 or min_tokens < 1:
            raise ValueError("FP8 projections need aligned dimensions and positive min_tokens")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.weight, self.bias = linear.weight, linear.bias
        self.min_tokens = min_tokens
        self.fp8_calls, self.eager_calls = 0, 0
        self.fp8_tokens, self.padded_fp8_tokens = 0, 0

    def forward(self, inputs):
        tokens = inputs.numel() // self.in_features
        if not inputs.is_cuda or inputs.dtype != torch.bfloat16 or self.weight.dtype != torch.bfloat16 or tokens < self.min_tokens:
            self.eager_calls += 1
            return F.linear(inputs, self.weight, self.bias)
        self.fp8_calls += 1
        self.fp8_tokens += tokens
        self.padded_fp8_tokens += (tokens + 15) // 16 * 16
        # No silent backend-error fallback: an unsupported CUDA operation fails
        # the experiment rather than being reported as FP8 performance.
        return FP8Projection.apply(inputs, self.weight, self.bias)


def install_fp8(model, min_tokens=16, families=("W_r", "W_k", "W_v", "W_o")):
    if isinstance(model, torch.jit.ScriptModule):
        raise TypeError("Use the isolated Python model and a matched Python control")
    selected = []
    for name, child in list(model.named_modules()):
        if (isinstance(child, torch.nn.Linear) and name.startswith("rwkv_modules.")
                and name.rsplit(".", 1)[-1] in families and child.in_features % 16 == 0 and child.out_features % 16 == 0):
            parent_name, _, attribute = name.rpartition(".")
            parent = model.get_submodule(parent_name)
            replacement = FP8Linear(child, min_tokens)
            setattr(parent, attribute, replacement)
            selected.append({"name": name, "shape": list(child.weight.shape), "minimum_tokens": min_tokens})
    if not selected:
        raise ValueError("No eligible projection was selected")
    return {"selected": selected, "forward": "per-tensor E4M3 _scaled_mm, slow accumulation",
            "backward": "straight-through BF16 with original operands", "sensitive_arithmetic": "unchanged",
            "quality_and_speed": "pending measured comparisons"}


def projection_counts(model):
    return {name: {"fp8_calls": child.fp8_calls, "eager_calls": child.eager_calls,
                   "fp8_tokens": child.fp8_tokens, "padded_fp8_tokens": child.padded_fp8_tokens}
            for name, child in model.named_modules() if isinstance(child, FP8Linear)}


def probe_fp8(device):
    device = torch.device(device)
    report = {"torch": str(torch.__version__), "cuda": torch.version.cuda, "device": str(device)}
    if device.type != "cuda" or not torch.cuda.is_available():
        return {**report, "status": "unavailable", "reason": "CUDA device required"}
    report.update(gpu=torch.cuda.get_device_name(device), capability=list(torch.cuda.get_device_capability(device)))
    try:
        with torch.random.fork_rng(devices=[device.index or 0]):
            torch.manual_seed(2025)
            x = torch.randn(32, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
            w = torch.randn(32, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
            y = FP8Projection.apply(x, w, None)
            y.float().square().mean().backward()
            torch.cuda.synchronize(device)
            if not all(torch.isfinite(value).all() for value in (y, x.grad, w.grad)):
                raise RuntimeError("Nonfinite FP8 probe output or gradient")
            expected = F.linear(x.detach(), w.detach())
            report.update(status="supported", max_absolute_forward_error=float((y - expected).float().abs().max()),
                          relative_forward_rms=float((y - expected).float().square().mean().sqrt() / expected.float().square().mean().sqrt()))
    except RuntimeError as exc:
        report.update(status="unavailable", reason=str(exc))
    return report
