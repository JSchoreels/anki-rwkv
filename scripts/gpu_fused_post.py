"""Isolated F08 normalization/bonus/gate prototype, beyond the active shift trial.

No model source or default backend is changed on disk. CUDA has no silent eager
fallback on compiler/backend errors. CPU uses the original eager expression.
Parameter-gradient atomic reductions can be nondeterministic on CUDA; numerical
and predictive validation remain necessary even when CPU math checks pass.
"""

import ast
import inspect
from pathlib import Path
import textwrap
import types

import numpy as np
import torch
import torch.nn.functional as functional

SOURCE = Path(__file__).with_name("kernels") / "rwkv_fused_post.cu"
_MODULES = {}


def compiler_options(dtype):
    import os
    import sysconfig
    options = ["--std=c++17", "--fmad=false"]
    if dtype == torch.bfloat16:
        runtime = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cuda_runtime/include"
        candidates = [runtime.parent.parent / "cuda_cccl/include"]
        candidates += [Path(os.environ[key]) / "include" for key in ("CUDA_PATH", "CUDA_HOME") if os.environ.get(key)]
        installed = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA GPU Computing Toolkit/CUDA"
        candidates += [path / "include" for path in sorted(installed.glob("v*"), reverse=True)]
        cccl = next((path for path in candidates if (path / "nv/target").exists()), None)
        if cccl is None:
            raise RuntimeError("BF16 compilation requires the CUDA CCCL headers")
        options += ["-DUSE_BF16", f"--include-path={runtime}", f"--include-path={cccl}"]
    elif dtype != torch.float32:
        raise ValueError("Fused post supports FP32 and BF16")
    return tuple(options)


def compile_module(device, dtype):
    from rwkv.model import nvrtc
    nvrtc.register()
    key = (device.index, dtype)
    if key not in _MODULES:
        with nvrtc._cupy.cuda.Device(device.index):
            module = nvrtc._cupy.RawModule(code=SOURCE.read_text(), options=compiler_options(dtype))
            _MODULES[key] = tuple(module.get_function(name) for name in ("fused_post_forward", "fused_post_backward"))
    return _MODULES[key]


def eager_post(x, r, k, v, gate, bonus, gamma, beta, eps):
    b, t, h, width = x.shape
    normalized = functional.group_norm(x.reshape(b * t, h * width), h, gamma, beta, eps).view(b, t, h * width)
    extra = ((r * bonus * k).sum(-1, keepdim=True) * v).view(b, t, h * width)
    return gate * (normalized + extra)


def _layout(x, r, k, v, gate, bonus, gamma, beta):
    if x.ndim != 4 or x.shape[-1] != 32 or x.numel() == 0:
        raise ValueError("Fused post requires nonempty B,T,H,32 recurrence output")
    b, t, h, width = x.shape
    if any(value.shape != x.shape for value in (r, k, v)) or gate.shape != (b, t, h * width):
        raise ValueError("Recurrence/gate shapes differ")
    if bonus.shape != (1, 1, h, width) or gamma.shape != (h * width,) or beta.shape != gamma.shape:
        raise ValueError("Bonus/normalization parameter shapes differ")
    if any(value.dtype != x.dtype or value.device != x.device for value in (r, k, v, gate, bonus, gamma, beta)):
        raise ValueError("Inputs must share dtype and device")
    return b * t * h, h


def _launch(function, groups, values, integers, device):
    from rwkv.model import nvrtc
    cupy = nvrtc._cupy
    args = tuple(np.uint64(value.data_ptr()) for value in values) + integers
    with cupy.cuda.Device(device.index), cupy.cuda.ExternalStream(torch.cuda.current_stream(device).cuda_stream):
        function(((groups + 3) // 4,), (128,), args)


class FusedPost(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, r, k, v, gate, bonus, gamma, beta, eps):
        groups, heads = _layout(x, r, k, v, gate, bonus, gamma, beta)
        if not x.is_cuda:
            raise ValueError("Custom fused post requires CUDA")
        values = tuple(value.contiguous() for value in (x, r, k, v, gate, bonus, gamma, beta))
        kernels = compile_module(x.device, x.dtype)
        output = torch.empty_like(values[4])
        means = torch.empty(groups, device=x.device, dtype=torch.float32)
        inverses = torch.empty_like(means)
        _launch(kernels[0], groups, (*values, output, means, inverses),
                (np.int32(groups), np.int32(heads), np.float32(eps)), x.device)
        ctx.save_for_backward(*values, means, inverses)
        ctx.groups, ctx.heads, ctx.kernels = groups, heads, kernels
        return output

    @staticmethod
    def backward(ctx, gradient):
        values = ctx.saved_tensors[:8]
        means, inverses = ctx.saved_tensors[8:]
        x, r, k, v, gate, bonus, gamma, beta = values
        outputs = tuple(torch.empty_like(value) for value in (x, r, k, v, gate))
        parameter_grads = tuple(torch.zeros(value.shape, device=x.device, dtype=torch.float32) for value in (bonus, gamma, beta))
        _launch(ctx.kernels[1], ctx.groups, (gradient.contiguous(), *values, means, inverses, *outputs, *parameter_grads),
                (np.int32(ctx.groups), np.int32(ctx.heads)), x.device)
        return (*outputs, *(value.to(x.dtype) for value in parameter_grads), None)


def fused_post(x, r, k, v, gate, bonus, gamma, beta, eps):
    _layout(x, r, k, v, gate, bonus, gamma, beta)
    if x.is_cuda:
        return FusedPost.apply(x, r, k, v, gate, bonus, gamma, beta, eps)
    return eager_post(x, r, k, v, gate, bonus, gamma, beta, eps)


def install_fused_post(model):
    """Replace only the cloned Python mixer's three post-recurrence expressions."""
    installed = []
    for name, mixer in model.named_modules():
        if type(mixer).__name__ != "RWKV7TimeMixer":
            continue
        if isinstance(mixer, torch.jit.ScriptModule):
            raise ValueError("Use the isolated Python model with its own matched control")
        tree = ast.parse(textwrap.dedent(inspect.getsource(type(mixer).forward)))
        method = tree.body[0]
        method.decorator_list = []
        starts = [index for index, node in enumerate(method.body)
                  if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                  and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "view"
                  and isinstance(node.value.func.value, ast.Call) and isinstance(node.value.func.value.func, ast.Attribute)
                  and node.value.func.value.func.attr == "out_group_norm"]
        if len(starts) != 1:
            raise ValueError("Expected exactly one original post-recurrence normalization block")
        index = starts[0]
        targets = [node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None
                   for node in method.body[index:index + 3]]
        if targets != ["out_BTC", "bonus_BTC", "out_BTC"]:
            raise ValueError("Original post-recurrence block changed")
        replacement = ast.parse("""out_BTC = _fused_post(out_BTHK, r_BTHK, k_BTHK, v_BTHK, g_BTC, self.bonus,
    self.out_group_norm.weight, self.out_group_norm.bias, self.out_group_norm.eps)
out_BTC = self.W_o(out_BTC)
""").body
        method.body[index:index + 3] = replacement
        namespace = {**type(mixer).forward.__globals__, "_fused_post": fused_post}
        exec(compile(ast.fix_missing_locations(tree), __file__, "exec"), namespace)
        mixer.forward = types.MethodType(namespace[method.name], mixer)
        installed.append(name)
    if not installed:
        raise ValueError("No eligible Python time mixer found")
    return {"modules": installed, "region": "GroupNorm, bonus reduction, output gate; W_o/dropout remain outside",
            "cuda_backward": "FP32 atomic parameter reductions; nondeterminism must be assessed",
            "cpu": "original eager expression", "cuda": "custom NVRTC kernel, no compiler-error fallback"}


def backward_math(inputs, gradient, eps):
    """CPU coefficient/chain-rule diagnostic against eager autograd.

    Use native CPU saved statistics to assess its backward contract. CUDA's
    moment reduction and BF16 dY*X intermediate differ and need GPU checks.
    """
    x, r, k, v, gate, bonus, gamma, beta = inputs
    b, t, h, width = x.shape
    cast = lambda value: value.to(x.dtype)
    xfloat = x.float()
    normalized, native_mean, native_inv = torch.ops.aten.native_group_norm.default(
        x.reshape(b * t, h * width), gamma, beta, b * t, h * width, 1, h, eps)
    mean = native_mean.float().view(b, t, h, 1)
    inv = native_inv.float().view(b, t, h, 1)
    normalized = normalized.view_as(x)
    inner = (r * bonus * k).sum(-1, keepdim=True)
    extra = inner * v
    dy = (gradient * gate).view_as(x)
    dg = gradient * (normalized + extra).view_as(gate)
    dv = dy * inner
    dinner = (dy * v).sum(-1, keepdim=True)
    dk = dinner * (r * bonus)
    drbonus = dinner * k
    dr = drbonus * bonus
    dbonus = (drbonus * r).sum((0, 1), keepdim=True)
    weighted = dy.float() * gamma.float().view(h, width)
    sum1 = (weighted * xfloat).sum(-1, keepdim=True)
    sum2 = weighted.sum(-1, keepdim=True)
    c2 = (sum2 * mean - sum1) * inv * inv * inv / width
    c3 = -c2 * mean - sum2 * inv / width
    dx = cast((inv * gamma.float().view(h, width)) * dy.float() + c2 * xfloat + c3)
    dgamma = cast(((dy.float() * xfloat - dy.float() * mean) * inv).sum((0, 1)).flatten())
    dbeta = cast(dy.float().sum((0, 1)).flatten())
    return dx, dr, dk, dv, dg, dbonus, dgamma, dbeta
