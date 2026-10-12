"""F08 opt-in masked loss reductions; separate from the active shift-only trial.

Seven FP32 per-position losses feed nine means. Original loss creation, objective
weights, binary predictions, label counts and finite-value checks are retained.
CUDA uses fixed partial/final reductions; compilation errors do not fall back.
CPU exercises the analytic custom backward with the original eager reductions.
"""

import ast
import inspect
from pathlib import Path
import textwrap
import types

import numpy as np
import torch

SOURCE = Path(__file__).with_name("kernels") / "rwkv_masked_means.cu"
NAMES = ("masked_partial", "masked_finish", "masked_backward")
FIELDS = ("ahead_avg", "ahead_raw_avg", "immediate_avg", "w_avg", "ahead_logits_mag_avg", "ahead_logits_diff_avg",
          "ahead_equalize_avg", "ahead_raw_equalize_avg", "immediate_binary_equalize_avg")
PAIRS = ((0, 0), (1, 0), (2, 1), (3, 0), (4, 0), (5, 0), (0, 2), (1, 2), (6, 3))
_MODULES = {}


def _layout(losses, masks):
    source = losses[0]
    if source.dtype != torch.float32 or source.numel() == 0 or source.numel() > 2**24:
        raise ValueError("Masked means require nonempty FP32 losses with at most 2^24 positions")
    if any(value.shape != source.shape or value.dtype != source.dtype or value.device != source.device for value in losses):
        raise ValueError("All seven per-position losses must share shape, dtype and device")
    if any(value.shape != source.shape or value.device != source.device or value.dtype not in (torch.int32, torch.int64, torch.bool) for value in masks):
        raise ValueError("Four binary masks must match the loss shape/device and use integer or bool dtype")
    # The model constructs binary masks from prepared labels on its original path.
    # No device-dependent branch or CUDA-to-host transfer is introduced here.
    return source.numel()


def eager_means(losses, masks):
    return torch.stack([(losses[index] * masks[mask]).sum() / (1e-8 + masks[mask].sum()) for index, mask in PAIRS])


def compile_module(device):
    from rwkv.model import nvrtc
    nvrtc.register()
    if device.index not in _MODULES:
        with nvrtc._cupy.cuda.Device(device.index):
            raw = nvrtc._cupy.RawModule(code=SOURCE.read_text(), options=("--std=c++17", "--fmad=false"))
            _MODULES[device.index] = tuple(raw.get_function(name) for name in NAMES)
    return _MODULES[device.index]


def launch(function, grid, values, integers, device):
    from rwkv.model import nvrtc
    cupy = nvrtc._cupy
    with cupy.cuda.Device(device.index), cupy.cuda.ExternalStream(torch.cuda.current_stream(device).cuda_stream):
        function(grid, (256,), tuple(np.uint64(value.data_ptr()) for value in values) + tuple(np.int32(value) for value in integers))


class MaskedMeans(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *values):
        losses, masks = values[:7], values[7:]
        n = _layout(losses, masks)
        masks = tuple(value.to(torch.int32).contiguous() for value in masks)
        source = losses[0]
        if source.is_cuda:
            kernels = compile_module(source.device)
            chunks = (n + 255) // 256
            partials = torch.empty(13, chunks, device=source.device, dtype=torch.float32)
            output = torch.empty(13, device=source.device, dtype=torch.float32)
            launch(kernels[0], (chunks, 13), (*(value.contiguous() for value in losses), *masks, partials), (n, chunks), source.device)
            launch(kernels[1], (13,), (partials, output), (chunks,), source.device)
            ctx.kernel = kernels[2]
        else:
            output = torch.cat((eager_means(losses, masks), torch.stack([value.sum().float() for value in masks])))
        ctx.save_for_backward(*masks, output)
        ctx.shape, ctx.n = source.shape, n
        return output[:9]

    @staticmethod
    def backward(ctx, gradient):
        masks, output = ctx.saved_tensors[:4], ctx.saved_tensors[4]
        if gradient.is_cuda:
            gradients = tuple(torch.empty(ctx.shape, device=gradient.device, dtype=torch.float32) for _ in range(7))
            launch(ctx.kernel, ((ctx.n + 255) // 256,), (gradient.contiguous(), *masks, output, *gradients), (ctx.n,), gradient.device)
        else:
            contributions = [[] for _ in range(7)]
            for field, (index, mask) in enumerate(PAIRS):
                contributions[index].append((gradient[field] / (1e-8 + output[9 + mask])) * masks[mask])
            gradients = tuple(sum(parts) for parts in contributions)
        return (*gradients, None, None, None, None)


def masked_means(*values):
    if len(values) != 11:
        raise ValueError("Expected seven losses and four masks")
    return MaskedMeans.apply(*values)


def install_fused_loss(model):
    """Bind a rewritten original loss method only on an isolated Python model."""
    if isinstance(model, torch.jit.ScriptModule):
        raise ValueError("Use a Python model with its own matched control")
    tree = ast.parse(textwrap.dedent(inspect.getsource(type(model)._get_loss)))
    method = tree.body[0]
    method.decorator_list = []
    found = []
    retained = []
    for node in method.body:
        target = node.targets[0].id if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) else None
        if target in FIELDS:
            found.append(target)
        else:
            retained.append(node)
    if set(found) != set(FIELDS) or len(found) != 9:
        raise ValueError("Expected the original nine masked averages exactly once")
    insert = [index for index, node in enumerate(retained) if isinstance(node, ast.Assign)
              and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "loss_avg"]
    if len(insert) != 1:
        raise ValueError("Original composite objective changed")
    replacement = ast.parse(
        ", ".join(FIELDS) + " = _masked_means(curve_loss, curve_raw_loss, p_loss, w_loss, "
        "ahead_logits_mag_loss, ahead_logits_diff_loss, p_binary_loss, ahead_mask, immediate_mask, "
        "ahead_equalize_mask, immediate_equalize_mask).unbind()\n").body
    retained[insert[0]:insert[0]] = replacement
    method.body = retained
    namespace = {**type(model)._get_loss.__globals__, "_masked_means": masked_means}
    exec(compile(ast.fix_missing_locations(tree), __file__, "exec"), namespace)
    model._get_loss = types.MethodType(namespace[method.name], model)
    return {"means": list(FIELDS), "cuda": "fixed partial/final FP32 reductions and one backward kernel; no global atomics",
            "cpu": "original means with analytic custom backward",
            "preserved": "loss creation, objective weights, detached reports, label counts, binary predictions and finite checks",
            "scope": "does not fuse BCE/CE/regularizer creation or hierarchy gather/cat"}
