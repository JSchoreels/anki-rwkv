"""Experimental F08: one NVRTC kernel for time-shift gathering and eight lerps."""

import numpy as np
import torch

_KERNELS = {}


def kernel(dtype, backward=False):
    from rwkv.model import nvrtc
    nvrtc.register()
    key = (dtype, backward)
    if key not in _KERNELS:
        scalar = "float" if dtype == torch.float32 else "__nv_bfloat16"
        prefix = "" if dtype == torch.float32 else "#include <cuda_bf16.h>\n"
        read = "x" if dtype == torch.float32 else "__bfloat162float(x)"
        write = "x" if dtype == torch.float32 else "__float2bfloat16(x)"
        code = prefix + f"""
typedef {scalar} F;
__device__ float read(F x) {{ return {read}; }}
__device__ F write(float x) {{ return {write}; }}
extern "C" __global__ void shift_lerp(const F* x, const long long* select,
    const F* w, F* out, int B, int T, int C) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = B*T*C;
    if (i >= 8*n) return;
    int j = i % n, c = j % C, bt = j/C, b = bt/T;
    float a = read(x[j]), z = read(x[(b*T+select[bt])*C+c]);
    float weight = read(w[(i/n)*C+c]);
    float d = z-a;
    out[i] = write(fabsf(weight) < 0.5f ? a+weight*d : z-d*(1-weight));
}}
"""
        if backward:
            code += """
extern "C" __global__ void lerp_partials(const F* x, const long long* select,
    const F* w, const F* grad, F* out, int B, int T, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = B*T*C;
    if (i >= 8*n) return;
    int j = i % n, c = j % C, bt = j/C, b = bt/T;
    float a = read(x[j]), z = read(x[(b*T+select[bt])*C+c]);
    float weight = read(w[(i/n)*C+c]), g = read(grad[i]);
    // Round at each original tensor-operation boundary before multiplication.
    float complement = read(write(__fsub_rn(1.0f, weight)));
    float delta = read(write(__fsub_rn(z, a)));
    out[i] = write(__fmul_rn(g, complement));
    out[8*n+i] = write(__fmul_rn(g, weight));
    out[16*n+i] = write(__fmul_rn(g, delta));
}
"""
        # Reuse the runtime's tested BF16 header and CUDA toolkit discovery.
        nvrtc.module(torch.device("cuda", torch.cuda.current_device()), dtype)
        options = ["--std=c++17"]
        if dtype == torch.bfloat16:
            import os
            import sysconfig
            from pathlib import Path
            runtime = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cuda_runtime/include"
            candidates = [runtime.parent.parent / "cuda_cccl/include"]
            candidates += [Path(os.environ[v]) / "include" for v in ("CUDA_PATH", "CUDA_HOME") if os.environ.get(v)]
            installed = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA GPU Computing Toolkit/CUDA"
            candidates += [p / "include" for p in sorted(installed.glob("v*"), reverse=True)]
            cccl = next(p for p in candidates if (p / "nv/target").exists())
            options += [f"--include-path={runtime}", f"--include-path={cccl}"]
        _KERNELS[key] = nvrtc._cupy.RawKernel(code, "lerp_partials" if backward else "shift_lerp", options=tuple(options))
    return _KERNELS[key]


class FusedShiftLerp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, select, weight):
        if not x.is_cuda or x.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Fused shift requires CUDA float32 or BF16")
        from rwkv.model import nvrtc
        x, select, weight = x.contiguous(), select.contiguous(), weight.contiguous()
        B, T, C = x.shape
        out = torch.empty((8, B, T, C), device=x.device, dtype=x.dtype)
        function = kernel(x.dtype)
        cupy = nvrtc._cupy
        with cupy.cuda.Device(x.device.index), cupy.cuda.ExternalStream(torch.cuda.current_stream(x.device).cuda_stream):
            args = tuple(np.uint64(v.data_ptr()) for v in (x, select, weight, out)) + tuple(np.int32(v) for v in (B, T, C))
            function(((out.numel() + 255) // 256,), (256,), args)
        ctx.save_for_backward(x, select, weight)
        return out

    @staticmethod
    def backward(ctx, grad):
        x, select, weight = ctx.saved_tensors
        indices = select.unsqueeze(-1).expand_as(x)
        shifted = torch.gather(x, 1, indices)
        dx = (grad * (1 - weight)).sum(0)
        ds = (grad * weight).sum(0)
        dx = dx + torch.zeros_like(x).scatter_add_(1, indices, ds)
        dw = (grad * (shifted - x).unsqueeze(0)).sum((1, 2), keepdim=True)
        return dx, None, dw


class ExactShiftLerp(torch.autograd.Function):
    """F34e fused forward, with the original autograd reduction graph backward."""
    @staticmethod
    def forward(ctx, x, select, weight):
        from rwkv.model import nvrtc
        x, select, weight = x.contiguous(), select.contiguous(), weight.contiguous()
        B, T, C = x.shape
        out = torch.empty((8, B, T, C), device=x.device, dtype=x.dtype)
        args = tuple(np.uint64(v.data_ptr()) for v in (x, select, weight, out)) + tuple(np.int32(v) for v in (B, T, C))
        nvrtc.raw_launch(kernel(x.dtype), x.device, ((out.numel()+255)//256,), (256,), args)
        ctx.save_for_backward(x, select, weight)
        return out

    @staticmethod
    def backward(ctx, grad):
        x, select, weight = ctx.saved_tensors
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            weight = weight.detach().requires_grad_()
            shifted = torch.gather(x, 1, select.unsqueeze(-1).expand_as(x))
            output = torch.lerp(x.unsqueeze(0), shifted.unsqueeze(0), weight)
            dx, dw = torch.autograd.grad(output, (x, weight), grad)
        return dx, None, dw


class ExplicitShiftLerp(ExactShiftLerp):
    """F34f: original derivative operations/reductions without graph rebuilding."""
    @staticmethod
    def backward(ctx, grad):
        x, select, weight = ctx.saved_tensors
        indices = select.unsqueeze(-1).expand_as(x)
        shifted = torch.gather(x, 1, indices)
        dx = (grad * (1 - weight)).sum(0)
        ds = (grad * weight).sum(0)
        dw = grad * (shifted - x).unsqueeze(0)
        reduce_dims = tuple(i for i in (1, 2) if dw.shape[i] != 1)
        if reduce_dims:
            dw = dw.sum(reduce_dims, keepdim=True)
        shifted_dx = torch.zeros_like(x).scatter_add_(1, indices, ds)
        return dx + shifted_dx, None, dw


class PartialShiftLerp(ExactShiftLerp):
    """F34g: fuse rounded derivative elements, retain original reduction kernels."""
    @staticmethod
    def backward(ctx, grad):
        from rwkv.model import nvrtc
        x, select, weight = ctx.saved_tensors
        grad = grad.contiguous()
        B, T, C = x.shape
        partials = torch.empty((3, 8, B, T, C), device=x.device, dtype=x.dtype)
        args = tuple(np.uint64(v.data_ptr()) for v in (x, select, weight, grad, partials)) + tuple(np.int32(v) for v in (B, T, C))
        nvrtc.raw_launch(kernel(x.dtype, True), x.device, ((grad.numel()+255)//256,), (256,), args)
        dx = partials[0].sum(0)
        ds = partials[1].sum(0)
        dw = partials[2]
        reduce_dims = tuple(i for i in (1, 2) if dw.shape[i] != 1)
        if reduce_dims:
            dw = dw.sum(reduce_dims, keepdim=True)
        indices = select.unsqueeze(-1).expand_as(x)
        shifted_dx = torch.zeros_like(x).scatter_add_(1, indices, ds)
        return dx + shifted_dx, None, dw
