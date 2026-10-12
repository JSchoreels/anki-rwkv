"""F32: GroupNorm weight/bias gradients for (N, C) inputs, bit for bit, across many SMs.

For N > 128 rows and HxW == 1, PyTorch's native_group_norm_backward computes
dgamma/dbeta in GammaBeta1dBackwardCUDAKernel2: one 32x16 block per 32
channels. Each (channel, row mod 32) chain is summed sequentially in float in
ascending row order, then the 32 chains of a channel are combined with a
__shfl_down_sync warp tree and rounded once to the parameter dtype. With 32
channels that is a single block on one SM. Here every chain runs in its own
warp lane with prefetched loads, and a second tiny kernel applies the same
tree. Forward and the input gradient remain PyTorch's own kernels.
"""

import numpy as np
import torch

_KERNELS = {}
UNROLL = 16


def kernel(dtype):
    from rwkv.model import nvrtc
    nvrtc.register()
    if dtype not in _KERNELS:
        scalar = "float" if dtype == torch.float32 else "__nv_bfloat16"
        prefix = "" if dtype == torch.float32 else "#include <cuda_bf16.h>\n"
        read = "x" if dtype == torch.float32 else "__bfloat162float(x)"
        write = "x" if dtype == torch.float32 else "__float2bfloat16(x)"
        code = prefix + f"""
typedef {scalar} F;
__device__ float read(F x) {{ return {read}; }}
__device__ F write(float x) {{ return {write}; }}
// PyTorch 2.9.1: dg_sum += (dy * x - dy * mean) * rstd, mean/rstd in the input dtype; db_sum += dy.
// Its sm_120 build contracts this to FFMA(FFMA(dy, x, -(dy*mean)), rstd, sum); float32 checks pin it.
__device__ float gamma_step(float sum, float dy, float x, float mean, float rstd) {{
    return __fmaf_rn(__fmaf_rn(dy, x, -__fmul_rn(dy, mean)), rstd, sum);
}}
// grid (32 residues, 2 kinds), block C == 32 channels: one chain per thread.
extern "C" __global__ void gamma_beta_chains(const F* dy, const F* x, const F* mean, const F* rstd,
    float* partial, long long N, int G) {{
    const int r = blockIdx.x, kind = blockIdx.y, c = threadIdx.x;
    const int D = 32 / G;
    const long long g = c / D;
    float sum = 0.0f;
    long long n = r;
    for (; n + 32LL * ({UNROLL} - 1) < N; n += 32LL * {UNROLL}) {{
        float dys[{UNROLL}], xs[{UNROLL}], means[{UNROLL}], rstds[{UNROLL}];
        #pragma unroll
        for (int u = 0; u < {UNROLL}; ++u) {{
            const long long row = n + 32LL * u;
            dys[u] = read(dy[row * 32 + c]);
            if (kind == 0) {{
                xs[u] = read(x[row * 32 + c]);
                means[u] = read(mean[row * G + g]);
                rstds[u] = read(rstd[row * G + g]);
            }}
        }}
        #pragma unroll
        for (int u = 0; u < {UNROLL}; ++u)
            sum = kind == 0 ? gamma_step(sum, dys[u], xs[u], means[u], rstds[u]) : __fadd_rn(sum, dys[u]);
    }}
    for (; n < N; n += 32) {{
        const float dyv = read(dy[n * 32 + c]);
        sum = kind == 0 ? gamma_step(sum, dyv, read(x[n * 32 + c]), read(mean[n * G + g]), read(rstd[n * G + g]))
                        : __fadd_rn(sum, dyv);
    }}
    partial[(kind * 32 + r) * 32 + c] = sum;
}}
// WarpReduceSum over the 32 residue chains of each channel: val += shfl_down(val, offset).
extern "C" __global__ void gamma_beta_tree(const float* partial, F* dgamma, F* dbeta) {{
    const int c = threadIdx.x, kind = blockIdx.x;
    float lane[32];
    for (int l = 0; l < 32; ++l) lane[l] = partial[(kind * 32 + l) * 32 + c];
    for (int offset = 16; offset > 0; offset /= 2)
        for (int l = 0; l < offset; ++l) lane[l] = __fadd_rn(lane[l], lane[l + offset]);
    (kind == 0 ? dgamma : dbeta)[c] = write(lane[0]);
}}
"""
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
        raw = nvrtc._cupy.RawModule(code=code, options=tuple(options))
        _KERNELS[dtype] = {name: raw.get_function(name) for name in ("gamma_beta_chains", "gamma_beta_tree")}
    return _KERNELS[dtype]


def gamma_beta_backward(grad, x, mean, rstd, groups):
    """dgamma, dbeta of native_group_norm for contiguous (N > 128, 32) inputs."""
    from rwkv.model import nvrtc
    N = x.shape[0]
    partial = torch.empty((2, 32, 32), device=x.device, dtype=torch.float32)
    dgamma = torch.empty(32, device=x.device, dtype=x.dtype)
    dbeta = torch.empty(32, device=x.device, dtype=x.dtype)
    functions = kernel(x.dtype)
    pointers = tuple(np.uint64(v.data_ptr()) for v in (grad, x, mean, rstd, partial))
    nvrtc.raw_launch(functions["gamma_beta_chains"], x.device, (32, 2), (32,), pointers + (np.int64(N), np.int32(groups)))
    nvrtc.raw_launch(functions["gamma_beta_tree"], x.device, (2,), (32,),
                     tuple(np.uint64(v.data_ptr()) for v in (partial, dgamma, dbeta)))
    return dgamma, dbeta


class ExactGroupNorm(torch.autograd.Function):
    """torch.nn.GroupNorm on a contiguous (N, 32) input with affine weight and bias."""

    @staticmethod
    def forward(ctx, x, weight, bias, groups: int, eps: float):
        if not x.is_cuda or x.dim() != 2 or x.shape[1] != 32 or not x.is_contiguous() or x.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Exact GroupNorm requires a contiguous 2-D CUDA float32/BF16 input with 32 channels")
        N, C = x.shape
        out, mean, rstd = torch.ops.aten.native_group_norm(x, weight, bias, N, C, 1, groups, eps)
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.groups = groups
        return out

    @staticmethod
    def backward(ctx, grad):
        x, weight, mean, rstd = ctx.saved_tensors
        N, C = x.shape
        grad = grad.contiguous()
        if N <= 128:
            # PyTorch uses its column-wise kernel here; keep it unchanged.
            dx, dgamma, dbeta = torch.ops.aten.native_group_norm_backward(
                grad, x, mean, rstd, weight, N, C, 1, ctx.groups, [True, True, True])
        else:
            dx, _, _ = torch.ops.aten.native_group_norm_backward(
                grad, x, mean, rstd, weight, N, C, 1, ctx.groups, [True, False, False])
            dgamma, dbeta = gamma_beta_backward(grad, x, mean, rstd, ctx.groups)
        return dx, dgamma, dbeta, None, None
