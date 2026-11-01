"""F26: time-shift gather whose backward avoids the deterministic sort, bit for bit.

With deterministic algorithms, PyTorch computes the gather backward as
index_put_(accumulate=True) over linearized indices: a radix sort, then a warp
per destination that sums duplicates in float in source order (warp-tree for
the first multiple-of-32 block, then sequential), adds the zero-initialized
destination and rounds once. This module builds the same per-destination source
lists from the (B, T) shift indices only and reproduces that arithmetic exactly.
Forward is torch.gather itself.

F27 does the same for index_select along rows: PyTorch's deterministic index_add
for row strides up to the warp width sums duplicates sequentially in float, in
source order, and adds the zero-initialized destination before rounding once.
"""

import numpy as np
import torch

_KERNELS = {}
_shared_orders = False
_orders = {}
_row_orders = {}
_prefetch_rows = False
_prefetch_shift = False


def set_prefetch_shift(enabled=False):
    """F46: pipeline shift source loads without changing lane accumulation order."""
    global _prefetch_shift
    _prefetch_shift = bool(enabled)


def set_prefetch_rows(enabled=False):
    """F35: row-selection backward with loads issued ahead of the unchanged sequential sum."""
    global _prefetch_rows
    _prefetch_rows = bool(enabled)


def set_shared_orders(enabled=False):
    """F38: reuse a split's shift-backward sort across its layers and mixers."""
    global _shared_orders
    _shared_orders = bool(enabled)
    _orders.clear()
    _row_orders.clear()


def shift_order(select):
    """Stable source order and per-destination starts for a contiguous (B, T) select."""
    B, T = select.shape
    if _shared_orders:
        # The entry holds `select`, so its storage cannot be reallocated while cached:
        # an equal pointer, layout and version counter means the same unmodified indices.
        key = (select.data_ptr(), B, T, select.stride(), select.dtype, select._version, select.device)
        cached = _orders.get(key)
        if cached is not None:
            return cached[1], cached[2]
    keys = (torch.arange(B, device=select.device).unsqueeze(1) * T + select).reshape(-1)
    sorted_keys, order = torch.sort(keys, stable=True)
    starts = torch.searchsorted(sorted_keys, torch.arange(B * T + 1, device=select.device))
    if _shared_orders:
        if len(_orders) >= 64:
            del _orders[next(iter(_orders))]
        _orders[key] = (select, order, starts)
    return order, starts


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
extern "C" __global__ void shift_backward(const F* grad, const long long* order, const long long* starts,
    F* out, long long rows, int C) {{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * C) return;
    int c = (int)(i % C);
    long long row = i / C;
    long long start = starts[row], count = starts[row + 1] - start;
    long long passes = count / 32;
    float lane[32];
    for (int l = 0; l < 32; l++) lane[l] = 0.0f;
    for (long long p = 0; p < passes; p++)
        for (int l = 0; l < 32; l++)
            lane[l] += read(grad[order[start + p * 32 + l] * C + c]);
    // __shfl_down_sync tree: lanes beyond the warp keep their own value.
    for (int offset = 16; offset > 0; offset /= 2) {{
        float next[32];
        for (int l = 0; l < 32; l++) next[l] = lane[l] + (l + offset < 32 ? lane[l + offset] : lane[l]);
        for (int l = 0; l < 32; l++) lane[l] = next[l];
    }}
    float gradient = lane[0];
    for (long long j = passes * 32; j < count; j++)
        gradient += read(grad[order[start + j] * C + c]);
    out[i] = write(0.0f + gradient);
}}
extern "C" __global__ void rows_backward(const F* grad, const long long* order, const long long* starts,
    F* out, long long rows, int C) {{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * C) return;
    int c = (int)(i % C);
    long long row = i / C;
    float gradient = 0.0f;
    for (long long j = starts[row]; j < starts[row + 1]; j++)
        gradient += read(grad[order[j] * C + c]);
    out[i] = write(0.0f + gradient);
}}
// Wide rows (C > 32): PyTorch's general indexing_backward_kernel reads the destination,
// adds one source in float and stores it in the destination dtype, once per duplicate
// in source order, starting from the zero-initialized destination.
extern "C" __global__ void rows_backward_wide(const F* grad, const long long* order, const long long* starts,
    F* out, long long rows, int C) {{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * C) return;
    int c = (int)(i % C);
    long long row = i / C;
    F weight = write(0.0f);
    for (long long j = starts[row]; j < starts[row + 1]; j++)
        weight = write(read(weight) + read(grad[order[j] * C + c]));
    out[i] = weight;
}}
// Pure row copy for index_copy with unique targets: out[target[i]] = src[i].
extern "C" __global__ void copy_rows(const F* src, const long long* target, F* out, long long n, int C) {{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * C) return;
    long long row = i / C;
    int c = (int)(i % C);
    out[target[row] * C + c] = src[i];
}}
// F35: the same sequential float sum in source order. Each step loads 64 values whose
// indices were fetched during the previous step, then fetches the next 64 indices
// before adding the values, so long destinations are not bound by dependent loads.
extern "C" __global__ void rows_backward_prefetch(const F* grad, const long long* order, const long long* starts,
    F* out, long long rows, int C) {{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * C) return;
    int c = (int)(i % C);
    long long row = i / C;
    float gradient = 0.0f;
    long long j = starts[row], end = starts[row + 1];
    long long source[64];
    float value[64];
    if (j + 64 <= end) {{
        #pragma unroll
        for (int u = 0; u < 64; u++) source[u] = order[j + u];
    }}
    for (; j + 64 <= end; j += 64) {{
        #pragma unroll
        for (int u = 0; u < 64; u++) value[u] = read(grad[source[u] * C + c]);
        if (j + 128 <= end) {{
            #pragma unroll
            for (int u = 0; u < 64; u++) source[u] = order[j + 64 + u];
        }}
        #pragma unroll
        for (int u = 0; u < 64; u++) gradient += value[u];
    }}
    for (; j < end; j++)
        gradient += read(grad[order[j] * C + c]);
    out[i] = write(0.0f + gradient);
}}
"""
        # Clone the accepted kernel's arithmetic; only the pass-load schedule differs.
        shift_code = code[code.index('extern "C" __global__ void shift_backward'):code.index('extern "C" __global__ void rows_backward')]
        shift_code = shift_code.replace("void shift_backward(", "void shift_backward_prefetch(")
        shift_code = shift_code.replace(
            """    for (long long p = 0; p < passes; p++)
        for (int l = 0; l < 32; l++)
            lane[l] += read(grad[order[start + p * 32 + l] * C + c]);""",
            """    long long source[32];
    float value[32];
    if (passes > 0) {
        #pragma unroll
        for (int l = 0; l < 32; l++) source[l] = order[start + l];
    }
    for (long long p = 0; p < passes; p++) {
        #pragma unroll
        for (int l = 0; l < 32; l++) value[l] = read(grad[source[l] * C + c]);
        if (p + 1 < passes) {
            #pragma unroll
            for (int l = 0; l < 32; l++) source[l] = order[start + (p + 1) * 32 + l];
        }
        #pragma unroll
        for (int l = 0; l < 32; l++) lane[l] += value[l];
    }""")
        code += shift_code
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
        raw = nvrtc._cupy.RawModule(code=code, options=tuple(options))
        _KERNELS[dtype] = {name: raw.get_function(name) for name in ("shift_backward", "shift_backward_prefetch", "rows_backward", "rows_backward_prefetch",
                                                                "rows_backward_wide", "copy_rows")}
    return _KERNELS[dtype]


def shift_backward(grad, select):
    """Gradient of torch.gather(x, 1, select[..., None].expand(-1, -1, C)) with respect to x."""
    from rwkv.model import nvrtc
    grad, select = grad.contiguous(), select.contiguous()
    B, T, C = grad.shape
    order, starts = shift_order(select)
    out = torch.empty_like(grad)
    args = tuple(np.uint64(v.data_ptr()) for v in (grad, order, starts, out)) + (np.int64(B * T), np.int32(C))
    name = "shift_backward_prefetch" if _prefetch_shift and grad.dtype == torch.bfloat16 else "shift_backward"
    nvrtc.raw_launch(kernel(grad.dtype)[name], grad.device, ((B * T * C + 255) // 256,), (256,), args)
    return out


class ExactShiftGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, select):
        if not x.is_cuda or x.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Exact shift gather requires CUDA float32 or BF16")
        ctx.save_for_backward(select)
        return torch.gather(x, 1, select.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

    @staticmethod
    def backward(ctx, grad):
        (select,) = ctx.saved_tensors
        return shift_backward(grad, select), None


def row_order(index, rows):
    """Stable source order and per-destination starts of an index_select backward (sources -1 excluded)."""
    index = index.contiguous().long()
    sorted_rows, order = torch.sort(index, stable=True)
    return order, torch.searchsorted(sorted_rows, torch.arange(rows + 1, device=index.device))


def rows_backward(grad, index, rows, orders=None):
    """Gradient of torch.index_select(x, 0, index) for a 2-D x with `rows` rows.

    Width <= 32 reproduces PyTorch's small-stride kernel (float sum, one rounding); wider rows reproduce its
    general kernel (one rounding to the destination dtype per source). `orders` = row_order(index, rows) reuses a
    sort across calls with the same index (wide kernels, interleaved streams).
    """
    from rwkv.model import nvrtc
    grad, index = grad.contiguous(), index.contiguous().long()
    C = grad.shape[1]
    if C > 32:
        order, starts = orders if orders is not None else row_order(index, rows)
        out = torch.empty((rows, C), device=grad.device, dtype=grad.dtype)
        args = tuple(np.uint64(v.data_ptr()) for v in (grad, order, starts, out)) + (np.int64(rows), np.int32(C))
        nvrtc.raw_launch(kernel(grad.dtype)["rows_backward_wide"], grad.device, ((rows * C + 255) // 256,), (256,), args)
        return out
    cached = _row_orders.get((index.data_ptr(), index.numel(), index._version, index.device))
    if orders is not None:
        cached = (None, *orders)
    if cached is None:
        sorted_rows, order = torch.sort(index, stable=True)
        starts = torch.searchsorted(sorted_rows, torch.arange(rows + 1, device=index.device))
    else:
        _, order, starts = cached
        if starts.numel() != rows + 1:
            raise ValueError("F53 row-order destination count mismatch")
    out = torch.empty((rows, C), device=grad.device, dtype=grad.dtype)
    args = tuple(np.uint64(v.data_ptr()) for v in (grad, order, starts, out)) + (np.int64(rows), np.int32(C))
    # FP32 measured slower with prefetching (worst real case 2.3 to 3.1 ms); R01 trains in BF16.
    name = "rows_backward_prefetch" if _prefetch_rows and grad.dtype == torch.bfloat16 else "rows_backward"
    nvrtc.raw_launch(kernel(grad.dtype)[name], grad.device, ((rows * C + 255) // 256,), (256,), args)
    return out


class ExactIndexSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, index):
        if not x.is_cuda or x.dim() != 2 or x.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Exact row selection requires 2-D CUDA float32/BF16 rows")
        ctx.save_for_backward(index)
        ctx.rows = x.shape[0]
        return torch.index_select(x, 0, index)

    @staticmethod
    def backward(ctx, grad):
        (index,) = ctx.saved_tensors
        return rows_backward(grad, index, ctx.rows), None


class ExactPaddedIndexSelect(ExactIndexSelect):
    """F50: forward maps padding to row zero; backward omits proven zero sources.

    Only hierarchy padding whose downstream gradient is zero may use this path.
    Stable sorting retains every real source's sequential accumulation order.
    """
    @staticmethod
    def forward(ctx, x, index):
        if not x.is_cuda or x.dim() != 2 or x.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Exact padded row selection requires CUDA float32/BF16 rows")
        ctx.save_for_backward(index)
        ctx.rows = x.shape[0]
        return torch.index_select(x, 0, index.clamp(min=0))


class PreGatheredIndexSelect(ExactIndexSelect):
    """F51: independent original backward nodes after batched forward loads."""
    @staticmethod
    def forward(ctx, x, index, loaded):
        ctx.save_for_backward(index)
        ctx.rows = x.shape[0]
        return loaded

    @staticmethod
    def backward(ctx, grad):
        index, = ctx.saved_tensors
        return rows_backward(grad, index, ctx.rows), None, None


class FusedPaddedIndexSelect(ExactIndexSelect):
    """F52: clamp padding directly in the row-load kernel."""
    @staticmethod
    def forward(ctx, x, index):
        from rwkv.model.exact_glue import kernel
        from rwkv.model import nvrtc
        if not x.is_cuda or x.dim() != 2 or x.shape[1] != 32 or x.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Fused padding gather requires CUDA float32/BF16 width 32")
        ctx.save_for_backward(index)
        ctx.rows = x.shape[0]
        out = torch.empty((index.numel(), x.shape[1]), device=x.device, dtype=x.dtype)
        args = tuple(np.uint64(t.data_ptr()) for t in (x, index, out)) + (np.int32(out.numel()),)
        nvrtc.raw_launch(kernel(x.dtype, 'padded_forward'), x.device, ((out.numel()+255)//256,), (256,), args)
        return out


class OrderedIndexSelect(torch.autograd.Function):
    """Wide-kernels port: ExactIndexSelect with a precomputed row_order (one sort reused across layers)."""
    @staticmethod
    def forward(ctx, x, index, order, starts):
        if not x.is_cuda or x.dim() != 2 or x.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Exact row selection requires 2-D CUDA float32/BF16 rows")
        ctx.save_for_backward(index, order, starts)
        ctx.rows = x.shape[0]
        return torch.index_select(x, 0, index)

    @staticmethod
    def backward(ctx, grad):
        index, order, starts = ctx.saved_tensors
        return rows_backward(grad, index, ctx.rows, (order, starts)), None, None, None


def copy_rows(x, target, source):
    """x.index_copy(0, target, source) for unique targets, without the deterministic index_put sort."""
    from rwkv.model import nvrtc
    out = x.clone()
    source, target = source.contiguous(), target.contiguous().long()
    n, C = source.shape
    if n:
        args = tuple(np.uint64(v.data_ptr()) for v in (source, target, out)) + (np.int64(n), np.int32(C))
        nvrtc.raw_launch(kernel(x.dtype)["copy_rows"], x.device, ((n * C + 255) // 256,), (256,), args)
    return out


class ExactIndexCopy(torch.autograd.Function):
    """index_copy along rows with unique targets; values and both gradients equal PyTorch's bit for bit
    (derivatives.yaml: self grad.index_fill(0, index, 0), source grad.index_select(0, index))."""
    @staticmethod
    def forward(ctx, x, target, source):
        if not x.is_cuda or x.dim() != 2 or x.dtype != source.dtype or x.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Exact index copy requires 2-D CUDA float32/BF16 rows of one dtype")
        ctx.save_for_backward(target)
        return copy_rows(x, target, source)

    @staticmethod
    def backward(ctx, grad):
        (target,) = ctx.saved_tensors
        return grad.index_fill(0, target, 0), None, grad.index_select(0, target)
