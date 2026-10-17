"""S09: fused CUDA step of the SM32b state-VQ QAT recurrence (NVRTC through CuPy, the rwkv/model/nvrtc.py pattern).

forward  : scripts/kernels/state_vq_wkv.cu `state_vq_wkv_forward`, one launch per split (one block per row and head
           runs all T steps; recurrence + rank-1 Gram-squaring factorization + joint residual VQ + log2 norm).
backward : the plain K=16 WKV backward of rwkv/model/nvrtc.py (`rwkv7_wkv_backward_kernel<1, float>`, the validated
           training kernel with a state checkpoint at EVERY step) fed the stored (quantized) state before every step,
           i.e. the straight-through gradient of expertium/model/state_vq_hooks.py VQStateWKV.backward.
The kernel computes StateCodec.wkv with factor "sq" (scripts/state_vq.py factorize_sq), so the recipe must be a "...p"
one; `factor_dtype` float32 (default) or float64 selects the factorization type. Picks can differ from the CPU torch
codec only on float32 near-ties (distance and log2 rounding), as for the earlier torch-on-CUDA path.

compile_only() checks both kernels with NVRTC on a machine without a usable GPU (no device, no context).
"""
from pathlib import Path
import os

import numpy as np
import torch

SOURCE = Path(__file__).with_name("kernels") / "state_vq_wkv.cu"
BACKWARD = "rwkv::rwkv7_wkv_backward_kernel<1, float>"
K = 16
_MODULES = {}


def _options(factor_dtype):
    options = ["--std=c++17"]
    if factor_dtype == torch.float64:
        options.append("-DFACTOR_DOUBLE")
    elif factor_dtype != torch.float32:
        raise ValueError(f"state_vq CUDA kernel: factor_dtype float32 or float64, not {factor_dtype}")
    return options


def backward_source():
    from rwkv.model import nvrtc
    return nvrtc.kernel_source(torch.float32, head=K)


def compile_only(arch="compute_120", factor_dtype=torch.float32, verbose_arch=None):
    """NVRTC compile of the forward and backward kernels without a CUDA device. Returns sizes (and, with
    verbose_arch='sm_120', the ptxas register report)."""
    if os.name == "nt":
        os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
    from cupy_backends.cuda.libs import nvrtc as N
    report = {}
    for name, source, extra, names in (("forward", SOURCE.read_text(), _options(factor_dtype), ()),
                                       ("backward", backward_source(), ["--std=c++17"], (BACKWARD,))):
        program = N.createProgram(source, name + ".cu", [], [])
        for expression in names:
            N.addNameExpression(program, expression)
        target = verbose_arch or arch
        options = extra + [f"--gpu-architecture={target}"] + (["--ptxas-options=-v"] if verbose_arch else [])
        try:
            N.compileProgram(program, options)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"NVRTC {name}: {exc}\n{N.getProgramLog(program)}") from exc
        log = N.getProgramLog(program)
        entry = {"ptx_bytes": len(N.getPTX(program)) if not verbose_arch else None}
        if verbose_arch:
            entry["ptxas"] = [line.strip() for line in log.splitlines() if "registers" in line or "spill" in line]
        report[name] = entry
    return report


def _functions(device, factor_dtype):
    from rwkv.model import nvrtc
    nvrtc.register()
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, factor_dtype)
    if key not in _MODULES:
        cupy = nvrtc._cupy
        with cupy.cuda.Device(index):
            fwd = cupy.RawModule(code=SOURCE.read_text(), options=tuple(_options(factor_dtype)))
            bwd = cupy.RawModule(code=backward_source(), options=("--std=c++17",), name_expressions=(BACKWARD,))
            _MODULES[key] = (fwd.get_function("state_vq_wkv_forward"), bwd.get_function(BACKWARD))
    return _MODULES[key]


def _launch(function, device, grid, block, args):
    from rwkv.model import nvrtc
    cupy = nvrtc._cupy
    values = tuple(np.uint64(a.data_ptr()) if isinstance(a, torch.Tensor) else a for a in args)
    with cupy.cuda.Device(device.index), cupy.cuda.ExternalStream(torch.cuda.current_stream(device).cuda_stream):
        function(grid, block, values)


class KernelCodec:
    """Device tensors and scalars of one StateCodec (rank 1, factor "sq") for the fused kernel."""

    def __init__(self, codec, device, factor_dtype=torch.float32):
        rec = codec.rec
        if rec.rank != 1 or rec.factor != "sq" or not codec.wkv_books[0]:
            raise ValueError("the fused state-VQ kernel needs a rank-1 recipe with factor 'sq' and coded directions "
                             f"(got rank {rec.rank}, factor {rec.factor!r})")
        if rec.norm_bits is None:
            raise ValueError("the fused state-VQ kernel needs quantized norms (norm_bits not None)")
        stages = codec.wkv_books[0]
        M = stages[0].shape[0]
        if any(cb.shape != (M, 2 * K) for cb in stages):
            raise ValueError("WKV codebook stages must all be [M, 32]")
        self.device, self.factor_dtype = device, factor_dtype
        self.books = torch.stack([cb.float().t().contiguous() for cb in stages]).to(device).contiguous()  # [S, 32, M]
        self.cn2 = torch.stack([n2.float() for n2 in codec.wkv_n2[0]]).to(device).contiguous()            # [S, M]
        self.stages, self.M = len(stages), M
        nq = codec.wkv_nq[0]
        if nq.bits == 0:
            self.norm = (np.int32(0), np.float32(0), np.float32(1), np.float32(1),
                         np.float32(2.0 ** ((nq.lo + nq.hi) / 2)))
        else:
            # torch casts the Python scalars of NormQuant to float32: lo, (hi - lo) in double first, levels exact
            self.norm = (np.int32(1), np.float32(nq.lo), np.float32(nq.hi - nq.lo), np.float32(2 ** nq.bits - 1),
                         np.float32(0))

    def forward(self, r, k, v, w, a, kd, skip):
        """Inputs float32 contiguous [B, T, H, 16], skip bool [B, T] -> out [B, T, H, 16], before [B, T, H, 16, 16]."""
        B, T, H, Kh = r.shape
        if Kh != K or r.dtype != torch.float32 or skip.dtype != torch.bool:
            raise ValueError("state-VQ kernel: float32 [B, T, H, 16] inputs and a bool skip mask")
        out = torch.empty_like(r)
        before = torch.empty(B, T, H, K, K, device=r.device, dtype=torch.float32)
        if B * T:
            fwd, _ = _functions(r.device, self.factor_dtype)
            _launch(fwd, r.device, (B, H), (K, K),
                    (np.int32(B), np.int32(T), np.int32(H), r, k, v, w, a, kd, skip, out, before, self.books, self.cn2,
                     np.int32(self.stages), np.int32(self.M), *self.norm))
        return out, before

    def backward(self, r, k, v, w, a, kd, skip, before, grad):
        return wkv_backward(r, k, v, w, a, kd, skip, before, grad)


class BackwardKernel:
    """impl "hybrid": the torch forward (any recipe) with the fused straight-through backward."""

    def backward(self, r, k, v, w, a, kd, skip, before, grad):
        return wkv_backward(r, k, v, w, a, kd, skip, before, grad)


def wkv_backward(r, k, v, w, a, kd, skip, before, grad):
    """Straight-through gradients along the stored trajectory `before` ([B, T, H, 16, 16], the state before every
    step): rwkv7_wkv_backward_kernel<1, float> with one checkpoint per step (L = T)."""
    B, T, H, Kh = r.shape
    if Kh != K or before.shape != (B, T, H, K, K) or not before.is_contiguous():
        raise ValueError(f"state-VQ backward: inputs [B, T, H, 16] and contiguous states [B, T, H, 16, 16], "
                         f"got {tuple(r.shape)} / {tuple(before.shape)}")
    grad = grad.float().contiguous()
    grads = tuple(torch.empty_like(t) for t in (r, k, v, w, a, kd))
    if B * T:
        _, bwd = _functions(r.device, torch.float32)
        _launch(bwd, r.device, (B, H), (K, K),
                (np.int32(B), np.int32(T), np.int32(H), r, k, v, w, a, kd, skip, grad, np.int32(T), before, *grads))
    return grads
