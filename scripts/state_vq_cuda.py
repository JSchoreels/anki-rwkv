"""S09: fused CUDA step of the SM32b state-VQ QAT recurrence (NVRTC through CuPy, the rwkv/model/nvrtc.py pattern).

forward  : scripts/kernels/state_vq_wkv.cu `state_vq_wkv_forward`, one launch per split (one block per row and head
           runs all T steps; recurrence + rank-1 Gram-squaring factorization + joint residual VQ + log2 norm).
backward : the plain K=16 WKV backward of rwkv/model/nvrtc.py (`rwkv7_wkv_backward_kernel<1, float>`, the validated
           training kernel with a state checkpoint at EVERY step) fed the stored (quantized) state before every step,
           i.e. the straight-through gradient of expertium/model/state_vq_hooks.py VQStateWKV.backward.
The kernel computes StateCodec.wkv with factor "sq" (scripts/state_vq.py factorize_sq), so the recipe must be a "...p"
one; `factor_dtype` float32 (default) or float64 selects the factorization type. Picks can differ from the CPU torch
codec only on float32 near-ties (distance and log2 rounding), as for the earlier torch-on-CUDA path.

SM32c (learned codebooks, carried state): `KernelCodec.forward(..., s0=, record=True)` also returns the final stored
state, the stage picks and q(n)^2 of every coded step; `wkv_backward(..., dstored=True)` runs a copy of the backward
kernel (`rwkv7_wkv_backward_dstored_kernel`, backward_dstored_source: one extra write) that also returns dL/d(stored
state) of every coded step, from which `codebook_grad` forms the centroid gradients. With the defaults the SM32b
kernels and launches are unchanged.

compile_only() checks both kernels with NVRTC on a machine without a usable GPU (no device, no context).
"""
from pathlib import Path
import os

import numpy as np
import torch

SOURCE = Path(__file__).with_name("kernels") / "state_vq_wkv.cu"
BACKWARD = "rwkv::rwkv7_wkv_backward_kernel<1, float>"
BACKWARD_DSTORED = "rwkv::rwkv7_wkv_backward_dstored_kernel<1, float>"
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


def _patch(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"state-VQ dstored patch: {old!r} found {text.count(old)} times in the backward kernel")
    return text.replace(old, new)


def backward_dstored_source():
    """backward_source() plus a renamed copy of the backward kernel with one more argument, `dstored_BTHKK`
    [B, T, H, 16, 16]: at every non-skipped step t it receives dL/d(stored state after t) (the carried gradient
    dS_xy_contrib before the step adds it), i.e. the gradient the straight-through rule hands to the quantizer's
    output. Every other line of the copy is the validated kernel's; the original kernel stays in the source unchanged."""
    source = backward_source()
    start = source.index("template <int CHUNK_LEN=32, typename F>\n__global__ void rwkv7_wkv_backward_kernel(")
    end = source.index("\n}\n", source.index("dS_xy_contrib += dS_xy_decay", start)) + 3
    body = source[start:end]
    body = _patch(body, "rwkv7_wkv_backward_kernel(", "rwkv7_wkv_backward_dstored_kernel(")
    body = _patch(body, "F* __restrict__ k_deformed_grad_BTHK\n    ) {",
                  "F* __restrict__ k_deformed_grad_BTHK,\n    float* __restrict__ dstored_BTHKK\n    ) {")
    body = _patch(body, "            if (!skip) {\n                dS_xy += dS_xy_contrib;",
                  "            if (!skip) {\n                dstored_BTHKK[get_index4(b, t, h, x, y, T, H, K, K)] = dS_xy_contrib;\n"
                  "                dS_xy += dS_xy_contrib;")
    namespace_end = source.rindex("\n}")
    return source[:namespace_end] + "\n" + body + source[namespace_end:]


def compile_only(arch="compute_120", factor_dtype=torch.float32, verbose_arch=None):
    """NVRTC compile of the forward and backward kernels without a CUDA device. Returns sizes (and, with
    verbose_arch='sm_120', the ptxas register report)."""
    if os.name == "nt":
        os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
    from cupy_backends.cuda.libs import nvrtc as N
    report = {}
    for name, source, extra, names in (("forward", SOURCE.read_text(), _options(factor_dtype), ()),
                                       ("backward", backward_source(), ["--std=c++17"], (BACKWARD,)),
                                       ("backward_dstored", backward_dstored_source(), ["--std=c++17"],
                                        (BACKWARD, BACKWARD_DSTORED))):
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


def _dstored_function(device):
    from rwkv.model import nvrtc
    nvrtc.register()
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, "dstored")
    if key not in _MODULES:
        cupy = nvrtc._cupy
        with cupy.cuda.Device(index):
            raw = cupy.RawModule(code=backward_dstored_source(), options=("--std=c++17",),
                                 name_expressions=(BACKWARD_DSTORED,))
            _MODULES[key] = raw.get_function(BACKWARD_DSTORED)
    return _MODULES[key]


def _launch(function, device, grid, block, args):
    from rwkv.model import nvrtc
    cupy = nvrtc._cupy
    values = tuple(np.uint64(a.data_ptr()) if isinstance(a, torch.Tensor) else (np.uint64(0) if a is None else a)
                   for a in args)
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

    def refresh(self, stages):
        """SM32c: take the current values of the learned stage tensors (`stages` [S, M, 32]) before a launch."""
        stages = stages.detach().float()
        self.books = stages.transpose(1, 2).contiguous()
        self.cn2 = (stages * stages).sum(-1).contiguous()

    def forward(self, r, k, v, w, a, kd, skip, s0=None, record=False):
        """Inputs float32 contiguous [B, T, H, 16], skip bool [B, T] -> out [B, T, H, 16], before [B, T, H, 16, 16].
        s0 [B, H, 16, 16] float32 (or None = zeros) = the state step 0 starts from. record=True returns
        (out, before, final [B, H, 16, 16], picks [B, T, H, S] int32 (-1 = not coded), cscale [B, T, H])."""
        B, T, H, Kh = r.shape
        if Kh != K or r.dtype != torch.float32 or skip.dtype != torch.bool:
            raise ValueError("state-VQ kernel: float32 [B, T, H, 16] inputs and a bool skip mask")
        if s0 is not None and (s0.shape != (B, H, K, K) or s0.dtype != torch.float32 or not s0.is_contiguous()):
            raise ValueError("state-VQ kernel: s0 must be contiguous float32 [B, H, 16, 16]")
        out = torch.empty_like(r)
        before = torch.empty(B, T, H, K, K, device=r.device, dtype=torch.float32)
        final = picks = cscale = None
        if record:
            final = s0.clone() if s0 is not None else torch.zeros(B, H, K, K, device=r.device, dtype=torch.float32)
            picks = torch.full((B, T, H, self.stages), -1, device=r.device, dtype=torch.int32)
            cscale = torch.zeros(B, T, H, device=r.device, dtype=torch.float32)
        if B * T:
            fwd, _ = _functions(r.device, self.factor_dtype)
            _launch(fwd, r.device, (B, H), (K, K),
                    (np.int32(B), np.int32(T), np.int32(H), r, k, v, w, a, kd, skip, out, before, self.books, self.cn2,
                     np.int32(self.stages), np.int32(self.M), *self.norm, s0, final, picks, cscale))
        if record:
            return out, before, final, picks, cscale
        return out, before

    def backward(self, r, k, v, w, a, kd, skip, before, grad, dstored=False):
        return wkv_backward(r, k, v, w, a, kd, skip, before, grad, dstored)


class BackwardKernel:
    """impl "hybrid": the torch forward (any recipe) with the fused straight-through backward."""

    def backward(self, r, k, v, w, a, kd, skip, before, grad, dstored=False):
        return wkv_backward(r, k, v, w, a, kd, skip, before, grad, dstored)


def wkv_backward(r, k, v, w, a, kd, skip, before, grad, dstored=False):
    """Straight-through gradients along the stored trajectory `before` ([B, T, H, 16, 16], the state before every
    step): rwkv7_wkv_backward_kernel<1, float> with one checkpoint per step (L = T). dstored=True: the patched copy
    (backward_dstored_source), returning (grads, dstored [B, T, H, 16, 16], zero on skipped steps)."""
    B, T, H, Kh = r.shape
    if Kh != K or before.shape != (B, T, H, K, K) or not before.is_contiguous():
        raise ValueError(f"state-VQ backward: inputs [B, T, H, 16] and contiguous states [B, T, H, 16, 16], "
                         f"got {tuple(r.shape)} / {tuple(before.shape)}")
    grad = grad.float().contiguous()
    grads = tuple(torch.empty_like(t) for t in (r, k, v, w, a, kd))
    if dstored:
        ds = torch.zeros(B, T, H, K, K, device=r.device, dtype=torch.float32)
        if B * T:
            _launch(_dstored_function(r.device), r.device, (B, H), (K, K),
                    (np.int32(B), np.int32(T), np.int32(H), r, k, v, w, a, kd, skip, grad, np.int32(T), before, *grads,
                     ds))
        return grads, ds
    if B * T:
        _, bwd = _functions(r.device, torch.float32)
        _launch(bwd, r.device, (B, H), (K, K),
                (np.int32(B), np.int32(T), np.int32(H), r, k, v, w, a, kd, skip, grad, np.int32(T), before, *grads))
    return grads


def codebook_grad(stages, picks, cscale, dstored, out=None):
    """SM32c centroid gradients of the joint (u, v) WKV catalog. stages [S, M, 32] (the values the forward used),
    picks [..., S'] int32 (the S' <= S stages of this codec, -1 = step not coded), cscale [...] = q(n)^2, dstored
    [..., 16, 16] = dL/d(stored state). The stored state is cscale acc_u acc_v^T with acc = the sum of the picked rows,
    so dL/dacc_u = cscale dstored acc_v and dL/dacc_v = cscale dstored^T acc_u, and every stage's picked row receives
    that 32-vector (embedding-style; the picks are not differentiated). Returns [S, M, 32] (accumulated into `out`)."""
    S_used = picks.shape[-1]
    picks = picks.reshape(-1, S_used).long()
    cscale = cscale.reshape(-1)
    dstored = dstored.reshape(-1, K, K)
    grad = torch.zeros_like(stages, dtype=torch.float32) if out is None else out
    rows = torch.nonzero(picks[:, 0] >= 0).flatten()
    if rows.numel() == 0:
        return grad
    picks, cscale, G = picks.index_select(0, rows), cscale.index_select(0, rows), dstored.index_select(0, rows)
    values = stages.detach().float()
    acc = values[0].index_select(0, picks[:, 0])
    for s in range(1, S_used):
        acc = acc + values[s].index_select(0, picks[:, s])
    u, v = acc[:, :K], acc[:, K:]
    gu = cscale[:, None] * (G @ v.unsqueeze(-1)).squeeze(-1)
    gv = cscale[:, None] * (G.transpose(1, 2) @ u.unsqueeze(-1)).squeeze(-1)
    g = torch.cat([gu, gv], dim=-1).to(grad.dtype)
    for s in range(S_used):
        grad[s].index_add_(0, picks[:, s], g)
    return grad
