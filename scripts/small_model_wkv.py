"""Forward-only WKV-7 recurrence of any length for head size 16 or 32 (CUDA via CuPy), with optional per-row state
rounding (the fork's int8/int4 card/note state precision). Inference only.

Generalises scripts/compress_wkv_cuda.py (head size 32 only) to the Expertium head-size-16 models (w64, w80), so the
small-model track can score every candidate in the full-history protocol (whole history as one sequence), the
protocol of the fork's 2.76M comparison (artifacts/compression/heldout-b-final.md). One block of K threads per
(batch row, head); thread i holds state row i. Same arithmetic as compress_wkv_cuda (summation order aside).

    wkv_long(r, k, v, w, a, kd, skip, bits) -> out      # tensors [B, T, H, K], skip [B, T]; bits 0 / 8 / 4
"""
import os
from pathlib import Path

import torch

from scripts import compress_wkv_cuda as _base

_kernels = {}
_handles = []


def _source(K):
    text = _base.SOURCE.replace("32", str(K))
    mask = "0xffffffffu" if K == 32 else f"0x{(1 << K) - 1:x}u"
    return text.replace("__syncwarp()", f"__syncwarp({mask})")


def _kernel(K):
    if K not in (16, 32):
        raise ValueError(f"head size {K} not supported (16 or 32)")
    if K not in _kernels:
        if os.name == "nt" and not _handles:
            directory = str(Path(torch.__file__).parent / "lib")
            os.environ["PATH"] = directory + os.pathsep + os.environ["PATH"]
            _handles.append(os.add_dll_directory(directory))
        import cupy
        _kernels[K] = cupy.RawKernel(_source(K), "wkv_q", options=("--std=c++14",))
    return _kernels[K]


def wkv_long(r, k, v, w, a, kd, skip, bits=0):
    B, T, H, K = r.shape
    if not r.is_cuda:
        return reference(r, k, v, w, a, kd, skip, bits)
    import cupy
    kernel = _kernel(K)
    tensors = [x.float().contiguous() for x in (r, k, v, w, a, kd)]
    skip = skip.to(torch.uint8).contiguous()
    out = torch.empty_like(tensors[0])
    if B * H == 0 or T == 0:
        return out
    with cupy.cuda.Device(r.device.index), cupy.cuda.ExternalStream(torch.cuda.current_stream(r.device).cuda_stream):
        args = tuple(cupy.from_dlpack(x) for x in (*tensors, skip, out))
        kernel((B * H,), (K,), (*args, cupy.int32(B), cupy.int32(T), cupy.int32(H), cupy.int32(bits)))
    return out


def reference(r, k, v, w, a, kd, skip, bits=0):
    """Float32 per-step reference (CPU smoke and self-test); same update and rounding as the kernel."""
    from rwkv.model.rwkv_ops import single_timestep
    B, T, H, K = r.shape
    r, k, v, w, a, kd = (x.float() for x in (r, k, v, w, a, kd))
    state = torch.zeros(B, H, K, K, dtype=torch.float32, device=r.device)
    out = torch.empty(B, T, H, K, dtype=torch.float32, device=r.device)
    for t in range(T):
        o, nxt = single_timestep(r[:, t], k[:, t], v[:, t], w[:, t], a[:, t], kd[:, t], state)
        out[:, t] = o
        if bits:
            qmax = 127.0 if bits == 8 else 7.0
            scale = nxt.abs().amax(-1, keepdim=True) / qmax
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            nxt = torch.clamp(torch.round(nxt / scale), -qmax, qmax) * scale
        state = torch.where(skip[:, t, None, None, None].bool(), state, nxt)
    return out


def self_test(device="cuda", seed=0):
    """Max abs error vs the float32 reference for K in (16, 32) and state bits (0, 8); random inputs incl. skips."""
    g = torch.Generator().manual_seed(seed)
    results = {}
    for K in (16, 32):
        B, T, H = 3, 500, 4
        r, k, v, a, kd = (torch.randn(B, T, H, K, generator=g) * 0.3 for _ in range(5))
        kd = torch.nn.functional.normalize(kd, dim=-1)
        a = torch.sigmoid(a)
        w = torch.exp(-torch.exp(-0.5 - torch.rand(B, T, H, K, generator=g)))
        skip = torch.rand(B, T, generator=g) < 0.3
        for bits in (0, 8):
            ref = reference(r, k * a, v, w, a, kd, skip, bits)
            got = wkv_long(*(x.to(device) for x in (r, k * a, v, w, a, kd)), skip.to(device), bits).cpu()
            results[f"K{K}/bits{bits}"] = float((got - ref).abs().max())
    return results


if __name__ == "__main__":
    print(self_test())
