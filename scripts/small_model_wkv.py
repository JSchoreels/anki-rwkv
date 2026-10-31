"""Forward-only WKV-7 recurrence of any length for head size 16 or 32 (CUDA via CuPy), with optional per-row state
rounding (the fork's int8/int4 card/note state precision). Inference only.

Generalises scripts/compress_wkv_cuda.py (head size 32 only) to the Expertium head-size-16 models (w64, w80), so the
small-model track can score every candidate in the full-history protocol (whole history as one sequence), the
protocol of the fork's 2.76M comparison (artifacts/compression/heldout-b-final.md). One block of K threads per
(batch row, head); thread i holds state row i. Same arithmetic as compress_wkv_cuda (summation order aside).

    wkv_long(r, k, v, w, a, kd, skip, bits) -> out      # tensors [B, T, H, K], skip [B, T]; bits 0 / 8 / 4

SM77: `wkv_long(..., state0=S0)` starts every (row, head) from S0 [B, H, K, K] (float32, row = value index) instead of
zeros: a separate kernel "wkv_q_s0" built from the same text with the state load and one more pointer (_state_source);
without state0 the stock kernel runs unchanged.

SM47: `bits | F16_SCALE` (8 | 256, 4 | 256) rounds every row scale (absmax / qmax) to IEEE float16 before the codes
are computed with it (round to nearest even, subnormals kept, clamped to the float16 maximum 65504; a scale that is 0,
or underflows to 0, or is NaN becomes 1 like the float32 path). Without the flag the kernel is unchanged.
"""
import os
from pathlib import Path

import torch

from scripts import compress_wkv_cuda as _base

_kernels = {}
_handles = []

F16_SCALE = 256   # flag on `bits`: float16 row scales (SM47)
F16_MAX = 65504.0

# The f32 -> f16 -> f32 rounding of a non-negative scale in the kernel: RNE to 10 mantissa bits in the normal range,
# to multiples of 2^-24 below 2^-14 (the float16 subnormals); NaN passes through (then becomes 1).
_F16_DEVICE = r"""
__device__ __forceinline__ float f16_round_scale(float x) {
    if (!(x < 65504.0f)) return x >= 65504.0f ? 65504.0f : x;
    if (x < 6.103515625e-05f) return rintf(x * 16777216.0f) * 5.9604644775390625e-08f;
    unsigned int u = __float_as_uint(x);
    u += 0x0FFFu + ((u >> 13) & 1u);
    return __uint_as_float(u & 0xFFFFE000u);
}
"""


def round_scale_f16(scale):
    """Torch twin of the kernel's f16_round_scale (CPU reference, scripts/state_vq.int_row_quant)."""
    return torch.clamp(scale, max=F16_MAX).half().float()


def _source(K):
    text = _base.SOURCE.replace("32", str(K))
    mask = "0xffffffffu" if K == 32 else f"0x{(1 << K) - 1:x}u"
    text = text.replace("__syncwarp()", f"__syncwarp({mask})")
    # SM47: the F16_SCALE flag (bits 0 / 4 / 8 keep the exact old arithmetic)
    for old, new in (('extern "C"', _F16_DEVICE + 'extern "C"'),
                     ("const float qmax = bits == 8 ? 127.0f : 7.0f;",
                      "const float qmax = (bits & 255) == 8 ? 127.0f : 7.0f;"),
                     ("float scale = m / qmax;\n",
                      "float scale = m / qmax;\n                if (bits & 256) scale = f16_round_scale(scale);\n")):
        if text.count(old) != 1:
            raise RuntimeError(f"kernel source changed: {old!r}")
        text = text.replace(old, new)
    return text


def _state_source(K):
    """SM77: the kernel text with the state loaded from `state0` (one more pointer), renamed wkv_q_s0."""
    text = _source(K)
    for old, new in (('void wkv_q(', 'void wkv_q_s0('),
                     ("const int B, const int T, const int H, const int bits) {",
                      "const int B, const int T, const int H, const int bits,\n"
                      "                                 const float* __restrict__ state0) {"),
                     (f"for (int j = 0; j < {K}; j++) S[j] = 0.0f;",
                      f"for (int j = 0; j < {K}; j++) S[j] = state0[((long long)bh * {K} + i) * {K} + j];")):
        if text.count(old) != 1:
            raise RuntimeError(f"kernel source changed (state rewrite): {old!r}")
        text = text.replace(old, new)
    return text


def _kernel(K, state=False):
    if K not in (16, 32):
        raise ValueError(f"head size {K} not supported (16 or 32)")
    if state:
        key = (K, "state")
        if key not in _kernels:
            _kernel(K)  # the DLL directory set-up
            import cupy
            _kernels[key] = cupy.RawKernel(_state_source(K), "wkv_q_s0", options=("--std=c++14",))
        return _kernels[key]
    if K not in _kernels:
        if os.name == "nt" and not _handles:
            directory = str(Path(torch.__file__).parent / "lib")
            os.environ["PATH"] = directory + os.pathsep + os.environ["PATH"]
            _handles.append(os.add_dll_directory(directory))
        import cupy
        _kernels[K] = cupy.RawKernel(_source(K), "wkv_q", options=("--std=c++14",))
    return _kernels[K]


def wkv_long(r, k, v, w, a, kd, skip, bits=0, state0=None):
    B, T, H, K = r.shape
    if not r.is_cuda:
        return reference(r, k, v, w, a, kd, skip, bits, state0)
    import cupy
    kernel = _kernel(K, state=state0 is not None)
    tensors = [x.float().contiguous() for x in (r, k, v, w, a, kd)]
    skip = skip.to(torch.uint8).contiguous()
    out = torch.empty_like(tensors[0])
    if B * H == 0 or T == 0:
        return out
    with cupy.cuda.Device(r.device.index), cupy.cuda.ExternalStream(torch.cuda.current_stream(r.device).cuda_stream):
        args = tuple(cupy.from_dlpack(x) for x in (*tensors, skip, out))
        scalars = (cupy.int32(B), cupy.int32(T), cupy.int32(H), cupy.int32(bits))
        if state0 is None:
            kernel((B * H,), (K,), (*args, *scalars))
        else:
            s0 = state0.float().contiguous()
            if s0.shape != (B, H, K, K) or s0.device != r.device:
                raise ValueError(f"state0 must be {(B, H, K, K)} on {r.device}, got {tuple(s0.shape)} {s0.device}")
            kernel((B * H,), (K,), (*args, *scalars, cupy.from_dlpack(s0)))
    return out


def reference(r, k, v, w, a, kd, skip, bits=0, state0=None, steps=None):
    """Float32 per-step reference (CPU smoke and self-test); same update and rounding as the kernel. `steps`: a list
    that receives each step's largest row scale (the state quantisation step; self_test_state's bits-8 bar)."""
    from rwkv.model.rwkv_ops import single_timestep
    B, T, H, K = r.shape
    r, k, v, w, a, kd = (x.float() for x in (r, k, v, w, a, kd))
    state = (torch.zeros(B, H, K, K, dtype=torch.float32, device=r.device) if state0 is None
             else state0.float().clone())
    out = torch.empty(B, T, H, K, dtype=torch.float32, device=r.device)
    for t in range(T):
        o, nxt = single_timestep(r[:, t], k[:, t], v[:, t], w[:, t], a[:, t], kd[:, t], state)
        out[:, t] = o
        if bits:
            qmax = 127.0 if (bits & 255) == 8 else 7.0
            scale = nxt.abs().amax(-1, keepdim=True) / qmax
            if bits & F16_SCALE:
                scale = round_scale_f16(scale)
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            if steps is not None:
                steps.append(float(scale.max()))
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
        for bits in (0, 8, 8 | F16_SCALE, 4 | F16_SCALE):
            ref = reference(r, k * a, v, w, a, kd, skip, bits)
            got = wkv_long(*(x.to(device) for x in (r, k * a, v, w, a, kd)), skip.to(device), bits).cpu()
            name = f"{bits & 255}h" if bits & F16_SCALE else f"{bits}"
            results[f"K{K}/bits{name}"] = float((got - ref).abs().max())
    return results


def self_test_state(device="cuda", seed=0):
    """SM77: max abs error of wkv_long(..., state0) vs the float32 reference from the same S0 (K 16 and 32, bits 0 / 8).

    bits 8 also reports what its bar needs (scripts/sm77_warm.py gpu_check, check L): the state is rounded to 8-bit codes
    every step, so a float32 summation-order difference of ~1e-7 that lands next to a rounding boundary moves one code
    by one step (row absmax / 127) and the output by up to step * |r|_1 from then on. The kernel and the reference sum
    in another order, so their bits-8 outputs are not continuous in that noise: the float32 reference itself against a
    float64 run of the same recurrence differs by 5e-4..1.2e-3 on these inputs, from S0 and from zero alike (CPU,
    10-31), while bits 0 agree to 1e-7. Keys: `/step` = the largest quantisation step of the reference run,
    `/bound` = step * max_t |r_t|_1 (one step on every element of a state row), `/frac_over_1e-4` = the share of output
    elements off by more than 1e-4 (code flips are rare events; a wrong kernel moves almost every element)."""
    g = torch.Generator().manual_seed(seed)
    results = {}
    for K in (16, 32):
        B, T, H = 3, 300, 4
        r, k, v, a, kd = (torch.randn(B, T, H, K, generator=g) * 0.3 for _ in range(5))
        kd = torch.nn.functional.normalize(kd, dim=-1)
        a = torch.sigmoid(a)
        w = torch.exp(-torch.exp(-0.5 - torch.rand(B, T, H, K, generator=g)))
        skip = torch.rand(B, T, generator=g) < 0.3
        state0 = torch.randn(B, H, K, K, generator=g) * 0.2
        for bits in (0, 8):
            steps = []
            ref = reference(r, k * a, v, w, a, kd, skip, bits, state0, steps)
            got = wkv_long(*(x.to(device) for x in (r, k * a, v, w, a, kd)), skip.to(device), bits, state0.to(device)).cpu()
            zero = wkv_long(*(x.to(device) for x in (r, k * a, v, w, a, kd)), skip.to(device), bits,
                            torch.zeros_like(state0).to(device)).cpu()
            stock = wkv_long(*(x.to(device) for x in (r, k * a, v, w, a, kd)), skip.to(device), bits).cpu()
            error = (got - ref).abs()
            results[f"K{K}/bits{bits}"] = float(error.max())
            if bits:
                results[f"K{K}/bits{bits}/step"] = max(steps)
                results[f"K{K}/bits{bits}/bound"] = max(steps) * float(r.abs().sum(-1).max())
                results[f"K{K}/bits{bits}/frac_over_1e-4"] = float((error > 1e-4).float().mean())
            results[f"K{K}/bits{bits}/zero_state_equals_stock_bitwise"] = bool(torch.equal(zero, stock))
    return results


if __name__ == "__main__":
    print(self_test())
