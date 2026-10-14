"""Forward-only RWKV-7 WKV recurrence for arbitrary sequence length on CUDA (CuPy/NVRTC), with optional per-row
state quantization (the fork's int8/int4 card/note state). Inference only (no backward).

The vendored NVRTC kernels accept at most 383 timesteps (enough for the R01 128-review windows); the full-history
protocol needs whole histories (up to ~1.7M rows in the user module). One CUDA block of 32 threads handles one
(batch row, head); thread i keeps state row i (32 floats, the value index) in registers and walks the sequence, so the
fork's per-row abs-max quantization is thread-local.

Per step (identical to rwkv.model.rwkv_ops.single_timestep / reference_rwkv7, summation order aside):
    x_i   = sum_j S_ij kd_j
    S'_ij = S_ij w_j - x_i a_j kd_j + v_i k_j
    out_i = sum_j S'_ij r_j                 (output always from the unrounded update)
    S     = S if skip_t else Q(S')          (Q = per-row symmetric abs-max rounding when bits > 0)

Registered as torch.ops.compress.wkv_q(r, k, v, w, a, kd, skip, bits) -> out; the schema is always defined (so the
TorchScript model compiles on CPU too), the CUDA implementation only by register_cuda().
"""
import os
from pathlib import Path

import torch

_defined = False
_cuda = None
_handles = []

SOURCE = r"""
extern "C" __global__ void wkv_q(const float* __restrict__ r, const float* __restrict__ k,
                                 const float* __restrict__ v, const float* __restrict__ w,
                                 const float* __restrict__ a, const float* __restrict__ kd,
                                 const unsigned char* __restrict__ skip, float* __restrict__ out,
                                 const int B, const int T, const int H, const int bits) {
    // One warp per (batch row, head); lane i owns state row i. The next step's inputs are prefetched into
    // registers while the current step computes, which hides most of the global-memory latency.
    const int bh = blockIdx.x;
    const int b = bh / H;
    const int h = bh - b * H;
    const int i = threadIdx.x;
    __shared__ float sr[32], sk[32], sw[32], sakd[32], skd[32];
    float S[32];
    float N[32];
    #pragma unroll
    for (int j = 0; j < 32; j++) S[j] = 0.0f;
    const float qmax = bits == 8 ? 127.0f : 7.0f;
    const long long stride = (long long)H * 32;
    long long base = ((long long)b * T * H + h) * 32 + i;
    float cr = 0.0f, ck = 0.0f, cw = 0.0f, ca = 0.0f, ckd = 0.0f, cv = 0.0f;
    unsigned char cs = 0;
    if (T > 0) {
        cr = r[base]; ck = k[base]; cw = w[base]; ca = a[base]; ckd = kd[base]; cv = v[base];
        cs = skip[(long long)b * T];
    }
    for (int t = 0; t < T; t++) {
        sr[i] = cr;
        sk[i] = ck;
        sw[i] = cw;
        skd[i] = ckd;
        sakd[i] = ca * ckd;
        const float vi = cv;
        const unsigned char skip_t = cs;
        const long long here = base;
        __syncwarp();
        if (t + 1 < T) {
            base += stride;
            cr = r[base]; ck = k[base]; cw = w[base]; ca = a[base]; ckd = kd[base]; cv = v[base];
            cs = skip[(long long)b * T + t + 1];
        }
        float x = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; j++) x += S[j] * skd[j];
        float o = 0.0f;
        float m = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            const float s = S[j] * sw[j] - x * sakd[j] + vi * sk[j];
            N[j] = s;
            o += s * sr[j];
            m = fmaxf(m, fabsf(s));
        }
        out[here] = o;
        if (!skip_t) {
            if (bits > 0) {
                float scale = m / qmax;
                if (!(scale > 0.0f)) scale = 1.0f;
                #pragma unroll
                for (int j = 0; j < 32; j++) S[j] = fminf(fmaxf(rintf(N[j] / scale), -qmax), qmax) * scale;
            } else {
                #pragma unroll
                for (int j = 0; j < 32; j++) S[j] = N[j];
            }
        }
        __syncwarp();
    }
}
"""


def define():
    """Define the op schema (idempotent). Needed before the patched TorchScript model is built."""
    global _defined
    if _defined:
        return
    try:
        torch.ops.compress.wkv_q  # already defined in this process
    except (AttributeError, RuntimeError):
        torch.library.define("compress::wkv_q",
                             "(Tensor r, Tensor k, Tensor v, Tensor w, Tensor a, Tensor kd, Tensor skip, int bits) -> Tensor")
    _defined = True


def _forward(r, k, v, w, a, kd, skip, bits):
    import cupy
    B, T, H, K = r.shape
    if K != 32:
        raise ValueError("head size must be 32")
    tensors = [x.float().contiguous() for x in (r, k, v, w, a, kd)]
    skip = skip.to(torch.uint8).contiguous()
    out = torch.empty_like(tensors[0])
    if B * H == 0 or T == 0:
        return out
    with cupy.cuda.Device(r.device.index), cupy.cuda.ExternalStream(torch.cuda.current_stream(r.device).cuda_stream):
        args = tuple(cupy.from_dlpack(x) for x in (*tensors, skip, out))
        _cuda((B * H,), (32,), (*args, cupy.int32(B), cupy.int32(T), cupy.int32(H), cupy.int32(bits)))
    return out


def register_cuda():
    global _cuda
    define()
    if _cuda is not None:
        return
    if os.name == "nt":
        directory = str(Path(torch.__file__).parent / "lib")
        os.environ["PATH"] = directory + os.pathsep + os.environ["PATH"]
        _handles.append(os.add_dll_directory(directory))
    import cupy
    _cuda = cupy.RawKernel(SOURCE, "wkv_q", options=("--std=c++14",))
    torch.library.impl("compress::wkv_q", "CUDA", _forward)


def self_test(device="cuda", seed=0):
    """Compare against the float32 reference recurrence on random inputs incl. skips; returns max abs errors."""
    from rwkv.model.rwkv_ops import single_timestep
    g = torch.Generator().manual_seed(seed)
    B, T, H, K = 3, 300, 4, 32
    r, k, v, a, kd = (torch.randn(B, T, H, K, generator=g) * 0.3 for _ in range(5))
    kd = torch.nn.functional.normalize(kd, dim=-1)
    a = torch.sigmoid(a)
    w = torch.exp(-torch.exp(-0.5 - torch.rand(B, T, H, K, generator=g)))
    skip = torch.rand(B, T, generator=g) < 0.3
    results = {}
    for bits in (0, 8, 4):
        state = torch.zeros(B, H, K, K, dtype=torch.float32)
        ref = torch.empty(B, T, H, K, dtype=torch.float32)
        for t in range(T):
            o, nxt = single_timestep(r[:, t], k[:, t] * a[:, t], v[:, t], w[:, t], a[:, t], kd[:, t], state)
            ref[:, t] = o
            if bits:
                qmax = 127.0 if bits == 8 else 7.0
                scale = nxt.abs().amax(-1, keepdim=True) / qmax
                scale = torch.where(scale > 0, scale, torch.ones_like(scale))
                nxt = torch.clamp(torch.round(nxt / scale), -qmax, qmax) * scale
            state = torch.where(skip[:, t, None, None, None], state, nxt)
        got = torch.ops.compress.wkv_q(*(x.to(device) for x in (r, k * a, v, w, a, kd)), skip.to(device), bits).cpu()
        results[bits] = float((got - ref).abs().max())
    return results
