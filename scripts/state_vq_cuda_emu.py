"""CPU execution of the S09 CUDA kernels (scripts/kernels/state_vq_wkv.cu and the K=16 WKV backward of
rwkv/model/nvrtc.py at checkpoint length 1) for scripts/check_state_vq_qat.py, on a machine whose GPU is not available.

The UNCHANGED kernel sources are compiled as C++20 with MSVC (Build Tools, found through vswhere/vcvars64) behind a
small shim: one OS thread per CUDA thread of a block (256), `__shared__` = function statics, __syncthreads = a
std::barrier over the block, warp shuffles = an exchange buffer between two barriers (CUDA lane / width semantics),
blocks run one after another. Float arithmetic is IEEE single/double as on the GPU, except that the GPU may contract
a*b+c into FMAs and its log2f/exp2f can differ by an ulp; a missing __syncthreads shows up as a race here as well.
Slow (about 10 ms per block step): for checks on a few rows only.
"""
from pathlib import Path
import ctypes
import os
import subprocess
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FORWARD = ROOT / "scripts" / "kernels" / "state_vq_wkv.cu"

SHIM = r"""
#include <cmath>
#include <cstdint>
#include <cstring>
#include <barrier>
#include <thread>
#include <vector>
#include <algorithm>
struct dim3 { unsigned x = 1, y = 1, z = 1; };
static thread_local dim3 threadIdx, blockIdx;
static thread_local int emu_tid = 0;
static dim3 blockDim, gridDim;
static std::barrier<>* emu_barrier = nullptr;
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __restrict__ __restrict
#define __shared__ static
inline void __syncthreads() { emu_barrier->arrive_and_wait(); }
inline float __int_as_float(int v) { float f; std::memcpy(&f, &v, 4); return f; }
static uint64_t emu_exchange_buffer[1024];
template <typename T> T emu_exchange(T value, int source) {
    uint64_t bits = 0; std::memcpy(&bits, &value, sizeof(T));
    emu_exchange_buffer[emu_tid] = bits;
    __syncthreads();
    uint64_t other = emu_exchange_buffer[source];
    __syncthreads();
    T out; std::memcpy(&out, &other, sizeof(T));
    return out;
}
template <typename T> T __shfl_xor_sync(unsigned, T value, int mask, int width = 32) {
    const int lane = emu_tid & 31, warp = emu_tid - lane;
    int src = lane ^ mask;
    if (src >= (lane / width + 1) * width) src = lane;   // later group: own value
    return emu_exchange(value, warp + src);
}
template <typename T> T __shfl_down_sync(unsigned, T value, unsigned delta, int width = 32) {
    const int lane = emu_tid & 31, warp = emu_tid - lane;
    const int src = ((lane % width) + (int)delta >= width) ? lane : lane + (int)delta;
    return emu_exchange(value, warp + src);
}
template <typename T> T __shfl_sync(unsigned, T value, int source, int width = 32) {
    const int lane = emu_tid & 31, warp = emu_tid - lane;
    return emu_exchange(value, warp + (lane / width) * width + (source % width));
}
template <class F> void emu_launch(dim3 grid, dim3 block, F body) {
    const int n = block.x * block.y * block.z;
    blockDim = block; gridDim = grid;
    std::barrier<> bar(n);
    emu_barrier = &bar;
    std::vector<std::thread> threads;
    for (int i = 0; i < n; i++) threads.emplace_back([&, i] {
        threadIdx = dim3{(unsigned)(i % block.x), (unsigned)((i / block.x) % block.y), (unsigned)(i / (block.x * block.y))};
        emu_tid = i;
        for (unsigned bz = 0; bz < grid.z; bz++) for (unsigned by = 0; by < grid.y; by++)
            for (unsigned bx = 0; bx < grid.x; bx++) {
                blockIdx = dim3{bx, by, bz};
                body();
                __syncthreads();
            }
    });
    for (auto& t : threads) t.join();
}
"""

ENTRY = r"""
#define EXPORT extern "C" __declspec(dllexport)
EXPORT void emu_forward(int B, int T, int H, const float* r, const float* k, const float* v, const float* w,
                        const float* a, const float* kd, const bool* skip, float* out, float* before,
                        const float* books, const float* cn2, int stages, int M, int norm_mode, float lo, float span,
                        float levels, float fixed) {
    emu_launch(dim3{(unsigned)B, (unsigned)H, 1}, dim3{16, 16, 1}, [&] {
        state_vq_wkv_forward(B, T, H, r, k, v, w, a, kd, skip, out, before, books, cn2, stages, M, norm_mode, lo, span,
                             levels, fixed);
    });
}
EXPORT void emu_backward(int B, int T, int H, const float* r, const float* k, const float* v, const float* w,
                         const float* a, const float* kd, const bool* skip, const float* grad, const float* states,
                         float* dr, float* dk, float* dv, float* dw, float* da, float* dkd) {
    emu_launch(dim3{(unsigned)B, (unsigned)H, 1}, dim3{16, 16, 1}, [&] {
        rwkv::rwkv7_wkv_backward_kernel<1, float>(B, T, H, r, k, v, w, a, kd, skip, grad, T, states, dr, dk, dv, dw,
                                                    da, dkd);
    });
}
"""


def _vcvars():
    base = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio"
    found = sorted(base.glob("*/*/VC/Auxiliary/Build/vcvars64.bat"), reverse=True)
    return found[0] if found else None


def available():
    return os.name == "nt" and _vcvars() is not None


_LIB = {}


def library(factor_double=False):
    """Build (once per process) the emulation DLL from the unchanged kernel sources; returns the ctypes handle."""
    if factor_double in _LIB:
        return _LIB[factor_double]
    vcvars = _vcvars()
    if vcvars is None:
        raise RuntimeError("no MSVC Build Tools (vcvars64.bat) for the CPU kernel emulation")
    from scripts.state_vq_cuda import backward_source
    folder = Path(tempfile.mkdtemp(prefix="state-vq-emu-"))
    backward = backward_source().replace("typedef long long int64_t;", "")
    defines = "#define FACTOR_DOUBLE\n" if factor_double else ""
    source = SHIM + defines + FORWARD.read_text() + "\n" + backward + "\n" + ENTRY
    (folder / "emu.cpp").write_text(source, encoding="utf-8")
    (folder / "build.bat").write_text(f'@call "{vcvars}" >nul\r\n'
                                      '@cl /nologo /std:c++20 /O2 /fp:strict /EHsc /LD emu.cpp /Fe:emu.dll\r\n')
    proc = subprocess.run(["cmd", "/c", str(folder / "build.bat")], cwd=folder, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"emulation build failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-2000:]}")
    lib = ctypes.CDLL(str(folder / "emu.dll"))
    _LIB[factor_double] = lib
    return lib


def _ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def forward(kernel_codec, r, k, v, w, a, kd, skip, factor_double=False):
    """CPU run of state_vq_wkv_forward with the KernelCodec's tensors (moved to CPU): (out, before)."""
    lib = library(factor_double)
    r, k, v, w, a, kd = (t.detach().float().contiguous().cpu() for t in (r, k, v, w, a, kd))
    skip = skip.contiguous().cpu()
    B, T, H, K = r.shape
    out = torch.zeros_like(r)
    before = torch.zeros(B, T, H, K, K)
    books, cn2 = kernel_codec.books.cpu().contiguous(), kernel_codec.cn2.cpu().contiguous()
    mode, lo, span, levels, fixed = kernel_codec.norm
    lib.emu_forward(ctypes.c_int(B), ctypes.c_int(T), ctypes.c_int(H), *map(_ptr, (r, k, v, w, a, kd, skip, out, before,
                                                                                books, cn2)),
                    ctypes.c_int(kernel_codec.stages), ctypes.c_int(kernel_codec.M), ctypes.c_int(int(mode)),
                    *(ctypes.c_float(float(x)) for x in (lo, span, levels, fixed)))
    return out, before


def backward(r, k, v, w, a, kd, skip, before, grad):
    """CPU run of rwkv7_wkv_backward_kernel<1, float> along `before`: (dr, dk, dv, dw, da, dkd)."""
    lib = library(False)
    r, k, v, w, a, kd, grad = (t.detach().float().contiguous().cpu() for t in (r, k, v, w, a, kd, grad))
    skip, before = skip.contiguous().cpu(), before.float().contiguous().cpu()
    B, T, H, _ = r.shape
    grads = [torch.zeros_like(r) for _ in range(6)]
    lib.emu_backward(ctypes.c_int(B), ctypes.c_int(T), ctypes.c_int(H),
                     *map(_ptr, (r, k, v, w, a, kd, skip, grad, before, *grads)))
    return tuple(grads)


if __name__ == "__main__":
    print("MSVC:", _vcvars())
    library()
    print("built", np.__version__)
