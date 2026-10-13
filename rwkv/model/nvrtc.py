"""Optional float32/BF16 CUDA recurrence compiled by NVRTC without a host compiler.

Uses the upstream sequential kernels with matching float/BF16 type helpers.
The full extension remains the backend for long sequences.
"""

import os
from pathlib import Path

import numpy as np
import torch

_library = None
_modules = {}
_dll_handles = []
_cupy = None
_checkpoint_length = 32
_raw_pointer_launch = False
_stream_launch = False
_streams = {}
_functions = {}
_driver_launch = False
_driver = None
_driver_parameters = None


def set_driver_launch(enabled=False):
    """F65a: launch the identical CuPy function through the CUDA driver ABI."""
    global _driver_launch
    _driver_launch = bool(enabled)


def driver_launch(function, grid, block, args, stream):
    """Copy exact by-value arguments into thread-local, typed driver parameter slots."""
    global _driver, _driver_parameters
    import ctypes
    import threading
    if _driver is None:
        _driver = ctypes.WinDLL('nvcuda.dll') if os.name == 'nt' else ctypes.CDLL('libcuda.so.1')
        launch = _driver.cuLaunchKernel
        launch.argtypes = [ctypes.c_void_p, *([ctypes.c_uint] * 7), ctypes.c_void_p,
                           ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        launch.restype = ctypes.c_int
        _driver_parameters = threading.local()
    types = {np.dtype('uint64'): ctypes.c_uint64, np.dtype('int64'): ctypes.c_int64,
             np.dtype('int32'): ctypes.c_int32}
    signature = tuple(arg.dtype for arg in args)
    if any(dtype not in types for dtype in signature):
        raise TypeError('Native launcher accepts only explicit uint64/int64/int32 argument bytes')
    cache = getattr(_driver_parameters, 'cache', None)
    if cache is None:
        cache = _driver_parameters.cache = {}
    entry = cache.get(signature)
    if entry is None:
        fields = [(f'a{i}', types[dtype]) for i, dtype in enumerate(signature)]
        struct = type('KernelParameters', (ctypes.Structure,), {'_fields_': fields})
        values = struct()
        pointers = (ctypes.c_void_p * len(fields))(*(ctypes.addressof(values) + getattr(struct, name).offset for name, _ in fields))
        entry = cache[signature] = values, pointers, tuple(name for name, _ in fields)
    values, pointers, names = entry
    for name, value in zip(names, args):
        setattr(values, name, int(value))
    dimensions = tuple(grid) + (1,) * (3 - len(grid)) + tuple(block) + (1,) * (3 - len(block))
    if len(dimensions) != 6 or any(type(v) is not int or v < 1 for v in dimensions):
        raise ValueError('Driver launch requires positive integer grid/block dimensions')
    # cuLaunchKernel copies host argument values before returning; Torch retains tensor
    # storage, exactly as in the CuPy path. Slots are thread-local, not shared by callers.
    status = _driver.cuLaunchKernel(function.ptr, *dimensions, 0, stream, pointers, None)
    if status:
        raise RuntimeError(f'CUDA driver launch failed with status {status}')


def set_stream_launch(enabled=False):
    """F38: launch raw kernels with a cached stream argument instead of per-call CuPy contexts."""
    global _stream_launch
    _stream_launch = bool(enabled)


def raw_launch(kernel, device, grid, block, args):
    """Enqueue a CuPy RawKernel on Torch's current stream of `device`."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    if _driver_launch:
        pointer = torch.cuda.current_stream(device).cuda_stream
        with _cupy.cuda.Device(index):
            function = _functions.get((id(kernel), index))
            if function is None:
                function = _functions[(id(kernel), index)] = (kernel, kernel.kernel)
            driver_launch(function[1], grid, block, args, pointer)
        return
    if _stream_launch and index == torch.cuda.current_device():
        # Same kernel, arguments and stream; only the per-call Device/ExternalStream
        # context setup (about 12 us of host time) is skipped.
        pointer = torch.cuda.current_stream(device).cuda_stream
        stream = _streams.get((index, pointer))
        if stream is None:
            stream = _streams[(index, pointer)] = _cupy.cuda.ExternalStream(pointer, index)
        function = _functions.get((id(kernel), index))
        if function is None:
            with _cupy.cuda.Device(index):
                function = _functions[(id(kernel), index)] = (kernel, kernel.kernel)
        function[1](grid, block, args, 0, stream)
        return
    with _cupy.cuda.Device(index), _cupy.cuda.ExternalStream(torch.cuda.current_stream(device).cuda_stream):
        kernel(grid, block, args)


def set_raw_pointer_launch(enabled=False):
    """Opt into same-stream pointer arguments without DLPack stream negotiation."""
    global _raw_pointer_launch
    _raw_pointer_launch = bool(enabled)


def set_checkpoint_length(length=32):
    """Select recomputation tiles before a forward/backward pair, never between them."""
    global _checkpoint_length
    if length not in (8, 16, 32):
        raise ValueError("NVRTC checkpoint length must be 8, 16, or 32")
    _checkpoint_length = length


def kernel_source(dtype=torch.float32):
    sources = Path(__file__).parent / "csrc" / "cuda"
    source = (sources / "rwkv7_cuda.cu").read_text()
    start = source.index("namespace rwkv {")
    end = source.index("template <int CHUNK_LEN=32, typename F>\nstd::tuple", start)
    kernels = source[start:end].replace("std::min(", "nvrtc_min(") + "\n}"
    helpers = (sources / "rwkv7_cuda_utils.h").read_text()
    helpers = helpers[helpers.index("__device__ inline int64_t get_index1"):]
    conversions = ""
    if dtype == torch.bfloat16:
        conversions = """
template <> __device__ inline float to_float<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }
template <> __device__ inline __nv_bfloat16 to_F<__nv_bfloat16>(float x) { return __float2bfloat16(x); }
"""
    prefix = "#include <cuda_bf16.h>\n" if dtype == torch.bfloat16 else ""
    return prefix + """typedef long long int64_t;
#define FULL_MASK 0xffffffff
template <typename F> __device__ inline float to_float(F x) { return x; }
template <typename F> __device__ inline F to_F(float x) { return x; }
__device__ inline int nvrtc_min(int x, int y) { return x < y ? x : y; }
""" + conversions + helpers + kernels


def module(device, dtype):
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, dtype, _checkpoint_length)
    if key not in _modules:
        with _cupy.cuda.Device(index):
            scalar = "float" if dtype == torch.float32 else "__nv_bfloat16"
            names = (f"rwkv::rwkv7_wkv_forward_kernel<{_checkpoint_length}, {scalar}>",
                     f"rwkv::rwkv7_wkv_backward_kernel<{_checkpoint_length}, {scalar}>")
            options = ["--std=c++17"]
            if dtype == torch.bfloat16:
                # cuda_bf16.h also needs the CCCL headers from a CUDA toolkit.
                import sysconfig
                runtime = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cuda_runtime/include"
                options.append(f"--include-path={runtime}")
                candidates = [runtime.parent.parent / "cuda_cccl/include"]
                for variable in ("CUDA_PATH", "CUDA_HOME"):
                    if os.environ.get(variable):
                        candidates.append(Path(os.environ[variable]) / "include")
                if os.name == "nt":
                    installed = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA GPU Computing Toolkit/CUDA"
                    candidates.extend(path / "include" for path in sorted(installed.glob("v*"), reverse=True))
                else:
                    candidates.append(Path("/usr/local/cuda/include"))
                cccl = next((path for path in candidates if (path / "nv/target").exists()), None)
                if cccl is None:
                    raise RuntimeError("BF16 NVRTC needs CCCL headers (nv/target); install a CUDA toolkit or nvidia-cuda-cccl-cu12")
                options.append(f"--include-path={cccl}")
            raw = _cupy.RawModule(code=kernel_source(dtype), options=tuple(options), name_expressions=names)
            _modules[key] = tuple(raw.get_function(name) for name in names)
    return _modules[key]


def validate(inputs, skip):
    r = inputs[0]
    if r.ndim != 4 or r.shape[-1] != 32 or not 0 < r.shape[1] < 384:
        raise ValueError("NVRTC recurrence requires head dimension 32 and 1–383 timesteps")
    if r.dtype not in (torch.float32, torch.bfloat16) or any(
            x.dtype != (torch.float32 if i == 3 else r.dtype) or x.device != r.device or x.shape != r.shape
            or not x.is_contiguous() for i, x in enumerate(inputs)):
        raise ValueError("NVRTC requires contiguous float32/BF16 inputs, float32 decay, and matching shapes")
    if not r.is_cuda or skip.dtype != torch.bool or skip.device != r.device or skip.shape != r.shape[:2]:
        raise ValueError("Invalid NVRTC device or skip mask")
    if not skip.is_contiguous():
        raise ValueError("NVRTC skip mask must be contiguous")
    return r.shape


def launch(kernel, device, shape, args):
    with _cupy.cuda.Device(device.index), _cupy.cuda.ExternalStream(torch.cuda.current_stream(device).cuda_stream):
        # DLPack shares storage and preserves the current Torch CUDA stream.
        # CuPy does not need a BF16 array type: uint16 exposes the same bits to
        # the kernel's BF16 pointer, with zero copying or numeric conversion.
        if _raw_pointer_launch:
            # The Torch allocator and autograd retain the tensor storage. Launch
            # on Torch's current stream; CUDA pointers are 64-bit ABI arguments.
            args = tuple(np.uint64(arg.data_ptr()) if isinstance(arg, torch.Tensor)
                         else np.int32(arg) for arg in args)
        else:
            args = tuple(_cupy.from_dlpack(arg.detach().view(torch.uint16) if arg.dtype == torch.bfloat16 else arg.detach()) if isinstance(arg, torch.Tensor)
                         else np.int32(arg) for arg in args)
        kernel((shape[0], shape[2]), (32, 32), args)


def forward(r, k, v, w, a, kd, skip):
    inputs = (r, k, v, w, a, kd)
    b, t, h, _ = shape = validate(inputs, skip)
    checkpoints = (t + _checkpoint_length - 1) // _checkpoint_length
    out = torch.empty_like(r)
    states = torch.empty((b, checkpoints, h, 32, 32), device=r.device, dtype=torch.float32)
    launch(module(r.device, r.dtype)[0], r.device, shape, (b, t, h, *inputs, skip, out, checkpoints, states))
    return out, states


def backward(r, k, v, w, a, kd, skip, states, grad):
    inputs = (r, k, v, w, a, kd)
    b, t, h, _ = shape = validate(inputs, skip)
    grad = grad.contiguous()
    if grad.shape != r.shape or grad.dtype != r.dtype or grad.device != r.device:
        raise ValueError("Invalid NVRTC output gradient")
    expected_states = (b, (t + _checkpoint_length - 1) // _checkpoint_length, h, 32, 32)
    if states.shape != expected_states or states.device != r.device or states.dtype != torch.float32:
        raise ValueError("Invalid NVRTC state checkpoints")
    gradients = tuple(torch.empty_like(x) for x in inputs)
    launch(module(r.device, r.dtype)[1], r.device, shape,
           (b, t, h, *inputs, skip, grad, states.shape[1], states, *gradients))
    return gradients


def register():
    """Install operators used by the existing RWKV7_WKV autograd class."""
    global _library, _cupy
    if _library is not None:
        return
    from rwkv.model import RWKV_CUDA
    if RWKV_CUDA is not None:
        raise RuntimeError("RWKV_CUDA is installed; select WKV_BACKEND='cuda' instead of 'nvrtc'")
    if os.name == "nt":
        directory = str(Path(torch.__file__).parent / "lib")
        os.environ["PATH"] = directory + os.pathsep + os.environ["PATH"]
        _dll_handles.append(os.add_dll_directory(directory))
    try:
        import cupy
    except ImportError as exc:
        raise RuntimeError("Install the NVRTC extra: uv sync --extra nvrtc") from exc
    _cupy = cupy
    library = torch.library.Library("rwkv", "FRAGMENT")
    arguments = "Tensor r, Tensor k, Tensor v, Tensor w, Tensor a, Tensor kd, Tensor skip"
    for suffix in ("float", "bfloat16"):
        library.define(f"rwkv7_wkv_forward_{suffix}({arguments}) -> (Tensor, Tensor)")
        library.define(f"rwkv7_wkv_backward_{suffix}({arguments}, Tensor states, Tensor grad) -> "
                       "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)")
        library.impl(f"rwkv7_wkv_forward_{suffix}", forward, "CUDA")
        library.impl(f"rwkv7_wkv_backward_{suffix}", backward, "CUDA")
    _library = library
