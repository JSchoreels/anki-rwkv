"""Opt-in F09 row-tiled recurrence. Never changes the default backend on disk."""

from contextlib import contextmanager
import os
from pathlib import Path
import sysconfig

import torch
from rwkv.model import nvrtc

ROOT = Path(__file__).resolve().parents[1]
_modules = {}


def source_options(dtype):
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Row-tiled recurrence supports float32 or BF16")
    prefix = nvrtc.kernel_source(dtype).split("namespace rwkv {")[0]
    source = prefix + (ROOT / "scripts/kernels/rwkv_row_tiles.cu").read_text()
    options = ["--std=c++17"]
    if dtype == torch.bfloat16:
        runtime = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cuda_runtime/include"
        options.append(f"--include-path={runtime}")
        candidates = [runtime.parent.parent / "cuda_cccl/include"]
        candidates.extend(Path(os.environ[name]) / "include" for name in ("CUDA_PATH", "CUDA_HOME") if os.environ.get(name))
        installed = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA GPU Computing Toolkit/CUDA"
        candidates.extend(path / "include" for path in sorted(installed.glob("v*"), reverse=True))
        candidates.append(Path("/usr/local/cuda/include"))
        cccl = next((path for path in candidates if (path / "nv/target").exists()), None)
        if cccl is None:
            raise RuntimeError("BF16 NVRTC requires CCCL headers (nv/target)")
        options.append(f"--include-path={cccl}")
    return source, tuple(options)


def compile_module(device, dtype, rows, tile):
    if rows not in (1, 2, 4) or tile not in (8, 16, 32):
        raise ValueError("Rows must be 1, 2 or 4; checkpoint tile must be 8, 16 or 32")
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = index, dtype, rows, tile
    if key not in _modules:
        source, options = source_options(dtype)
        scalar = "float" if dtype == torch.float32 else "__nv_bfloat16"
        names = tuple(f"rwkv_future::{name}_kernel<{tile}, {rows}, {scalar}>" for name in ("forward", "backward"))
        with nvrtc._cupy.cuda.Device(index):
            raw = nvrtc._cupy.RawModule(code=source, options=options, name_expressions=names)
            _modules[key] = tuple(raw.get_function(name) for name in names)
    return _modules[key]


@contextmanager
def row_tiled_backend(rows=4, tile=8):
    """Hold this context through every associated forward AND backward call."""
    if rows not in (1, 2, 4) or tile not in (8, 16, 32):
        raise ValueError("Invalid row/tile configuration")
    nvrtc.register()
    original_module, original_launch = nvrtc.module, nvrtc.launch
    original_tile = nvrtc._checkpoint_length

    def module(device, dtype):
        return compile_module(device, dtype, rows, tile)

    def launch(kernel, device, shape, args):
        import numpy as np
        # Torch owns and retains storage; enqueue on its current stream.
        arguments = tuple(np.uint64(arg.data_ptr()) if isinstance(arg, torch.Tensor) else np.int32(arg) for arg in args)
        with nvrtc._cupy.cuda.Device(device.index), nvrtc._cupy.cuda.ExternalStream(torch.cuda.current_stream(device).cuda_stream):
            kernel((shape[0], shape[2]), (32, 32 // rows), arguments)

    nvrtc.set_checkpoint_length(tile)
    nvrtc.module, nvrtc.launch = module, launch
    try:
        yield {"rows_per_thread": rows, "checkpoint_tile": tile, "threads_per_block": 1024 // rows}
    finally:
        try:
            torch.cuda.synchronize()
        finally:
            # CUDA errors must not leave the experimental backend installed.
            nvrtc.module, nvrtc.launch = original_module, original_launch
            nvrtc.set_checkpoint_length(original_tile)
