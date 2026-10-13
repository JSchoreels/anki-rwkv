"""Opt-in F09 row-tiled recurrence. Never changes the default backend on disk."""

from contextlib import contextmanager
import os
from pathlib import Path
import sysconfig

import torch
from rwkv.model import nvrtc

ROOT = Path(__file__).resolve().parents[1]
_modules = {}


def source_options(dtype, exact=False, compact_previous=False, next_step_prefetch=False, load_pipeline=0):
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Row-tiled recurrence supports float32 or BF16")
    prefix = nvrtc.kernel_source(dtype).split("namespace rwkv {")[0]
    # F31 pins every float operation to the production instruction sequence.
    kernel = "rwkv_row_tiles_exact.cu" if exact else "rwkv_row_tiles.cu"
    source = prefix + (ROOT / "scripts/kernels" / kernel).read_text()
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
    if compact_previous:
        if not exact:
            raise ValueError("F54 compact previous state requires the exact kernel")
        options.append("-DCOMPACT_PREVIOUS_STATE=1")
    if next_step_prefetch:
        options.append("-DNEXT_STEP_PREFETCH=1")
    if load_pipeline not in (0, 1, 2, 3):
        raise ValueError("Invalid recurrence load pipeline")
    if load_pipeline:
        if not exact:
            raise ValueError("Load pipelines require exact arithmetic")
        options.append(("", "-DRECOMPUTE_PREFETCH=1", "-DROW_GRAD_PREFETCH=1", "-DFORWARD_PREFETCH=1")[load_pipeline])
    return source, tuple(options)


def compile_module(device, dtype, rows, tile, exact=False, compact_previous=False, next_step_prefetch=False, load_pipeline=0):
    if rows not in (1, 2, 4, 8) or tile not in (8, 16, 32):
        raise ValueError("Rows must be 1, 2, 4 or 8; checkpoint tile must be 8, 16 or 32")
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = index, dtype, rows, tile, exact, compact_previous, next_step_prefetch, load_pipeline
    if key not in _modules:
        source, options = source_options(dtype, exact, compact_previous, next_step_prefetch, load_pipeline)
        scalar = "float" if dtype == torch.float32 else "__nv_bfloat16"
        namespace = "rwkv_exact" if exact else "rwkv_future"
        names = tuple(f"{namespace}::{name}_kernel<{tile}, {rows}, {scalar}>" for name in ("forward", "backward"))
        with nvrtc._cupy.cuda.Device(index):
            raw = nvrtc._cupy.RawModule(code=source, options=options, name_expressions=names)
            _modules[key] = tuple(raw.get_function(name) for name in names)
    return _modules[key]


@contextmanager
def row_tiled_backend(rows=4, tile=8, exact=False, compact_previous=False, next_step_prefetch=False, wave_aware=False, load_pipeline=0):
    """Hold this context through every associated forward AND backward call."""
    if rows not in (1, 2, 4, 8) or tile not in (8, 16, 32):
        raise ValueError("Invalid row/tile configuration")
    nvrtc.register()
    original_module, original_launch = nvrtc.module, nvrtc.launch
    original_tile = nvrtc._checkpoint_length

    def module(device, dtype):
        if wave_aware:
            # Forward remains at the accepted layout; launch chooses backward by B.
            return tuple((device, dtype, i) for i in (0, 1))
        return compile_module(device, dtype, rows, tile, exact, compact_previous, next_step_prefetch, load_pipeline)

    def launch(kernel, device, shape, args):
        import numpy as np
        launch_rows = rows
        if wave_aware:
            index = kernel[2]
            dtype = kernel[1]
            # Keep forward at four rows; choose eight-row backward for measured long
            # partial-wave shapes and large short histories, retaining four elsewhere.
            selected = (504 < shape[0] <= 1008 and shape[1] >= 128) or (shape[0] >= 1000 and shape[1] <= 16)
            launch_rows = 8 if index == 1 and selected else rows
            kernel = compile_module(device, dtype, launch_rows, tile, exact, compact_previous, next_step_prefetch, load_pipeline)[index]
        # Torch owns and retains storage; enqueue on its current stream.
        arguments = tuple(np.uint64(arg.data_ptr()) if isinstance(arg, torch.Tensor) else np.int32(arg) for arg in args)
        nvrtc.raw_launch(kernel, device, (shape[0], shape[2]), (32, 32 // launch_rows), arguments)

    nvrtc.set_checkpoint_length(tile)
    nvrtc.module, nvrtc.launch = module, launch
    try:
        yield {"rows_per_thread": rows, "checkpoint_tile": tile, "threads_per_block": 1024 // rows,
               "compact_previous_state": compact_previous, "recurrence_prefetch": next_step_prefetch, "wave_aware_backward": wave_aware, "recurrence_load_pipeline": load_pipeline}
    finally:
        try:
            torch.cuda.synchronize()
        finally:
            # CUDA errors must not leave the experimental backend installed.
            nvrtc.module, nvrtc.launch = original_module, original_launch
            nvrtc.set_checkpoint_length(original_tile)
