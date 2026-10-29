"""Opt-in F09 row-tiled recurrence. Never changes the default backend on disk."""

from contextlib import contextmanager
import os
from pathlib import Path
import sysconfig

import torch
from rwkv.model import nvrtc

ROOT = Path(__file__).resolve().parents[1]
_modules = {}


# Wide-kernels port: head size 16 (Expertium w64/w80) by text rewrite, like nvrtc.head_source; K=32 text unchanged.
HEAD16_REWRITES = (("T, H, 32)", "T, H, 16)", 23), ("L, H, 32, 32)", "L, H, 16, 16)", 3), ("[32 * 33]", "[16 * 17]", 4),
                   (" * 33 + ", " * 17 + ", 10), ("Kd[32], A[32]", "Kd[16], A[16]", 1),
                   ("const int WARPS = 32 / ROWS;", "const int WARPS = 16 / ROWS;", 2), ("j < 32;", "j < 16;", 1),
                   ("for (int offset = 16; offset > 0; offset /= 2)", "for (int offset = 8; offset > 0; offset /= 2)", 1),
                   ("__shfl_down_sync(FULL_MASK, value, offset)", "__shfl_down_sync(FULL_MASK, value, offset, 16)", 1),
                   ("__shfl_sync(FULL_MASK, dot, 0)", "__shfl_sync(FULL_MASK, dot, 0, 16)", 2),
                   ("__shfl_sync(FULL_MASK, warp_sum(remove[q]), 0)",
                    "__shfl_sync(FULL_MASK, warp_sum(remove[q]), 0, 16)", 1))


def head16_source(text):
    """Rewrite the exact row-tiled kernels (F31) from head size 32 to 16: index strides, shared tiles, warp
    reductions of width 16 (two state rows per 32-lane warp), as the production NVRTC K=16 rewrite does."""
    for old, new, count in HEAD16_REWRITES:
        if text.count(old) != count:
            raise RuntimeError(f"K=16 row-tile rewrite expected {count}x {old!r}, found {text.count(old)}")
        text = text.replace(old, new)
    return text


# SM21 (docs/SMALL_MODEL.md, carried state): the exact forward started from an initial state S0 [B, H, K, K] (float32)
# and writing the final state, as rwkv.model.nvrtc.STATE_REWRITES does for the generic kernel. Applied after the K=16
# rewrite ({K} = head size) and only when `state` is requested, so the stock text and its modules are unchanged.
STATE_REWRITES = (("    F* out, int L, float* checkpoints) {",
                   "    F* out, int L, float* checkpoints, const float* state0, float* state_final) {", 1),
                  ("    for (int q = 0; q < ROWS; ++q) state[q] = 0.0f;\n",
                   "    for (int q = 0; q < ROWS; ++q)\n"
                   "        state[q] = state0[get_index4(b, 0, h, threadIdx.y + q * WARPS, y, 1, H, {K}, {K})];\n", 1),
                  ("            if (skipped) state[q] = previous;\n        }\n    }\n}",
                   "            if (skipped) state[q] = previous;\n        }\n    }\n    #pragma unroll\n"
                   "    for (int q = 0; q < ROWS; ++q)\n"
                   "        state_final[get_index4(b, 0, h, threadIdx.y + q * WARPS, y, 1, H, {K}, {K})] = state[q];\n}", 1))


# SM77 (learned initial state): the exact backward also writes dL/dS0 [B, H, K, K] (float32). After the loop the
# registers carry[q] hold the gradient every step passed to the state before it, i.e. dL/dS0 of rows x = threadIdx.y
# + q * WARPS; the rewrite adds an output pointer and one store per row. Applied after STATE_REWRITES, only when
# `state_grad` is requested.
STATE_GRAD_REWRITES = (("    float* dw, F* da, F* dkd) {", "    float* dw, F* da, F* dkd, float* state0_grad) {", 1),
                       ("-__fmul_rn(kdy, remove[q])));\n            }\n            __syncthreads();\n        }\n    }\n}",
                        "-__fmul_rn(kdy, remove[q])));\n            }\n            __syncthreads();\n        }\n    }\n"
                        "    #pragma unroll\n    for (int q = 0; q < ROWS; ++q)\n"
                        "        state0_grad[get_index4(b, 0, h, threadIdx.y + q * WARPS, y, 1, H, {K}, {K})] = carry[q];\n}", 1))


def state_grad_source(text, head=32):
    for old, new, count in STATE_GRAD_REWRITES:
        if text.count(old) != count:
            raise RuntimeError(f"row-tile state-gradient rewrite expected {count}x {old!r}, found {text.count(old)}")
        text = text.replace(old, new.replace("{K}", str(head)))
    return text


def state_source(text, head=32):
    for old, new, count in STATE_REWRITES:
        if text.count(old) != count:
            raise RuntimeError(f"row-tile state rewrite expected {count}x {old!r}, found {text.count(old)}")
        text = text.replace(old, new.replace("{K}", str(head)))
    return text


def source_options(dtype, exact=False, compact_previous=False, next_step_prefetch=False, load_pipeline=0, head=32,
                   state=False, state_grad=False):
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Row-tiled recurrence supports float32 or BF16")
    prefix = nvrtc.kernel_source(dtype).split("namespace rwkv {")[0]
    # F31 pins every float operation to the production instruction sequence.
    kernel = "rwkv_row_tiles_exact.cu" if exact else "rwkv_row_tiles.cu"
    text = (ROOT / "scripts/kernels" / kernel).read_text()
    if head == 16:
        if not exact:
            raise ValueError("Head size 16 row tiles exist only for the exact kernel")
        text = head16_source(text)
    elif head != 32:
        raise ValueError("Row-tiled recurrence supports head sizes 16 and 32")
    if state:
        if not exact:
            raise ValueError("SM21 initial/final state exists only for the exact row-tiled kernel")
        text = state_source(text, head)
    if state_grad:
        if not state:
            raise ValueError("SM77 state gradient needs the state forward")
        text = state_grad_source(text, head)
    source = prefix + text
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
    if load_pipeline not in (0, 1, 2, 3, 4, 5, 6):
        raise ValueError("Invalid recurrence load pipeline")
    if load_pipeline:
        if not exact:
            raise ValueError("Load pipelines require exact arithmetic")
        options.extend(((), ("-DRECOMPUTE_PREFETCH=1",), ("-DROW_GRAD_PREFETCH=1",), ("-DFORWARD_PREFETCH=1",),
                        ("-DFORWARD_V_PREFETCH=1",), ("-DFORWARD_PREFETCH=1", "-DFORWARD_V_PREFETCH=1"),
                        ("-DRECOMPUTE_V_PREFETCH=1",))[load_pipeline])
    return source, tuple(options)


def compile_module(device, dtype, rows, tile, exact=False, compact_previous=False, next_step_prefetch=False, load_pipeline=0,
                   head=32, state=False, state_grad=False):
    if rows not in (1, 2, 4, 8) or tile not in (8, 16, 32):
        raise ValueError("Rows must be 1, 2, 4 or 8; checkpoint tile must be 8, 16 or 32")
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = index, dtype, rows, tile, exact, compact_previous, next_step_prefetch, load_pipeline
    if head != 32:
        key += (head,)
    if state:
        key += ("state",)
    if state_grad:
        key += ("state-grad",)
    if key not in _modules:
        source, options = source_options(dtype, exact, compact_previous, next_step_prefetch, load_pipeline, head, state,
                                         state_grad)
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
    original_state_module = nvrtc.state_module
    original_state_grad_module = nvrtc.state_grad_module

    def module(device, dtype, head=32):
        if wave_aware or load_pipeline in (7, 8):
            if head != 32:
                raise ValueError("Wave-aware and adaptive pipelines exist only for head size 32")
            # Forward remains at the accepted layout; launch chooses backward by B.
            return tuple((device, dtype, i) for i in (0, 1))
        return compile_module(device, dtype, rows, tile, exact, compact_previous, next_step_prefetch, load_pipeline, head)

    def state_module(device, dtype, head=32):
        # SM21: the forward with initial/final state in this backend's layout; the backward stays module()[1].
        if wave_aware or load_pipeline in (7, 8):
            raise NotImplementedError("SM21 state carry supports the fixed row-tile layouts only")
        return compile_module(device, dtype, rows, tile, exact, compact_previous, next_step_prefetch, load_pipeline,
                              head, state=True)[0]

    def state_grad_module(device, dtype, head=32):
        # SM77: (forward with state, backward writing dS0) in this backend's layout
        if wave_aware or load_pipeline in (7, 8):
            raise NotImplementedError("SM77 state gradient supports the fixed row-tile layouts only")
        return compile_module(device, dtype, rows, tile, exact, compact_previous, next_step_prefetch, load_pipeline,
                              head, state=True, state_grad=True)

    def launch(kernel, device, shape, args, head=32):
        import numpy as np
        launch_rows = rows
        if load_pipeline in (7, 8):
            # Long histories amortize the pipeline live set; short histories have
            # many independent rows and little sequential latency to hide.
            index, dtype = kernel[2], kernel[1]
            selected = shape[1] >= 32 if load_pipeline == 7 else shape[1] <= 16
            pipeline = 5 if selected else 0
            kernel = compile_module(device, dtype, rows, tile, exact, compact_previous, next_step_prefetch, pipeline)[index]
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
        nvrtc.raw_launch(kernel, device, (shape[0], shape[2]), (head, head // launch_rows), arguments)

    nvrtc.set_checkpoint_length(tile)
    nvrtc.module, nvrtc.launch = module, launch
    nvrtc.state_module = state_module
    nvrtc.state_grad_module = state_grad_module
    try:
        yield {"rows_per_thread": rows, "checkpoint_tile": tile, "threads_per_block": 1024 // rows,
               "compact_previous_state": compact_previous, "recurrence_prefetch": next_step_prefetch, "wave_aware_backward": wave_aware, "recurrence_load_pipeline": load_pipeline}
    finally:
        try:
            torch.cuda.synchronize()
        finally:
            # CUDA errors must not leave the experimental backend installed.
            nvrtc.module, nvrtc.launch = original_module, original_launch
            nvrtc.state_module = original_state_module
            nvrtc.state_grad_module = original_state_grad_module
            nvrtc.set_checkpoint_length(original_tile)
