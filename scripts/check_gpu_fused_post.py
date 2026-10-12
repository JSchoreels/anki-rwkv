"""F08 analytic backward, isolated integration and offline compiler checks.

Default execution stays on CPU. --cuda is a small functional test and refuses
while the approved first batch is active. Neither mode validates predictive
quality or complete training speed.
"""

from argparse import ArgumentParser
from dataclasses import replace
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance


def inputs_for(shape, dtype, device="cpu"):
    import torch
    b, t, h, width = shape
    values = [torch.randn(shape, dtype=dtype, device=device) * scale for scale in (1, .2, .2, .2)]
    values += [torch.randn(b, t, h * width, dtype=dtype, device=device),
               torch.randn(1, 1, h, width, dtype=dtype, device=device) * .1,
               1 + torch.randn(h * width, dtype=dtype, device=device) * .1,
               torch.randn(h * width, dtype=dtype, device=device) * .1]
    return tuple(value.requires_grad_() for value in values)


def cpu_math():
    import torch
    from scripts.gpu_fused_post import backward_math, eager_post, fused_post
    results = []
    torch.manual_seed(20251011)
    for dtype in (torch.float32, torch.bfloat16):
        atol, rtol = ((1e-6, 1e-5) if dtype == torch.float32 else (0., 0.))
        for shape in ((1, 1, 1, 32), (2, 3, 2, 32), (1, 5, 4, 32)):
            inputs = inputs_for(shape, dtype)
            expected = eager_post(*inputs, 64e-5)
            actual = fused_post(*inputs, 64e-5)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            gradient = torch.randn_like(expected)
            expected_gradients = torch.autograd.grad(expected, inputs, gradient)
            with torch.no_grad():
                math_gradients = backward_math(inputs, gradient, 64e-5)
            errors = {}
            for name, left, right in zip(("x", "r", "k", "v", "gate", "bonus", "gamma", "beta"), expected_gradients, math_gradients):
                torch.testing.assert_close(right, left, atol=atol, rtol=rtol)
                assert torch.isfinite(right).all()
                errors[name] = float((right.float() - left.float()).abs().max())
            results.append({"shape": list(shape), "dtype": str(dtype), "gradient_max_abs": errors,
                            "atol": atol, "rtol": rtol, "forward": "exact CPU eager fallback",
                            "backward": "CPU coefficient/chain-rule formulas use native CPU saved statistics; compared with original eager autograd. CUDA BF16 moments and dY*X rounding need GPU checks."})
    return results


def integration(device="cpu", dtype=None):
    import torch
    from scripts.check_gpu_shape_buckets import fixture
    from scripts.gpu_future_models import capacity_variant, python_model_class
    from scripts.gpu_fused_post import install_fused_post
    results = []
    dtype = torch.float32 if dtype is None else dtype
    atol, rtol = ((4e-6, 4e-5) if dtype == torch.float32 else (.015, .08))
    for name in ("baseline", "expertium"):
        package = "rwkv" if name == "baseline" else name
        architecture = importlib.import_module(package + ".architecture").DEFAULT_ANKI_RWKV_CONFIG
        small = capacity_variant(architecture, width_heads=1, remove_layers=20)
        small = replace(small, modules=[(stream, replace(config, n_layers=2, total_layers=2)) for stream, config in small.modules])
        torch.manual_seed(20251011)
        expected = python_model_class(name)(small).float().to(device).eval()
        with torch.no_grad():
            for param in expected.parameters():
                param.add_(torch.randn_like(param) * .01)
        actual = python_model_class(name)(small).float().to(device).eval()
        actual.load_state_dict(expected.state_dict())
        selected = install_fused_post(actual)
        assert len(selected["modules"]) == 10  # Both layers of all five streams.
        assert list(actual.state_dict()) == list(expected.state_dict())
        expected.selective_cast(dtype)
        actual.selective_cast(dtype)
        source = fixture().to(device)
        source.start = source.start.to(dtype)
        left, right = expected.get_loss(source), actual.get_loss(source)
        errors = {}
        for field in ("average_loss", "p_curve", "p_imm", "p_imm_all"):
            torch.testing.assert_close(getattr(right, field), getattr(left, field), atol=atol, rtol=rtol)
            errors[field] = float((getattr(right, field) - getattr(left, field)).detach().abs().max())
        left.average_loss.backward()
        right.average_loss.backward()
        maximum = 0.
        for (left_name, left_param), (right_name, right_param) in zip(expected.named_parameters(), actual.named_parameters()):
            assert left_name == right_name and (left_param.grad is None) == (right_param.grad is None)
            if left_param.grad is not None:
                torch.testing.assert_close(right_param.grad, left_param.grad, atol=atol, rtol=rtol)
                assert torch.isfinite(right_param.grad).all()
                maximum = max(maximum, float((right_param.grad - left_param.grad).abs().max()))
        results.append({"model": name, "device": str(device), "dtype": str(dtype), "prediction_max_abs": errors,
                        "all_parameter_gradient_max_abs": maximum, "installed": selected,
                        "atol": atol, "rtol": rtol,
                        "architecture": "two layers in all five streams, width32, dropout disabled"})
    return results


def offline_compile(folder):
    import torch
    from scripts.gpu_fused_post import SOURCE, compiler_options
    # Loading compiler DLLs does not initialize a CUDA context.
    if os.name == "nt":
        directory = str(Path(torch.__file__).parent / "lib")
        os.environ["PATH"] = directory + os.pathsep + os.environ["PATH"]
        dll_handle = os.add_dll_directory(directory)
    os.environ["CUPY_CACHE_DIR"] = str(ROOT / "artifacts/cache/cupy-cache")
    from cupy.cuda.compiler import _NVRTCProgram
    result = []
    for dtype in (torch.float32, torch.bfloat16):
        name = str(dtype).split(".")[-1]
        log = io.StringIO()
        program = _NVRTCProgram(SOURCE.read_text(), "fused-post-" + name + ".cu")
        ptx, _ = program.compile((*compiler_options(dtype), "--gpu-architecture=compute_120"), log_stream=log)
        path = folder / (name + ".ptx")
        path.write_bytes(ptx)
        row = {"dtype": str(dtype), "target": "sm_120", "compiler_log": log.getvalue(),
               "ptx": path.relative_to(ROOT).as_posix(), "nvrtc": "compiled"}
        assembler = shutil.which("ptxas")
        if assembler:
            completed = subprocess.run([assembler, "--gpu-name=sm_120", "--verbose", str(path),
                                        "-o", str(folder / (name + ".cubin"))], capture_output=True, text=True)
            row.update(ptxas_exit_code=completed.returncode, ptxas_log=completed.stdout + completed.stderr)
            if completed.returncode:
                raise RuntimeError(row["ptxas_log"])
        result.append(row)
    return result


def cuda_checks():
    import torch
    from scripts.gpu_fused_post import eager_post, fused_post
    from rwkv.model import nvrtc
    nvrtc.register()
    nvrtc.set_raw_pointer_launch(True)
    torch.cuda.set_per_process_memory_fraction(.72, 0)
    torch.manual_seed(20251011)
    results = []
    for dtype in (torch.float32, torch.bfloat16):
        atol, rtol = ((8e-6, 8e-5) if dtype == torch.float32 else (.025, .075))
        for shape in ((1, 1, 1, 32), (2, 3, 2, 32), (1, 5, 4, 32)):
            inputs = inputs_for(shape, dtype, "cuda:0")
            expected = eager_post(*inputs, 64e-5)
            actual = fused_post(*inputs, 64e-5)
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            gradient = torch.randn_like(expected)
            left = torch.autograd.grad(expected, inputs, gradient)
            right = torch.autograd.grad(actual, inputs, gradient)
            errors = {}
            for name, a, b in zip(("x", "r", "k", "v", "gate", "bonus", "gamma", "beta"), left, right):
                torch.testing.assert_close(b, a, atol=atol, rtol=rtol)
                assert torch.isfinite(b).all()
                errors[name] = float((b.float() - a.float()).abs().max())
            results.append({"dtype": str(dtype), "shape": list(shape), "output_max_abs": float((actual.float() - expected.float()).abs().max()),
                            "gradient_max_abs": errors, "atol": atol, "rtol": rtol})
    model_results = [row for dtype in (torch.float32, torch.bfloat16) for row in integration(torch.device("cuda:0"), dtype)]
    nvrtc.set_raw_pointer_launch(False)
    return {"kernel_parity": results, "model_integration": model_results,
            "timing": "not measured; these are functional checks only"}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory under artifacts")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--offline-compile", action="store_true")
    args = parser.parse_args()
    folder = args.output.resolve()
    if not folder.is_relative_to(ROOT / "artifacts") or folder.exists():
        parser.error("Preserve existing evidence; use a new artifact directory")
    folder.mkdir(parents=True)
    report = {"status": "running", "provenance": process_provenance(),
              "scope": "synthetic CPU math/integration and optional offline compiler/CUDA functional checks; no held-out quality or training-speed claim"}
    try:
        if args.cuda:
            import psutil
            state, lock = ROOT / "artifacts/benchmarks/gpu-batch1-500/process.json", ROOT / "artifacts/gpu-batch1.lock"
            if ((state.exists() and json.loads(state.read_text()).get("status") == "running")
                    or (lock.exists() and psutil.pid_exists(json.loads(lock.read_text())["pid"]))):
                raise RuntimeError("Verify the first-batch worker has exited before running CUDA checks")
        import torch
        torch.set_num_threads(1)
        assert not torch.cuda.is_initialized()
        report["cpu_math"] = cpu_math()
        report["cpu_integration"] = integration()
        if args.offline_compile:
            report["offline_compile"] = offline_compile(folder)
        assert not torch.cuda.is_initialized()
        report["cpu_cuda_context_initialized"] = False
        if args.cuda:
            report["cuda"] = cuda_checks()
        report["status"] = "passed"
        print("F08 post-recurrence checks passed: " + ", ".join(key for key in report if key.startswith(("cpu", "offline", "cuda"))))
    except Exception as exc:
        report.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        (folder / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
