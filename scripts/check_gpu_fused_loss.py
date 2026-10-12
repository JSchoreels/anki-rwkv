"""CPU backward/integration and offline compilation for F08 masked reductions.

CUDA checks are opt-in, synthetic and blocked while the approved batch runs.
No complete-training speed or held-out quality is inferred from these checks.
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


def reduction_checks(device="cpu"):
    import torch
    from scripts.gpu_fused_loss import eager_means, masked_means
    torch.manual_seed(20251011)
    rows = []
    for shape in ((1, 1), (2, 129), (3, 257)):
        for mode in ("mixed", "no_curve", "no_instant", "no_equalize", "no_labels"):
            # Transposed inputs exercise the contiguous copy and gradient shape.
            losses = tuple(torch.randn(tuple(reversed(shape)), device=device).T.requires_grad_() for _ in range(7))
            label = torch.rand(shape, device=device) > .2
            curve = torch.rand(shape, device=device) > .5
            equal = torch.rand(shape, device=device) > .4
            if mode == "no_curve":
                curve.zero_()
            elif mode == "no_instant":
                curve.fill_(True)
            elif mode == "no_equalize":
                equal.zero_()
            elif mode == "no_labels":
                label.zero_()
            masks = (label & curve, label & ~curve, label & curve & equal, label & ~curve & equal)
            expected = eager_means(losses, masks)
            actual = masked_means(*losses, *masks)
            atol, rtol = ((0., 0.) if str(device) == "cpu" else (3e-6, 3e-5))
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            weights = torch.randn(9, device=device)
            left = torch.autograd.grad(expected, losses, weights)
            right = torch.autograd.grad(actual, losses, weights)
            errors = []
            for a, b in zip(left, right):
                torch.testing.assert_close(b, a, atol=1e-8, rtol=1e-6)
                assert torch.isfinite(b).all()
                errors.append(float((b - a).abs().max()))
            assert torch.isfinite(actual).all()
            if mode == "no_labels":
                assert actual.count_nonzero() == 0 and all(value.count_nonzero() == 0 for value in right)
            rows.append({"shape": list(shape), "mode": mode, "forward_max_abs": float((actual - expected).detach().abs().max()),
                         "seven_gradient_max_abs": errors, "forward_atol": atol, "forward_rtol": rtol,
                         "gradient_atol": 1e-8, "gradient_rtol": 1e-6,
                         "input_layout": "transposed; noncontiguous when both dimensions exceed one"})
    return rows


def integration(device="cpu", dtype=None):
    import torch
    from scripts.check_gpu_shape_buckets import fixture
    from scripts.gpu_future_models import capacity_variant, python_model_class
    from scripts.gpu_fused_loss import install_fused_loss
    from scripts.gpu_fused_post import install_fused_post
    dtype = torch.float32 if dtype is None else dtype
    atol, rtol = ((5e-6, 5e-5) if dtype == torch.float32 else (.015, .08))
    rows = []
    for name in ("baseline", "expertium"):
        package = "rwkv" if name == "baseline" else name
        architecture = importlib.import_module(package + ".architecture").DEFAULT_ANKI_RWKV_CONFIG
        architecture = capacity_variant(architecture, width_heads=1, remove_layers=20)
        architecture = replace(architecture, modules=[(stream, replace(config, n_layers=2, total_layers=2))
                                                       for stream, config in architecture.modules])
        torch.manual_seed(20251011)
        control = python_model_class(name)(architecture).float().to(device).eval()
        with torch.no_grad():
            for param in control.parameters():
                param.add_(torch.randn_like(param) * .01)
        state = {key: value.clone() for key, value in control.state_dict().items()}
        control.selective_cast(dtype)
        batch = fixture().to(device)
        batch.start = batch.start.to(dtype)
        # Exercise both equalized tasks and absent labels in the full model.
        batch.labels[..., 5] = (torch.arange(batch.labels.numel() // 7, device=device).reshape(batch.labels.shape[:2]) % 3 == 0)
        batch.labels[-1, -1, 4] = 0
        expected = control.get_loss(batch)
        expected.average_loss.backward()
        expected_gradients = {key: param.grad.clone() for key, param in control.named_parameters() if param.grad is not None}
        for combined in (False, True):
            candidate = python_model_class(name)(architecture).float().to(device).eval()
            candidate.load_state_dict(state)
            installed = {"loss": install_fused_loss(candidate)}
            if combined:
                installed["post"] = install_fused_post(candidate)
            assert list(candidate.state_dict()) == list(control.state_dict())
            candidate.selective_cast(dtype)
            actual = candidate.get_loss(batch)
            errors = {}
            # Compare every returned field, not just the composite objective.
            for field in expected._fields:
                left, right = getattr(expected, field), getattr(actual, field)
                if isinstance(left, torch.Tensor):
                    assert left.dtype == right.dtype and left.shape == right.shape
                    if left.is_floating_point():
                        torch.testing.assert_close(right, left, atol=atol, rtol=rtol)
                        errors[field] = float((right - left).detach().abs().max())
                    else:
                        assert torch.equal(right, left)
                else:
                    assert right == left
            actual.average_loss.backward()
            maximum = 0.
            for key, param in candidate.named_parameters():
                assert (param.grad is not None) == (key in expected_gradients)
                if key in expected_gradients:
                    torch.testing.assert_close(param.grad, expected_gradients[key], atol=atol, rtol=rtol)
                    assert torch.isfinite(param.grad).all()
                    maximum = max(maximum, float((param.grad - expected_gradients[key]).abs().max()))
            rows.append({"model": name, "device": str(device), "dtype": str(dtype), "combined_post_fusion": combined,
                         "all_statistic_fields_checked": list(expected._fields), "floating_field_max_abs": errors,
                         "all_parameter_gradient_max_abs": maximum, "atol": atol, "rtol": rtol,
                         "installed": installed, "architecture": "width32, two layers in each of five streams, dropout disabled"})
    return rows


def offline_compile(folder):
    import torch
    from scripts.gpu_fused_loss import SOURCE
    if os.name == "nt":
        directory = str(Path(torch.__file__).parent / "lib")
        os.environ["PATH"] = directory + os.pathsep + os.environ["PATH"]
        dll_handle = os.add_dll_directory(directory)
    os.environ["CUPY_CACHE_DIR"] = str(ROOT / "artifacts/cache/cupy-cache")
    from cupy.cuda.compiler import _NVRTCProgram
    log = io.StringIO()
    program = _NVRTCProgram(SOURCE.read_text(), "fused-masked-means.cu")
    ptx, _ = program.compile(("--std=c++17", "--fmad=false", "--gpu-architecture=compute_120"), log_stream=log)
    path = folder / "masked-means.ptx"
    path.write_bytes(ptx)
    row = {"target": "sm_120", "dtype": "FP32", "nvrtc": "compiled", "compiler_log": log.getvalue(),
           "ptx": path.relative_to(ROOT).as_posix()}
    assembler = shutil.which("ptxas")
    if assembler:
        completed = subprocess.run([assembler, "--gpu-name=sm_120", "--verbose", str(path),
                                    "-o", str(folder / "masked-means.cubin")], capture_output=True, text=True)
        row.update(ptxas_exit_code=completed.returncode, ptxas_log=completed.stdout + completed.stderr)
        if completed.returncode:
            raise RuntimeError(row["ptxas_log"])
    return row


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New artifact directory")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--offline-compile", action="store_true")
    args = parser.parse_args()
    folder = args.output.resolve()
    if not folder.is_relative_to(ROOT / "artifacts") or folder.exists():
        parser.error("Preserve previous evidence; use a new directory below artifacts")
    folder.mkdir(parents=True)
    result = {"status": "running", "provenance": process_provenance(),
              "scope": "synthetic reduction/backward/full-model integration and optional compiler/CUDA checks; no complete-training or predictive validation"}
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
        result["runtime"] = {"torch": torch.__version__, "cuda_build": torch.version.cuda}
        result["cpu_reductions"] = reduction_checks()
        result["cpu_integration"] = integration()
        if args.offline_compile:
            result["offline_compile"] = offline_compile(folder)
        assert not torch.cuda.is_initialized()
        result["cpu_cuda_context_initialized"] = False
        if args.cuda:
            from rwkv.model import nvrtc
            nvrtc.register()
            nvrtc.set_raw_pointer_launch(True)
            torch.cuda.set_per_process_memory_fraction(.72, 0)
            result["cuda_reductions"] = reduction_checks("cuda:0")
            result["cuda_integration"] = [row for dtype in (torch.float32, torch.bfloat16) for row in integration("cuda:0", dtype)]
            nvrtc.set_raw_pointer_launch(False)
        result["status"] = "passed"
        print("F08 masked reduction checks passed: " + ", ".join(key for key in result if key.startswith(("cpu", "offline", "cuda"))))
    except Exception as exc:
        result.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        (folder / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
