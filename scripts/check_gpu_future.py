"""Focused CPU checks for F10/F11/F13 and optional isolated F09/F10 CUDA checks."""

from argparse import ArgumentParser
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance


def cpu_checks(output):
    import torch
    from scripts.gpu_future_models import TaskModel, capacity_variant, load_pretrained, python_model_class
    from scripts.gpu_training_targets import AccuracyStopper, METRICS, compare_budgets, learning_rate
    from scripts.gpu_fp8 import install_fp8, projection_counts, quantize
    from rwkv.model.srs_model import PreparedBatch
    results = {}
    length = 4
    labels = torch.tensor([[[86400, 0, 0, 1, 1, 0, 0], [86400, 0, 1, 3, 1, 0, 1],
                            [172800, 0, 1, 4, 1, 0, 0], [172800, 0, 1, 2, 1, 0, 1]]], dtype=torch.float32)
    batch = PreparedBatch(1, torch.randn(length, 92), [[torch.arange(length)] for _ in range(5)],
                          [[length] for _ in range(5)], [[torch.tensor([0, 0, 1, 2])] for _ in range(5)],
                          [[torch.tensor([False, True, False, True])] for _ in range(5)], labels,
                          torch.arange(length).view(1, -1))
    for name in ("baseline", "expertium"):
        package = "rwkv" if name == "baseline" else name
        import importlib
        architecture = importlib.import_module(package + ".architecture").DEFAULT_ANKI_RWKV_CONFIG
        original_class = importlib.import_module(package + ".model.srs_model").SrsRWKV
        variant = capacity_variant(architecture, width_heads=1, remove_layers=20)
        assert architecture.d_model == (128 if name == "baseline" else 32)
        assert all(config.n_layers >= 1 and config.d_model // config.n_heads == 32 for _, config in variant.modules)
        parity_architecture = (replace(variant, modules=[(stream, replace(config, n_layers=2, total_layers=2))
                                                         for stream, config in variant.modules]) if name == "baseline" else variant)
        original = original_class(parity_architecture).float().eval()
        with torch.no_grad():
            for param in original.parameters():
                param.add_(torch.randn_like(param) * 0.01)
        state = original.state_dict()
        plain = python_model_class(name)(parity_architecture).float().eval()
        plain.load_state_dict(state)
        expected, actual = original.get_loss(batch), plain.get_loss(batch)
        for field in ("average_loss", "p_curve", "p_imm", "p_imm_all"):
            torch.testing.assert_close(getattr(actual, field), getattr(expected, field), atol=3e-6, rtol=3e-5)
        expected.average_loss.backward()
        actual.average_loss.backward()
        max_error = 0.0
        for (left_name, left), (right_name, right) in zip(original.named_parameters(), plain.named_parameters()):
            assert left_name == right_name
            if left.grad is not None:
                torch.testing.assert_close(right.grad, left.grad, atol=3e-6, rtol=3e-5)
                max_error = max(max_error, float((right.grad - left.grad).abs().max()))
        tasks, gradients, parameter_counts = {}, {}, {}
        for task in ("joint", "curve", "instant"):
            base = python_model_class(name)(parity_architecture).float()
            base.load_state_dict(state)
            specialist = TaskModel(base, task).eval()
            stats = specialist.get_loss(batch)
            assert torch.isfinite(stats.average_loss)
            if task != "instant":
                torch.testing.assert_close(stats.p_curve, expected.p_curve, atol=3e-6, rtol=3e-5)
            else:
                assert stats.p_curve is None and stats.ahead_n == 0
            if task != "curve":
                torch.testing.assert_close(stats.p_imm, expected.p_imm, atol=3e-6, rtol=3e-5)
            else:
                assert stats.p_imm is None and stats.imm_n == 0
            if task == "joint":
                torch.testing.assert_close(stats.average_loss, expected.average_loss, atol=3e-6, rtol=3e-5)
            stats.average_loss.backward()
            gradients[task] = {key: p.grad.clone() for key, p in specialist.named_parameters() if p.grad is not None}
            parameter_counts[task] = sum(p.numel() for p in specialist.parameters())
            tasks[task] = float(stats.average_loss.detach())
            if task == "curve":
                assert not any(key.startswith(("head_p.", "p_linear.")) for key in specialist.state_dict())
            if task == "instant":
                assert not any(key.startswith(("head_w.", "w_linear.", "ahead_linear.", "head_ahead_logits.")) for key in specialist.state_dict())
        for key, joint in gradients["joint"].items():
            combined = gradients["curve"].get(key, torch.zeros_like(joint)) + gradients["instant"].get(key, torch.zeros_like(joint))
            torch.testing.assert_close(combined, joint, atol=4e-6, rtol=5e-5)
        assert parameter_counts["curve"] < parameter_counts["joint"] and parameter_counts["instant"] < parameter_counts["joint"]
        fp8_source = python_model_class(name)(parity_architecture).float().eval()
        fp8_source.load_state_dict(state)
        configuration = install_fp8(fp8_source)
        fp8_stats = fp8_source.get_loss(batch)
        torch.testing.assert_close(fp8_stats.average_loss, expected.average_loss, atol=3e-6, rtol=3e-5)
        assert all(row["fp8_calls"] == 0 and row["eager_calls"] > 0 for row in projection_counts(fp8_source).values())
        one_layer = python_model_class(name)(variant).float().eval()
        assert torch.isfinite(one_layer.get_loss(batch).average_loss)
        results[name] = {"python_vs_scripted_gradient_max_abs": max_error, "specialist_gradient_decomposition": "passed",
                         "tiny_fixture_parameter_counts": parameter_counts, "synthetic_objectives": tasks,
                         "fp8_cpu_fallback": "passed; CUDA FP8 execution remains pending", "fp8_selection": configuration,
                         "architecture_variant": asdict(variant), "one_layer_initialization_and_forward": "passed",
                         "parity_architecture": asdict(parity_architecture)}
    for value in (torch.zeros(32, 32), torch.randn(32, 32), torch.tensor([0, 1e-30, 1e20, -1e20])):
        quantized, scale = quantize(value)
        assert torch.isfinite(quantized.float()).all() and torch.isfinite(scale) and scale > 0
    unaligned = torch.randn(17, 32)
    before, original_scale = quantize(unaligned)
    after, padded_scale = quantize(torch.nn.functional.pad(unaligned, (0, 0, 0, 15)))
    assert torch.equal(before.float(), after[:17].float()) and torch.equal(original_scale, padded_scale)
    schedules = {}
    for schedule in ("constant", "cosine", "wsd"):
        values = [learning_rate(step, 100, schedule=schedule) for step in range(1, 101)]
        assert values[0] == 7e-4 / 5 and values[4] == 7e-4
        assert all(0 <= value <= 7e-4 for value in values)
        assert values[-1] == (7e-4 if schedule == "constant" else 0)
        schedules[schedule] = {"first": values[0], "peak": max(values), "last": values[-1]}
    stopper = AccuracyStopper({key: 0.5 for key in METRICS})
    assert not stopper.observe({key: 0.49 for key in METRICS}, 10)
    assert not stopper.observe({**{key: 0.49 for key in METRICS}, METRICS[-1]: 0.51}, 20)
    assert not stopper.observe({key: 0.49 for key in METRICS}, 30)
    assert stopper.observe({key: 0.49 for key in METRICS}, 40)
    comparison = compare_budgets([{"update": 1, "training_seconds": 10, "raw_review_presentations": 100}],
                                 [{"update": 1, "training_seconds": 7, "raw_review_presentations": 100},
                                  {"update": 2, "training_seconds": 14, "raw_review_presentations": 200}])
    assert comparison["training_seconds"]["candidate"]["update"] == 1
    assert comparison["raw_review_presentations"]["candidate"]["update"] == 1
    # Explicit pretraining lineage and forbidden-user handling, using a new fixture checkpoint.
    fixture = output / "pretraining-fixture.pth"
    torch.save(plain.state_dict(), fixture)
    manifest = {"sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(), "dataset_revision": "synthetic-fixture",
                "training_users": [1], "validation_users": [2], "training_reviews": 4,
                "provenance": {"kind": "synthetic CPU fixture; not a training result"}}
    manifest_path = output / "pretraining-fixture.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    load_pretrained(plain, fixture, manifest_path, [99])
    try:
        load_pretrained(plain, fixture, manifest_path, [1])
    except ValueError:
        pass
    else:
        raise AssertionError("Pretraining must reject reserved-user leakage")
    results["training_targets"] = {"schedules": schedules, "target_confirmation": "passed", "no_checkpoint_extrapolation": "passed",
                                   "pretraining_manifest_and_leak_guard": "passed"}
    results["scope"] = "CPU functional checks on synthetic data; not predictive quality, speed or GPU backend support"
    return results


def cuda_checks():
    import os
    import psutil
    import torch
    lock = ROOT / "artifacts/gpu-batch1.lock"
    if lock.exists():
        owner = json.loads(lock.read_text())
        if psutil.pid_exists(owner["pid"]):
            raise RuntimeError("Approved GPU suite is live; do not run competing GPU checks")
    state = ROOT / "artifacts/benchmarks/gpu-batch1-500/process.json"
    if state.exists() and json.loads(state.read_text()).get("status") == "running":
        raise RuntimeError("Verify the GPU suite is terminal before running CUDA checks")
    os.environ["CUPY_CACHE_DIR"] = str(ROOT / "artifacts/cache/cupy-cache")
    from rwkv.model import nvrtc
    from rwkv.model.rwkv_ops import RWKV7_WKV
    from scripts.gpu_row_tiles import compile_module, row_tiled_backend
    from scripts.gpu_fp8 import FP8Projection, probe_fp8, quantize
    nvrtc.register()
    results = {"recurrence": []}
    device = torch.device("cuda:0")
    for dtype in (torch.float32, torch.bfloat16):
        for heads in (1, 4):
            for length in (1, 7, 8, 9, 32, 128, 255, 383):
                torch.manual_seed(2025 + length)
                shape = (2, length, heads, 32)
                values = [(torch.randn(shape, device=device) * 0.1).to(dtype) for _ in range(6)]
                values[3] = torch.full(shape, 0.9, device=device)
                skip = torch.zeros(2, length, device=device, dtype=torch.bool)
                skip[0, ::3] = True
                skip[1] = True
                upstream = [x.detach().clone().requires_grad_() for x in values]
                gradient = torch.randn(shape, device=device, dtype=dtype)
                nvrtc.set_checkpoint_length(8)
                expected = RWKV7_WKV.apply(*upstream, skip)
                left_grads = torch.autograd.grad(expected, upstream, gradient)
                for rows in (1, 2, 4):
                    candidate = [x.detach().clone().requires_grad_() for x in values]
                    with row_tiled_backend(rows, 8):
                        actual = RWKV7_WKV.apply(*candidate, skip)
                        right_grads = torch.autograd.grad(actual, candidate, gradient)
                    atol, rtol = ((3e-6, 3e-5) if dtype == torch.float32 else (0.001, 0.02))
                    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
                    for left, right in zip(left_grads, right_grads):
                        torch.testing.assert_close(right, left, atol=atol, rtol=rtol)
                    kernels = compile_module(device, dtype, rows, 8)
                    results["recurrence"].append({"dtype": str(dtype), "heads": heads, "length": length, "rows": rows,
                                                  "forward_max_abs": float((actual - expected).abs().max()),
                                                  "gradient_max_abs": [float((a - b).abs().max()) for a, b in zip(right_grads, left_grads)],
                                                  "kernel_attributes": [kernel.attributes for kernel in kernels]})
    nvrtc.set_checkpoint_length(32)
    results["fp8_probe"] = probe_fp8(device)
    if results["fp8_probe"]["status"] == "supported":
        x = torch.randn(2, 17, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(64, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
        bias = torch.randn(64, device=device, dtype=torch.bfloat16, requires_grad=True)
        y = FP8Projection.apply(x, w, bias)
        qx, sx = quantize(x.flatten(0, 1))
        qw, sw = quantize(w)
        reference = ((qx.float() * sx) @ (qw.float() * sw).t()).to(torch.bfloat16) + bias
        torch.testing.assert_close(y.flatten(0, 1), reference, atol=0.25, rtol=0.02)
        gradient = torch.randn_like(y)
        dx, dw, db = torch.autograd.grad(y, (x, w, bias), gradient)
        torch.testing.assert_close(dx, (gradient.flatten(0, 1) @ w).view_as(x), atol=0, rtol=0)
        torch.testing.assert_close(dw, gradient.flatten(0, 1).t() @ x.flatten(0, 1), atol=0, rtol=0)
        torch.testing.assert_close(db, gradient.float().flatten(0, 1).sum(0).to(torch.bfloat16), atol=0, rtol=0)
        results["fp8_scaled_reference_and_backward"] = "passed"
    return results


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/benchmarks/gpu-future/cpu-check")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT / "artifacts") or (args.output / "result.json").exists():
        parser.error("Use a new output directory below workspace artifacts")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "provenance": process_provenance()}
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(2025)
    try:
        report["cpu"] = cpu_checks(args.output)
        if args.cuda:
            report["cuda"] = cuda_checks()
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {args.output / 'result.json'}")


if __name__ == "__main__":
    main()
