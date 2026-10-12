"""F05 CPU structural/gradient checks; no GPU or broad quality measurements."""

from dataclasses import replace
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance


def fixture():
    import torch
    from rwkv.model.srs_model import PreparedBatch
    gather = [[torch.tensor([2, 0, 1, 3, 7, -1]), torch.tensor([4, 5, 6, -1])],
              [torch.tensor([0, 2, 4]), torch.tensor([1, 3, 6, 7, 8, -1])],
              [torch.tensor([5, 3, 4, -1]), torch.tensor([0, 1, 2, 6, 7, -1, -1, -1])],
              [torch.tensor([1, 0, 2]), torch.tensor([5, 4, 6, 8, 7])],
              [torch.arange(8)]]
    lengths = [[2, 4], [1, 3], [2, 4], [3, 5], [4]]
    skips, shifts = [], []
    for groups, lens in zip(gather, lengths):
        stream_skips, stream_shifts = [], []
        for group, length in zip(groups, lens):
            skip = group.reshape(-1, length) < 0
            for row in skip:
                if length > 2:
                    row[1] = True  # A query can skip recurrence while retaining a label.
            select = torch.zeros_like(skip, dtype=torch.long)
            for row, destination in zip(skip, select):
                previous = 0
                for time in range(length):
                    destination[time] = previous
                    if not row[time]:
                        previous = time
            stream_skips.append(skip.flatten())
            stream_shifts.append(select.flatten())
        skips.append(stream_skips)
        shifts.append(stream_shifts)
    labels = torch.tensor([[[86400, 0, 0, 1, 1, 0, 0], [86400, 0, 1, 3, 1, 0, 1],
                            [172800, 0, 1, 4, 1, 0, 0], [172800, 0, 1, 2, 1, 0, 1]]] * 2, dtype=torch.float32)
    return PreparedBatch(2, torch.randn(8, 92), gather, lengths, shifts, skips, labels, torch.arange(8).reshape(2, 4))


def checks():
    import importlib
    import torch
    from scripts.gpu_future_models import TaskModel, capacity_variant, python_model_class
    from scripts.gpu_shape_buckets import BucketPolicy, BucketStatistics, bucket_batch
    torch.set_num_threads(1)
    torch.manual_seed(20251011)
    original = fixture()
    policy = BucketPolicy((1, 2, 4, 8), 20)  # Small synthetic fixture, deliberately loose test limit.
    result = bucket_batch(original, policy)
    assert result.admitted
    bucket = result.batch
    assert bucket.sub_gather_lens[:-1] == [[1, 2, 4, 8]] * 4
    assert bucket.sub_gather_lens[-1] == [8] and bucket.num_data == policy.capacity(original.num_data)
    torch.testing.assert_close(bucket.labels.reshape(-1, 7)[result.output_positions], original.labels.reshape(-1, 7), rtol=0, atol=0)
    assert torch.equal(bucket.label_review_th.flatten()[result.output_positions], original.label_review_th.flatten())
    old_counts = original.labels[..., 4].sum()
    assert bucket.labels[..., 4].sum() == old_counts
    # Different original hierarchical tensor shapes collapse to the same bucket.
    # The added source sequence is entirely padded and is never gathered later.
    variant = replace(original, sub_gather=[[value.clone() for value in group] for group in original.sub_gather],
                      skips=[[value.clone() for value in group] for group in original.skips],
                      time_shift_selects=[[value.clone() for value in group] for group in original.time_shift_selects])
    variant.sub_gather[0][0] = torch.cat((variant.sub_gather[0][0], torch.tensor([-1, -1])))
    variant.skips[0][0] = torch.cat((variant.skips[0][0], torch.tensor([True, True])))
    variant.time_shift_selects[0][0] = torch.cat((variant.time_shift_selects[0][0], torch.tensor([0, 0])))
    for group in variant.sub_gather[1]:
        group[group >= 6] += 2
    alternative = bucket_batch(variant, policy)
    assert alternative.admitted and original.sub_gather[0][0].shape != variant.sub_gather[0][0].shape
    fields = ("sub_gather", "time_shift_selects", "skips")
    assert all(torch.equal(a, b) for name in fields for left, right in zip(getattr(bucket, name), getattr(alternative.batch, name)) for a, b in zip(left, right))
    invalid_dependency = replace(original, sub_gather=[[value.clone() for value in group] for group in original.sub_gather])
    invalid_dependency.sub_gather[1][0][0] = 5  # Stream 0's padded output.
    assert not bucket_batch(invalid_dependency, policy).admitted
    # No truncation or silently enlarged production padding budget.
    rejected = bucket_batch(original, BucketPolicy((1, 2, 4, 8), 1))
    assert not rejected.admitted and rejected.batch is original
    too_long = bucket_batch(original, BucketPolicy((1, 2), 20))
    assert not too_long.admitted and too_long.batch is original
    statistics = BucketStatistics(policy)
    statistics.prepare(original)
    statistics.policy = BucketPolicy((1, 2, 4, 8), 1)
    statistics.prepare(original)
    assert statistics.report()["admitted"] == statistics.report()["eager_fallbacks"] == 1
    results = {"structural": "labels, review IDs, skipped padding, sequence remapping and canonical signature checked",
               "fallbacks": "excess work, unsupported sequence length and a real position depending on padding preserve the original batch",
               "fixture_work": result.metrics, "fixture_counters": statistics.report()}
    for name in ("baseline", "expertium"):
        package = "rwkv" if name == "baseline" else name
        architecture = importlib.import_module(package + ".architecture").DEFAULT_ANKI_RWKV_CONFIG
        architecture = capacity_variant(architecture, width_heads=1, remove_layers=20)
        # Same all-five-stream small architecture on each side. The Python clone
        # defines the baseline's one-layer initialization endpoint.
        base = TaskModel(python_model_class(name)(architecture)).float().eval()
        with torch.no_grad():
            for param in base.parameters():
                param.add_(torch.randn_like(param) * 0.01)
        actual = TaskModel(python_model_class(name)(architecture)).float().eval()
        actual.load_state_dict(base.state_dict())
        expected_stats, actual_stats = base.get_loss(original), actual.get_loss(bucket)
        torch.testing.assert_close(actual_stats.average_loss, expected_stats.average_loss, atol=3e-6, rtol=3e-5)
        errors = {}
        for field in ("p_curve", "p_imm", "p_imm_all"):
            expected, prediction = getattr(expected_stats, field), getattr(actual_stats, field)
            prediction = prediction.reshape(-1, *prediction.shape[2:])[result.output_positions]
            expected = expected.reshape_as(prediction)
            torch.testing.assert_close(prediction, expected, atol=3e-6, rtol=3e-5)
            errors[field] = float((prediction - expected).detach().abs().max())
        expected_stats.average_loss.backward()
        actual_stats.average_loss.backward()
        gradient_error = 0.0
        for (left_name, left), (right_name, right) in zip(base.named_parameters(), actual.named_parameters()):
            assert left_name == right_name
            assert (left.grad is None) == (right.grad is None)
            if left.grad is not None:
                torch.testing.assert_close(right.grad, left.grad, atol=3e-6, rtol=3e-5)
                gradient_error = max(gradient_error, float((right.grad - left.grad).abs().max()))
        results[name] = {"prediction_max_abs": errors, "all_parameter_gradient_max_abs": gradient_error,
                         "precision": "FP32 CPU; dropout disabled", "atol": 3e-6, "rtol": 3e-5}
    return results


def cuda_checks():
    """Run only after verifying the active suite has exited; synthetic checks only."""
    import importlib
    import psutil
    import torch
    from scripts.gpu_future_models import capacity_variant
    from scripts.gpu_shape_buckets import BucketPolicy, BucketedGraphStage
    from rwkv.model import nvrtc
    state = ROOT / "artifacts/benchmarks/gpu-batch1-500/process.json"
    lock = ROOT / "artifacts/gpu-batch1.lock"
    if (state.exists() and json.loads(state.read_text()).get("status") == "running") or (lock.exists() and psutil.pid_exists(json.loads(lock.read_text())["pid"])):
        raise RuntimeError("Verify the first-batch worker has exited before running CUDA checks")
    torch.cuda.set_per_process_memory_fraction(0.72, 0)
    nvrtc.register()
    nvrtc.set_checkpoint_length(32)
    nvrtc.set_raw_pointer_launch(True)
    device = torch.device("cuda:0")
    results = []
    torch.manual_seed(2025)
    source = fixture()
    for name in ("baseline", "expertium"):
        package = "rwkv" if name == "baseline" else name
        architecture = importlib.import_module(package + ".architecture").DEFAULT_ANKI_RWKV_CONFIG
        architecture = capacity_variant(architecture, width_heads=1, remove_layers=20)
        if name == "baseline":
            architecture = replace(architecture, modules=[(stream, replace(config, n_layers=2, total_layers=2)) for stream, config in architecture.modules])
        cls = importlib.import_module(package + ".model.srs_model").SrsRWKV
        master = cls(architecture).float().to(device)
        with torch.no_grad():
            for parameter in master.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.01)
        for dtype in (torch.float32, torch.bfloat16):
            def factory():
                return cls(architecture).float().to(device).selective_cast(dtype)
            base = factory().eval()
            base.copy_downcast_(master, dtype)
            original = source.to(device)
            original.start = original.start.to(dtype)
            expected = base.get_loss(original)
            expected.average_loss.backward()
            expected_loss = float(expected.average_loss.detach())
            gradients = {key: parameter.grad.detach().clone() for key, parameter in base.named_parameters() if parameter.grad is not None}
            expected = None
            base.zero_grad(set_to_none=True)
            stage = BucketedGraphStage(base, factory, BucketPolicy((1, 2, 4, 8), 20))
            maximum_gradient_error, maximum_loss_error = 0.0, 0.0
            for _ in range(3):
                prepared = stage.prepare_cpu(source)
                assert prepared.admitted
                bucket = prepared.batch.to(device)
                bucket.start = bucket.start.to(dtype)
                active, stats = stage.backward(master, bucket, dtype, True, 2042, training=False)
                atol, rtol = (3e-5, 3e-4) if dtype == torch.float32 else (0.004, 0.04)
                torch.testing.assert_close(stats.average_loss, torch.tensor(expected_loss, device=device), atol=atol, rtol=rtol)
                maximum_loss_error = max(maximum_loss_error, abs(float(stats.average_loss.detach()) - expected_loss))
                for key, parameter in active.named_parameters():
                    assert (parameter.grad is not None) == (key in gradients)
                    if key in gradients:
                        torch.testing.assert_close(parameter.grad, gradients[key], atol=atol, rtol=rtol)
                        maximum_gradient_error = max(maximum_gradient_error, float((parameter.grad - gradients[key]).detach().abs().max()))
                parameter = active = stats = bucket = None
            assert stage.graphs.metrics["captures"] == 1 and stage.graphs.metrics["hits"] == 1
            # Training mode is baked into capture. Switching must reset the
            # cached eval graph, produce finite gradients, and use eager first.
            prepared = stage.prepare_cpu(source)
            bucket = prepared.batch.to(device)
            bucket.start = bucket.start.to(dtype)
            active, stats = stage.backward(master, bucket, dtype, True, 2042, training=True)
            assert stage.mode_changes == 1 and not stage.graphs.cache
            assert all(torch.isfinite(parameter.grad).all() for parameter in active.parameters() if parameter.grad is not None)
            active = stats = bucket = None
            original.start = original.start.to(dtype)
            active, stats = stage.backward(master, original, dtype, False, 2042, training=True)
            assert not stage.graphs.cache
            results.append({"model": name, "dtype": str(dtype), "loss_max_abs": maximum_loss_error,
                            "all_parameter_gradient_max_abs": maximum_gradient_error,
                            "atol": atol, "rtol": rtol, "stage": stage.report(),
                            "checks": "eager bucket, second-use capture, replay, dropout-mode reset and explicit eager fallback"})
            active = stats = stage = base = original = gradients = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        master = None
        torch.cuda.empty_cache()
    nvrtc.set_raw_pointer_launch(False)
    return results


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/benchmarks/gpu-future/shape-buckets-cpu-v1.json")
    parser.add_argument("--cuda", action="store_true", help="Synthetic CUDA graph/parity checks after the live suite exits")
    args = parser.parse_args()
    provenance = process_provenance()
    target = args.output.resolve()
    if not target.is_relative_to(ROOT / "artifacts") or target.exists():
        raise FileExistsError("Preserve earlier evidence; choose a new result filename")
    target.parent.mkdir(parents=True, exist_ok=True)
    result = {"provenance": provenance, "scope": "synthetic structure/gradient checks, with CUDA capture/replay checks only when --cuda is requested; no throughput or 500-user quality claim"}
    try:
        result.update(status="passed", checks=checks())
        if args.cuda:
            result["cuda_checks"] = cuda_checks()
        print(f"Hierarchical buckets: {'CPU and CUDA' if args.cuda else 'CPU'} structure, losses and all parameter gradients passed for both models")
    except Exception as exc:
        result.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
