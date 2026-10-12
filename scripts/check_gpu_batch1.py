"""Input, gradient, stream and shape-cache checks before the 500-user suite."""

from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from rwkv.model import nvrtc
from rwkv.model.fused_shift import FusedShiftLerp
from rwkv.data_processing import get_rwkv_data
from rwkv.prepare_batch import prepare
from scripts.gpu_batch1 import (DATA, DynamicGraphs, ReducedSyncLoss, batch_tensors,
                                build_sample_cache, make_samples, pack_batch, prepare_job,
                                split_frame, transfer_batches)
from scripts.benchmark_gpu_trial import benchmark_models
from scripts.experiment_provenance import process_provenance
from scripts.trial_batches import TrialBatch


def check():
    provenance = process_provenance()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.72, 0)
    nvrtc.register()
    device = torch.device("cuda:0")
    results = {"provenance": provenance, "data_path": str(DATA),
               "dataset_manifest": json.loads((DATA / "download_manifest.json").read_text())["revision"]}
    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(2025)
        x = torch.randn(3, 17, 64, device=device, dtype=dtype, requires_grad=True)
        weight = torch.randn(8, 1, 1, 64, device=device, dtype=dtype, requires_grad=True)
        select = torch.arange(17, device=device).sub(1).clamp_min(0).repeat(3, 1)
        grad = torch.randn(8, 3, 17, 64, device=device, dtype=dtype)
        shifted = torch.gather(x, 1, select.unsqueeze(-1).expand_as(x))
        original = torch.lerp(x.unsqueeze(0), shifted.unsqueeze(0), weight)
        expected = torch.autograd.grad(original, (x, weight), grad)
        fused = FusedShiftLerp.apply(x, select, weight)
        actual = torch.autograd.grad(fused, (x, weight), grad)
        atol, rtol = (3e-6, 3e-6) if dtype == torch.float32 else (0.125, 0.025)
        torch.testing.assert_close(fused, original, atol=atol, rtol=rtol)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        results[str(dtype)] = {"forward_max_abs": float((fused - original).detach().abs().max()),
                               "gradient_max_abs": [float((a - b).abs().max()) for a, b in zip(actual, expected)],
                               "atol": atol, "rtol": rtol}
    source_frame = get_rwkv_data(DATA, 17)
    row = {"user_id": 17, "history_decile": 1, "available_reviews": len(source_frame),
           "train_range": [0, 512], "validation_range": [512, 1024]}
    frame = split_frame(source_frame, row, "train")
    samples = make_samples(frame, 17, 128)
    batch = prepare(samples, seed=2042)
    packed = pack_batch(batch)
    for a, b in zip(batch_tensors(batch), batch_tensors(packed)):
        assert torch.equal(a, b)
    paths, cache = build_sample_cache([row], 512, ROOT / "artifacts/diagnostics/check-temp/batch1-cache")
    cached = prepare_job(row, 0, 512, 2042, {"sample_cache": True}, cache_path=paths[(17, 0)]).batch
    local = prepare_job(row, 0, 512, 2042, {"worker_local": True, "packed_ipc": True}).batch
    for other in (cached, local):
        for a, b in zip(batch_tensors(batch), batch_tensors(other)):
            assert torch.equal(a, b)
    results["cpu_input_checks"] = {"packed": "exact", "cached": "exact", "worker_local": "exact", "cache": cache}
    for model_name in ("baseline", "expertium"):
        config = {"MODEL": model_name, "DTYPE": "bfloat16", "WKV_BACKEND": "nvrtc", "SEED": 2025}
        torch.manual_seed(2025)
        master, model, dtype = benchmark_models(config, device)
        # Exercise initially-zero projections and dropout-free full parameter gradients.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.01)
        model.eval()
        original = batch.to(device)
        original.start = original.start.to(dtype)
        stats = model.get_loss(original)
        stats.average_loss.backward()
        gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        loss = stats.average_loss.detach().clone()
        model.zero_grad(set_to_none=True)
        reduced = ReducedSyncLoss(model).get_loss(original)
        torch.testing.assert_close(reduced.average_loss, loss)
        reduced.average_loss.backward()
        for name, parameter in model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(parameter.grad, gradients[name], atol=0.002, rtol=0.04)
        model.zero_grad(set_to_none=True)
        less_padding = prepare(samples, seed=2042, padding_factor=0.3).to(device)
        less_padding.start = less_padding.start.to(dtype)
        value = model.get_loss(less_padding)
        torch.testing.assert_close(value.average_loss, loss, atol=2e-4, rtol=2e-4)
        value.average_loss.backward()
        for name, parameter in model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(parameter.grad, gradients[name], atol=0.002, rtol=0.04)
        model.zero_grad(set_to_none=True)
        item = TrialBatch(batch, 0, 512, 4, 2042, 0)
        staged = list(transfer_batches(iter([(row, item), (row, item)]), dtype, True))
        for _, _, moved in staged:
            for a, b in zip(batch_tensors(original), batch_tensors(moved)):
                assert torch.equal(a, b)
        del staged, value, less_padding, reduced, gradients, stats, master
        # Independent graph parameters; exact-shape admission and shape fallback.
        nvrtc.set_raw_pointer_launch(True)
        manager = DynamicGraphs(model, lambda: benchmark_models(config, device)[1])
        assert manager.select(original) is model
        graph = manager.select(original)
        graph.load_state_dict(model.state_dict())
        graph.eval()
        torch.manual_seed(2042)
        captured = graph.graph_update(original, 2042)
        torch.testing.assert_close(captured.average_loss, loss, atol=2e-4, rtol=2e-4)
        assert manager.select(original) is graph
        changed = replace(original, num_data=original.num_data + 1)
        assert manager.select(changed) is model
        results[model_name] = {"reduced_sync": "loss/all parameter gradients within BF16 tolerance (atol 0.002, rtol 0.04)", "padding": "loss/all parameter gradients within BF16 tolerance (atol 0.002, rtol 0.04)",
                               "async_transfer": "exact", "graphs": manager.metrics}
        del manager, graph, captured, model, original
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        nvrtc.set_raw_pointer_launch(False)
    results["status"] = "passed"
    path = ROOT / "artifacts/diagnostics/check-temp/gpu-batch1-check.json"
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    try:
        check()
    except Exception as exc:
        import traceback
        (ROOT / "artifacts/diagnostics/check-temp/gpu-batch1-check-failure.json").write_text(
            json.dumps({"provenance": process_provenance(), "status": "failed",
                        "error": str(exc), "traceback": traceback.format_exc()}, indent=2) + "\n")
        raise
