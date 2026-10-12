"""Focused GPU parity check for F32 exact GroupNorm weight/bias gradients.

Compares ExactGroupNorm with torch.nn.GroupNorm bitwise (output, input, weight
and bias gradients) on synthetic (N, 32) inputs, then the whole Expertium loss
and every parameter gradient on real R01 windows with the accepted options,
F32 on versus off. No training, optimizer update or evaluation runs.
"""

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--options", type=Path, required=True, help="Accepted case selection.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        parser.error("Use a new output under workspace artifacts")
    options = json.loads((ROOT / args.options).read_text(encoding="utf-8"))
    options = options.get("expertium", options).get("options", options)
    from scripts.benchmark_r01_components import windows
    data_path, selected = windows("expertium")
    os.environ["GPU_BATCH1_DATA_PATH"] = str(ROOT / data_path)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["CUPY_CACHE_DIR"] = str(ROOT / "artifacts/cache/cupy-cache")
    import torch
    from scripts.experiment_provenance import process_provenance
    provenance = process_provenance()
    torch._C._set_graph_executor_optimize(False)
    torch.use_deterministic_algorithms(True, warn_only=False)
    from rwkv.model import nvrtc
    from rwkv.model.exact_group_norm import ExactGroupNorm
    nvrtc.register()
    nvrtc.set_checkpoint_length(32)
    device = torch.device("cuda:0")
    result = {"provenance": provenance, "options": options, "torch": torch.__version__,
              "scope": "bitwise parity of F32 GroupNorm weight/bias backward against PyTorch; no training", "synthetic": [], "model": []}
    for dtype in (torch.bfloat16, torch.float32):
        for N in (1, 100, 128, 129, 1000, 4097, 40824, 62712, 109134, 131072):
            for seed in range(3):
                generator = torch.Generator(device=device).manual_seed(seed * 1000 + N)
                x = (torch.randn(N, 32, generator=generator, device=device) * 3).to(dtype)
                norm = torch.nn.GroupNorm(1, 32, eps=64e-5).to(device).to(dtype)
                with torch.no_grad():
                    norm.weight.copy_(torch.randn(32, generator=generator, device=device))
                    norm.bias.copy_(torch.randn(32, generator=generator, device=device))
                grad = torch.randn(N, 32, generator=generator, device=device).to(dtype)
                values = []
                for exact in (False, True):
                    norm.zero_grad(set_to_none=True)
                    leaf = x.clone().requires_grad_()
                    out = ExactGroupNorm.apply(leaf, norm.weight, norm.bias, 1, 64e-5) if exact else norm(leaf)
                    out.backward(grad)
                    values.append((out.detach(), leaf.grad, norm.weight.grad.clone(), norm.bias.grad.clone()))
                equal = [bool(torch.equal(a, b)) for a, b in zip(*values)]
                result["synthetic"].append({"dtype": str(dtype), "rows": N, "seed": seed,
                                            "output_input_weight_bias_equal": equal, "bitwise_equal": all(equal)})

    from rwkv.data_processing import get_rwkv_data
    from scripts.benchmark_gpu_trial import benchmark_models, fresh_checkpoint
    from scripts.gpu_batch1 import split_frame
    from scripts.gpu_b01 import prepare_job, map_batch, materialize
    from contextlib import ExitStack
    kernels = ExitStack()
    if options.get("row_tiles_exact"):
        from scripts.gpu_row_tiles import row_tiled_backend
        kernels.enter_context(row_tiled_backend(rows=options["row_tiles_exact"], tile=32, exact=True))
    span = options["batch_chunks"] * 128
    torch.cuda.set_per_process_memory_fraction(options["gpu_memory_fraction"], 0)
    config = {"MODEL": "expertium", "DTYPE": "bfloat16", "SEED": 2025, "WKV_BACKEND": "nvrtc"}
    master, base, dtype = benchmark_models(config, device)
    master.load_state_dict(fresh_checkpoint("expertium", 2025)["model"])
    base.copy_downcast_(master, dtype)
    for module in base.modules():
        if hasattr(module, "exact_shift"):
            module.exact_shift = options.get("exact_shift", False)
        if hasattr(module, "exact_index"):
            module.exact_index = options.get("exact_index", False)
    base.train()
    batches = []
    for row, length in selected:
        frame = split_frame(get_rwkv_data(Path(os.environ["GPU_BATCH1_DATA_PATH"]), row["user_id"]), row, "train")
        for offset in range(0, length, span):
            end = min(length, offset + span)
            item = prepare_job(row, offset, end, 2025 + row["user_id"] + offset, options, frame.iloc[offset:end])
            batches.append((row["user_id"], offset, item.seed, item.reviews, materialize(map_batch(item.batch, lambda x: x.to(device)), dtype)))

    def gradients(batch, seed, exact):
        for module in base.modules():
            if hasattr(module, "exact_group_norm"):
                module.exact_group_norm = exact
        torch.manual_seed(seed)
        base.zero_grad(set_to_none=True)
        stats = base.get_loss(batch)
        stats.average_loss.backward()
        torch.cuda.synchronize()
        return stats.average_loss.detach().clone(), [p.grad.detach().clone() if p.grad is not None else None for p in base.parameters()]

    for user, offset, seed, reviews, batch in batches:
        reference_loss, reference = gradients(batch, seed, False)
        loss, candidate = gradients(batch, seed, True)
        pairs = list(zip(reference, candidate))
        missing = sum((a is None) != (b is None) for a, b in pairs)
        present = [(a, b) for a, b in pairs if a is not None and b is not None]
        equal = missing == 0 and all(torch.equal(a, b) for a, b in present)
        result["model"].append({"user_id": user, "offset": offset, "reviews": reviews, "loss_bitwise_equal": bool(torch.equal(reference_loss, loss)),
                                "parameter_gradients": len(present), "gradients_bitwise_equal": equal,
                                "max_abs_gradient_difference": max(float((a.float() - b.float()).abs().max()) for a, b in present)})
    kernels.close()
    result["summary"] = {"synthetic_cases": len(result["synthetic"]), "synthetic_bitwise_equal": sum(c["bitwise_equal"] for c in result["synthetic"]),
                         "model_batches": len(result["model"]),
                         "model_bitwise_equal": sum(c["loss_bitwise_equal"] and c["gradients_bitwise_equal"] for c in result["model"])}
    result["passed"] = (result["summary"]["synthetic_cases"] == result["summary"]["synthetic_bitwise_equal"]
                        and result["summary"]["model_batches"] == result["summary"]["model_bitwise_equal"])
    result["status"] = "passed" if result["passed"] else "failed"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "passed": result["passed"]}))


if __name__ == "__main__":
    main()
