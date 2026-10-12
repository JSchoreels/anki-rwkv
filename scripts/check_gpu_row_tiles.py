"""Focused GPU parity check for F09 row-tiled recurrence kernels at the accepted tile 32.

Compares the production NVRTC kernels with the F09 row layouts bitwise: forward
outputs, state checkpoints and all six input gradients on synthetic edge cases,
then the whole Expertium loss and every parameter gradient on real R01 windows
with the accepted 0020 options. Exact equality is reported separately from
maximum absolute differences. No training, optimizer update or evaluation runs.
"""

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def synthetic(torch, b, t, skip_mode, generator):
    device = torch.device("cuda:0")
    shape = (b, t, 1, 32)

    def normal(scale=1.0):
        return (torch.randn(shape, generator=generator, device=device) * scale).to(torch.bfloat16).contiguous()

    r, k, v = normal(), normal(0.5), normal()
    w = torch.exp(-torch.exp(torch.randn(shape, generator=generator, device=device) - 1.0)).contiguous()
    a = torch.sigmoid(torch.randn(shape, generator=generator, device=device)).to(torch.bfloat16).contiguous()
    kd = torch.nn.functional.normalize(torch.randn(shape, generator=generator, device=device), dim=-1).to(torch.bfloat16).contiguous()
    if skip_mode == "none":
        skip = torch.zeros((b, t), dtype=torch.bool, device=device)
    elif skip_mode == "all":
        skip = torch.ones((b, t), dtype=torch.bool, device=device)
    elif skip_mode == "every-third":
        skip = (torch.arange(t, device=device) % 3 == 2).expand(b, t).contiguous()
    else:
        skip = (torch.rand((b, t), generator=generator, device=device) < 0.3).contiguous()
    grad = normal()
    return (r, k, v, w, a, kd), skip, grad


def compare(torch, reference, candidate):
    equal = all(torch.equal(x, y) for x, y in zip(reference, candidate))
    difference = max(float((x.float() - y.float()).abs().max()) if x.numel() else 0.0 for x, y in zip(reference, candidate))
    return equal, difference


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--options", type=Path, required=True, help="Accepted case selection.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", default="1,2,4")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Output must remain under workspace artifacts")
    if output.exists():
        raise FileExistsError("Preserve the existing check")
    rows_list = [int(value) for value in args.rows.split(",")]
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
    from scripts.gpu_row_tiles import row_tiled_backend
    nvrtc.register()
    nvrtc.set_checkpoint_length(32)
    nvrtc.set_raw_pointer_launch(False)
    device = torch.device("cuda:0")
    result = {"provenance": provenance, "options": options, "checkpoint_tile": 32, "rows_per_thread": rows_list,
              "scope": "bitwise parity of F09 row layouts against the production NVRTC recurrence; no training", "synthetic": [], "model": []}

    def run_op(inputs, skip, grad):
        out, states = torch.ops.rwkv.rwkv7_wkv_forward_bfloat16(*inputs, skip)
        gradients = torch.ops.rwkv.rwkv7_wkv_backward_bfloat16(*inputs, skip, states, grad)
        torch.cuda.synchronize()
        return (out, states, *gradients)

    generator = torch.Generator(device=device)
    cases = []
    for b in (1, 5, 97):
        for t in (1, 2, 31, 32, 33, 64, 65, 100, 383):
            for skip_mode in ("none", "all", "every-third", "random"):
                cases.append((b, t, skip_mode))
    for index, (b, t, skip_mode) in enumerate(cases):
        generator.manual_seed(20261012 + index)
        inputs, skip, grad = synthetic(torch, b, t, skip_mode, generator)
        reference = run_op(inputs, skip, grad)
        for rows in rows_list:
            with row_tiled_backend(rows=rows, tile=32):
                candidate = run_op(inputs, skip, grad)
            equal, difference = compare(torch, reference, candidate)
            result["synthetic"].append({"batch": b, "timesteps": t, "skip": skip_mode, "rows_per_thread": rows,
                                        "bitwise_equal": equal, "max_abs_difference": difference})

    from rwkv.data_processing import get_rwkv_data
    from scripts.benchmark_gpu_trial import benchmark_models, fresh_checkpoint
    from scripts.gpu_batch1 import split_frame
    from scripts.gpu_b01 import prepare_job, map_batch, materialize
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
            batches.append((row["user_id"], offset, item.seed, item.reviews,
                            materialize(map_batch(item.batch, lambda x: x.to(device)), dtype)))

    def gradients(batch, seed):
        torch.manual_seed(seed)
        base.zero_grad(set_to_none=True)
        stats = base.get_loss(batch)
        stats.average_loss.backward()
        torch.cuda.synchronize()
        return stats.average_loss.detach().clone(), [p.grad.detach().clone() if p.grad is not None else None for p in base.parameters()]

    for user, offset, seed, reviews, batch in batches:
        reference_loss, reference_grads = gradients(batch, seed)
        for rows in rows_list:
            with row_tiled_backend(rows=rows, tile=32):
                loss, grads = gradients(batch, seed)
            pairs = [(x, y) for x, y in zip(reference_grads, grads) if x is not None or y is not None]
            missing = sum((x is None) != (y is None) for x, y in pairs)
            present = [(x, y) for x, y in pairs if x is not None and y is not None]
            equal, difference = compare(torch, [x for x, _ in present], [y for _, y in present])
            result["model"].append({"user_id": user, "offset": offset, "reviews": reviews, "rows_per_thread": rows,
                                    "loss_bitwise_equal": bool(torch.equal(reference_loss, loss)),
                                    "parameter_gradients": len(present), "gradient_presence_mismatches": missing,
                                    "gradients_bitwise_equal": equal and missing == 0, "max_abs_gradient_difference": difference})
    result["summary"] = {str(rows): {
        "synthetic_cases": sum(1 for case in result["synthetic"] if case["rows_per_thread"] == rows),
        "synthetic_bitwise_equal": sum(case["bitwise_equal"] for case in result["synthetic"] if case["rows_per_thread"] == rows),
        "model_batches": sum(1 for case in result["model"] if case["rows_per_thread"] == rows),
        "model_bitwise_equal": sum(case["loss_bitwise_equal"] and case["gradients_bitwise_equal"]
                                   for case in result["model"] if case["rows_per_thread"] == rows)} for rows in rows_list}
    result["passed"] = {rows: value["synthetic_cases"] == value["synthetic_bitwise_equal"] and value["model_batches"] == value["model_bitwise_equal"]
                        for rows, value in result["summary"].items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "passed": result["passed"]}, indent=1))


if __name__ == "__main__":
    main()
