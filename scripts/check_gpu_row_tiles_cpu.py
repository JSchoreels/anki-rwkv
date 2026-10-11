"""F09 recurrence redesign: CPU equation audit and mocked backend cleanup.

The CPU model exercises row ownership, checkpoint reconstruction and the six
derivative equations against independent matrix/autograd references. It does not
execute CUDA source or establish GPU parity, speed or predictive quality.
"""

from argparse import ArgumentParser
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re
import sys
import traceback
from unittest.mock import patch
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_provenance import process_provenance


def compilation_evidence():
    path = ROOT / "artifacts/benchmarks/gpu-future/row-tiles-compile/result.json"
    previous = json.loads(path.read_text(encoding="utf-8"))
    provenance = previous["provenance"]
    archive = ROOT / provenance["source_archive"]
    if not archive.exists():
        archive = ROOT / "artifacts/sources/experiment-sources" / provenance["source_tree_sha256"] / "source.zip"
    dependencies = ("scripts/kernels/rwkv_row_tiles.cu", "scripts/gpu_row_tiles.py",
                    "rwkv/model/nvrtc.py", "rwkv/model/csrc/cuda/rwkv7_cuda.cu",
                    "rwkv/model/csrc/cuda/rwkv7_cuda_utils.h")
    matches = {}
    with ZipFile(archive) as snapshot:
        for name in dependencies:
            old, current = snapshot.read(name), (ROOT / name).read_bytes()
            matches[name] = {"sha256": hashlib.sha256(current).hexdigest(), "same_bytes": old == current}
    # Cleanup-only changes in the context manager do not change generated source.
    # Bind compilation reuse to source_options itself when its enclosing file differs.
    if not matches["scripts/gpu_row_tiles.py"]["same_bytes"]:
        import ast
        with ZipFile(archive) as snapshot:
            old_tree = ast.parse(snapshot.read("scripts/gpu_row_tiles.py").decode("utf-8"))
        new_tree = ast.parse((ROOT / "scripts/gpu_row_tiles.py").read_text(encoding="utf-8"))
        def source_options(tree):
            return ast.dump(next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "source_options"))
        matches["scripts/gpu_row_tiles.py"]["source_options_unchanged"] = source_options(old_tree) == source_options(new_tree)
    reusable = previous["status"] == "complete" and all(
        item["same_bytes"] or item.get("source_options_unchanged", False) for item in matches.values())
    resources = []
    for case in previous["cases"]:
        log = case["ptxas_log"]
        backward = log.split("Function properties for")[-1]
        resources.append({"name": case["name"], "rows": case["rows"], "tile": case["tile"],
                          "threads_per_block": 1024 // case["rows"],
                          "backward_registers_per_thread": int(re.search(r"Used (\d+) registers", backward)[1]),
                          "backward_stack_bytes_per_thread": int(re.search(r"(\d+) bytes stack frame", backward)[1]),
                          "spill_store_bytes": int(re.search(r"(\d+) bytes spill stores", backward)[1]),
                          "spill_load_bytes": int(re.search(r"(\d+) bytes spill loads", backward)[1])})
    return {"reusable": reusable, "result": path.relative_to(ROOT).as_posix(),
            "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_archive": archive.relative_to(ROOT).as_posix(), "source_matches": matches,
            "resources": resources, "scope": "Prior CPU compilation; not measured runtime occupancy or speed"}


def tiled_equations(values, skip, rows, tile):
    """CPU matrix model of row-owned states, checkpoints and analytical backward."""
    import torch
    dtype = values[0].dtype
    batch, length, heads, width = values[0].shape
    row_order = torch.arange(width).reshape(rows, width // rows).T.flatten()
    inverse = torch.argsort(row_order)
    assert torch.equal(row_order[inverse], torch.arange(width))
    inputs = [value.detach().float().permute(0, 2, 1, 3) for value in values]
    state = torch.zeros(batch, heads, width, width)
    checkpoints, output = [], []

    def transition(state, t):
        r, k, v, w, a, kd = [value[:, :, t] for value in inputs]
        dot = state @ kd.unsqueeze(-1)
        updated = state * w.unsqueeze(-2) - dot * (a * kd).unsqueeze(-2)
        return updated + v.index_select(-1, row_order).unsqueeze(-1) * k.unsqueeze(-2)

    for t in range(length):
        if t % tile == 0:
            checkpoints.append(state.index_select(-2, inverse).clone())
        updated = transition(state, t)
        prediction = (updated @ inputs[0][:, :, t].unsqueeze(-1)).squeeze(-1)
        output.append(prediction.index_select(-1, inverse))
        state = torch.where(skip[:, t, None, None, None], state, updated)
    output = torch.stack(output, dim=2).permute(0, 2, 1, 3).to(dtype)

    def backward(upstream):
        gradient = upstream.float().permute(0, 2, 1, 3)
        grads = [torch.empty_like(value) for value in inputs]
        carry = torch.zeros_like(state)
        for start in reversed(range(0, length, tile)):
            current = checkpoints[start // tile].index_select(-2, row_order)
            states, previous = [], []
            for t in range(start, min(length, start + tile)):
                previous.append(current)
                updated = transition(current, t)
                states.append(updated)
                current = torch.where(skip[:, t, None, None, None], current, updated)
            for t in reversed(range(start, min(length, start + tile))):
                r, k, v, w, a, kd = [value[:, :, t] for value in inputs]
                g = gradient[:, :, t]
                skipped = skip[:, t, None, None, None]
                ds = g.index_select(-1, row_order).unsqueeze(-1) * r.unsqueeze(-2)
                ds += torch.where(skipped, torch.zeros_like(carry), carry)
                s = states[t - start].index_select(-2, inverse)
                p = previous[t - start].index_select(-2, inverse)
                natural_ds = ds.index_select(-2, inverse)
                product = p.transpose(-2, -1) @ natural_ds
                transpose_dot = (product.transpose(-2, -1) @ kd.unsqueeze(-1)).squeeze(-1)
                grads[0][:, :, t] = (s.transpose(-2, -1) @ g.unsqueeze(-1)).squeeze(-1)
                grads[1][:, :, t] = (natural_ds.transpose(-2, -1) @ v.unsqueeze(-1)).squeeze(-1)
                grads[2][:, :, t] = (natural_ds @ k.unsqueeze(-1)).squeeze(-1)
                grads[3][:, :, t] = product.diagonal(dim1=-2, dim2=-1)
                grads[4][:, :, t] = -transpose_dot * kd
                grads[5][:, :, t] = -(product @ (a * kd).unsqueeze(-1)).squeeze(-1) - a * transpose_dot
                removed = ds @ (a * kd).unsqueeze(-1)
                carry = torch.where(skipped, carry, torch.zeros_like(carry)) + ds * w.unsqueeze(-2) - removed * kd.unsqueeze(-2)
        return [value.permute(0, 2, 1, 3).to(original.dtype) for value, original in zip(grads, values)]
    return output, backward


def equation_checks():
    import torch
    from rwkv.model.rwkv_ops import reference_rwkv7
    cases = []
    lengths = (1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 127, 128, 129, 255, 383)
    for dtype in (torch.float32, torch.bfloat16):
        for heads in (1, 4):
            for length in lengths:
                generator = torch.Generator(device="cpu").manual_seed(2025 + length + heads)
                shape = (3, length, heads, 32)
                values = [(torch.randn(shape, generator=generator) * 0.1).to(dtype) for _ in range(6)]
                values[3] = 0.8 + torch.rand(shape, generator=generator) * 0.18
                skip = torch.zeros(3, length, dtype=torch.bool)
                skip[1] = True
                skip[2, ::3] = True
                differentiable = [value.clone().requires_grad_() for value in values]
                expected = reference_rwkv7(*differentiable, skip)
                upstream = (torch.randn(shape, generator=generator) * 0.1).to(dtype)
                expected_grads = torch.autograd.grad(expected, differentiable, upstream)
                atol, rtol = (3e-6, 3e-5) if dtype == torch.float32 else (0.001, 0.02)
                for rows in (1, 2, 4):
                    for tile in (8, 16, 32):
                        actual, backward = tiled_equations(values, skip, rows, tile)
                        actual_grads = backward(upstream)
                        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
                        for left, right in zip(actual_grads, expected_grads):
                            torch.testing.assert_close(left, right, atol=atol, rtol=rtol)
                        cases.append({"dtype": str(dtype), "heads": heads, "length": length, "rows": rows, "tile": tile,
                                      "forward_max_abs": float((actual.float() - expected.detach().float()).abs().max()),
                                      "gradient_max_abs": [float((left.float() - right.float()).abs().max())
                                                           for left, right in zip(actual_grads, expected_grads)]})
            print(f"CPU equation checks: {dtype}, {heads} heads complete", flush=True)
    return {"case_count": len(cases), "cases": cases, "skip_patterns": ["none", "all", "every third timestep"],
            "reference": "Existing independent matrix recurrence with Torch autograd",
            "scope": "CPU equations and row/checkpoint indexing only; CUDA code was not executed"}


def backend_cleanup_checks(cases):
    import torch
    from rwkv.model import nvrtc
    from scripts.gpu_row_tiles import row_tiled_backend
    original = nvrtc.module, nvrtc.launch, nvrtc._checkpoint_length
    for failure in (None, "body", "synchronize"):
        with patch.object(nvrtc, "register"), patch.object(torch.cuda, "synchronize") as synchronize:
            if failure == "synchronize":
                synchronize.side_effect = RuntimeError("synthetic synchronize failure")
            try:
                with row_tiled_backend(rows=2, tile=8):
                    assert nvrtc._checkpoint_length == 8
                    if failure == "body":
                        raise RuntimeError("synthetic body failure")
            except RuntimeError as exc:
                assert failure and str(exc) == f"synthetic {failure} failure"
            else:
                assert failure is None
            finally:
                restored = (nvrtc.module, nvrtc.launch, nvrtc._checkpoint_length) == original
                # Always restore this CPU fixture, even on an assertion failure.
                nvrtc.module, nvrtc.launch = original[:2]
                nvrtc.set_checkpoint_length(original[2])
            cases.append({"failure": failure, "restored": restored, "mocked_sync_calls": synchronize.call_count})
            assert restored, f"F09 backend was not restored after {failure} failure"
    return cases


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        parser.error("Use a new directory under workspace artifacts")
    output.mkdir(parents=True)
    report = {"status": "running", "track": "F09 (recurrence kernel redesign)", "provenance": process_provenance(),
              "scope": "CPU-only investigation; no CUDA initialization, launches, training, speed or held-out loss measurement"}
    try:
        import torch
        torch.set_num_threads(1)
        with ExitStack() as guards:
            guards.enter_context(patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU audit must not initialize CUDA")))
            report["compilation"] = compilation_evidence()
            assert report["compilation"]["reusable"], "Prior compilation no longer matches generated-source dependencies"
            report["equations"] = equation_checks()
            report["backend_cleanup"] = []
            backend_cleanup_checks(report["backend_cleanup"])
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {output / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
