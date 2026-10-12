"""F50 boundary/cancellation/signed-zero checks for the unchanged row kernels."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists() or not output.resolve().is_relative_to(ROOT / "artifacts"):
        parser.error("Use a new artifacts directory")
    import torch
    from rwkv.model import exact_index, nvrtc
    from scripts.experiment_provenance import process_provenance
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    nvrtc.register()
    nvrtc.set_stream_launch(True)
    cases = []
    for dtype in (torch.bfloat16, torch.float32):
        for count in (0, 1, 31, 32, 63, 64, 65, 127, 128, 129, 513):
            for layout in ("prefix", "suffix", "interleaved", "all"):
                index = torch.arange(count, device="cuda") % 3
                padding = torch.arange(count, device="cuda") % 3 == 1
                if layout == "prefix":
                    padding = torch.arange(count, device="cuda") < count // 2
                elif layout == "suffix":
                    padding = torch.arange(count, device="cuda") >= count // 2
                elif layout == "all":
                    padding = torch.ones(count, device="cuda", dtype=torch.bool)
                index[padding] = -1
                # Cancellation and wide exponents distinguish reordered float sums.
                values = torch.tensor([1e10, 1, -1e10, 0.125, -0.125, -0.0, 0.0], device="cuda", dtype=dtype)
                grad = values[torch.arange(count * 32, device="cuda") % len(values)].view(count, 32)
                grad[padding] = -0.0 if count % 2 else 0.0
                for prefetch in (False, True):
                    exact_index.set_prefetch_rows(prefetch)
                    reference = exact_index.rows_backward(grad, index.clamp(min=0), 5)
                    candidate = exact_index.rows_backward(grad, index, 5)
                    bits = torch.int16 if dtype == torch.bfloat16 else torch.int32
                    equal = torch.equal(reference.view(bits), candidate.view(bits))
                    x = torch.arange(5 * 32, device="cuda", dtype=dtype).view(5, 32)
                    y = exact_index.ExactPaddedIndexSelect.apply(x, index)
                    forward_equal = torch.equal(y.view(bits), x.index_select(0, index.clamp(min=0)).view(bits))
                    cases.append({"dtype": str(dtype), "count": count, "layout": layout, "prefetch": prefetch,
                                  "backward_equal": equal, "forward_equal": forward_equal})
    torch.cuda.synchronize()
    passed = all(c["backward_equal"] and c["forward_equal"] for c in cases)
    output.mkdir(parents=True)
    result = {"status": "passed" if passed else "failed", "provenance": process_provenance(),
              "torch": torch.__version__, "summary": {"cases": len(cases), "all_bitwise_equal": passed}, "cases": cases}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"]))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
