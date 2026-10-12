"""F46 synthetic shift backward: exact deterministic sums at warp boundaries."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if output.exists() or not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Use a fresh artifacts output")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    from rwkv.model import exact_index
    from scripts.experiment_provenance import process_provenance
    result = {"provenance": process_provenance(), "torch": torch.__version__, "status": "running", "cases": [],
              "scope": "BF16/FP32, C32, B2; all-to-first, identity and random destinations; 1/31/32/33/63/64/65/127/128/129/512/4096; zero/signed-zero and cancellation gradients; compare deterministic Torch and accepted kernel"}
    generator = torch.Generator(device="cuda").manual_seed(46)
    try:
        for dtype in (torch.bfloat16, torch.float32):
            for length in (1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 512, 4096):
                for pattern in ("first", "identity", "random"):
                    select = (torch.zeros((2, length), device="cuda", dtype=torch.long) if pattern == "first" else
                              torch.arange(length, device="cuda").expand(2, -1).contiguous() if pattern == "identity" else
                              torch.randint(length, (2, length), device="cuda", generator=generator))
                    x = torch.zeros((2, length, 32), device="cuda", dtype=dtype, requires_grad=True)
                    grad = torch.randn(x.shape, device="cuda", dtype=dtype, generator=generator)
                    grad[:, ::3, :] = 0.0
                    grad[:, ::5, :] = -0.0
                    if length >= 33:
                        grad[:, 1, :] = 256.0
                        grad[:, 32, :] = -256.0
                    torch.gather(x, 1, select.unsqueeze(-1).expand_as(x)).backward(grad)
                    exact_index.set_prefetch_shift(False)
                    accepted = exact_index.shift_backward(grad, select)
                    exact_index.set_prefetch_shift(True)
                    candidate = exact_index.shift_backward(grad, select)
                    bits = torch.int16 if dtype == torch.bfloat16 else torch.int32
                    equal = torch.equal(candidate.view(bits), x.grad.view(bits)) and torch.equal(candidate.view(bits), accepted.view(bits))
                    result["cases"].append({"dtype": str(dtype), "length": length, "pattern": pattern, "equal": equal})
            print(f"[shift] {dtype}: boundary cases complete", flush=True)
        result["status"] = "passed" if all(case["equal"] for case in result["cases"]) else "failed"
        result["summary"] = {"cases": len(result["cases"]), "equal": sum(case["equal"] for case in result["cases"])}
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        exact_index.set_prefetch_shift(False)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"status": result["status"], "summary": result.get("summary"), "error": result.get("error")}))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
