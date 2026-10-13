"""F54 synthetic bitwise output/checkpoint/six-gradient kernel check."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--load-pipeline", type=int, default=0, choices=(0, 1, 2, 3, 4, 5, 6))
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--wave-aware", action="store_true")
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists() or not output.resolve().is_relative_to(ROOT / "artifacts"):
        parser.error("Use a new artifact directory")
    import torch
    from rwkv.model import nvrtc
    from scripts.gpu_row_tiles import row_tiled_backend, compile_module
    from scripts.check_gpu_row_tiles import synthetic
    from scripts.experiment_provenance import process_provenance
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    nvrtc.register()
    nvrtc.set_stream_launch(True)
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device)
    cases = []

    def run(inputs, skip, grad, compact):
        with row_tiled_backend(rows=4, tile=32, exact=True, compact_previous=False if args.prefetch or args.wave_aware or args.load_pipeline else compact,
                               next_step_prefetch=compact and args.prefetch, wave_aware=compact and args.wave_aware,
                               load_pipeline=args.load_pipeline if compact else 0):
            out, states = nvrtc.forward(*inputs, skip)
            gradients = nvrtc.backward(*inputs, skip, states, grad)
        return (out, states, *gradients)

    for dtype in (torch.bfloat16, torch.float32):
        for b in ((1, 5, 512) if args.wave_aware else (1, 5, 97)):
            for t in (1, 2, 31, 32, 33, 64, 65, 100, 383):
                for mode in ("none", "all", "every-third", "random"):
                    generator.manual_seed(20261012 + len(cases))
                    inputs, skip, grad = synthetic(torch, b, t, mode, generator)
                    if dtype == torch.float32:
                        inputs = tuple(value.float() for value in inputs)
                        grad = grad.float()
                    a, c = run(inputs, skip, grad, False), run(inputs, skip, grad, True)
                    equal = all(torch.equal(x.view(torch.int16 if x.dtype == torch.bfloat16 else torch.int32),
                                            y.view(torch.int16 if y.dtype == torch.bfloat16 else torch.int32)) for x, y in zip(a, c))
                    cases.append({"dtype": str(dtype), "batch": b, "timesteps": t, "skip": mode, "bitwise_equal": equal})
    attributes = {}
    for dtype in (torch.bfloat16, torch.float32):
        attributes[str(dtype)] = {str(compact): compile_module(device, dtype, 4, 32, True, False if args.load_pipeline else compact,
                                                                 next_step_prefetch=compact and args.prefetch,
                                                                 load_pipeline=args.load_pipeline if compact else 0)[1].attributes
                                  for compact in (False, True)}
    passed = all(c["bitwise_equal"] for c in cases)
    result = {"status": "passed" if passed else "failed", "provenance": process_provenance(), "torch": torch.__version__,
              "load_pipeline": args.load_pipeline, "prefetch": args.prefetch,
              "summary": {"cases": len(cases), "all_bitwise_equal": passed}, "kernel_attributes": attributes, "cases": cases}
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"]))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
