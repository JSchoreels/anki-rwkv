"""SM21 kernel check: WKV-7 started from a carried state S0, returning the final state S_T (docs/SMALL_MODEL.md,
"Design 2026-10-16: training with carried state (SM21)", work item (a)).

Kernels under test (rwkv/model/wkv_state.py): the generic NVRTC forward (`rwkv.model.nvrtc.forward_with_state`) and
the exact row-tiled forward (`scripts.gpu_row_tiles`, inside `row_tiled_backend(rows=4, tile=32, exact=True)`, the
trainer's `row_tiles_exact: 4`), each with the stock backward kernel. Reference: `wkv_with_state` (CPU, float32, the
prototype's reference) and the same recurrence in float64.

--cpu (no GPU, no CUDA context): every source rewrite applies exactly once and compiles with NVRTC (compile only, for
       compute_120), the stock kernel text is unchanged by the rewrite helpers, and the CPU reference started from
       zeros equals `reference_rwkv7` bit for bit, chained halves equal the whole sequence bit for bit, and no gradient
       reaches S0.
default (GPU, a ~5-minute queue ticket; never run outside the queue): per backend, dtype (float32, bfloat16), head
       size (16, 32), timesteps (1 ... 700) and skip pattern:
  Z  zero state: the state kernel with S0 = 0 vs the stock kernel (RWKV7_WKV): output, checkpoints and the six input
     gradients bit for bit (reported; the stock path itself is untouched by construction);
  R  random S0: output and final state vs the float64 reference (and the float32 CPU reference): max |diff|;
  G  input gradients with random S0 vs float64 autograd through the reference: max |diff| / max |reference|;
  C  chain: the sequence cut in two, the second half started from the first half's final state, equals the whole
     sequence bit for bit (output and final state).
Pass: R and G within tolerance, C bitwise, Z bitwise. Prints ALL OK and writes the JSON report.

  .venv/Scripts/python.exe -X utf8 scripts/check_state_carry_kernels.py --cpu --output artifacts/small-model/check-state-carry-kernels-cpu.json
  .venv/Scripts/python.exe -X utf8 scripts/check_state_carry_kernels.py --output artifacts/small-model/check-state-carry-kernels-gpu.json
"""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOLERANCE = {"float32": {"out": 2e-4, "state": 2e-4, "grad": 2e-4},
             "bfloat16": {"out": 3e-2, "state": 2e-4, "grad": 3e-2}}


def reference64(r, k, v, w, a, kd, skip, state0, grad=None):
    """The reference recurrence in float64 on the CPU; with `grad`, also the input gradients (autograd)."""
    import torch
    from rwkv.model.wkv_state import single_timestep
    inputs = [x.detach().cpu().double().requires_grad_(grad is not None) for x in (r, k, v, w, a, kd)]
    skip = skip.cpu()
    B, T, H, K = r.shape
    state = state0.detach().cpu().double().clone()
    outs = []
    with torch.enable_grad():
        for t in range(T):
            out, nxt = single_timestep(*(x[:, t] for x in inputs), state)
            outs.append(out)
            state = torch.where(skip[:, t].view(B, 1, 1, 1), state, nxt)
        out = torch.stack(outs, dim=1)
        grads = None
        if grad is not None:
            grads = torch.autograd.grad(out, inputs, grad.detach().cpu().double())
    return out.detach(), state.detach(), grads


def synthetic(torch, b, t, h, k, skip_mode, dtype, generator, device):
    shape = (b, t, h, k)

    def normal(scale=1.0):
        return (torch.randn(shape, generator=generator, device=device) * scale).to(dtype).contiguous()

    r, kk, v = normal(), normal(0.5), normal()
    w = torch.exp(-torch.exp(torch.randn(shape, generator=generator, device=device) - 1.0)).contiguous()
    a = torch.sigmoid(torch.randn(shape, generator=generator, device=device)).to(dtype).contiguous()
    kd = torch.nn.functional.normalize(torch.randn(shape, generator=generator, device=device), dim=-1).to(dtype).contiguous()
    kk = (kk * a).contiguous()  # as the time mixer: k = k_scale * normalize(k) * a, kd = k_scale * normalize(k)
    if skip_mode == "none":
        skip = torch.zeros((b, t), dtype=torch.bool, device=device)
    elif skip_mode == "random":
        skip = torch.rand((b, t), generator=generator, device=device) < 0.4
    elif skip_mode == "first":
        skip = torch.zeros((b, t), dtype=torch.bool, device=device)
        skip[:, 0] = True
    elif skip_mode == "tail":  # right padding, as in a batch split
        skip = torch.zeros((b, t), dtype=torch.bool, device=device)
        skip[:, t // 2:] = True
    else:
        raise ValueError(skip_mode)
    state0 = (torch.randn((b, h, k, k), generator=generator, device=device) * 0.5).contiguous()
    grad = normal()
    return (r, kk, v, w, a, kd), skip.contiguous(), state0, grad


def cpu_checks(report):
    import torch
    from rwkv.model import nvrtc
    from rwkv.model.rwkv_ops import reference_rwkv7
    from rwkv.model.wkv_state import wkv_with_state
    from scripts import gpu_row_tiles
    from cupy.cuda import nvrtc as compiler
    checks = {}
    fails = []
    # 1. rewrites and compile (NVRTC only, no device)
    compiled = []
    for dtype in (torch.float32, torch.bfloat16):
        scalar = "float" if dtype == torch.float32 else "__nv_bfloat16"
        for head in (32, 16):
            stock = nvrtc.kernel_source(dtype, head)
            tiles_source, tiles_options = gpu_row_tiles.source_options(dtype, exact=True, head=head, state=True)
            variants = (("generic", nvrtc.state_kernel_source(dtype, head),
                         [f"rwkv::rwkv7_wkv_forward_kernel<{tile}, {scalar}>" for tile in (8, 16, 32)],
                         nvrtc._compile_options(dtype)),
                        ("row_tiles_exact", tiles_source,
                         [f"rwkv_exact::{kind}_kernel<32, {rows}, {scalar}>" for kind in ("forward", "backward")
                          for rows in (1, 2, 4, 8)], list(tiles_options)))
            for label, source, names, options in variants:
                started = time.perf_counter()
                program = compiler.createProgram(source, f"{label}.cu", [], [])
                for name in names:
                    compiler.addNameExpression(program, name)
                ok, log = True, ""
                try:
                    compiler.compileProgram(program, [*options, "--gpu-architecture=compute_120"])
                except Exception as error:  # noqa: BLE001 - report the NVRTC log
                    ok, log = False, f"{error}: {compiler.getProgramLog(program)[:2000]}"
                compiled.append({"kernel": label, "dtype": scalar, "head": head, "compiled": ok, "log": log,
                                 "state_pointers": source.count("state0"), "seconds": round(time.perf_counter() - started, 1)})
                fails += [] if ok else [f"compile {label} {scalar} K={head}"]
            # the stock text is what the stock modules compile; the rewrite helpers must not touch it
            if nvrtc.kernel_source(dtype, head) != stock or "state0" in stock:
                fails.append(f"stock text changed {scalar} K={head}")
            stock_tiles, _ = gpu_row_tiles.source_options(dtype, exact=True, head=head)
            if "state0" in stock_tiles:
                fails.append(f"stock row-tile text changed {scalar} K={head}")
    checks["nvrtc_compile"] = compiled
    # 2. CPU reference semantics
    generator = torch.Generator().manual_seed(2026)
    reference = []
    for b, t, h, k, skip_mode in ((3, 1, 4, 16, "none"), (5, 37, 4, 16, "random"), (4, 100, 2, 32, "first"),
                                  (2, 64, 4, 16, "tail")):
        inputs, skip, state0, grad = synthetic(torch, b, t, h, k, skip_mode, torch.float32, generator, "cpu")
        stock = reference_rwkv7(*inputs, skip)
        zero_out, _ = wkv_with_state(*inputs, skip, None)
        zeros_out, _ = wkv_with_state(*inputs, skip, torch.zeros_like(state0))
        full_out, full_state = wkv_with_state(*inputs, skip, state0)
        cut = max(1, t // 3)
        first_out, first_state = wkv_with_state(*(x[:, :cut] for x in inputs), skip[:, :cut], state0)
        second_out, second_state = wkv_with_state(*(x[:, cut:].contiguous() for x in inputs), skip[:, cut:], first_state)
        chained = torch.cat([first_out, second_out], dim=1) if cut < t else first_out
        chained_state = second_state if cut < t else first_state
        r64, s64, _ = reference64(*inputs, skip, state0)
        leaf = state0.clone().requires_grad_(True)
        x = [i.clone().requires_grad_(True) for i in inputs]
        out, _ = wkv_with_state(*x, skip, leaf)
        out.backward(grad)
        row = {"batch": b, "timesteps": t, "heads": h, "head": k, "skip": skip_mode,
               "zero_state_equals_reference_rwkv7": bool(torch.equal(stock, zero_out) and torch.equal(stock, zeros_out)),
               "chain_bitwise": bool(torch.equal(chained, full_out) and torch.equal(chained_state, full_state)),
               "max_abs_vs_float64_out": float((full_out.double() - r64).abs().max()),
               "max_abs_vs_float64_state": float((full_state.double() - s64).abs().max()),
               "no_gradient_into_state0": leaf.grad is None}
        reference.append(row)
        if not (row["zero_state_equals_reference_rwkv7"] and row["chain_bitwise"] and row["no_gradient_into_state0"]
                and row["max_abs_vs_float64_out"] < 1e-4):
            fails.append(f"reference {b}x{t} {skip_mode}")
    checks["cpu_reference"] = reference
    report["checks"] = checks
    return fails


def gpu_checks(report, quick=False):
    import torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=False)
    from expertium.model.rwkv_ops import RWKV7_WKV
    from rwkv.model import nvrtc
    from rwkv.model.wkv_state import RWKV7_WKV_State, wkv_with_state
    from scripts.gpu_row_tiles import row_tiled_backend
    from contextlib import nullcontext
    nvrtc.register()
    nvrtc.set_checkpoint_length(32)
    nvrtc.set_raw_pointer_launch(False)
    nvrtc.set_max_timesteps(8192)
    device = torch.device("cuda:0")
    report["device"] = torch.cuda.get_device_name(0)
    generator = torch.Generator(device=device)
    rows, fails = [], []
    shapes = [(1, 1), (5, 2), (97, 31), (33, 32), (7, 33), (64, 100), (3, 383), (2, 700)]
    if quick:
        shapes = shapes[:3]
    backends = (("generic", nullcontext), ("row_tiles_exact4", lambda: row_tiled_backend(rows=4, tile=32, exact=True)))
    case = 0
    for backend, context in backends:
        with context():
            for dtype in (torch.float32, torch.bfloat16):
                name = "float32" if dtype == torch.float32 else "bfloat16"
                tolerance = TOLERANCE[name]
                for head in (16, 32):
                    for b, t in shapes:
                        for skip_mode in ("none", "random", "first", "tail"):
                            case += 1
                            generator.manual_seed(20261018 + case)
                            inputs, skip, state0, grad = synthetic(torch, b, t, 4, head, skip_mode, dtype, generator, device)
                            leaves = [x.clone().requires_grad_(True) for x in inputs]
                            # Z: zero state vs the stock kernel
                            stock_out, stock_states = (torch.ops.rwkv.rwkv7_wkv_forward_float(*inputs, skip) if dtype == torch.float32
                                                       else torch.ops.rwkv.rwkv7_wkv_forward_bfloat16(*inputs, skip))
                            zero = torch.zeros_like(state0)
                            z_out, z_states, z_final = nvrtc.forward_with_state(*inputs, skip, zero)
                            stock_grads = nvrtc.backward(*inputs, skip, stock_states, grad)
                            z_leaves = [x.clone().requires_grad_(True) for x in inputs]
                            z_auto, _ = RWKV7_WKV_State.apply(*z_leaves, skip, zero)
                            z_auto.backward(grad)
                            s_leaves = [x.clone().requires_grad_(True) for x in inputs]
                            RWKV7_WKV.apply(*s_leaves, skip).backward(grad)
                            zero_bitwise = (torch.equal(stock_out, z_out) and torch.equal(stock_states, z_states)
                                            and all(torch.equal(x.grad, y.grad) for x, y in zip(z_leaves, s_leaves))
                                            and all(torch.equal(x, y.grad) for x, y in zip(stock_grads, s_leaves)))
                            # R, G: random state
                            out, final = RWKV7_WKV_State.apply(*leaves, skip, state0)
                            out.backward(grad)
                            r64, s64, g64 = reference64(*inputs, skip, state0, grad)
                            r32, s32 = wkv_with_state(*(x.cpu() for x in inputs), skip.cpu(), state0.cpu())
                            out_err = float((out.detach().cpu().double() - r64).abs().max())
                            state_err = float((final.cpu().double() - s64).abs().max())
                            out_err32 = float((out.detach().cpu().float() - r32.float()).abs().max())
                            state_err32 = float((final.cpu() - s32).abs().max())
                            grad_err = max(float((x.grad.cpu().double() - g).abs().max()) / max(float(g.abs().max()), 1.0)
                                           for x, g in zip(leaves, g64))
                            # C: chain
                            cut = max(1, t // 3)
                            first_out, _, first_final = nvrtc.forward_with_state(
                                *(x[:, :cut].contiguous() for x in inputs), skip[:, :cut].contiguous(), state0)
                            if cut < t:
                                second_out, _, second_final = nvrtc.forward_with_state(
                                    *(x[:, cut:].contiguous() for x in inputs), skip[:, cut:].contiguous(), first_final)
                                chained, chained_final = torch.cat([first_out, second_out], dim=1), second_final
                            else:
                                chained, chained_final = first_out, first_final
                            chain_bitwise = torch.equal(chained, out.detach()) and torch.equal(chained_final, final)
                            ok = (out_err <= tolerance["out"] * max(1.0, float(r64.abs().max()))
                                  and state_err <= tolerance["state"] * max(1.0, float(s64.abs().max()))
                                  and grad_err <= tolerance["grad"] and chain_bitwise and zero_bitwise)
                            rows.append({"backend": backend, "dtype": name, "head": head, "batch": b, "timesteps": t,
                                         "skip": skip_mode, "zero_state_bitwise_vs_stock": zero_bitwise,
                                         "max_abs_out_vs_float64": out_err, "max_abs_state_vs_float64": state_err,
                                         "max_abs_out_vs_cpu_float32": out_err32, "max_abs_state_vs_cpu_float32": state_err32,
                                         "max_rel_grad_vs_float64": grad_err, "chain_bitwise": chain_bitwise, "pass": ok})
                            if not ok:
                                fails.append(f"{backend} {name} K={head} {b}x{t} {skip_mode}")
    torch.cuda.synchronize()
    report["checks"] = {"synthetic": rows,
                        "summary": {key: max(row[key] for row in rows) for key in
                                    ("max_abs_out_vs_float64", "max_abs_state_vs_float64", "max_rel_grad_vs_float64")}
                        | {"zero_state_bitwise_all": all(row["zero_state_bitwise_vs_stock"] for row in rows),
                           "chain_bitwise_all": all(row["chain_bitwise"] for row in rows), "cases": len(rows)}}
    return fails


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="CPU-only part: rewrites, NVRTC compile, reference")
    parser.add_argument("--quick", action="store_true", help="GPU: three shapes only")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    os.environ.setdefault("CUPY_CACHE_DIR", str(ROOT / "artifacts/cache/cupy-cache"))
    started = time.perf_counter()
    report = {"mode": "cpu" if args.cpu else "gpu", "torch": torch.__version__, "tolerance": TOLERANCE}
    try:
        from scripts.experiment_provenance import process_provenance
        report["provenance"] = process_provenance()
    except Exception as error:  # noqa: BLE001 - provenance is informative here
        report["provenance_error"] = repr(error)
    fails = cpu_checks(report) if args.cpu else gpu_checks(report, args.quick)
    report["fails"] = fails
    report["passed"] = not fails
    report["seconds"] = round(time.perf_counter() - started, 1)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print("ALL OK" if not fails else f"FAILED: {fails[:10]}")
    print(f"report {output} ({report['seconds']} s)")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
