"""Diagnostic: exact rows=4/tile=32 WKV backward compiled with register caps; time and bitwise check."""
import json, os, sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2])
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from scripts.r01_workflow import configure_runtime
configure_runtime(ROOT, 1)
import torch
from rwkv.model import nvrtc
import scripts.gpu_row_tiles as tiles
from scripts.gpu_row_tiles import row_tiled_backend, source_options
from expertium.model.rwkv_ops import RWKV7_WKV

nvrtc.register()
nvrtc.set_stream_launch(True)
dev = torch.device("cuda:0")
cp = nvrtc._cupy
base_compile = tiles.compile_module


def capped(cap):
    def compile_module(device, dtype, rows, tile, exact=False):
        key = (0, dtype, rows, tile, exact, cap)
        if key not in tiles._modules:
            source, options = source_options(dtype, exact)
            scalar = "float" if dtype == torch.float32 else "__nv_bfloat16"
            names = tuple(f"rwkv_exact::{n}_kernel<{tile}, {rows}, {scalar}>" for n in ("forward", "backward"))
            extra = (f"--maxrregcount={cap}",) if cap else ()
            raw = cp.RawModule(code=source, options=options + extra, name_expressions=names)
            tiles._modules[key] = tuple(raw.get_function(n) for n in names)
        return tiles._modules[key]
    return compile_module


def inputs(B, T, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    n = lambda *s: torch.randn(*s, generator=g, device=dev)
    r = n(B, T, 1, 32).bfloat16()
    k = torch.nn.functional.normalize(n(B, T, 1, 32), dim=-1).bfloat16()
    v = torch.nn.functional.normalize(n(B, T, 1, 32), dim=-1).bfloat16()
    w = torch.exp(-torch.exp(-0.5 - torch.nn.functional.softplus(-n(B, T, 1, 32))))
    a = torch.sigmoid(n(B, T, 1, 32)).bfloat16()
    kd = torch.nn.functional.normalize(n(B, T, 1, 32), dim=-1).bfloat16()
    skip = torch.rand(B, T, generator=g, device=dev) < 0.3
    skip[:, 0] = False
    return [r, k * a, v, w, a, kd], skip, n(B, T, 1, 32).bfloat16()


def bits(x):
    return x.view(torch.int16 if x.dtype == torch.bfloat16 else torch.int32)


result = []
for B, T in [(512, 256), (742, 256), (869, 256), (36177, 2), (3359, 10)]:
    tensors, skip, gout = inputs(B, T, B * 1000 + T)
    reference = None
    for cap in (None, 64, 72, 56, 48):
        tiles.compile_module = capped(cap)
        try:
            f, b = tiles.compile_module(dev, torch.bfloat16, 4, 32, True)
            attrs = b.attributes
            with row_tiled_backend(rows=4, tile=32, exact=True):
                leaves = [t.detach().clone().requires_grad_(True) for t in tensors]
                for _ in range(2):
                    RWKV7_WKV.apply(*leaves, skip).backward(gout)
                torch.cuda.synchronize()
                times = []
                for _ in range(7):
                    for leaf in leaves:
                        leaf.grad = None
                    out = RWKV7_WKV.apply(*leaves, skip)
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    out.backward(gout)
                    e1.record()
                    torch.cuda.synchronize()
                    times.append(e0.elapsed_time(e1))
                values = [out.detach()] + [leaf.grad for leaf in leaves]
        finally:
            tiles.compile_module = base_compile
        if reference is None:
            reference = values
        equal = all(torch.equal(bits(x), bits(y)) for x, y in zip(values, reference))
        row = {"B": B, "T": T, "maxrregcount": cap, "regs": attrs["num_regs"], "local_bytes": attrs["local_size_bytes"],
               "bwd_ms_median": round(sorted(times)[3], 4), "bitwise_equal_to_uncapped": equal}
        result.append(row)
        print(json.dumps(row), flush=True)
OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
