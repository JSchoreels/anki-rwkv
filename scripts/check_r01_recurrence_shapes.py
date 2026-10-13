"""Diagnostic: exact row-tiled WKV forward/backward time per real E33 split shape, by rows/thread and tile.

Also checks bitwise equality of outputs and input gradients against rows=4/tile=32 (production)."""
import json, os, sys
from pathlib import Path

from argparse import ArgumentParser
p=ArgumentParser()
p.add_argument("--output", type=Path, required=True)
args=p.parse_args()
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / args.output / "result.json"
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from scripts.r01_workflow import configure_runtime
configure_runtime(ROOT, 1)
import torch
from rwkv.model import nvrtc
from scripts.gpu_row_tiles import row_tiled_backend
from expertium.model.rwkv_ops import RWKV7_WKV

nvrtc.register()
nvrtc.set_stream_launch(True)
dev = torch.device("cuda:0")
shapes = [(742, 256), (869, 256), (512, 256), (36177, 2), (7764, 4), (3359, 10), (318, 26), (695, 68), (4240, 12)]
configs = [(4, False, False), (4, True, False), (4, True, True), (8, True, False), (4, True, "wave")]
result = {"shapes": shapes, "configs": configs, "rows": []}


def inputs(B, T, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    def n(*s):
        return torch.randn(*s, generator=g, device=dev)
    r = n(B, T, 1, 32).bfloat16()
    k = torch.nn.functional.normalize(n(B, T, 1, 32), dim=-1).bfloat16()
    v = torch.nn.functional.normalize(n(B, T, 1, 32), dim=-1).bfloat16()
    w = torch.exp(-torch.exp(-0.5 - torch.nn.functional.softplus(-n(B, T, 1, 32))))
    a = torch.sigmoid(n(B, T, 1, 32)).bfloat16()
    kd = torch.nn.functional.normalize(n(B, T, 1, 32), dim=-1).bfloat16()
    skip = torch.rand(B, T, generator=g, device=dev) < 0.3
    skip[:, 0] = False
    return [r, k * a, v, w, a, kd], skip, n(B, T, 1, 32).bfloat16()


for B, T in shapes:
    tensors, skip, gout = inputs(B, T, B * 1000 + T)
    reference = None
    for rows, prefetch, compact in configs:
        tile = 32
        with row_tiled_backend(rows=rows, tile=tile, exact=True, next_step_prefetch=prefetch, compact_previous=compact is True, wave_aware=compact == "wave"):
            leaves = [t.detach().clone().requires_grad_(True) for t in tensors]
            for _ in range(2):
                out = RWKV7_WKV.apply(*leaves, skip)
                out.backward(gout)
            torch.cuda.synchronize()
            ef = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            fwd = bwd = 0.0
            reps = 5
            for _ in range(reps):
                for leaf in leaves:
                    leaf.grad = None
                ef[0].record()
                out = RWKV7_WKV.apply(*leaves, skip)
                ef[1].record()
                out.backward(gout)
                ef[2].record()
                torch.cuda.synchronize()
                fwd += ef[0].elapsed_time(ef[1])
                bwd += ef[1].elapsed_time(ef[2])
            values = [out.detach()] + [leaf.grad for leaf in leaves]
        if reference is None:
            reference = values
            equal = True
        else:
            equal = all(x.dtype == y.dtype and torch.equal(x.view(torch.int16 if x.dtype == torch.bfloat16 else torch.int32),
                                                           y.view(torch.int16 if y.dtype == torch.bfloat16 else torch.int32))
                        for x, y in zip(values, reference))
        row = {"B": B, "T": T, "rows": rows, "tile": tile, "prefetch": prefetch, "compact": compact, "fwd_ms": round(fwd / reps, 4), "bwd_ms": round(bwd / reps, 4), "bitwise_equal_to_r4_t32": equal}
        result["rows"].append(row)
        print(json.dumps(row), flush=True)
from scripts.experiment_provenance import process_provenance
result.update(status="passed" if all(r["bitwise_equal_to_r4_t32"] for r in result["rows"]) else "failed", provenance=process_provenance(), torch=torch.__version__)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
