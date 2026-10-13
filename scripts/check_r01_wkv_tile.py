"""F90 native exact 4-row WKV: tile 16 (and 8) outputs and all six input gradients bitwise against tile 32; timings informative only."""
import argparse
from pathlib import Path
import sys, json, os
sys.path.insert(0, os.getcwd())
os.environ.setdefault('CUPY_CACHE_DIR', os.path.join(os.getcwd(), 'artifacts/cache/cupy-cache'))
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
from scripts.experiment_provenance import process_provenance
provenance=process_provenance()
import torch
from rwkv.model import nvrtc
from rwkv.model.native_launch import load
from scripts.gpu_row_tiles import compile_module
nvrtc.register(); nvrtc.set_stream_launch(True); load()
dev = torch.device('cuda', 0)
handles = {t: tuple(int(f.kernel.ptr) for f in compile_module(dev, torch.bfloat16, 4, t, True, False, False, 0)) for t in (32, 16, 8)}
g = torch.Generator(device='cuda'); g.manual_seed(7)
out = {}
for B, T in ((48, 256), (256, 128), (1024, 32), (4096, 8), (160, 256), (512, 256), (2048, 64), (16384, 8), (1024, 128), (1, 1), (3, 33)):
    def mk(scale=1.0, dtype=torch.bfloat16):
        return (torch.randn((B, T, 1, 32), device='cuda', generator=g) * scale).to(dtype)
    r, k, v = mk(), mk(), mk()
    kk = torch.nn.functional.normalize(mk().float(), dim=-1).to(torch.bfloat16)
    a = torch.sigmoid(mk().float()).to(torch.bfloat16)
    w = torch.exp(-torch.exp(-0.5 - torch.nn.functional.softplus(mk().float())))
    skip = torch.rand((B, T), device='cuda', generator=g) < 0.1
    grad = mk()
    res = {}
    for tile, (fh, bh) in handles.items():
        vals = [x.detach().clone().requires_grad_(True) for x in (r, k, v, w, a, kk)]
        def step():
            o = torch.ops.rwkv_native.wkv(*vals, skip, fh, bh, 4, tile)
            o.backward(grad)
            return o
        for _ in range(3): step()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(20): o = step()
        e1.record(); torch.cuda.synchronize()
        for x in vals: x.grad = None
        o = step(); torch.cuda.synchronize()
        res[tile] = (e0.elapsed_time(e1) / 20, [o.detach().clone()] + [x.grad.detach().clone() for x in vals])
    base = res[32][1]
    eq = {t: all(torch.equal(x.view(torch.int16) if x.dtype == torch.bfloat16 else x.view(torch.int32),
                             y.view(torch.int16) if y.dtype == torch.bfloat16 else y.view(torch.int32))
                 for x, y in zip(base, res[t][1])) for t in (16, 8)}
    out[f'{B}x{T}'] = {'ms': {t: round(res[t][0], 3) for t in res}, 'bitwise_vs_32': eq}
passed=all(all(v['bitwise_vs_32'].values()) for v in out.values())
record={'status':'passed' if passed else 'failed','cases':out,'provenance':provenance,'torch':torch.__version__,
        'scope':'native exact 4-row BF16 WKV forward output and all input gradients, tiles 16/8 vs 32, 10% skipped steps, default stream'}
(args.output/'result.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':record['status'],'cases':len(out)}))
if not passed: raise SystemExit(1)
