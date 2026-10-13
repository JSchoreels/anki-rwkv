"""Synthetic exact checks for F34 bonus products and F51/F52 row loading."""
from pathlib import Path
import sys, json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.r01_workflow import configure_runtime
configure_runtime(Path(__file__).resolve().parents[1], 1)
import torch
from rwkv.model.exact_glue import ExactBonusProducts
from rwkv.model.exact_index import ExactPaddedIndexSelect, FusedPaddedIndexSelect, PreGatheredIndexSelect
from scripts.experiment_provenance import process_provenance

def same(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))

def main():
    from argparse import ArgumentParser
    p = ArgumentParser(); p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    torch.use_deterministic_algorithms(True); torch.manual_seed(2025)
    records = []
    for dtype in (torch.bfloat16, torch.float32):
        for n in (1, 31, 32, 257, 2048):
            r, k = [torch.randn(n, 1, 1, 32, device='cuda', dtype=dtype) for _ in range(2)]
            b = torch.randn(1, 1, 1, 32, device='cuda', dtype=dtype)
            b.flatten()[:6] = torch.tensor([0., -0., 1., -1., 1e-10, -1e-10], device='cuda', dtype=dtype)
            dy = torch.randn_like(r)
            a = [v.clone().requires_grad_() for v in (r, b, k)]
            z = a[0] * a[1] * a[2]; ga = torch.autograd.grad(z, a, dy)
            c = [v.clone().requires_grad_() for v in (r, b, k)]
            y = ExactBonusProducts.apply(*c); gc = torch.autograd.grad(y, c, dy)
            records.append(dict(mode='bonus', dtype=str(dtype), n=n, exact=same(z,y) and all(same(x,y) for x,y in zip(ga,gc))))
            index = torch.randint(-1, 17, (n,), device='cuda')
            grad = torch.randn(n, 32, device='cuda', dtype=dtype)
            grad[index < 0] = 0
            x = torch.randn(17, 32, device='cuda', dtype=dtype, requires_grad=True)
            ref = ExactPaddedIndexSelect.apply(x, index); gx, = torch.autograd.grad(ref, x, grad)
            for mode, fn in [('padded', FusedPaddedIndexSelect), ('preloaded', PreGatheredIndexSelect)]:
                loaded = x.detach().index_select(0, index.clamp(min=0))
                out = fn.apply(x, index, loaded) if mode == 'preloaded' else fn.apply(x,index)
                gg, = torch.autograd.grad(out, x, grad)
                records.append(dict(mode=mode, dtype=str(dtype), n=n, exact=same(ref,out) and same(gx,gg)))
    result = dict(status='passed' if all(r['exact'] for r in records) else 'failed', cases=records, provenance=process_provenance())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], cases=len(records), failures=[r for r in records if not r['exact']])))
    if result['status'] != 'passed': raise SystemExit(1)

if __name__ == '__main__': main()
