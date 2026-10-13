"""Eight-way lerp edge shapes, weights, signed zeros and original reductions."""
from pathlib import Path
import sys, json
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.r01_workflow import configure_runtime
configure_runtime(ROOT, 1)
import torch
from rwkv.model.fused_shift import ExactShiftLerp, ExplicitShiftLerp, PartialShiftLerp
from scripts.experiment_provenance import process_provenance


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--partials', action='store_true')
    parser.add_argument('--explicit', action='store_true')
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(2025)
    records = []
    for dtype in (torch.float32, torch.bfloat16):
        for B,T in ((1,1),(2,7),(3,32),(2,128)):
            for edge in (False, True):
                x = torch.randn(B,T,32,device='cuda',dtype=dtype)
                w = torch.randn(8,1,1,32,device='cuda',dtype=dtype)
                if edge:
                    x.flatten()[:6] = torch.tensor([0.,-0.,1.,-1.,1e-10,-1e-10],device='cuda',dtype=dtype)
                    w.flatten()[:8] = torch.tensor([0.,-0.,.5,-.5,1.,-1.,2.,1e-10],device='cuda',dtype=dtype)
                select = torch.randint(T,(B,T),device='cuda')
                dy = torch.randn(8,B,T,32,device='cuda',dtype=dtype)
                a,b = x.clone().requires_grad_(),w.clone().requires_grad_()
                shifted = torch.gather(a,1,select.unsqueeze(-1).expand_as(a))
                y = torch.lerp(a.unsqueeze(0),shifted.unsqueeze(0),b)
                ga = torch.autograd.grad(y,(a,b),dy)
                c,d = x.clone().requires_grad_(),w.clone().requires_grad_()
                z = (PartialShiftLerp if args.partials else ExplicitShiftLerp if args.explicit else ExactShiftLerp).apply(c,select,d)
                gb = torch.autograd.grad(z,(c,d),dy)
                same = lambda u,v: torch.equal(u.contiguous().view(torch.uint8),v.contiguous().view(torch.uint8))
                records.append(dict(dtype=str(dtype),B=B,T=T,edge=edge,forward=same(y,z),gradients=[same(u,v) for u,v in zip(ga,gb)]))
    passed = all(r['forward'] and all(r['gradients']) for r in records)
    result = dict(status='passed' if passed else 'failed',cases=records,provenance=process_provenance())
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],cases=len(records),failed=[r for r in records if not r['forward'] or not all(r['gradients'])])))
    if not passed: raise SystemExit(1)


if __name__ == '__main__': main()
