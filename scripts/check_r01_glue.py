"""F34 synthetic bitwise forward/every-input-gradient cases."""
from pathlib import Path
import sys, json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.r01_workflow import configure_runtime
ROOT=Path(__file__).resolve().parents[1]
configure_runtime(ROOT,1)
import torch
from rwkv.model.exact_glue import ExactGateGlue, ExactChannelGlue
from scripts.experiment_provenance import process_provenance

def same(a,b):
    return torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))

def main():
    from argparse import ArgumentParser
    p=ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(2025)
    records=[]
    for dtype in (torch.bfloat16,torch.float32):
        for n in (1,31,32,257,65536):
            values=[torch.randn(n,device='cuda',dtype=dtype) for _ in range(4)]
            edges=torch.tensor([0.,-0.,1.,-1.,1e-10,-1e-10],device='cuda',dtype=dtype)
            for v in values:v[:min(n,6)]=edges[:min(n,6)]
            for mode in ('gate1','gate2','channel'):
                inputs=[v.detach().clone().requires_grad_() for v in values[:3 if mode.startswith('gate') else 1]]
                a=(inputs[2]*(inputs[0]+inputs[1])) if mode.startswith('gate') else torch.square(torch.relu(inputs[0]))
                ga=torch.autograd.grad(a,inputs,values[3])
                other=[v.detach().clone().requires_grad_() for v in inputs]
                b=ExactGateGlue.apply(*other,mode=='gate2') if mode.startswith('gate') else ExactChannelGlue.apply(other[0])
                gb=torch.autograd.grad(b,other,values[3])
                records.append(dict(dtype=str(dtype),n=n,mode=mode,forward=same(a,b),gradients=[same(x,y) for x,y in zip(ga,gb)]))
    result=dict(status='passed' if all(r['forward'] and all(r['gradients']) for r in records) else 'failed',cases=records,provenance=process_provenance())
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],cases=len(records),failed=[r for r in records if not r['forward'] or not all(r['gradients'])])))
    if result['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
