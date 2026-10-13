"""Draft F65c same-ATen/same-cubin GroupNorm transport parity fixture."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    from scripts.experiment_provenance import process_provenance
    provenance=process_provenance();os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from rwkv.model import nvrtc
    from rwkv.model.native_launch import load
    from rwkv.model.exact_group_norm import ExactGroupNorm,kernel
    nvrtc.register();nvrtc.set_stream_launch(True);build=load();functions=kernel(torch.bfloat16)
    handles=[int(functions[name].kernel.ptr) for name in ('gamma_beta_chains','gamma_beta_tree')]
    generator=torch.Generator(device='cuda');cases=[]
    for index,(n,groups,eps) in enumerate((n,g,e) for n in (1,127,128,129,1024,7420) for g in (1,2,4,8) for e in (1e-5,1e-3)):
        generator.manual_seed(20261013+index)
        source=[torch.randn((n,32),device='cuda',generator=generator).to(torch.bfloat16),torch.randn(32,device='cuda',generator=generator).to(torch.bfloat16),torch.randn(32,device='cuda',generator=generator).to(torch.bfloat16)]
        grad=torch.randn((n,32),device='cuda',generator=generator).to(torch.bfloat16)
        stream=torch.cuda.Stream() if index%2 else torch.cuda.default_stream();results=[]
        for native in (False,True):
            values=[v.detach().clone().requires_grad_(True) for v in source];stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                out=torch.ops.rwkv_native.group_norm(*values,groups,eps,*handles) if native else ExactGroupNorm.apply(*values,groups,eps)
                out.backward(grad)
            stream.synchronize();results.append((out.detach().clone(),*[v.grad.detach().clone() for v in values]))
        equal=[]
        for a,b in zip(*results):
            bits=torch.int16 if a.dtype==torch.bfloat16 else torch.int32
            equal.append(a.dtype==b.dtype and torch.equal(a.view(bits),b.view(bits)))
        cases.append({'n':n,'groups':groups,'eps':eps,'nondefault_stream':bool(index%2),'component_exact':equal,'bitwise_equal':all(equal)})
    record={'status':'passed' if all(c['bitwise_equal'] for c in cases) else 'failed','cases':cases,'build':build,'provenance':provenance,'scope':'48 BF16-input/BF16-affine GroupNorm output+allgradient bit comparisons; N<=128 original ATen path, N>128 identical accepted CuPy gamma/beta kernels; no speed claim'}
    (args.output/'result.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');print(json.dumps({'status':record['status'],'cases':len(cases)}))
    if record['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
