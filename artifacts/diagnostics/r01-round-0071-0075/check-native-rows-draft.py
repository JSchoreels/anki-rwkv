"""F65e native row selection transport parity against accepted F35/F50 kernels."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    from scripts.experiment_provenance import process_provenance
    provenance=process_provenance();os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from rwkv.model import nvrtc
    from rwkv.model.native_launch import load
    from rwkv.model.exact_index import ExactIndexSelect,ExactPaddedIndexSelect,kernel,set_prefetch_rows
    torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=False
    nvrtc.register();nvrtc.set_stream_launch(True);set_prefetch_rows(True);build=load()
    handle=int(kernel(torch.bfloat16)['rows_backward_prefetch'].kernel.ptr)
    generator=torch.Generator(device='cuda');generator.manual_seed(20261013);cases=[]
    for rows,c,padding,nondefault in ((r,c,p,s) for r in (1,129,65536) for c in (1,16,32) for p in (False,True) for s in (False,True)):
        source=torch.randn((rows,c),device='cuda',generator=generator).to(torch.bfloat16)
        index=torch.randint(rows,(rows*2,),device='cuda',generator=generator)
        if padding:index[::5]=-1
        grad=torch.randn((rows*2,c),device='cuda',generator=generator).to(torch.bfloat16)
        if padding:grad[index<0]=0
        stream=torch.cuda.Stream() if nondefault else torch.cuda.default_stream();results=[]
        for native in (False,True):
            value=source.detach().clone().requires_grad_(True);stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                out=torch.ops.rwkv_native.rows(value,index,handle,padding) if native else (ExactPaddedIndexSelect if padding else ExactIndexSelect).apply(value,index)
                out.backward(grad)
            stream.synchronize();results.append((out.detach().clone(),value.grad.detach().clone()))
        equal=[torch.equal(a.view(torch.int16),b.view(torch.int16)) for a,b in zip(*results)]
        cases.append({'rows':rows,'channels':c,'padding':padding,'nondefault_stream':nondefault,'component_exact':equal,'bitwise_equal':all(equal),'padding_gradient_zero':bool((grad[index<0]==0).all().item()) if padding else None})
    record={'status':'passed' if all(c['bitwise_equal'] for c in cases) else 'failed','cases':cases,'build':build,'provenance':provenance,'scope':'36 BF16 output/input-gradient bit comparisons, normal/padded indices and default/nondefault streams, accepted rows_backward_prefetch kernel and unchanged stable sort/searchsorted. Padding upstream gradients explicitly zero; no speed claim.'}
    (args.output/'result.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');print(json.dumps({'status':record['status'],'cases':len(cases)}))
    if record['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
