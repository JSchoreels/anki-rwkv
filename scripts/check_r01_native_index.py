"""F65d same-kernel shift-gather/padded row-select transport parity fixture (BF16, accepted F35/F38 settings)."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def bits_equal(a,b):
    view=torch.int16 if a.dtype==torch.bfloat16 else torch.int32
    return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().view(view),b.contiguous().view(view))

def main():
    global torch
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    from scripts.experiment_provenance import process_provenance
    provenance=process_provenance();os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    torch.use_deterministic_algorithms(True)
    from rwkv.model import nvrtc, exact_index
    from rwkv.model.native_launch import load
    from rwkv.model.exact_index import ExactShiftGather,ExactPaddedIndexSelect,ExactIndexSelect,kernel,shift_order
    nvrtc.register();nvrtc.set_stream_launch(True);exact_index.set_shared_orders(True);exact_index.set_prefetch_rows(True)
    build=load();functions=kernel(torch.bfloat16)
    shift_handle,rows_handle=(int(getattr(functions[n],'kernel',functions[n]).ptr) for n in ('shift_backward','rows_backward_prefetch'))
    generator=torch.Generator(device='cuda');cases=[]
    shapes=[(1,1),(1,7),(3,33),(16,64),(37,256),(256,40),(2048,8)]
    for index,(B,T) in enumerate(shapes):
        generator.manual_seed(20261014+index)
        x=torch.randn((B,T,32),device='cuda',generator=generator).to(torch.bfloat16)
        positions=torch.arange(T,device='cuda').expand(B,T)
        back=torch.randint(0,4,(B,T),device='cuda',generator=generator)
        select=(positions-back).clamp(min=0).contiguous()
        if index%3==2: select[:,:]=0  # every source on one destination per row
        grad=torch.randn((B,T,32),device='cuda',generator=generator).to(torch.bfloat16)
        stream=torch.cuda.Stream() if index%2 else torch.cuda.default_stream();results=[]
        for native in (False,True):
            value=x.detach().clone().requires_grad_(True);stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                if native:
                    order,starts=shift_order(select)
                    out=torch.ops.rwkv_native.shift_gather(value,select,order,starts,shift_handle)
                else:
                    out=ExactShiftGather.apply(value,select)
                out.backward(grad)
            stream.synchronize();results.append((out.detach().clone(),value.grad.detach().clone()))
        equal=[bits_equal(a,b) for a,b in zip(*results)]
        cases.append({'op':'shift_gather','B':B,'T':T,'nondefault_stream':bool(index%2),'component_exact':equal,'bitwise_equal':all(equal)})
    row_cases=[(1,1,0.0,False),(64,300,0.2,False),(4096,30000,0.3,True),(50,60000,0.9,True),(7420,7420,0.0,False),(9000,47595,0.5,True)]
    for index,(rows,count,padding,clamped) in enumerate(row_cases):
        generator.manual_seed(20261114+index)
        x=torch.randn((rows,32),device='cuda',generator=generator).to(torch.bfloat16)
        gather=torch.randint(0,rows,(count,),device='cuda',generator=generator)
        gather[torch.rand(count,device='cuda',generator=generator)<padding]=-1
        if index==3: gather[:count//2]=0
        grad=torch.randn((count,32),device='cuda',generator=generator).to(torch.bfloat16)
        stream=torch.cuda.Stream() if index%2 else torch.cuda.default_stream();results=[]
        for native in (False,True):
            value=x.detach().clone().requires_grad_(True);stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                index_in=gather.clamp(min=0) if clamped else gather
                if native:
                    out=torch.ops.rwkv_native.padded_rows(value,index_in,rows_handle)
                else:
                    out=ExactIndexSelect.apply(value,index_in) if clamped else ExactPaddedIndexSelect.apply(value,index_in)
                out.backward(grad)
            stream.synchronize();results.append((out.detach().clone(),value.grad.detach().clone()))
        equal=[bits_equal(a,b) for a,b in zip(*results)]
        cases.append({'op':'padded_rows','rows':rows,'sources':count,'padding_fraction':padding,'pre_clamped':clamped,
                      'nondefault_stream':bool(index%2),'component_exact':equal,'bitwise_equal':all(equal)})
    record={'status':'passed' if all(c['bitwise_equal'] for c in cases) else 'failed','cases':cases,'build':build,'provenance':provenance,
            'torch':torch.__version__,
            'scope':'BF16 output and input-gradient bit comparisons of native F65d ops against the Python F26 shift (F38 shared orders) and F27/F50 row-select (F35 prefetch) paths; same CuPy kernel handles; no training'}
    (args.output/'result.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');print(json.dumps({'status':record['status'],'cases':len(cases),'exact':sum(c['bitwise_equal'] for c in cases)}))
    if record['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
