"""F65e native F34 glue parity fixture: native ops against the unfused PyTorch expressions and Python F34 Functions (BF16)."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def bits_equal(a,b):
    view=torch.int16 if a.dtype==torch.bfloat16 else torch.int32
    return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().view(view),b.contiguous().view(view))

def run(fn,sources,grad,stream):
    values=[v.detach().clone().requires_grad_(True) for v in sources];stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out=fn(*values);out.backward(grad)
    stream.synchronize()
    return [out.detach().clone()]+[v.grad.detach().clone() for v in values]

def main():
    global torch
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    from scripts.experiment_provenance import process_provenance
    provenance=process_provenance();os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    torch.use_deterministic_algorithms(True)
    from rwkv.model import nvrtc
    from rwkv.model.native_launch import load
    from rwkv.model.exact_glue import ExactGateGlue,ExactChannelGlue,ExactBonusProducts,kernel
    nvrtc.register();nvrtc.set_stream_launch(True);build=load()
    h={name:int(kernel(torch.bfloat16,name).kernel.ptr) for name in ('gate_forward','gate_backward','bonus_forward','channel_forward')}
    ops={
        'gate':(3,lambda a,b,g:g*(a+b),lambda a,b,g:ExactGateGlue.apply(a,b,g,True),
                lambda a,b,g:torch.ops.rwkv_native.gate_glue(a,b,g,h['gate_forward'],h['gate_backward'])),
        'bonus':(3,lambda r,b,k:r*b*k,lambda r,b,k:ExactBonusProducts.apply(r,b,k),
                 lambda r,b,k:torch.ops.rwkv_native.bonus_products(r,b,k,h['bonus_forward'])),
        'channel':(1,lambda x:torch.square(torch.nn.functional.relu(x)),lambda x:ExactChannelGlue.apply(x),
                   lambda x:torch.ops.rwkv_native.channel_glue(x,h['channel_forward'])),
    }
    generator=torch.Generator(device='cuda');cases=[]
    shapes=[(1,1),(3,33),(37,256),(256,40),(2048,8)]
    for index,(B,T) in enumerate((s for s in shapes for _ in range(2))):
        for name,(arity,reference,python,native) in ops.items():
            generator.manual_seed(20261015+index*7+len(name))
            if name=='bonus':
                sources=[torch.randn((B,T,1,32),device='cuda',generator=generator).to(torch.bfloat16),
                         torch.randn((1,1,1,32),device='cuda',generator=generator).to(torch.bfloat16),
                         torch.randn((B,T,1,32),device='cuda',generator=generator).to(torch.bfloat16)]
                shape=(B,T,1,32)
            else:
                shape=(B,T,128 if name=='channel' else 32)
                sources=[torch.randn(shape,device='cuda',generator=generator).to(torch.bfloat16) for _ in range(arity)]
            if name=='channel' and index%2:
                sources[0].view(-1)[::3]=0  # exact zeros exercise the threshold boundary
            grad=torch.randn(shape,device='cuda',generator=generator).to(torch.bfloat16)
            stream=torch.cuda.Stream() if index%2 else torch.cuda.default_stream()
            results=[run(fn,sources,grad,stream) for fn in (reference,python,native)]
            versus_reference=[bits_equal(a,b) for a,b in zip(results[0],results[2])]
            versus_python=[bits_equal(a,b) for a,b in zip(results[1],results[2])]
            cases.append({'op':name,'B':B,'T':T,'nondefault_stream':bool(index%2),'native_vs_unfused':versus_reference,
                          'native_vs_python_f34':versus_python,'bitwise_equal':all(versus_reference) and all(versus_python)})
    record={'status':'passed' if all(c['bitwise_equal'] for c in cases) else 'failed','cases':cases,'build':build,'provenance':provenance,
            'torch':torch.__version__,
            'scope':'BF16 output and every input-gradient bit comparison of native F65e gate/bonus/channel glue against unfused PyTorch and the Python F34 Functions (fused gate backward); no training'}
    (args.output/'result.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':record['status'],'cases':len(cases),'exact':sum(c['bitwise_equal'] for c in cases)}))
    if record['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
