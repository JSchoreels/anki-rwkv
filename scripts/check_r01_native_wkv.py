"""Native WKV transport: same-cubin output/gradient bits, skip cases and current streams."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    from scripts.experiment_provenance import process_provenance
    provenance=process_provenance()
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from rwkv.model import nvrtc
    from rwkv.model.native_launch import load
    from scripts.gpu_row_tiles import compile_module,row_tiled_backend
    from scripts.check_gpu_row_tiles import synthetic
    from expertium.model.rwkv_ops import RWKV7_WKV
    from scripts.experiment_provenance import process_provenance
    nvrtc.register()
    nvrtc.set_stream_launch(True)
    build=load()
    generator=torch.Generator(device='cuda')
    cases=[]
    def bits(tensor):
        return tensor.view(torch.int16 if tensor.dtype==torch.bfloat16 else torch.int32)
    for index,(b,t,skip_mode) in enumerate([(b,t,s) for b in (1,5,97) for t in (1,31,32,33,64,256) for s in ('none','all','every-third','random')]):
        generator.manual_seed(20261013+index)
        values,skip,grad=synthetic(torch,b,t,skip_mode,generator)
        # Synthetic fixture supplies BF16 payload/FP32 decay, precisely the real training ABI.
        forward,backward=compile_module(torch.device('cuda',0),torch.bfloat16,4,32,True)
        stream=torch.cuda.Stream() if index%2 else torch.cuda.default_stream()
        results=[]
        for native in (False,True):
            inputs=[v.detach().clone().requires_grad_(True) for v in values]
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream),row_tiled_backend(rows=4,tile=32,exact=True):
                if native:
                    out=torch.ops.rwkv_native.wkv(*inputs,skip,int(forward.kernel.ptr),int(backward.kernel.ptr),4,32)
                else:
                    out=RWKV7_WKV.apply(*inputs,skip)
                out.backward(grad)
            stream.synchronize()
            results.append((out.detach().clone(),*[v.grad.detach().clone() for v in inputs]))
        equal_components=[torch.equal(bits(a),bits(b)) for a,b in zip(*results)]
        equal=all(equal_components)
        cases.append({'batch':b,'timesteps':t,'skip':skip_mode,'nondefault_stream':bool(index%2),'bitwise_equal':equal,
                      'component_exact':equal_components})
    record={'status':'passed' if all(c['bitwise_equal'] for c in cases) else 'failed','cases':cases,
            'scope':'72same-cubin BF16/FP32-decay synthetic forward/backward cases; same current streams and exact sequential arithmetic; no training timing',
            'build':build,'provenance':provenance}
    (args.output/'result.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':record['status'],'cases':len(cases),'exact_cases':sum(c['bitwise_equal'] for c in cases)}))
    if record['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()
