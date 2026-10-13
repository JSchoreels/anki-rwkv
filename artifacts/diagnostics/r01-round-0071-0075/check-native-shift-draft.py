"""F65d native shift transport: exact arithmetic, mutation/reuse and bounded cache."""
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
    from rwkv.model.exact_index import ExactShiftGather,kernel,set_shared_orders
    torch.use_deterministic_algorithms(True);torch.utils.deterministic.fill_uninitialized_memory=False
    nvrtc.register();nvrtc.set_stream_launch(True);set_shared_orders(True);build=load()
    handle=int(kernel(torch.bfloat16)['shift_backward'].kernel.ptr)
    generator=torch.Generator(device='cuda');generator.manual_seed(20261013)
    torch.ops.rwkv_native.shift_cache_stats(True);torch.cuda.reset_peak_memory_stats()
    cases=[];max_entries=0;max_storage=0;max_orders=0;seen={};reused_addresses=0
    def compare(x,select,stream,label):
        nonlocal max_entries,max_storage,max_orders
        grad=torch.randn(x.shape,device='cuda',generator=generator).to(torch.bfloat16);results=[]
        for native in (False,True):
            value=x.detach().clone().requires_grad_(True);stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                out=torch.ops.rwkv_native.shift(value,select,handle,True) if native else ExactShiftGather.apply(value,select)
                out.backward(grad)
            stream.synchronize();results.append((out.detach().clone(),value.grad.detach().clone()))
        exact=[torch.equal(a.view(torch.int16),b.view(torch.int16)) for a,b in zip(*results)]
        entries,storage,orders=torch.ops.rwkv_native.shift_cache_stats(False)
        max_entries=max(max_entries,entries);max_storage=max(max_storage,storage);max_orders=max(max_orders,orders)
        cases.append({'label':label,'shape':list(x.shape),'component_exact':exact,'bitwise_equal':all(exact),'cache_entries':entries})
    for b,t,c in ((1,1,1),(3,17,16),(37,256,32),(742,256,32)):
        for repeated in (False,True):
            for nondefault in (False,True):
                x=torch.randn((b,t,c),device='cuda',generator=generator).to(torch.bfloat16)
                select=torch.zeros((b,t),device='cuda',dtype=torch.int64) if repeated else torch.randint(t,(b,t),device='cuda',generator=generator)
                stream=torch.cuda.Stream() if nondefault else torch.cuda.default_stream()
                compare(x,select,stream,'shape-and-stream');compare(x,select,stream,'same-owner-cache-hit')
    x=torch.randn((2,33,32),device='cuda',generator=generator).to(torch.bfloat16)
    select=torch.zeros((2,33),device='cuda',dtype=torch.int64)
    for version in range(24):
        select.copy_(torch.randint(33,(2,33),device='cuda',generator=generator))
        compare(x,select,torch.cuda.default_stream(),'mutated-same-owner-'+str(version))
    # A small select view intentionally retains a larger shared transfer storage: expose its real bytes.
    bulk=torch.zeros(262144,device='cuda',dtype=torch.int64);select=bulk[:66].view(2,33)
    compare(x,select,torch.cuda.default_stream(),'bulk-storage-view')
    bulk_stats=list(torch.ops.rwkv_native.shift_cache_stats(False));del bulk,select
    for index in range(160):
        select=torch.randint(33,(2,33),device='cuda',generator=generator)
        pointer=select.data_ptr()
        if pointer in seen:reused_addresses+=1
        seen[pointer]=index
        compare(x,select,torch.cuda.default_stream(),'fresh-owner-'+str(index));del select
    cache_stats=list(torch.ops.rwkv_native.shift_cache_stats(False))
    record={'status':'passed' if all(c['bitwise_equal'] for c in cases) and max_entries<=64 else 'failed',
            'cases':cases,'build':build,'provenance':provenance,'cache':{'capacity':64,'max_entries':max_entries,'max_unique_selection_storage_bytes':max_storage,'max_order_bytes':max_orders,'bulk_view_stats':bulk_stats,'final_stats':cache_stats,'fresh_selection_reused_addresses':reused_addresses},
            'max_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'scope':'BF16 forward and every input gradient; stable accepted sort/searchsorted and kernel; default/nondefault streams, same-owner reuse, in-place version changes, bulk-storage views and 160 fresh batches. No throughput claim.'}
    (args.output/'result.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':record['status'],'cases':len(cases),'cache':record['cache']}))
    if record['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
