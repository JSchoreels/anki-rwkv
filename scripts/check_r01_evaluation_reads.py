"""F60 paired held-out evaluation screen; training estimator/protocol remain unchanged."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch._C._set_graph_executor_optimize(False)
    from scripts.benchmark_r01_components import COHORT
    from scripts.benchmark_gpu_trial import benchmark_models
    from scripts.compare import evaluate
    from scripts.gpu_r01_training import evaluation_job, tensor_hash
    from scripts.gpu_batch1 import split_frame
    from rwkv.data_processing import get_rwkv_data
    from scripts.trial_batches import preparation_pool
    from rwkv.model import nvrtc
    from scripts.experiment_provenance import process_provenance
    cohort=json.loads(COHORT.read_text(encoding='utf-8'))
    row=max(cohort['stages']['100']['evaluation_rows'],key=lambda row:row['available_reviews'])
    section=split_frame(get_rwkv_data(ROOT/cohort['dataset_path'],row['user_id']),row,'validation').iloc[:32768]
    with preparation_pool(6) as pool:
        futures=[pool.submit(evaluation_job,section.iloc[i:i+1024],row['user_id'],2025+row['user_id']+row['validation_range'][0]+i) for i in range(0,len(section),1024)]
        batches=[future.result() for future in futures]
    nvrtc.register()
    master,base,dtype=benchmark_models({'MODEL':'expertium','DTYPE':'bfloat16','SEED':2025,'WKV_BACKEND':'nvrtc'},torch.device('cuda'))
    checkpoint=ROOT/'artifacts/retraining/gpu-retrain-r01/iteration-0042/expertium/E33-F48-F50/master.pth'
    master.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cuda'))
    base=None
    digest=hashlib.sha256()
    for batch in batches:
        for tensor in (batch.start,batch.labels,batch.label_review_th):
            digest.update(tensor.numpy().tobytes())
    def items():
        return ({'batch':batch.to('cuda')} for batch in batches)
    with torch.no_grad():
        probe=master.get_loss(batches[0].to('cuda'))
        scalar_dtypes=[str(v.dtype) for v in (probe.ahead_avg,probe.ahead_raw_avg,probe.imm_avg)]
        stack_dtype=str(torch.stack((probe.ahead_avg,probe.ahead_raw_avg,probe.imm_avg)).dtype)
    probe=None
    evaluate(master,items())
    records=[]
    for repeat in range(4):
        for option in ((False,True) if repeat%2==0 else (True,False)):
            torch.cuda.synchronize()
            start=time.perf_counter()
            values=evaluate(master,items(),deferred_metric_reads=option)
            torch.cuda.synchronize()
            records.append({'repeat':repeat,'deferred_metric_reads':option,'seconds':time.perf_counter()-start,'values':values})
    reference=records[0]['values']
    record={'status':'passed' if all(r['values']==reference for r in records) else 'failed',
            'scope':'one held-out user fixed first32768reviews; fp32 accepted0042 master; same transferred inputs; existing finite checks/counts and Python accumulation order',
            'user_id':row['user_id'],'reviews':len(section),'batches':len(batches),'validation_input_sha256':digest.hexdigest(),
            'trained_parameter_sha256':tensor_hash(master.state_dict()),'checkpoint':str(checkpoint.relative_to(ROOT)),
            'runs':records,'scalar_dtypes':scalar_dtypes,'stack_dtype':stack_dtype,'provenance':process_provenance()}
    (a.output/'result.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':record['status'],'runs':[{k:r[k] for k in ('repeat','deferred_metric_reads','seconds')} for r in records]}))
    if record['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()
