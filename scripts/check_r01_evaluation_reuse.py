"""Verify F67 proof against saved full/replica checkpoints, and reject missing/tampered proof."""
import argparse
from copy import deepcopy
import json
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    import torch
    from scripts.run_gpu_batch1 import tensor_hash
    from scripts.r01_evaluation_reuse import prove_reuse, REPORT_FIELDS, PROTOCOL_FIELDS, CONTEXT_FIELDS, RUNTIME_FIELDS
    from scripts.experiment_provenance import process_provenance
    iteration=ROOT/'artifacts/retraining/gpu-retrain-r01/iteration-0070'
    relative=Path('expertium/E33-F48-F50-F34g')
    full_dir=iteration/relative
    candidate_dir=iteration/'replicates/r2'/relative
    full=json.loads((full_dir/'result.json').read_text(encoding='utf-8'))
    candidate=json.loads((candidate_dir/'result.json').read_text(encoding='utf-8'))
    protocol=json.loads((iteration/'protocol.json').read_text(encoding='utf-8'))
    checkpoint_checks=[]
    for directory,report in ((full_dir,full),(candidate_dir,candidate)):
        actual=tensor_hash(torch.load(directory/'master.pth',map_location='cpu',weights_only=True))
        checkpoint_checks.append({'directory':str(directory.relative_to(ROOT)),'hash':actual,'passed':actual==report['trained_parameter_sha256']})
    historical=prove_reuse(candidate,protocol,full_dir,protocol,candidate_checkpoint=candidate_dir/'master.pth')
    # Historical records cannot establish missing driver/NVRTC context. Never fill it retrospectively.
    # A clearly marked fixture tests the positive API branch with explicit supplied context.
    fixture=args.output/'explicit-context-fixture'
    fixture.mkdir()
    context={key:'TEST FIXTURE ONLY; not historical metadata' for key in CONTEXT_FIELDS}
    context.update(deterministic_algorithms=True,prepared_input_sha256=full['validation_input_sha256'],
                   runtime={key:'TEST FIXTURE ONLY; not historical metadata' for key in RUNTIME_FIELDS})
    full=deepcopy(full);full['evaluation_context']=context
    candidate=deepcopy(candidate);candidate['evaluation_context']=context
    (fixture/'result.json').write_text(json.dumps(full,indent=2)+'\n',encoding='utf-8')
    shutil.copy2(full_dir/'master.pth',fixture/'master.pth')
    def prove(report, proto):
        return prove_reuse(report,proto,fixture,protocol,candidate_checkpoint=candidate_dir/'master.pth')
    proof=prove(candidate,protocol)
    negative=[]
    for field in REPORT_FIELDS:
        broken=deepcopy(candidate)
        broken.pop(field)
        negative.append({'missing_report_field':field,'refused':prove(broken,protocol)['status']=='refused'})
    for field in PROTOCOL_FIELDS:
        broken=deepcopy(protocol)
        broken.pop(field)
        negative.append({'missing_protocol_field':field,'refused':prove(candidate,broken)['status']=='refused'})
    for field in ('trained_parameter_sha256','validation_input_sha256','implementation_sha256'):
        broken=deepcopy(candidate);broken[field]='0'*64
        negative.append({'unequal_field':field,'refused':prove(broken,protocol)['status']=='refused'})
    negative.append({'missing_checkpoint':True,'refused':prove_reuse(candidate,protocol,fixture,protocol)['status']=='refused'})
    # F74 T1 additions: other options, other scope, incomplete runtime manifest, chained (already reused) origin.
    broken=deepcopy(candidate);broken['options']={**broken['options'],'recurrence_prefetch':not broken['options'].get('recurrence_prefetch',False)}
    negative.append({'unequal_options':True,'refused':prove(broken,protocol)['status']=='refused'})
    broken=deepcopy(candidate);broken['smoke_only']=True
    negative.append({'smoke_candidate_full_origin':True,'refused':prove(broken,protocol)['status']=='refused'})
    for key in ('driver_version','gpu_name'):
        broken=deepcopy(candidate);broken['evaluation_context']['runtime'][key]=None
        negative.append({'incomplete_runtime':key,'refused':prove(broken,protocol)['status']=='refused'})
    chained=args.output/'chained-origin-fixture';chained.mkdir()
    reused=deepcopy(full);reused['evaluation_reuse']={'status':'reused'}
    (chained/'result.json').write_text(json.dumps(reused,indent=2)+'\n',encoding='utf-8');shutil.copy2(full_dir/'master.pth',chained/'master.pth')
    negative.append({'chained_origin':True,'refused':prove_reuse(candidate,protocol,chained,protocol,candidate_checkpoint=candidate_dir/'master.pth')['status']=='refused'})
    for field in CONTEXT_FIELDS:
        broken=deepcopy(candidate);broken['evaluation_context'].pop(field)
        negative.append({'missing_context_field':field,'refused':prove(broken,protocol)['status']=='refused'})
    passed=all(row['passed'] for row in checkpoint_checks) and proof['status']=='reusable' and all(row['refused'] for row in negative) and historical['status']=='refused'
    metric_equal=proof.get('per_user')==candidate['per_user'] and proof.get('evaluation')==candidate['evaluation']
    record={'status':'passed' if passed and metric_equal else 'failed','explicit_context_fixture_proof':proof,'historical_proof':historical,'checkpoint_checks':checkpoint_checks,
            'negative_checks':negative,'saved_replica_metrics_exact':metric_equal,'evaluation_actually_skipped':False,
            'scope':'0070 full/r2 final checkpoints verified; historical reuse refused because runtime manifest missing; positive explicit-context fixture and negative guards (F74: options, scope, runtime manifest, chained origin); runner integration is exercised separately by the F74 smoke',
            'provenance':process_provenance()}
    (args.output/'result.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':record['status'],'proof_status':proof['status'],'negative_checks':len(negative),'metric_equal':metric_equal,'evaluation_actually_skipped':False}))
    if record['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()
