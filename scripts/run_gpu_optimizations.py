"""Run isolated fixed-batch optimization benchmarks and save an incremental report."""
from pathlib import Path
from datetime import datetime, timezone
import json, math, os, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiment_provenance import process_provenance
OUT=ROOT/'artifacts/benchmarks/perf-optimizations'
VARIANTS=['reference','preparation-cpu','preparation-gpu','preparation-gpu-packed','wkv-tile16','wkv-tile8','cast-fusion','raw-launch','fused-adamw','cuda-graph','reference-repeat']


def save(report):
    temporary=OUT/'comparison.tmp'
    temporary.write_text(json.dumps(report,indent=2)+'\n')
    temporary.replace(OUT/'comparison.json')


def rates(report, model):
    return {item['variant']:item['result']['reviews_per_second'] for item in report['results']
            if item['model']==model and item.get('status')=='complete'}


def main():
    provenance=process_provenance()
    OUT.mkdir(parents=True,exist_ok=True)
    # Finish with a margin before the user's one-hour shutdown fallback.
    deadline=time.time()+35*60
    report={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
            'provenance':provenance,
            'batch_sizes':{'baseline':128,'expertium':256},'cpu_workers':{'baseline':4,'expertium':6},
            'precision':'bfloat16 with float32 master weights','power_cap_changed':False,
            'steps':64,'warmup':6,'results':[],'selection_threshold':1.03}
    save(report)
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
    import pandas as pd
    import torch
    from rwkv import data_processing as processing
    from scripts.benchmark_gpu_trial import fresh_checkpoint, run_variant
    from scripts.benchmark_gpu_optimizations import VARIANTS as SETTINGS
    from scripts.trial_batches import preparation_pool
    from scripts.gpu_power import limited_gpu
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.72,0)
    source=processing.get_rwkv_data(ROOT/'data/anki-revlogs-10k',17)
    common=source.iloc[4096:4096+8192].copy();del source
    checkpoints={model:fresh_checkpoint(model) for model in ('baseline','expertium')}
    frames={}
    for model,chunks in report['batch_sizes'].items():
        needed=(6+64+1)*chunks*128
        frames[model]=pd.concat([common]*((needed+len(common)-1)//len(common)),ignore_index=True)
    active_pool=None

    def measure(model,variant,overrides=None):
        if time.time()>deadline-180:
            raise RuntimeError('Time budget reached; saving results before existing shutdown')
        folder=OUT/model/variant;folder.mkdir(parents=True,exist_ok=True)
        print(f'Starting {model}/{variant}',flush=True)
        started=time.time()
        result_file=folder/'result.json'
        if variant=='reference' and result_file.exists():
            item=json.loads(result_file.read_text())
            exit_code=0
            print('Reusing reference already measured in this session',flush=True)
        elif variant=='cuda-graph' or (overrides or {}).get('CUDA_GRAPH'):
            command=[sys.executable,str(ROOT/'scripts/benchmark_gpu_optimizations.py'),
                     '--model',model,'--variant',variant]
            if overrides:
                command+=['--overrides',json.dumps(overrides)]
            with (folder/'console.log').open('w') as log:
                try:
                    completed=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                             timeout=min(300,max(30,deadline-time.time()-120)))
                    exit_code=completed.returncode
                except subprocess.TimeoutExpired:
                    exit_code=-1
            item=(json.loads(result_file.read_text()) if result_file.exists() else
                  {'model':model,'variant':variant,'status':'failed','error':'Timeout or process exited before saving; see console.log'})
        else:
            config={'MODEL':model,'DTYPE':'bfloat16','WKV_BACKEND':'nvrtc','CHUNK_REVIEWS':128,
                    'BATCH_CHUNKS':report['batch_sizes'][model],'PREFETCH_WORKERS':report['cpu_workers'][model],
                    'PREFETCH_BATCHES':report['cpu_workers'][model]*2,'REVIEWS_PER_USER':0,'SEED':2025,
                    **SETTINGS.get(variant,{}),**(overrides or {})}
            item={'model':model,'variant':variant,'config':config,'warmup_updates':6,'measured_updates':64,
                  'provenance':provenance,
                  'initialization':'fresh weights and optimizer, seed 2025',
                  'data':'user 17 reviews 4097-12288 repeated; same input seeds and labels'}
            try:
                with limited_gpu(0,0,folder):
                    result=run_variant(variant,frames[model],config,checkpoints[model],folder,6,64,active_pool)
                item.update(status='complete',result=result);exit_code=0
            except Exception as exc:
                item.update(status='failed',error=str(exc),error_type=type(exc).__name__);exit_code=1
                exc.__traceback__=None
                import gc
                gc.collect();torch.cuda.empty_cache()
            result_file.write_text(json.dumps(item,indent=2)+'\n')
        item['process_exit_code']=exit_code;item['elapsed_wall_seconds']=time.time()-started
        report['results'].append(item);save(report)
        if item['status']=='complete':
            result=item['result']
            assert len(result['update_seconds'])==64 and all(math.isfinite(v) and v>0 for v in result['update_seconds'])
            assert math.isfinite(result['last_loss']) and math.isfinite(result['last_gradient_norm'])
            expected=64*result['reviews_per_update']/sum(result['update_seconds'])
            assert math.isclose(expected,result['reviews_per_second'],rel_tol=1e-12)
            print(f"Finished {model}/{variant}: {result['reviews_per_second']:,.0f} reviews/s; GPU path {result['training_only_reviews_per_second']:,.0f}",flush=True)
        else:
            print(f"Finished {model}/{variant}: {item['status']}: {item.get('error','')[:240]}",flush=True)
        return item

    try:
        for model in ('baseline','expertium'):
            with preparation_pool(report['cpu_workers'][model]) as active_pool:
                for variant in VARIANTS:
                    measure(model,variant)
                available=rates(report,model)
                reference=(available['reference']+available['reference-repeat'])/2
                overrides={}
                prep=max(('preparation-cpu','preparation-gpu','preparation-gpu-packed'),key=lambda name:available.get(name,0))
                if available.get(prep,0)>reference*1.03:
                    overrides['PREPARATION']={'preparation-cpu':'cpu','preparation-gpu':'gpu','preparation-gpu-packed':'gpu-packed'}[prep]
                tile=max(('wkv-tile16','wkv-tile8'),key=lambda name:available.get(name,0))
                if available.get(tile,0)>reference*1.03:
                    overrides['CHECKPOINT_LENGTH']=16 if tile=='wkv-tile16' else 8
                for variant,key in [('cast-fusion','FAST_COPIES'),('raw-launch','RAW_POINTERS'),('fused-adamw','FUSED_ADAMW'),('cuda-graph','CUDA_GRAPH')]:
                    if available.get(variant,0)>reference*1.03:
                        overrides[key]=True
                report.setdefault('combined_configuration',{})[model]=overrides
                if overrides:
                    measure(model,'combined',overrides)
        report['status']='complete'
    except Exception as exc:
        report.update(status='incomplete',error=str(exc))
    report['finished_utc']=datetime.now(timezone.utc).isoformat()
    save(report)
    lines=['# Fixed-batch GPU optimization benchmarks','',
           'Same real-history window, fresh seed 2025, BF16/FP32 master weights, 6 warmup and 64 measured updates.',
           'Complete throughput includes preparation waits, transfers, forward/backward and AdamW. Startup and file loads are excluded.',
           'Default: 128 chunks / 4 workers. Expertium: 256 chunks / 6 workers. Batch sizes and worker counts stay fixed across variants.',
           '','| Model | Experiment | Reviews/s | Transfer + training reviews/s | Peak allocated MiB | Status |',
           '|---|---|---:|---:|---:|---|']
    for item in report['results']:
        result=item.get('result',{})
        lines.append(f"| {item['model']} | {item['variant']} | {result.get('reviews_per_second',0):,.0f} | {result.get('training_only_reviews_per_second',0):,.0f} | {result.get('peak_allocated_mib',0):,.0f} | {item['status']} |")
    lines+=['','CUDA Graph results apply to one repeated shape. General histories require a bounded shape cache or bucketing.',
            'Tile size changes internal recomputation, not review chunks or optimizer batch size.',
            'Preparation and recurrence output/gradient parity are checked separately. Throughput measurements are not model-quality evaluations.',
            '',f"Run status: {report['status']}. Power cap was not changed."]
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
    print(f"Suite {report['status']}; report saved in {OUT}",flush=True)
    return 0 if report['status']=='complete' else 1


if __name__=='__main__':
    raise SystemExit(main())
