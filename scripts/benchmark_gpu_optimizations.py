"""Benchmark four GPU optimization families at fixed per-model batch sizes."""
from argparse import ArgumentParser
from pathlib import Path
import gc, json, sys, traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import pandas as pd
import torch
from rwkv import data_processing as processing
from scripts.benchmark_gpu_trial import fresh_checkpoint, run_variant
from scripts.trial_batches import preparation_pool
from scripts.gpu_power import limited_gpu
from scripts.experiment_provenance import process_provenance

VARIANTS={
    'reference':{},
    'preparation-cpu':{'PREPARATION':'cpu'},
    'preparation-gpu':{'PREPARATION':'gpu'},
    'preparation-gpu-packed':{'PREPARATION':'gpu-packed'},
    'wkv-tile16':{'CHECKPOINT_LENGTH':16},
    'wkv-tile8':{'CHECKPOINT_LENGTH':8},
    'cast-fusion':{'FAST_COPIES':True},
    'raw-launch':{'RAW_POINTERS':True},
    'cuda-graph':{'CUDA_GRAPH':True},
    'fused-adamw':{'FUSED_ADAMW':True},
}


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=('baseline','expertium'),required=True)
    parser.add_argument('--variant',required=True)
    parser.add_argument('--steps',type=int,default=64)
    parser.add_argument('--warmup',type=int,default=6)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--overrides',default='{}')
    args=parser.parse_args()
    provenance=process_provenance()
    overrides=json.loads(args.overrides)
    if args.variant not in VARIANTS and args.variant not in ('combined','combined-gpu-packed','reference-repeat'):
        parser.error('Unknown variant')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.72,0)
    chunks,workers=(128,4) if args.model=='baseline' else (256,6)
    output=ROOT/'artifacts/benchmarks/perf-optimizations'/args.model/args.variant
    output.mkdir(parents=True,exist_ok=True)
    config={'MODEL':args.model,'DTYPE':'bfloat16','WKV_BACKEND':'nvrtc','CHUNK_REVIEWS':128,
            'BATCH_CHUNKS':chunks,'PREFETCH_WORKERS':workers,'PREFETCH_BATCHES':workers*2,
            'REVIEWS_PER_USER':0,'SEED':2025,**VARIANTS.get(args.variant,{}),**overrides}
    report={'model':args.model,'variant':args.variant,'config':config,'torch':str(torch.__version__),
            'provenance':provenance,
            'gpu':torch.cuda.get_device_name(),'warmup_updates':args.warmup,'measured_updates':args.steps,
            'initialization':'fresh weights and optimizer, seed 2025',
            'data':'user 17 reviews 4097-12288 repeated; same input seeds and labels',
            'timing':'complete updates including preparation waits, transfers, forward/backward and optimizer; excludes cold setup/file loads'}
    try:
        if config.get("CUDA_GRAPH"):
            from rwkv.model import nvrtc
            from scripts.check_gpu_optimizations import check_copies_and_graphs
            nvrtc.register()
            nvrtc.set_raw_pointer_launch(True)
            report["graph_verification"] = check_copies_and_graphs(graph=True, model_names=(args.model,))
        source=processing.get_rwkv_data(ROOT/'data/anki-revlogs-10k',17)
        common=source.iloc[4096:4096+8192].copy();del source
        needed=(args.warmup+args.steps+1)*chunks*128
        frame=pd.concat([common]*((needed+len(common)-1)//len(common)),ignore_index=True)
        checkpoint=fresh_checkpoint(args.model)
        with limited_gpu(0,0,output),preparation_pool(workers) as pool:
            result=run_variant(args.variant,frame,config,checkpoint,output,args.warmup,args.steps,pool,profile=args.profile)
        report.update(status='complete',result=result)
    except Exception as exc:
        report.update(status='failed',error=str(exc),error_type=type(exc).__name__)
        traceback.print_exc()
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f"{args.model}/{args.variant}: {report['status']}",flush=True)
    return 0 if report['status']=='complete' else 1


if __name__=='__main__':
    raise SystemExit(main())
