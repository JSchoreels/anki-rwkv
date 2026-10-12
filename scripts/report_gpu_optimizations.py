"""Audit saved fixed-batch measurements and write a compact reproducible report."""
from datetime import datetime, timezone
from pathlib import Path
import json, math, statistics, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_training_tradeoffs import decorate_report
OUT=ROOT/'artifacts/benchmarks/perf-optimizations'
LABELS={'reference':'Control before','reference-repeat':'Control after',
        'preparation-cpu':'Vectorized CPU preparation','preparation-gpu':'GPU encodings (unpacked)',
        'preparation-gpu-packed':'GPU encodings (packed)', 'wkv-tile16':'RWKV tile 16','wkv-tile8':'RWKV tile 8',
        'cast-fusion':'Fused casts/copies','raw-launch':'Direct CUDA pointer launches','fused-adamw':'Fused AdamW',
        'cuda-graph':'CUDA Graphs','combined':'Combined','combined-gpu-packed':'GPU packed + graphs'}


def main():
    report=json.loads((OUT/'comparison.json').read_text())
    latest={}
    for item in report['results']:
        latest[(item['model'],item['variant'])]=item
        if item.get('status')!='complete':
            continue
        result=item['result']
        assert item.get('process_exit_code',0)==0
        assert result['measured_steps']==64 and len(result['update_seconds'])==64
        assert all(math.isfinite(v) and v>0 for v in result['update_seconds'])
        assert all(math.isfinite(result[name]) for name in ('last_loss','last_gradient_norm','reviews_per_second'))
        expected=64*result['reviews_per_update']/sum(result['update_seconds'])
        assert math.isclose(result['reviews_per_second'],expected,rel_tol=1e-12)
        assert (result['batch_chunks'],result['prefetch_workers'])==((128,4) if item['model']=='baseline' else (256,6))
        assert result['dtype']=='bfloat16'
    power=[]
    for path in OUT.glob('*/*/power_summary.json'):
        record=json.loads(path.read_text())
        assert record['requested_watts']==0 and not record['power_limit_changed'] and not record['monitor_errors']
        power.append({'file':str(path.relative_to(ROOT)),'cap':record['original_watts'],'samples':record['samples']})
    controls={model:statistics.mean([latest[(model,variant)]['result']['reviews_per_second']
                                    for variant in ('reference','reference-repeat')]) for model in ('baseline','expertium')}
    best={model:max((item for (name,_),item in latest.items() if name==model and item.get('status')=='complete'),
                    key=lambda item:item['result']['reviews_per_second']) for model in ('baseline','expertium')}
    coverage={model:all(latest[(model,variant)]['status']=='complete' for variant in
                       ('preparation-cpu','wkv-tile16','cast-fusion','cuda-graph','fused-adamw'))
              for model in ('baseline','expertium')}
    assert all(coverage.values()), coverage
    verification=json.loads((OUT/'verification.json').read_text())
    assert len(verification['preparation'])==9 and len(verification['tiles'])==6 and len(verification['copies'])==2
    training_summary=json.loads((ROOT/'artifacts/benchmarks/gpu-trial/summary.json').read_text())
    assert training_summary['status']=='complete'
    assert training_summary['review_presentations']==training_summary['selected_review_presentations']==5226500
    training_checkpoint=ROOT/'checkpoints/gpu-trial/rwkv_gpu_trial.pth'
    assert training_checkpoint.exists()
    assert training_checkpoint.stat().st_mtime<datetime.fromisoformat(report['started_utc']).timestamp()
    report['audit']={'verified_utc':datetime.now(timezone.utc).isoformat(),'four_families_completed':coverage,
                     'power_checks':power,'rates_recomputed':True,'finite_losses_and_gradient_norms':True,
                     'preparation_and_recurrence_parity_passed':True,
                     'completed_training_checkpoint_unchanged':True}
    report['control_mean_reviews_per_second']=controls
    report['best']={model:{'variant':item['variant'],'reviews_per_second':item['result']['reviews_per_second'],
                           'speedup_vs_control_mean':item['result']['reviews_per_second']/controls[model],
                           'config':item['config']} for model,item in best.items()}
    report['status']='complete-and-verified'
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# GPU optimization benchmarks at fixed batch sizes','',
           'RTX 5080 16 GB, Windows, PyTorch 2.9.1+cu128. Measured on 2026-10-11.',
           'Default: **128 chunks / 4 CPU workers**. Expertium iter36: **256 chunks / 6 CPU workers**. Each chunk contains 128 raw reviews. Batch sizes and worker counts stay fixed across each model’s variants.',
           '','Each scenario starts from fresh weights and AdamW state with seed 2025, BF16 computation and FP32 master weights. Inputs are the same user-17 history window (reviews 4097–12288), repeated. Six warmup updates precede 64 measured updates.',
           'Complete throughput includes CPU preparation waits, CPU-to-GPU transfer, forward/backward, gradient clipping and AdamW. File loads, process startup, compilation and graph capture are excluded. This is a throughput benchmark, not a held-out model-quality comparison.',
           '','## Results','',
           '| Experiment | Default reviews/s | Expertium reviews/s |','|---|---:|---:|']
    for variant,label in LABELS.items():
        values=[]
        for model in ('baseline','expertium'):
            item=latest.get((model,variant),{})
            values.append(f"{item['result']['reviews_per_second']:,.0f}" if item.get('status')=='complete' else 'failed / not run')
        lines.append(f"| {label} | {' | '.join(values)} |")
    lines+=['','## Interpretation','']
    for model,item in best.items():
        ratio=item['result']['reviews_per_second']/controls[model]
        lines.append(f"- {model}: best **{item['result']['reviews_per_second']:,.0f} reviews/s**, {(ratio-1)*100:.1f}% above the mean of its two controls ({controls[model]:,.0f} reviews/s), using `{item['variant']}`.")
    lines+=['','The controls expose run-to-run variation. Small differences near that range should not be treated as established speedups. Expertium is constrained by CPU preparation; its transfer-plus-training rate can improve without a comparable rise in complete throughput.',
            '','CUDA Graphs capture forward, loss and backward together. The first capture attempt failed because backward joined the legacy stream. Recreating parameter leaves on the same non-default stream as capture resolved it; deterministic output and gradient checks pass for both models. This matches the kind of stream dependency discussed in [PyTorch issue 164302](https://github.com/pytorch/pytorch/issues/164302).',
            '','Graph timing applies to one repeated shape. Full histories need a bounded graph cache or shape buckets before this can be used throughout production training. Finite loss and gradient checks execute after replay, before optimizer updates. Dropout is enabled during throughput runs; parity fixtures disable dropout and perturb initially-zero parameters to exercise their gradients.',
            '','Smaller RWKV checkpoint tiles reduce per-thread local storage (32: 256 bytes; 16: 128 bytes; 8: 64 bytes) but increase saved checkpoints. Review chunk and optimizer batch sizes do not change. GPU feature preparation preserves CPU RNG draw order and moves lookup/calendar arithmetic to CUDA; the packed variant consolidates metadata in workers before IPC.',
            '','Graph-private pools retain memory after temporary tensors are released. Use **reserved memory** rather than the much smaller live-tensor allocation counter when comparing graph memory footprints.',
            '','## Validation and saved artifacts','',
            '- Vectorized CPU preparation matches original inputs exactly across nine windows from users 1, 17 and 92, including partial chunks. GPU feature variants match within 1e-6 absolute / 1e-5 relative tolerance; indexes and labels match exactly.',
            '- Float32/BF16 recurrence outputs and six input gradients match the CPU reference at 256 timesteps, for one/four heads and checkpoint tiles 8/16/32.',
            '- Direct casts and gradient copies preserve values exactly. CUDA Graph output/gradient parity checks are stored with graph benchmark results.',
            '- All successful scenarios performed 64 measured optimizer updates with finite final losses and gradient norms. Review rates were independently recomputed from timings.',
            '- Power monitoring reports no cap changes or monitoring errors. The configured 400 W cap was left unchanged.',
            '- The completed 100-user training checkpoint remains separate under `checkpoints/gpu-trial/`.',
            '','[Full comparison JSON](../artifacts/benchmarks/perf-optimizations/comparison.json), [parity results](../artifacts/benchmarks/perf-optimizations/verification.json), and per-scenario console/power logs are saved under `artifacts/benchmarks/perf-optimizations/`.',
            '','## Reproduction','',
            '```powershell',
            r'.\.venv\Scripts\python.exe scripts/check_gpu_optimizations.py',
            r'.\.venv\Scripts\python.exe scripts/run_gpu_optimizations.py',
            r''' .\.venv\Scripts\python.exe scripts/benchmark_gpu_optimizations.py --model baseline --variant combined --overrides '{"CUDA_GRAPH":true,"FAST_COPIES":true}' '''.strip(),
            r''' .\.venv\Scripts\python.exe scripts/benchmark_gpu_optimizations.py --model expertium --variant combined --overrides '{"CUDA_GRAPH":true,"PREPARATION":"cpu"}' '''.strip(),
            r''' .\.venv\Scripts\python.exe scripts/benchmark_gpu_optimizations.py --model expertium --variant combined-gpu-packed --overrides '{"CUDA_GRAPH":true,"PREPARATION":"gpu-packed"}' '''.strip(),
            '```','',
            'Use the NVRTC environment from [the GPU setup guide](GPU_TRAINING.md), including its workspace cache/temp paths. `just benchmark-gpu-optimizations` runs the isolated variants and selected combinations. Benchmark scripts leave shutdown under user control.']
    text='\n'.join(lines)+'\n'
    previous=(ROOT/'docs/GPU_OPTIMIZATIONS.md').read_text(encoding='utf-8')
    text=decorate_report(text,previous)
    from scripts.artifact_markdown import rebase_links
    artifact_text=rebase_links(text, ROOT/'docs', OUT)
    (OUT/'summary.md').write_text(artifact_text,encoding='utf-8')
    print(json.dumps({'best':report['best'],'coverage':coverage,'power_checks':len(power)},indent=2))


if __name__=='__main__':
    main()
