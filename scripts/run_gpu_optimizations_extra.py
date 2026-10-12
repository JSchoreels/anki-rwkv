"""Finish capture retest and packed feature benchmarks, preserving earlier results."""
from datetime import datetime, timezone
from pathlib import Path
import json, math, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/benchmarks/perf-optimizations'


def save(report):
    temporary=OUT/'comparison.tmp'
    temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(OUT/'comparison.json')


def main():
    report=json.loads((OUT/'comparison.json').read_text())
    report['status']='running-final-checks';save(report)
    experiments=[('baseline','cuda-graph',{}),('baseline','combined',{'CUDA_GRAPH':True,'FAST_COPIES':True}),
                 ('baseline','preparation-gpu-packed',{}),('expertium','preparation-gpu-packed',{})]
    if '--packed-combined-only' in sys.argv:
        experiments=[('expertium','combined-gpu-packed',{'CUDA_GRAPH':True,'PREPARATION':'gpu-packed'})]
    for model,variant,overrides in experiments:
        folder=OUT/model/variant;folder.mkdir(parents=True,exist_ok=True)
        if variant=='cuda-graph' and (folder/'result.json').exists():
            (folder/'first-attempt-result.json').write_bytes((folder/'result.json').read_bytes())
            (folder/'first-attempt-console.log').write_bytes((folder/'console.log').read_bytes())
        command=[sys.executable,str(ROOT/'scripts/benchmark_gpu_optimizations.py'),'--model',model,'--variant',variant]
        if overrides:
            command+=['--overrides',json.dumps(overrides)]
        print(f'Starting final {model}/{variant}',flush=True)
        started=time.time()
        with (folder/'console.log').open('w') as log:
            try:
                completed=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=240)
                code=completed.returncode
            except subprocess.TimeoutExpired:
                code=-1
        item=(json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists()
              else {'model':model,'variant':variant,'status':'failed','error':'Timed out or missing result'})
        item.update(process_exit_code=code,elapsed_wall_seconds=time.time()-started)
        report['results'].append(item);save(report)
        result=item.get('result',{})
        if item['status']=='complete':
            assert len(result['update_seconds'])==64
            assert all(math.isfinite(v) and v>0 for v in result['update_seconds'])
            assert math.isfinite(result['last_loss']) and math.isfinite(result['last_gradient_norm'])
            print(f"Finished final {model}/{variant}: {result['reviews_per_second']:,.0f} reviews/s",flush=True)
        else:
            print(f"Final {model}/{variant} failed: {item.get('error','')[:160]}",flush=True)
    report.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat())
    save(report)
    print('All final cases saved',flush=True)


if __name__=='__main__':
    main()
