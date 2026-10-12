"""Save compact raw comparison using existing R01 parity/metrics helpers."""
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.r01_metrics import case_metrics
from scripts.report_gpu_r01_next import parity
from scripts.profile_gpu_resources import intervals, summarize
from scripts.r01_workflow import write_json

R = ROOT / 'artifacts/retraining/gpu-retrain-r01'
cases = [R / f'iteration-{n}/expertium/E30-F37-F42' for n in ('0033', '0040')]
results = [json.loads((p / 'result.json').read_text()) for p in cases]
metrics = [case_metrics(p) for p in cases]
for metric, case, result in zip(metrics, cases, results):
    metric.update({k: result.get(k) for k in ('case_elapsed_seconds', 'model_setup_seconds', 'pool_prestart_seconds', 'pool_ready_wait_seconds')})
    rows = [json.loads(line) for line in (case / 'resources.jsonl').read_text().splitlines()]
    weights = intervals(rows)
    selected = [(row, w) for row, w in zip(rows, weights) if row['phase'].startswith('training')]
    total_cpu, tree_cpu, used_weights = [], [], []
    for row, w in selected:
        ncores = len(row['cpu_per_core_percent'])
        cpus = [p.get('cpu_percent_one_core_100') for p in row['processes']]
        if ncores and cpus and all(x is not None for x in cpus):
            total_cpu.append(sum(row['cpu_per_core_percent']) / ncores)
            tree_cpu.append(sum(cpus) / ncores)
            used_weights.append(w)
    metric['sampled_total_host_cpu_percent'] = summarize(total_cpu, used_weights)
    metric['sampled_experiment_tree_cpu_percent'] = summarize(tree_cpu, used_weights)
    metric['background_cpu_residual_proxy_percent'] = summarize([a-b for a,b in zip(total_cpu, tree_cpu)], used_weights)
    metric['prelaunch_background'] = json.loads((case.parents[1] / 'background-load.json').read_text())
    profile = json.loads((case.parents[1] / 'profiles/expertium-E30-F37-F42/result.json').read_text())
    metric['profile_workload'] = {k:v for k,v in profile['workload'].items() if not isinstance(v, dict)}

changes = {k: {'reference': metrics[0][k], 'repeat': metrics[1][k],
               'absolute_change': metrics[1][k]-metrics[0][k],
               'percent_change': (metrics[1][k]/metrics[0][k]-1)*100 if metrics[0][k] else None}
           for k in metrics[0] if isinstance(metrics[0][k], (int,float)) and isinstance(metrics[1].get(k), (int,float))}
record = {'reference': str(cases[0].relative_to(ROOT)), 'repeat': str(cases[1].relative_to(ROOT)),
          'metrics': {'reference':metrics[0], 'repeat':metrics[1]}, 'changes':changes,
          'parity': parity(*results, *cases),
          'configuration_equal': {k: results[0].get(k)==results[1].get(k) for k in ('config','options','execution_mode','training_dropout','master_dtype','float32_matmul_precision','cuda_matmul_allow_tf32','torch','cuda','gpu','implementation_sha256')},
          'provenance': [r['provenance'] for r in results],
          'loss_metrics': {k: {'reference':results[0]['evaluation'][k], 'repeat':results[1]['evaluation'][k], 'absolute_change':results[1]['evaluation'][k]-results[0]['evaluation'][k], 'percent_change':(results[1]['evaluation'][k]/results[0]['evaluation'][k]-1)*100} for k in results[0]['evaluation'] if 'log_loss' in k},
          'gradient_tensor_hashes_available_in_full_runs': False,
          'background_interpretation': 'Residual CPU proxy subtracts summed experiment-tree CPU/core-count from sampled total host CPU. Sample windows are not perfectly synchronized; this is descriptive, not per-app attribution. GPU activity is whole-device and cannot isolate desktop activity.',
          'timing_interpretation': 'One repeat under intended desktop use cannot estimate a distribution or establish desktop-load causality; accepted 0033 reference is unchanged.'}
updates = [[json.loads(s) for s in (p/'updates.jsonl').read_text().splitlines()] for p in cases]
record['logged_loss_and_gradient_norm_float_bits_equal'] = all(struct.pack('!d',a[k])==struct.pack('!d',b[k]) for a,b in zip(*updates) for k in ('loss','gradient_norm')) and len(updates[0])==len(updates[1])
write_json(R/'iteration-0040/repeat-comparison.json', record)
print(json.dumps({'changes':changes, 'cpu_summary':[{'total':m['sampled_total_host_cpu_percent']['mean'],'experiment':m['sampled_experiment_tree_cpu_percent']['mean'],'residual_proxy':m['background_cpu_residual_proxy_percent']['mean']} for m in metrics], 'parity':record['parity']['exact_recorded_parity'], 'configuration':record['configuration_equal']}))
