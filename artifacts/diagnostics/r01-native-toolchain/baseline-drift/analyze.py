"""Read-only historical/current baseline audit; launches no GPU work."""
import json
from pathlib import Path
from scripts.r01_metrics import case_metrics

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'artifacts/retraining/gpu-retrain-r01'
OUT = Path(__file__).resolve().parent
paths = {'historical': BASE / 'iteration-0042/expertium/E33-F48-F50',
         'current': BASE / 'iteration-0072/replicates/r1/expertium/E33-F48-F50'}
reports = {k: json.loads((p / 'result.json').read_text(encoding='utf-8')) for k, p in paths.items()}
proof_fields = ['options', 'config', 'review_presentations', 'updates', 'training_users', 'evaluation_users',
                'initial_parameter_sha256', 'trained_parameter_sha256', 'validation_input_sha256', 'evaluation', 'torch', 'cuda', 'gpu']
metrics = {k: case_metrics(p) for k, p in paths.items()}
for k, profile in [('historical', BASE / 'iteration-0042/profiles/expertium-E33-F48-F50/result.json'),
                   ('current', OUT / 'current-profile/result.json')]:
    phases = json.loads(profile.read_text(encoding='utf-8'))['phases']
    phase = next(v for name, v in phases.items() if name.startswith('training'))
    for key in ['nvidia_utilization.gpu', 'worker_pool_cpu_percent', 'nvidia_memory.used', 'process_tree_rss_bytes']:
        metrics[k][key] = phase['counters'][key]['mean']
    metrics[k]['gpu_copy_seconds'] = reports[k]['gpu_copy_milliseconds'] / 1000
result = {'status': 'recorded evidence; causal explanation pending',
          'scope': 'same-configuration historical control versus current control; unmatched dates/source/runtime conditions',
          'equality_checks': {key: reports['historical'].get(key) == reports['current'].get(key) for key in proof_fields},
          'case_paths': {k: str(p.relative_to(ROOT)) for k, p in paths.items()},
          'metrics': metrics, 'experiment_provenance': {k: r['provenance'] for k, r in reports.items()},
          'pending': ['Matched instrumentation/event-lifetime audit preserving existing host-wall estimator',
                      'Default CUDA kernel/source performance audit with same numerical behavior',
                      'Historical thread environment and GPU clocks were not fully recorded; do not infer them from current values'],
          'new_gpu_runs': 0}
(OUT / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'all_equal': all(result['equality_checks'].values()),
                  'training_gap_seconds': metrics['current']['training_seconds'] - metrics['historical']['training_seconds'],
                  'status': result['status']}))
