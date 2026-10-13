"""Read recorded evidence only; no new training, timing or evaluations."""
import json, statistics, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
BASE = Path('artifacts/diagnostics/r01-round-0060-0062')
read = lambda p: json.loads((ROOT/p).read_text(encoding='utf-8'))
accepted = read(Path('artifacts/retraining/gpu-retrain-r01/iteration-0042/expertium/E33-F48-F50/result.json'))
audit = []
for iteration in ('0060', '0061', '0062'):
    out = Path(f'artifacts/retraining/gpu-retrain-r01/iteration-{iteration}')
    c = read(out/'comparison.json')['comparisons']['expertium']
    b = c['replicated_blocks']
    assert b['final']
    a_paths = [Path(p[0]) for p in b['pairs']]
    b_paths = [Path(p[1]) for p in b['pairs']]
    paired = BASE/f'{iteration}-paired.json'
    with (ROOT/BASE/f'{iteration}-paired.log').open('w', encoding='utf-8') as log:
        subprocess.run([sys.executable, 'scripts/r01_paired_timing.py', '--a', *map(str,a_paths), '--b', *map(str,b_paths), '--output', str(paired)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    timing = read(paired)
    results_a = [read(p/'result.json') for p in a_paths]
    results_b = [read(p/'result.json') for p in b_paths]
    states = list(timing['machine_state'].values())
    evaluations_exact = all(r['evaluation'] == accepted['evaluation'] and r['validation_input_sha256'] == accepted['validation_input_sha256'] and r['trained_parameter_sha256'] == accepted['trained_parameter_sha256'] for r in results_a+results_b)
    assert evaluations_exact
    audit.append({'iteration': iteration, 'blocks': b['n'], 'mean_percent': b['mean_percent'], 'lower_bound_percent': b['lower_bound_percent'], 'decision': b['decision'],
                  'reference_mean_reviews_per_second': statistics.mean(r['reviews_per_second'] for r in results_a), 'candidate_mean_reviews_per_second': statistics.mean(r['reviews_per_second'] for r in results_b),
                  'paired_wall': timing['wall'], 'allocator_retries': sum(s['allocator_stats']['num_alloc_retries'] for s in states), 'ooms': sum(s['allocator_stats']['num_ooms'] for s in states),
                  'sm_clock_median_values': sorted(set(s.get('busy_sm_clock_median_mhz') for s in states), key=str), 'peak_reserved_values_mib': sorted(set(s['peak_reserved_mib'] for s in states)), 'all_evaluations_exact': evaluations_exact})
(ROOT/BASE/'round-audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
for a in audit:
    print(json.dumps({k:v for k,v in a.items() if k != 'paired_wall'}))
