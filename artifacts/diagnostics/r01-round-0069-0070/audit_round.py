"""Audit recorded final campaign cases; no training or new evaluation."""
from pathlib import Path
import json, statistics
ROOT = Path(__file__).resolve().parents[3]
R = ROOT / 'artifacts/retraining/gpu-retrain-r01'
OUT = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
accepted = read(R / 'iteration-0042/expertium/E33-F48-F50/result.json')
rows = []
for iteration in ('0069', '0070'):
    folder = R / ('iteration-' + iteration)
    protocol = read(folder / 'protocol.json')
    comparison = read(folder / 'comparison.json')['comparisons']['expertium']
    blocks = comparison['replicated_blocks']
    track = comparison['track']
    paths = [folder / 'expertium' / track] + [p.parent for p in sorted((folder / 'replicates').glob('r*/expertium/*/result.json'))]
    cases = []
    for path in paths:
        d = read(path / 'result.json')
        background = read(path.parents[1] / 'background-load.json')
        assert background['total_cpu_percent'] <= 10, str(path)
        if 'python_gpu_processes' in background:
            assert background['python_gpu_processes'] == 0, str(path)
        power = [json.loads(line) for line in (path / 'gpu_power.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        clocks = [float(sample['clocks.sm']) for sample in power if sample.get('clocks.sm') not in (None, 'N/A', '[N/A]')]
        parity = {key: d[key] == accepted[key] for key in ('initial_parameter_sha256', 'trained_parameter_sha256', 'validation_input_sha256', 'evaluation')}
        cases.append(dict(background_cpu_percent=background['total_cpu_percent'], other_python_gpu_processes=background.get('python_gpu_processes'), path=path.relative_to(ROOT).as_posix(), status=d['status'], exact=all(parity.values()), parity=parity,
                          reviews_per_second=d['reviews_per_second'], peak_reserved_mib=d['peak_reserved_mib'],
                          allocator_retries=d['allocator_stats']['num_alloc_retries'], ooms=d['allocator_stats']['num_ooms'],
                          sm_clock_median=statistics.median(clocks) if clocks else None,
                          experiment_git_commit=d['provenance']['git_commit'], git_dirty=d['provenance']['git_dirty'],
                          source_archive=d['provenance']['source_archive']))
    assert all(c['status']=='complete' and c['exact'] and not c['ooms'] and not c['allocator_retries'] for c in cases)
    assert protocol['round_iteration_budget']==23 and protocol['round_iteration_index']==int(iteration)-47
    assert protocol['round_relative_loss_limit']==0.001
    assert blocks['final'], 'Frozen look must be resolved before final audit'
    row=dict(iteration=iteration, blocks=blocks['n'], mean_percent=blocks['mean_percent'], lower_bound_percent=blocks['lower_bound_percent'],
             decision=blocks['decision'], all_cases_exact=True, all_quiet_gates_passed=True, allocator_retries=sum(c['allocator_retries'] for c in cases),
             ooms=sum(c['ooms'] for c in cases), peak_reserved_values_mib=sorted(set(c['peak_reserved_mib'] for c in cases)),
             sm_clock_median_values=sorted(set(c['sm_clock_median'] for c in cases)), cases=cases,
             round_iteration_budget=23, round_iteration_index=protocol['round_iteration_index'], loss_ceiling=0.001,
             background=read(folder / 'background-load.json'))
    rows.append(row)
(OUT / 'round-audit.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
print(json.dumps([dict(iteration=r['iteration'],blocks=r['blocks'],mean_percent=r['mean_percent'],decision=r['decision'],cases=len(r['cases']),exact=r['all_cases_exact'],allocator_retries=r['allocator_retries'],ooms=r['ooms']) for r in rows]))
