"""Exactly 0060-0062, inherited count, frozen looks; stop on failure."""
from pathlib import Path
import json, subprocess, sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.r01_workflow import run_logged
BASE = Path('artifacts/diagnostics/r01-round-0060-0062')
RUNS = Path('artifacts/retraining/gpu-retrain-r01')
def run(args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)
for iteration, mode, suffix, mechanism in (
    ('0060', 4, 'd', 'forward per-row value loads'),
    ('0061', 5, 'e', 'forward scalar and per-row value loads'),
    ('0062', 6, 'f', 'recomputation per-row value loads without scalar prefetch'),
):
    out = RUNS/f'iteration-{iteration}'
    track = f'E33-F48-F50-F58{suffix}'
    comparison_path = ROOT/out/'comparison.json'
    if comparison_path.exists():
        prior = json.loads(comparison_path.read_text(encoding='utf-8'))['comparisons']['expertium'].get('replicated_blocks', {})
        if prior.get('final'):
            print('[resume]', iteration, 'completed blocks retained', flush=True)
            continue
    if not (ROOT/out/'protocol.json').exists():
        run(['scripts/r01_iterate.py', '--iteration', iteration, '--previous', f'{int(iteration)-1:04d}', '--agent', 'codex', '--candidate', track,
             '--set', f'recurrence_load_pipeline={mode}',
             '--explanation', f'Pipeline {mechanism} before current shuffle work, preserving arithmetic and reduction order. Test independently on accepted 0042; omit unproven scalar F58 after inconclusive P3b attribution.',
             '--evidence', 'P3b 20 alternating real-batch pairs: F58 mean -0.226%, descriptive interval [-1.35,+0.90]%; distinct row-value load schedule passes 216 synthetic cases and five real batches.',
             '--check-reason', f'Verify {mechanism} preserves all updates, RNG and evaluation hashes', '--preflight', BASE/'preflight.json',
             '--check', BASE/f'synthetic-{mode}/result.json', '--check', BASE/f'real-{mode}/result.json'])
    else:
        run(['scripts/r01_iterate.py', '--iteration', iteration, '--previous', f'{int(iteration)-1:04d}', '--agent', 'codex', '--candidate', track, '--from-step', 'report'])
    plan = json.loads((ROOT/RUNS/f'plans/iteration-{iteration}-codex.json').read_text(encoding='utf-8'))
    control = plan['control_tracks']['expertium']
    for look in (2, 4, 6):
        run_logged('replicates', [sys.executable, 'scripts/r01_replicates.py', 'run', '--output', str(out), '--sequence', ','.join([control,track,control,track]), '--reason', f'Contemporaneous frozen look {look}: {mechanism}; quiet-machine paired blocks'], BASE/f'{iteration}-look-{look}.log', root=ROOT, heartbeat_seconds=60)
        run(['scripts/report_gpu_r01_next.py', '--output', out, '--brief'])
        b = json.loads((ROOT/out/'comparison.json').read_text(encoding='utf-8'))['comparisons']['expertium']['replicated_blocks']
        print('[blocks]', iteration, json.dumps({k:v for k,v in b.items() if k not in ('blocks','runs')}), flush=True)
        if b['decision'] != 'extend':
            break
    else:
        raise RuntimeError('Unexpected extension beyond frozen cap')
