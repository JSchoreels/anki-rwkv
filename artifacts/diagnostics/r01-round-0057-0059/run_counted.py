"""Sequential authorized 0057-0059 continuation; stop on failure, no automatic retries."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.r01_workflow import run_logged

BASE = Path('artifacts/diagnostics/r01-round-0057-0059')
RUNS = Path('artifacts/retraining/gpu-retrain-r01')

def run(args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)

for iteration, mode, suffix, mechanism in (
    ('0057', 1, 'a', 'recomputation input loads'),
    ('0058', 2, 'b', 'per-row backward gradient loads'),
    ('0059', 3, 'c', 'forward recurrence scalar loads'),
):
    out = RUNS / f'iteration-{iteration}'
    track = f'E33-F48-F50-F58-F58{suffix}'
    comparison_path = ROOT/out/'comparison.json'
    if comparison_path.exists():
        previous = json.loads(comparison_path.read_text(encoding='utf-8'))['comparisons']['expertium'].get('replicated_blocks', {})
        if previous.get('final'):
            print('[resume]', iteration, 'completed blocks retained', flush=True)
            continue
    if iteration != '0057' and not (ROOT/out/'protocol.json').exists():
        run(['scripts/r01_iterate.py', '--iteration', iteration,
             '--previous', f'{int(iteration)-1:04d}', '--agent', 'codex',
             '--candidate', track, '--set', 'recurrence_prefetch=true',
             '--set', f'recurrence_load_pipeline={mode}',
             '--explanation', f'Pipeline {mechanism} ahead of current recurrence work, preserving arithmetic and reduction order. Replace the previous pipeline mode; retain inherited scalar F58 and omit rejected slices.',
             '--evidence', 'Changed F58 mechanism independently passes 216 synthetic BF16/FP32 cases and five real loss/every-gradient batches; modes are alternatives, not no-op replays.',
             '--check-reason', f'Verify {mechanism} preserves all updates, RNG and evaluation hashes',
             '--preflight', BASE/'preflight.json',
             '--check', BASE/f'synthetic-{mode}/result.json',
             '--check', BASE/f'real-{mode}-final/result.json'])
    elif iteration != '0057':
        run(['scripts/r01_iterate.py', '--iteration', iteration,
             '--previous', f'{int(iteration)-1:04d}', '--agent', 'codex',
             '--candidate', track, '--from-step', 'report'])
    plan = json.loads((ROOT/RUNS/f'plans/iteration-{iteration}-codex.json').read_text(encoding='utf-8'))
    # The frozen selection identifies the current accepted reference even if it changes.
    selection = json.loads((ROOT/out/'selection.json').read_text(encoding='utf-8'))
    protocol = json.loads((ROOT/out/'protocol.json').read_text(encoding='utf-8'))
    control = plan['control_tracks']['expertium']
    for look in (2, 4, 6):
        run_logged('replicates', [sys.executable, 'scripts/r01_replicates.py', 'run',
                   '--output', str(out), '--sequence', ','.join([control,track,control,track]),
                   '--reason', f'Contemporaneous frozen look {look}: changed {mechanism}; quiet-machine paired blocks'],
                   BASE/f'{iteration}-look-{look}.log', root=ROOT, heartbeat_seconds=60)
        run(['scripts/report_gpu_r01_next.py', '--output', out, '--brief'])
        c = json.loads((ROOT/out/'comparison.json').read_text(encoding='utf-8'))['comparisons']['expertium']
        b = c['replicated_blocks']
        print('[blocks]', iteration, json.dumps(b), flush=True)
        if b['decision'] != 'extend':
            break
    else:
        raise RuntimeError('Unexpected extension beyond frozen six-block cap')
