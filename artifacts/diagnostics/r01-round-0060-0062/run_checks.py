import json, subprocess, sys
from pathlib import Path
base = Path('artifacts/diagnostics/r01-round-0060-0062')
accepted = json.loads((base/'accepted.json').read_text(encoding='utf-8'))
for mode in (4, 5, 6):
    (base/f'pipeline-{mode}.json').write_text(json.dumps(dict(accepted, recurrence_load_pipeline=mode), indent=2)+'\n', encoding='utf-8')
    subprocess.run([sys.executable, 'scripts/r01_preflight.py', '--output', str(base/'preflight.json'), '--run', 'scripts/check_r01_compact_previous.py', '--load-pipeline', str(mode), '--output', str(base/f'synthetic-{mode}')], check=True)
    subprocess.run([sys.executable, 'scripts/r01_preflight.py', '--output', str(base/'preflight.json'), '--run', 'scripts/check_r01_exact_options.py', '--a', str(base/'accepted.json'), '--b', str(base/f'pipeline-{mode}.json'), '--output', str(base/f'real-{mode}')], check=True)
