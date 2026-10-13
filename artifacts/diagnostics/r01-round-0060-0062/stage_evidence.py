"""Build an explicit evidence pathspec, retaining linked source archives, excluding checkpoints/caches."""
import json
from pathlib import Path
root = Path(__file__).resolve().parents[3]
base = root/'artifacts/retraining/gpu-retrain-r01'
diagnostics = root/'artifacts/diagnostics/r01-round-0060-0062'
extensions = {'.json', '.jsonl', '.log', '.md', '.py', '.csv', '.txt'}
paths = {p for p in diagnostics.rglob('*') if p.is_file() and p.suffix in extensions}
for iteration in ('0060','0061','0062'):
    paths.update(p for p in (base/f'iteration-{iteration}').rglob('*') if p.is_file() and p.suffix in extensions)
    paths.update(p for p in base.glob(f'iteration-{iteration}-*') if p.is_file() and p.suffix in extensions)
    paths.add(base/f'plans/iteration-{iteration}-codex.json')
paths.update(base.glob('codex-round-0060-0062-*'))
paths.add(base/'README.md')
paths.update(root/'docs'/name for name in ('GPU_OPTIMIZATIONS.md','GPU_OPTIMIZATIONS_LOG.md','R01_HOST_PATH.md','R01_NEXT_ROUND_PLAN.md'))
def references(value):
    if isinstance(value, dict):
        for v in value.values(): references(v)
    elif isinstance(value, list):
        for v in value: references(v)
    elif isinstance(value,str) and value.startswith('artifacts/sources/'):
        candidate = (root/value).resolve()
        if candidate.is_relative_to(root/'artifacts/sources') and candidate.is_file(): paths.add(candidate)
for p in list(paths):
    if p.suffix == '.json':
        references(json.loads(p.read_text(encoding='utf-8-sig')))
relative = sorted(p.relative_to(root).as_posix() for p in paths)
(diagnostics/'evidence-pathspec.nul').write_bytes(('\0'.join(relative)+'\0').encode('utf-8'))
print(f'evidence: {len(paths)} files; {sum(p.stat().st_size for p in paths)/1048576:.2f} MiB; source zips {sum(p.suffix==".zip" for p in paths)}')
