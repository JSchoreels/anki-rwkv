"""Freeze recorded R01 source in a local repository without copying the dataset."""

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def build(archive, runtime):
    archive, runtime = (ROOT / archive).resolve(), (ROOT / runtime).resolve()
    if not runtime.is_relative_to(ROOT / 'artifacts') or runtime.exists():
        raise ValueError('Use a new runtime directory under workspace artifacts')
    runtime.mkdir(parents=True)
    with ZipFile(archive) as snapshot:
        manifest = json.loads(snapshot.read('source-manifest.json'))
        for name, digest in manifest['files'].items():
            target = runtime / name
            if not target.resolve().is_relative_to(runtime):
                raise ValueError('Source archive path escapes runtime')
            content = snapshot.read(name)
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError('Source archive digest mismatch')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    for name in ('.gitignore', 'AGENT.md', 'docs/GPU_RETRAIN_LOOP.md',
                 'docs/GPU_OPTIMIZATIONS_LOG.md'):
        target = runtime / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    base = ROOT / 'artifacts/retraining/gpu-retrain-r01'
    names = ['cohorts-user-holdout-v1.json', 'iterations']
    names.extend(path.name for path in sorted(base.glob('iteration-*')) if path.is_dir())
    for name in names:
        source = base / name
        target = runtime / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns('*.pth', '*.safetensors', '__pycache__'))
        else:
            shutil.copy2(source, target)
    for result in (runtime / 'artifacts/retraining/gpu-retrain-r01').rglob('result.json'):
        case = json.loads(result.read_text(encoding='utf-8'))
        provenance = case.get('provenance', {})
        if provenance.get('source_archive'):
            source = ROOT / provenance['source_archive']
            if not source.exists() and provenance.get('git_commit') == 'synthetic-only':
                continue
            target = runtime / provenance['source_archive']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    cache = ROOT / 'artifacts/cache/cupy-cache'
    if cache.exists():
        shutil.copytree(cache, runtime / 'artifacts/cache/cupy-cache')
    record = {'origin_git_commit': manifest['git_commit'],
              'origin_source_archive': archive.relative_to(ROOT).as_posix(),
              'origin_source_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'source_file_count': len(manifest['files']),
              'runtime': runtime.relative_to(ROOT).as_posix(),
              'purpose': 'Immutable experiment source; independent local Git identity; dataset read through a junction; exported results retain process provenance unchanged.'}
    (runtime / 'isolation.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record))


def export(runtime, iteration):
    runtime = (ROOT / runtime).resolve()
    if not runtime.is_relative_to(ROOT / 'artifacts'):
        raise ValueError('Runtime must remain under workspace artifacts')
    relative = Path('artifacts/retraining/gpu-retrain-r01') / f'iteration-{iteration}'
    copied = []
    for source in (runtime / relative).rglob('*'):
        if not source.is_file():
            continue
        target = ROOT / relative / source.relative_to(runtime / relative)
        if target.exists():
            if target.read_bytes() != source.read_bytes():
                # Existing evidence must never be overwritten during export.
                raise ValueError(f'Export would overwrite differing evidence: {target}')
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(target.relative_to(ROOT).as_posix())
    for source in (runtime / 'artifacts/sources/experiment-sources').rglob('source.zip'):
        target = ROOT / source.relative_to(runtime)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError('Source archive collision')
        if not target.exists():
            shutil.copy2(source, target)
    record = json.loads((runtime / 'isolation.json').read_text())
    record['exported_files'] = copied
    (ROOT / relative / 'source-isolation.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'exported_files': len(copied), 'iteration': iteration}))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--export', dest='iteration')
    args = parser.parse_args()
    if args.iteration:
        export(args.runtime, args.iteration)
    elif args.archive:
        build(args.archive, args.runtime)
    else:
        parser.error('Use --archive or --export ITERATION')


if __name__ == '__main__':
    main()
