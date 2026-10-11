"""Run frozen R01 code while holding the shared workspace GPU lock."""

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
import sys
import psutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    args, forwarded = parser.parse_known_args()
    runtime = (ROOT / args.runtime).resolve()
    if not runtime.is_relative_to(ROOT / 'artifacts') or not (runtime / 'isolation.json').exists():
        parser.error('Use a recorded isolated runtime under workspace artifacts')
    lock = ROOT / 'artifacts/gpu-r01.lock'
    for path in (lock, ROOT / 'artifacts/gpu-batch1.lock'):
        if path.exists() and psutil.pid_exists(json.loads(path.read_text())['pid']):
            raise RuntimeError(f'GPU experiment is already active: {path}')
    lock.write_text(json.dumps({'pid': os.getpid(), 'isolated_runtime': args.runtime.as_posix()}), encoding='utf-8')
    try:
        return subprocess.run([sys.executable, str(runtime / 'scripts/run_gpu_r01.py'), *forwarded], cwd=runtime).returncode
    finally:
        if lock.exists() and json.loads(lock.read_text()).get('pid') == os.getpid():
            lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
