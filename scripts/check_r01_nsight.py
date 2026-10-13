"""Ordinary-rights Nsight diagnostics; preserve failures, never change permissions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tool', choices=('ncu', 'nsys'), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    target = [sys.executable, '-X', 'utf8', 'scripts/check_r01_dispatch_profile.py', '--options',
              'artifacts/retraining/gpu-retrain-r01/iteration-0042/selection.json', '--output',
              str(args.output / 'dispatch.json'), '--passes', '1', '--timeline-only']
    if args.tool == 'ncu':
        command = ['C:/Program Files/NVIDIA Corporation/Nsight Compute 2026.3.0/target/windows-desktop-win7-x64/ncu.exe',
                   '--kernel-name', 'regex:rwkv_exact.*backward_kernel', '--launch-count', '1', '--section', 'WarpStateStats',
                   '--export', str(args.output / 'recurrence'), *target]
    else:
        command = ['C:/Program Files/NVIDIA Corporation/Nsight Systems 2026.3.2/target-windows-x64/nsys.exe',
                   'profile', '--sample=none', '--cpuctxsw=none', '--trace=cuda,nvtx',
                   '--output', str(args.output / 'timeline'), *target]
    with (args.output / 'tool.log').open('w', encoding='utf-8') as log:
        try:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=120)
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            log.write('\nOwn bounded diagnostic timed out at120s; no retries or permission changes.\n')
            returncode = -1
    from scripts.experiment_provenance import process_provenance
    record = {'status': 'diagnostic completed' if returncode == 0 else 'diagnostic failed',
              'returncode': returncode, 'command': command, 'provenance': process_provenance(),
              'permissions_changed': False}
    (args.output / 'result.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:record[k] for k in ('status','returncode','permissions_changed')}))

if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    main()
