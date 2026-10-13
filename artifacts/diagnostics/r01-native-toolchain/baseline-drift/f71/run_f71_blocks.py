"""Run uncounted F71 A/B replicate blocks in the f71 worktree under the shared `gpu` claim.

  python run_f71_blocks.py --worktree ../rwkv-f71 --blocks 4

Follows docs/R01_PARALLEL_SESSIONS.md: each timing run holds the main checkout's `gpu` claim
(scripts/r01_claim.py) for its duration only, so other sessions' GPU checks can run between runs but
never during one. Before launching it also waits while any R01 runner from another checkout is alive.
Results stay in the worktree's own artifacts tree (never in the main checkout's gpu-retrain-r01 records).
"""
from argparse import ArgumentParser
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

MAIN = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(MAIN))
from scripts.r01_claim import acquire, release  # noqa: E402

OWNER = "baseline-drift-f71"
CONTROL, CANDIDATE = "E33-F48-F50-F65b", "E33-F48-F50-F65b-F71"
MARKERS = ("run_gpu_r01.py", "r01_replicates.py", "run_isolated_gpu_r01.py", "gpu_r01_training", "r01_iterate.py")


def foreign_runners():
    mine = {os.getpid(), *(p.pid for p in psutil.Process().children(recursive=True))}
    found = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(proc.info["cmdline"] or [])
        except psutil.Error:
            continue
        if proc.info["pid"] not in mine and any(m in cmd for m in MARKERS):
            found.append(f"{proc.info['pid']}: {cmd[:160]}")
    return found


def main():
    parser = ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--output", default="artifacts/retraining/gpu-retrain-r01/iteration-9171")
    args = parser.parse_args()
    worktree = args.worktree.resolve()
    reason = "Uncounted F71 worker-per-core A/B diagnostic (human request 2026-10-13: pin workers on each core)"
    for block in range(args.blocks):
        for track in (CONTROL, CANDIDATE):
            acquire("gpu", OWNER, f"F71 timing replicate block {block + 1} {track} (worktree rwkv-f71)", wait_minutes=60)
            try:
                while busy := foreign_runners():
                    print(f"[f71] waiting: other R01 process active {busy[:2]}", flush=True)
                    time.sleep(60)
                code = subprocess.run([str(worktree / ".venv/Scripts/python.exe"), "-X", "utf8", "scripts/r01_replicates.py", "run",
                                       "--output", args.output, "--sequence", track, "--reason", reason],
                                      cwd=worktree, capture_output=True, text=True)
            finally:
                release("gpu", OWNER)
            lines = [line for line in (code.stdout + code.stderr).splitlines() if "reviews/s" in line or "Error" in line]
            print(f"[f71] block {block + 1} {track}: exit {code.returncode} {lines[-1] if lines else ''}", flush=True)
            if code.returncode:
                print((code.stdout + code.stderr)[-1500:], flush=True)
                return 1
            time.sleep(35)  # let another session's queued GPU claim go first if it is waiting
    print("[f71] ALL-DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
