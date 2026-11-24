"""One-line factory liveness check: prints BUSY, QUIET or IDLE and exits 0, 3 or 4.

BUSY  - the GPU queue has a running or waiting ticket, the GPU is loaded, or a launcher/scorer process is alive.
QUIET - nothing runs, but a GPU quiet window (coordination/queue/quiet.json) is active, so idle is expected.
IDLE  - nothing runs and no quiet window: the factory is wasting time; start the next backlog item.

Used by the main session's hourly check (cron) so a dead watcher never leaves the GPU idle for hours.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
QUIET = ROOT / "coordination" / "queue" / "quiet.json"
PATTERNS = ("launch-sm", "run-sm", "launch_", "r01_queue.py run", "score_", "train_", "_report.py")


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except Exception as exc:  # noqa: BLE001 - a broken probe must not hide the other probes
        return f"ERROR {exc}"


def queue_state():
    out = _run([str(PY if PY.exists() else sys.executable), str(ROOT / "scripts" / "r01_queue.py"), "status"])
    busy = [line.strip() for line in out.splitlines()
            if line.startswith("[queue] running:") or line.startswith("[queue] waiting:")]
    return [b for b in busy if not b.endswith(": none")]


def gpu_state():
    out = _run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"])
    try:
        util, mem = (int(x) for x in out.strip().splitlines()[0].split(","))
    except Exception:  # noqa: BLE001
        return None, out.strip()[:80]
    return (util >= 20 or mem >= 4000), f"GPU {util}% {mem} MiB"


def processes():
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|bash' -and $_.CommandLine } | "
          "ForEach-Object { $_.CommandLine }")
    out = _run(["powershell", "-NoProfile", "-Command", ps])
    hits = []
    for line in out.splitlines():
        if "factory_idle_check" in line or "shell-snapshots" in line:
            continue
        if any(p in line for p in PATTERNS):
            hits.append(line.strip()[:120])
    return hits


def quiet_now():
    try:
        windows = json.loads(QUIET.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - no file or malformed file means no window
        return None
    now = datetime.now()
    for w in windows if isinstance(windows, list) else []:
        try:
            if datetime.fromisoformat(w["start"]) <= now < datetime.fromisoformat(w["end"]):
                return f"{w['start']} to {w['end']} ({w.get('reason', '')})"
        except Exception:  # noqa: BLE001
            continue
    return None


def main():
    q = queue_state()
    gpu_busy, gpu_txt = gpu_state()
    procs = processes()
    stamp = datetime.now().strftime("%m-%d %H:%M")
    if q or gpu_busy or procs:
        why = "; ".join(q + ([gpu_txt] if gpu_busy else []) + [f"{len(procs)} job process(es)"] * bool(procs))
        print(f"BUSY {stamp}: {why}")
        return 0
    window = quiet_now()
    if window:
        print(f"QUIET {stamp}: quiet window {window}; {gpu_txt}")
        return 3
    print(f"IDLE {stamp}: queue empty, {gpu_txt}, no launcher or scorer alive")
    return 4


if __name__ == "__main__":
    sys.exit(main())
