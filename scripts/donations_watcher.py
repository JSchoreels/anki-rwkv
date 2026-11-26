"""Watch the donation folders and refresh the data campaign page when it changes (replaces the 4-hour schedule).

Every --interval seconds (default 60) it stats the files of every donation folder (publish_donations_page.
DEFAULT_FOLDERS: names, sizes, mtimes; no file is opened) and compares that fingerprint with the one stored in
pages/data/donations.json. When they differ and the folders have been stable for --settle seconds (so a Dropbox upload in progress is not read half-written), it runs
scripts/update_donations_site.py, which recomputes the stats, commits only the aggregate json files and pushes.
On failure (e.g. a git lock while another commit runs) it retries on a later cycle with a growing back-off.
Every 10 minutes it also runs scripts/ensure_pages_deployed.py: if GitHub's newest site deploy failed, a fresh
deploy of main is dispatched (never a re-run of an old run, which would republish stale data).

Single instance: a lock file holds the watcher's pid; a second start exits. Log: %LOCALAPPDATA%/anki-rwkv-donations/
watcher.log (outside the repository, aggregate lines only). Stop: create the file STOP next to the log, or kill
the pid in watcher.pid.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from publish_donations_page import DEFAULT_FOLDERS, OUT, fingerprint  # noqa: E402

PY = REPO / ".venv" / "Scripts" / "python.exe"
HOME = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "anki-rwkv-donations"
LOG, PID, STOP = HOME / "watcher.log", HOME / "watcher.pid", HOME / "STOP"


def log(msg):
    HOME.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")


def pid_alive(pid):
    if pid <= 0:
        return False
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return str(pid) in out


def published_fingerprint():
    try:
        return json.loads(OUT.read_text(encoding="utf-8")).get("source_fingerprint")
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", action="append", help="donation folder (repeatable; default: DEFAULT_FOLDERS)")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--settle", type=int, default=120)
    args = ap.parse_args()
    folders = [Path(f) for f in (args.folder or DEFAULT_FOLDERS)]
    HOME.mkdir(parents=True, exist_ok=True)
    if PID.exists():
        try:
            old = int(PID.read_text().strip())
        except ValueError:
            old = 0
        if old != os.getpid() and pid_alive(old):
            print(f"watcher already running (pid {old})")
            return 0
    PID.write_text(str(os.getpid()))
    STOP.unlink(missing_ok=True)
    log(f"start pid {os.getpid()} folders {len(folders)} interval {args.interval}s settle {args.settle}s")
    last_fp, stable_since, backoff_until, fails = None, time.time(), 0.0, 0
    next_deploy_check = 0.0
    try:
        while not STOP.exists():
            try:
                fp = fingerprint(folders)
            except Exception as exc:  # noqa: BLE001 - folder briefly unavailable (Dropbox, drive sleep)
                log(f"fingerprint failed: {exc}")
                time.sleep(args.interval)
                continue
            now = time.time()
            if fp != last_fp:
                last_fp, stable_since = fp, now
            if fp != published_fingerprint() and now - stable_since >= args.settle and now >= backoff_until:
                r = subprocess.run([str(PY), "-X", "utf8", str(REPO / "scripts" / "update_donations_site.py")],
                                   cwd=REPO, capture_output=True, text=True, encoding="utf-8")
                line = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
                log(line)
                if r.returncode or line.startswith("FAIL"):
                    fails += 1
                    backoff_until = now + min(3600, 120 * 2 ** min(fails, 5))
                else:
                    fails = 0
            if now >= next_deploy_check:
                # every 10 minutes: if GitHub's last site deploy failed (transient artifact errors), deploy main afresh
                next_deploy_check = now + 600
                d = subprocess.run([str(PY), str(REPO / "scripts" / "ensure_pages_deployed.py")], cwd=REPO,
                                   capture_output=True, text=True, encoding="utf-8")
                line = (d.stdout.strip().splitlines() or ["(no output)"])[-1]
                if not line.startswith(("OK", "RUNNING")):
                    log("site deploy: " + line)
            time.sleep(args.interval)
        log("stop file found: exit")
    finally:
        if PID.exists() and PID.read_text().strip() == str(os.getpid()):
            PID.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
