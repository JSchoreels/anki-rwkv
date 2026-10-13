"""Named mutual-exclusion claims for parallel sessions sharing this checkout.

Claims are directories under artifacts/coordination/claims/<name>; os.mkdir is atomic,
so exactly one owner holds a claim. See docs/R01_PARALLEL_SESSIONS.md for the names.
"""
from argparse import ArgumentParser, REMAINDER
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CLAIMS = ROOT / "artifacts" / "coordination" / "claims"
NAMES = ("gpu", "training-source", "ledger", "plan", "git")


def _path(name):
    if name not in NAMES:
        raise ValueError(f"Unknown claim {name!r}; use one of {', '.join(NAMES)}")
    return CLAIMS / name


def holder(name):
    try:
        return json.loads((_path(name) / "owner.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None if not _path(name).exists() else {"owner": "?", "purpose": "owner.json missing"}


def acquire(name, owner, purpose, wait_minutes=0.0, poll_seconds=30.0):
    CLAIMS.mkdir(parents=True, exist_ok=True)
    deadline, announced = time.monotonic() + wait_minutes * 60, False
    while True:
        try:
            _path(name).mkdir()
        except FileExistsError:
            current = holder(name) or {}
            if current.get("owner") == owner:
                return current
            if time.monotonic() >= deadline:
                raise SystemExit(f"[claim] {name} held by {current.get('owner')} since {current.get('since')}: {current.get('purpose')}")
            if not announced:
                print(f"[claim] waiting for {name}, held by {current.get('owner')}: {current.get('purpose')}", flush=True)
                announced = True
            time.sleep(poll_seconds)
            continue
        record = {"owner": owner, "purpose": purpose, "since": datetime.now(timezone.utc).isoformat(timespec="seconds"), "pid": os.getpid()}
        (_path(name) / "owner.json").write_text(json.dumps(record), encoding="utf-8")
        return record


def release(name, owner, force=False):
    current = holder(name)
    if current is None:
        return
    if current.get("owner") != owner and not force:
        raise SystemExit(f"[claim] {name} is held by {current.get('owner')}, not {owner}; only the human may pass --force")
    (_path(name) / "owner.json").unlink(missing_ok=True)
    _path(name).rmdir()


def main():
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("acquire", "release", "run"):
        p = sub.add_parser(action)
        p.add_argument("--claim", action="append", required=True, choices=NAMES)
        p.add_argument("--owner", required=True)
        if action != "release":
            p.add_argument("--purpose", required=True)
            p.add_argument("--wait-minutes", type=float, default=0.0)
        else:
            p.add_argument("--force", action="store_true")
        if action == "run":
            p.add_argument("command", nargs=REMAINDER)
    sub.add_parser("status")
    args = parser.parse_args()
    if args.action == "status":
        for name in NAMES:
            current = holder(name)
            print(f"{name}: " + ("free" if current is None else f"{current.get('owner')} since {current.get('since')} - {current.get('purpose')}"))
        return
    names = sorted(set(args.claim), key=NAMES.index)  # fixed order prevents deadlock
    if args.action == "release":
        for name in names:
            release(name, args.owner, args.force)
        return
    held = []
    try:
        for name in names:
            acquire(name, args.owner, args.purpose, args.wait_minutes)
            held.append(name)
        if args.action == "acquire":
            held = []
            print("[claim] acquired " + ", ".join(names))
            return
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            raise SystemExit("run needs a command after --")
        local = ROOT / command[0]
        executable = str(local) if local.is_file() else shutil.which(command[0]) or command[0]
        sys.exit(subprocess.call([executable, *command[1:]], cwd=ROOT))
    finally:
        for name in reversed(held):
            release(name, args.owner)


if __name__ == "__main__":
    main()
