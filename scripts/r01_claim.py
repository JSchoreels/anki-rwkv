"""Named mutual-exclusion claims for parallel sessions sharing this checkout.

Claims are directories under artifacts/coordination/claims/<name>; os.mkdir is atomic,
so exactly one owner holds a claim. See docs/R01_PARALLEL_SESSIONS.md for the names.

Waiting for a claim goes through the FIFO queue in scripts/r01_queue.py: `acquire` and `run`
register a ticket, block on directory-change notifications (no polling) until every earlier
ticket that needs one of the same claims has finished, then hold the claim directories.
`run` releases them when the command exits; `acquire` keeps them until `release`.
"""
from argparse import ArgumentParser, REMAINDER
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
COORDINATION = Path(os.environ.get("R01_COORDINATION_ROOT") or ROOT / "artifacts" / "coordination")
CLAIMS = COORDINATION / "claims"
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


def acquire(name, owner, purpose, wait_minutes=0.0):
    """Hold one claim: immediately when free, otherwise through the queue for up to wait_minutes."""
    if wait_minutes:
        from scripts.r01_queue import detach, hold

        detach(hold(owner, purpose, [name], max_wait_minutes=wait_minutes))
        return holder(name)
    CLAIMS.mkdir(parents=True, exist_ok=True)
    try:
        _path(name).mkdir()
    except FileExistsError:
        current = holder(name) or {}
        if current.get("owner") == owner:
            return current
        raise SystemExit(f"[claim] {name} held by {current.get('owner')} since {current.get('since')}: {current.get('purpose')}")
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
            p.add_argument("--wait-minutes", type=float, default=0.0, help="0 fails at once when a claim is busy; otherwise wait in the queue this long")
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
    sys.path.insert(0, str(ROOT))
    from scripts.r01_queue import detach, finish, hold, run_command

    command = None
    if args.action == "run":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            raise SystemExit("run needs a command after --")
    record = hold(args.owner, args.purpose, names, max_wait_minutes=args.wait_minutes, command=command)
    if args.action == "acquire":
        detach(record)
        print("[claim] acquired " + ", ".join(names))
        return
    code = None
    try:
        code = run_command(command, record=record)
    finally:
        finish(record, code)
    sys.exit(code)


if __name__ == "__main__":
    main()
