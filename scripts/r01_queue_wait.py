"""Block until one of the GPU queue tickets finishes, then print which and exit.

A finished ticket's JSON file leaves artifacts/coordination/queue/tickets/. Run this as a background command so a
waiting agent costs no tokens: the command exits when a ticket finishes, which wakes the session once per job.

    python scripts/r01_queue_wait.py              # every ticket, including ones queued after the start
    python scripts/r01_queue_wait.py 000310 000312
    python scripts/r01_queue_wait.py --all        # wait until every watched ticket is gone
"""
import argparse
import json
import os
import time
from pathlib import Path

COORDINATION = Path(os.environ.get("R01_COORDINATION_ROOT") or Path(__file__).resolve().parents[1] / "artifacts" / "coordination")
TICKETS = COORDINATION / "queue" / "tickets"
STALE = COORDINATION / "queue" / "stale"
HISTORY = COORDINATION / "queue" / "history.jsonl"


def present():
    return {p.stem for p in TICKETS.glob("*.json")}


def history_size():
    try:
        return HISTORY.stat().st_size
    except OSError:
        return 0


def finished_since(offset):
    """Tickets that finished (history.jsonl) after the given byte offset, including ones that came and went between two polls."""
    try:
        with HISTORY.open("rb") as handle:
            handle.seek(offset)
            data = handle.read()
    except OSError:
        return set(), offset
    end = data.rfind(b"\n") + 1  # leave a half-written last line for the next poll
    names = set()
    for line in data[:end].decode("utf-8", "replace").splitlines():
        try:
            names.add(json.loads(line)["ticket"])
        except (ValueError, KeyError, TypeError):
            pass
    return names, offset + end


def label(ticket):
    # a finished run can race a sweep that also files it under stale/, so history wins
    if ticket in finished_since(0)[0] or not (STALE / f"{ticket}.json").exists():
        return ticket
    return f"{ticket} (removed: cancelled or stale)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tickets", nargs="*", help="ticket number prefixes (default: all present now)")
    parser.add_argument("--all", action="store_true", help="exit only once every watched ticket has finished")
    parser.add_argument("--interval", type=float, default=30.0, help="seconds between checks")
    args = parser.parse_args()
    follow = not args.tickets  # no filter: also watch tickets queued after the start
    offset = history_size()
    now = present()
    watched = {t for t in now if follow or any(t.startswith(p) for p in args.tickets)}
    if not watched and not follow:
        print("no matching tickets in the queue")
        return
    print(f"watching: {', '.join(sorted(watched)) or 'nothing yet'}{' (+ new tickets)' if follow else ''}", flush=True)
    while True:
        now = present()
        if follow:
            watched |= now
            quick, offset = finished_since(offset)  # tickets that were queued and finished within one poll interval
            watched |= quick
        done = watched - now
        if done and (not args.all or done == watched):
            print(f"finished: {', '.join(label(t) for t in sorted(done))}")
            left = sorted(watched - done)
            print(f"still queued: {', '.join(left) if left else 'none'}")
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
