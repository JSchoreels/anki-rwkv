"""Block until one of the GPU queue tickets finishes, then print which and exit.

A finished ticket's JSON file leaves artifacts/coordination/queue/tickets/. Run this as a background command so a
waiting agent costs no tokens: the command exits when a ticket finishes, which wakes the session once per job.

    python scripts/r01_queue_wait.py              # every ticket present now
    python scripts/r01_queue_wait.py 000310 000312
    python scripts/r01_queue_wait.py --all        # wait until every watched ticket is gone
"""
import argparse
import time
from pathlib import Path

TICKETS = Path(__file__).resolve().parents[1] / "artifacts" / "coordination" / "queue" / "tickets"


def present():
    return {p.stem for p in TICKETS.glob("*.json")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tickets", nargs="*", help="ticket number prefixes (default: all present now)")
    parser.add_argument("--all", action="store_true", help="exit only once every watched ticket has finished")
    parser.add_argument("--interval", type=float, default=30.0, help="seconds between checks")
    args = parser.parse_args()
    now = present()
    watched = {t for t in now if not args.tickets or any(t.startswith(p) for p in args.tickets)}
    if not watched:
        print("no matching tickets in the queue")
        return
    print(f"watching: {', '.join(sorted(watched))}", flush=True)
    while True:
        done = watched - present()
        if done and (not args.all or done == watched):
            print(f"finished: {', '.join(sorted(done))}")
            left = sorted(watched - done)
            print(f"still queued: {', '.join(left) if left else 'none'}")
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
