"""Seconds a GPU job of MINUTES must wait so it neither starts in nor runs into a quiet window (SM39 launcher helper).

  python quiet_wait.py MINUTES   -> prints "<seconds> <reason>" (0 = clear now)

Reads the queue's quiet.json like scripts/r01_queue.py (_quiet_hit: a window holds a ticket when now < end and
now + expected > start; "scope": "timing" windows do not hold accuracy jobs). If quiet.json cannot be parsed, the
queue itself would ignore every window, so this falls back to the standing rule of the coordination rules file
(daily 21:00-01:00, Sunday 2026-10-25 from 16:00) and says so.
"""
from datetime import datetime, timedelta
import json
import sys
from pathlib import Path

QUIET = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json")


def default_windows(now):
    out = []
    day = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    for _ in range(10):
        start = day.replace(hour=16 if day.date().isoformat() == "2026-10-25" else 21)
        out.append((start, day + timedelta(days=1, hours=1), "default rule (quiet.json unreadable)"))
        day += timedelta(days=1)
    return out


def windows(now):
    try:
        raw = json.loads(QUIET.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        if isinstance(error, FileNotFoundError):
            return [], "no quiet.json"
        return default_windows(now), f"quiet.json unreadable ({type(error).__name__}); default 21:00-01:00 rule used"
    found = []
    for w in raw if isinstance(raw, list) else []:
        try:
            start, end = (datetime.fromisoformat(w[k]) for k in ("start", "end"))
        except (KeyError, TypeError, ValueError):
            continue
        if w.get("scope", "all") == "timing":
            continue
        start, end = (m.astimezone().replace(tzinfo=None) if m.tzinfo else m for m in (start, end))
        if end > start:
            found.append((start, end, w.get("reason", "")))
    return found, "quiet.json"


def main():
    minutes = float(sys.argv[1])
    now = datetime.now()
    found, source = windows(now)
    wait_until = now
    reason = ""
    moved = True
    while moved:  # a later window may also be reached after waiting for an earlier one
        moved = False
        for start, end, why in found:
            if wait_until < end and wait_until + timedelta(minutes=minutes) > start:
                wait_until, reason, moved = end + timedelta(minutes=1), f"window {start:%m-%d %H:%M}-{end:%m-%d %H:%M} ({source})", True
    print(int((wait_until - now).total_seconds()), reason or f"clear ({source})")


if __name__ == "__main__":
    main()
