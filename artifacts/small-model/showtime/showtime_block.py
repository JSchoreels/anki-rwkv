#!/usr/bin/env python
"""Break check for the show-time launcher (CPU). Prints "NOW <seconds to the end of the break>" inside a break, else
"NEXT <minutes to the next break>". Breaks: artifacts/coordination/queue/quiet.json (scope all/gpu) plus, for each night D,
the default D 20:00 (Sunday 16:00) -> D+1 01:00 unless a quiet.json all/gpu window overlaps it (then quiet.json's window
replaces it; same rule as the stage-4 launcher of the instant-only small model (SM57)).
  python showtime_block.py [--now 2026-11-08T15:00] [--quiet PATH]
"""
from argparse import ArgumentParser
from datetime import datetime, timedelta
from pathlib import Path
import json
import re

QUIET = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json")
DAILY = (20, 1)       # 20:00 -> 01:00
SUNDAY_START = 16     # Sundays from 16:00


def windows(quiet_path, now):
    text = Path(quiet_path).read_text(encoding="utf-8")
    try:
        wins = [(w["start"], w["end"]) for w in json.loads(text) if w.get("scope", "all") in ("all", "gpu")]
    except (ValueError, KeyError, TypeError):   # malformed JSON: every start/end pair, scope assumed "all"
        wins = re.findall(r'"start"\s*:\s*"([^"]+)"[^{}]*?"end"\s*:\s*"([^"]+)"', text, re.S)
    quiet = [(datetime.fromisoformat(s), datetime.fromisoformat(e)) for s, e in wins]
    blocks = list(quiet)
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    for k in range(-1, 14):
        d = day + timedelta(days=k)
        start = SUNDAY_START if d.weekday() == 6 else DAILY[0]
        s, e = d + timedelta(hours=start), d + timedelta(days=1, hours=DAILY[1])
        if not any(qs < e and qe > s for qs, qe in quiet):
            blocks.append((s, e))
    blocks.sort()
    merged = []
    for s, e in blocks:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def block_line(quiet_path, now):
    try:
        merged = windows(quiet_path, now)
    except OSError as error:
        return f"ERR quiet.json unreadable: {error}"
    for s, e in merged:
        if s <= now < e:
            return f"NOW {int((e - now).total_seconds()) + 60}"
        if s > now:
            return f"NEXT {int((s - now).total_seconds() // 60)}"
    return "NEXT 100000"


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--now", default="")
    ap.add_argument("--quiet", default=str(QUIET))
    a = ap.parse_args()
    print(block_line(a.quiet, datetime.fromisoformat(a.now) if a.now else datetime.now()))
