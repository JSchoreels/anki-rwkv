"""The smaller instant-only model (SM101): break windows for the launcher, identical to the show-time instant-only model,
stage 2 (SM98)'s sm98_block.py (= the show-time instant-only small model (SM96)'s sm96_block.py without its hard hold):
quiet.json windows of scope all / gpu plus, for each night D, the default D 20:00 (Sunday 16:00) -> D+1 01:00 unless a
quiet.json all / gpu window overlaps it (then quiet.json's window replaces that night's break). No hard hold.

  python sm101_block.py [--quiet FILE] [--now 2026-11-10T14:00]
      -> "NOW <seconds to its end + 60>" or "NEXT <minutes>"
"""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import re

QUIET = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json")
DAILY, SUNDAY_START = (20, 1), 16


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quiet", default=str(QUIET))
    p.add_argument("--now", default="")
    args = p.parse_args()
    now = datetime.fromisoformat(args.now) if args.now else datetime.now()
    try:
        merged = windows(args.quiet, now)
    except OSError as error:
        print(f"ERR quiet.json unreadable: {error}")
        return 0
    for s, e in merged:
        if s <= now < e:
            print(f"NOW {int((e - now).total_seconds()) + 60}")
            return 0
        if s > now:
            print(f"NEXT {int((s - now).total_seconds() // 60)}")
            return 0
    print("NEXT 100000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
