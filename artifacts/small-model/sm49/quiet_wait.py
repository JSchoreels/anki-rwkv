"""Seconds to wait before a run of N minutes may start without touching a quiet window of
artifacts/coordination/queue/quiet.json (0 = start now). Reads start/end pairs with a regex when the file is not valid
JSON (on 2026-10-21 it held unescaped quotes in a reason, which made the queue's json.loads ignore every window).
usage: quiet_wait.py MINUTES"""
import json
import re
import sys
from datetime import datetime, timedelta

QUIET = "C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json"
minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 0
try:
    text = open(QUIET, encoding="utf-8").read()
except OSError:
    print(0)
    sys.exit(0)
try:
    pairs = [(w["start"], w["end"]) for w in json.loads(text)]
except (ValueError, KeyError, TypeError):
    pairs = re.findall(r'"start"\s*:\s*"([^"]+)"\s*,\s*"end"\s*:\s*"([^"]+)"', text)
now = datetime.now()
wait = 0.0
for s, e in pairs:
    try:
        start, end = datetime.fromisoformat(s), datetime.fromisoformat(e)
    except ValueError:
        continue
    if start <= now + timedelta(minutes=minutes) and now < end:
        wait = max(wait, (end - now).total_seconds())
print(int(wait))
