"""Compact state for Small and Big (SM131): quiet windows for the launcher, quiet.json windows of scope all / gpu ONLY.
No daily or Sunday break (user 11-19 ~14:10: there is no planned daily break any more; the user says when a pause is
needed and the main session then adds a quiet.json window). Re-read at every check. Otherwise the code of the compact
state for Tiny (SM129)'s sm129_block.py without its default nightly windows.

  python sm131_block.py [--quiet FILE] [--now 2026-11-19T14:00]
      -> "NOW <seconds to its end + 60>" or "NEXT <minutes>" (NEXT 100000 = no window ahead), "ERR ..." if unreadable
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re

QUIET = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json")


def windows(quiet_path):
    text = Path(quiet_path).read_text(encoding="utf-8")
    try:
        wins = [(w["start"], w["end"]) for w in json.loads(text) if w.get("scope", "all") in ("all", "gpu")]
    except (ValueError, KeyError, TypeError):   # malformed JSON: every start/end pair, scope assumed "all"
        wins = re.findall(r'"start"\s*:\s*"([^"]+)"[^{}]*?"end"\s*:\s*"([^"]+)"', text, re.S)
    blocks = sorted((datetime.fromisoformat(s), datetime.fromisoformat(e)) for s, e in wins)
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
        merged = windows(args.quiet)
    except (OSError, ValueError) as error:
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
