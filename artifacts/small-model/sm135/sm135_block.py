"""Six-layer Tiny with longer recovery (SM135): break windows for the launcher, re-read at every check.

quiet.json windows of scope all / gpu ONLY (user 11-19 ~14:10: no planned daily or Sunday break anymore). Unlike the
six-layer Tiny (SM130)'s sm130_block.py (a copy of the older block helpers, patched mid-run), this file has no daily
20:00-01:00 or Sunday 16:00 window at all. A malformed quiet.json is read leniently (every start/end pair, scope "all").

  python sm135_block.py [--quiet FILE] [--now 2026-11-20T14:00]
      -> "NOW <seconds to its end + 60>" or "NEXT <minutes>" (NEXT 100000 = no window ahead)
  python sm135_block.py selftest
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import tempfile

QUIET = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json")


def windows(quiet_path):
    text = Path(quiet_path).read_text(encoding="utf-8")
    try:
        wins = [(w["start"], w["end"]) for w in json.loads(text) if w.get("scope", "all") in ("all", "gpu")]
    except (ValueError, KeyError, TypeError, AttributeError):   # malformed JSON: every start/end pair, scope assumed "all"
        wins = re.findall(r'"start"\s*:\s*"([^"]+)"[^{}]*?"end"\s*:\s*"([^"]+)"', text, re.S)
    blocks = sorted((datetime.fromisoformat(s), datetime.fromisoformat(e)) for s, e in wins)
    merged = []
    for s, e in blocks:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def answer(quiet_path, now):
    for s, e in windows(quiet_path):
        if s <= now < e:
            return f"NOW {int((e - now).total_seconds()) + 60}"
        if s > now:
            return f"NEXT {int((s - now).total_seconds() // 60)}"
    return "NEXT 100000"


def selftest():
    q = [{"start": "2026-11-20T21:00", "end": "2026-11-21T01:00", "scope": "all"},
         {"start": "2026-11-20T10:00", "end": "2026-11-20T11:00", "scope": "timing"},
         {"start": "2026-11-22T03:00", "end": "2026-11-22T04:00", "scope": "gpu"}]
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "quiet.json"
        p.write_text(json.dumps(q), encoding="utf-8")
        ok = {"no_daily_break_at_20h": answer(p, datetime(2026, 11, 20, 20, 30)) == "NEXT 30",
              "inside_all_window": answer(p, datetime(2026, 11, 20, 23, 0)) == f"NOW {2 * 3600 + 60}",
              "timing_scope_ignored": answer(p, datetime(2026, 11, 20, 10, 30)) == "NEXT 630",
              "no_sunday_break": answer(p, datetime(2026, 11, 22, 16, 30)) == "NEXT 100000",
              "gpu_scope_counts": answer(p, datetime(2026, 11, 22, 2, 0)) == "NEXT 60"}
        p.write_text('[{"start": "2026-11-20T21:00", "end": "2026-11-21T01:00", "scope": "timing",}]', encoding="utf-8")
        ok["malformed_read_as_all"] = answer(p, datetime(2026, 11, 20, 22, 0)).startswith("NOW ")
    src = Path(__file__).read_text(encoding="utf-8").split("def selftest", 1)[0]
    ok["no_hardcoded_hours"] = not re.search(r"DAILY|SUNDAY|weekday|hours=", src)
    good = all(ok.values())
    print(f"[sm135-block] selftest {json.dumps(ok)} -> {'OK' if good else 'FAILED'}")
    return 0 if good else 1


def main():
    if sys.argv[1:] == ["selftest"]:
        return selftest()
    p = argparse.ArgumentParser()
    p.add_argument("--quiet", default=str(QUIET))
    p.add_argument("--now", default="")
    args = p.parse_args()
    now = datetime.fromisoformat(args.now) if args.now else datetime.now()
    try:
        print(answer(args.quiet, now))
    except OSError as error:
        print(f"ERR quiet.json unreadable: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
