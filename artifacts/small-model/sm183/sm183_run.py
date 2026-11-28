#!/usr/bin/env python
"""Tiny cool-down with session inputs (SM183): the trainer entry (copy of sm181_run.py, reading decay.json "sm183"). `scripts/expertium_decay.py` of the run tree (SM179_CODE, default
C:/Users/jonat/workspace/wt/sm179-run, branch sm179-run, committed and clean), unchanged, after two guards: the
environment's SM179_INPUTS equals the arm folder's decay.json "sm183.inputs" ("" = unset), and the run tree has no source edits.

  cd C:/Users/jonat/workspace/wt/sm179-run
  SM179_INPUTS=session python -X utf8 <this file> smoke|run|stop --out artifacts/small-model/sm183-th-ses-p1
"""
import json
import os
from pathlib import Path
import subprocess
import sys

CODE = Path(os.environ.get("SM179_CODE", "C:/Users/jonat/workspace/wt/sm179-run")).resolve()


def main():
    argv = sys.argv[1:]
    out = Path(argv[argv.index("--out") + 1])
    out = out if out.is_absolute() else CODE / out
    want = json.loads((out / "decay.json").read_text())["sm183"]["inputs"]
    have = os.environ.get("SM179_INPUTS", "")
    if have != want:
        raise SystemExit(f"SM179_INPUTS={have!r} but {out.name} needs {want!r}")
    dirty = subprocess.run(["git", "-C", str(CODE), "status", "--porcelain", "rwkv", "scripts", "expertium"],
                           capture_output=True, text=True).stdout.strip()
    if dirty and argv[0] != "stop":
        raise SystemExit(f"run tree has source edits:\n{dirty}")
    os.chdir(CODE)
    sys.path.insert(0, str(CODE))
    from rwkv import extra_inputs
    if extra_inputs.MODE != want:
        raise SystemExit(f"rwkv.extra_inputs.MODE {extra_inputs.MODE!r} != {want!r}")
    sys.argv = [str(CODE / "scripts" / "expertium_decay.py")] + argv
    from scripts import expertium_decay
    return expertium_decay.main()


if __name__ == "__main__":
    sys.exit(main())
