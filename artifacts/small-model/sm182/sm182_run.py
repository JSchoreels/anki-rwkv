#!/usr/bin/env python
"""AUC ceiling (SM182): the trainer entry. `scripts/expertium_decay.py` of the run tree (SM182_CODE, default
C:/Users/jonat/workspace/wt/sm182-run, branch sm182-run, committed and clean), unchanged, after two guards: the arm
folder's decay.json carries the "pair_rank" option it was prepared with, and the run tree has no source edits.

  cd C:/Users/jonat/workspace/wt/sm182-run
  python -X utf8 <this file> smoke|run|stop --out C:/Users/jonat/workspace/wt/sm182-run/artifacts/small-model/sm182-tiny-r
"""
import json
import os
from pathlib import Path
import subprocess
import sys

CODE = Path(os.environ.get("SM182_CODE", "C:/Users/jonat/workspace/wt/sm182-run")).resolve()


def main():
    argv = sys.argv[1:]
    out = Path(argv[argv.index("--out") + 1])
    out = out if out.is_absolute() else CODE / out
    decay = json.loads((out / "decay.json").read_text())
    if decay["options"].get("pair_rank") != decay["sm182"]["pair_rank"]:
        raise SystemExit(f"{out.name}: pair_rank option differs from the prepared one")
    dirty = subprocess.run(["git", "-C", str(CODE), "status", "--porcelain", "rwkv", "scripts", "expertium"],
                           capture_output=True, text=True).stdout.strip()
    if dirty and argv[0] != "stop":
        raise SystemExit(f"run tree has source edits:\n{dirty}")
    os.chdir(CODE)
    sys.path.insert(0, str(CODE))
    sys.argv = [str(CODE / "scripts" / "expertium_decay.py")] + argv
    from scripts import expertium_decay
    return expertium_decay.main()


if __name__ == "__main__":
    sys.exit(main())
