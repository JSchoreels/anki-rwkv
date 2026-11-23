#!/usr/bin/env python
"""Rank-2 low-rank state cool-down (SM149): the trainer entry. `scripts/expertium_decay.py` of the run tree
(SM149_CODE, default C:/Users/jonat/workspace/wt/sm149-run = origin/sm148 unchanged), with one difference: a
`"state_vq"` track option whose recipe is a low-rank label (e.g. {"recipe": "lr2q8x8"}) is installed by this folder's
prototype hook (sm149_qat.install) instead of the state-VQ codebooks. Every other option, and a folder without
"state_vq" (the control arm), runs the unchanged trainer. No source file of the run tree is edited, so the trainer's
source watch sees the tree it hashed at start.

  cd C:/Users/jonat/workspace/wt/sm149-run
  SM149_CODE=$PWD python -X utf8 <this file> smoke --out artifacts/small-model/sm149-<tag>-<arm>
  SM149_CODE=$PWD python -X utf8 <this file> run   --out artifacts/small-model/sm149-<tag>-<arm>
  (stop / save as in expertium_decay.py)
"""
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
os.environ.setdefault("SM149_CODE", "C:/Users/jonat/workspace/wt/sm149-run")
CODE = Path(os.environ["SM149_CODE"]).resolve()
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CODE))

import sm149_qat  # noqa: E402  (puts SM149_CODE on sys.path too)
from scripts import state_vq  # noqa: E402

_original_install = state_vq.install_training


def install_training(model, value, device, *args, **kwargs):
    recipe = value.get("recipe") if isinstance(value, dict) else value
    if isinstance(recipe, str) and recipe.startswith("lr"):
        info = sm149_qat.install(model, recipe=recipe)
        return {**info, "device": str(device), "code_tree": str(CODE)}
    return _original_install(model, value, device, *args, **kwargs)


state_vq.install_training = install_training   # gpu_r01_training imports it from scripts.state_vq at call time


def _report_calls():
    import json
    if sm149_qat.CALLS["window"]:   # the trainer process only (prefetch workers re-import this file and never train)
        print(f"[sm149] low-rank hook calls: {json.dumps(sm149_qat.CALLS)}", flush=True)


__import__("atexit").register(_report_calls)

if __name__ == "__main__":
    os.chdir(CODE)
    from scripts import expertium_decay
    sys.exit(expertium_decay.main())
