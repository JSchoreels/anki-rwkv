#!/usr/bin/env python
"""State-energy penalty (SM171): the trainer entry. `scripts/expertium_decay.py` of the run tree (SM171_CODE, default
C:/Users/jonat/workspace/wt/sm171-run = cea0f931, detached, unchanged) with one difference: a `"state_vq"` track option
whose recipe is "sm171-energy" (e.g. {"recipe": "sm171-energy", "lambda": 0.01}) is installed by this folder's hook
(sm171_energy.install) instead of the state-VQ codebooks; lambda 0 = the stock trainer plus a no-grad diagnostic.
No source file of the run tree is edited, so the trainer's source watch sees the tree it hashed at start.

  cd C:/Users/jonat/workspace/wt/sm171-run
  SM171_CODE=$PWD python -X utf8 <this file> smoke --out artifacts/small-model/sm171-<arm>
  SM171_CODE=$PWD python -X utf8 <this file> run   --out artifacts/small-model/sm171-<arm>
  (stop / save as in expertium_decay.py). The penalty diagnostics go to <out>/energy.jsonl (smoke: energy-smoke.jsonl).
"""
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
os.environ.setdefault("SM171_CODE", "C:/Users/jonat/workspace/wt/sm171-run")
CODE = Path(os.environ["SM171_CODE"]).resolve()
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CODE))

import sm171_energy  # noqa: E402
from scripts import state_vq  # noqa: E402

_original_install = state_vq.install_training


def _log_path():
    argv = sys.argv
    if "--out" in argv:
        out = CODE / argv[argv.index("--out") + 1]
        return str(out / ("energy-smoke.jsonl" if "smoke" in argv[1:2] else "energy.jsonl"))
    return None


def install_training(model, value, device, *args, **kwargs):
    if isinstance(value, dict) and value.get("recipe") == "sm171-energy":
        info = sm171_energy.install(model, value["lambda"], log=_log_path())
        return {**info, "device": str(device), "code_tree": str(CODE)}
    return _original_install(model, value, device, *args, **kwargs)


state_vq.install_training = install_training   # gpu_r01_training imports it from scripts.state_vq at call time


def _at_exit():
    import json
    if sm171_energy.CALLS["card_note"]:   # the trainer process only (prefetch workers re-import this file, never train)
        sm171_energy.flush()
        print(f"[sm171] hook calls: {json.dumps(sm171_energy.CALLS)}", flush=True)


__import__("atexit").register(_at_exit)

if __name__ == "__main__":
    os.chdir(CODE)
    from scripts import expertium_decay
    sys.exit(expertium_decay.main())
