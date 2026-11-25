#!/usr/bin/env python
"""Sharpness-aware cool-down (SM172): the trainer entry. `scripts/expertium_decay.py` of the run tree (SM172_CODE,
default C:/Users/jonat/workspace/wt/sm172-run = cea0f931, detached, unchanged; the tree the SM171 control trained in)
with one difference: a `"state_vq"` track option {"recipe": "sm172-sam", "lambda": 0.0, "rho": R} installs the SM171
hook at lambda 0 (the control's path: stock arithmetic plus a no-grad dropped-share diagnostic) and SAM of radius R
(sm172_sam.install). No source file of the run tree is edited, so the trainer's source watch sees the tree it hashed.

  cd C:/Users/jonat/workspace/wt/sm172-run
  SM172_CODE=$PWD python -X utf8 <this file> smoke --out <abs>/artifacts/small-model/sm172-<arm>
  SM172_CODE=$PWD python -X utf8 <this file> run   --out <abs>/artifacts/small-model/sm172-<arm>
  (stop / save as in expertium_decay.py). Diagnostics: <out>/energy.jsonl (SM171 dropped share, first passes only) and
  <out>/sam.jsonl (SAM: gradient norm, loss at w and w + e, bfloat16 survival of e); smoke: *-smoke.jsonl.
"""
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
os.environ.setdefault("SM172_CODE", "C:/Users/jonat/workspace/wt/sm172-run")
CODE = Path(os.environ["SM172_CODE"]).resolve()
os.environ["SM171_CODE"] = str(CODE)
sys.path.insert(0, str(HERE.parent / "sm171"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CODE))

import sm171_energy  # noqa: E402
import sm172_sam  # noqa: E402
from scripts import state_vq  # noqa: E402

_original_install = state_vq.install_training


def _log_path(kind):
    argv = sys.argv
    if "--out" in argv:
        out = CODE / argv[argv.index("--out") + 1]
        return str(out / (f"{kind}-smoke.jsonl" if "smoke" in argv[1:2] else f"{kind}.jsonl"))
    return None


def install_training(model, value, device, *args, **kwargs):
    if isinstance(value, dict) and value.get("recipe") == "sm172-sam":
        if set(value) != {"recipe", "lambda", "rho"} or value["lambda"] != 0.0:
            raise ValueError(f"SM172: state_vq {value}")
        energy = sm171_energy.install(model, 0.0, log=_log_path("energy"))
        sam = sm172_sam.install(model, value["rho"], log=_log_path("sam"))
        return {"recipe": "sm172-sam", "lambda": 0.0, "rho": float(value["rho"]), "sam": sam, "energy_hook": energy,
                "device": str(device), "code_tree": str(CODE)}
    return _original_install(model, value, device, *args, **kwargs)


state_vq.install_training = install_training   # gpu_r01_training imports it from scripts.state_vq at call time


def _at_exit():
    import json
    if sm171_energy.CALLS["card_note"]:   # the trainer process only (prefetch workers re-import this file, never train)
        sm171_energy.flush()
        sm172_sam.flush()
        print(f"[sm172] hook calls: {json.dumps(sm171_energy.CALLS)}, SAM: {json.dumps(sm172_sam.STATS)}", flush=True)


__import__("atexit").register(_at_exit)

if __name__ == "__main__":
    os.chdir(CODE)
    from scripts import expertium_decay
    sys.exit(expertium_decay.main())
