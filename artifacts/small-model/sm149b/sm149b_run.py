#!/usr/bin/env python
"""Rank-2 states with a tail-energy penalty (SM149b): the trainer entry. A copy of SM149's sm149_run.py: the run
tree's `scripts/expertium_decay.py` (SM149_CODE, default C:/Users/jonat/workspace/wt/sm149-run = origin/sm148
unchanged) with one difference: a `"state_vq"` track option whose recipe is a low-rank label is installed by this
folder's hook (sm149b_qat.install) with the tail-energy penalty:

  "state_vq": {"recipe": "lr2q8x8", "tail_lambda": 1.0, "tail_stride": 8}
  lambda: env SM149B_LAMBDA if set, else the option's "tail_lambda", else 1.0
  stride: env SM149B_STRIDE if set, else the option's "tail_stride", else 8

The effective values land in result.json (state_vq.tail_penalty). The mean tail of every update pass is appended to
<out>/tail.jsonl (`run`; `smoke`: <out>/tail-smoke.jsonl), one JSON line per pass, flushed every 50 passes, and
summarised in the training log every 100 passes ("[sm149b] tail energy, ..."). No source file of the run tree is
edited (the trainer's source watch hashes the unedited tree).

  cd C:/Users/jonat/workspace/wt/sm149-run
  SM149_CODE=$PWD python -X utf8 <this file> smoke --out artifacts/small-model/sm149b-<tag>-<arm>
  SM149_CODE=$PWD python -X utf8 <this file> run   --out artifacts/small-model/sm149b-<tag>-<arm>
  (stop / save as in expertium_decay.py)
"""
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
os.environ.setdefault("SM149_CODE", "C:/Users/jonat/workspace/wt/sm149-run")
CODE = Path(os.environ["SM149_CODE"]).resolve()
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CODE))

import sm149b_qat  # noqa: E402
from scripts import state_vq  # noqa: E402

_original_install = state_vq.install_training


def _setting(value, env, key, default, cast):
    if os.environ.get(env):
        return cast(os.environ[env]), "env " + env
    if isinstance(value, dict) and key in value:
        return cast(value[key]), "option " + key
    return default, "default"


def install_training(model, value, device, *args, **kwargs):
    recipe = value.get("recipe") if isinstance(value, dict) else value
    if isinstance(recipe, str) and recipe.startswith("lr"):
        lam, lam_from = _setting(value, "SM149B_LAMBDA", "tail_lambda", 1.0, float)
        stride, stride_from = _setting(value, "SM149B_STRIDE", "tail_stride", 8, int)
        info = sm149b_qat.install(model, recipe=recipe, tail_lambda=lam, tail_stride=stride)
        info["tail_penalty"].update(lambda_from=lam_from, stride_from=stride_from, log=sm149b_qat.LOG["path"])
        print(f"[sm149b] hook installed: recipe {recipe}, tail lambda {lam:g} ({lam_from}), stride {stride} "
              f"({stride_from}), tail log {sm149b_qat.LOG['path']}", flush=True)
        return {**info, "device": str(device), "code_tree": str(CODE)}
    return _original_install(model, value, device, *args, **kwargs)


state_vq.install_training = install_training   # gpu_r01_training imports it from scripts.state_vq at call time


def _tail_log_path(argv):
    if len(argv) < 2 or argv[1] not in ("run", "smoke") or "--out" not in argv:
        return None
    out = Path(argv[argv.index("--out") + 1])
    out = out if out.is_absolute() else CODE / out
    return str(out / ("tail.jsonl" if argv[1] == "run" else "tail-smoke.jsonl"))


def _at_exit():
    if sm149b_qat.CALLS["window"]:   # the trainer process only (prefetch workers re-import this file and never train)
        sm149b_qat.flush_log(final=True)
        print(f"[sm149b] low-rank hook calls: {json.dumps(sm149b_qat.CALLS)}; tail log lines written "
              f"{sm149b_qat.LOG['written']}", flush=True)
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            print(f"[sm149b] process peak GPU memory: allocated {torch.cuda.max_memory_allocated() / 2 ** 20:,.0f} MiB, "
                  f"reserved {torch.cuda.max_memory_reserved() / 2 ** 20:,.0f} MiB", flush=True)


__import__("atexit").register(_at_exit)

if __name__ == "__main__":
    sm149b_qat.LOG["path"] = _tail_log_path(sys.argv)
    os.chdir(CODE)
    from scripts import expertium_decay
    sys.exit(expertium_decay.main())
