#!/usr/bin/env bash
# EV01 promotion P3 retry (EV07 full-run smoke, fast_evaluation false/true). Job 4's attempt failed before training:
# the native launch DLL is cached only in the main checkout, so read it from there (read-only) via GPU_R01_CONTROLLER_ROOT.
set -u
export MSYS_NO_PATHCONV=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export GPU_R01_CONTROLLER_ROOT=C:/Users/jonat/workspace/anki-rwkv
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
W=C:/Users/jonat/workspace/wt/eval-speed
P=$W/artifacts/eval-speed/promotion
for name in run-reference run-fast; do
  echo "[job5] P3 $name $(date)"
  $PY -X utf8 $W/scripts/run_full_training.py run --run artifacts/eval-speed/promotion/p3/$name > $P/p3/$name-retry.log 2>&1
  echo "[job5] P3 $name exit $? $(date)"
  tail -c 400 $P/p3/$name-retry.log
done
echo "[job5] done $(date)"
