#!/usr/bin/env bash
# EV01 promotion P1: fast path (512 chunks, input hash) on the 300 quick users for epochs 01-09.
set -u
export MSYS_NO_PATHCONV=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
W=C:/Users/jonat/workspace/wt/eval-speed
M=C:/Users/jonat/workspace/anki-rwkv/artifacts/full-run/run-8k-20261013
OUT=$W/artifacts/eval-speed/promotion/p1
mkdir -p $OUT
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,power.draw --format=csv > $OUT/nvidia-smi-start.csv 2>&1
for e in 01 02 03 04 05 06 07 08 09; do
  echo "[job3] epoch $e $(date)"
  $PY -W ignore $W/scripts/fast_evaluation.py --checkpoint $M/epoch-$e/expertium/E89-F65e-F77a/master.pth --users $M/run.json:quick_validation_users --output $OUT/epoch-$e-fast --workers 7 --batch-chunks 512 --input-hash --quiet > $OUT/epoch-$e-fast.log 2>&1
  grep -E '"evaluation_seconds"' $OUT/epoch-$e-fast.log
done
echo "[job3] done $(date)"
