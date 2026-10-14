#!/usr/bin/env bash
# EV01 promotion P3 (EV07 full-run smoke, fast_evaluation false/true) and P2 (upstream 2.76M baseline, fast vs reference).
set -u
export MSYS_NO_PATHCONV=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
W=C:/Users/jonat/workspace/wt/eval-speed
M=C:/Users/jonat/workspace/anki-rwkv/artifacts/full-run/run-8k-20261013
P=$W/artifacts/eval-speed/promotion
CK=C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,power.draw --format=csv > $P/nvidia-smi-job4-start.csv 2>&1
for name in run-reference run-fast; do
  echo "[job4] P3 $name $(date)"
  $PY -X utf8 $W/scripts/run_full_training.py run --run artifacts/eval-speed/promotion/p3/$name > $P/p3/$name.log 2>&1
  echo "[job4] P3 $name exit $? $(date)"
  tail -c 300 $P/p3/$name.log
done
echo "[job4] P2 baseline fast $(date)"
$PY -W ignore $W/scripts/fast_evaluation.py --model baseline --checkpoint $CK --users @$P/p2/users.json --run-json $M/run.json --output $P/p2/baseline-fast --workers 7 --input-hash --quiet > $P/p2/baseline-fast.log 2>&1
grep -E '"evaluation_seconds"' $P/p2/baseline-fast.log || tail -c 1500 $P/p2/baseline-fast.log
echo "[job4] P2 baseline reference $(date)"
$PY -W ignore $W/scripts/fast_evaluation.py --model baseline --checkpoint $CK --users @$P/p2/users.json --run-json $M/run.json --output $P/p2/baseline-reference --mode reference --workers 5 --quiet > $P/p2/baseline-reference.log 2>&1
grep -E '"evaluation_seconds"' $P/p2/baseline-reference.log || tail -c 1500 $P/p2/baseline-reference.log
echo "[job4] done $(date)"
