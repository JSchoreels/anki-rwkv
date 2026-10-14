#!/usr/bin/env bash
# EV01 GPU job 2: worker/batch sweep on the 300 quick users (epoch-07), then the 2,000-user final set:
# epoch-10 (parity against the run's own recorded final evaluation, with input hash) and epoch-07.
set -u
export MSYS_NO_PATHCONV=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
W=C:/Users/jonat/workspace/wt/eval-speed
M=C:/Users/jonat/workspace/anki-rwkv/artifacts/full-run/run-8k-20261013
E7=$M/epoch-07/expertium/E89-F65e-F77a
E10=$M/epoch-10/expertium/E89-F65e-F77a
OUT=$W/artifacts/eval-speed/gpu
WORKERS=${WORKERS:-7}
CHUNKS=${CHUNKS:-512}
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,power.draw --format=csv > $OUT/nvidia-smi-job2-start.csv 2>&1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >> $OUT/nvidia-smi-job2-start.csv 2>&1
for cfg in 7:512 7:1024 10:512; do
  w=${cfg%%:*}; c=${cfg##*:}
  echo "[job2] fast quick300 workers $w chunks $c $(date)"
  $PY -W ignore $W/scripts/fast_evaluation.py --checkpoint $E7/master.pth --users $M/run.json:quick_validation_users --output $OUT/quick300-fast-w$w-c$c --workers $w --batch-chunks $c > $OUT/quick300-fast-w$w-c$c.log 2>&1
  grep -E '"evaluation_seconds"|"reviews_per_second"' $OUT/quick300-fast-w$w-c$c.log
done
echo "[job2] fast final2000 epoch-10 with input hash $(date)"
$PY -W ignore $W/scripts/fast_evaluation.py --checkpoint $E10/master.pth --users $M/run.json:validation_users+final_test_users --output $OUT/final2000-epoch10-fast-hash --workers $WORKERS --batch-chunks $CHUNKS --input-hash > $OUT/final2000-epoch10-fast-hash.log 2>&1
grep -E '"evaluation_seconds"|"reviews_per_second"' $OUT/final2000-epoch10-fast-hash.log
$PY $W/scripts/check_fast_evaluation.py compare --reference $E10/result.json --candidate $OUT/final2000-epoch10-fast-hash/result.json --output $OUT/compare-final2000-recorded-vs-fast.json
echo "[job2] fast final2000 epoch-07 $(date)"
$PY -W ignore $W/scripts/fast_evaluation.py --checkpoint $E7/master.pth --users $M/run.json:validation_users+final_test_users --output $OUT/final2000-epoch07-fast --workers $WORKERS --batch-chunks $CHUNKS > $OUT/final2000-epoch07-fast.log 2>&1
grep -E '"evaluation_seconds"|"reviews_per_second"' $OUT/final2000-epoch07-fast.log
echo "[job2] exact mode: one section per forward (batch-chunks 8), quick300 $(date)"
$PY -W ignore $W/scripts/fast_evaluation.py --checkpoint $E7/master.pth --users $M/run.json:quick_validation_users --output $OUT/quick300-fast-w$WORKERS-c8 --workers $WORKERS --batch-chunks 8 > $OUT/quick300-fast-w$WORKERS-c8.log 2>&1
grep -E '"evaluation_seconds"|"reviews_per_second"' $OUT/quick300-fast-w$WORKERS-c8.log
$PY $W/scripts/check_fast_evaluation.py compare --reference $OUT/quick300-reference/result.json --candidate $OUT/quick300-fast-w$WORKERS-c8/result.json --output $OUT/compare-quick300-reference-vs-c8.json
echo "[job2] done $(date)"
