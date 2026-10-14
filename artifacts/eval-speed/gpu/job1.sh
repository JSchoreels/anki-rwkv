#!/usr/bin/env bash
# EV01 GPU job 1: forward bitwise diagnostic, fast vs reference on the 300 quick-validation users (epoch-07).
set -u
export MSYS_NO_PATHCONV=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
W=C:/Users/jonat/workspace/wt/eval-speed
M=C:/Users/jonat/workspace/anki-rwkv/artifacts/full-run/run-8k-20261013
CK=$M/epoch-07/expertium/E89-F65e-F77a/master.pth
OUT=$W/artifacts/eval-speed/gpu
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,power.draw --format=csv > $OUT/nvidia-smi-job1-start.csv 2>&1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >> $OUT/nvidia-smi-job1-start.csv 2>&1
echo "[job1] forward diagnostic $(date)"
$PY -W ignore $W/scripts/check_fast_evaluation.py forward --users 3 --max-sections 64 --output $OUT/forward-quick3.json
echo "[job1] fast quick300 $(date)"
$PY -W ignore $W/scripts/fast_evaluation.py --checkpoint $CK --users $M/run.json:quick_validation_users --output $OUT/quick300-fast --workers 5 > $OUT/quick300-fast.log 2>&1
tail -c 1500 $OUT/quick300-fast.log
echo "[job1] fast quick300 with input hash $(date)"
$PY -W ignore $W/scripts/fast_evaluation.py --checkpoint $CK --users $M/run.json:quick_validation_users --output $OUT/quick300-fast-hash --workers 5 --input-hash > $OUT/quick300-fast-hash.log 2>&1
tail -c 600 $OUT/quick300-fast-hash.log
echo "[job1] reference quick300 $(date)"
$PY -W ignore $W/scripts/fast_evaluation.py --checkpoint $CK --users $M/run.json:quick_validation_users --output $OUT/quick300-reference --mode reference --workers 5 > $OUT/quick300-reference.log 2>&1
tail -c 1500 $OUT/quick300-reference.log
echo "[job1] compare $(date)"
$PY $W/scripts/check_fast_evaluation.py compare --reference $OUT/quick300-reference/result.json --candidate $OUT/quick300-fast/result.json --output $OUT/compare-quick300.json
$PY $W/scripts/check_fast_evaluation.py compare --reference $M/epoch-07/expertium/E89-F65e-F77a/result.json --candidate $OUT/quick300-reference/result.json --output $OUT/compare-recorded-vs-reference.json
$PY $W/scripts/check_fast_evaluation.py compare --reference $M/epoch-07/expertium/E89-F65e-F77a/result.json --candidate $OUT/quick300-fast/result.json --output $OUT/compare-recorded-vs-fast.json
echo "[job1] done $(date)"
