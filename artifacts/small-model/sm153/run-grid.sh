#!/bin/bash
# SM153 grid pass: 2 CPU processes at a time, 1 thread each.
cd C:/Users/jonat/workspace/wt/sm153
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
L=artifacts/small-model/sm153/logs
run() { $PY -X utf8 scripts/sm153_shrink.py grid --model $1 > $L/grid-$1.out 2>&1; echo "[$(date +%H:%M:%S)] grid $1 exit $?" >> $L/chain.log; }
echo "[$(date +%H:%M:%S)] start" >> $L/chain.log
run sm130th & run sm113 & wait
run sm118 & run sm130io & wait
run sm99
echo "[$(date +%H:%M:%S)] grid done" >> $L/chain.log
