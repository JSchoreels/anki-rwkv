#!/bin/bash
# SM153 apply + recal (combined) + report: 2 lanes, 1 thread each.
cd C:/Users/jonat/workspace/wt/sm153
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
L=artifacts/small-model/sm153/logs
X="$PY -X utf8 scripts/sm153_shrink.py"
step() { $X "${@:2}" > $L/$1.out 2>&1; echo "[$(date +%H:%M:%S)] $1 exit $?" >> $L/apply-chain.log; }
lane() { for m in "$@"; do step apply-$m apply --model $m; step recal-$m recal --model $m --arm combined; done; }
echo "[$(date +%H:%M:%S)] start" >> $L/apply-chain.log
lane sm99 sm118 sm130io &
( lane sm113 sm130th; step recal-raw-check recal --model sm113 --arm raw --limit 3 ) &
wait
step report report
echo "[$(date +%H:%M:%S)] all done" >> $L/apply-chain.log
