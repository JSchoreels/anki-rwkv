#!/usr/bin/env bash
# 2026-10-16 queue Q1 -> Q2 -> Q3 (docs/SMALL_MODEL.md "Queue 2026-10-16"); each step is its own queue ticket.
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/small-model
SM=$WT/artifacts/small-model
SC=$WT/scripts/small_model_heldout.py
CKPT=expertium/E89-F65e-F77a/master.pth
TEACHER=C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth
LBL=s-w64kd-gate-ctx2048
echo "[chain $(date +%H:%M:%S)] Q1 SM04 CPU timing"
$PY $QUEUE run --owner small-model --purpose "small-model: SM04 CPU inference timing (1 thread)" --claim gpu --timing --expected-minutes 15 -- $PY -X utf8 $WT/scripts/small_model_cpu_timing.py --threads 1 --out $SM/cpu-timing.json
echo "[chain $(date +%H:%M:%S)] Q1 exit $?"
echo "[chain $(date +%H:%M:%S)] Q2 SM13 in-window@2048"
$PY $QUEUE run --owner small-model --purpose "small-model: SM13 in-window@2048 fork vs ctx2048 dec" --claim gpu --expected-minutes 15 -- $PY -X utf8 $SC --users-file $SM/gate-users.json --window 2048 --model fork=fork::$TEACHER --model $LBL-dec=expertium:w64:$SM/scale-w64kd-gate-cached-ctx2048-decay/$CKPT --only fork/w8+s8 --only $LBL-dec/w8+s8 --time-limit 900 --out $SM/heldout.jsonl
echo "[chain $(date +%H:%M:%S)] Q2 exit $?"
echo "[chain $(date +%H:%M:%S)] Q3 S06 speed profile"
$PY $QUEUE run --owner small-model --purpose "small-model: S06 long-context speed profile" --claim gpu --timing --expected-minutes 25 -- "C:/Program Files/Git/usr/bin/bash.exe" $SM/jobs/speed-ctx.sh
echo "[chain $(date +%H:%M:%S)] Q3 exit $?"
echo "[chain $(date +%H:%M:%S)] done"
