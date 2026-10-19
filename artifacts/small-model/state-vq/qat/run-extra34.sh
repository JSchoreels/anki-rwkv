#!/usr/bin/env bash
# 2026-10-19 (user: "try on more than 34 users"): SM32b check on the 34 other users of users68 (CPU, 3 processes x 3 threads).
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; WT=C:/Users/jonat/workspace/wt/state-vq; cd $WT
QAT=$WT/artifacts/small-model/scale-w64kd-gate-cached-ctx32768-cont-decay-qat378x412p/expertium/E89-F65e-F77a/master.pth
DEC10=C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/scale-w64kd-gate-cached-ctx32768-cont-decay/expertium/E89-F65e-F77a/master.pth
Q=$WT/artifacts/small-model/state-vq/qat; U=$WT/artifacts/small-model/state-vq/users68-extra34.json
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
echo "[extra34 $(date '+%m-%d %H:%M:%S')] start"
$PY -X utf8 scripts/small_model_heldout.py --device cpu --users-file $U --time-limit 100000 --out $Q/qat-vq-x34.jsonl --model qat=expertium:w64:$QAT --only qat/w8+vq378x412 > $Q/logs/qat-vq-x34.log 2>&1 &
$PY -X utf8 scripts/small_model_heldout.py --device cpu --users-file $U --time-limit 100000 --out $Q/dec10-vq-x34.jsonl --model dec10=expertium:w64:$DEC10 --only dec10/w8+vq378x412 > $Q/logs/dec10-vq-x34.log 2>&1 &
$PY -X utf8 scripts/small_model_heldout.py --device cpu --users-file $U --time-limit 100000 --out $Q/s8-x34.jsonl --model dec10=expertium:w64:$DEC10 --model qat=expertium:w64:$QAT --only dec10/w8+s8 --only qat/w8+s8 > $Q/logs/s8-x34.log 2>&1 &
wait
echo "[extra34 $(date '+%m-%d %H:%M:%S')] done"
