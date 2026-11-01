#!/usr/bin/env bash
# 2026-10-17 chain b (after Q11): (1) SM24: per-review dumps of fork w8+s8 and the best student (ctx16384c-dec9-L32k) on
# the 464 users, for the review-type / same-day breakdown (scores go to a separate jsonl so heldout.jsonl is untouched);
# (2) Q12: epoch 10 at 32768 x 1 from ctx16384-cont e9, decay at 32768, 464-user scoring (labels s-w64kd-gate-ctx32768c-e10/-dec10).
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/small-model
SM=$WT/artifacts/small-model
TEACHER=C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth
ST=$SM/scale-w64kd-gate-cached-ctx16384-cont-decay-L32k/expertium/E89-F65e-F77a/master.pth
cd "$WT" || exit 1
for pass in 1 2; do
  "$PY" "$QUEUE" run --owner small-model --purpose "small-model: SM24 per-review dumps, fork + ctx16384c-dec9-L32k, 464 users (pass $pass)" \
    --claim gpu --expected-minutes 60 -- "$PY" -X utf8 scripts/small_model_heldout.py --model fork=fork::$TEACHER \
    --model st=expertium:w64:$ST --only fork/w8+s8 --only st/w8+s8 --time-limit 3300 --out $SM/sm24-scores.jsonl --dump $SM/sm24-dump
  n=$(cat $SM/sm24-scores.jsonl 2>/dev/null | wc -l); echo "[chain-b $(date +%H:%M:%S)] SM24 rows after pass $pass: $n"
  [ "$n" -ge 928 ] && break
done
TIER=gate FROM=9 TO=10 CHUNK=32768 BATCH_CHUNKS=1 RUN_TAG=ctx16384-cont SRC_LBL=s-w64kd-gate-ctx16384c FIRST_MIN=270 \
  bash $SM/runbook-2026-10-16b.sh cont > $SM/runbook-2026-10-17b-cont32k.log 2>&1
echo "[chain-b $(date +%H:%M:%S)] cont exit $?"
