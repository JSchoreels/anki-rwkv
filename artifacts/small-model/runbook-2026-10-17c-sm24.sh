#!/usr/bin/env bash
# SM24 after chain b's cont stage (its first SM24 step failed: the queue needs absolute script paths). Waits for
# "cont exit" in runbook-2026-10-17b-chain.log, then dumps fork w8+s8 and ctx16384c-dec9-L32k per review on 464 users
# (separate jsonl; heldout.jsonl untouched), up to 2 passes.
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/small-model
SM=$WT/artifacts/small-model
TEACHER=C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth
ST=$SM/scale-w64kd-gate-cached-ctx16384-cont-decay-L32k/expertium/E89-F65e-F77a/master.pth
until grep -q "cont exit" $SM/runbook-2026-10-17b-chain.log; do sleep 120; done
for pass in 1 2; do
  "$PY" "$QUEUE" run --owner small-model --purpose "small-model: SM24 per-review dumps, fork + ctx16384c-dec9-L32k, 464 users (pass $pass)" \
    --claim gpu --expected-minutes 60 -- "$PY" -X utf8 $WT/scripts/small_model_heldout.py --model fork=fork::$TEACHER \
    --model st=expertium:w64:$ST --only fork/w8+s8 --only st/w8+s8 --time-limit 3300 --out $SM/sm24-scores.jsonl --dump $SM/sm24-dump
  n=$(cat $SM/sm24-scores.jsonl 2>/dev/null | wc -l); echo "[sm24 $(date +%H:%M:%S)] rows after pass $pass: $n"
  [ "$n" -ge 928 ] && break
done
"$PY" -X utf8 $SM/feature-screen/sm24_bytype.py > $SM/feature-screen/sm24-bytype.txt 2>&1
echo "[sm24 $(date +%H:%M:%S)] done"
