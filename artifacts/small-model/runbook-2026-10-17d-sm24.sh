#!/usr/bin/env bash
# SM24 (replaces runbook-2026-10-17c-sm24.sh, which only waited): after chain b's cont stage, dump fork w8+s8 and the
# newest student per review on the 464 users and break the gap down by review type / same-day (user 2026-10-17: report
# the review-state slice beside the bar). Student = Q12 dec10 (scale-w64kd-gate-cached-ctx32768-cont-decay) if it
# exists, else ctx16384c-dec9-L32k; tag "st" either way, the checkpoint path is in every sm24-scores.jsonl row.
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/small-model
SM=$WT/artifacts/small-model
TEACHER=C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth
CKPT=expertium/E89-F65e-F77a/master.pth
until grep -q "cont exit" $SM/runbook-2026-10-17b-chain.log; do sleep 120; done
ST=$SM/scale-w64kd-gate-cached-ctx32768-cont-decay/$CKPT; NAME=ctx32768c-dec10
[ -f "$ST" ] || { ST=$SM/scale-w64kd-gate-cached-ctx16384-cont-decay-L32k/$CKPT; NAME=ctx16384c-dec9-L32k; }
echo "[sm24 $(date +%H:%M:%S)] student $NAME = $ST"
for pass in 1 2; do
  "$PY" "$QUEUE" run --owner small-model --purpose "small-model: SM24 per-review dumps, fork + $NAME, 464 users (pass $pass)" \
    --claim gpu --expected-minutes 60 -- "$PY" -X utf8 $WT/scripts/small_model_heldout.py --model fork=fork::$TEACHER \
    --model st=expertium:w64:$ST --only fork/w8+s8 --only st/w8+s8 --time-limit 3300 --out $SM/sm24-scores.jsonl --dump $SM/sm24-dump
  n=$(cat $SM/sm24-scores.jsonl 2>/dev/null | wc -l); echo "[sm24 $(date +%H:%M:%S)] rows after pass $pass: $n"
  [ "$n" -ge 928 ] && break
done
"$PY" -X utf8 $SM/feature-screen/sm24_bytype.py > $SM/feature-screen/sm24-bytype.txt 2>&1
echo "[sm24 $(date +%H:%M:%S)] done ($NAME)"
