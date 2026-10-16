#!/usr/bin/env bash
# 2026-10-16 Q5 (S05 batched scoring check) -> Q6 (ctx4096 decay probe from e6), after Q4 extend is done.
# Plan: docs/SMALL_MODEL.md "Queue 2026-10-16" rows Q5/Q6 and "Re-assessment 2026-10-16 ~03:30". Exit codes: read the [queue] lines.
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/small-model
S05=C:/Users/jonat/workspace/wt/s05-batched
SM=$WT/artifacts/small-model
SC=$WT/scripts/small_model_heldout.py
DEC_PY=$WT/scripts/expertium_decay.py
CKPT=expertium/E89-F65e-F77a/master.pth
TEACHER=C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth
RUN=$WT/artifacts/accuracy/experiments/scale-w64kd-gate-cached-ctx2048
LBL=s-w64kd-gate-ctx2048
K=6
log() { echo "[chain56 $(date +%H:%M:%S)] $*"; }
until grep -q "done: extend" $SM/runbook-2026-10-16-extend.log; do
  if grep -q -E "stop|failed" $SM/runbook-2026-10-16-extend.log && ! pgrep -f runbook-2026-10-16.sh >/dev/null 2>&1 && ! ps -ef | grep -v grep | grep -q runbook-2026-10-16.sh; then log "extend ended without done (see its log); continuing with Q5 only"; K=; break; fi
  sleep 60
done
now=$(date +%H%M)
log "Q4 finished; Q5 S05 batched scoring check"
if [ "$now" -lt 0800 ]; then
  $PY $QUEUE run --owner small-model --purpose "small-model: S05 batched scoring tolerance + timing" --claim gpu --timing --expected-minutes 12 -- $PY -X utf8 $S05/scripts/small_model_heldout.py --users-file $SM/gate-users.json --batch-users 8 --model $LBL-e2=expertium:w64:$RUN/run/epoch-02/$CKPT --only $LBL-e2/w8+s8 --out $SM/s05-check.jsonl
  rc=$?; log "Q5 queue exit $rc"
  $PY -X utf8 $S05/scripts/small_model_batch_compare.py --reference $SM/heldout.jsonl --candidate $SM/s05-check.jsonl --label $LBL-e2/w8+s8 --allow-missing-input-sha --out $SM/s05-gpu-check.json
  log "Q5 compare exit $?"
else
  log "after 08:00: Q5 skipped (timing window), moves to after 20:00"
fi
[ -n "$K" ] || { log "no e6: Q6 skipped"; exit 1; }
[ -e "$RUN/run/epoch-0$K/$CKPT" ] || { log "no epoch-0$K checkpoint: Q6 skipped"; exit 1; }
log "Q6 prepare cal-ctx4096-e$K"
$PY -X utf8 $DEC_PY prepare --source-run $RUN/run --epoch $K --out $SM/cal-ctx4096-e$K --fraction 0.25 --shape one_minus_sin --evaluate none --seed 1 --extra-options "{\"distillation\": {\"teacher\": \"$TEACHER\", \"alpha\": 0.9, \"decay_alpha\": 0.5, \"cache\": \"C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-teacher/fork-bf16-s2025\"}, \"chunk_reviews\": 4096, \"batch_chunks\": 8}" || { log "Q6 prepare failed"; exit 1; }
$PY $QUEUE run --owner small-model --purpose "small-model: Q6 ctx4096 decay probe from e$K" --claim gpu --expected-minutes 40 -- "C:/Program Files/Git/usr/bin/bash.exe" -c "'$PY' -X utf8 '$DEC_PY' smoke --out '$SM/cal-ctx4096-e$K' && '$PY' -X utf8 '$DEC_PY' run --out '$SM/cal-ctx4096-e$K'"
rc=$?; log "Q6 decay queue exit $rc"
[ -e "$SM/cal-ctx4096-e$K/$CKPT" ] || { log "no Q6 checkpoint: scoring skipped"; exit 1; }
$PY $QUEUE run --owner small-model --purpose "small-model: Q6 scoring cal-ctx4096-e$K (464 users)" --claim gpu --expected-minutes 16 -- $PY -X utf8 $SC --model cal-ctx4096-e$K=expertium:w64:$SM/cal-ctx4096-e$K/$CKPT --only cal-ctx4096-e$K/w8+s8 --time-limit 3000 --out $SM/heldout.jsonl
log "Q6 scoring queue exit $?"
log "done"
