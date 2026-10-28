#!/usr/bin/env bash
# SM74 epoch weight averaging (user 10-28 ~17:40, "yeah we can try that"): is the leak-free small model (SM72s) still
#   in one valley at its constant learning rate? Epoch-to-epoch weight steps stay ~11% of the weight norm with no
#   consistent direction since epoch 12, while curve swings +-0.001. If the uniform average of late epochs scores
#   better than every one of them, they sit in one basin, and the average is a free candidate next to the cool-down.
# Arms (458-user set, leak-free scoring exactly as launch-sm72s.sh's 464 epoch scoring):
#   sm74-avg15-20 = mean of epochs 15..20, sm74-avg10-20 = mean of epochs 10..20 (master.pth, float32).
# Start: waits for the sm72s launcher's "epoch 20/20:" line (epoch 20 trained, scored and recorded), so its
#   checkpoint is final; the scoring ticket then goes through the R01 queue (the cool-down from epoch 20 waits ~25
#   min; it is segmented and resumable, so nothing is lost).
# Breaks: quiet.json (scope all/gpu) checked before the ticket; the ticket must end before the next window.
# Scorer code: wt/sm72s (the run's own code, read only). Outputs here.
# Launch: WMI hidden; log artifacts/small-model/sm74/launch-sm74.log. Stop: touch .../sm74/STOP.
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
AVG=C:/Users/jonat/workspace/anki-rwkv/scripts/average_checkpoints.py
S=C:/Users/jonat/workspace/anki-rwkv/artifacts/small-model/sm74
WT=C:/Users/jonat/workspace/wt/sm72s
SC=$WT/scripts/small_model_heldout.py
RUN=$WT/artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes-leakfree/run
CKPT=expertium/E89-F65e-F77a/master.pth
SM72SLOG=$WT/artifacts/small-model/sm72s/launch-sm72s.log
USERS464=$WT/artifacts/small-model/sm72s/eval-users-464-fit.json
OUT=$S/sm74-458.jsonl
LOG=$S/launch-sm74.log
MINUTES=27   # 2 models x ~12 min (launch-sm72s.sh MIN_SCORE464) + margin
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS
log() { echo "[sm74 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$S/STOP" ] && { log "STOP: STOP file present"; return 0; }; return 1; }

free_minutes() {  # minutes to the next quiet.json window (scope all/gpu), 0 inside one
  Q="$QUIET" "$PY" -X utf8 - <<'PYEOF'
import json, os
from datetime import datetime
now = datetime.now()
wins = sorted((datetime.fromisoformat(w["start"]), datetime.fromisoformat(w["end"]))
              for w in json.load(open(os.environ["Q"], encoding="utf-8")) if w.get("scope", "all") in ("all", "gpu"))
for s, e in wins:
    if s <= now < e: print(0); raise SystemExit
    if s > now: print(int((s - now).total_seconds() // 60)); raise SystemExit
print(100000)
PYEOF
}
scored() { F="$OUT" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
p = Path(os.environ['F'])
rows = [json.loads(l) for l in open(p, encoding='utf-8')] if p.exists() else []
print(min(len({r['user_id'] for r in rows if r.get('label') == l and 'benchmark_existing' in r}) for l in ('avg15-20/w8+s8', 'avg10-20/w8+s8')))"; }

log "started"
until grep -q "epoch 20/20:" "$SM72SLOG"; do
  stopped && exit 1
  sleep 120
done
log "epoch 20 recorded; averaging"
ck() { for k in "$@"; do printf '%s ' "$RUN/epoch-$(printf %02d "$k")/$CKPT"; done; }
"$PY" -X utf8 "$AVG" --out "$S/avg15-20/$CKPT" $(ck 15 16 17 18 19 20) >> "$LOG" 2>&1 || { log "STOP: average 15-20 failed"; exit 1; }
"$PY" -X utf8 "$AVG" --out "$S/avg10-20/$CKPT" $(ck 10 11 12 13 14 15 16 17 18 19 20) >> "$LOG" 2>&1 || { log "STOP: average 10-20 failed"; exit 1; }

cd "$WT" || exit 1
pass=1
while [ $pass -le 2 ]; do
  [ "$(scored)" -ge 458 ] && break
  while true; do
    stopped && exit 1
    FREE=$(free_minutes) || { log "quiet.json check failed"; sleep 600; continue; }
    [ "$FREE" -ge $(( MINUTES + 3 )) ] && break
    log "waiting: $FREE min to the next window"; sleep 600
  done
  log "enqueue: scoring pass $pass ($MINUTES min expected, $FREE min to the next window)"
  "$PY" "$QUEUE" run --owner small-model --purpose "small-model: SM74 epoch weight averaging, leak-free scoring on the 458-user set (pass $pass)" \
    --claim gpu --expected-minutes "$MINUTES" --max-wait-minutes $(( FREE - MINUTES - 3 )) --launched-by "detached launcher sm74/launch-sm74.sh" -- \
    env -u RWKV_DELETED_CARDS "$PY" -X utf8 "$SC" --pool 464 --users-file "$USERS464" \
    --model "avg15-20=expertium:w64:$S/avg15-20/$CKPT" --model "avg10-20=expertium:w64:$S/avg10-20/$CKPT" --variants w8+s8 \
    --time-limit 2400 --split-existing --deleted-cards neighbour --out "$OUT" >> "$S/score.out" 2>&1
  rc=$?; log "exit $rc: scoring pass $pass"
  [ "$rc" = 75 ] && { sleep 60; continue; }
  [ "$rc" = 0 ] || { log "STOP: scoring failed"; exit 1; }
  pass=$((pass + 1))
done
log "scored users (min of both arms): $(scored)"
log "done"
