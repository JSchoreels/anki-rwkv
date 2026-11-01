#!/usr/bin/env bash
# 2026-10-19 00:40 (user: "you can go on, I won't use the comp until tomorrow"): GPU chain after the SM40v2 runbook
# (wt/sm21v2, one-epoch 32k decay of carried state v2). Steps, each one a set of queue tickets:
#   1. 732 pool: retry the 3 long users (1534, 1601, 4625) that ran out of memory at the 0.8 cap, at 0.92, one pass per label
#      (fork, dec10, carry v1 A / B, v2 dec12, SM40v2 if its checkpoint exists).
#   2. Spaced-review slice (SM24) for the new students on the 464 users: per-review dumps of SM40v2 and v2 dec12 (fork dumps
#      of 10-17 are reused), then the by-type / same-day / stage breakdown -> feature-screen/sm24-bytype-<name>.{txt,json}.
#   3. 2,225 pool at the 0.8 cap: fork, dec10, v2 dec12, SM40v2 (~4 h).
#   4. 2,225 pool: one 0.92 retry for the users that ran out of memory in step 3.
# Hold: create $SM/HOLD-chain to pause before the next step (a running ticket is not touched); delete it to resume.
# Launch: nohup bash <this file> > $SM/launch-chain-after-sm40v2.log 2>&1 & disown
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/small-model
SM=$WT/artifacts/small-model
V2W=C:/Users/jonat/workspace/wt/sm21v2/artifacts/small-model
SM40LOG=$V2W/runbook-2026-10-18-sm40v2.log
CKPT=expertium/E89-F65e-F77a/master.pth
POOLS="$SM/runbook-eval-pools.sh"
ONLY2225='^fork|ctx32768c-dec10|sc-lanes-dec12'
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
log() { echo "[chain $(date '+%m-%d %H:%M:%S')] $*"; }
hold() { local said=0; while [ -f "$SM/HOLD-chain" ]; do [ "$said" = 0 ] && log "HOLD-chain present: waiting" && said=1; sleep 120; done; }
cd "$WT" || exit 1

log "waiting for the SM40v2 runbook to end"
until grep -aqE "^\[sm40v2 [0-9: -]+\] (done|ended with exit)" "$SM40LOG" 2>/dev/null; do sleep 120; done
log "SM40v2: $(grep -aE '^\[sm40v2 [0-9: -]+\] (done|ended with exit)' "$SM40LOG" | tail -1)"

hold; log "step 1: 732 pool retry of the long users at 0.92"
HELDOUT_GPU_MEMORY_FRACTION=0.92 PASSES=1 POOL=732 bash "$POOLS"
log "step 1 exit $?"

hold; log "step 2: SM24 spaced-review slice dumps (464 users)"
for entry in "sm40v2|$V2W/scale-w64kd-gate-sc-lanes-cont-decay1ep32k/$CKPT" "v2dec12|$V2W/scale-w64kd-gate-sc-lanes-cont-decay/$CKPT"; do
  IFS='|' read -r name ckpt <<< "$entry"
  [ -f "$ckpt" ] || { log "SKIP SM24 $name: no checkpoint $ckpt"; continue; }
  out=$SM/sm24-scores-$name.jsonl; dump=$SM/sm24-dump-$name
  for pass in 1 2; do
    n=$(grep -c benchmark "$out" 2>/dev/null || echo 0); [ "$n" -ge 464 ] && break
    "$PY" "$QUEUE" run --owner small-model --purpose "small-model: SM24 per-review dumps, $name, 464 users (pass $pass)" \
      --claim gpu --expected-minutes 30 -- "$PY" -X utf8 $WT/scripts/small_model_heldout.py --model st=expertium:w64:$ckpt \
      --only st/w8+s8 --time-limit 1680 --out "$out" --dump "$dump"
    log "SM24 $name pass $pass exit $?, rows $(grep -c benchmark "$out" 2>/dev/null || echo 0)"
  done
  SM24_DUMP=$dump SM24_FORK_DUMP=$SM/sm24-dump SM24_JSON=$SM/feature-screen/sm24-bytype-$name.json \
    "$PY" -X utf8 $SM/feature-screen/sm24_bytype.py > $SM/feature-screen/sm24-bytype-$name.txt 2>&1
  log "SM24 $name breakdown exit $? -> feature-screen/sm24-bytype-$name.txt"
done

hold; log "step 3: 2,225 pool at the 0.8 cap ($ONLY2225)"
ONLY="$ONLY2225" POOL=2225 bash "$POOLS"
log "step 3 exit $?"

hold; log "step 4: 2,225 pool retry of out-of-memory users at 0.92"
HELDOUT_GPU_MEMORY_FRACTION=0.92 PASSES=1 ONLY="$ONLY2225" POOL=2225 bash "$POOLS"
log "step 4 exit $?"
log "chain done"
