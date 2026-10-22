#!/usr/bin/env bash
# SM58 detached launcher (disagreement test, small vs big model: UM+ and slope). Steps:
#   0. wait for the line "all arms finished" (or "STOP:") in the SM59 launcher log (SM59 check D needs ~17 GB RAM: never run beside it)
#   1. users.json (732 pool minus the long users 1534, 1601, 4625)
#   2. CPU: own-collection predictions of SM36 (full history, float32, 2 threads, ~2-3 min), not during a quiet window
#   3. GPU, via the queue (cap 0.8): big-model dump for the users the 464-user dumps (sm24-dump) do not cover; SM36 dump (all users)
#   4. only if INCLUDE-SM55 exists now: SM55 collection predictions + SM55 dump
#   5. CPU: sm58_umplus.py -> summary.{json,md}, graphs.html, PNGs (not during a quiet window)
# Log lines: "[sm58 ...] done" at the end, "STOP: <reason>" on failure. STOP file in this folder: exit without enqueuing.
# DRYRUN=1: no waiting for SM59, enqueues nothing, prints the plan and commands, ends with "DRYRUN OK".
#   nohup bash launch-sm58.sh > launch-sm58.log 2>&1 & disown
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm58
S=$WT/artifacts/small-model/sm58
SM59LOG=C:/Users/jonat/workspace/wt/sm59/artifacts/small-model/sm59/launch-sm59.log
SM36=C:/Users/jonat/workspace/wt/sm42/artifacts/small-model/scale-w64kd-gate-sc-lanes-cont-decay1ep32k-muon/expertium/E89-F65e-F77a/master.pth
SM55=C:/Users/jonat/workspace/wt/sm55/artifacts/small-model/scale-w64kd-gate-sc-lanes-cont-decay1ep32k-muon-final8k/expertium/E89-F65e-F77a/master.pth
BIG=C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth
DRYRUN=${DRYRUN:-0}
MAXT=60
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HELDOUT_GPU_MEMORY_FRACTION=0.8
log() { echo "[sm58 $(date '+%m-%d %H:%M:%S')] $*"; }
stop() { log "STOP: $*"; exit 1; }
cd "$WT" || stop "no worktree $WT"

# wait_clear MINUTES WHAT: sleep until a job of MINUTES would neither start in nor reach a quiet window
wait_clear() {
  local qw w
  while :; do
    [ -f "$S/STOP" ] && { log "STOP file found; exiting"; exit 0; }
    qw=$("$PY" "$S/quiet_wait.py" "$1"); w=${qw%% *}
    [ "${w:-0}" -gt 0 ] 2>/dev/null || return 0
    [ "$DRYRUN" = 1 ] && { log "DRYRUN: $2 would wait ${w}s (${qw#* })"; return 0; }
    log "waiting ${w}s for $2 (${qw#* })"; sleep $(( w < 1800 ? w : 1800 ))
  done
}

if [ "$DRYRUN" != 1 ]; then
  log "started; waiting for SM59 ($SM59LOG)"
  n=0
  while :; do
    [ -f "$S/STOP" ] && { log "STOP file found; exiting"; exit 0; }
    if grep -qE '\[sm59-launch [^]]*\] (all arms finished|STOP:)' "$SM59LOG" 2>/dev/null; then log "SM59 log: $(grep -E '\[sm59-launch [^]]*\] (all arms finished|STOP:)' "$SM59LOG" | tail -1 | cut -c1-120)"; break; fi
    [ $((n % 24)) -eq 0 ] && log "waiting for SM59 ($(tail -1 "$SM59LOG" 2>/dev/null | cut -c1-120))"
    n=$((n + 1)); sleep 300
  done
else
  log "DRYRUN: not waiting for SM59"
fi

for f in "$SM36" "$BIG"; do [ -f "$f" ] || stop "missing checkpoint $f"; done
"$PY" -X utf8 "$S/sm58_plan.py" users > "$S/users.count" || stop "users.json failed"
log "users: $(cat "$S/users.count") (users, reviews); long users 1534/1601/4625 skipped"

collect() {  # collect TAG CKPT
  wait_clear 15 "own-collection scoring of $1 (CPU)"
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $PY $S/sm58_own_preds.py --tag $1 --checkpoint $2"; return 0; fi
  free=$(powershell -NoProfile -c "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)" | tr -dc '0-9')
  [ "${free:-0}" -ge 6 ] || log "low free RAM ($free GB), scoring anyway (needs ~3 GB)"
  "$PY" -X utf8 "$S/sm58_own_preds.py" --tag "$1" --checkpoint "$2" > "$S/collection-$1.log" 2>&1 || stop "collection scoring of $1 failed (see collection-$1.log)"
  log "collection $1: $(tail -1 "$S/collection-$1.log")"
}

dump() {  # dump TAG KIND RATE CKPT   (KIND fork => --model fork=fork::CKPT)
  local tag=$1 rate=$3 ckpt=$4 spec nusers nrev minutes t rc
  if [ "$2" = fork ]; then spec="fork=fork::$ckpt"; else spec="$tag=expertium:w64:$ckpt"; fi
  for pass in 1 2; do
    read -r nusers nrev minutes <<< "$("$PY" -X utf8 "$S/sm58_plan.py" todo "$tag" "$rate" "$S/todo-$tag.json")"
    [ "${nusers:-0}" = 0 ] && { log "dump $tag: complete (failed users: $(cat "$S/failed-$tag.json"))"; return 0; }
    t=$(( minutes > MAXT ? MAXT : minutes ))
    log "dump $tag pass $pass: $nusers users / $nrev reviews, $minutes min total, ticket $t min"
    wait_clear "$t" "GPU ticket dump-$tag"
    cmd=("$PY" -X utf8 "$WT/scripts/small_model_heldout.py" --model "$spec" --only "$tag/w8+s8" --pool 732 --users-file "$S/todo-$tag.json"
         --time-limit $(( (t - 2) * 60 )) --out "$S/heldout-sm58.jsonl" --dump "$S/dump")
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue (expected $t min): ${cmd[*]}"; return 0; fi
    "$PY" "$QUEUE" run --owner sm58 --purpose "SM58 disagreement test: per-review dump of $tag, $nusers users (pass $pass), cap 0.8" \
      --claim gpu --expected-minutes "$t" --launched-by "detached launcher launch-sm58.sh" -- "${cmd[@]}"
    rc=$?; log "dump $tag pass $pass exit $rc"
    [ "$rc" != 0 ] && stop "GPU ticket for $tag failed (exit $rc)"
  done
  read -r nusers nrev minutes <<< "$("$PY" -X utf8 "$S/sm58_plan.py" todo "$tag" "$rate" "$S/todo-$tag.json")"
  log "dump $tag: $nusers users still without a dump after 2 passes (failed: $(cat "$S/failed-$tag.json"))"
}

collect sm36 "$SM36"
dump fork fork 29 "$BIG"
dump sm36 st 11 "$SM36"
MODELS=sm36
if [ -f "$S/INCLUDE-SM55" ]; then
  [ -f "$SM55" ] || stop "INCLUDE-SM55 exists but no checkpoint $SM55"
  log "INCLUDE-SM55 present: adding the 8k model (SM55)"
  collect sm55 "$SM55"
  dump sm55 st 11 "$SM55"
  MODELS=sm36,sm55
else
  log "no INCLUDE-SM55: SM36 vs big only"
fi
wait_clear 30 "analysis (CPU)"
if [ "$DRYRUN" = 1 ]; then
  log "DRYRUN: $PY $S/sm58_umplus.py --models $MODELS"; log "DRYRUN OK"; exit 0
fi
"$PY" -X utf8 "$S/sm58_umplus.py" --models "$MODELS" > "$S/umplus.log" 2>&1 || stop "analysis failed (see umplus.log)"
log "analysis: $(tail -1 "$S/umplus.log")"
log "done (models $MODELS; outputs summary.json, summary.md, graphs.html, um-plus-*.png in $S)"
