#!/usr/bin/env bash
# Big long cards with 6-bit factors (SM165): F6 (w8+lr2.16.4q6x8) rows for the 50 long-card users of the low-rank int8
# states test (SM148), Big two-head (SM91s2) and Big Instant-Only (SM99), CPU stream engine (bitwise equal to the stock
# CPU scorer), flags of the bounded-memory long-card scorer (SM155) first use. design.md is the pre-registration.
#   bash launch-sm165.sh          main: preconditions, then ticket 1 (gate) and ticket 2 (pool) through the queue
#   bash launch-sm165.sh gate     ticket body: stock vs stream on users 2388 + 2883 (F6, T24) + reused-row check
#   bash launch-sm165.sh pool     ticket body: one fresh process per user (both models, F6), WORKERS at a time
#   DRYRUN=1 bash launch-sm165.sh    logs the commands, scores nothing, no queue
# Resumable: finished users (cpu/users/u-<user>.jsonl) and cpu/gate/ok are kept. Stop between users: touch cpu/STOP.
# Log cpu/launch-sm165.log; end lines "SM165 DONE" (all 50 users) or "SM165 STOP" (cause on an earlier "STOP:" line).
set -u
MODE=${1:-main}
WT=C:/Users/jonat/workspace/wt/sm165
S=$WT/artifacts/small-model/sm165
C=$S/cpu
D=$C/users
G=$C/gate
GW=$WT/work/sm165-gate   # gate rows + dumps (dumps never committed); rows and outs copied to cpu/gate
DUMP=$WT/work/sm165-dump
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
SC=$WT/scripts/small_model_heldout.py
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
BASH_EXE="C:/Program Files/Git/usr/bin/bash.exe"
CBLOCK=$WT/artifacts/small-model/sm136/sm136_block.py
CMP=$WT/artifacts/small-model/sm155/sm155_compare.py
NAME="Big long cards with 6-bit factors (SM165)"
WORKERS=${SM165_WORKERS:-3}
DRYRUN=${DRYRUN:-0}
CHUNK=16384
F6=w8+lr2.16.4q6x8
T24=w8+lr2.16.4q8x8
mkdir -p "$D" "$G" "$GW" "$DUMP"
LOG=$C/launch-sm165.log
[ "$DRYRUN" = 1 ] && LOG=$C/launch-sm165-dryrun.log
log() { echo "[sm165 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
source "$WT/artifacts/small-model/sm148/models.sh"
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START SM11_INPUTS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4   # as SM148 / SM155 (the scorer sets 2 torch threads itself)
FLAGS=(--inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets 0,1,3,5,9,16,32,128,300,1000)
mspec() { echo "$MODELS" | awk -F'|' -v t="$1" '$1 == t {print $1 "=" $2 ":" $3}'; }
msha() { echo "$MODELS" | awk -F'|' -v t="$1" '$1 == t {print $3 " " $4}'; }
BIG2=sm91s2-big-showtime
BIGI=sm99-big-inst-e2
PARGS=(--model "$(mspec $BIG2)" --only "$BIG2/$F6" --model "$(mspec $BIGI)" --only "$BIGI/$F6")
# the 50 long-card users (artifacts/small-model/sm148/users-longcard.json), largest first (review counts of the reused rows)
USERS="4965 2289 1714 1399 596 2393 1717 4912 4132 2119 4566 1015 1458 3260 4620 4479 954 2807 1497 4282 3982 3220 1678 4380 \
1376 4898 4540 3876 1331 1624 2499 563 1323 4896 2820 3152 3432 446 2824 4374 4727 4811 966 3502 4407 280 1215 3882 2883 1866"

score() {  # score OUTFILE USERSFILE ENGINE [extra]: one scorer process
  local out=$1 users=$2 eng=$3; shift 3
  if [ "$DRYRUN" = 1 ]; then echo "    DRYRUN cmd: $SC --device cpu --pool 732 --users-file $users $* ${FLAGS[*]} --dump-scope all --engine $eng --chunk-rows $CHUNK --out $out" >> "$LOG"; return 0; fi
  CUDA_VISIBLE_DEVICES= PYTORCH_JIT=0 "$PY" -X utf8 -W ignore "$SC" --device cpu --pool 732 --users-file "$users" \
    "$@" "${FLAGS[@]}" --time-limit 1000000 --dump-scope all --engine "$eng" --chunk-rows $CHUNK --out "$out"
}

wait_quiet() {  # hold while a quiet window (scope all / cpu) is on, or one starts within MINUTES
  local b
  while true; do
    [ -e "$C/STOP" ] && return 1
    b=$("$PY" -X utf8 "$CBLOCK" --scopes all,cpu | tr -d '\r')
    case "$b" in
      NOW*) log "quiet window, ${b#NOW }s left"; sleep "$(( ${b#NOW } > 600 ? 600 : ${b#NOW } ))";;
      NEXT*) [ "${b#NEXT }" -ge "$1" ] && return 0; log "next quiet window in ${b#NEXT } min (< $1 min needed); waiting"; sleep 300;;
      *) log "window check failed: $b"; sleep 600;;
    esac
  done
}

gate_body() {  # stock vs stream bitwise on 2388 + 2883 (Big two-head (SM91s2), F6 and T24), then T24 vs reused rows
  local eng
  [ -f "$G/ok" ] && { log "gate: already passed"; return 0; }
  echo '{"users": [2388, 2883]}' > "$G/users.json"
  for eng in parallel stream; do
    rm -rf "$GW/dump-$eng" "$GW/$eng.jsonl"
    score "$GW/$eng.jsonl" "$G/users.json" "$eng" --model "$(mspec $BIG2)" --only "$BIG2/$F6" --only "$BIG2/$T24" \
      --dump "$GW/dump-$eng" > "$G/$eng.out" 2>&1 || { log "STOP: gate $eng exit $?"; return 1; }
    cp "$GW/$eng.jsonl" "$G/$eng.jsonl" 2>/dev/null
  done
  [ "$DRYRUN" = 1 ] && return 0
  "$PY" -X utf8 "$CMP" "$GW" > "$G/compare.txt" 2>&1
  log "gate: $(tail -1 "$G/compare.txt")"
  grep -q "^4/4 pass; worst max|dp| 0.00e+00" "$G/compare.txt" || { log "STOP: gate not bitwise equal (cpu/gate/compare.txt)"; return 1; }
  "$PY" -X utf8 "$S/sm165_report.py" gate-reuse > "$G/reuse.txt" 2>&1 || { log "STOP: gate T24 rows differ from the reused rows ($(tail -1 "$G/reuse.txt"))"; return 1; }
  log "gate: $(tail -1 "$G/reuse.txt")"
  touch "$G/ok"
}

one() {  # one USER: both Big models, F6, stream engine, one fresh process
  local u=$1 t0 rc
  echo "{\"users\": [$u]}" > "$D/users-$u.json"
  rm -f "$D/u-$u.part.jsonl" "$DUMP/${u}_"*.npz
  t0=$(date +%s)
  score "$D/u-$u.part.jsonl" "$D/users-$u.json" stream "${PARGS[@]}" --dump "$DUMP" > "$D/score-$u.out" 2>&1
  rc=$?
  [ "$DRYRUN" = 1 ] && return 0
  if [ $rc = 0 ] && [ "$(grep -c '"benchmark"\|"nonfinite"' "$D/u-$u.part.jsonl")" = 2 ]; then mv "$D/u-$u.part.jsonl" "$D/u-$u.jsonl"
    log "user $u done in $(( ($(date +%s) - t0) / 60 )) min: $(grep -a '^\[small-model\] 1/1' "$D/score-$u.out" | tail -1 | cut -c1-110)"
  elif grep -q "u-$u.part.jsonl" "$C/memguard-kill" 2>/dev/null; then
    echo "$u killed by the memory guard (exit $rc)" >> "$D/killed.txt"; rm -f "$C/STOP"; log "user $u killed by the memory guard"
  else echo "$u exit $rc" >> "$D/failed.txt"; log "user $u failed (exit $rc, see cpu/users/score-$u.out)"; fi
}

pool_body() {
  local wp MG u p alive PIDS=() n=0
  wp=$(cat /proc/$$/winpid 2>/dev/null || echo 0)
  if [ "$DRYRUN" != 1 ]; then
    rm -f "$C/memguard.off"
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$S/memguard.ps1")" -Dir "$(cygpath -w "$C")" \
      -ParentPid "$wp" -Match "small_model_heldout.py&--device cpu&sm165" > /dev/null 2>&1 &
    MG=$!
  fi
  log "pool: start (workers $WORKERS, chunk rows $CHUNK, label $F6)"
  for u in $USERS; do
    [ -f "$D/u-$u.jsonl" ] && continue
    grep -q "^$u " "$D/killed.txt" 2>/dev/null && continue
    if [ "$DRYRUN" = 1 ]; then n=$((n + 1)); [ $n -le 2 ] && one "$u"; continue; fi
    while true; do   # a free slot
      alive=(); for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive+=("$p"); done; PIDS=("${alive[@]}")
      [ ${#PIDS[@]} -lt "$WORKERS" ] && break; sleep 20
    done
    wait_quiet 20 || break
    one "$u" & PIDS+=($!)
    sleep 30   # stagger the preparation peaks
  done
  [ "$DRYRUN" = 1 ] && { log "DRYRUN pool: $n users to score"; return 0; }
  [ ${#PIDS[@]} -gt 0 ] && wait "${PIDS[@]}"   # only our scorer jobs, never the memory guard (SM155 launcher bug)
  touch "$C/memguard.off"; sleep 12; kill "$MG" 2>/dev/null
  cat "$D"/u-*.jsonl > "$C/sm165-f6-rows.jsonl" 2>/dev/null
  log "pool: end, $(ls "$D"/u-*.jsonl 2>/dev/null | wc -l) of 50 users"
}

q() {  # q MINUTES MODE PURPOSE: one queue ticket (cpu-heavy) running this script in MODE
  local minutes=$1 mode=$2 purpose=$3 rc t0
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN ticket: $NAME: $purpose (claim cpu-heavy, expected $minutes min)"; "${mode}_body"; return $?; fi
  log "enqueue: $purpose (claim cpu-heavy, expected $minutes min)"
  t0=$(date +%s)
  "$PY" "$QUEUE" run --owner sm165 --purpose "$NAME: $purpose" --claim cpu-heavy --expected-minutes "$minutes" \
    --launched-by "detached launcher sm165/launch-sm165.sh" -- "$BASH_EXE" "$S/launch-sm165.sh" "$mode" >> "$C/queue.out" 2>&1
  rc=$?
  log "exit $rc: $purpose ($(( ($(date +%s) - t0 + 30) / 60 )) min; $(grep -a 'registered for' "$C/queue.out" | tail -1 | cut -c1-40))"
  return $rc
}

case "$MODE" in
  gate) export PATH="/usr/bin:/mingw64/bin:$PATH"; gate_body; exit $? ;;
  pool) export PATH="/usr/bin:/mingw64/bin:$PATH"; pool_body; exit $? ;;
  main) ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac

rm -f "$C/done" "$C/failed" "$C/stopped"
log "started: $(git -C "$WT" log --oneline -1 | cut -c1-90) (DRYRUN=$DRYRUN, workers $WORKERS)"
if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv artifacts/small-model/sm165/*.sh artifacts/small-model/sm165/*.py artifacts/small-model/sm165/*.ps1 artifacts/small-model/sm165/design.md)" ]; then
  log "STOP: source or sm165 files not committed"; log "SM165 STOP"; touch "$C/failed"; exit 1; fi
for t in $BIG2 $BIGI; do
  set -- $(msha $t)
  [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ] || { log "STOP: sha256 mismatch for $t"; log "SM165 STOP"; touch "$C/failed"; exit 1; }
done
log "checkpoints: sha256 OK"
if ! q 10 gate "bitwise gate, users 2388 + 2883, F6 and T24, stock vs stream (CPU)"; then
  log "STOP: gate failed (cpu/gate/compare.txt, cpu/gate/reuse.txt); nothing scored"; log "SM165 STOP"; touch "$C/failed"; exit 1; fi
q 90 pool "F6 on the 50 long-card users, Big two-head (SM91s2) and Big Instant-Only (SM99), CPU stream engine" || log "pool ticket exit $?"
[ "$DRYRUN" = 1 ] && { log "DRYRUN end"; exit 0; }
n=$(ls "$D"/u-*.jsonl 2>/dev/null | wc -l)
if [ -e "$C/STOP" ]; then touch "$C/stopped"; log "STOP: stopped by cpu/STOP ($n of 50 users)"; log "SM165 STOP"
elif [ "$n" = 50 ]; then touch "$C/done"; log "SM165 DONE (50 of 50 users)"
else touch "$C/failed"; log "STOP: ended with $n of 50 users (cpu/users/killed.txt, failed.txt)"; log "SM165 STOP"; fi
