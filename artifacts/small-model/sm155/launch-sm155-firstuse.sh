#!/usr/bin/env bash
# Bounded-memory long-card scorer (SM155): first use (decided by the user; NOT started on 11-24).
# Both Big models (Big two-head (SM91s2), Big Instant-Only (SM99)) x {T24 w8+lr2.16.4q8x8, REF w8+s8hx8} with --engine
# stream and the long-card flags of the low-rank int8 states test (SM148), on
#   1. the 13 Big long-card users the stock CPU scorer could not fit in SM148 (596 1015 1399 1714 1717 2119 2289 2393
#      4132 4479 4566 4912 4965; 87k to 151k reviews), then
#   2. the 5 Big users that never fit on the GPU (296 2073 2267 3735 4852; 424k to 531k reviews).
# One fresh process per user (both models, both labels), LC_WORKERS at a time (default 2), memory guard 5.9 GB per CPU
# process, quiet.json windows of scope all / cpu respected (artifacts/small-model/sm136/sm136_block.py). First a gate:
# user 2388, Big two-head (SM91s2), REF and T24, stock vs stream must be bitwise equal in this launcher's environment.
# Resumable: finished users (firstuse/big/u-<user>.jsonl) are skipped. Stop between users: touch firstuse/STOP.
# Ends with firstuse/done (all rows), firstuse/failed (some user failed or was killed) or firstuse/stopped.
set -u
WT=C:/Users/jonat/workspace/wt/sm155
S=$WT/artifacts/small-model/sm155
F=$S/firstuse
D=$F/big
DUMP=$WT/work/sm155-firstuse-dump
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
SC=$WT/scripts/small_model_heldout.py
CBLOCK=$WT/artifacts/small-model/sm136/sm136_block.py
LC_WORKERS=${LC_WORKERS:-2}
CHUNK=16384
mkdir -p "$D" "$DUMP"
rm -f "$F/done" "$F/failed" "$F/stopped"
LOG=$F/launch-firstuse.log
log() { echo "[sm155 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
source "$WT/artifacts/small-model/sm148/models.sh"
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START SM11_INPUTS
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4   # as the SM148 CPU chain (the scorer sets 2 torch threads itself)
FLAGS=(--inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets 0,1,3,5,9,16,32,128,300,1000)
GARGS=()
for tag in sm91s2-big-showtime sm99-big-inst-e2; do
  GARGS+=(--model "$(echo "$MODELS" | awk -F'|' -v t="$tag" '$1 == t {print $1 "=" $2 ":" $3}')")
  GARGS+=(--only "$tag/w8+lr2.16.4q8x8" --only "$tag/w8+s8hx8")
done
USERS="4479 1015 4566 2119 4132 4912 1717 2393 596 1399 1714 2289 4965 2073 296 2267 4852 3735"   # by review count within each group

score() {  # score OUTFILE USERSFILE ENGINE [extra]: one scorer process
  local out=$1 users=$2 eng=$3; shift 3
  CUDA_VISIBLE_DEVICES= PYTORCH_JIT=0 "$PY" -X utf8 -W ignore "$SC" --device cpu --pool 732 --users-file "$users" \
    "$@" "${FLAGS[@]}" --time-limit 1000000 --dump-scope all --engine "$eng" --chunk-rows $CHUNK --out "$out"
}

wait_quiet() {  # hold while a quiet window (scope all / cpu) is on, or one starts within MINUTES
  local b
  while true; do
    [ -e "$F/STOP" ] && return 1
    b=$("$PY" -X utf8 "$CBLOCK" --scopes all,cpu | tr -d '\r')
    case "$b" in
      NOW*) log "quiet window, ${b#NOW }s left"; sleep "$(( ${b#NOW } > 600 ? 600 : ${b#NOW } ))";;
      NEXT*) [ "${b#NEXT }" -ge "$1" ] && return 0; log "next quiet window in ${b#NEXT } min (< $1 min needed); waiting"; sleep 300;;
      *) log "window check failed: $b"; sleep 600;;
    esac
  done
}

gate() {   # bitwise gate in this environment: user 2388, Big two-head (SM91s2), REF and T24
  local g=$F/gate eng
  [ -f "$g/ok" ] && return 0
  mkdir -p "$g"; echo '{"users": [2388]}' > "$g/users.json"
  for eng in parallel stream; do
    score "$g/$eng.jsonl" "$g/users.json" "$eng" --model "${GARGS[1]}" --only "sm91s2-big-showtime/w8+lr2.16.4q8x8" \
      --only "sm91s2-big-showtime/w8+s8hx8" --dump "$g/dump-$eng" > "$g/$eng.out" 2>&1 || { log "gate: $eng exit $?"; return 1; }
  done
  "$PY" -X utf8 "$S/sm155_compare.py" "$g" > "$g/compare.txt" 2>&1
  log "gate: $(tail -1 "$g/compare.txt")"
  grep -q "^2/2 pass; worst max|dp| 0.00e+00" "$g/compare.txt" && touch "$g/ok"
}

one() {  # one USER: both Big models x {T24, REF}, stream engine, one fresh process
  local u=$1 t0 rc
  echo "{\"users\": [$u]}" > "$D/users-$u.json"
  rm -f "$D/u-$u.part.jsonl" "$DUMP/${u}_"*.npz
  t0=$(date +%s)
  score "$D/u-$u.part.jsonl" "$D/users-$u.json" stream "${GARGS[@]}" --dump "$DUMP" > "$D/score-$u.out" 2>&1
  rc=$?
  if [ $rc = 0 ]; then mv "$D/u-$u.part.jsonl" "$D/u-$u.jsonl"
    log "user $u done in $(( ($(date +%s) - t0) / 60 )) min: $(grep -a '^\[small-model\] 1/1' "$D/score-$u.out" | tail -1 | cut -c1-110)"
  elif grep -q "u-$u.part.jsonl" "$F/memguard-kill" 2>/dev/null; then
    echo "$u killed by the memory guard (exit $rc)" >> "$D/killed.txt"; rm -f "$F/STOP"; log "user $u killed by the memory guard"
  else echo "$u exit $rc" >> "$D/failed.txt"; log "user $u failed (exit $rc, see $D/score-$u.out)"; fi
}

log "start (workers $LC_WORKERS, chunk rows $CHUNK)"
wp=$(cat /proc/$$/winpid 2>/dev/null || echo 0)
rm -f "$F/memguard.off"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$S/memguard.ps1")" -Dir "$(cygpath -w "$F")" \
  -ParentPid "$wp" -Match "small_model_heldout.py&--device cpu&sm155" > /dev/null 2>&1 &
MG=$!
if ! gate; then log "STOP: the bitwise gate failed (firstuse/gate/compare.txt); nothing scored"; touch "$F/failed"
  touch "$F/memguard.off"; kill $MG 2>/dev/null; exit 1; fi
PIDS=()
for u in $USERS; do
  [ -f "$D/u-$u.jsonl" ] && continue
  grep -q "^$u " "$D/killed.txt" 2>/dev/null && continue
  while true; do   # a free slot
    alive=(); for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive+=("$p"); done; PIDS=("${alive[@]}")
    [ ${#PIDS[@]} -lt "$LC_WORKERS" ] && break; sleep 20
  done
  wait_quiet 90 || break
  one "$u" & PIDS+=($!)
  sleep 60   # stagger the preparation peaks
done
wait
touch "$F/memguard.off"; sleep 12; kill $MG 2>/dev/null
cat "$D"/u-*.jsonl > "$F/sm155-big-longcard-rows.jsonl" 2>/dev/null
n=$(ls "$D"/u-*.jsonl 2>/dev/null | wc -l)
if [ -e "$F/STOP" ]; then touch "$F/stopped"; log "stopped by STOP ($n of 18 users)"
elif [ "$n" = 18 ]; then touch "$F/done"; log "done (18 of 18 users)"
else touch "$F/failed"; log "ended with $n of 18 users (killed.txt / failed.txt)"; fi
