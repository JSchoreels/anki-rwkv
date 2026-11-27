#!/usr/bin/env bash
# Today's models on donor data (SM178): the chain. Adapted from SM164's launch-sm164-gpu.sh (same queue, cap and pass
# rules, same scorer: wt/sm164 commit c30a0070, unchanged, wrapped by sm178_score.py).
#   1 labels (CPU, 2 processes, no ticket): FSRS-6 benchmark test reviews of the 218 donor users -> work/sm178/run/label-db
#   2 wait for gpu/smoke-ok (the smoke is run and checked by hand: smoke.sh + sm178_report.py smoke)
#   3 one GPU ticket per size (2 models, label REF = w8+s8hx8), order Tiny, Small, Big, all 218 users
#     (users-all.json); passes at cap 0.8, then 0.92 after a pass without progress (memory only) -> gpu/sm178-rows.jsonl
#   Scoring flags = SM164's: show-time inputs (e2s-query), leak-free (--deleted-cards neighbour), --nonfinite-rows,
#   --split-existing, --prior-buckets, PYTORCH_JIT=0.
# Log: gpu/launch-sm178.log; last line "SM178 DONE ..." or "SM178 STOP ..." (cause on an earlier "STOP:" line).
#   stop: touch gpu/STOP (before the next ticket). Resume: remove gpu/STOP, launch again (finished users are kept).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm178
S=$WT/artifacts/small-model/sm178
G=$S/gpu
RUN=$WT/work/sm178/run
mkdir -p "$G" "$RUN"
LOG=$G/launch-sm178.log
WRAP=$S/sm178_score.py
BLOCK=C:/Users/jonat/workspace/wt/sm164/artifacts/small-model/sm135/sm135_block.py
. "$S/models.sh"
UALL=$S/users-all.json
ROWS=$G/sm178-rows.jsonl
TICKETS=$G/tickets.jsonl
STOPF=$G/STOP
NAME="Today's models on donor data (SM178)"
EDGES=0,1,3,5,9,16,32,128,300,1000
# size|expected minutes (SM164's REF tickets scaled to 39.9M reviews, +40% for the 1M+ review users)
JOBS="
tiny|20
small|25
big|35
"
MIN_RETRY=30
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START SM11_INPUTS SM157_STATS
log() { echo "[sm178 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$STOPF" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }
block() { "$PY" -X utf8 "$BLOCK" | tr -d '\r'; }
jline() { "$PY" -X utf8 -c "import json, sys; print(json.dumps(dict(zip(sys.argv[1::2], [int(v) if v.lstrip('-').isdigit() else v for v in sys.argv[2::2]]))))" "$@" | tr -d '\r'; }

wait_free() {  # wait_free MINUTES WHAT: returns once MINUTES fit before the next quiet window (FREE = minutes to it)
  local n=0 b s
  while true; do
    stopped && return 1
    b=$(block)
    case "$b" in
      NOW*) s=${b#NOW }; [ $((n % 6)) = 0 ] && log "$2: quiet window, ${s}s left"; n=$((n + 1)); [ "$s" -gt 600 ] && s=600; sleep "$s";;
      NEXT*) FREE=${b#NEXT }
             [ "$FREE" -ge $(( $1 + 3 )) ] && return 0
             [ $((n % 6)) = 0 ] && log "$2 needs $1 min, $FREE min before the next quiet window; waiting"; n=$((n + 1)); sleep 600;;
      *) log "window check failed: $b"; sleep 600;;
    esac
  done
}

q() {  # q MINUTES KIND PURPOSE -- CMD...
  local minutes=$1 kind=$2 purpose=$3 rc t0 mx base wpid maxwait; shift 4
  while true; do
    wait_free "$minutes" "$purpose" || return 1
    maxwait=$(( FREE - minutes - 3 )); [ "$maxwait" -lt 1 ] && maxwait=1
    log "enqueue: $purpose (expected $minutes min, max wait $maxwait min)"
    t0=$(date +%s); base=$(gpu_used); rm -f "$G/.gpu-max"
    ( m=0; while true; do u=$(gpu_used); [ -n "$u" ] && [ "$u" -gt "$m" ] && { m=$u; echo "$m" > "$G/.gpu-max"; }; sleep 30; done ) &
    wpid=$!
    "$PY" "$QUEUE" run --owner sm178 --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm178/launch-sm178.sh" -- "$@"
    rc=$?; kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null; mx=$(cat "$G/.gpu-max" 2>/dev/null | tr -dc '0-9'); mx=${mx:-0}
    log "exit $rc: $purpose (nvidia-smi max $mx MiB, baseline $base MiB)"
    [ "$rc" = 75 ] && { log "queue wait would reach the next quiet window; re-planning"; sleep 60; continue; }
    jline kind "$kind" purpose "$purpose" exit "$rc" minutes "$(( ($(date +%s) - t0 + 30) / 60 ))" nvidia_smi_max_mib "$mx" nvidia_smi_baseline_mib "${base:-0}" >> "$TICKETS"
    return $rc
  done
}

min_users() {  # min_users LABEL...: users of users-all.json with a row (benchmark or 'nonfinite') for every label
  F="$ROWS" U="$UALL" "$PY" -X utf8 -c "
import json, os, sys
from pathlib import Path
p = Path(os.environ['F'])
want = set(json.loads(Path(os.environ['U']).read_text())['users'])
got = {}
if p.exists():
    for raw in open(p, 'rb'):
        try:
            r = json.loads(raw.decode('utf-8', errors='replace'))
        except ValueError:
            continue
        if r.get('user_id') in want and ('benchmark' in r or ('nonfinite' in r and 'error' not in r)):
            got.setdefault(r.get('label'), set()).add(r['user_id'])
print(min(len(got.get(l, ())) for l in sys.argv[1:]))" "$@" | tr -d '\r'; }

group_args() {  # group_args SIZE: GARGS (--model / --only) and GLBLS (labels) of one size
  GARGS=(); GLBLS=()
  local tag ka path digest role size
  while IFS='|' read -r tag ka path digest role size; do
    [ -z "$tag" ] && continue
    [ "$1" = "$size" ] || continue
    GARGS+=(--model "$tag=$ka:$path" --only "$tag/$REF"); GLBLS+=("$tag/$REF@e2s-query")
  done <<< "$(printf '%s\n' "$MODELS" | tr -d '\r')"
}

SCORE_FLAGS=(--inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets "$EDGES")
ENVS=(env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS -u SM157_STATS PYTORCH_JIT=0)

passes() {  # passes SIZE MINUTES
  local size=$1 min1=$2 cap=0.8 pass have prev=-1 minutes tl want=218
  for pass in 1 2 3 4 5; do
    have=$(min_users "${GLBLS[@]}")
    [ "$have" -ge "$want" ] && break
    if [ "$have" = "$prev" ]; then
      if [ "$cap" = 0.8 ]; then cap=0.92; log "no progress on $size ($have of $want users): next pass at cap 0.92 (memory only, same numerics)"
      else log "no progress at cap 0.92 on $size ($have of $want users); going on"; break; fi
    fi
    prev=$have
    minutes=$min1; [ "$pass" -gt 1 ] && minutes=$MIN_RETRY
    tl=$(( minutes * 60 * 6 / 5 ))
    q $minutes score "REF, $size (2 models), 218 donor users, e2s-query inputs (pass $pass, $have done, cap $cap)" -- \
      "${ENVS[@]}" HELDOUT_GPU_MEMORY_FRACTION=$cap "$PY" -X utf8 -W ignore "$WRAP" score "$RUN" -- --users-file "$UALL" \
      "${GARGS[@]}" "${SCORE_FLAGS[@]}" --time-limit $tl --out "$ROWS" >> "$G/score.out" 2>&1 \
      || { log "STOP: scoring ticket failed ($size)"; return 1; }
  done
  have=$(min_users "${GLBLS[@]}")
  log "$size: $have of $want users scored for every label (${#GLBLS[@]} labels)"
  return 0
}

labels() {
  [ -e "$RUN/labels-done" ] && { log "labels kept: $RUN/labels-done"; return 0; }
  log "labels: start (CPU, 2 processes)"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" -X utf8 -W ignore "$WRAP" labels "$RUN" 2 "$UALL" >> "$G/labels.out" 2>&1 \
    || { log "STOP: labels failed (see $G/labels.out)"; return 1; }
  grep -q "DONE" "$G/labels.out" || { log "STOP: labels did not finish"; return 1; }
  touch "$RUN/labels-done"; log "labels: done"
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1 | cut -c1-90); scorer $(git -C C:/Users/jonat/workspace/wt/sm164 log --oneline -1 | cut -c1-60), dirty scorer files: $(git -C C:/Users/jonat/workspace/wt/sm164 status --porcelain -- scripts expertium rwkv | wc -l)"
rc=0
stopped && rc=1
if [ $rc -eq 0 ]; then
  while IFS='|' read -r tag ka path digest role size; do
    [ -z "$tag" ] && continue
    [ "$(sha256sum "$path" | cut -d' ' -f1)" = "$digest" ] || { log "STOP: sha256 of $tag differs"; rc=1; }
  done <<< "$(printf '%s\n' "$MODELS" | tr -d '\r')"
fi
[ $rc -eq 0 ] && { labels || rc=1; }
if [ $rc -eq 0 ]; then
  i=0
  while [ ! -e "$G/smoke-ok" ]; do
    stopped && { rc=1; break; }
    [ $i -ge 24 ] && { log "STOP: no gpu/smoke-ok after 2 h"; rc=1; break; }
    [ $i = 0 ] && log "waiting for gpu/smoke-ok"; i=$((i + 1)); sleep 300
  done
fi
if [ $rc -eq 0 ]; then
  mapfile -t JOBL < <(printf '%s\n' "$JOBS" | tr -d '\r' | grep -v '^$')
  for job in "${JOBL[@]}"; do
    IFS='|' read -r size minutes <<< "$job"
    stopped && { rc=1; break; }
    group_args "$size"
    passes "$size" "$minutes" || { rc=1; break; }
  done
fi
if [ $rc -eq 0 ]; then log "SM178 DONE: rows in $ROWS (read-out: sm178_report.py report)"; else log "SM178 STOP: ended with exit $rc"; fi
exit $rc
