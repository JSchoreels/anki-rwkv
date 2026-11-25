#!/usr/bin/env bash
# Tiny ensemble on the 1,500 users (SM169): the GPU chain (design.md "Run"). Adapted from the deploy-time ensemble test
# (SM160)'s launch-sm160-gpu.sh (same queue, cap and pass rules); the two six-layer Tiny models, T24 only, with --dump.
#   0 preconditions  files, clean source (scripts/ expertium/ rwkv/ and this folder's scripts), sha256 pins, report
#                    selftest, user lists, the 1,446 copied 732-part dumps
#   1 GPU smoke      ticket: the 2 smoke users x 2 models x T24 with the dump, into the run's own rows / dump folder;
#                    sm169_report.py smoke: rows within 1e-5 of SM164's rows, dumps reproduce the rows
#   2 the new part   users-1500-new.json (665 users, --pool 1500); passes at cap 0.8, then 0.92 after a pass without
#                    progress (memory only) -> gpu/sm169-rows-1500.jsonl, dumps work/sm169-dump/new (never committed)
# Log: gpu/launch-sm169-gpu.log; last line "SM169 GPU DONE ..." or "SM169 GPU STOP ..." (cause on an earlier "STOP:" line).
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm169/artifacts/small-model/sm169/launch-sm169-gpu.sh
#   launch   WMI hidden (HANDOFF.md), stdout to gpu/launch-sm169-gpu.nohup
#   resume   remove gpu/STOP, then the launch command again (finished users and gpu/smoke-ok are kept)
#   stop     touch C:/Users/jonat/workspace/wt/sm169/artifacts/small-model/sm169/gpu/STOP (before the next ticket)
# Do not edit scripts/ expertium/ rwkv/ of the sm169 worktree while it runs.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm169
SM=$WT/artifacts/small-model
S=$SM/sm169
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm169-gpu.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm169-gpu-dryrun.log
SC=$WT/scripts/small_model_heldout.py
RULE=$S/sm169_report.py
BLOCK=$SM/sm135/sm135_block.py
. "$S/models.sh"
T24=w8+lr2.16.4q8x8
EDGES=0,1,3,5,9,16,32,128,300,1000
UNEW=$S/users-1500-new.json
USMOKE=$S/users-smoke.json
ROWS=$G/sm169-rows-1500.jsonl
DNEW=$WT/work/sm169-dump/new; DE732=$WT/work/sm169-dump/e732   # absolute (the queue refuses relative paths)
TICKETS=$G/tickets.jsonl
STOPF=$G/STOP
NAME="Tiny ensemble on the 1,500 users (SM169)"
WANT=665
MIN_SMOKE=10; MIN_NEW=50; MIN_RETRY=15
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START SM11_INPUTS SM157_STATS
log() { echo "[sm169 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$STOPF" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }
free_mib() { awk '/MemFree/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 99999; }
block() { "$PY" -X utf8 "$BLOCK" | tr -d '\r'; }
jline() { "$PY" -X utf8 -c "import json, sys; print(json.dumps(dict(zip(sys.argv[1::2], [int(v) if v.lstrip('-').isdigit() else v for v in sys.argv[2::2]]))))" "$@" | tr -d '\r'; }

wait_free() {  # wait_free MINUTES WHAT: returns once MINUTES fit before the next quiet window (FREE = minutes to it)
  local n=0 b
  while true; do
    stopped && return 1
    b=$(block)
    case "$b" in
      NOW*) local s=${b#NOW }
            if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $2 would wait inside a quiet window (${s}s left)"; FREE=100000; return 0; fi
            [ $((n % 6)) = 0 ] && log "$2: quiet window, ${s}s left"; n=$((n + 1)); nap "$s";;
      NEXT*) FREE=${b#NEXT }
             [ "$FREE" -ge $(( $1 + 3 )) ] && return 0
             if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $2 ($1 min) would wait: $FREE min to the next quiet window"; FREE=100000; return 0; fi
             [ $((n % 6)) = 0 ] && log "$2 needs $1 min, $FREE min before the next quiet window; waiting"; n=$((n + 1)); nap $(( FREE * 60 + 60 ));;
      *) log "window check failed: $b"; nap 600;;
    esac
  done
}

ram_gate() {  # ram_gate WHAT: hold while < 3 GB RAM is free, at most 30 min, then go on
  local i=0
  while [ "$DRYRUN" != 1 ] && [ "$(free_mib)" -lt 3072 ]; do
    if [ $i -ge 6 ]; then log "$1: still $(free_mib) MiB RAM free after 30 min; going on"; return 0; fi
    [ $i = 0 ] && log "$1: $(free_mib) MiB RAM free (< 3 GB), holding"; i=$((i + 1)); sleep 300
  done
  return 0
}

q() {  # q MINUTES KIND PURPOSE -- CMD...   one ticket that must end before the next quiet window
  local minutes=$1 kind=$2 purpose=$3 rc t0 mx base wpid; shift 4
  while true; do
    wait_free "$minutes" "$purpose" || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $NAME: $purpose (expected $minutes min)"; echo "    cmd: $*" >> "$LOG"; return 0; fi
    local maxwait=$(( FREE - minutes - 3 )); [ "$maxwait" -lt 1 ] && maxwait=1
    log "enqueue: $purpose (expected $minutes min, max wait $maxwait min, $FREE min to the next quiet window)"
    t0=$(date +%s); base=$(gpu_used); rm -f "$G/.gpu-max"
    ( m=0; while true; do u=$(gpu_used); [ -n "$u" ] && [ "$u" -gt "$m" ] && { m=$u; echo "$m" > "$G/.gpu-max"; }; sleep 30; done ) &
    wpid=$!
    "$PY" "$QUEUE" run --owner sm169 --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm169/launch-sm169-gpu.sh" -- "$@"
    rc=$?; kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null; mx=$(cat "$G/.gpu-max" 2>/dev/null | tr -dc '0-9'); mx=${mx:-0}
    log "exit $rc: $purpose (nvidia-smi max $mx MiB, baseline $base MiB)"
    [ "$rc" = 75 ] && { log "queue wait would reach the next quiet window; re-planning"; sleep 60; continue; }
    jline kind "$kind" purpose "$purpose" exit "$rc" minutes "$(( ($(date +%s) - t0 + 30) / 60 ))" nvidia_smi_max_mib "$mx" nvidia_smi_baseline_mib "${base:-0}" >> "$TICKETS"
    return $rc
  done
}

min_users() {  # min_users ROWS LABEL...: users with a row (benchmark, or a 'nonfinite' row) for every label (the minimum)
  F="$1" "$PY" -X utf8 -c "
import json, os, sys
from pathlib import Path
p = Path(os.environ['F'])
got = {}
if p.exists():
    for raw in open(p, 'rb'):
        try:
            r = json.loads(raw.decode('utf-8', errors='replace'))
        except ValueError:
            continue
        if 'benchmark' in r or ('nonfinite' in r and 'error' not in r):
            got.setdefault(r.get('label'), set()).add(r['user_id'])
print(min(len(got.get(l, ())) for l in sys.argv[1:]))" "${@:2}" | tr -d '\r'; }

GARGS=(); GLBLS=()
while IFS='|' read -r tag ka path digest role size; do
  [ -z "$tag" ] && continue
  GARGS+=(--model "$tag=$ka:$path" --only "$tag/$T24"); GLBLS+=("$tag/$T24@e2s-query")
done <<< "$(tr -d '\r' <<< "$MODELS")"

SCORE_FLAGS=(--inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets "$EDGES")
ENVS=(env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS -u SM157_STATS PYTORCH_JIT=0)

smoke() {
  [ -e "$G/smoke-ok" ] && return 0
  q $MIN_SMOKE smoke "GPU smoke: 2 smoke users, 2 models x T24 with the per-review dump, vs SM164's rows" -- \
    "${ENVS[@]}" HELDOUT_GPU_MEMORY_FRACTION=0.8 "$PY" -X utf8 -W ignore "$SC" --pool 1500 --users-file "$USMOKE" \
    "${GARGS[@]}" "${SCORE_FLAGS[@]}" --dump "$DNEW" --time-limit 900 --out "$ROWS" >> "$G/score.out" 2>&1 \
    || { log "STOP: smoke ticket failed (see $G/score.out)"; return 1; }
  [ "$DRYRUN" = 1 ] && return 0
  local out; out=$("$PY" -X utf8 "$RULE" smoke 2>&1 | tr -d '\r')
  echo "$out" > "$G/smoke-check.txt"; log "$out"
  case "$out" in *"smoke OK"*) touch "$G/smoke-ok"; return 0;; esac
  log "STOP: GPU smoke mismatch (see $G/smoke-check.txt, $G/smoke-check.json)"; return 1
}

passes() {  # score GARGS / GLBLS on the new part until WANT users have every label
  local what="T24, the six-layer Tiny (2 models), ids >= 5000 of the 1,500 pool (users-1500-new.json)" cap=0.8 pass have prev=-1 minutes tl
  for pass in 1 2 3 4 5; do
    stopped && return 1
    have=$(min_users "$ROWS" "${GLBLS[@]}")
    [ "$have" -ge "$WANT" ] && break
    if [ "$have" = "$prev" ]; then
      if [ "$cap" = 0.8 ]; then cap=0.92; log "no progress ($have of $WANT users): next pass at cap 0.92 (memory only, same numerics)"
      else log "no progress at cap 0.92 ($have of $WANT users); going on"; break; fi
    fi
    prev=$have
    minutes=$MIN_NEW; [ "$pass" -gt 1 ] && minutes=$MIN_RETRY
    tl=$(( minutes * 60 * 6 / 5 ))
    ram_gate "$what"
    q $minutes score "$what with the per-review dump, e2s-query inputs (pass $pass, $have done, cap $cap)" -- \
      "${ENVS[@]}" HELDOUT_GPU_MEMORY_FRACTION=$cap "$PY" -X utf8 "$SC" --pool 1500 --users-file "$UNEW" \
      "${GARGS[@]}" "${SCORE_FLAGS[@]}" --dump "$DNEW" --time-limit $tl --out "$ROWS" >> "$G/score.out" 2>&1 \
      || { log "STOP: scoring ticket failed (pass $pass)"; return 1; }
    [ "$DRYRUN" = 1 ] && break
  done
  [ "$DRYRUN" = 1 ] && return 0
  have=$(min_users "$ROWS" "${GLBLS[@]}")
  log "new part: $have of $WANT users scored for both models"
  return 0
}

preconditions() {
  for f in "$BLOCK" "$RULE" "$S/models.sh" "$S/design.md" "$UNEW" "$USMOKE" "$SC" \
           "$SM/sm160/sm160_report.py" "$SM/sm160/gpu/sm160-rows-732.jsonl" "$SM/sm160/gpu/wfit-T24.json"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv artifacts/small-model/sm160 artifacts/small-model/sm169/*.sh artifacts/small-model/sm169/*.py artifacts/small-model/sm169/*.json artifacts/small-model/sm169/design.md)" ]; then
    log "STOP: uncommitted source changes"; return 1; fi
  local tag ka path digest role size got n
  while IFS='|' read -r tag ka path digest role size; do
    [ -z "$tag" ] && continue
    got=$(sha256sum "$path" | cut -d' ' -f1)
    [ "$got" = "$digest" ] || { log "STOP: sha256 of $tag is $got, pinned $digest"; return 1; }
  done <<< "$(tr -d '\r' <<< "$MODELS")"
  n=$(ls "$DE732" 2>/dev/null | grep -c "_w8+lr2.16.4q8x8@e2s-query.npz$")
  [ "$n" = 1446 ] || { log "STOP: $n copied 732-part dumps in $DE732 (need 1446)"; return 1; }
  "$PY" -X utf8 "$RULE" selftest >> "$LOG" 2>&1 || { log "STOP: report selftest failed"; return 1; }
  "$PY" -X utf8 "$RULE" users >> "$LOG" 2>&1 || { log "STOP: user lists differ from their rule"; return 1; }
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1 | cut -c1-90) (DRYRUN=$DRYRUN; dirty source files: $(git -C "$WT" status --porcelain -- scripts expertium rwkv | wc -l))"
rc=0
stopped && rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
[ $rc -eq 0 ] && { smoke || rc=1; }
[ $rc -eq 0 ] && { passes || rc=1; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "SM169 GPU DONE: rows in $ROWS, dumps in $DNEW (the read-out is the next agent's step)"; fi
else
  log "SM169 GPU STOP: ended with exit $rc"
fi
exit $rc
