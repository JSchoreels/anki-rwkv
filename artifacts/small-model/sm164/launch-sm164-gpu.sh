#!/usr/bin/env bash
# Release-candidate re-check of the final state formats (SM164): the GPU chain (design.md "Run"). Adapted from the deploy-time
# ensemble (SM160)'s launch-sm160-gpu.sh (same queue, cap and pass rules); plain labels, no dump.
#   0 preconditions  files, clean source (scripts/ expertium/ rwkv/ and this folder's files), model sha256, SM160 rows sha256,
#                    report selftest, users lists unchanged
#   1 GPU smoke      ticket: 5 smoke users x 6 models x REF / T24 / F6 -> gpu/smoke-gpu-rows.jsonl; sm164_report.py smoke:
#                    rows within 1e-5 of SM152's GPU smoke rows (3 users) and of SM160's REF / T24 rows (2 users id >= 5000)
#   2 the ids >= 5000 part of the 1,500 pool, only what no earlier run scored (one ticket per job, order Tiny, Small, Big):
#       tiny  T24 + REF  users-new-rest.json (665)     small F6  users-new-all.json (765)   small REF users-new-rest.json (665)
#       big   F6         users-big-all.json  (754)     big   REF users-big-rest.json (654)
#     passes at cap 0.8, then 0.92 after a pass without progress (memory only) -> gpu/sm164-rows-new.jsonl
#   All scoring: --pool 1500, show-time inputs (e2s-query), leak-free (--deleted-cards neighbour), --nonfinite-rows,
#   --split-existing, --prior-buckets, PYTORCH_JIT=0 (SM148 / SM152's flags).
# Log: gpu/launch-sm164-gpu.log; last line "SM164 GPU DONE ..." or "SM164 GPU STOP ..." (cause on an earlier "STOP:" line).
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm164/artifacts/small-model/sm164/launch-sm164-gpu.sh
#   launch   WMI hidden (HANDOFF.md), stdout to gpu/launch-sm164-gpu.nohup
#   resume   remove gpu/STOP, then the launch command again (finished users and gpu/smoke-ok are kept)
#   stop     touch C:/Users/jonat/workspace/wt/sm164/artifacts/small-model/sm164/gpu/STOP (before the next ticket)
# Do not edit scripts/ expertium/ rwkv/ of the sm164 worktree while it runs.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm164
SM=$WT/artifacts/small-model
S=$SM/sm164
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm164-gpu.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm164-gpu-dryrun.log
SC=$WT/scripts/small_model_heldout.py
RULE=$S/sm164_report.py
BLOCK=$SM/sm135/sm135_block.py
. "$S/models.sh"; . "$S/labels.sh"
USMOKE=$S/users-smoke.json
SMOKEROWS=$G/smoke-gpu-rows.jsonl
ROWS=$G/sm164-rows-new.jsonl
SRC160=C:/Users/jonat/workspace/wt/sm160/artifacts/small-model/sm160/gpu/sm160-rows-wfit.jsonl
SRC160_SHA=c03f1ac776dd4b07187474d0e1c731f4c67567419234e80de4ca11ddf37931e5
TICKETS=$G/tickets.jsonl
STOPF=$G/STOP
NAME="Release-candidate re-check of the final state formats (SM164)"
# JOBS: size|label names|users file|expected minutes (SM160's per-label 732-set minutes x the list's reviews / 42.8M, +25%)
JOBS="
tiny|T24 REF|users-new-rest.json|55
small|F6|users-new-all.json|60
small|REF|users-new-rest.json|30
big|F6|users-big-all.json|80
big|REF|users-big-rest.json|35
"
MIN_SMOKE=15; MIN_RETRY=40
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START SM11_INPUTS SM157_STATS
log() { echo "[sm164 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
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
    "$PY" "$QUEUE" run --owner sm164 --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm164/launch-sm164-gpu.sh" -- "$@"
    rc=$?; kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null; mx=$(cat "$G/.gpu-max" 2>/dev/null | tr -dc '0-9'); mx=${mx:-0}
    log "exit $rc: $purpose (nvidia-smi max $mx MiB, baseline $base MiB)"
    [ "$rc" = 75 ] && { log "queue wait would reach the next quiet window; re-planning"; sleep 60; continue; }
    jline kind "$kind" purpose "$purpose" exit "$rc" minutes "$(( ($(date +%s) - t0 + 30) / 60 ))" nvidia_smi_max_mib "$mx" nvidia_smi_baseline_mib "${base:-0}" >> "$TICKETS"
    return $rc
  done
}

min_users() {  # min_users ROWS USERSFILE LABEL...: users of USERSFILE with a row (benchmark or 'nonfinite') for every label (the minimum)
  F="$1" U="$2" "$PY" -X utf8 -c "
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
print(min(len(got.get(l, ())) for l in sys.argv[1:]))" "${@:3}" | tr -d '\r'; }

group_args() {  # group_args SIZE "NAMES": GARGS (--model / --only) and GLBLS (labels) of one size (all = every model)
  GARGS=(); GLBLS=()
  local tag ka path digest role size n
  while IFS='|' read -r tag ka path digest role size; do
    [ -z "$tag" ] && continue
    [ "$1" = all ] || [ "$1" = "$size" ] || continue
    GARGS+=(--model "$tag=$ka:$path")
    for n in $2; do GARGS+=(--only "$tag/${!n}"); GLBLS+=("$tag/${!n}@e2s-query"); done
  done <<< "$MODELS"
}

SCORE_FLAGS=(--pool 1500 --inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets "$EDGES")
ENVS=(env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS -u SM157_STATS PYTORCH_JIT=0)

smoke() {
  [ -e "$G/smoke-ok" ] && { log "smoke kept: $G/smoke-ok"; return 0; }
  group_args all "REF T24 F6"
  q $MIN_SMOKE smoke "GPU smoke: 5 smoke users, 6 models x REF / T24 / F6, vs SM152's GPU smoke rows and SM160's rows" -- \
    "${ENVS[@]}" HELDOUT_GPU_MEMORY_FRACTION=0.8 "$PY" -X utf8 -W ignore "$SC" --users-file "$USMOKE" \
    "${GARGS[@]}" "${SCORE_FLAGS[@]}" --time-limit 900 --out "$SMOKEROWS" >> "$G/score.out" 2>&1 \
    || { log "STOP: smoke ticket failed (see $G/score.out)"; return 1; }
  [ "$DRYRUN" = 1 ] && return 0
  local out; out=$("$PY" -X utf8 "$RULE" smoke 2>&1 | tr -d '\r')
  echo "$out" > "$G/smoke-check.txt"; log "$out"
  case "$out" in *"smoke OK"*) touch "$G/smoke-ok"; return 0;; esac
  log "STOP: GPU smoke mismatch (see $G/smoke-check.txt, $G/smoke-check.json)"; return 1
}

passes() {  # passes WHAT USERSFILE MINUTES: score GARGS / GLBLS until every user of USERSFILE has every label
  local what=$1 uf=$2 min1=$3 cap=0.8 pass have prev=-1 minutes tl want
  want=$("$PY" -X utf8 -c "import json, sys; print(len(json.load(open(sys.argv[1]))['users']))" "$uf" | tr -d '\r')
  for pass in 1 2 3 4 5; do
    have=$(min_users "$ROWS" "$uf" "${GLBLS[@]}")
    [ "$have" -ge "$want" ] && break
    if [ "$have" = "$prev" ]; then
      if [ "$cap" = 0.8 ]; then cap=0.92; log "no progress on $what ($have of $want users): next pass at cap 0.92 (memory only, same numerics)"
      else log "no progress at cap 0.92 on $what ($have of $want users); going on"; break; fi
    fi
    prev=$have
    minutes=$min1; [ "$pass" -gt 1 ] && minutes=$MIN_RETRY
    tl=$(( minutes * 60 * 6 / 5 ))
    ram_gate "$what"
    q $minutes score "$what, e2s-query inputs (pass $pass, $have done, cap $cap)" -- \
      "${ENVS[@]}" HELDOUT_GPU_MEMORY_FRACTION=$cap "$PY" -X utf8 "$SC" --users-file "$uf" \
      "${GARGS[@]}" "${SCORE_FLAGS[@]}" --time-limit $tl --out "$ROWS" >> "$G/score.out" 2>&1 \
      || { log "STOP: scoring ticket failed ($what)"; return 1; }
    [ "$DRYRUN" = 1 ] && break
  done
  [ "$DRYRUN" = 1 ] && return 0
  have=$(min_users "$ROWS" "$uf" "${GLBLS[@]}")
  log "$what: $have of $want users scored for every label (${#GLBLS[@]} labels)"
  return 0
}

preconditions() {
  for f in "$BLOCK" "$RULE" "$S/models.sh" "$S/labels.sh" "$S/design.md" "$USMOKE" "$SC" "$SRC160" \
           "$S/users-new-all.json" "$S/users-new-rest.json" "$S/users-big-all.json" "$S/users-big-rest.json" \
           "$SM/sm152/gpu/smoke-gpu-rows.jsonl" "$SM/sm152/gpu/sm152-rows-732.jsonl" "$SM/sm148/gpu/sm148-rows-732.jsonl"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv artifacts/small-model/sm164/*.sh artifacts/small-model/sm164/*.py artifacts/small-model/sm164/*.json artifacts/small-model/sm164/design.md)" ]; then
    log "STOP: uncommitted source changes"; return 1; fi
  local tag ka path digest role size got
  while IFS='|' read -r tag ka path digest role size; do
    [ -z "$tag" ] && continue
    got=$(sha256sum "$path" | cut -d' ' -f1)
    [ "$got" = "$digest" ] || { log "STOP: sha256 of $tag is $got, pinned $digest"; return 1; }
  done <<< "$MODELS"
  got=$(sha256sum "$SRC160" | cut -d' ' -f1)
  [ "$got" = "$SRC160_SHA" ] || { log "STOP: SM160 rows changed (sha256 $got, pinned $SRC160_SHA)"; return 1; }
  "$PY" -X utf8 "$RULE" selftest >> "$LOG" 2>&1 || { log "STOP: report selftest failed"; return 1; }
  "$PY" -X utf8 "$RULE" users >> "$LOG" 2>&1 || { log "STOP: users lists differ from their rule"; return 1; }
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1 | cut -c1-90) (DRYRUN=$DRYRUN; dirty source files: $(git -C "$WT" status --porcelain -- scripts expertium rwkv | wc -l))"
rc=0
stopped && rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
[ $rc -eq 0 ] && { smoke || rc=1; }
if [ $rc -eq 0 ]; then
  mapfile -t JOBL < <(printf '%s\n' "$JOBS" | tr -d '\r' | grep -v '^$')   # an array: no ticket can read the job list from stdin
  for job in "${JOBL[@]}"; do
    IFS='|' read -r size names uf minutes <<< "$job"
    stopped && { rc=1; break; }
    group_args "$size" "$names"
    passes "$names, $size (2 models), ids >= 5000 of the 1,500 pool ($uf)" "$S/$uf" "$minutes" || { rc=1; break; }
  done
fi
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "SM164 GPU DONE: rows in $ROWS (the read-out is the next agent's step: sm164_report.py report)"; fi
else
  log "SM164 GPU STOP: ended with exit $rc"
fi
exit $rc
