#!/usr/bin/env bash
# Deploy-time ensemble of the two instant predictions (SM160): the GPU chain (design.md "Run"). Adapted from output-aware
# T24 on the 732 set (SM158)'s launch-sm158-gpu.sh (same queue, cap and pass rules); plain labels only, with --dump.
#   0 preconditions  files, clean source (scripts/ expertium/ rwkv/ and this folder's scripts), report selftest, users
#   1 GPU smoke      ticket: 3 smoke users x 6 models x T24 and REF, with the dump -> gpu/smoke-gpu-rows.jsonl;
#                    sm160_report.py smoke: rows within 1e-5 of SM148's GPU smoke rows, dumps reproduce the rows
#   for each label, T24 (the decision) then REF (information):
#   2 training users one ticket, the 100 users of users-wfit.json x 6 models (--pool 1500) -> gpu/sm160-rows-wfit.jsonl
#   3 w fit          sm160_report.py fit --label L -> gpu/wfit-L.json (CPU, training users only), BEFORE any 732 row of L
#   4 the 732 set    one ticket per size (Tiny, Small, Big), 2 models; passes at cap 0.8, then 0.92 after a pass without
#                    progress (memory only) -> gpu/sm160-rows-732.jsonl
#   All scoring: show-time inputs (e2s-query), leak-free (--deleted-cards neighbour), --nonfinite-rows, --split-existing,
#   --prior-buckets, PYTORCH_JIT=0 (SM148's flags), --dump work/sm160-dump/{smoke,wfit,e732} (never committed).
# Log: gpu/launch-sm160-gpu.log; "SM160 T24 PHASE DONE" after the decision label; last line "SM160 GPU DONE ..." or
# "SM160 GPU STOP ...".
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm160/artifacts/small-model/sm160/launch-sm160-gpu.sh
#   launch   WMI hidden (HANDOFF.md), stdout to gpu/launch-sm160-gpu.nohup
#   resume   remove gpu/STOP, then the launch command again (finished users, gpu/smoke-ok and gpu/wfit-*.json are kept)
#   stop     touch C:/Users/jonat/workspace/wt/sm160/artifacts/small-model/sm160/gpu/STOP (before the next ticket)
#   SKIP_REF=1 skips the REF label (information only).
# Do not edit scripts/ expertium/ rwkv/ of the sm160 worktree while it runs.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm160
SM=$WT/artifacts/small-model
S=$SM/sm160
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm160-gpu.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm160-gpu-dryrun.log
SC=$WT/scripts/small_model_heldout.py
RULE=$S/sm160_report.py
BLOCK=$SM/sm135/sm135_block.py
. "$S/models.sh"; . "$S/labels.sh"
USERS732=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json
UWFIT=$S/users-wfit.json
USMOKE=$S/users-smoke.json
SMOKEROWS=$G/smoke-gpu-rows.jsonl
WFITROWS=$G/sm160-rows-wfit.jsonl
ROWS732=$G/sm160-rows-732.jsonl
DSMOKE=work/sm160-dump/smoke; DWFIT=work/sm160-dump/wfit; D732=work/sm160-dump/e732   # relative to the worktree root
TICKETS=$G/tickets.jsonl
STOPF=$G/STOP
NAME="Deploy-time ensemble of the two instant predictions (SM160)"
declare -A MIN=([tiny]=40 [small]=55 [big]=95)      # per label and size: SM148's 732 wall minutes per label x 1.3 (dump)
declare -A WANT=([tiny]=723 [small]=723 [big]=718)  # Big: the 5 largest users never fit the scorer (SM148)
WANT_WFIT=100
MIN_SMOKE=20; MIN_WFIT=45; MIN_RETRY=40
PHASES="T24 REF"; [ "${SKIP_REF:-0}" = 1 ] && PHASES="T24"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START SM11_INPUTS SM157_STATS
log() { echo "[sm160 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
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
    "$PY" "$QUEUE" run --owner sm160 --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm160/launch-sm160-gpu.sh" -- "$@"
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

SCORE_FLAGS=(--inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets "$EDGES")
ENVS=(env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS -u SM157_STATS PYTORCH_JIT=0)

smoke() {
  [ -e "$G/smoke-ok" ] && return 0
  group_args all "T24 REF"
  q $MIN_SMOKE smoke "GPU smoke: 3 smoke users, 6 models x T24 / REF with the per-review dump, vs SM148's GPU smoke rows" -- \
    "${ENVS[@]}" HELDOUT_GPU_MEMORY_FRACTION=0.8 "$PY" -X utf8 -W ignore "$SC" --pool 732 --users-file "$USMOKE" \
    "${GARGS[@]}" "${SCORE_FLAGS[@]}" --dump "$DSMOKE" --time-limit 1800 --out "$SMOKEROWS" >> "$G/score.out" 2>&1 \
    || { log "STOP: smoke ticket failed (see $G/score.out)"; return 1; }
  [ "$DRYRUN" = 1 ] && return 0
  local out; out=$("$PY" -X utf8 "$RULE" smoke 2>&1 | tr -d '\r')
  echo "$out" > "$G/smoke-check.txt"; log "$out"
  case "$out" in *"smoke OK"*) touch "$G/smoke-ok"; return 0;; esac
  log "STOP: GPU smoke mismatch (see $G/smoke-check.txt, $G/smoke-check.json)"; return 1
}

passes() {  # passes WHAT WANT POOL USERSFILE ROWS DUMP MINUTES: score GARGS / GLBLS until WANT users have every label
  local what=$1 want=$2 pool=$3 uf=$4 rows=$5 dump=$6 min1=$7 cap=0.8 pass have prev=-1 minutes tl
  for pass in 1 2 3 4 5; do
    have=$(min_users "$rows" "${GLBLS[@]}")
    [ "$have" -ge "$want" ] && break
    if [ "$have" = "$prev" ]; then
      if [ "$cap" = 0.8 ]; then cap=0.92; log "no progress on $what ($have of $want users): next pass at cap 0.92 (memory only, same numerics)"
      else log "no progress at cap 0.92 on $what ($have of $want users); going on"; break; fi
    fi
    prev=$have
    minutes=$min1; [ "$pass" -gt 1 ] && minutes=$MIN_RETRY
    tl=$(( minutes * 60 * 6 / 5 ))
    ram_gate "$what"
    q $minutes score "$what with the per-review dump, e2s-query inputs (pass $pass, $have done, cap $cap)" -- \
      "${ENVS[@]}" HELDOUT_GPU_MEMORY_FRACTION=$cap "$PY" -X utf8 "$SC" --pool "$pool" --users-file "$uf" \
      "${GARGS[@]}" "${SCORE_FLAGS[@]}" --dump "$dump" --time-limit $tl --out "$rows" >> "$G/score.out" 2>&1 \
      || { log "STOP: scoring ticket failed ($what)"; return 1; }
    [ "$DRYRUN" = 1 ] && break
  done
  [ "$DRYRUN" = 1 ] && return 0
  have=$(min_users "$rows" "${GLBLS[@]}")
  log "$what: $have of $want users scored for every label (${#GLBLS[@]} labels)"
  return 0
}

phase() {  # phase LABELNAME: training users, the w fit, then the 732 set per size
  local L=$1 size out
  stopped && return 1
  group_args all "$L"
  passes "$L on the 100 training users (6 models)" $WANT_WFIT 1500 "$UWFIT" "$WFITROWS" "$DWFIT" $MIN_WFIT || return 1
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: w fit (sm160_report.py fit --label $L) would write gpu/wfit-$L.json"
  elif [ ! -s "$G/wfit-$L.json" ]; then
    out=$("$PY" -X utf8 "$RULE" fit --label "$L" 2>&1 | tr -d '\r') || { log "STOP: w fit failed: $out"; return 1; }
    log "$out"
  else log "w fit $L kept: $G/wfit-$L.json"; fi
  for size in tiny small big; do
    stopped && return 1
    group_args "$size" "$L"
    passes "$L, $size (2 models), the 732 set" ${WANT[$size]} 732 "$USERS732" "$ROWS732" "$D732" ${MIN[$size]} || return 1
  done
  return 0
}

preconditions() {
  for f in "$BLOCK" "$RULE" "$S/models.sh" "$S/labels.sh" "$S/design.md" "$USERS732" "$UWFIT" "$USMOKE" "$SC" \
           "$SM/sm148/gpu/smoke-gpu-rows.jsonl"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv artifacts/small-model/sm160/*.sh artifacts/small-model/sm160/*.py artifacts/small-model/sm160/*.json artifacts/small-model/sm160/design.md)" ]; then
    log "STOP: uncommitted source changes"; return 1; fi
  local tag ka path digest role size got
  while IFS='|' read -r tag ka path digest role size; do
    [ -z "$tag" ] && continue
    got=$(sha256sum "$path" | cut -d' ' -f1)
    [ "$got" = "$digest" ] || { log "STOP: sha256 of $tag is $got, pinned $digest"; return 1; }
  done <<< "$MODELS"
  "$PY" -X utf8 "$RULE" selftest >> "$LOG" 2>&1 || { log "STOP: report selftest failed"; return 1; }
  "$PY" -X utf8 "$RULE" users >> "$LOG" 2>&1 || { log "STOP: users-wfit.json differs from its rule"; return 1; }
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1 | cut -c1-90) (DRYRUN=$DRYRUN; phases $PHASES; dirty source files: $(git -C "$WT" status --porcelain -- scripts expertium rwkv | wc -l))"
rc=0
stopped && rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
[ $rc -eq 0 ] && { smoke || rc=1; }
for L in $PHASES; do
  [ $rc -eq 0 ] || break
  phase "$L" || rc=1
  [ $rc -eq 0 ] && [ "$L" = T24 ] && [ "$DRYRUN" != 1 ] && log "SM160 T24 PHASE DONE: the decision label is scored (rows $ROWS732, fit $G/wfit-T24.json)"
done
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "SM160 GPU DONE: rows in $ROWS732 and $WFITROWS, dumps in $WT/$D732 and $WT/$DWFIT (labels $PHASES; the read-out is the next agent's step)"; fi
else
  log "SM160 GPU STOP: ended with exit $rc"
fi
exit $rc
