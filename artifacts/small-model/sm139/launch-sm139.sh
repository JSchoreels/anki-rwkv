#!/usr/bin/env bash
# Release-candidate checks for the family (SM139) (artifacts/small-model/sm139/design.md, pre-registered 2026-11-21 before
#   any GPU ticket). No training. Every GPU step is one scripts/r01_queue.py ticket (owner sm139, claim gpu):
#   1 smoke   the 2 smoke users (users-smoke.json), all 16 labels, GPU -> gpu/smoke-gpu-rows.jsonl, then
#             sm139_report.py smoke vs the committed CPU rows (cpu/smoke-cpu-rows.jsonl): |GPU - CPU| <= 1e-5 or STOP
#   2 scoring one ticket per (size, pool), passes at cap 0.8, then 0.92 after a pass without progress (memory only, same
#             numerics); order Tiny, Small, Big on the 732 set (723 users), then the same on the extension (771 users of
#             users-sm93-score.json). Labels: release candidates at w8+s8 and w8+s8hx8, the published 11-13 models of
#             Small and Tiny at w8+s8 (Big's published models are the candidates themselves).
#   3 report  sm139_report.py report (CPU) -> sm139-result.{json,md}
#   All scoring: show-time inputs (e2s-query), leak-free (--deleted-cards neighbour), --nonfinite-rows, --split-existing.
# Breaks: quiet.json windows of scope all / gpu only (sm135_block.py); no planned daily or Sunday break.
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm139/artifacts/small-model/sm139/launch-sm139.sh
#   launch   WMI hidden (HANDOFF.md); log artifacts/small-model/sm139/gpu/launch-sm139.log
#   resume   remove gpu/STOP, then the launch command again (finished users and steps are skipped)
#   stop     touch C:/Users/jonat/workspace/wt/sm139/artifacts/small-model/sm139/gpu/STOP
# Do not edit scripts/ expertium/ rwkv/ of the sm139 worktree while it runs.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm139
SM=$WT/artifacts/small-model
S=$SM/sm139
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm139.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm139-dryrun.log
SC=$WT/scripts/small_model_heldout.py
RULE=$S/sm139_report.py
BLOCK=$SM/sm135/sm135_block.py
. "$S/models.sh"
W=C:/Users/jonat/workspace/wt
USERS732=$W/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json
USERSEXT=$W/sm57/artifacts/small-model/sm57/stage4/users-sm93-score.json
USMOKE=$S/users-smoke.json
CPUSMOKE=$S/cpu/smoke-cpu-rows.jsonl
GPUSMOKE=$G/smoke-gpu-rows.jsonl
ROWS732=$G/sm139-rows-732.jsonl
ROWSEXT=$G/sm139-rows-ext.jsonl
TICKETS=$G/tickets.jsonl
STOPF=$G/STOP
NAME="Release-candidate checks for the family (SM139)"
# expected minutes of pass 1 per (size, pool) (design.md "GPU time"); later passes MIN_RETRY
declare -A MIN=([tiny-732]=30 [small-732]=50 [big-732]=80 [tiny-ext]=40 [small-ext]=65 [big-ext]=105)
MIN_SMOKE=12; MIN_RETRY=20
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START SM11_INPUTS
log() { echo "[sm139 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$STOPF" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }
free_mib() { awk '/MemFree/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 99999; }
block() { "$PY" -X utf8 "$BLOCK" | tr -d '\r'; }
jline() { "$PY" -X utf8 -c "import json, sys; print(json.dumps(dict(zip(sys.argv[1::2], [int(v) if v.lstrip('-').isdigit() else v for v in sys.argv[2::2]]))))" "$@" | tr -d '\r'; }

wait_free() {  # wait_free MINUTES WHAT: returns once MINUTES fit before the next quiet window (sets FREE = minutes to it)
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
    "$PY" "$QUEUE" run --owner sm139 --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm139/launch-sm139.sh" -- "$@"
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

group_args() {  # group_args SIZE: sets GARGS (--model / --only) and GLBLS (labels) of one size; SIZE=all for the smoke
  GARGS=(); GLBLS=()
  local tag ka path digest role size
  while IFS='|' read -r tag ka path digest role size; do
    [ -z "$tag" ] && continue
    [ "$1" = all ] || [ "$1" = "$size" ] || continue
    GARGS+=(--model "$tag=$ka:$path" --only "$tag/w8+s8"); GLBLS+=("$tag/w8+s8@e2s-query")
    [ "$role" = rc ] && { GARGS+=(--only "$tag/w8+s8hx8"); GLBLS+=("$tag/w8+s8hx8@e2s-query"); }
  done <<< "$MODELS"
}

smoke() {
  [ -e "$G/smoke-ok" ] && return 0
  group_args all
  q $MIN_SMOKE smoke "GPU smoke: 2 users, the 16 labels (6 release candidates at w8+s8 and w8+s8hx8, 4 published models at w8+s8), vs the CPU rows" -- \
    env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS HELDOUT_GPU_MEMORY_FRACTION=0.8 "$PY" -X utf8 "$SC" --pool 732 \
    --users-file "$USMOKE" "${GARGS[@]}" --inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --time-limit 600 \
    --out "$GPUSMOKE" >> "$G/score.out" 2>&1 || { log "STOP: smoke ticket failed (see $G/score.out)"; return 1; }
  [ "$DRYRUN" = 1 ] && return 0
  local out; out=$("$PY" -X utf8 "$RULE" smoke --gpu "$GPUSMOKE" --cpu "$CPUSMOKE" 2>&1 | tr -d '\r')
  echo "$out" > "$G/smoke-check.txt"; log "$out"
  case "$out" in *"smoke OK"*) touch "$G/smoke-ok"; return 0;; esac
  log "STOP: GPU smoke mismatch (see $G/smoke-check.txt)"; return 1
}

score() {  # score SIZE POOL: one size on one pool, passes with the cap rule
  local size=$1 pool=$2 cap=0.8 pass have prev=-1 minutes uf rows want tl what sp
  if [ "$pool" = 732 ]; then sp=732; uf=$USERS732; rows=$ROWS732; want=723; what="the 732 set minus 9 never-fit (723 users)"
  else sp=1500; uf=$USERSEXT; rows=$ROWSEXT; want=771; what="the 1,500 validation users outside the 732 set + 6 check users (771 users)"; fi
  group_args "$size"
  for pass in 1 2 3 4 5; do
    have=$(min_users "$rows" "${GLBLS[@]}")
    [ "$have" -ge "$want" ] && break
    if [ "$have" = "$prev" ]; then
      if [ "$cap" = 0.8 ]; then cap=0.92; log "no progress on $size/$pool ($have of $want users): next pass at cap 0.92 (memory only, same numerics)"
      else log "no progress at cap 0.92 on $size/$pool ($have of $want users); going on"; break; fi
    fi
    prev=$have
    minutes=${MIN[$size-$pool]}; [ "$pass" -gt 1 ] && minutes=$MIN_RETRY
    tl=$(( minutes * 60 * 6 / 5 ))
    ram_gate "scoring $size/$pool"
    q $minutes score "scoring $size ($(echo "${GLBLS[@]}" | sed 's/@e2s-query//g')) on e2s-query inputs, $what (pass $pass, $have done, cap $cap)" -- \
      env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS HELDOUT_GPU_MEMORY_FRACTION=$cap "$PY" -X utf8 "$SC" --pool $sp \
      --users-file "$uf" "${GARGS[@]}" --inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --time-limit $tl \
      --out "$rows" >> "$G/score.out" 2>&1 || { log "STOP: scoring ticket failed ($size/$pool)"; return 1; }
    [ "$DRYRUN" = 1 ] && break
  done
  [ "$DRYRUN" = 1 ] && return 0
  have=$(min_users "$rows" "${GLBLS[@]}")
  log "$size/$pool: $have of $want users scored for every label (${#GLBLS[@]} labels)"
  return 0
}

preconditions() {
  for f in "$BLOCK" "$RULE" "$S/models.sh" "$S/design.md" "$S/design-pins.json" "$USERS732" "$USERSEXT" "$USMOKE" "$CPUSMOKE" "$SC"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/"; return 1; fi
  "$PY" -X utf8 "$BLOCK" selftest >> "$LOG" 2>&1 || { log "STOP: block selftest failed"; return 1; }
  "$PY" -X utf8 "$RULE" check >> "$LOG" 2>&1 || { log "STOP: pins changed (models, users files or showtime_report.py)"; return 1; }
  "$PY" -X utf8 "$RULE" selftest >> "$LOG" 2>&1 || { log "STOP: report selftest failed"; return 1; }
  return 0
}

final_check() {  # every expected output exists (non-empty, except the smoke-ok marker) before done
  local f
  [ -e "$G/smoke-ok" ] || { log "STOP: expected output missing: $G/smoke-ok"; return 1; }
  for f in "$GPUSMOKE" "$G/smoke-check.txt" "$ROWS732" "$ROWSEXT" "$S/sm139-result.json" "$S/sm139-result.md" "$TICKETS"; do
    [ -s "$f" ] || { log "STOP: expected output missing or empty: $f"; return 1; }
  done
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1 | cut -c1-90) (DRYRUN=$DRYRUN; dirty source files: $(git -C "$WT" status --porcelain -- scripts expertium rwkv | wc -l))"
rc=0
stopped && rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
[ $rc -eq 0 ] && { smoke || rc=1; }
for step in tiny-732 small-732 big-732 tiny-ext small-ext big-ext; do
  [ $rc -eq 0 ] || break
  stopped && { rc=1; break; }
  score "${step%-*}" "${step#*-}" || rc=1
done
STATUS="not run"
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $PY -X utf8 $RULE report"
  else
    out=$("$PY" -X utf8 "$RULE" report 2>&1 | tr -d '\r'); r=$?
    log "report exit $r: $(echo "$out" | tail -1 | cut -c1-600)"
    if [ $r -ne 0 ]; then log "STOP: report exit $r (rows missing?)"; rc=1
    else STATUS=$("$PY" -X utf8 -c "import json; print(json.load(open(r'$S/sm139-result.json'))['status'])" | tr -d '\r'); fi
  fi
fi
[ $rc -eq 0 ] && [ "$DRYRUN" != 1 ] && { final_check || rc=1; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done (status $STATUS)"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
