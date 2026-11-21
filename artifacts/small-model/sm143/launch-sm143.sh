#!/usr/bin/env bash
# Per-user adaptation (SM143) pilot (artifacts/small-model/sm143/design.md, pre-registered 2026-11-21 before any GPU
#   ticket). Every GPU step is one scripts/r01_queue.py ticket (owner sm143, claim gpu, cap 0.8, then 0.92 after a pass
#   without progress: memory only, same numerics):
#   1 smoke   the 2 smoke users, Small and Tiny, every variant at 0 steps (GPU rows vs the committed CPU rows,
#             sm143_check.py smoke, |GPU - CPU| <= 1e-5); failure -> STOP
#   1b steps  every variant with its steps on user 2388 (the full fine-tune's training kernel on the GPU); a failure
#             only skips step 5
#   2 small   variants a,b,c of the longer Small Instant-Only model (SM113) on the 200 pilot users
#   3 big     the big instant-only model (SM99), unadapted, stock scorer with --dump on the 200 pilot users
#   4 tiny    variants a,b,c of the six-layer Tiny Instant-Only model (SM130) on the 200 pilot users
#   5 full    variant d (+ a,b,c again) of Small on the 50 users of users-pilot-d.json (same rows file; the newer row wins)
#   6 report  sm143_report.py report (CPU) -> sm143-result.{json,md}
#   All: show-time inputs (e2s-query), leak-free (--deleted-cards neighbour), fp32.
# Breaks: quiet.json windows of scope all / gpu only (sm135_block.py); no planned daily or Sunday break.
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm143/artifacts/small-model/sm143/launch-sm143.sh
#   launch   WMI hidden (HANDOFF.md); log artifacts/small-model/sm143/gpu/launch-sm143.log
#   resume   remove gpu/STOP, then the launch command again (finished users and steps are skipped)
#   stop     touch C:/Users/jonat/workspace/wt/sm143/artifacts/small-model/sm143/gpu/STOP
# Do not edit scripts/ expertium/ rwkv/ of the sm143 worktree while it runs.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm143
SM=$WT/artifacts/small-model
S=$SM/sm143
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm143.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm143-dryrun.log
AD=$WT/scripts/sm143_adapt.py
SC=$WT/scripts/small_model_heldout.py
CHECK=$S/sm143_check.py
RULE=$S/sm143_report.py
BLOCK=C:/Users/jonat/workspace/wt/sm135/artifacts/small-model/sm135/sm135_block.py
. "$S/models.sh"
IFS='|' read -r ST SK SP SH SR <<< "$SMALL"
IFS='|' read -r TT TK TP TH TR <<< "$TINY"
IFS='|' read -r BT BK BP BH BR <<< "$BIG"
USMOKE=$S/users-smoke.json
UPILOT=$S/users-pilot.json
UD=$S/users-pilot-d.json
ROWS_S=$G/rows-small.jsonl
ROWS_T=$G/rows-tiny.jsonl
ROWS_B=$G/rows-big.jsonl
NPZ_S=$G/npz-small
NPZ_T=$G/npz-tiny
DUMP_B=$G/dump-big
TICKETS=$G/tickets.jsonl
STOPF=$G/STOP
NAME="Per-user adaptation (SM143)"
declare -A MIN=([small]=45 [big]=30 [tiny]=35 [full]=150)
MIN_SMOKE=20; MIN_RETRY=25
WANT=200; WANT_D=50
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START SM11_INPUTS
log() { echo "[sm143 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
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
    "$PY" "$QUEUE" run --owner sm143 --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm143/launch-sm143.sh" -- "$@"
    rc=$?; kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null; mx=$(cat "$G/.gpu-max" 2>/dev/null | tr -dc '0-9'); mx=${mx:-0}
    log "exit $rc: $purpose (nvidia-smi max $mx MiB, baseline $base MiB)"
    [ "$rc" = 75 ] && { log "queue wait would reach the next quiet window; re-planning"; sleep 60; continue; }
    jline kind "$kind" purpose "$purpose" exit "$rc" minutes "$(( ($(date +%s) - t0 + 30) / 60 ))" nvidia_smi_max_mib "$mx" nvidia_smi_baseline_mib "${base:-0}" >> "$TICKETS"
    return $rc
  done
}

n_done() {  # n_done ROWS LABEL [VARIANT]: users with a complete row of LABEL (containing VARIANT if given)
  F="$1" "$PY" -X utf8 -c "
import json, os, sys
from pathlib import Path
p = Path(os.environ['F'])
got = set()
if p.exists():
    for raw in open(p, 'rb'):
        try:
            r = json.loads(raw.decode('utf-8', errors='replace'))
        except ValueError:
            continue
        if r.get('label') == sys.argv[1] and 'error' not in r and ('benchmark' in r or 'nonfinite' in r):
            if len(sys.argv) < 3 or sys.argv[2] in r.get('variants', [sys.argv[2]]):
                got.add(r['user_id'])
print(len(got))" "${@:2}" | tr -d '\r'; }

smoke() {  # 0-step rows of both models on the GPU vs the committed CPU rows (<= 1e-5); failure -> STOP
  [ -e "$G/smoke-ok" ] && return 0
  local out1 out2 E="env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS HELDOUT_GPU_MEMORY_FRACTION=0.8"
  q $MIN_SMOKE smoke "GPU smoke: 2 users, Small and Tiny, every variant at 0 steps vs the CPU rows" -- \
    bash -c "$E '$PY' -X utf8 -W ignore '$AD' --model '$ST=$SK:$SP' --users-file '$USMOKE' --fsrs '$FSRS' --variants a,b,c,d \
      --steps-scale 0 --out '$G/smoke-gpu-small.jsonl' && $E '$PY' -X utf8 -W ignore '$AD' --model '$TT=$TK:$TP' \
      --users-file '$USMOKE' --fsrs '$FSRS' --variants a,b,c,d --steps-scale 0 --out '$G/smoke-gpu-tiny.jsonl'" \
    >> "$G/adapt.out" 2>&1 || { log "STOP: smoke ticket failed (see $G/adapt.out)"; return 1; }
  [ "$DRYRUN" = 1 ] && return 0
  out1=$("$PY" -X utf8 "$CHECK" smoke --gpu "$G/smoke-gpu-small.jsonl" --cpu "$S/cpu/zero-small.jsonl" 2>&1 | tr -d '\r')
  out2=$("$PY" -X utf8 "$CHECK" smoke --gpu "$G/smoke-gpu-tiny.jsonl" --cpu "$S/cpu/zero-tiny.jsonl" 2>&1 | tr -d '\r')
  printf '%s\n%s\n' "$out1" "$out2" > "$G/smoke-check.txt"; log "small: $out1 | tiny: $out2"
  case "$out1" in *"smoke OK"*) ;; *) log "STOP: GPU smoke mismatch (see $G/smoke-check.txt)"; return 1;; esac
  case "$out2" in *"smoke OK"*) touch "$G/smoke-ok"; return 0;; esac
  log "STOP: GPU smoke mismatch (see $G/smoke-check.txt)"; return 1
}

smoke_steps() {  # every variant with its steps on user 2388 (the full fine-tune's training kernel on the GPU).
  # A failure only disables the full fine-tune step (variants a, b, c are checked again by their own tickets).
  [ -e "$G/steps-ok" ] || [ -e "$G/steps-failed" ] && return 0
  local out
  q $MIN_SMOKE smoke "GPU real-step smoke: every variant with its steps on user 2388" -- \
    env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS HELDOUT_GPU_MEMORY_FRACTION=0.8 "$PY" -X utf8 -W ignore "$AD" \
    --model "$ST=$SK:$SP" --users-file "$S/users-smoke-one.json" --fsrs "$FSRS" --variants a,b,c,d --out "$G/smoke-gpu-steps.jsonl" \
    >> "$G/adapt.out" 2>&1
  [ "$DRYRUN" = 1 ] && return 0
  out=$("$PY" -X utf8 "$CHECK" steps --rows "$G/smoke-gpu-steps.jsonl" 2>&1 | tr -d '\r')
  echo "$out" >> "$G/smoke-check.txt"; log "steps: $out"
  case "$out" in *"steps OK"*) touch "$G/steps-ok";; *) touch "$G/steps-failed"; log "real-step smoke failed: the full fine-tune step is skipped (see $G/adapt.out)";; esac
  return 0
}

step() {  # step NAME: one pilot step with the cap rule
  local name=$1 cap=0.8 pass have prev=-1 minutes want label rows cmd tl
  case "$name" in
    small) want=$WANT; label="$ST/fp32@e2s-query"; rows=$ROWS_S;;
    tiny)  want=$WANT; label="$TT/fp32@e2s-query"; rows=$ROWS_T;;
    big)   want=$WANT; label="$BT/fp32@e2s-query"; rows=$ROWS_B;;
    full)  want=$WANT_D; label="$ST/fp32@e2s-query"; rows=$ROWS_S;;
  esac
  for pass in 1 2 3 4; do
    if [ "$name" = full ]; then have=$(n_done "$rows" "$label" d); else have=$(n_done "$rows" "$label"); fi
    [ "$have" -ge "$want" ] && break
    if [ "$have" = "$prev" ]; then
      if [ "$cap" = 0.8 ]; then cap=0.92; log "no progress on $name ($have of $want users): next pass at cap 0.92 (memory only, same numerics)"
      else log "no progress at cap 0.92 on $name ($have of $want users); going on"; break; fi
    fi
    prev=$have
    minutes=${MIN[$name]}; [ "$pass" -gt 1 ] && minutes=$MIN_RETRY
    tl=$(( minutes * 60 * 6 / 5 ))
    ram_gate "$name"
    local E="env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS HELDOUT_GPU_MEMORY_FRACTION=$cap"
    case "$name" in
      small) q $minutes adapt "variants a,b,c of the longer Small Instant-Only model (SM113), 200 pilot users (pass $pass, $have done, cap $cap)" -- \
               $E "$PY" -X utf8 -W ignore "$AD" --model "$ST=$SK:$SP" --users-file "$UPILOT" --fsrs "$FSRS" --variants a,b,c \
               --out "$rows" --save "$NPZ_S" --time-limit $tl >> "$G/adapt.out" 2>&1;;
      tiny)  q $minutes adapt "variants a,b,c of the six-layer Tiny Instant-Only model (SM130), 200 pilot users (pass $pass, $have done, cap $cap)" -- \
               $E "$PY" -X utf8 -W ignore "$AD" --model "$TT=$TK:$TP" --users-file "$UPILOT" --fsrs "$FSRS" --variants a,b,c \
               --out "$rows" --save "$NPZ_T" --time-limit $tl >> "$G/adapt.out" 2>&1;;
      big)   q $minutes score "the big instant-only model (SM99), unadapted, fp32 on e2s-query inputs with --dump, 200 pilot users (pass $pass, $have done, cap $cap)" -- \
               $E "$PY" -X utf8 -W ignore "$SC" --pool 732 --users-file "$UPILOT" --model "$BT=$BK:$BP" --only "$BT/fp32" \
               --inputs e2s-query --deleted-cards neighbour --nonfinite-rows --dump "${DUMP_B#$WT/}" --time-limit $tl \
               --out "${rows#$WT/}" >> "$G/score.out" 2>&1;;
      full)  q $minutes adapt "variant d (full fine-tune) + a,b,c of the longer Small Instant-Only model (SM113), the 50 users of the d subset (pass $pass, $have done, cap $cap)" -- \
               $E "$PY" -X utf8 -W ignore "$AD" --model "$ST=$SK:$SP" --users-file "$UPILOT" --d-users-file "$UD" --fsrs "$FSRS" \
               --variants a,b,c,d --out "$rows" --save "$NPZ_S" --time-limit $tl >> "$G/adapt.out" 2>&1;;
    esac
    [ $? -ne 0 ] && { log "STOP: ticket failed ($name)"; return 1; }
    [ "$DRYRUN" = 1 ] && break
  done
  [ "$DRYRUN" = 1 ] && return 0
  if [ "$name" = full ]; then have=$(n_done "$rows" "$label" d); else have=$(n_done "$rows" "$label"); fi
  log "$name: $have of $want users done"
  return 0
}

preconditions() {
  local f
  for f in "$BLOCK" "$RULE" "$CHECK" "$AD" "$SC" "$S/models.sh" "$S/design.md" "$USMOKE" "$S/users-smoke-one.json" "$UPILOT" "$UD" \
           "$S/cpu/zero-small.jsonl" "$S/cpu/zero-tiny.jsonl" "$SP" "$TP" "$BP"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  [ -d "$FSRS" ] || { log "STOP: missing $FSRS"; return 1; }
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/"; return 1; fi
  "$PY" -X utf8 "$BLOCK" selftest >> "$LOG" 2>&1 || { log "STOP: block selftest failed"; return 1; }
  "$PY" -X utf8 "$AD" selftest >> "$LOG" 2>&1 || { log "STOP: causality selftest failed"; return 1; }
  "$PY" -X utf8 "$RULE" selftest >> "$LOG" 2>&1 || { log "STOP: report selftest failed"; return 1; }
  "$PY" -X utf8 "$CHECK" pins --models "$S/models.sh" --pins "$S/design-pins.json" >> "$LOG" 2>&1 || { log "STOP: pins changed"; return 1; }
  return 0
}

final_check() {  # every expected output exists (non-empty) before done
  local f
  [ -e "$G/smoke-ok" ] || { log "STOP: expected output missing: $G/smoke-ok"; return 1; }
  for f in "$G/smoke-check.txt" "$ROWS_S" "$ROWS_T" "$ROWS_B" "$S/sm143-result.json" "$S/sm143-result.md" "$TICKETS"; do
    [ -s "$f" ] || { log "STOP: expected output missing or empty: $f"; return 1; }
  done
  [ -d "$NPZ_S" ] && [ -d "$DUMP_B" ] || { log "STOP: per-user arrays missing ($NPZ_S or $DUMP_B)"; return 1; }
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1 | cut -c1-90) (DRYRUN=$DRYRUN; dirty source files: $(git -C "$WT" status --porcelain -- scripts expertium rwkv | wc -l))"
rc=0
stopped && rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
[ $rc -eq 0 ] && { smoke || rc=1; }
[ $rc -eq 0 ] && { smoke_steps || rc=1; }
for s in small big tiny full; do
  [ $rc -eq 0 ] || break
  stopped && { rc=1; break; }
  if [ "$s" = full ] && [ "$DRYRUN" != 1 ] && [ ! -e "$G/steps-ok" ]; then log "full fine-tune step skipped (real-step smoke failed)"; continue; fi
  step "$s" || rc=1
done
STATUS="not run"
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $PY -X utf8 $RULE report --small $ROWS_S --tiny $ROWS_T --small-npz $NPZ_S --big-dump $DUMP_B"
  else
    out=$("$PY" -X utf8 "$RULE" report --small "$ROWS_S" --tiny "$ROWS_T" --small-npz "$NPZ_S" --big-dump "$DUMP_B" \
          --big-label "$BT/fp32@e2s-query" --out-json "$S/sm143-result.json" --out-md "$S/sm143-result.md" 2>&1 | tr -d '\r'); r=$?
    log "report exit $r: $(echo "$out" | tail -1 | cut -c1-600)"
    if [ $r -ne 0 ]; then log "STOP: report exit $r (rows missing?)"; rc=1
    else STATUS=$("$PY" -X utf8 -c "import json; print(json.load(open(r'$S/sm143-result.json'))['status'])" | tr -d '\r'); fi
  fi
fi
[ $rc -eq 0 ] && [ "$DRYRUN" != 1 ] && { final_check || rc=1; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done (status $STATUS)"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
