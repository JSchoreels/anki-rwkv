#!/usr/bin/env bash
# Six-layer Tiny with the card-only state (SM134) (artifacts/small-model/sm134/design.md), designed 2026-11-20 (no training):
#   are the six-layer Tiny models of six-layer Tiny (SM130) still within +1.00% of Small per head with the card-only state
#   w8+s4hgx4c8hx8n (619 B per card, notes 1,315 B)?  Scores ONLY the card-only format for the two SM130 final models (the
#   two-head six-layer Tiny, the Instant-Only six-layer Tiny); published w8+s8 and compact rows are SM130's (inputs/sm130-rows.jsonl).
#   Show-time inputs, leak-free, 723 users of the 732 set.
# Rule (sm134_rule.py decide, pre-registered): go(chain) iff every head's 95% CI upper bound of
#   100 x (card-only six-layer / Small published w8+s8 - 1) <= +1.00%; verdict go iff both chains go.
# Order: preconditions (sha256 of both models, rule selftest) -> CPU proofs (sm134_check.py, ALL OK, under memguard.ps1) ->
#   ONE GPU ticket: GPU smoke (3 users, 3 formats, GPU vs CPU / published rows; a mismatch = STOP), card-only scoring of both
#   models (cap 0.8) -> retry tickets for missing users (cap 0.8, then 0.92 on no progress) -> rule decide.
# Quiet windows: quiet.json scope all/gpu ONLY (sm134_block.py, re-read before every ticket); NO daily or Sunday break.
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm134/artifacts/small-model/sm134/launch-sm134.sh
#   launch   WMI hidden (HANDOFF.md); log artifacts/small-model/sm134/gpu/launch-sm134.log
#   resume   the launch command again (finished users are skipped by the scorer; the smoke is skipped once gpu/smoke-ok exists)
#   stop     touch C:/Users/jonat/workspace/wt/sm134/artifacts/small-model/sm134/gpu/STOP (between tickets)
# Read only: wt/sm130 (models, rows), wt/sm113 (rows), wt/sm72s (users). Never edits wt/sm130.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm134
S=$WT/artifacts/small-model/sm134
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm134.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm134-dryrun.log
RULE=$S/sm134_rule.py
OUTROWS=$G/sm134-rows.jsonl
TICKETS=$G/tickets.jsonl
STOPF=$G/STOP
W=C:/Users/jonat/workspace/wt
TRACK=E89-F65e-F77a
CKTH=$W/sm130/artifacts/small-model/sm130-th-dec2ep32k-p2/expertium/$TRACK/master.pth
CKIO=$W/sm130/artifacts/small-model/sm130-io-dec2ep32k-p2/expertium/$TRACK/master.pth
SHATH=48a5e724cf38b53a32d570d4d034791f1da2cfcbe878572ff03ad434d7fb75af
SHAIO=23f037282379199bd113a5297737c920fcbefd702b7b76eecc0095b8a31186a2
TTH=sm130-th-dec2ep32k-t91s2; TIO=sm130-io-dec2ep32k-t99
LABELS="$TTH/w8+s4hgx4c8hx8n@e2s-query $TIO/w8+s4hgx4c8hx8n@e2s-query"
USERS732=$W/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json
NUSERS=723
MIN_SMOKE=5; MIN_SCORE=50; MIN_RETRY=20
NAME="Six-layer Tiny with the card-only state (SM134)"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START SM11_INPUTS
log() { echo "[sm134 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$STOPF" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }
free_mib() { awk '/MemFree/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 99999; }
block() { "$PY" -X utf8 "$S/sm134_block.py"; }
jline() { "$PY" -X utf8 -c "import json, sys; print(json.dumps(dict(zip(sys.argv[1::2], [int(v) if v.lstrip('-').isdigit() else v for v in sys.argv[2::2]]))))" "$@"; }

wait_free() {  # wait_free MINUTES WHAT: returns once MINUTES fit before the next quiet.json window (sets FREE = minutes to it)
  local n=0
  while true; do
    stopped && return 1
    local b; b=$(block)
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

ram_gate() {  # hold while < 3 GB RAM is free, at most 30 min
  local i=0
  while [ "$DRYRUN" != 1 ] && [ "$(free_mib)" -lt 3072 ]; do
    [ $i -ge 6 ] && { log "$1: still $(free_mib) MiB RAM free after 30 min; going on"; return 0; }
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
    "$PY" "$QUEUE" run --owner sm134 --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm134/launch-sm134.sh" -- "$@"
    rc=$?; kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null; mx=$(cat "$G/.gpu-max" 2>/dev/null || echo 0)
    log "exit $rc: $purpose (nvidia-smi max $mx MiB, baseline $base MiB)"
    [ "$rc" = 75 ] && { log "queue wait would reach the next quiet window; re-planning"; sleep 60; continue; }
    jline kind "$kind" purpose "$purpose" exit "$rc" minutes "$(( ($(date +%s) - t0 + 30) / 60 ))" nvidia_smi_max_mib "${mx:-0}" nvidia_smi_baseline_mib "${base:-0}" >> "$TICKETS"
    return $rc
  done
}

min_users() {  # min_users ROWS LABEL...: the smallest number of scored users over the labels (unparseable lines skipped)
  F="$1" "$PY" -X utf8 -c "
import json, os, sys
from pathlib import Path
p = Path(os.environ['F'])
got, bad = {}, 0
if p.exists():
    for raw in open(p, 'rb'):
        line = raw.decode('utf-8', errors='replace').strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            if 'benchmark' in r or ('nonfinite' in r and 'error' not in r):   # amendment 11-19: nonfinite rows are scored
                got.setdefault(r.get('label'), set()).add(r['user_id'])
        except (ValueError, KeyError, TypeError, AttributeError):
            bad += 1
if bad:
    print(f'{bad} unparseable line(s) in {p.name} skipped', file=sys.stderr)
print(min(len(got.get(l, ())) for l in sys.argv[1:]))" "${@:2}" 2>> "$LOG"; }

cpu_checks() {  # the CPU proofs (sm134_check.py) under the memory guard
  [ -e "$G/cpu-ok" ] && return 0
  [ "$DRYRUN" = 1 ] && { log "DRYRUN: $PY -X utf8 $S/sm134_check.py (under memguard.ps1)"; return 0; }
  wait_free 10 "CPU proofs" || return 1
  ram_gate "CPU proofs"
  rm -f "$G/memguard.off"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$S/memguard.ps1")" -Dir "$(cygpath -w "$G")" \
    -Match "sm134_check.py;sm134_rule.py" > /dev/null 2>&1 &
  local mg=$!
  CUDA_VISIBLE_DEVICES= "$PY" -X utf8 "$S/sm134_check.py" >> "$G/check.out" 2>&1; local rc=$?
  touch "$G/memguard.off"; sleep 12; kill "$mg" 2>/dev/null
  [ -e "$G/memguard-kill" ] && { log "STOP: memory guard killed a CPU proof process ($(tail -1 "$G/memguard-kill"))"; return 1; }
  [ "$rc" = 0 ] || { log "STOP: CPU proofs failed (see $G/check.out, $S/check-cpu-sm134.json)"; return 1; }
  log "CPU proofs ALL OK ($(grep -c '^PASS' "$G/check.out") PASS lines, $("$PY" -c "import json; print(json.load(open(r'$S/check-cpu-sm134.json'))['seconds'])") s)"
  touch "$G/cpu-ok"
}

preconditions() {
  for f in "$S/sm134_block.py" "$RULE" "$S/sm134_check.py" "$S/sm134_ticket.sh" "$S/smoke-users.json" "$S/inputs/sm130-rows.jsonl"            "$S/cpu/smoke-cpu-rows.jsonl" "$CKTH" "$CKIO" "$USERS732" "$WT/expertium/variants.py"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  grep -q 'w64-c{C}-d{D}p{P}u{U}\|c1-d2p1u1' "$WT/expertium/variants.py" || { log "STOP: preset w64-c1-d2p1u1 missing in expertium/variants.py"; return 1; }
  local sha
  for pair in "$CKTH $SHATH" "$CKIO $SHAIO"; do
    set -- $pair
    sha=$(sha256sum "$1" | cut -d' ' -f1); [ "$sha" = "$2" ] || { log "STOP: $1 changed (sha256 $sha)"; return 1; }
  done
  "$PY" -X utf8 "$RULE" selftest >> "$LOG" 2>&1 || { log "STOP: rule selftest failed"; return 1; }
  return 0
}

gpu_passes() {  # one chained ticket (smoke + score), then retry tickets for missing users
  local pass cap=0.8 have prev=-1 steps minutes sig
  for pass in 1 2 3 4 5; do
    have=$(min_users "$OUTROWS" $LABELS)
    steps=""; minutes=0
    [ -e "$G/smoke-ok" ] || { steps="smoke"; minutes=$MIN_SMOKE; }
    [ "$have" -ge $NUSERS ] || { steps="$steps score"; minutes=$(( minutes + ( have > 600 ? MIN_RETRY : MIN_SCORE ) )); }
    [ -z "$steps" ] && { log "all labels scored: $have of $NUSERS users"; return 0; }
    [ "$DRYRUN" = 1 ] && steps="smoke score" && minutes=$(( MIN_SMOKE + MIN_SCORE ))
    sig="$have"
    if [ "$sig" = "$prev" ]; then
      if [ "$cap" = 0.8 ]; then cap=0.92; log "no progress ($have of $NUSERS users): next pass at cap 0.92 (memory only, same numerics)"
      else log "no progress at cap 0.92 ($have of $NUSERS users); going on to the rule (it will be pending if users are missing)"; return 0; fi
    fi
    prev=$sig
    ram_gate "GPU pass $pass"
    q $minutes score "scoring pass $pass:$steps ($have of $NUSERS users done, cap $cap)" --       bash "$S/sm134_ticket.sh" "$cap" $steps
    local rc=$?
    [ "$DRYRUN" = 1 ] && return 0
    case $rc in
      0) ;;
      3) log "STOP: GPU smoke mismatch (GPU vs CPU / published rows, see $G/smoke-check.json)"; return 1;;
      *) stopped && return 1; log "STOP: scoring ticket failed (exit $rc, see $G/score.out)"; return 1;;
    esac
  done
  log "5 passes done ($(min_users "$OUTROWS" $LABELS) of $NUSERS users); going on to the rule"
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN; quiet.json windows only, no daily break; card-only w8+s4hgx4c8hx8n for the two SM130 final models)"
rc=0
wait_free 10 "preparation" || rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
[ $rc -eq 0 ] && { cpu_checks || rc=1; }
[ $rc -eq 0 ] && { gpu_passes || rc=1; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: then $PY -X utf8 $RULE decide"; log "DRYRUN OK"; exit 0; fi
  out=$("$PY" -X utf8 "$RULE" decide 2>&1); r=$?
  log "rule decide exit $r: $(echo "$out" | tail -1 | cut -c1-700)"
  if [ "$r" = 0 ]; then
    log "done (verdict $("$PY" -X utf8 -c "import json; print(json.load(open(r'$S/sm134-decision.json'))['verdict'])"))"
  elif [ "$r" = 2 ]; then
    log "STOP: the rule is pending (card-only rows missing for some of the 722 users; see the line above)"; rc=1
  else
    log "STOP: the rule did not decide (exit $r)"; rc=1
  fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
