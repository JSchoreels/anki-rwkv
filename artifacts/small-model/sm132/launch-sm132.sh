#!/usr/bin/env bash
# Again pressed as Hard (SM132) (artifacts/small-model/sm132/design.md), designed 2026-11-19 (no training):
#   how much worse is RWKV when people press Hard instead of Again with probability p? Every Again review is recorded as
#   Hard in the model inputs with probability p (seed [132, user], nested in p; grade one-hot + the card state Anki would
#   record; targets stay true). The show-time two-head cool-down for Small (SM118) at p = 0, 5, 10, 20, 30, 50% and a
#   grade-only arm at 20%; the show-time big model (SM91s2) at 10% and 20% (report only). Published format w8+s8,
#   show-time inputs, leak-free, 723 users of the 732 set.
# Readout (sm132_rule.py decide, pre-registered): the user's bet holds for a head iff the 95% CI upper bound of
#   100 x (corrupted / clean - 1) is <= +2.0% at every p in {5, 10, 20}%; verdict holds iff both heads hold.
# Order: CPU proofs (sm132_check.py, ALL OK required, under memguard.ps1) -> ONE GPU ticket: GPU smoke (2 users; a
#   mismatch = STOP), s0 s5 s10 s20 s30 s50 s20g b10 b20 -> retry tickets for missing users (cap 0.8, then 0.92 on no
#   progress, then go on: the readout runs on the users every p has, at least 718 of 722) -> rule decide.
# Quiet windows: quiet.json scope all/gpu ONLY (sm132_block.py, re-read before every ticket); NO daily or Sunday break.
#   A ticket that cannot end before the next quiet window waits for its end.
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm132/artifacts/small-model/sm132/launch-sm132.sh
#   launch   WMI hidden (HANDOFF.md); log artifacts/small-model/sm132/gpu/launch-sm132.log
#   resume   the launch command again (finished users are skipped by the scorer; the smoke is skipped once gpu/smoke-ok exists)
#   stop     touch C:/Users/jonat/workspace/wt/sm132/artifacts/small-model/sm132/gpu/STOP (between tickets)
# Read only: wt/sm118, wt/sm91s2 (models, rows), wt/sm72s (users). Never touches wt/sm130 or wt/sm131.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm132
S=$WT/artifacts/small-model/sm132
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm132.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm132-dryrun.log
RULE=$S/sm132_rule.py
OUTROWS=$G/sm132-rows.jsonl
TICKETS=$G/tickets.jsonl
STOPF=$G/STOP
W=C:/Users/jonat/workspace/wt
TRACK=E89-F65e-F77a
CK118=$W/sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2/expertium/$TRACK/master.pth
CK91=$W/sm91s2/artifacts/small-model/sm91s2-big-showtime/expertium/big-showtime/master.pth
SHA118=b4fe1175d25a7e6abff70e87ac0af1f4d76db9622d54f7cb53196fbf35117eb6
SHA91=67ee8e9b0d0169733242c40646e0d2e5637845a25775762002ffc4e6396f9590
T118=sm118-e26-dec2ep32k-t91s2; T91=sm91s2-big-showtime
STEPS="s0 s5 s10 s20 s30 s50 s20g b10 b20"
USERS732=$W/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json
NUSERS=723
MIN_SMOKE=5; MIN_SMALL=12; MIN_BIG=16; MIN_RETRY=4
NAME="Again pressed as Hard (SM132)"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START SM11_INPUTS
log() { echo "[sm132 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$STOPF" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }
free_mib() { awk '/MemFree/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 99999; }
block() { "$PY" -X utf8 "$S/sm132_block.py"; }
jline() { "$PY" -X utf8 -c "import json, sys; print(json.dumps(dict(zip(sys.argv[1::2], [int(v) if v.lstrip('-').isdigit() else v for v in sys.argv[2::2]]))))" "$@"; }
label_of() {  # label_of STEP
  case "$1" in
    s20g) echo "$T118/w8+s8@e2s-query+aah20g";;
    s*) echo "$T118/w8+s8@e2s-query+aah${1#s}";;
    b*) echo "$T91/w8+s8@e2s-query+aah${1#b}";;
  esac
}

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
    "$PY" "$QUEUE" run --owner sm132 --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm132/launch-sm132.sh" -- "$@"
    rc=$?; kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null; mx=$(cat "$G/.gpu-max" 2>/dev/null || echo 0)
    log "exit $rc: $purpose (nvidia-smi max $mx MiB, baseline $base MiB)"
    [ "$rc" = 75 ] && { log "queue wait would reach the next quiet window; re-planning"; sleep 60; continue; }
    jline kind "$kind" purpose "$purpose" exit "$rc" minutes "$(( ($(date +%s) - t0 + 30) / 60 ))" nvidia_smi_max_mib "${mx:-0}" nvidia_smi_baseline_mib "${base:-0}" >> "$TICKETS"
    return $rc
  done
}

n_users() {  # n_users ROWS LABEL: scored users of LABEL (unparseable lines skipped and counted in the log)
  F="$1" "$PY" -X utf8 -c "
import json, os, sys
from pathlib import Path
p = Path(os.environ['F'])
got, bad = set(), 0
if p.exists():
    for raw in open(p, 'rb'):
        line = raw.decode('utf-8', errors='replace').strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            if r['label'] == sys.argv[1] and ('benchmark' in r or ('nonfinite' in r and 'error' not in r)):
                got.add(r['user_id'])
        except (ValueError, KeyError, TypeError, AttributeError):
            bad += 1
if bad:
    print(f'{bad} unparseable line(s) in {p.name} skipped', file=sys.stderr)
print(len(got))" "$2" 2>> "$LOG"; }

cpu_checks() {  # the CPU proofs (sm132_check.py) under the memory guard
  [ -e "$G/cpu-ok" ] && return 0
  [ "$DRYRUN" = 1 ] && { log "DRYRUN: $PY -X utf8 $S/sm132_check.py (under memguard.ps1)"; return 0; }
  wait_free 15 "CPU proofs" || return 1
  ram_gate "CPU proofs"
  rm -f "$G/memguard.off"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$S/memguard.ps1")" -Dir "$(cygpath -w "$G")" \
    -Match "sm132_check.py;sm132_rule.py;small_model_heldout.py&--device&cpu" > /dev/null 2>&1 &
  local mg=$!
  CUDA_VISIBLE_DEVICES= "$PY" -X utf8 "$S/sm132_check.py" >> "$G/check.out" 2>&1; local rc=$?
  touch "$G/memguard.off"; sleep 12; kill "$mg" 2>/dev/null
  [ -e "$G/memguard-kill" ] && { log "STOP: memory guard killed a CPU proof process ($(tail -1 "$G/memguard-kill"))"; return 1; }
  [ "$rc" = 0 ] || { log "STOP: CPU proofs failed (see $G/check.out, $S/check-cpu-sm132.json)"; return 1; }
  log "CPU proofs ALL OK ($(grep -c '^PASS' "$G/check.out") PASS lines, $("$PY" -c "import json; print(json.load(open(r'$S/check-cpu-sm132.json'))['seconds'])") s)"
  touch "$G/cpu-ok"
}

preconditions() {
  for f in "$S/sm132_block.py" "$RULE" "$S/sm132_check.py" "$S/sm132_ticket.sh" "$S/smoke-users.json" \
           "$S/cpu/smoke-cpu-rows.jsonl" "$CK118" "$CK91" "$USERS732"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  local sha
  for pair in "$CK118 $SHA118" "$CK91 $SHA91"; do
    set -- $pair
    sha=$(sha256sum "$1" | cut -d' ' -f1); [ "$sha" = "$2" ] || { log "STOP: $1 changed (sha256 $sha)"; return 1; }
  done
  "$PY" -X utf8 "$RULE" selftest >> "$LOG" 2>&1 || { log "STOP: rule selftest failed"; return 1; }
  return 0
}

gpu_passes() {  # one chained ticket (smoke + every step), then retry tickets for steps with missing users
  local pass cap=0.8 prev=-1 steps minutes sig st n done_s
  for pass in 1 2 3 4 5 6; do
    steps=""; minutes=0; sig=""
    [ -e "$G/smoke-ok" ] || { steps="smoke"; minutes=$MIN_SMOKE; }
    for st in $STEPS; do
      n=$(n_users "$OUTROWS" "$(label_of "$st")"); sig="$sig$n/"
      [ "$n" -ge $NUSERS ] && continue
      steps="$steps $st"
      if [ "$n" -gt 600 ]; then minutes=$(( minutes + MIN_RETRY )); else
        case "$st" in b*) minutes=$(( minutes + MIN_BIG ));; *) minutes=$(( minutes + MIN_SMALL ));; esac; fi
    done
    [ -z "$steps" ] && { log "all steps scored ($sig of $NUSERS users)"; return 0; }
    [ "$DRYRUN" = 1 ] && steps="smoke $STEPS" && minutes=$(( MIN_SMOKE + 7 * MIN_SMALL + 2 * MIN_BIG ))
    if [ "$steps" = "smoke" ] || [ "$sig" != "$prev" ] || [ "$pass" = 1 ]; then :; else
      if [ "$cap" = 0.8 ]; then cap=0.92; log "no progress ($sig of $NUSERS users): next pass at cap 0.92 (memory only, same numerics)"
      else log "no progress at cap 0.92 ($sig of $NUSERS users); going on to the rule (it runs on the users every p has)"; return 0; fi
    fi
    prev=$sig
    ram_gate "GPU pass $pass"
    q $minutes score "scoring pass $pass:$steps (users done per step $sig of $NUSERS, cap $cap)" -- \
      bash "$S/sm132_ticket.sh" "$cap" $steps
    local rc=$?
    [ "$DRYRUN" = 1 ] && return 0
    case $rc in
      0) ;;
      3) log "STOP: GPU smoke mismatch (GPU vs CPU rows, see $G/smoke-check.json)"; return 1;;
      *) stopped && return 1; log "STOP: scoring ticket failed (exit $rc, see $G/score.out)"; return 1;;
    esac
  done
  log "6 passes done; going on to the rule"
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN; quiet.json windows only, no daily break; steps $STEPS)"
rc=0
wait_free 10 "preparation" || rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
[ $rc -eq 0 ] && { cpu_checks || rc=1; }
[ $rc -eq 0 ] && { gpu_passes || rc=1; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: then $PY -X utf8 $RULE decide"; log "DRYRUN OK"; exit 0; fi
  out=$("$PY" -X utf8 "$RULE" decide 2>&1); r=$?
  log "rule decide exit $r: $(echo "$out" | tail -1 | cut -c1-900)"
  if [ "$r" = 0 ]; then
    log "done (verdict $("$PY" -X utf8 -c "import json; print('the bet ' + json.load(open(r'$S/sm132-decision.json'))['verdict'])"))"
  elif [ "$r" = 2 ]; then
    log "STOP: the rule is pending (fewer than 718 of 722 users at p = 0 and every p; see the line above)"; rc=1
  else
    log "STOP: the rule did not decide (exit $r)"; rc=1
  fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
