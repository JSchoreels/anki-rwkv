#!/usr/bin/env bash
# Compact state for Small and Big (SM131) (artifacts/small-model/sm131/design.md), designed 2026-11-19 (no training):
#   is the compact integer state of the compact int state test (SM47) (w8+s4c8nhx8: int4 cards + int8 notes, float16 row
#   scales, int8 token-shift vectors) free against the fork's default stored state (w8+s4c8n) for the bigger models?
#   Big: the show-time big model (SM91s2) and the show-time instant-only big model (SM99), scored at fork default, compact
#   and card-only (w8+s4hgx4c8hx8n, report only); Small: the show-time two-head cool-down for Small (SM118) and the
#   longer Small Instant-Only model (SM113) at compact and card-only (fork-default rows from SM129, same tree).
#   Show-time inputs, leak-free, 723 users of the 732 set.
# Rule (sm131_rule.py decide, pre-registered): free(model) iff every head's 95% CI upper bound of
#   100 x (compact / fork default - 1) < +0.05%; verdict per size: free iff both models free.
# Order: CPU proofs (sm131_check.py, ALL OK required, under memguard.ps1) -> ONE GPU ticket: GPU smoke (2 users, four
#   formats, GPU vs CPU rows; a mismatch = STOP), Big, Small -> retry tickets for missing users (cap 0.8, then 0.92 on no
#   progress, then go on: the rule runs on the users both formats have, at least 718 of 722) -> rule decide.
# Quiet windows: quiet.json scope all/gpu ONLY (sm131_block.py, re-read before every ticket); NO daily or Sunday break
#   (user 11-19 ~14:10). A ticket that cannot end before the next quiet window waits for its end.
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm131/artifacts/small-model/sm131/launch-sm131.sh
#   launch   WMI hidden (HANDOFF.md); log artifacts/small-model/sm131/gpu/launch-sm131.log
#   resume   the launch command again (finished users are skipped by the scorer; the smoke is skipped once gpu/smoke-ok exists)
#   stop     touch C:/Users/jonat/workspace/wt/sm131/artifacts/small-model/sm131/gpu/STOP (between tickets)
# Read only: wt/sm91s2, wt/sm99, wt/sm118, wt/sm113 (models, rows), wt/sm72s (users). Never touches wt/sm130.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm131
S=$WT/artifacts/small-model/sm131
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm131.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm131-dryrun.log
RULE=$S/sm131_rule.py
OUTROWS=$G/sm131-rows.jsonl
TICKETS=$G/tickets.jsonl
STOPF=$G/STOP
W=C:/Users/jonat/workspace/wt
TRACK=E89-F65e-F77a
CK91=$W/sm91s2/artifacts/small-model/sm91s2-big-showtime/expertium/big-showtime/master.pth
CK99=$W/sm99/artifacts/small-model/sm99-big-inst-e2/expertium/big-inst/master.pth
CK118=$W/sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2/expertium/$TRACK/master.pth
CK113=$W/sm113/artifacts/small-model/sm113-inst-e35st-st-dec2ep32k-t99-p2/expertium/$TRACK/master.pth
SHA91=67ee8e9b0d0169733242c40646e0d2e5637845a25775762002ffc4e6396f9590
SHA99=c49e45d698194da033caa1231f598f875922a4a2a3e921714a3fd433d37e5768
SHA118=b4fe1175d25a7e6abff70e87ac0af1f4d76db9622d54f7cb53196fbf35117eb6
SHA113=e4574e2aa8b2cfe6360dbd86f2a75b71d23ea81677b00ffcc9c3abce6b4cb736
T91=sm91s2-big-showtime; T99=sm99-big-inst-e2; T118=sm118-e26-dec2ep32k-t91s2; T113=sm113-inst-e35st-st-dec2ep32k-t99
BIG_LABELS="$T91/w8+s4c8n@e2s-query $T91/w8+s4c8nhx8@e2s-query $T91/w8+s4hgx4c8hx8n@e2s-query $T99/w8+s4c8n@e2s-query $T99/w8+s4c8nhx8@e2s-query $T99/w8+s4hgx4c8hx8n@e2s-query"
SMALL_LABELS="$T118/w8+s4c8nhx8@e2s-query $T118/w8+s4hgx4c8hx8n@e2s-query $T113/w8+s4c8nhx8@e2s-query $T113/w8+s4hgx4c8hx8n@e2s-query"
USERS732=$W/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json
NUSERS=723
MIN_SMOKE=5; MIN_BIG=95; MIN_SMALL=50; MIN_BIG_RETRY=25; MIN_SMALL_RETRY=15
NAME="Compact state for Small and Big (SM131)"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START SM11_INPUTS
log() { echo "[sm131 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$STOPF" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }
free_mib() { awk '/MemFree/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 99999; }
block() { "$PY" -X utf8 "$S/sm131_block.py"; }
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
    "$PY" "$QUEUE" run --owner sm131 --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm131/launch-sm131.sh" -- "$@"
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
            if 'benchmark' in r:
                got.setdefault(r.get('label'), set()).add(r['user_id'])
        except (ValueError, KeyError, TypeError, AttributeError):
            bad += 1
if bad:
    print(f'{bad} unparseable line(s) in {p.name} skipped', file=sys.stderr)
print(min(len(got.get(l, ())) for l in sys.argv[1:]))" "${@:2}" 2>> "$LOG"; }

cpu_checks() {  # the CPU proofs (sm131_check.py) under the memory guard
  [ -e "$G/cpu-ok" ] && return 0
  [ "$DRYRUN" = 1 ] && { log "DRYRUN: $PY -X utf8 $S/sm131_check.py (under memguard.ps1)"; return 0; }
  wait_free 10 "CPU proofs" || return 1
  ram_gate "CPU proofs"
  rm -f "$G/memguard.off"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$S/memguard.ps1")" -Dir "$(cygpath -w "$G")" \
    -Match "sm131_check.py;sm131_bytes.py;sm131_rule.py" > /dev/null 2>&1 &
  local mg=$!
  CUDA_VISIBLE_DEVICES= "$PY" -X utf8 "$S/sm131_check.py" >> "$G/check.out" 2>&1; local rc=$?
  touch "$G/memguard.off"; sleep 12; kill "$mg" 2>/dev/null
  [ -e "$G/memguard-kill" ] && { log "STOP: memory guard killed a CPU proof process ($(tail -1 "$G/memguard-kill"))"; return 1; }
  [ "$rc" = 0 ] || { log "STOP: CPU proofs failed (see $G/check.out, $S/check-cpu-sm131.json)"; return 1; }
  log "CPU proofs ALL OK ($(grep -c '^PASS' "$G/check.out") PASS lines, $("$PY" -c "import json; print(json.load(open(r'$S/check-cpu-sm131.json'))['seconds'])") s)"
  touch "$G/cpu-ok"
}

preconditions() {
  for f in "$S/sm131_block.py" "$RULE" "$S/sm131_check.py" "$S/sm131_bytes.py" "$S/sm131_ticket.sh" "$S/smoke-users.json" \
           "$S/cpu/smoke-cpu-rows.jsonl" "$CK91" "$CK99" "$CK118" "$CK113" "$USERS732"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  local sha
  for pair in "$CK91 $SHA91" "$CK99 $SHA99" "$CK118 $SHA118" "$CK113 $SHA113"; do
    set -- $pair
    sha=$(sha256sum "$1" | cut -d' ' -f1); [ "$sha" = "$2" ] || { log "STOP: $1 changed (sha256 $sha)"; return 1; }
  done
  "$PY" -X utf8 "$RULE" selftest >> "$LOG" 2>&1 || { log "STOP: rule selftest failed"; return 1; }
  return 0
}

gpu_passes() {  # one chained ticket (smoke + big + small), then retry tickets for missing users
  local pass cap=0.8 hb hs prev=-1 steps minutes sig
  for pass in 1 2 3 4 5 6; do
    hb=$(min_users "$OUTROWS" $BIG_LABELS); hs=$(min_users "$OUTROWS" $SMALL_LABELS)
    steps=""; minutes=0
    [ -e "$G/smoke-ok" ] || { steps="smoke"; minutes=$MIN_SMOKE; }
    [ "$hb" -ge $NUSERS ] || { steps="$steps big"; minutes=$(( minutes + ( hb > 600 ? MIN_BIG_RETRY : MIN_BIG ) )); }
    [ "$hs" -ge $NUSERS ] || { steps="$steps small"; minutes=$(( minutes + ( hs > 600 ? MIN_SMALL_RETRY : MIN_SMALL ) )); }
    [ -z "$steps" ] && { log "all labels scored: Big $hb, Small $hs of $NUSERS users"; return 0; }
    [ "$DRYRUN" = 1 ] && steps="smoke big small" && minutes=$(( MIN_SMOKE + MIN_BIG + MIN_SMALL ))
    sig="$hb/$hs"
    if [ "$sig" = "$prev" ]; then
      if [ "$cap" = 0.8 ]; then cap=0.92; log "no progress (Big $hb, Small $hs of $NUSERS users): next pass at cap 0.92 (memory only, same numerics)"
      else log "no progress at cap 0.92 (Big $hb, Small $hs of $NUSERS users); going on to the rule (it runs on the users both formats have)"; return 0; fi
    fi
    prev=$sig
    ram_gate "GPU pass $pass"
    q $minutes score "scoring pass $pass:$steps (Big $hb, Small $hs of $NUSERS users done, cap $cap)" -- \
      bash "$S/sm131_ticket.sh" "$cap" $steps
    local rc=$?
    [ "$DRYRUN" = 1 ] && return 0
    case $rc in
      0) ;;
      3) log "STOP: GPU smoke mismatch (GPU vs CPU rows, see $G/smoke-check.json)"; return 1;;
      *) stopped && return 1; log "STOP: scoring ticket failed (exit $rc, see $G/score.out)"; return 1;;
    esac
  done
  log "6 passes done (Big $(min_users "$OUTROWS" $BIG_LABELS), Small $(min_users "$OUTROWS" $SMALL_LABELS) of $NUSERS users); going on to the rule"
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN; quiet.json windows only, no daily break; fork default w8+s4c8n, compact w8+s4c8nhx8, card-only w8+s4hgx4c8hx8n)"
rc=0
wait_free 10 "preparation" || rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
[ $rc -eq 0 ] && { cpu_checks || rc=1; }
[ $rc -eq 0 ] && { gpu_passes || rc=1; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: then $PY -X utf8 $RULE decide"; log "DRYRUN OK"; exit 0; fi
  out=$("$PY" -X utf8 "$RULE" decide 2>&1); r=$?
  log "rule decide exit $r: $(echo "$out" | tail -1 | cut -c1-600)"
  if [ "$r" = 0 ]; then
    log "done (verdict $("$PY" -X utf8 -c "import json; v=json.load(open(r'$S/sm131-decision.json'))['verdict']; print('Small ' + v['Small'] + ', Big ' + v['Big'])"))"
  elif [ "$r" = 2 ]; then
    log "STOP: the rule is pending (fewer than 718 of 722 users in both formats for a model; see the line above)"; rc=1
  else
    log "STOP: the rule did not decide (exit $r)"; rc=1
  fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
