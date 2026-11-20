#!/usr/bin/env bash
# Compact state for Tiny (SM129) (artifacts/small-model/sm129/design.md), designed 2026-11-18 (no training):
#   score the current Tiny candidates, the show-time two-head cool-down for Tiny (SM124) and the longer cool-down for Tiny
#   (SM115), with the compact integer state of the compact int state test (SM47) (w8+s4c8nhx8: int4 cards + int8 notes,
#   float16 row scales, int8 token-shift vectors), the card-only arm C' of the second integer-state test (SM49)
#   (w8+s4hgx4c8hx8n) and the fork's default (w8+s4c8n, report only), show-time inputs, leak-free, 723 users of the 732
#   set; then the Small references at the fork's w8+s4c8n (report only). Published w8+s8 rows are reused (committed).
# Rule (sm129_rule.py decide, pre-registered): per arm and model, every head's 95% CI upper bound of
#   100 x (Tiny arm / Small - 1) <= +1.00% (two-head vs the show-time two-head cool-down for Small (SM118), Instant-Only
#   vs the longer Small Instant-Only model (SM113)); go = both models pass.
# CPU proofs first (sm129_check.py -> check-cpu-sm129.json, ALL OK required; under memguard.ps1).
# Breaks: sm129_block.py = quiet.json (scope all/gpu) + the daily 20:00-01:00 (Sunday from 16:00); re-read at every check.
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm129/artifacts/small-model/sm129/launch-sm129.sh
#   launch   WMI hidden (HANDOFF.md); log artifacts/small-model/sm129/gpu/launch-sm129.log
#   resume   the launch command again (finished users are skipped by the scorer)
#   stop     touch C:/Users/jonat/workspace/wt/sm129/artifacts/small-model/sm129/gpu/STOP (between tickets)
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm129
S=$WT/artifacts/small-model/sm129
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm129.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm129-dryrun.log
SC=$WT/scripts/small_model_heldout.py
RULE=$S/sm129_rule.py
OUTROWS=$G/sm129-rows.jsonl
TICKETS=$G/tickets.jsonl
STOPF=$G/STOP
W=C:/Users/jonat/workspace/wt
TRACK=E89-F65e-F77a
CK124=$W/sm124/artifacts/small-model/sm124-c1-e29st-dec2ep32k-t91s2-p2/expertium/$TRACK/master.pth
CK115=$W/sm115/artifacts/small-model/sm115-c1-e38st-st-dec2ep32k-t99-p2/expertium/$TRACK/master.pth
CK118=$W/sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2/expertium/$TRACK/master.pth
CK113=$W/sm113/artifacts/small-model/sm113-inst-e35st-st-dec2ep32k-t99-p2/expertium/$TRACK/master.pth
SHA124=c78b24d8ee187b58fd8c23c90476fb68a1aab88a137cc1f6a64c03530b7dcb3c
SHA115=c59d70495f7a81541d3a70329313354b5f0ec07c9fdd98ced58f849c5890fb0a
SHA118=b4fe1175d25a7e6abff70e87ac0af1f4d76db9622d54f7cb53196fbf35117eb6
SHA113=e4574e2aa8b2cfe6360dbd86f2a75b71d23ea81677b00ffcc9c3abce6b4cb736
T124=sm124-c1-e29st-dec2ep32k-t91s2; T115=sm115-c1-e38st-st-dec2ep32k-t99
T118=sm118-e26-dec2ep32k-t91s2; T113=sm113-inst-e35st-st-dec2ep32k-t99
S72=$W/sm72s/artifacts/small-model/sm72s
USERS732=$S72/eval-users-732-fit.json
MODE=neighbour
MIN_TINY=45; MIN_SMALL=20; MIN_SCORE_RETRY=20; TL=3600
NAME="Compact state for Tiny (SM129)"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START SM11_INPUTS
log() { echo "[sm129 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$STOPF" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }
free_mib() { awk '/MemFree/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 99999; }
block() { "$PY" -X utf8 "$S/sm129_block.py"; }
jline() { "$PY" -X utf8 -c "import json, sys; print(json.dumps(dict(zip(sys.argv[1::2], [int(v) if v.lstrip('-').isdigit() else v for v in sys.argv[2::2]]))))" "$@"; }

wait_free() {  # wait_free MINUTES WHAT: returns once MINUTES fit before the next break (sets FREE = minutes to it)
  local n=0
  while true; do
    stopped && return 1
    local b; b=$(block)
    case "$b" in
      NOW*) local s=${b#NOW }
            if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $2 would wait inside a break (${s}s left)"; FREE=100000; return 0; fi
            [ $((n % 6)) = 0 ] && log "$2: break, ${s}s left"; n=$((n + 1)); nap "$s";;
      NEXT*) FREE=${b#NEXT }
             [ "$FREE" -ge $(( $1 + 3 )) ] && return 0
             if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $2 ($1 min) would wait: $FREE min to the next break"; FREE=100000; return 0; fi
             [ $((n % 6)) = 0 ] && log "$2 needs $1 min, $FREE min before the next break; waiting"; n=$((n + 1)); nap $(( FREE * 60 + 60 ));;
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

q() {  # q MINUTES KIND PURPOSE -- CMD...   one ticket that must end before the next break
  local minutes=$1 kind=$2 purpose=$3 rc t0 mx base wpid; shift 4
  while true; do
    wait_free "$minutes" "$purpose" || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $NAME: $purpose (expected $minutes min)"; echo "    cmd: $*" >> "$LOG"; return 0; fi
    local maxwait=$(( FREE - minutes - 3 )); [ "$maxwait" -lt 1 ] && maxwait=1
    log "enqueue: $purpose (expected $minutes min, max wait $maxwait min, $FREE min to the next break)"
    t0=$(date +%s); base=$(gpu_used); rm -f "$G/.gpu-max"
    ( m=0; while true; do u=$(gpu_used); [ -n "$u" ] && [ "$u" -gt "$m" ] && { m=$u; echo "$m" > "$G/.gpu-max"; }; sleep 30; done ) &
    wpid=$!
    "$PY" "$QUEUE" run --owner sm129 --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm129/launch-sm129.sh" -- "$@"
    rc=$?; kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null; mx=$(cat "$G/.gpu-max" 2>/dev/null || echo 0)
    log "exit $rc: $purpose (nvidia-smi max $mx MiB, baseline $base MiB)"
    [ "$rc" = 75 ] && { log "queue wait would reach the next break; re-planning"; sleep 60; continue; }
    jline kind "$kind" purpose "$purpose" exit "$rc" minutes "$(( ($(date +%s) - t0 + 30) / 60 ))" nvidia_smi_max_mib "${mx:-0}" nvidia_smi_baseline_mib "${base:-0}" >> "$TICKETS"
    return $rc
  done
}

min_users() {  # min_users ROWS LABEL...: the smallest number of scored users over the labels
  F="$1" "$PY" -X utf8 -c "
import json, os, sys
from pathlib import Path
p = Path(os.environ['F'])
got = {}
if p.exists():
    for r in map(json.loads, open(p, encoding='utf-8')):
        if 'benchmark' in r:
            got.setdefault(r.get('label'), set()).add(r['user_id'])
print(min(len(got.get(l, ())) for l in sys.argv[1:]))" "${@:2}"; }

score() {  # score TAG ARCH CKPT MIN1 STRICT VARIANT...: 723 users of the 732 set, show-time, all variants in one pass per ticket
  local tag=$1 arch=$2 ck=$3 min1=$4 strict=$5 cap=0.8 pass have prev=-1 minutes; shift 5
  local only=() lbls=() v
  for v in "$@"; do only+=(--only "$tag/$v"); lbls+=("$tag/$v@e2s-query"); done
  for pass in 1 2 3 4 5; do
    have=$(min_users "$OUTROWS" "${lbls[@]}")
    [ "$have" -ge 723 ] && break
    if [ "$have" = "$prev" ]; then
      if [ "$cap" = 0.8 ]; then cap=0.92; log "no progress on $tag ($have of 723 users): next pass at cap 0.92 (memory only, same numerics)"
      elif [ "$strict" = 1 ]; then log "STOP: no progress on $tag ($have of 723 users)"; return 1
      else log "no progress at cap 0.92 on $tag ($have of 723 users); going on"; break; fi
    fi
    prev=$have
    minutes=$min1; [ "$pass" -gt 1 ] && minutes=$MIN_SCORE_RETRY
    ram_gate "scoring $tag"
    q $minutes score "scoring $tag at $* on e2s-query inputs, the 732 set minus 9 never-fit (723 users) (pass $pass, $have done, cap $cap)" -- \
      env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START HELDOUT_GPU_MEMORY_FRACTION=$cap "$PY" -X utf8 "$SC" --pool 732 --users-file "$USERS732" \
      --model "$tag=expertium:$arch:$ck" "${only[@]}" --inputs e2s-query --time-limit $TL --split-existing \
      --deleted-cards $MODE --out "$OUTROWS" >> "$G/score.out" 2>&1 || { log "STOP: scoring ticket failed ($tag)"; return 1; }
    [ "$DRYRUN" = 1 ] && break
  done
  [ "$DRYRUN" = 1 ] && return 0
  have=$(min_users "$OUTROWS" "${lbls[@]}")
  log "$tag: $have users scored for each of ${lbls[*]}"
  [ "$strict" != 1 ] || [ "$have" -ge 723 ] || { log "STOP: $tag scored on $have of 723 users"; return 1; }
  return 0
}

cpu_checks() {  # the CPU proofs (sm129_check.py) under the memory guard
  [ -e "$G/cpu-ok" ] && return 0
  [ "$DRYRUN" = 1 ] && { log "DRYRUN: $PY -X utf8 $S/sm129_check.py (under memguard.ps1)"; return 0; }
  wait_free 15 "CPU proofs" || return 1
  ram_gate "CPU proofs"
  rm -f "$G/memguard.off"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$S/memguard.ps1")" -Dir "$(cygpath -w "$G")" \
    -Match "sm129_check.py;--device cpu&sm124-c1-e29st-dec2ep32k-t91s2-p2" > /dev/null 2>&1 &
  local mg=$!
  CUDA_VISIBLE_DEVICES= "$PY" -X utf8 "$S/sm129_check.py" >> "$G/check.out" 2>&1; local rc=$?
  touch "$G/memguard.off"; sleep 12; kill "$mg" 2>/dev/null
  [ -e "$G/memguard-kill" ] && { log "STOP: memory guard killed a CPU proof process ($(tail -1 "$G/memguard-kill"))"; return 1; }
  [ "$rc" = 0 ] || { log "STOP: CPU proofs failed (see $G/check.out, $S/check-cpu-sm129.json)"; return 1; }
  log "CPU proofs ALL OK ($(grep -c '^PASS' "$G/check.out") PASS lines, $("$PY" -c "import json; print(json.load(open(r'$S/check-cpu-sm129.json'))['seconds'])") s)"
  touch "$G/cpu-ok"
}

preconditions() {
  for f in "$S/sm129_block.py" "$RULE" "$S/sm129_check.py" "$S/sm129_bytes.py" "$CK124" "$CK115" "$CK118" "$CK113" "$USERS732"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  local sha
  for pair in "$CK124 $SHA124" "$CK115 $SHA115" "$CK118 $SHA118" "$CK113 $SHA113"; do
    set -- $pair
    sha=$(sha256sum "$1" | cut -d' ' -f1); [ "$sha" = "$2" ] || { log "STOP: $1 changed (sha256 $sha)"; return 1; }
  done
  "$PY" -X utf8 "$RULE" selftest >> "$LOG" 2>&1 || { log "STOP: rule selftest failed"; return 1; }
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN; arms compact w8+s4c8nhx8, card-only w8+s4hgx4c8hx8n, fork w8+s4c8n)"
rc=0
wait_free 10 "preparation" || rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
[ $rc -eq 0 ] && { cpu_checks || rc=1; }
[ $rc -eq 0 ] && { score "$T124" w64-c1 "$CK124" $MIN_TINY 1 w8+s4c8nhx8 w8+s4hgx4c8hx8n w8+s4c8n || rc=1; }
[ $rc -eq 0 ] && { score "$T115" w64-c1 "$CK115" $MIN_TINY 1 w8+s4c8nhx8 w8+s4hgx4c8hx8n w8+s4c8n || rc=1; }
[ $rc -eq 0 ] && { score "$T118" w64 "$CK118" $MIN_SMALL 0 w8+s4c8n || log "Small fork-precision scoring (report only) incomplete; going on"; }
[ $rc -eq 0 ] && { score "$T113" w64 "$CK113" $MIN_SMALL 0 w8+s4c8n || log "Small fork-precision scoring (report only) incomplete; going on"; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: then $PY -X utf8 $RULE decide"; log "DRYRUN OK"; exit 0; fi
  out=$("$PY" -X utf8 "$RULE" decide 2>&1); r=$?
  log "rule decide exit $r: $(echo "$out" | tail -1 | cut -c1-600)"
  if [ "$r" = 0 ]; then
    log "done (verdict $("$PY" -X utf8 -c "import json; v=json.load(open(r'$S/sm129-decision.json'))['verdict']; print('compact ' + v['compact'] + ', card-only ' + v['card-only'])"))"
  else
    log "STOP: the rule did not decide (exit $r)"; rc=1
  fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
