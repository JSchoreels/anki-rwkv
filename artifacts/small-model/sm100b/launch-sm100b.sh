#!/usr/bin/env bash
# SM100b (the show-time error sensitivity test (SM100), follow-up: model comparison without timeout reviews):
#   the show-time retrain of the release (SM92) (read only checkpoint, sha256 pinned below) scored on the 723 users of the 732 set
#   with e2s-query inputs and per-review dumps, ONE GPU queue ticket (sm100b_score.py, ~15 min), then the pre-registered rule
#   (sm100b_rule.py decide, CPU): SM96 minus SM92, instant, e2s-query, 722 users, (a) all (b) without timeouts (c) timeouts only.
# Starts no ticket before 2026-11-10 01:00 (sm100b_block.py hard hold + the daily 20:00-01:00 break + quiet.json).
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm100/artifacts/small-model/sm100b/launch-sm100b.sh
#   launch   WMI hidden (HANDOFF.md); log artifacts/small-model/sm100b/gpu/launch-sm100b.log
#   stop     touch C:/Users/jonat/workspace/wt/sm100/artifacts/small-model/sm100b/gpu/STOP
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm100
S=$WT/artifacts/small-model/sm100b
S100=$WT/artifacts/small-model/sm100
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm100b.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm100b-dryrun.log
CKPT=C:/Users/jonat/workspace/wt/sm92/artifacts/small-model/sm92-e26-dec-1ep32k-muon/expertium/E89-F65e-F77a/master.pth
CKPT_SHA=2091f8a879a56cf96a3df3b820fdacfa64c6467e8502a3a5eb17436524100d7f
TICKETS=$G/tickets.jsonl
STOPF=$G/STOP
MIN_SCORE=${MIN_SCORE:-25}    # 1 pass ~12-15 min (SM96: 12 min per 723-user pass) + retry
NAME="the show-time error sensitivity test (SM100), follow-up"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START
log() { echo "[sm100b $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$STOPF" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }
free_mib() { awk '/MemFree/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 99999; }
block() { "$PY" -X utf8 "$S/sm100b_block.py"; }
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

ram_gate() {  # hold while < 3 GB RAM is free, at most 30 min, then go on
  local i=0
  while [ "$DRYRUN" != 1 ] && [ "$(free_mib)" -lt 3072 ]; do
    [ $i -ge 6 ] && { log "$1: still $(free_mib) MiB RAM free after 30 min; going on"; return 0; }
    [ $i = 0 ] && log "$1: $(free_mib) MiB RAM free (< 3 GB), holding"; i=$((i + 1)); sleep 300
  done
  return 0
}

q() {  # q MINUTES KIND PURPOSE -- CMD...   one ticket that must end before the next break; timed in $TICKETS
  local minutes=$1 kind=$2 purpose=$3 rc t0; shift 4
  while true; do
    wait_free "$minutes" "$purpose" || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $NAME: $purpose (expected $minutes min)"; echo "    cmd: $*" >> "$LOG"; return 0; fi
    local maxwait=$(( FREE - minutes - 3 )); [ "$maxwait" -lt 1 ] && maxwait=1
    log "enqueue: $purpose (expected $minutes min, max wait $maxwait min, $FREE min to the next break, $(free_mib) MiB RAM free)"
    t0=$(date +%s)
    "$PY" "$QUEUE" run --owner sm100b --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm100b/launch-sm100b.sh" -- "$@"
    rc=$?; log "exit $rc: $purpose"
    [ "$rc" = 75 ] && { log "queue wait would reach the next break; re-planning"; sleep 60; continue; }
    jline kind "$kind" purpose "$purpose" exit "$rc" minutes "$(( ($(date +%s) - t0 + 30) / 60 ))" >> "$TICKETS"
    return $rc
  done
}

preconditions() {
  for f in "$S100/check-cpu-sm100.json" "$S/sm100b_rule.py" "$S/sm100b_score.py" "$S/sm100b_block.py" "$CKPT"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  grep -q '"all_ok": true' "$S100/check-cpu-sm100.json" || { log "STOP: the CPU proofs are not ALL OK"; return 1; }
  [ "$(sha256sum "$CKPT" | cut -c1-64)" = "$CKPT_SHA" ] || { log "STOP: checkpoint sha256 differs from $CKPT_SHA"; return 1; }
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv artifacts/small-model/sm100/*.py artifacts/small-model/sm100b/*.py)" ]; then
    log "STOP: uncommitted changes in scripts/ expertium/ rwkv/ or the sm100 scripts"; return 1; fi
  "$PY" -X utf8 "$S/sm100b_rule.py" selftest >> "$LOG" 2>&1 || { log "STOP: rule selftest failed"; return 1; }
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN; SM92 checkpoint on 723 users: e2s-query, dumps on)"
rc=0
preconditions || rc=1
if [ $rc -eq 0 ]; then
  ram_gate "scoring"
  q $MIN_SCORE score "scoring the show-time retrain of the release (SM92) on the 723 users of the 732 set with show-time inputs (1 pass, per-review dumps, cap 0.8, 0.92 memory-only retry)" -- \
    "$PY" -X utf8 "$S/sm100b_score.py" $([ "$DRYRUN" = 1 ] && echo --dry-run) >> "$G/score.out" 2>&1
  r=$?
  if [ "$DRYRUN" = 1 ]; then "$PY" -X utf8 "$S/sm100b_score.py" --dry-run >> "$LOG" 2>&1; fi
  [ $r = 3 ] && { log "some variant is short of 723 users (see $G/score.out); the rule decides whether its 722 users are there"; r=0; }
  [ $r = 0 ] || { log "STOP: scoring ticket failed (exit $r; see $G/score.out)"; rc=1; }
fi
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: then $PY -X utf8 $S/sm100b_rule.py decide"
  else
    out=$("$PY" -X utf8 "$S/sm100b_rule.py" decide 2>&1); r=$?
    log "rule exit $r: $(echo "$out" | tail -1 | cut -c1-900)"
    [ $r = 0 ] || rc=1
  fi
fi
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done (verdict $("$PY" -X utf8 -c "import json; print(json.load(open(r'$S/sm100b-decision.json')).get('verdict'))" 2>/dev/null))"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
