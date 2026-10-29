#!/usr/bin/env bash
# Smaller model (SM73) stage 1 (design: anki-rwkv/artifacts/small-model/sm73/DESIGN.md): the released leak-free small
#   model (SM72s, cool-down from epoch 20; read only from wt/sm72s) scored with a smaller stored card/note state on the
#   732 set (723 fit users), leak-free scoring exactly as SM72s's readout (--split-existing --deleted-cards neighbour,
#   no RWKV_DELETED_CARDS in the scorer, cap 0.8), paired against its own int8-state rows (TAG/w8+s8, wt/sm72s).
#   Bar (user 10-29): <= +0.50% curve AND <= +0.40% instant vs the big model (1 leak-free epoch).
#   Order (stop rule): w8+s4 first (int4 card and note state, the fork's existing int4 format, 3,018 B per card + note,
#   -34%); if it passes, stop there (the smallest format already fits). If it fails, w8+s4c8n (int4 card, int8 note,
#   3,530 B, -22%). No training, no snapshots: each ticket is ~12 min and resumable per user (the scorer skips scored
#   users; a second pass fills gaps).
# Breaks: quiet.json ONLY (scope all/gpu), re-read at every check (sleeps <= 10 min); a ticket starts only if it ends
#   before the next window (CPU steps too).
#
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm73/artifacts/small-model/sm73/launch-sm73.sh
#   launch   WMI hidden (HANDOFF-sm73.md); log artifacts/small-model/sm73/launch-sm73.log
#   resume   the launch command again (scored users and decided formats are skipped)
#   stop     touch C:/Users/jonat/workspace/wt/sm73/artifacts/small-model/sm73/STOP (checked before every ticket)
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm73
S=$WT/artifacts/small-model/sm73
G=$S/gpu
LOG=$S/launch-sm73.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$S/launch-sm73-dryrun.log
SC=$WT/scripts/small_model_heldout.py
RULE=$S/sm73_rule.py
SM72S=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model                                                  # read only
CK=$SM72S/scale-w64kd-final-sc-lanes-leakfree-dec20-1ep32k-muon/expertium/E89-F65e-F77a/master.pth
REFJ=$SM72S/sm72s/gpu/small-lf-neighbour.jsonl
BIGJ=C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/gpu-big/bigleak-neighbour.jsonl          # read only
USERS732=$SM72S/sm72s/eval-users-732-fit.json   # 732 pool minus 9 never-fit = 723
TAG=s-w64kd-final-sc-lanes-leakfree-dec20-1ep32k-muon
OUT=$G/sm73-formats-732.jsonl
FORMATS="w8+s4 w8+s4c8n"
MIN1=${MIN1:-15}; MIN2=8    # SM72s's int8 pass on the 723 users: 11.7 min
MODE=neighbour
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS   # scorer cap 0.8, no transform in the scorer
log() { echo "[sm73 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
mkdir -p "$G"

block() {  # -> "NOW <seconds to its end>" inside a scope all/gpu quiet.json window, else "NEXT <minutes to the next start>"
  Q="$QUIET" "$PY" -X utf8 - <<'PYEOF'
import json, os, re
from datetime import datetime
now = datetime.now()
try:
    text = open(os.environ["Q"], encoding="utf-8").read()
except OSError as error:
    print(f"ERR quiet.json unreadable: {error}"); raise SystemExit
try:
    wins = [(w["start"], w["end"]) for w in json.loads(text) if w.get("scope", "all") in ("all", "gpu")]
except (ValueError, KeyError, TypeError):  # malformed JSON: take every start/end pair textually, scope assumed "all"
    wins = re.findall(r'"start"\s*:\s*"([^"]+)"[^{}]*?"end"\s*:\s*"([^"]+)"', text, re.S)
blocks = sorted((datetime.fromisoformat(s), datetime.fromisoformat(e)) for s, e in wins)
merged = []
for s, e in blocks:
    if merged and s <= merged[-1][1]:
        merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    else:
        merged.append((s, e))
for s, e in merged:
    if s <= now < e:
        print(f"NOW {int((e - now).total_seconds()) + 60}"); raise SystemExit
    if s > now:
        print(f"NEXT {int((s - now).total_seconds() // 60)}"); raise SystemExit
print("NEXT 100000")
PYEOF
}

stopped() { [ -e "$S/STOP" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }   # re-read quiet.json every <= 10 min

wait_free() {  # wait_free MINUTES WHAT: returns once MINUTES fit before the next window (sets FREE = minutes to it)
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
             if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $2 ($1 min) would wait: $FREE min to the next window"; FREE=100000; return 0; fi
             [ $((n % 6)) = 0 ] && log "$2 needs $1 min, $FREE min before the next window; waiting"; n=$((n + 1)); nap $(( FREE * 60 + 60 ));;
      *) log "window check failed: $b"; nap 600;;
    esac
  done
}

q() {  # q MINUTES PURPOSE -- CMD...   one ticket that must end before the next window
  local minutes=$1 purpose=$2 rc; shift 3
  while true; do
    wait_free "$minutes" "$purpose" || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $purpose (expected $minutes min)"; echo "    cmd: $*" >> "$LOG"; return 0; fi
    local maxwait=$(( FREE - minutes - 3 )); [ "$maxwait" -lt 1 ] && maxwait=1
    log "enqueue: $purpose (expected $minutes min, max wait $maxwait min, $FREE min to the next window)"
    "$PY" "$QUEUE" run --owner small-model --purpose "small-model: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm73/launch-sm73.sh" -- "$@"
    rc=$?; log "exit $rc: $purpose"
    [ "$rc" = 75 ] && { log "queue wait would reach the next window; re-planning"; sleep 60; continue; }
    return $rc
  done
}
cpu_step() { wait_free "$1" "$2"; }   # CPU work also stays out of the windows
need() { if [ "$DRYRUN" = 1 ]; then log "DRYRUN warning (would stop): $1"; return 0; fi; log "STOP: $1"; return 1; }
label_users() { F="$1" LB="$2" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
p = Path(os.environ['F'])
print(len({r['user_id'] for r in map(json.loads, open(p, encoding='utf-8')) if r.get('label') == os.environ['LB'] and 'benchmark_existing' in r}) if p.exists() else 0)"; }
decided() { [ -f "$S/sm73-formats.jsonl" ] && grep -q "\"label\": \"$TAG/$1\"" "$S/sm73-formats.jsonl"; }
verdict() { grep "\"label\": \"$TAG/$1\"" "$S/sm73-formats.jsonl" | tail -1 | grep -q '"pass": true'; }

preconditions() {
  for f in "$CK" "$REFJ" "$BIGJ" "$USERS732" "$SC" "$RULE"; do
    [ -f "$f" ] || { need "missing $f" || return 2; }
  done
  local n; n=$(label_users "$REFJ" "$TAG/w8+s8")
  [ "$n" -ge 723 ] || { need "reference int8 rows: $n users (want >= 723)" || return 2; }
  if [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    need "uncommitted source changes in scripts/ expertium/ rwkv/" || return 2; fi
  # the scorer reads the variant names this stage uses
  for v in $FORMATS; do
    WTR="$WT" V="$v" "$PY" -X utf8 -c "
import os, sys
sys.path.insert(0, os.environ['WTR'])
os.environ.setdefault('RWKV_NO_JIT', '1')
from scripts.small_model_heldout import parse_variant
print('[sm73-pre] variant', os.environ['V'], '->', parse_variant(os.environ['V']))" >> "$LOG" 2>&1 || { need "scorer does not parse $v" || return 2; }
  done
  log "inputs OK: checkpoint $(sha256sum "$CK" | cut -c1-12) (SM72s cool-down from epoch 20), int8 reference rows $n users"
  return 0
}

score_format() {  # score_format VARIANT: 723 users of the 732 set, 2 passes, then the decision
  local v=$1 pass
  for pass in 1 2; do
    if [ "$DRYRUN" = 1 ] || [ "$(label_users "$OUT" "$TAG/$v")" -lt 723 ]; then
      q $([ $pass = 1 ] && echo $MIN1 || echo $MIN2) "smaller model (SM73) stage 1: leak-free small model (SM72s, cool-down from epoch 20) at $v state on the 732 set (723 users, pass $pass)" -- \
        env -u RWKV_DELETED_CARDS "$PY" -X utf8 "$SC" --pool 732 --users-file "$USERS732" --model "$TAG=expertium:w64:$CK" --only "$TAG/$v" \
        --time-limit 2400 --split-existing --deleted-cards $MODE --out "$OUT" >> "$G/score.out" 2>&1 || { log "STOP: scoring ticket failed ($v)"; return 1; }
      [ "$DRYRUN" = 1 ] && break
    fi
  done
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: would run: $PY -X utf8 $RULE decide $TAG/$v"; return 0; fi
  local got; got=$(label_users "$OUT" "$TAG/$v")
  [ "$got" -ge 723 ] || { log "STOP: $v scored on $got of 723 users after 2 passes"; return 1; }
  cpu_step 2 "decide $v" || return 1
  local out rc; out=$("$PY" -X utf8 "$RULE" decide "$TAG/$v"); rc=$?
  log "$out"
  [ "$rc" = 2 ] && { log "STOP: decision for $v lacks users"; return 1; }
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1 | cut -c1-120) (DRYRUN=$DRYRUN)"
rc=0
preconditions >> "$LOG" 2>&1 || rc=$?
if [ $rc -eq 0 ]; then
  for v in $FORMATS; do
    if [ "$DRYRUN" != 1 ] && decided "$v"; then log "$v already decided: $(verdict "$v" && echo PASS || echo FAIL)"
    else score_format "$v" || { rc=1; break; }
    fi
    if [ "$DRYRUN" != 1 ] && verdict "$v"; then log "stop rule: $v passes the bar; larger formats not scored"; break; fi
    [ "$DRYRUN" = 1 ] && log "DRYRUN: next format only if $v fails"
  done
fi
if [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ]; then
  "$PY" -X utf8 "$RULE" summary >> "$LOG" 2>&1 || log "summary failed"
fi
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
