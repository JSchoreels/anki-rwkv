#!/usr/bin/env bash
# Show-time scoring of the headline models (user 11-08): leak-free (RWKV_DELETED_CARDS=neighbour), w8+s8, --inputs e2s-query
# (what the Anki fork gives at a query). 732 set = 723 users (732 minus 9 never-fit) at cap 0.8 + the 3 long users at cap 0.92
# (the SM92 passes). Rows: gpu/showtime-rows.jsonl (732 set + long), gpu/showtime-rows-1500.jsonl (1,500 validation users).
# Reused, NOT rescored: the release (SM63, sm72e25), the show-time retrain (SM92) and the big model (1 leak-free epoch) 732-set rows.
# Order: 1 instant-only small model (SM57) stage 1 | 2 SM57 stage 4 | 3 the 1,500 pool: stage 1, stage 4, the release |
#        4 previous release (cool-down from epoch 20, SM72e20) | earlier release (cool-down from epoch 15, SM72) |
#        small model cool-down from epoch 15 (SM56) | cool-down from epoch 10 (SM56 dec10) | shipped big model (fork RWKV).
# Each ticket goes through the GPU queue (scripts/r01_queue.py run --claim gpu), cap 0.8 (0.92 only for the long users / a pass
# without progress), and starts only if its expected minutes end before the next break (showtime_block.py: quiet.json + daily
# 20:00-01:00, Sunday from 16:00). A model whose checkpoint is missing is skipped and logged.
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/anki-rwkv/artifacts/small-model/showtime/launch-showtime.sh
#   launch   WMI hidden (HANDOFF.md); log gpu/launch-showtime.log (last line "[showtime ...] done")
#   resume   the launch command again (steps whose rows are complete are skipped; a pass 2 picks up missing users)
#   stop     touch C:/Users/jonat/workspace/anki-rwkv/artifacts/small-model/showtime/gpu/STOP
# The scorer runs from the clean worktree wt/sm57 (it has the 1,500-pool option): do not edit scripts/ expertium/ rwkv/ there.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
H=C:/Users/jonat/workspace/anki-rwkv/artifacts/small-model/showtime
BLOCK="$PY -X utf8 $H/showtime_block.py"
G=$H/gpu
mkdir -p "$G"
LOG=$G/launch-showtime.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-showtime-dryrun.log
WT=C:/Users/jonat/workspace/wt/sm57
SC=$WT/scripts/small_model_heldout.py
S72=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model
USERS732=$S72/sm72s/eval-users-732-fit.json
ULONG=$S72/sm72s/users-732-long.json
USERS1500=$WT/artifacts/small-model/sm57/stage4/users-sm93-score.json
ROWS=$G/showtime-rows.jsonl
ROWS1500=$G/showtime-rows-1500.jsonl
TRACK=E89-F65e-F77a
MODE=neighbour
INP=e2s-query
MIN_SCORE=20; MIN_RETRY=15; MIN_LONG=20; MIN_POOL=30
TL732=1800; TLLONG=1380; TLPOOL=2400
NAME="show-time scoring of the headline models"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START
log() { echo "[showtime $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$G/STOP" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }
free_mib() { awk '/MemFree/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 99999; }
block() { $BLOCK; }

wait_free() {  # wait_free MINUTES WHAT: returns once MINUTES (+3) fit before the next break (FREE = minutes to it)
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
      *) log "break check failed: $b"; nap 600;;
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
q() {  # q MINUTES PURPOSE -- CMD...   one ticket that must end before the next break
  local minutes=$1 purpose=$2 rc; shift 3
  while true; do
    wait_free "$minutes" "$purpose" || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $NAME: $purpose (expected $minutes min)"; echo "    cmd: $*" >> "$LOG"; return 0; fi
    ram_gate "$purpose"
    local maxwait=$(( FREE - minutes - 3 )); [ "$maxwait" -lt 1 ] && maxwait=1
    log "enqueue: $purpose (expected $minutes min, max wait $maxwait min, $FREE min to the next break)"
    "$PY" "$QUEUE" run --owner small-model --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher showtime/launch-showtime.sh" -- "$@"
    rc=$?; log "exit $rc: $purpose"
    [ "$rc" = 75 ] && { log "queue wait would reach the next break; re-planning"; sleep 60; continue; }
    return $rc
  done
}
label_users() { F="$1" LB="$2" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
p = Path(os.environ['F'])
print(len({r['user_id'] for r in map(json.loads, open(p, encoding='utf-8')) if r.get('label') == os.environ['LB'] and 'benchmark' in r}) if p.exists() else 0)"; }

score() {  # score TAG SPEC WHAT ROWS POOL USERS_FILE EXPECT MIN1 TL CAP0 WHERE: passes at CAP0; a pass without progress -> cap 0.92; none at 0.92 -> go on
  local tag=$1 spec=$2 what=$3 rows=$4 pool=$5 uf=$6 expect=$7 min1=$8 tl=$9 cap=${10} where=${11} pass have prev=-1 minutes lb
  lb="$tag/w8+s8@$INP"
  for pass in 1 2 3 4 5; do
    have=$(label_users "$rows" "$lb")
    [ "$have" -ge "$expect" ] && break
    if [ "$have" = "$prev" ]; then
      if [ "$cap" != 0.92 ]; then cap=0.92; log "no progress on $lb ($have of $expect users): next pass at cap 0.92 (memory only, same numerics)"
      else log "no progress at cap 0.92 on $lb ($have of $expect users): the rest do not fit; going on"; break; fi
    fi
    prev=$have
    minutes=$min1; [ "$pass" -gt 1 ] && minutes=$MIN_RETRY
    q $minutes "scoring $what on show-time inputs, $where (pass $pass, $have done, cap $cap)" -- \
      env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START HELDOUT_GPU_MEMORY_FRACTION=$cap "$PY" -X utf8 "$SC" --pool $pool --users-file "$uf" \
      --model "$tag=$spec" --only "$tag/w8+s8" --inputs $INP --time-limit $tl --split-existing \
      --deleted-cards $MODE --out "$rows" >> "$G/score.out" 2>&1 || { log "FAILED: scoring ticket failed ($lb, $where)"; return 1; }
    [ "$DRYRUN" = 1 ] && break
  done
  [ "$DRYRUN" = 1 ] || log "$lb: $(label_users "$rows" "$lb") users scored in $(basename "$rows") ($where)"
  return 0
}
score732() {  # score732 TAG SPEC WHAT: 723 users at 0.8, then the 3 long users at 0.92
  score "$1" "$2" "$3" "$ROWS" 732 "$USERS732" 723 $MIN_SCORE $TL732 0.8 "the 732 set minus 9 never-fit (723 users)" || return 1
  score "$1" "$2" "$3" "$ROWS" 732 "$ULONG" 726 $MIN_LONG $TLLONG 0.92 "the 3 long users of the 732 set" || log "long users: not all scored for $1 (the 722-user pool does not need them)"
  return 0
}
score1500() {  # 771 users of the 1,500 pool (765 new + 6 check users re-scored); the rest do not fit
  score "$1" "$2" "$3" "$ROWS1500" 1500 "$USERS1500" 771 $MIN_POOL $TLPOOL 0.8 "the 1,500 validation users outside the 732 set + 6 check users"
}

# ---- models: tag | checkpoint | plain name (ID in parentheses)
SM57=C:/Users/jonat/workspace/wt/sm57/artifacts/small-model
CK_S1=$SM57/sm57-inst-e26-dec-1ep32k-muon/expertium/$TRACK/master.pth
CK_S4=$SM57/sm57s4-inst-e30-dec-1ep32k-muon/expertium/$TRACK/master.pth
CK_REL=$S72/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/expertium/$TRACK/master.pth
CK_E20=$S72/scale-w64kd-final-sc-lanes-leakfree-dec20-1ep32k-muon/expertium/$TRACK/master.pth
CK_E15=$S72/scale-w64kd-final-sc-lanes-leakfree-dec15-1ep32k-muon/expertium/$TRACK/master.pth
CK_SM56E15=C:/Users/jonat/workspace/wt/sm56/artifacts/small-model/scale-w64kd-final-sc-lanes-dec15-1ep32k-muon/expertium/$TRACK/master.pth
CK_SM56E10=C:/Users/jonat/workspace/wt/sm56/artifacts/small-model/scale-w64kd-final-sc-lanes-dec10-1ep32k-muon/expertium/$TRACK/master.pth
CK_BIG=C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth
T_S1=sm57-inst-dec1ep32k
T_S4=sm57s4-inst-e30-dec1ep32k
T_REL=s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon
T_E20=s-w64kd-final-sc-lanes-leakfree-dec20-1ep32k-muon
T_E15=s-w64kd-final-sc-lanes-leakfree-dec15-1ep32k-muon
T_SM56E15=sm56dec15
T_SM56E10=sm56dec10
T_BIG=fork-shipped-big
N_S1="the instant-only small model (SM57) stage 1"
N_S4="the instant-only small model (SM57) stage 4"
N_REL="the release (SM63)"
N_E20="the previous release, cool-down from epoch 20 (SM72e20)"
N_E15="the earlier release, cool-down from epoch 15 (SM72)"
N_SM56E15="the small model, cool-down from epoch 15 (SM56)"
N_SM56E10="the previous release, cool-down from epoch 10 (SM56 dec10)"
N_BIG="the shipped big model (fork RWKV)"
specof() { case "$1" in "$T_BIG") echo "fork::$2";; *) echo "expertium:w64:$2";; esac; }
have_ck() { [ -f "$1" ] || { log "MISSING checkpoint (model skipped): $2 -> $1"; return 1; }; [ "$DRYRUN" = 1 ] && log "checkpoint OK: $2 -> $1"; return 0; }

log "started (DRYRUN=$DRYRUN): $NAME; scorer tree $(git -C "$WT" log --oneline -1)"
FAILED=""; SKIPPED=""
for f in "$QUEUE" "$SC" "$USERS732" "$ULONG" "$USERS1500" "$H/showtime_block.py"; do
  [ -f "$f" ] || { log "STOP: missing $f"; exit 1; }
done
if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
  log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/ of $WT (SourceWatch refuses them)"; exit 1; fi

run732() {  # run732 TAG CK NAME
  have_ck "$2" "$3" || { SKIPPED="$SKIPPED $3;"; return 0; }
  stopped && return 1
  score732 "$1" "$(specof "$1" "$2")" "$3" || FAILED="$FAILED $3 (732 set);"
  return 0
}
run1500() {  # run1500 TAG CK NAME
  have_ck "$2" "$3" || { SKIPPED="$SKIPPED $3 (1,500);"; return 0; }
  stopped && return 1
  score1500 "$1" "$(specof "$1" "$2")" "$3" || FAILED="$FAILED $3 (1,500);"
  return 0
}
{ run732 $T_S1 "$CK_S1" "$N_S1" &&
  run732 $T_S4 "$CK_S4" "$N_S4" &&
  run1500 $T_S1 "$CK_S1" "$N_S1" &&
  run1500 $T_S4 "$CK_S4" "$N_S4" &&
  run1500 $T_REL "$CK_REL" "$N_REL" &&
  run732 $T_E20 "$CK_E20" "$N_E20" &&
  run732 $T_E15 "$CK_E15" "$N_E15" &&
  run732 $T_SM56E15 "$CK_SM56E15" "$N_SM56E15" &&
  run732 $T_SM56E10 "$CK_SM56E10" "$N_SM56E10" &&
  run732 $T_BIG "$CK_BIG" "$N_BIG"; } || { log "STOP: stopped by the STOP file"; exit 1; }
[ -n "$SKIPPED" ] && log "skipped (checkpoint missing):$SKIPPED"
if [ -n "$FAILED" ]; then log "STOP: finished with failed steps:$FAILED"; exit 1; fi
if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done"; fi
exit 0
