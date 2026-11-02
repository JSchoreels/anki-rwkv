#!/usr/bin/env bash
# Show-time interval retrain (SM91) step 2 (artifacts/small-model/sm91s2/DESIGN.md): the big model (fork RWKV, 2,762,884
# parameters) gets ONE added epoch from the big model (1 leak-free epoch) with the leak-free retrain (SM72) recipe
# (scripts/sm72_big_epoch.py: users 5000-10000, shuffle seed 1, fresh AdamW, warmup 200, 3.5e-4 -> 0 one_minus_sin,
# 2 strided lanes x 8,192 carried state, budgeted carry table 768 MiB, cap 0.8, RWKV_DELETED_CARDS=neighbour), the only
# change being the track option "show_time_intervals": true (the duration-leak scoring (SM31) e2s-query inputs, as the
# show-time interval retrain (SM91) step 1 small model). Then scoring on e2s-query and normal inputs, then the rule.
# Rule (sm91s2_rule.py, CPU): 722 users of the 732 set, new - the big model (1 leak-free epoch) both on e2s-query (the
#   reference rows are SM31's, reused): go to step 3 if one head gains >= 0.10 points with its 95% CI below zero.
# Steps: (after 01:00) label-window pre-flight + CPU proofs (scripts/sm91s2.py cpu-check -> check-cpu-sm91s2.json; STOP
#   unless ALL OK) -> wait for the show-time interval retrain (SM91) step 1 launcher to end (its log ends with "done" or
#   "STOP: ended") -> trainer smoke -> the epoch in segments (snapshot every 50 users; a snapshot stop on gpu/STOP, on a
#   quiet window or <= 2 min before one; the next segment resumes) -> scoring e2s-query then normal (723 users at cap
#   0.8, then the 3 long users at 0.92) -> rule.
# Breaks: quiet.json (scope all/gpu) plus the daily 20:00-01:00 break, re-read at every check; CPU steps too.
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm91s2/artifacts/small-model/sm91s2/launch-sm91s2.sh
#   launch   WMI hidden (artifacts/small-model/sm91s2/HANDOFF.md); log artifacts/small-model/sm91s2/gpu/launch-sm91s2.log
#   resume   the launch command again (finished steps are skipped; an interrupted epoch resumes from its snapshot)
#   stop     touch C:/Users/jonat/workspace/wt/sm91s2/artifacts/small-model/sm91s2/gpu/STOP (between tickets; a running
#            epoch gets a snapshot stop at the next user boundary, ~1 min; a hard kill loses <= 50 users, ~5 min)
# Do not edit scripts/ expertium/ rwkv/ of the sm91s2 worktree while it runs (SourceWatch stops the training).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm91s2
SM=$WT/artifacts/small-model
S=$SM/sm91s2
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm91s2.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm91s2-dryrun.log
EPOCH="$PY -X utf8 $WT/scripts/sm72_big_epoch.py"
SC=$WT/scripts/small_model_heldout.py
RULE=$S/sm91s2_rule.py
CHECK=$S/check-cpu-sm91s2.json
OUT=$SM/sm91s2-big-showtime
TRACK=big-showtime
CK=$OUT/expertium/$TRACK/master.pth
TAG=sm91s2-big-showtime
OUTROWS=$G/sm91s2-inputs.jsonl
BIGCK=C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72-big-leakfree-1ep/expertium/big-leakfree-1ep/master.pth  # read only
SM72OUT=C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72-big-leakfree-1ep                                         # read only
SM31ROWS=C:/Users/jonat/workspace/wt/sm31/artifacts/small-model/sm31/gpu/sm31-inputs.jsonl                                  # read only
BIGROWS=C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/gpu-big/bigleak-neighbour.jsonl                          # read only
S72=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s                                                             # read only
USERS732=$S72/eval-users-732-fit.json; ULONG=$S72/users-732-long.json
SM91LOG=C:/Users/jonat/workspace/wt/sm91/artifacts/small-model/sm91/gpu/launch-sm91.log                                       # read only
MODE=neighbour
LF="env RWKV_DELETED_CARDS=$MODE"
RATE=${RATE:-11000}   # training reviews/s for ticket lengths: SM72 step 1 11,458, better teacher (SM65) 11,339
MIN_SMOKE=10; MIN_E2S=50; MIN_NORMAL=30; MIN_RETRY=15; MIN_LONG=20   # SM31 big model e2s-query pass 1: 45 min on 723 users
TL732=2400; TLLONG=1380
WAIT_SM91_MAX=${WAIT_SM91_MAX:-360}   # minutes (from the first check after 01:00) to wait for SM91 step 1 before going ahead
NAME="show-time interval retrain (SM91) step 2"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START   # scorer cap 0.8; trainer cap 0.8 (options)
log() { echo "[sm91s2 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$G/STOP" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }

block() {  # -> "NOW <seconds to its end>" inside a quiet window (quiet.json scope all/gpu, or the daily break), else "NEXT <minutes>"
  Q="$QUIET" "$PY" -X utf8 - <<'PYEOF'
import json, os, re
from datetime import datetime, timedelta
now = datetime.now()
try:
    text = open(os.environ["Q"], encoding="utf-8").read()
except OSError as error:
    print(f"ERR quiet.json unreadable: {error}"); raise SystemExit
try:
    wins = [(w["start"], w["end"]) for w in json.loads(text) if w.get("scope", "all") in ("all", "gpu")]
except (ValueError, KeyError, TypeError):  # malformed JSON: take every start/end pair textually, scope assumed "all"
    wins = re.findall(r'"start"\s*:\s*"([^"]+)"[^{}]*?"end"\s*:\s*"([^"]+)"', text, re.S)
blocks = [(datetime.fromisoformat(s), datetime.fromisoformat(e)) for s, e in wins]
day = now.replace(hour=0, minute=0, second=0, microsecond=0)
blocks += [(day + timedelta(days=d, hours=20), day + timedelta(days=d + 1, hours=1)) for d in range(-1, 9)]  # daily break
blocks.sort()
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
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $NAME: $purpose (expected $minutes min)"; echo "    cmd: $*" >> "$LOG"; return 0; fi
    local maxwait=$(( FREE - minutes - 3 )); [ "$maxwait" -lt 1 ] && maxwait=1
    log "enqueue: $purpose (expected $minutes min, max wait $maxwait min, $FREE min to the next window)"
    "$PY" "$QUEUE" run --owner small-model --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm91s2/launch-sm91s2.sh" -- "$@"
    rc=$?; log "exit $rc: $purpose"
    [ "$rc" = 75 ] && { log "queue wait would reach the next window; re-planning"; sleep 60; continue; }
    return $rc
  done
}
cpu_step() { wait_free "$1" "$2"; }   # CPU work also stays out of the windows
label_users() { F="$1" LB="$2" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
p = Path(os.environ['F'])
print(len({r['user_id'] for r in map(json.loads, open(p, encoding='utf-8')) if r.get('label') == os.environ['LB'] and 'benchmark' in r}) if p.exists() else 0)"; }
done_in() { [ -f "$CK" ] && R="$OUT/expertium/$TRACK/result.json" "$PY" -X utf8 -c "
import json, os, sys
r = json.load(open(os.environ['R']))
sys.exit(0 if r.get('status') == 'complete' and not r.get('smoke_only') else 1)" 2>/dev/null; }
train_left() {  # reviews left in the epoch (protocol.json shrinks on every resume; the latest snapshot of this attempt)
  O="$OUT" T="$TRACK" "$PY" -X utf8 - <<'PYEOF'
import json, os
from pathlib import Path
out = Path(os.environ["O"])
d = json.loads((out / "decay.json").read_text())
total = d.get("training_reviews_after_trims", d["training_reviews"])
p = json.loads((out / "protocol.json").read_text())
done = (p["full_run"].get("lr_resume") or {}).get("reviews_offset", 0)
latest = out / "expertium" / os.environ["T"] / "snapshots" / "latest.json"
if latest.exists():
    done += json.loads(latest.read_text())["review_presentations"]
print(max(total - done, 0))
PYEOF
}
cpu_ok() { [ -f "$CHECK" ] && grep -q '"ok": true' "$CHECK" && ! grep -q '"ok": false' "$CHECK"; }

cpu_proofs() {  # after 01:00: label-window pre-flight (trims as SM72 step 1), then the CPU proofs; STOP unless ALL OK
  if [ "$DRYRUN" = 1 ]; then
    log "DRYRUN: label-window pre-flight: $LF $PY -X utf8 $WT/scripts/sm72_label_windows.py --decay $OUT --trim --out $G/check-label-windows-sm91s2.json"
    log "DRYRUN: CPU proofs: $PY -X utf8 $WT/scripts/sm91s2.py cpu-check -> $CHECK"; return 0; fi
  if cpu_ok; then log "CPU proofs: ALL OK already ($CHECK)"; return 0; fi
  [ -f "$CHECK" ] && { log "STOP: CPU proofs not ALL OK ($CHECK exists with a failure; fix, delete it, relaunch)"; return 1; }
  cpu_step 40 "CPU proofs" || return 1
  if ! grep -q '"label_windows"' "$OUT/decay.json"; then
    $LF "$PY" -X utf8 "$WT/scripts/sm72_label_windows.py" --decay "$OUT" --trim --out "$G/check-label-windows-sm91s2.json" >> "$G/check.out" 2>&1 \
      || { log "STOP: label-window pre-flight failed (see $G/check.out)"; return 1; }
    log "label-window pre-flight: $("$PY" -c "import json; d = json.load(open(r'$OUT/decay.json')); print(d['label_windows'], d.get('training_reviews_after_trims'))")"
  fi
  log "CPU proofs started (scripts/sm91s2.py cpu-check)"
  CUDA_VISIBLE_DEVICES= "$PY" -X utf8 "$WT/scripts/sm91s2.py" cpu-check --out "$CHECK" >> "$G/check.out" 2>&1
  local r=$?
  log "CPU proofs exit $r: $(grep '\[sm91s2-check\]' "$G/check.out" | tail -1 | cut -c1-600)"
  cpu_ok && [ "$r" = 0 ] || { log "STOP: CPU proofs not ALL OK ($CHECK, $G/check.out)"; return 1; }
  return 0
}

preconditions() {
  for f in "$BIGCK" "$SM72OUT/decay.json" "$SM72OUT/protocol.prepared.json" "$SM31ROWS" "$BIGROWS" "$USERS732" "$ULONG" \
           "$OUT/decay.json" "$OUT/protocol.json" "$RULE"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/ (SourceWatch refuses them)"; return 1; fi
  return 0
}

wait_sm91() {  # do not overtake the show-time interval retrain (SM91) step 1 launcher's remaining scoring tickets
  [ "$DRYRUN" = 1 ] && { log "DRYRUN: would wait for SM91 step 1 ($SM91LOG ends with done / STOP: ended)"; return 0; }
  [ -e "$G/sm91-waited" ] && return 0
  local start gone=0; start=$(date +%s)
  while true; do
    stopped && return 1
    if tail -1 "$SM91LOG" 2>/dev/null | grep -qE '\] (done|STOP: ended)'; then
      log "the show-time interval retrain (SM91) step 1 launcher ended: $(tail -1 "$SM91LOG")"; break; fi
    if powershell -NoProfile -c "if (Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | ? { \$_.CommandLine -match 'launch-sm91(-tonight)?\.sh' }) { exit 0 } else { exit 1 }"; then
      gone=0
    else
      gone=$((gone + 1))
      [ "$gone" -ge 3 ] && { log "the SM91 step 1 launcher process is gone (15 min) and its log does not end with done / STOP: $(tail -1 "$SM91LOG" | cut -c1-200); going ahead"; break; }
    fi
    if [ $(( ($(date +%s) - start) / 60 )) -ge "$WAIT_SM91_MAX" ]; then log "waited $WAIT_SM91_MAX min for SM91 step 1; going ahead"; break; fi
    sleep 300
  done
  touch "$G/sm91-waited"
  return 0
}

run_epoch() {  # sm72_big_epoch.py run in segments to the next window or the end; resumable from snapshots
  local rc qpid stop_sent b left need seg what
  while ! done_in; do
    stopped && return 1
    wait_free 30 "the epoch" || return 1
    left=$(train_left); need=$(( left / RATE / 60 + 12 ))
    seg=$(( FREE - 5 )); [ "$seg" -gt "$need" ] && seg=$need
    what="the big model (fork RWKV) one added epoch on show-time intervals from the big model (1 leak-free epoch) (users 5000-10000, carried state 2 lanes x 8,192, AdamW 3.5e-4 -> 0, deleted-card inputs removed), $(( left / 1000000 ))M of 370M reviews left"
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $NAME: $what (segment to the next window or the end, expected $seg min of ~$need)"
      echo "    cmd: $LF $EPOCH run --out $OUT" >> "$LOG"; return 0; fi
    rm -f "$OUT/control/STOP"
    log "enqueue: $what (expected $seg min of ~$need, $FREE min to the next window)"
    "$PY" "$QUEUE" run --owner small-model --purpose "$NAME: $what (segment to the next window or the end)" --claim gpu \
      --expected-minutes "$seg" --max-wait-minutes $(( FREE - 20 > 1 ? FREE - 20 : 1 )) --launched-by "detached launcher sm91s2/launch-sm91s2.sh" -- \
      $LF $EPOCH run --out "$OUT" >> "$G/train.out" 2>&1 &
    qpid=$!
    stop_sent=0
    while kill -0 "$qpid" 2>/dev/null; do
      b=$(block)
      if [ -e "$G/STOP" ] || [[ "$b" == NOW* ]] || { [[ "$b" == NEXT* ]] && [ "${b#NEXT }" -le 2 ]; }; then
        # re-requested every 30 s: `run` clears control/STOP when it starts, so a request made while queued is repeated
        [ "$stop_sent" = 0 ] && log "requesting a snapshot stop ($b$([ -e "$G/STOP" ] && echo ', gpu/STOP'))"
        $EPOCH stop --out "$OUT" > /dev/null 2>&1; stop_sent=1
      fi
      sleep 30
    done
    wait "$qpid"; rc=$?; log "exit $rc: epoch segment ($(train_left) reviews left)"
    [ "$rc" = 75 ] && { sleep 60; continue; }
    done_in && break
    [ "$stop_sent" = 0 ] && { log "STOP: epoch segment ended without completing and without a stop request (exit $rc); see $G/train.out"; return 1; }
  done
  [ "$DRYRUN" = 1 ] || log "epoch complete: $("$PY" -X utf8 -c "import json; r=json.load(open(r'$OUT/expertium/$TRACK/result.json')); print(f\"{r.get('review_presentations', 0):,} reviews (last attempt), {r.get('training_seconds', 0)/60:.1f} min training, {r.get('reviews_per_second', 0):,.0f} reviews/s, update offset end {r.get('update_offset_end')}\")")"
  return 0
}

score() {  # score INPUTS MIN_PASS1: 723 users at cap 0.8 (a pass without progress -> one pass at 0.92), then the 3 long users at 0.92
  local v=$1 m1=$2 pass have prev=-1 cap=0.8 minutes lb what
  lb="$TAG/w8+s8"; [ "$v" != normal ] && lb="$lb@$v"
  what="scoring the show-time big model (big model (1 leak-free epoch) + 1 show-time epoch) on $v inputs"
  for pass in 1 2 3 4; do
    have=$(label_users "$OUTROWS" "$lb")
    [ "$have" -ge 723 ] && break
    if [ "$have" = "$prev" ]; then
      if [ "$cap" = 0.8 ]; then cap=0.92; log "no progress on $lb ($have of 723 users): next pass at cap 0.92 (memory only, same numerics)"
      else log "STOP: no progress on $lb ($have of 723 users)"; return 1; fi
    fi
    prev=$have
    minutes=$m1; [ "$pass" -gt 1 ] && minutes=$MIN_RETRY
    q $minutes "$what, the 732 set minus 9 never-fit (723 users) (pass $pass, $have done, cap $cap)" -- \
      env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START HELDOUT_GPU_MEMORY_FRACTION=$cap "$PY" -X utf8 "$SC" --pool 732 --users-file "$USERS732" \
      --model "$TAG=fork::$CK" --only "$TAG/w8+s8" --inputs "$v" --time-limit $TL732 --split-existing \
      --deleted-cards $MODE --out "$OUTROWS" >> "$G/score.out" 2>&1 || { log "STOP: scoring ticket failed ($lb)"; return 1; }
    [ "$DRYRUN" = 1 ] && break
  done
  [ "$DRYRUN" = 1 ] || [ "$(label_users "$OUTROWS" "$lb")" -ge 723 ] || { log "STOP: $lb scored on $(label_users "$OUTROWS" "$lb") of 723 users"; return 1; }
  for pass in 1 2; do
    if [ "$DRYRUN" = 1 ] || [ "$(label_users "$OUTROWS" "$lb")" -lt 726 ]; then
      q $MIN_LONG "$what, the 3 long users of the 732 set (pass $pass, cap 0.92)" -- \
        env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START HELDOUT_GPU_MEMORY_FRACTION=0.92 "$PY" -X utf8 "$SC" --pool 732 --users-file "$ULONG" \
        --model "$TAG=fork::$CK" --only "$TAG/w8+s8" --inputs "$v" --time-limit $TLLONG --split-existing \
        --deleted-cards $MODE --out "$OUTROWS" >> "$G/score.out" 2>&1 \
        || { stopped && return 1; log "long users ticket failed ($lb); continuing with $(label_users "$OUTROWS" "$lb") users (the rule pool does not need them)"; break; }
      [ "$DRYRUN" = 1 ] && break
    fi
  done
  [ "$DRYRUN" = 1 ] || log "$lb: $(label_users "$OUTROWS" "$lb") users scored"
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1 | cut -c1-160) (DRYRUN=$DRYRUN; start $BIGCK; out $OUT)"
rc=0
[ $rc -eq 0 ] && { preconditions || rc=1; }
[ $rc -eq 0 ] && { cpu_proofs || rc=1; }
[ $rc -eq 0 ] && { wait_sm91 || rc=1; }
if [ $rc -eq 0 ] && [ ! -e "$G/smoke-ok" ] && ! done_in; then
  q $MIN_SMOKE "trainer smoke of the show-time big-model epoch (fork RWKV, show-time inputs, 2 users x 512 reviews, scratch copy)" -- \
    $LF $EPOCH smoke --out "$OUT" >> "$G/smoke.out" 2>&1 \
    || { log "STOP: smoke failed (see $G/smoke.out)"; rc=1; }
  [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ] && { touch "$G/smoke-ok"; log "smoke: $(grep '"smoke"' "$G/smoke.out" | tail -1 | cut -c1-200)"; }
fi
[ $rc -eq 0 ] && { run_epoch || rc=1; }
[ $rc -eq 0 ] && [ "$DRYRUN" != 1 ] && { [ -f "$CK" ] || { log "STOP: no checkpoint $CK"; rc=1; }; }
[ $rc -eq 0 ] && { score e2s-query $MIN_E2S || rc=1; }
[ $rc -eq 0 ] && { score normal $MIN_NORMAL || rc=1; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then
    log "DRYRUN: then $PY -X utf8 $RULE decide"
  else
    cpu_step 3 "rule" || rc=1
    if [ $rc -eq 0 ]; then out=$("$PY" -X utf8 "$RULE" decide 2>&1); r=$?
      log "rule exit $r: $(echo "$out" | tail -1)"; [ "$r" = 0 ] || rc=1; fi
  fi
fi
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
