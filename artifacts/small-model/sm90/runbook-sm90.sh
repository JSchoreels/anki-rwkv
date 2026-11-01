#!/usr/bin/env bash
# Codebook state compression sweep (SM90): GPU runbook (detached, break-aware, resumable).
#   1. GPU check (check_state_vq_score_gpu.py gpu vs the CPU references in check-ref/, every screen recipe; discrete
#      codec rule of AGENT.md; also reviews/s per recipe). STOP on a failure.
#   2. Screen: every ladder recipe + diagnostics + w8+s8 + w8+s4c8n on the 40 screen users (screen-users.json), leak-free.
#   3. sm90_rule.py screen -> finalists.txt (pre-registered pick, DESIGN.md).
#   4. Finalists on the 723 fit users of the 732 set, one label at a time (w8+s4c8n first), <= 110-min tickets.
#   5. sm90_rule.py final -> sm90-summary.{md,json}.
# Every ticket: queue claim gpu, purpose prefix "codebook state compression sweep (SM90): ", GPU memory cap 0.8 (the
# scorer's default; HELDOUT_GPU_MEMORY_FRACTION unset), never started unless it fits before the next quiet window
# (quiet.json + the daily 20:00-01:00 break). Re-running resumes (rows already written are skipped).
#   DRYRUN=1 bash runbook-sm90.sh   prints the plan (no tickets)
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm90
H=$WT/artifacts/small-model/sm90
G=$H/gpu
LOG=$H/runbook-sm90.log
CK=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/expertium/E89-F65e-F77a/master.pth
BOOKS=$H/books/codebooks-sm90.npz
BOOKS_SHA=$(cat $H/books/codebooks-sm90.sha256 2>/dev/null)
SC=$WT/scripts/small_model_heldout.py
USERS732=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json
SCREEN_USERS=$H/screen-users.json
SCREEN_ROWS=$G/screen-sm90.jsonl
FINAL_ROWS=$G/final-sm90.jsonl
PFX="codebook state compression sweep (SM90)"
SCREEN_LABELS=$(cat $H/screen-labels.txt)
DRYRUN=${DRYRUN:-0}
MIN_CHECK=${MIN_CHECK:-35}
MIN_SCREEN=${MIN_SCREEN:-75}
MIN_FINAL=${MIN_FINAL:-110}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS
mkdir -p "$G"
log() { echo "[sm90 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; [ "$DRYRUN" = 1 ] && echo "[sm90] $*"; }

free_minutes() {  # minutes to the next quiet window (quiet.json scope all/gpu, or the daily 20:00-01:00 break); 0 = inside one
  Q="$QUIET" "$PY" -X utf8 - <<'PYEOF'
import json, os
from datetime import datetime, timedelta
now = datetime.now()
try:
    wins = [(datetime.fromisoformat(w["start"]), datetime.fromisoformat(w["end"]))
            for w in json.loads(open(os.environ["Q"], encoding="utf-8").read()) if w.get("scope", "all") in ("all", "gpu")]
except Exception:
    print(0); raise SystemExit
day = now.replace(hour=0, minute=0, second=0, microsecond=0)
wins += [(day + timedelta(days=d, hours=20), day + timedelta(days=d + 1, hours=1)) for d in range(-1, 3)]
if any(s <= now < e for s, e in wins):
    print(0); raise SystemExit
print(int(min([(s - now).total_seconds() // 60 for s, e in wins if s > now] + [100000])))
PYEOF
}

MAXWAIT=0
wait_window() {  # wait_window MIN: block until a ticket of MIN minutes fits before the next quiet window
  local f
  while true; do
    f=$(free_minutes)
    if [ "$f" -ge $(( $1 + 5 )) ]; then MAXWAIT=$(( f - $1 - 3 )); return 0; fi
    [ "$DRYRUN" = 1 ] && { MAXWAIT=1; return 0; }
    log "waiting: $f min to the next quiet window, ticket needs $1"
    sleep 600
  done
}

q() {  # q MIN PURPOSE -- command...: one queue ticket; retried after a max-wait expiry (exit 75)
  local min=$1 purpose=$2 rc attempt
  shift 3
  for attempt in 1 2 3 4 5 6 7 8; do
    wait_window "$min"
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN ticket ($min min): $PFX: $purpose"; return 0; fi
    log "enqueue ($min min, max wait $MAXWAIT min): $purpose"
    "$PY" "$QUEUE" run --owner small-model --purpose "$PFX: $purpose" --claim gpu --expected-minutes "$min" \
      --max-wait-minutes "$MAXWAIT" --launched-by "detached runbook sm90/runbook-sm90.sh" -- "$@"
    rc=$?
    log "ticket exit $rc: $purpose"
    [ "$rc" = 75 ] || return $rc
  done
  return 75
}

label_users() {  # label_users ROWS LABEL USERS_FILE: users of USERS_FILE with an error-free row of LABEL
  "$PY" -X utf8 -c "
import json, sys
want = set(json.load(open(sys.argv[3]))['users'])
got = set()
try:
    for line in open(sys.argv[1]):
        r = json.loads(line)
        if r.get('label') == sys.argv[2] and 'error' not in r and r['user_id'] in want:
            got.add(r['user_id'])
except FileNotFoundError:
    pass
print(len(got))" "$1" "$2" "$3"
}

cpu_ok() {  # the rule scripts are seconds of CPU; still never inside a quiet window
  while [ "$(free_minutes)" -lt 1 ] && [ "$DRYRUN" != 1 ]; do sleep 600; done
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) DRYRUN=$DRYRUN"
if [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then log "STOP: scripts/expertium/rwkv not clean"; exit 1; fi
got=$("$PY" -X utf8 -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$BOOKS")
if [ "$got" != "$BOOKS_SHA" ]; then log "STOP: codebooks sha256 $got != pinned $BOOKS_SHA"; exit 1; fi

# 1. GPU check
if ! "$PY" -X utf8 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('passed') else 1)" "$H/check-sm90-gpu.json" 2>/dev/null; then
  q $MIN_CHECK "GPU check of the codec path vs the CPU references (every screen recipe, 2 users) and reviews/s" -- \
    "$PY" -X utf8 $WT/scripts/check_state_vq_score_gpu.py gpu --cases $H/check-cases.json --users 2 --timing-users 2 \
    --output $H/check-sm90-gpu.json --ref-dir $H/check-ref >> "$G/check.out" 2>&1 || { log "STOP: GPU check ticket failed (see gpu/check.out)"; exit 1; }
  if [ "$DRYRUN" != 1 ] && ! "$PY" -X utf8 -c "import json,sys; r=json.load(open(sys.argv[1])); print(r.get('passed'), r.get('fails')); sys.exit(0 if r.get('passed') else 1)" "$H/check-sm90-gpu.json" >> "$LOG" 2>&1; then
    log "STOP: GPU check failed (check-sm90-gpu.json)"; exit 1
  fi
fi
log "GPU check passed"

# 2. screen
ONLY=""; for l in $SCREEN_LABELS; do ONLY="$ONLY --only $l"; done
NS=$("$PY" -X utf8 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['users']))" "$SCREEN_USERS")
for pass in 1 2 3 4; do
  todo=0
  for l in $SCREEN_LABELS; do [ "$(label_users "$SCREEN_ROWS" "$l" "$SCREEN_USERS")" -lt "$NS" ] && todo=1; done
  [ "$todo" = 0 ] && break
  q $MIN_SCREEN "screen of the budget ladder on 40 users of the 732 set (pass $pass)" -- \
    "$PY" -X utf8 "$SC" --pool 732 --users-file "$SCREEN_USERS" --model "rel=expertium:w64:$CK" $ONLY --vq-codebooks "$BOOKS" \
    --time-limit $(( (MIN_SCREEN - 10) * 60 )) --split-existing --deleted-cards neighbour --out "$SCREEN_ROWS" >> "$G/screen.out" 2>&1 \
    || { log "STOP: screen ticket failed (see gpu/screen.out)"; exit 1; }
  [ "$DRYRUN" = 1 ] && break
done
cpu_ok
if [ "$DRYRUN" != 1 ]; then
  "$PY" -X utf8 $H/sm90_rule.py screen >> "$G/rule.out" 2>&1 || { log "STOP: screen read-out incomplete (gpu/rule.out)"; exit 1; }
  log "screen done: finalists $(tr '\n' ' ' < $H/finalists.txt)"
else
  [ -f $H/finalists.txt ] || printf 'rel/w8+s4c8n\nrel/w8+vqB3a\n' > $G/finalists-dryrun.txt
fi

# 4. finalists on the 732 set (723 fit users)
FIN=$H/finalists.txt; [ "$DRYRUN" = 1 ] && [ ! -f $FIN ] && FIN=$G/finalists-dryrun.txt
NF=$("$PY" -X utf8 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['users']))" "$USERS732")
for l in $(cat $FIN); do
  prev=-1
  for pass in 1 2 3 4 5 6; do
    have=$(label_users "$FINAL_ROWS" "$l" "$USERS732")
    [ "$have" -ge "$NF" ] && break
    if [ "$have" = "$prev" ] && [ "$DRYRUN" != 1 ]; then log "STOP: no progress on $l ($have of $NF users)"; exit 1; fi
    prev=$have
    q $MIN_FINAL "scoring $l on the 723 users of the 732 set (pass $pass, $have done)" -- \
      "$PY" -X utf8 "$SC" --pool 732 --users-file "$USERS732" --model "rel=expertium:w64:$CK" --only "$l" --vq-codebooks "$BOOKS" \
      --time-limit $(( (MIN_FINAL - 10) * 60 )) --split-existing --deleted-cards neighbour --out "$FINAL_ROWS" >> "$G/final.out" 2>&1 \
      || { log "STOP: scoring ticket failed for $l (see gpu/final.out)"; exit 1; }
    [ "$DRYRUN" = 1 ] && break
  done
  log "$l: $(label_users "$FINAL_ROWS" "$l" "$USERS732") of $NF users"
done
cpu_ok
[ "$DRYRUN" = 1 ] && { log "DRYRUN done"; exit 0; }
"$PY" -X utf8 $H/sm90_rule.py final >> "$G/rule.out" 2>&1
log "final read-out exit $?: artifacts/small-model/sm90/sm90-summary.md"
log "done"
