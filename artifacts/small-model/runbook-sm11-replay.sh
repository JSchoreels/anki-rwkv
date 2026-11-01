#!/usr/bin/env bash
# SM11 GPU replay ticket(s): the fork's full-history targets for the 5,000 gate training users (docs/SMALL_MODEL.md, SM11;
# HANDOFF-SM11.md). NOT launched by the agent that wrote it. Launch (detached, from any directory):
#   nohup bash C:/Users/jonat/workspace/wt/sm11/artifacts/small-model/runbook-sm11-replay.sh > \
#     C:/Users/jonat/workspace/wt/sm11/artifacts/small-model/runbook-sm11-replay.log 2>&1 & disown
# DRYRUN=1 stubs the queue and the CPU check (echo only), validates the arguments through `status`, and logs "DRYRUN OK".
# Each queue ticket runs the export with --time-limit (no new user after LIMIT s) so the GPU is free between tickets;
# the export resumes per user, so the loop simply enqueues the next ticket until `status` reports all users complete.
# Stop between tickets: touch $OUT/STOP (the loop checks it before each enqueue; the queue also holds a ticket that
# would run into a GPU quiet window, and a cancelled ticket loses at most the user in progress).
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=${WT:-C:/Users/jonat/workspace/wt/sm11}
SM=$WT/artifacts/small-model
EXPORT=$WT/scripts/sm11_replay_export.py
CHECK=$WT/scripts/check_sm11_replay_targets.py
RUN=${RUN:-$WT/artifacts/accuracy/experiments/scale-w64kd-gate/run/run.json}
OUT=${OUT:-C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/fork-fp32-gate5k}
LIMIT=${LIMIT:-5100}            # seconds after which a ticket starts no new user
MINUTES=${MINUTES:-95}          # LIMIT + the longest user (3.9M reviews, 13 passes, ~7 min) + startup
MAX_TICKETS=${MAX_TICKETS:-8}   # ~386M replayed reviews at ~19k/s = ~5.6 h = 4-5 tickets
export DRYRUN=${DRYRUN:-0}
log() { echo "[sm11 $(date '+%m-%d %H:%M:%S')] $*"; }
q() {  # q MINUTES PURPOSE -- CMD...
  local minutes=$1 purpose=$2; shift 3
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $purpose (expected $minutes min)"; echo "    cmd: $*"; return 0; fi
  log "enqueue: $purpose (expected $minutes min)"
  "$PY" "$QUEUE" run --owner small-model --purpose "small-model: $purpose" --claim gpu --expected-minutes "$minutes" \
    --launched-by "detached runbook runbook-sm11-replay.sh" -- "$@"
  local rc=$?; log "exit $rc: $purpose"; return $rc
}
status() { "$PY" -X utf8 "$EXPORT" status --out "$OUT" --run "$RUN" "$@"; }
field() { "$PY" -X utf8 -c "import json,sys; print(json.loads(sys.argv[1])[sys.argv[2]])" "$ST" "$1" 2>/dev/null; }

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN) out $OUT"
if [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN warning (would stop): uncommitted source changes"; else log "uncommitted source changes; stop"; exit 2; fi
fi
free_gb=$(df -BG --output=avail "$(dirname "$OUT")" 2>/dev/null | tail -1 | tr -dc 0-9)
[ -z "$free_gb" ] && free_gb=$(df -BG --output=avail C:/ | tail -1 | tr -dc 0-9)
log "free disk: ${free_gb} GB (need ~3.7 GB)"
[ "${free_gb:-0}" -ge 10 ] || { log "less than 10 GB free; stop"; exit 2; }
if [ "$DRYRUN" = 1 ]; then
  log "DRYRUN: would run the CPU check $CHECK (~100 s, must print ALL OK)"
else
  "$PY" -X utf8 "$CHECK" --output "$SM/check-sm11-replay-targets-runbook.json" > "$SM/check-sm11-replay-targets-runbook.txt" 2>&1
  grep -q "ALL OK" "$SM/check-sm11-replay-targets-runbook.txt" || { log "SM11 check did not print ALL OK; stop"; exit 2; }
  log "SM11 check passed"
fi
previous=-1
for ticket in $(seq 1 "$MAX_TICKETS"); do
  ST=$(status); code=$?
  if [ "$code" = 0 ]; then log "export complete"; break; fi
  done_users=$(field complete_users); done_users=${done_users:-0}
  log "status: ${done_users} users complete, $(field reviews_done) of $(field reviews_total) reviews"
  if [ "$DRYRUN" != 1 ] && [ "$done_users" = "$previous" ]; then log "no progress in the last ticket; stop (see $OUT/progress.jsonl)"; exit 1; fi
  previous=$done_users
  [ -e "$OUT/STOP" ] && { log "STOP file present; stop"; exit 0; }
  q "$MINUTES" "SM11 fork full-history replay targets, gate 5k users (ticket $ticket, resumable)" -- \
    "$PY" -X utf8 "$EXPORT" export --out "$OUT" --run "$RUN" --device cuda --time-limit "$LIMIT"
  rc=$?
  if [ "$rc" != 0 ] && [ "$rc" != 3 ]; then log "export ticket exit $rc; stop"; exit 1; fi
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: status validated the arguments (exit $code; 3 = not complete yet)"; break; fi
done
if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; exit 0; fi
status || { log "export incomplete after $MAX_TICKETS tickets; rerun this runbook (it resumes)"; exit 1; }
"$PY" -X utf8 "$EXPORT" verify --out "$OUT" --run "$RUN" --sample 8 || { log "verify failed (see $OUT/verify.json)"; exit 1; }
log "done: $(cat "$OUT/status.json" | tr -d '\n' | cut -c1-400)"
