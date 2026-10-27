#!/usr/bin/env bash
# On every queue change (scripts/r01_queue.py touches artifacts/coordination/queue/changed when a ticket is registered,
# starts, finishes or is removed; checked every 10 s, then DEBOUNCE s so a finish and the next start land in one cycle),
# and at least every 5 minutes: rebuild pages/data/factory.json (scripts/publish_factory_page.py); when it changed
# materially, commit that one file and push, which deploys the GitHub Page. Runs until killed.
#   start:  nohup bash scripts/publish_factory_loop.sh > /dev/null 2>&1 &
#   stop:   kill "$(cat artifacts/factory-page/loop.pid)"
# A cycle is skipped when .git/index.lock exists, when anything is staged (never commits other paths), or when the
# checkout is not on main. Log: artifacts/factory-page/loop.log (artifacts/ is gitignored).
cd "$(dirname "$0")/.." || exit 1
PY=.venv/Scripts/python.exe
LOGDIR=artifacts/factory-page
LOG=$LOGDIR/loop.log
FILE=pages/data/factory.json
INTERVAL=${INTERVAL:-300}
POLL=${POLL:-10}
DEBOUNCE=${DEBOUNCE:-20}
CHANGED=artifacts/coordination/queue/changed
MARK=$LOGDIR/last-cycle
mkdir -p "$LOGDIR"
echo $$ > "$LOGDIR/loop.pid"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
trap 'log "loop stopped"; exit 0' TERM INT
log "loop started (pid $$, every ${INTERVAL}s)"

cycle() {
  [ -e .git/index.lock ] && { log "skip: .git/index.lock exists"; return; }
  [ -n "$(git diff --cached --name-only)" ] && { log "skip: staged changes from someone else"; return; }
  [ "$(git rev-parse --abbrev-ref HEAD)" = main ] || { log "skip: not on main"; return; }
  out=$("$PY" scripts/publish_factory_page.py 2>&1); rc=$?
  if [ $rc -eq 3 ]; then log "no material change"; return; fi
  if [ $rc -ne 0 ]; then log "publisher failed (rc $rc): $(echo "$out" | tail -n 3 | tr '\n' ' ')"; return; fi
  summary=$(echo "$out" | sed -n 's/^SUMMARY //p' | tail -n 1)
  [ -e .git/index.lock ] && { log "skip commit: .git/index.lock appeared"; return; }
  git add "$FILE" || { log "git add failed"; return; }
  msg=$(printf 'Pages: factory status %s\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n' "$summary")
  if git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit -q -m "$msg" -- "$FILE" >> "$LOG" 2>&1; then
    log "committed: $summary"
  else
    log "commit failed or nothing to commit"; git reset -q -- "$FILE"; return
  fi
  if git push -q >> "$LOG" 2>&1; then log "pushed"
  elif git pull -q --rebase --autostash >> "$LOG" 2>&1 && git push -q >> "$LOG" 2>&1; then log "pushed after rebase"
  else log "push failed (will retry next cycle)"; git rebase --abort >> "$LOG" 2>&1; fi
}

# Sleep until INTERVAL has passed or the queue marker is newer than the start of the last cycle.
pause() {
  local waited=0
  while [ "$waited" -lt "$INTERVAL" ]; do
    sleep "$POLL" &
    wait $!
    waited=$((waited + POLL))
    if [ "$CHANGED" -nt "$MARK" ]; then
      sleep "$DEBOUNCE" &
      wait $!
      log "queue changed"
      return
    fi
  done
}

while true; do
  touch "$MARK"
  cycle
  pause
done
