#!/usr/bin/env bash
# Replace runbook f (pid 4606) by f2 (break moved to 22:00) once f idles in wait_slot after the speed profile.
# Only kills f while it is sleeping in wait_slot (never during a ticket); f2 appends to f's log.
SM=/c/Users/jonat/workspace/wt/small-model/artifacts/small-model
LOG=$SM/runbook-2026-10-17f-evening-night.log
until grep -aq "speed report written" $LOG && grep -a "speed report written" -A100 $LOG | grep -aq "break window: waiting"; do
  if grep -aq "enqueue: SM35" $LOG; then echo "[switch $(date +%T)] f enqueued SM35 itself; not switching" >> $LOG; exit 0; fi
  sleep 30
done
kill 4606 2>/dev/null; sleep 2; pkill -P 4606 sleep 2>/dev/null
echo "[switch $(date '+%m-%d %H:%M:%S')] runbook f stopped while waiting; starting f2 (break 22:00-02:00)" >> $LOG
cd /c/Users/jonat/workspace/wt/small-model && nohup bash $SM/runbook-2026-10-17f2-rest.sh >> $LOG 2>&1 &
