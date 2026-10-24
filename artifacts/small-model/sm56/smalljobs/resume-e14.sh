#!/usr/bin/env bash
# User 10-24 ~16:45: "Tu peux lancer epoch 14". Resume the SM56 launcher for epoch 14 only, then pause again before epoch 15.
#   1. wait until the small-jobs runbook has scored epoch 13 (else the launcher would enqueue the same scoring)
#   2. remove STOP, relaunch launch-sm56.sh (records e13, trains e14)
#   3. once the launcher has enqueued e14's scoring, write STOP again: e14 is scored, then the launcher stops before epoch 15
export PATH=/usr/bin:/mingw64/bin:$PATH
S=C:/Users/jonat/workspace/wt/sm56/artifacts/small-model/sm56; J=$S/smalljobs; LOG=$J/resume-e14.log
log() { echo "[resume-e14 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
log "started"
until grep -q "exit [0-9]*: SM56 scoring s-w64kd-final-sc-lanes-e13" "$J/run-smalljobs.log"; do [ -e "$J/CANCEL" ] && { log CANCEL; exit 1; }; sleep 30; done
grep -q "exit 0: SM56 scoring s-w64kd-final-sc-lanes-e13" "$J/run-smalljobs.log" || { log "STOP: e13 scoring failed; not resuming"; exit 1; }
rm -f "$S/STOP"; log "STOP removed; relaunching launch-sm56.sh"
n0=$(wc -l < "$S/launch-sm56.log")
( cd C:/Users/jonat/workspace/wt/sm56 && echo "launched-e14 (resume-e14.sh)" >> "$S/launch-sm56.log" && nohup bash "$S/launch-sm56.sh" >> "$S/launch-sm56.log" 2>&1 ) &
until tail -n +"$n0" "$S/launch-sm56.log" | grep -q "enqueue: SM56 scoring s-w64kd-final-sc-lanes-e14"; do
  tail -n +"$n0" "$S/launch-sm56.log" | grep -q "STOP:" && { log "launcher stopped early: $(tail -n +"$n0" "$S/launch-sm56.log" | grep 'STOP:' | head -1)"; exit 1; }
  sleep 30; done
echo "User 2026-10-24 ~16:45: epoch 14 only, then pause again; wait for the user's word before epoch 15." > "$S/STOP"
log "e14 scoring enqueued; STOP written (launcher stops before epoch 15)"
wait
log "launcher exited"
