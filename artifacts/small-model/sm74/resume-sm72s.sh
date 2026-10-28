#!/usr/bin/env bash
# SM74 hand-over (10-28 18:35): the sm72s cool-down from e20 took the GPU 40 s before SM74's scoring ticket, which
#   would then wait until the cool-down ends (~09:00 next day). The sm72s STOP file was touched (documented pause: snapshot
#   stop, launcher exits "STOP: ended"). This script waits for SM74's launcher to end and for the sm72s launcher to have
#   exited, removes the STOP file and restarts the sm72s launcher (its documented resume: finished work is skipped,
#   the cool-down resumes from its snapshot). Launched hidden via WMI; exec keeps the launcher in this process.
set -u
S74=C:/Users/jonat/workspace/anki-rwkv/artifacts/small-model/sm74
S=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s
LOG74=$S74/launch-sm74.log
LOG=$S/launch-sm72s.log
log() { echo "[sm74-resume $(date '+%m-%d %H:%M:%S')] $*" >> "$S74/resume-sm72s.log"; }
log "started"
n0=$(wc -l < "$LOG")
until tail -n +"$((n0 + 1))" "$LOG" | grep -q "STOP: ended with exit"; do sleep 30; done
log "sm72s launcher exited"
until tail -1 "$LOG74" | grep -qE "\] done$|STOP"; do sleep 60; done
log "sm74 ended: $(tail -1 "$LOG74")"
rm -f "$S/STOP"
log "STOP removed; restarting the sm72s launcher"
export PATH=/usr/bin:/mingw64/bin:$PATH
exec bash C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/launch-sm72s.sh >> "$S/launch-sm72s.nohup" 2>&1
