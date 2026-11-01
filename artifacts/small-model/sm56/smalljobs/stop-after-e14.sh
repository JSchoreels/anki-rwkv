#!/usr/bin/env bash
# User 10-24 ~16:45: epoch 14 only. Once the launcher has enqueued e14's scoring, write STOP: it stops before epoch 15.
export PATH=/usr/bin:/mingw64/bin:$PATH
S=C:/Users/jonat/workspace/wt/sm56/artifacts/small-model/sm56; LOG=$S/smalljobs/resume-e14.log
until grep -q "enqueue: SM56 scoring s-w64kd-final-sc-lanes-e14 " "$S/launch-sm56.log"; do
  [ -e "$S/smalljobs/CANCEL" ] && exit 1; sleep 30; done
echo "User 2026-10-24 ~16:45: epoch 14 only, then pause again; wait for the user's word before epoch 15." > "$S/STOP"
echo "[resume-e14 $(date '+%m-%d %H:%M:%S')] e14 scoring enqueued; STOP written (launcher stops before epoch 15)" >> "$LOG"
