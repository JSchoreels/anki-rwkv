#!/usr/bin/env bash
# Wait for chain-q1-q3.sh to finish, then launch Q4 (runbook-2026-10-16.sh extend), only if before 05:00.
SM=C:/Users/jonat/workspace/wt/small-model/artifacts/small-model
until grep -q "\] done" $SM/chain-q1-q3.log; do sleep 30; done
now=$(date +%H%M)
if [ "$now" -ge 0500 ]; then echo "[q4 $(date +%H:%M:%S)] chain done after 05:00, NOT launching Q4"; exit 1; fi
echo "[q4 $(date +%H:%M:%S)] chain done, launching extend"
TIER=gate EPOCHS=6 RUN_TAG=ctx2048 nohup "C:/Program Files/Git/usr/bin/bash.exe" $SM/runbook-2026-10-16.sh extend > $SM/runbook-2026-10-16-extend.log 2>&1 &
disown
echo "[q4 $(date +%H:%M:%S)] launched"
