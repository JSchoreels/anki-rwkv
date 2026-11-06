#!/usr/bin/env bash
# One extra segment of the joint cool-down 20:00-21:45 on 11-06: the user moved tonight's break start to 22:00
# ("you can go on until 10PM without issue tonight", 11-06 ~16:00). The running launcher (launch-sm57.sh) keeps its
# hard-coded 20:00-01:00 daily break: it snapshot-stops the joint segment at ~19:58 and then sleeps until 01:00. This
# script waits for that stop, runs one more resumable segment through the queue (same command as the launcher's), and
# requests a snapshot stop at 21:45. At 01:00 the launcher resumes from the newest snapshot (or sees the cool-down
# complete and goes on to scoring). Separate from the launcher; never edits it.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm57
G=$WT/artifacts/small-model/sm57/gpu
D=$WT/artifacts/small-model/sm57-joint-e26-dec-1ep32k-muon
DEC_PY=$WT/scripts/expertium_decay.py
L=$G/extra-segment-1106.log
log() { echo "[extra $(date '+%m-%d %H:%M:%S')] $*" >> "$L"; }
log "start pid $$"
# 1. wait until the launcher logged the end of the joint segment (its 19:58 snapshot stop)
until grep -q "exit .*: joint: .* segment" "$G/launch-sm57.log"; do
  [ "$(date +%H%M)" -ge 2130 ] && { log "no joint segment end by 21:30; nothing done"; exit 0; }
  sleep 30
done
log "launcher: $(grep "exit .*: joint: .* segment" "$G/launch-sm57.log" | tail -1)"
"$PY" -X utf8 -c "
import json, sys
r = json.load(open(sys.argv[1]))
sys.exit(0 if r.get('status') == 'complete' else 1)" "$D/expertium/E89-F65e-F77a/result.json" 2>/dev/null \
  && { log "joint cool-down already complete; nothing to do"; exit 0; }
while [ "$(date +%H%M)" -lt 2000 ]; do sleep 20; done
now=$(date +%H%M); [ "$now" -ge 2125 ] && { log "too late ($now); nothing done"; exit 0; }
mins=$(( (21*60+45) - (10#${now:0:2}*60 + 10#${now:2:2}) ))
rm -f "$D/control/STOP"
log "enqueue extra segment, expected $mins min"
"$PY" "$QUEUE" run --owner sm57 --purpose "the instant-only small model (SM57): joint: the release's one-epoch 32k Muon cool-down from epoch 26, extra segment 20:00-21:45 (user moved tonight's break to 22:00)" \
  --claim gpu --expected-minutes "$mins" --max-wait-minutes 20 --launched-by "sm57/gpu/extra-segment-1106.sh" -- \
  env RWKV_DELETED_CARDS=neighbour "$PY" -X utf8 "$DEC_PY" run --out "$D" >> "$G/train.out" 2>&1 &
qpid=$!
stop_sent=0
while kill -0 "$qpid" 2>/dev/null; do
  if [ "$stop_sent" = 0 ] && [ "$(date +%H%M)" -ge 2145 ]; then
    log "requesting a snapshot stop"; "$PY" -X utf8 "$DEC_PY" stop --out "$D" > /dev/null 2>&1; stop_sent=1
  fi
  sleep 20
done
wait "$qpid"; log "exit $? (stop sent $stop_sent)"
