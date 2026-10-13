#!/bin/sh
# Interleaved blocks in the disposable sandbox clone (results never enter this workspace's gpu-retrain-r01 records).
# A = default Windows scheduling, C = B plus main process on CPUs 0,1,14,15, B = external pin_workers.py (one preparation worker per physical core).
HERE=/c/Users/jonat/workspace/anki-rwkv/artifacts/diagnostics/r01-native-toolchain/baseline-drift/live
SB=/c/Users/jonat/workspace/r01-drift-sandbox
cd "$SB" || exit 1
PY=.venv/Scripts/python.exe
OUT=artifacts/retraining/gpu-retrain-r01/iteration-9072
REASON="Uncounted live drift diagnostic: default scheduler vs workers pinned one per physical core (human-authorized 2026-10-13)"
for mode in $MODES; do
  n=$(( $(ls -d $OUT/replicates/r* 2>/dev/null | wc -l) + 1 ))
  if [ "$mode" != A ]; then EXTRA=""; [ "$mode" = C ] && EXTRA="--main-cpus 0,1,14,15";
    $PY -X utf8 $HERE/pin_workers.py --log $HERE/pins-r$n.json --timeout 1500 $EXTRA > $HERE/pins-r$n.out 2>&1 &
    W=$!
  fi
  $PY -X utf8 scripts/r01_replicates.py run --output $OUT --sequence E33-F48-F50 --reason "$REASON mode=$mode" > $HERE/replicate-r$n.log 2>&1
  code=$?
  if [ "$mode" != A ]; then wait $W; cat $HERE/pins-r$n.out; fi
  echo "r$n mode=$mode exit=$code $(grep -o 'r[0-9]* E33-F48-F50: [0-9,]* reviews/s' $HERE/replicate-r$n.log | tail -1)"
  [ $code -ne 0 ] && { tail -4 $HERE/replicate-r$n.log; exit 1; }
done
echo ALL-DONE
