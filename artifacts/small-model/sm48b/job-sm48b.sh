#!/usr/bin/env bash
# SM48b (one GPU queue ticket, --timing): --lanes 1/2/4 with the int state variants (no lock) and lane review budgets.
# 40 SM48 timing users of the 732 pool (timing-users.json), SM36 w8+s8, w8+s4c8n, w8+s8hx8, w8+s4c8nhx8, default memory cap 0.8.
# runs: 1-l1, 2-l2, 3-l4 (budget 400k), 4-l4b250 (250k), 5-l4b150 (150k), 6-l4lock (400k, HELDOUT_INT_LANE_LOCK=1 = the SM47 lock),
# 7-l1 (repeat). Each run is compared with run 1 (compare_heldout_rows.py). STOP on a failed run or compare.
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm48b
R=artifacts/small-model/sm48b
O=$WT/$R/timing
CK=C:/Users/jonat/workspace/wt/sm42/artifacts/small-model/scale-w64kd-gate-sc-lanes-cont-decay1ep32k-muon/expertium/E89-F65e-F77a/master.pth
L=sm36
log() { echo "[sm48b $(date '+%m-%d %H:%M:%S')] $*"; }
cd "$WT" || exit 1
unset HELDOUT_GPU_MEMORY_FRACTION HELDOUT_CPU_LANES_TEST HELDOUT_INT_LANE_LOCK
RUNS=("1-l1|l1||" "2-l2|l2|--lanes 2|" "3-l4|l4|--lanes 4|" "4-l4b250|l4b250|--lanes 4 --lane-review-budget 250000|" \
      "5-l4b150|l4b150|--lanes 4 --lane-review-budget 150000|" "6-l4lock|l4lock|--lanes 4|1" "7-l1|l1||")
if [ -n "$DRYRUN" ]; then for r in "${RUNS[@]}"; do log "DRYRUN $r"; done; log "DRYRUN ok"; exit 0; fi
mkdir -p "$O"
rm -f "$O"/run-*.jsonl "$O"/run-*.log "$O"/smi-*.csv "$O"/compare-*.json "$O/runs.jsonl"
for r in "${RUNS[@]}"; do
  IFS='|' read -r name arm flags lock <<< "$r"
  log "run $name (${flags:-lanes 1}${lock:+, int-variant lock})"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 500 > "$O/smi-$name.csv" 2>/dev/null &
  smi=$!
  t0=$(date +%s.%N)
  # shellcheck disable=SC2086
  HELDOUT_INT_LANE_LOCK=$lock "$PY" -X utf8 -W ignore scripts/small_model_heldout.py --pool 732 --users-file "$R/timing-users.json" \
    --model "$L=expertium:w64:$CK" --only "$L/w8+s8" --only "$L/w8+s4c8n" --only "$L/w8+s8hx8" --only "$L/w8+s4c8nhx8" \
    --time-limit 100000 --out "$R/timing/run-$name.jsonl" $flags > "$O/run-$name.log" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  kill $smi 2>/dev/null; wait $smi 2>/dev/null
  secs=$(awk "BEGIN{printf \"%.1f\", $t1 - $t0}")
  echo "{\"run\": \"$name\", \"arm\": \"$arm\", \"flags\": \"$flags\", \"lock\": ${lock:-0}, \"exit\": $rc, \"process_seconds\": $secs}" >> "$O/runs.jsonl"
  log "run $name exit $rc, ${secs}s; $(grep -a 'review-evals/s' "$O/run-$name.log" | tail -1)"
  [ $rc -eq 0 ] || { log "STOP: run $name failed"; tail -5 "$O/run-$name.log"; exit 1; }
  if [ "$name" != "1-l1" ]; then
    "$PY" -X utf8 scripts/compare_heldout_rows.py "$R/timing/run-1-l1.jsonl" "$R/timing/run-$name.jsonl" \
      --out "$R/timing/compare-$name.json" || log "FAIL: run $name rows differ from run 1 beyond tolerance (continuing)"
  fi
done
"$PY" -X utf8 "$R/sm48b_summary.py" || log "summary exit $?"
log "done"
