#!/usr/bin/env bash
# SM48 (one GPU queue ticket, --timing): held-out scoring speed A/B on 40 users of the 732 pool (timing-users.json,
# 2,255,532 reviews), SM36 w8+s8 and w8+s4c8n, default GPU memory cap 0.8, 4 prep workers.
#   arms: old = --legacy-metrics (dict metrics, one user at a time: the pre-SM48 path), arr = numpy metrics,
#         l2 / l4 = numpy metrics + --lanes 2 / 4
#   order old arr l2 l4 l4 l2 arr old (both orders cancel warm-up); nvidia-smi samples every 0.5 s per run
#   every run is compared with the first old run (scripts/compare_heldout_rows.py, AGENT.md evaluation tolerance;
#   bitwise counts recorded); STOP on a failed compare or a failed run
#   then timing/sm48_summary.py -> timing/summary.md / summary.json
# DRYRUN=1: print the runs and check the inputs, no GPU.
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm48
R=artifacts/small-model/sm48
O=$WT/$R/timing
CK=C:/Users/jonat/workspace/wt/sm42/artifacts/small-model/scale-w64kd-gate-sc-lanes-cont-decay1ep32k-muon/expertium/E89-F65e-F77a/master.pth
L=sm36
log() { echo "[sm48 $(date '+%m-%d %H:%M:%S')] $*"; }
cd "$WT" || exit 1
unset HELDOUT_GPU_MEMORY_FRACTION HELDOUT_CPU_LANES_TEST
for f in "$PY" "$CK" "$R/timing-users.json" scripts/small_model_heldout.py scripts/heldout_fast_metrics.py \
         scripts/compare_heldout_rows.py "$R/sm48_summary.py"; do
  [ -e "$f" ] || { log "STOP: missing $f"; exit 1; }
done
log "start $(git log -1 --format='%h %s' | cut -c1-120); dirty: $(git status --porcelain | wc -l) paths"
ARMS=(old arr l2 l4 l4 l2 arr old)
declare -A FLAGS=([old]="--legacy-metrics" [arr]="" [l2]="--lanes 2" [l4]="--lanes 4")
if [ -n "$DRYRUN" ]; then
  for i in "${!ARMS[@]}"; do a=${ARMS[$i]}; log "DRYRUN run $((i+1)) $a: small_model_heldout.py ... ${FLAGS[$a]}"; done
  log "DRYRUN ok"; exit 0
fi
mkdir -p "$O"
rm -f "$O"/run-*.jsonl "$O"/run-*.log "$O"/smi-*.csv "$O"/compare-*.json "$O/runs.jsonl"
for i in "${!ARMS[@]}"; do
  a=${ARMS[$i]}; name="$((i+1))-$a"
  log "run $name (${FLAGS[$a]:-default})"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 500 > "$O/smi-$name.csv" 2>/dev/null &
  smi=$!
  t0=$(date +%s.%N)
  # shellcheck disable=SC2086
  "$PY" -X utf8 -W ignore scripts/small_model_heldout.py --pool 732 --users-file "$R/timing-users.json" \
    --model "$L=expertium:w64:$CK" --only "$L/w8+s8" --only "$L/w8+s4c8n" --time-limit 100000 \
    --out "$R/timing/run-$name.jsonl" ${FLAGS[$a]} > "$O/run-$name.log" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  kill $smi 2>/dev/null; wait $smi 2>/dev/null
  secs=$(awk "BEGIN{printf \"%.1f\", $t1 - $t0}")
  echo "{\"run\": \"$name\", \"arm\": \"$a\", \"flags\": \"${FLAGS[$a]}\", \"exit\": $rc, \"process_seconds\": $secs}" >> "$O/runs.jsonl"
  log "run $name exit $rc, ${secs}s; $(grep -a 'review-evals/s' "$O/run-$name.log" | tail -1)"
  [ $rc -eq 0 ] || { log "STOP: run $name failed (tail of $O/run-$name.log)"; tail -5 "$O/run-$name.log"; exit 1; }
  if [ "$name" != "1-old" ]; then
    "$PY" -X utf8 scripts/compare_heldout_rows.py "$R/timing/run-1-old.jsonl" "$R/timing/run-$name.jsonl" \
      --out "$R/timing/compare-$name.json" || { log "STOP: run $name rows differ from run 1-old beyond tolerance"; exit 1; }
  fi
done
"$PY" -X utf8 "$R/sm48_summary.py" || log "summary exit $?"
log "done"
