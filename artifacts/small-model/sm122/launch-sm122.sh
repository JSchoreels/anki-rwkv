#!/usr/bin/env bash
# Retrieval-effort probe (SM122): CPU only, no GPU, no queue ticket (design: artifacts/small-model/sm122/design.md).
#   Runs the three model passes in parallel (each python process: 4 threads, below-normal priority, CUDA hidden, checks
#   quiet.json scope all/cpu and the daily 20:00-01:00 break before every user and sleeps through a window), then the
#   report. Per-user arrays in sm122/work/ (gitignored); log in sm122/run/launch-sm122.log.
#   stop: touch C:/Users/jonat/workspace/wt/sm122/artifacts/small-model/sm122/run/STOP (checked before every user)
#   resume: launch again (finished users are skipped)
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm122
S=$WT/artifacts/small-model/sm122
R=$S/run
mkdir -p "$R"
LOG=$R/launch-sm122.log
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=-1
log() { echo "[sm122 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1)"
if [ -n "$(git -C "$WT" status --porcelain -- scripts)" ]; then log "STOP: uncommitted changes in scripts/"; exit 1; fi
pids=""
for m in ${MODELS:-sm92 sm106 sm91s2}; do
  "$PY" -X utf8 "$WT/scripts/sm122_effort.py" run --model "$m" >> "$R/run-$m.out" 2>&1 &
  pids="$pids $!"
  log "run $m: pid $!"
done
rc=0
for p in $pids; do wait "$p" || rc=$?; done
log "runs ended (last nonzero rc $rc): $(ls "$S"/work/*/DONE 2>/dev/null | wc -l) of 3 DONE"
[ -e "$R/STOP" ] && { log "STOP file present: no report"; exit 1; }
"$PY" -X utf8 "$WT/scripts/sm122_effort.py" report >> "$R/report.out" 2>&1
log "report exit $?: $(tail -1 "$R/report.out" | cut -c1-200)"
log "done"
