#!/usr/bin/env bash
# The small model in the review simulator (SM41), the changing-learner simulation (SM41g): the 20 users of part A
# (diag/sample.json) x 1 split, 60 days, the lanes of learner-drift/lanes.txt, scenarios improve (--drift 0.5,0.2),
# decline (--drift -0.5,-0.2) and none (no drift), all with --ivl-hist -> learner-drift/<scenario>/rows/<user>.jsonl.
# Jobs (learner-drift/jobs.txt: improve and decline first, largest users first, then none) are claimed by workers via
# lock directories. 2 workers start at once; 2 more once the real-data check (SM41r) runner says "done" (CPU cap of
# 4 sm41_ processes). Below-normal priority, 1 thread, CPU only. A worker claims no new job from 40 min before a
# quiet.json window (scope all / cpu) to its end; the engine itself also holds per split from 10 min before.
# Resumable: finished splits are skipped (the engine's done keys); rerun after deleting learner-drift/locks.
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm41
H=$WT/artifacts/small-model/sm41
O=$H/learner-drift
F=$H/fullrun/fits
RC=$H/realcheck/runner.log
LANES=$(tr -d '\r\n' < "$O/lanes.txt")
cd "$WT"
mkdir -p "$O/locks"
echo "start $(date '+%F %T') pid $$" >> "$O/runner.log"

hold() {   # wait while a quiet window runs or starts within 40 min
  local said=0
  while "$PY" -c "
import datetime as dt, sys
sys.path.insert(0, '.')
from scripts import sm41_drift as DR
sys.exit(0 if DR.hold_reason(dt.datetime.now() + dt.timedelta(minutes=30)) else 1)" 2>/dev/null; do
    [ $said = 0 ] && echo "worker $1 holding for a quiet window $(date '+%F %T')" >> "$O/runner.log" && said=1
    sleep 120
  done
}

worker() {
  while read -r sc u sp; do
    [ -z "$sc" ] && continue
    mkdir "$O/locks/$sc-$u" 2>/dev/null || continue
    hold "$1"
    case $sc in
      improve) X="--drift 0.5,0.2" ;;
      decline) X="--drift=-0.5,-0.2" ;;
      *) X="" ;;
    esac
    mkdir -p "$O/$sc/rows" "$O/$sc/logs"
    echo "worker $1 $sc $u $sp $(date '+%F %T')" >> "$O/runner.log"
    "$PY" -X utf8 scripts/sm41_engine.py run --user "$u" --split "$sp" --days 60 --lanes "$LANES" $X --ivl-hist \
      --fsrs7-fits "$F/$u.jsonl" --min-free-gb 3 --out "$O/$sc/rows/$u.jsonl" >> "$O/$sc/logs/$u.log" 2>&1
  done < "$O/jobs.txt"
}

worker 1 & worker 2 &
( while ! grep -q '^done ' "$RC" 2>/dev/null; do
    [ "$(ls "$O/locks" | wc -l)" -ge "$(grep -c . "$O/jobs.txt")" ] && exit 0
    sleep 60
  done
  echo "real-data check done, 2 more workers $(date '+%F %T')" >> "$O/runner.log"
  worker 3 & worker 4 & wait ) &
wait
echo "done $(date '+%F %T')" >> "$O/runner.log"
