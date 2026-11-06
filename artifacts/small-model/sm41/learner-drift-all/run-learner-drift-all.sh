#!/usr/bin/env bash
# The small model in the review simulator (SM41): the changing-learner simulation (SM41g) confirmation on all 117
# users (304 splits of fullrun/users.json). One fused run per user (all its splits, shared warm-up): tf + 48 policy
# lanes x 3 scenarios (improving g0.5_0.2, declining gm0.5_m0.2, no drift) = 145 lanes (learner-drift-all/lanes.txt),
# 60 days, --ivl-hist, the full run's pre-split FSRS-7 refits -> learner-drift-all/rows/<user>.jsonl.
# Jobs (jobs.txt, largest users first) are claimed via lock dirs. 4 workers, 1 thread each, below-normal priority,
# CPU only; a split is held while < 3 GB RAM is free. Nothing starts before 2026-11-07 00:41. A worker claims no new
# job from 40 min before a quiet.json window (scope all / cpu) to its end; the engine also holds per split from 10 min
# before. Resumable: finished splits are skipped; rerun after deleting learner-drift-all/locks.
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm41
H=$WT/artifacts/small-model/sm41
O=$H/learner-drift-all
F=$H/fullrun/fits
LANES=$(tr -d '\r\n' < "$O/lanes.txt")
START="2026-11-07 00:41:00"   # the user ended the 11-06 break early at 00:41
cd "$WT"
mkdir -p "$O/locks" "$O/rows" "$O/logs"
echo "start $(date '+%F %T') pid $$" >> "$O/runner.log"

while [ "$(date '+%F %T')" \< "$START" ]; do sleep 60; done

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
  while read -r u sps; do
    [ -z "$u" ] && continue
    mkdir "$O/locks/$u" 2>/dev/null || continue
    hold "$1"
    echo "worker $1 $u $sps $(date '+%F %T')" >> "$O/runner.log"
    "$PY" -X utf8 scripts/sm41_engine.py run --user "$u" --splits "$sps" --days 60 --lanes "$LANES" --ivl-hist \
      --fsrs7-fits "$F/$u.jsonl" --min-free-gb 3 --out "$O/rows/$u.jsonl" >> "$O/logs/$u.log" 2>&1
  done < "$O/jobs.txt"
}

worker 1 & worker 2 & worker 3 & worker 4 &
wait
echo "done $(date '+%F %T')" >> "$O/runner.log"
