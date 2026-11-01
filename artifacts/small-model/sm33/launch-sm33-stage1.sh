#!/usr/bin/env bash
# Grade-order sanity (SM33), stage 1: how often the released small model (the leak-free small model (SM72s) 25 epochs
# + 1-epoch 32k Muon cool-down, s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon, fp32) breaks
# interval(Again) <= interval(Hard) <= interval(Good) <= interval(Easy). scripts/sm33_grade_order.py.
# One GPU ticket: check (probe exactness on CUDA, user 4, first 20,000 reviews) then measure (723 users of the 732
# pool, 3,000 sampled reviews per user, 4 what-if rows each); then the CPU report.
#   launch   WMI hidden (artifacts/small-model/sm33/HANDOFF.md); log artifacts/small-model/sm33/gpu/launch-stage1.log
#   resume   the launch command again (users already measured are skipped)
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm33
S=$WT/artifacts/small-model/sm33
G=$S/gpu
OUT=$S/work/stage1-released-fp32
mkdir -p "$G" "$OUT"
LOG=$G/launch-stage1.log
M=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/expertium/E89-F65e-F77a/master.pth
USERS=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json
MIN=40
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION
log() { echo "[sm33 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

free_minutes() {  # minutes to the next quiet window (quiet.json scope all/gpu, or the daily 20:00-01:00 break); 0 = inside one
  Q="$QUIET" "$PY" -X utf8 - <<'PYEOF'
import json, os
from datetime import datetime, timedelta
now = datetime.now()
try:
    wins = [(datetime.fromisoformat(w["start"]), datetime.fromisoformat(w["end"]))
            for w in json.loads(open(os.environ["Q"], encoding="utf-8").read()) if w.get("scope", "all") in ("all", "gpu")]
except Exception:
    print(0); raise SystemExit
day = now.replace(hour=0, minute=0, second=0, microsecond=0)
wins += [(day + timedelta(days=d, hours=20), day + timedelta(days=d + 1, hours=1)) for d in range(-1, 3)]
if any(s <= now < e for s, e in wins):
    print(0); raise SystemExit
print(int(min([(s - now).total_seconds() // 60 for s, e in wins if s > now] + [100000])))
PYEOF
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1)"
FREE=$(free_minutes)
if [ "$FREE" -lt $((MIN + 5)) ]; then log "STOP: only $FREE min to the next quiet window (need $MIN)"; exit 0; fi
MAXWAIT=$((FREE - MIN - 3))
log "enqueue (expected $MIN min, max wait $MAXWAIT min)"
"$PY" "$QUEUE" run --owner small-model --purpose "grade-order sanity (SM33): stage 1, CUDA probe check + what-if intervals of the released small model on the 732 pool (723 users, 3,000 reviews each)" \
  --claim gpu --expected-minutes $MIN --max-wait-minutes $MAXWAIT --launched-by "detached launcher sm33/launch-sm33-stage1.sh" -- \
  bash -c "\"$PY\" -X utf8 $WT/scripts/sm33_grade_order.py check --model $M --device cuda --user 4 --max-rows 20000 --probes 3000 --json $G/check-cuda.json \
    && \"$PY\" -X utf8 $WT/scripts/sm33_grade_order.py measure --model $M --device cuda --users-file $USERS --probes 3000 --time-limit 2100 --out $OUT" \
  >> "$G/stage1.out" 2>&1
rc=$?
log "ticket exit $rc"
[ "$rc" = 0 ] || exit $rc
"$PY" -X utf8 "$WT/scripts/sm33_grade_order.py" report --out "$OUT" >> "$G/stage1.out" 2>&1
log "report exit $?: $OUT/sm33-stage1.json"
cp "$OUT/sm33-stage1.json" "$S/sm33-stage1.json" 2>/dev/null
log "done"
