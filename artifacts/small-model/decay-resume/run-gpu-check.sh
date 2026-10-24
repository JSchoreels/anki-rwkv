#!/usr/bin/env bash
# GPU check of the resumable decay (scripts/check_decay_resume_gpu.py), one queue ticket, ~15 min:
# a 0.3% decay (20 users, 1.8M reviews) from the full retrain (SM56) epoch 10 with the SM56/SM55 decay options
# (extra-sm56-decay.json), run uninterrupted (A) and stopped at half + resumed (B); results in check/check.json.
#   bash C:/Users/jonat/workspace/wt/decay-resume/artifacts/small-model/decay-resume/run-gpu-check.sh
# Do not edit scripts/ expertium/ rwkv/ of this worktree while it runs (SourceWatch).
set -u
WT=C:/Users/jonat/workspace/wt/decay-resume
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
SRC=C:/Users/jonat/workspace/wt/sm56/artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes/run   # read only
D=$WT/artifacts/small-model/decay-resume
cd "$WT" || exit 1
"$PY" "$QUEUE" run --owner small-model --purpose "small-model: resumable decay GPU check (A uninterrupted vs B stop + resume, 20 users)" \
  --claim gpu --expected-minutes 20 --max-wait-minutes 240 --launched-by "run-gpu-check.sh" -- \
  "$PY" -X utf8 "$WT/scripts/check_decay_resume_gpu.py" --source-run "$SRC" --epoch 10 \
  --extra-options "$(cat "$D/extra-sm56-decay.json")" --out "$D/check" 2>&1 | tee "$D/run-gpu-check.log"
