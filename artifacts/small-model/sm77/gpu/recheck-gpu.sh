#!/usr/bin/env bash
# SM77: one short queue ticket re-running only the GPU check with the new P / L bars (commit 81690dfa); no smoke, no
# gpu-check-ok marker (the launcher's own ticket 0 still runs on relaunch). Output: check-gpu-sm77-recheck.json, recheck.out.
WT=C:/Users/jonat/workspace/wt/sm77
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
G=$WT/artifacts/small-model/sm77/gpu
cd "$WT" || exit 1
"$PY" C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py run --owner small-model \
  --purpose "small-model: tiny architecture additions (SM77): GPU check re-run with the new P / L bars (check only, ~2 min)" \
  --claim gpu --expected-minutes 10 --max-wait-minutes 1080 --launched-by "detached sm77/gpu/recheck-gpu.sh" -- \
  env RWKV_DELETED_CARDS=neighbour "$PY" -X utf8 "$WT/scripts/sm77_warm.py" gpu-check \
  --out "$G/check-gpu-sm77-recheck.json" --ext "$WT/artifacts/small-model/sm77-ext-e20" >> "$G/recheck.out" 2>&1
echo "[recheck] exit $? $(date '+%m-%d %H:%M:%S')" >> "$G/recheck.out"
