#!/usr/bin/env bash
# SM21 v2 (strided lanes) GPU check ticket (HANDOFF-SM21.md in the sm21v2 worktree): the trainer's v2 carried-state path
# on the GPU (P empty table vs stock, X exact carry vs full history with zero and seeded lane starts, T per-update cost
# stock / v1 / v2, M table size and peak memory on the largest training user). The S0/S_T kernels themselves passed in
# the v1 parity ticket (000435, check-state-carry-kernels-gpu.json) and are unchanged. ~10-15 min.
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm21v2
cd "$WT" || exit 2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"$PY" -X utf8 "$WT/scripts/check_state_carry_lanes_gpu.py" --output "$WT/artifacts/small-model/check-state-carry-lanes-gpu.json"
rc=$?
echo "[sm21v2-gpu-check] exit $rc"
exit $rc
