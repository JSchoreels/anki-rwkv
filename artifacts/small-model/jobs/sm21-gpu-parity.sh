#!/usr/bin/env bash
# SM21 GPU parity ticket (HANDOFF-SM21.md): kernel parity (S0/S_T kernels vs the CPU reference) then the trainer's
# carried-state path on the GPU (empty table vs stock, exact carry vs full history, update cost). ~10-15 min.
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/state-carry
cd "$WT" || exit 2
"$PY" -X utf8 "$WT/scripts/check_state_carry_kernels.py" --output "$WT/artifacts/small-model/check-state-carry-kernels-gpu.json"
a=$?
"$PY" -X utf8 "$WT/scripts/check_state_carry_gpu.py" --output "$WT/artifacts/small-model/check-state-carry-gpu.json"
b=$?
echo "[sm21-parity] kernels exit $a, trainer exit $b"
exit $((a | b))
