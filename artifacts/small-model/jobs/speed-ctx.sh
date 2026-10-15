#!/usr/bin/env bash
# Queue job (small-model, Q3 / S06): long-context training speed profile, w64 wide kernels, no KD, 160 users.
# Refuses to start if other processes hold more than 3.6 GB of VRAM (skip rule of the queue row).
# Finished cases are skipped. Per-case outputs (incl. gpu_power.jsonl) are in $OUT/<case>.
cd /c/Users/jonat/workspace/wt/small-model || exit 1
export PATH="/usr/bin:/bin:$PATH"
export GPU_R01_CONTROLLER_ROOT='C:\Users\jonat\workspace\anki-rwkv'
PY=/c/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
OUT=artifacts/small-model/speed-ctx-20261016
CASES="${CASES:-w64-ctx128-b256 w64-ctx1024-b32 w64-ctx2048-b16 w64-ctx2048-b32}"
MAX_OTHERS_MB="${MAX_OTHERS_MB:-3686}"   # 3.6 GB
mkdir -p "$OUT"
{ date; nvidia-smi; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; } >> "$OUT/all.log" 2>&1
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' \r')
case "$USED" in ''|*[!0-9]*) echo "cannot read VRAM in use ('$USED'); refusing to start" | tee -a "$OUT/all.log"; exit 3;; esac
if [ "$USED" -gt "$MAX_OTHERS_MB" ]; then
  echo "VRAM in use by others is ${USED} MiB > ${MAX_OTHERS_MB} MiB at start; skipping S06 (queue row skip rule)" | tee -a "$OUT/all.log"
  exit 2
fi
if [ ! -e "$OUT/speed.json" ]; then
  "$PY" -X utf8 scripts/expertium_speed.py prepare --out "$OUT" --users 160 || exit 1
fi
for CASE in $CASES; do
  if [ -e "$OUT/$CASE" ]; then continue; fi
  "$PY" -X utf8 scripts/expertium_speed.py case --out "$OUT" --case "$CASE"
  echo "$CASE exit $? at $(date)" >> "$OUT/all.log"
done
"$PY" -X utf8 scripts/expertium_speed.py report --out "$OUT"
