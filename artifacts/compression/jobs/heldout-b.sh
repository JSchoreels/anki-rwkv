#!/usr/bin/env bash
# Held-out evaluation B chunk (resumable): next users of the stratified cohort, all six variants, <= 3,200 s of scoring.
# Usage: heldout-b.sh [count]   (count = cohort prefix to cover, default 1000)
set -euo pipefail
cd /c/Users/jonat/workspace/wt/compression
PY=/c/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu --format=csv
$PY scripts/compress_heldout_b.py --count "${1:-1000}" --workers 4 --time-limit 3200 \
  --variants fp32,fp32+s8,w8-pc+s8,w4-g32+s8,w4-g64+s8,w4-g32+s4 --out artifacts/compression/heldout-b.jsonl
