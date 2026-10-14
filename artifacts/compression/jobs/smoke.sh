#!/usr/bin/env bash
# GPU smoke: 40 QAT steps (w4 per-channel) for memory/throughput, then 6 held-out users x 2 variants.
set -euo pipefail
cd /c/Users/jonat/workspace/wt/compression
PY=/c/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
FORK='C:\Users\jonat\workspace\anki\qt\aqt\rwkv_inference\RWKV_trained_on_5000_10000.pth'
nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu --format=csv
$PY scripts/compress_qat.py --bits 4 --group 0 --max-steps 40 --time-budget 600 --warmup 10 \
  --segment 16384 --batch-reviews 49152 --out artifacts/compression/qat/smoke-w4pc
$PY scripts/compress_score_heldout.py --weights "fork=$FORK" --variants fp32,w4-pc --users 6 \
  --out artifacts/compression/heldout-smoke.jsonl
