#!/usr/bin/env bash
# QAT fine-tune of the fork's 2.76M RWKV. Usage: qat.sh <bits> <group|0> <name> [budget_seconds]
#   qat.sh 4 0 w4pc-run1        int4 per-channel (smallest int4: 1.64 MB with f32 scales)
#   qat.sh 2 32 w2g32-run1      ternary, group 32 (1.21 MB)
#   qat.sh 0 0 fp32-control1    same fine-tune without quantization (control)
# Each job: ~2,400 s of training + startup/compile; keep under 60 min.
set -euo pipefail
cd /c/Users/jonat/workspace/wt/compression
PY=/c/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
BITS=$1; GROUP=$2; NAME=$3; BUDGET=${4:-2400}
nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu --format=csv
$PY scripts/compress_qat.py --bits "$BITS" --group "$GROUP" --lr 1e-4 --warmup 30 --time-budget "$BUDGET" \
  --segment 16384 --batch-reviews 49152 --workers 6 --save-every 900 --out "artifacts/compression/qat/$NAME"
