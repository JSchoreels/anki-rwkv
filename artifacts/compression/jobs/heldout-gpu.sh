#!/usr/bin/env bash
# Held-out GPU evaluation (full history, 300 validation users outside 5000-10000). Extra tag=path weights may follow,
# e.g. heldout-gpu.sh qat-w4pc=artifacts/compression/qat/w4pc-run1/final.pth
set -euo pipefail
cd /c/Users/jonat/workspace/wt/compression
PY=/c/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
FORK='C:\Users\jonat\workspace\anki\qt\aqt\rwkv_inference\RWKV_trained_on_5000_10000.pth'
EXP='C:\Users\jonat\workspace\anki-rwkv\artifacts\full-run\run-8k-20261013\epoch-10\expertium\E89-F65e-F77a\master.pth'
ONLY="--only fork:fp32 --only fork:w8-pc --only fork:w4-g32 --only fork:w4-pc --only fork:w3-g32 --only fork:w3-pc --only fork:w2-g32 --only expertium-e10:fp32"
WEIGHTS=(--weights "fork=$FORK" --weights "expertium-e10=$EXP")
for extra in "$@"; do
  tag=${extra%%=*}; WEIGHTS+=(--weights "$extra"); ONLY="$ONLY --only $tag:fp32"
  case $tag in *w4pc*) ONLY="$ONLY --only $tag:w4-pc";; *w2g32*) ONLY="$ONLY --only $tag:w2-g32";; *w3pc*) ONLY="$ONLY --only $tag:w3-pc";; esac
done
$PY scripts/compress_score_heldout.py "${WEIGHTS[@]}" --variants fp32 $ONLY --users 300 --workers 4 \
  --time-limit 3000 --out artifacts/compression/heldout-scores.jsonl
