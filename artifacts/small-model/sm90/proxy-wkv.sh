#!/bin/bash
cd C:/Users/jonat/workspace/wt/sm90
export OMP_NUM_THREADS=4
C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 artifacts/small-model/sm90/proxy_sm90.py --books wkv-only-n.npz --parts wkv --out artifacts/small-model/sm90/proxy-sm90-wkv.json > artifacts/small-model/sm90/logs/proxy-wkv.log 2>&1
echo "[proxy-wkv] done $(date +%H:%M:%S)"
