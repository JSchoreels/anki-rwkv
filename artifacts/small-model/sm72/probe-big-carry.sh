#!/usr/bin/env bash
# SM72 probe: big-model training speed/memory with carried state (strided lanes), 5 layouts (one queue ticket)
W=C:/Users/jonat/workspace/wt/sm72
for c in 16384x1 8192x2 4096x4 2048x8 2048x16; do
  env RWKV_DELETED_CARDS=neighbour C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 $W/scripts/sm72_big_epoch.py run --out $W/artifacts/small-model/sm72-big-probe-carry-$c
  echo "probe $c exit $?"
done
