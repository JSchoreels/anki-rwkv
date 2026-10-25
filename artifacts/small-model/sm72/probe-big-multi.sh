#!/usr/bin/env bash
# SM72 probe: big-model training speed/memory at 16,384 reviews per update in 3 window lengths (one queue ticket)
W=C:/Users/jonat/workspace/wt/sm72
for c in 8192 4096 2048; do
  env RWKV_DELETED_CARDS=neighbour C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 $W/scripts/sm72_big_epoch.py run --out $W/artifacts/small-model/sm72-big-probe-$c
  echo "probe $c exit $?"
done
