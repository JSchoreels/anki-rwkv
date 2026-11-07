#!/usr/bin/env bash
# The grade order by construction (SM95): GPU scorer smoke (one queue ticket of launch-sm95.sh): the reference
# checkpoint on 2 users of the 732 set, raw and with --grade-rectify (w8+s8, leak-free), into gpu/smoke-rect.jsonl.
set -e
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm95
S=$WT/artifacts/small-model/sm95
CK=C:/Users/jonat/workspace/wt/sm77/artifacts/small-model/sm77-ref-e21-dec-q32k-muon/expertium/E89-F65e-F77a/master.pth
for flag in "" "--grade-rectify"; do
  tag=sm77-ref-e21-q32k; [ -n "$flag" ] && tag=sm77-ref-e21-q32k-rect
  "$PY" -X utf8 "$WT/scripts/small_model_heldout.py" --pool 732 --users-file "$S/smoke-users.json" --model "$tag=expertium:w64:$CK" \
    --only "$tag/w8+s8" --time-limit 600 --split-existing --deleted-cards neighbour $flag --out "$S/gpu/smoke-rect.jsonl"
done
