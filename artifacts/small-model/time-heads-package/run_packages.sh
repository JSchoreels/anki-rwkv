#!/bin/bash
# Package the longer Small Instant-Only model (SM113) and the longer cool-down for Tiny (SM115) for the fork (CPU).
set -e
cd "$(dirname "$0")/../../.."
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
export RWKV_DELETED_CARDS=neighbour CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
$PY -X utf8 scripts/small_model_export.py package --arch w64 --label sm113 \
  --checkpoint C:/Users/jonat/workspace/wt/sm113/artifacts/small-model/sm113-inst-e35st-st-dec2ep32k-t99-p2/expertium/E89-F65e-F77a/master.pth \
  --out artifacts/models/sm113
$PY -X utf8 scripts/small_model_export.py package --arch w64-c1 --label sm115 \
  --checkpoint C:/Users/jonat/workspace/wt/sm115/artifacts/small-model/sm115-c1-e38st-st-dec2ep32k-t99-p2/expertium/E89-F65e-F77a/master.pth \
  --out artifacts/models/sm115
echo PACKAGES-DONE
