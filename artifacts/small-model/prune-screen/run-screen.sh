#!/usr/bin/env bash
# CPU screen scorer for one shard of the 68 feature-screen users (design 2.1). 2 torch threads per process.
set -u
SHARD=$1; EXTRA="${2:-}"
cd C:/Users/jonat/workspace/wt/prune || exit 1
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
P=C:/Users/jonat/workspace/wt/sm21v2/artifacts/small-model/scale-w64kd-gate-sc-lanes-cont-decay1ep32k/expertium/E89-F65e-F77a/master.pth
S=artifacts/small-model/prune-screen
M="--model P_dec=expertium:w64:$P"; ONLY="--only P_dec/w8+s8"
for a in w64-d2p2u2 w64-d2p1u1 w64-d2p1u1-nocmix w64-d2p3u3 w64-d4p1u3 w64-d4p3u1 $EXTRA; do
  [ -f $S/$a.pth ] || $PY -X utf8 scripts/small_model_prune.py depth --src $P --arch $a --out $S/$a.pth > /dev/null
  M="$M --model $a=expertium:$a:$S/$a.pth"; ONLY="$ONLY --only $a/w8+s8"
done
exec $PY -X utf8 scripts/small_model_heldout.py --device cpu --users-file C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/feature-screen/shard$SHARD.json \
  $M $ONLY --time-limit 100000 --out $S/screen-s$SHARD.jsonl
