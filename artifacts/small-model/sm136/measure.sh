#!/usr/bin/env bash
# measure.sh USER TAG=kind:arch:ckpt LABEL... ; MODE=peak (default, peakrun.py) or prof (${PROF:-memprof.py}); optional USERFILE env
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
S=C:/Users/jonat/workspace/wt/sm136/artifacts/small-model/sm136
u=$1; model=$2; shift 2
only=""; for l in "$@"; do only="$only --only $l"; done
echo "{\"users\": [$u]}" > $S/cpu/r2/users-$u.json
export OMP_NUM_THREADS=${NT:-4} MKL_NUM_THREADS=${NT:-4} CUDA_VISIBLE_DEVICES=
unset RWKV_DELETED_CARDS EXPERTIUM_END_TO_START SM11_INPUTS
ARGS="--device cpu --inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets 0,32,128,300,1000 ${RC:+--row-chunk $RC} --pool 732 --users-file $S/cpu/r2/users-$u.json --model $model $only --time-limit 1000000 --out ${OUT:-$S/cpu/r2/meas-$u-$$.jsonl}"
if [ "${MODE:-peak}" = prof ]; then $PY -X utf8 -W ignore $S/${PROF:-memprof.py} $ARGS 2>&1 | tail -${TAILN:-30} | cut -c1-300
else $PY -X utf8 $S/peakrun.py $PY -X utf8 -W ignore C:/Users/jonat/workspace/wt/sm136/scripts/small_model_heldout.py $ARGS 2>&1 | tail -3 | cut -c1-400; fi
