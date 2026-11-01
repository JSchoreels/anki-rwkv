#!/usr/bin/env bash
# SM32c reference rows (CPU, 2 processes x 3 threads, alongside the GPU): the best model SM42 with codebooks fitted
# after training (post-training VQ, the k-means file) and with int8 state, on the 68 users of the SM32b re-check.
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; WT=C:/Users/jonat/workspace/wt/sm32c; cd $WT
SM42=C:/Users/jonat/workspace/wt/sm42/artifacts/small-model/scale-w64kd-gate-cached-ctx32768-cont-decay1ep32k-lanes/expertium/E89-F65e-F77a/master.pth
O=$WT/artifacts/small-model/sm32c; U=$O/users68.json
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
echo "[sm42-ptq $(date '+%m-%d %H:%M:%S')] start"
$PY -X utf8 scripts/small_model_heldout.py --device cpu --users-file $U --time-limit 100000 --out $O/sm42-vq.jsonl --model sm42=expertium:w64:$SM42 --only sm42/w8+vq378x412 > $O/logs/sm42-vq.log 2>&1 &
$PY -X utf8 scripts/small_model_heldout.py --device cpu --users-file $U --time-limit 100000 --out $O/sm42-s8.jsonl --model sm42=expertium:w64:$SM42 --only sm42/w8+s8 > $O/logs/sm42-s8.log 2>&1 &
wait
echo "[sm42-ptq $(date '+%m-%d %H:%M:%S')] done"
