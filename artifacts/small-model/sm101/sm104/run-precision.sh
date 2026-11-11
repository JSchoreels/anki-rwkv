#!/bin/bash
# The tiny model (SM101) vs the show-time instant-only model, stage 2 (SM98): f32 / int8 / int4-card+int8-note state,
# f32 weights, leak-free (neighbour), show-time inputs (e2s-query), CPU, 80 random users of the 732 set (SM78 sample).
# 4 processes x 2 threads. Run from the worktree root.
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
D=artifacts/small-model/sm101/sm104
T101=C:/Users/jonat/workspace/wt/sm101/artifacts/small-model/sm101-c1-e38st-st-dec-1ep32k-muon/expertium/E89-F65e-F77a/master.pth
T98=C:/Users/jonat/workspace/wt/sm98/artifacts/small-model/sm98-inst-e35st-st-dec-1ep32k-muon/expertium/E89-F65e-F77a/master.pth
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 "$PY" -X utf8 scripts/small_model_heldout.py --device cpu --pool 732 \
    --users-file $D/users$i.json --model sm101=expertium:w64-c1:$T101 --model sm98=expertium:w64:$T98 \
    --variants fp32,fp32+s8,fp32+s4c8n --deleted-cards neighbour --inputs e2s-query --time-limit 200000 \
    --out $D/scores$i.jsonl > $D/log$i.txt 2>&1 &
done
wait
echo done
