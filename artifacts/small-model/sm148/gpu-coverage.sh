#!/usr/bin/env bash
# SM148 coverage: the 5 Big users never scored on the GPU (OOM at cap 0.92 with all 8 grid points in one process).
# One ticket (queue, cap 0.92), one fresh process per (model, label) so no cached memory carries over; same numerics.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm148; S=$WT/artifacts/small-model/sm148; G=$S/gpu
cd "$WT" || exit 1
. "$S/models.sh"; . "$S/labels.sh"
cat > "$G/cov-ticket-body.sh" <<'B'
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm148; S=$WT/artifacts/small-model/sm148; G=$S/gpu
cd "$WT"; . "$S/models.sh"; . "$S/labels.sh"
unset RWKV_DELETED_CARDS EXPERTIUM_END_TO_START SM11_INPUTS
export HELDOUT_GPU_MEMORY_FRACTION=0.92 PYTORCH_JIT=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while IFS='|' read -r tag ka path digest role size; do
  [ "$size" = big ] || continue
  for n in $GPU_LABELS; do
    "$PY" -X utf8 scripts/small_model_heldout.py --pool 732 --users-file "$G/users-cov5.json" --model "$tag=$ka:$path" --only "$tag/${!n}" \
      --inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets "$EDGES" --time-limit 3600 \
      --out "$G/sm148-rows-732.jsonl" >> "$G/score-cov.out" 2>&1
    echo "$tag $n exit $?" >> "$G/cov.log"
  done
done <<< "$(echo "$MODELS" | sed '/^$/d')"
B
"$PY" "$QUEUE" run --owner sm148 --purpose "Low-rank int8 states (SM148): coverage, the 5 Big users that ran out of GPU memory (one process per model and label, cap 0.92)" \
  --claim gpu --expected-minutes 40 --max-wait-minutes 60 --launched-by "detached sm148/gpu-coverage.sh" -- bash "$G/cov-ticket-body.sh" >> "$G/cov.log" 2>&1
echo "ticket exit $?" >> "$G/cov.log"; touch "$G/COV-DONE"
