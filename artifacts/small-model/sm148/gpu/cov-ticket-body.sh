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
