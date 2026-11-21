#!/usr/bin/env bash
# Release-candidate checks for the family (SM139): CPU proof rows (run once before any GPU ticket).
#   the 2 smoke users (users-smoke.json), every label of the run (6 RC models x {w8+s8, w8+s8hx8} + 4 published x w8+s8),
#   CPU, show-time inputs, leak-free (neighbour), --nonfinite-rows -> cpu/smoke-cpu-rows.jsonl (the GPU smoke reference)
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm139
S=$WT/artifacts/small-model/sm139
. "$S/models.sh"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
args=()
while IFS='|' read -r tag ka path sha role size; do
  [ -z "$tag" ] && continue
  args+=(--model "$tag=$ka:$path" --only "$tag/w8+s8")
  [ "$role" = rc ] && args+=(--only "$tag/w8+s8hx8")
done <<< "$MODELS"
cd "$WT" || exit 1
env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS -u HELDOUT_GPU_MEMORY_FRACTION "$PY" -X utf8 scripts/small_model_heldout.py \
  --device cpu --workers 1 --pool 732 --users-file "$S/users-smoke.json" "${args[@]}" --inputs e2s-query --split-existing \
  --deleted-cards neighbour --nonfinite-rows --out "$S/cpu/smoke-cpu-rows.jsonl"
