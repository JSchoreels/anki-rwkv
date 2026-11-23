#!/usr/bin/env bash
# Smaller shift vectors and factors on T24 states (SM152): CPU proof rows (run once before any GPU ticket; design.md
# "Proofs").
#   1 the 3 smoke users (users-smoke.json), the 6 release-candidate models x SMOKE_LABELS, CPU (Hooks / patched fork
#     path, PYTORCH_JIT=0), show-time inputs, leak-free, --nonfinite-rows, --prior-buckets, --dump-scope all into
#     work/sm152-dump-smoke -> cpu/smoke-cpu-rows.jsonl (also the GPU smoke reference)
#   2 sm152_check.py cpu -> check-cpu-sm152.json
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm152
S=$WT/artifacts/small-model/sm152
DUMP=$WT/work/sm152-dump-smoke
. "$S/models.sh"; . "$S/labels.sh"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$S/cpu"
args=()
while IFS='|' read -r tag ka path sha role size; do
  [ -z "$tag" ] && continue
  args+=(--model "$tag=$ka:$path")
  for n in $SMOKE_LABELS; do args+=(--only "$tag/${!n}"); done
done <<< "$MODELS"
cd "$WT" || exit 1
rm -rf "$DUMP"; rm -f "$S/cpu/smoke-cpu-rows.jsonl"
env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS -u HELDOUT_GPU_MEMORY_FRACTION CUDA_VISIBLE_DEVICES= PYTORCH_JIT=0 \
  "$PY" -X utf8 -W ignore scripts/small_model_heldout.py --device cpu --workers 1 --pool 732 --users-file "$S/users-smoke.json" \
  --inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets "$EDGES" --time-limit 100000 \
  "${args[@]}" --dump "$DUMP" --dump-scope all --out "$S/cpu/smoke-cpu-rows.jsonl" > "$S/cpu/smoke-cpu.out" 2>&1 \
  || { echo "smoke rows failed"; exit 1; }
"$PY" -X utf8 -W ignore "$S/sm152_check.py" cpu
