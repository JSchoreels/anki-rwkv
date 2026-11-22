#!/usr/bin/env bash
# Tiered card state by answer count (SM144): CPU proof rows (run once before any GPU ticket; design.md "Proofs").
#   1 the 3 smoke users (users-smoke.json), the 6 release-candidate models x SMOKE_LABELS, CPU (Hooks / patched fork
#     path), show-time inputs, leak-free, --nonfinite-rows, --prior-buckets -> cpu/smoke-cpu-rows.jsonl (also the GPU
#     smoke reference)
#   2 the 4 expertium models (Small, Tiny) x REF U8 A16 AN16 with --emulate-gpu-path (the GPU code path's CPU twin)
#     -> cpu/emulate-rows.jsonl
#   3 sm144_check.py cpu -> check-cpu-sm144.json
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm144
S=$WT/artifacts/small-model/sm144
. "$S/models.sh"; . "$S/labels.sh"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$S/cpu"
args=(); eargs=()
while IFS='|' read -r tag ka path sha role size; do
  [ -z "$tag" ] && continue
  args+=(--model "$tag=$ka:$path")
  for n in $SMOKE_LABELS; do args+=(--only "$tag/${!n}"); done
  if [ "${ka%%:*}" = expertium ]; then eargs+=(--model "$tag=$ka:$path"); for n in REF U8 A16 AN16; do eargs+=(--only "$tag/${!n}"); done; fi
done <<< "$MODELS"
cd "$WT" || exit 1
run() { env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS -u HELDOUT_GPU_MEMORY_FRACTION CUDA_VISIBLE_DEVICES= "$PY" -X utf8 -W ignore \
  scripts/small_model_heldout.py --device cpu --workers 1 --pool 732 --users-file "$S/users-smoke.json" --inputs e2s-query --split-existing \
  --deleted-cards neighbour --nonfinite-rows --prior-buckets "$EDGES" --time-limit 100000 "$@"; }
rm -f "$S/cpu/smoke-cpu-rows.jsonl" "$S/cpu/emulate-rows.jsonl"
run "${args[@]}" --out "$S/cpu/smoke-cpu-rows.jsonl" > "$S/cpu/smoke-cpu.out" 2>&1 || { echo "smoke rows failed"; exit 1; }
run "${eargs[@]}" --emulate-gpu-path --out "$S/cpu/emulate-rows.jsonl" > "$S/cpu/emulate.out" 2>&1 || { echo "emulate rows failed"; exit 1; }
"$PY" -X utf8 -W ignore "$S/sm144_check.py" cpu
