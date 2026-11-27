#!/usr/bin/env bash
# Today's models on donor data (SM178): GPU smoke, two queue tickets (run by hand, then sm178_report.py smoke):
#   a) six-layer Tiny two-head (SM130) at w8+s8 on SM177's 20 CPU smoke users (must reproduce 0.334454 / 0.323671)
#   b) the six models at REF (w8+s8hx8) on 3 of those users (rows finite, sane)
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
S=C:/Users/jonat/workspace/wt/sm178/artifacts/small-model/sm178
G=$S/gpu; RUN=C:/Users/jonat/workspace/wt/sm178/work/sm178/run-smoke
mkdir -p "$G"
. "$S/models.sh"
NAME="Today's models on donor data (SM178)"
ENVS=(env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS -u SM157_STATS PYTORCH_JIT=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4)
FLAGS=(--inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets 0,1,3,5,9,16,32,128,300,1000)
GARGS=()
while IFS='|' read -r tag ka path digest role size; do
  [ -z "$tag" ] && continue
  GARGS+=(--model "$tag=$ka:$path" --only "$tag/$REF")
  [ "$tag" = sm130-th-dec2ep32k-t91s2 ] && TINY="$tag=$ka:$path"
done <<< "$(printf '%s\n' "$MODELS" | tr -d '\r')"
"$PY" "$QUEUE" run --owner sm178 --purpose "$NAME: GPU smoke a, six-layer Tiny two-head w8+s8 on SM177's 20 smoke users" \
  --claim gpu --expected-minutes 5 --max-wait-minutes 60 --launched-by "agent sm178 smoke.sh" -- \
  "${ENVS[@]}" HELDOUT_GPU_MEMORY_FRACTION=0.8 "$PY" -X utf8 -W ignore "$S/sm178_score.py" score "$RUN" -- \
  --users-file "$S/users-sm177-smoke.json" --model "$TINY" --only sm130-th-dec2ep32k-t91s2/w8+s8 "${FLAGS[@]}" \
  --time-limit 900 --out "$G/smoke-rows.jsonl" >> "$G/smoke.out" 2>&1 || { echo "smoke a failed"; exit 1; }
"$PY" "$QUEUE" run --owner sm178 --purpose "$NAME: GPU smoke b, 6 models REF on 3 donor users" \
  --claim gpu --expected-minutes 5 --max-wait-minutes 60 --launched-by "agent sm178 smoke.sh" -- \
  "${ENVS[@]}" HELDOUT_GPU_MEMORY_FRACTION=0.8 "$PY" -X utf8 -W ignore "$S/sm178_score.py" score "$RUN" -- \
  --users-file "$S/users-smoke3.json" "${GARGS[@]}" "${FLAGS[@]}" \
  --time-limit 900 --out "$G/smoke-rows.jsonl" >> "$G/smoke.out" 2>&1 || { echo "smoke b failed"; exit 1; }
echo "smoke tickets done"
