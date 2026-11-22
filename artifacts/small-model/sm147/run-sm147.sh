#!/usr/bin/env bash
# Low-rank structure of the card and note states (SM147): CPU run (design.md "Run"). Writes cpu/*.jsonl, logs cpu/*.out,
# "SM147 RUN DONE" (or "SM147 RUN FAILED") as the last line of cpu/run.out.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm147
S=$WT/artifacts/small-model/sm147
. "$S/models.sh"
EDGES=0,1,3,5,9,16,32,128,300,1000
cd "$WT" || exit 1
mkdir -p "$S/cpu"
run() { env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS -u HELDOUT_GPU_MEMORY_FRACTION \
  PYTORCH_JIT=0 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PY" -X utf8 -W ignore \
  scripts/small_model_heldout.py --device cpu --workers 1 --pool 732 --inputs e2s-query --split-existing \
  --deleted-cards neighbour --nonfinite-rows --prior-buckets "$EDGES" --time-limit 100000 "$@"; }
grid() {   # grid SIZES LABELS -> --model/--only args
  local sizes=$1 labels=$2 out=()
  while IFS='|' read -r tag ka path sha role size; do
    [ -z "$tag" ] && continue
    case " $sizes " in *" $size "*) ;; *) continue;; esac
    out+=(--model "$tag=$ka:$path"); for v in $labels; do out+=(--only "$tag/$v"); done
  done <<< "$MODELS"
  printf '%s\n' "${out[@]}"
}
LABELS="w8+lrsv w8+s8hx8 w8+lr1 w8+lr2 w8+lr4"
powershell -NoProfile -ExecutionPolicy Bypass -File "$S/memguard.ps1" -Dir "$(cygpath -w "$S/cpu")" \
  -Match "small_model_heldout.py&sm147" -LimitGB 5.9 &
echo "$(date '+%m-%d %H:%M:%S') start $(git rev-parse HEAD) dirty=$(git status --porcelain scripts | wc -l)"
# step 1: smoke proof rows (C2, C3)
mapfile -t a < <(grid "big small tiny" "w8 w8+lrsv w8+s8hx8")
run --users-file "$S/users-smoke.json" "${a[@]}" --out "$S/cpu/smoke-rows.jsonl" > "$S/cpu/smoke.out" 2>&1 \
  || { echo "SM147 RUN FAILED (smoke)"; touch "$S/cpu/memguard.off"; exit 1; }
echo "$(date '+%m-%d %H:%M:%S') smoke done"
# step 2: 8 users x 6 models x 5 labels, two processes
mapfile -t A < <(grid "big" "$LABELS"); mapfile -t B < <(grid "small tiny" "$LABELS")
run --users-file "$S/users.json" "${A[@]}" --out "$S/cpu/rows-big.jsonl" > "$S/cpu/rows-big.out" 2>&1 & pa=$!
run --users-file "$S/users.json" "${B[@]}" --out "$S/cpu/rows-small-tiny.jsonl" > "$S/cpu/rows-small-tiny.out" 2>&1 & pb=$!
wait $pa; ra=$?; wait $pb; rb=$?
touch "$S/cpu/memguard.off"
echo "$(date '+%m-%d %H:%M:%S') rows done (big $ra, small/tiny $rb)"
if [ $ra -ne 0 ] || [ $rb -ne 0 ] || [ -e "$S/cpu/memguard-kill" ]; then echo "SM147 RUN FAILED"; exit 1; fi
cat "$S/cpu/rows-big.jsonl" "$S/cpu/rows-small-tiny.jsonl" > "$S/cpu/sm147-rows.jsonl"
echo "SM147 RUN DONE"
