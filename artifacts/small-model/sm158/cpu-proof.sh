#!/usr/bin/env bash
# Output-aware T24 on the 732 set (SM158): CPU proofs before the launch (design.md "Proofs before the launch").
#   1 the 3 smoke users x the 6 release models x REF, Q2, T24, Q2RK, T24RK on the CPU, two processes (Big; Small + Tiny),
#     SM157_STATS = ../sm157/sm157-stats.json (SM157's fit, not refit) -> cpu/smoke-cpu-rows.jsonl
#   2 sm158_check.py cpu -> check-cpu-sm158.json: REF, Q2, T24 == SM148's committed CPU smoke rows; Q2RK, T24RK ==
#     SM157's committed CPU probe rows of the same users (bit for bit)
# The rows of step 1 are also the CPU reference of the GPU smoke ticket (launch-sm158-gpu.sh).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm158
S=$WT/artifacts/small-model/sm158
STATS=$WT/artifacts/small-model/sm157/sm157-stats.json
. "$S/models.sh"; . "$S/labels.sh"
mkdir -p "$S/cpu"
run() { env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS -u HELDOUT_GPU_MEMORY_FRACTION \
  PYTORCH_JIT=0 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 SM157_STATS="$STATS" "$PY" -X utf8 -W ignore \
  scripts/small_model_heldout.py --device cpu --workers 1 --pool 732 --inputs e2s-query --split-existing \
  --deleted-cards neighbour --nonfinite-rows --prior-buckets "$EDGES" --time-limit 100000 "$@"; }
grid() {   # grid SIZES LABEL-NAMES -> --model/--only args
  local sizes=$1 names=$2 out=()
  while IFS='|' read -r tag ka path sha role size; do
    [ -z "$tag" ] && continue
    case " $sizes " in *" $size "*) ;; *) continue;; esac
    out+=(--model "$tag=$ka:$path"); for n in $names; do out+=(--only "$tag/${!n}"); done
  done <<< "$MODELS"
  printf '%s\n' "${out[@]}"
}
cd "$WT" || exit 1
echo "[sm158 cpu-proof] start $(date '+%m-%d %H:%M:%S') $(git log --oneline -1 | cut -c1-60)"
rm -f "$S/cpu/smoke-cpu-rows"*.jsonl
mapfile -t A < <(grid "big" "$SMOKE_OLD $SMOKE_NEW"); mapfile -t B < <(grid "small tiny" "$SMOKE_OLD $SMOKE_NEW")
run --users-file "$S/users-smoke.json" "${A[@]}" --out "$S/cpu/smoke-cpu-rows-big.jsonl" > "$S/cpu/smoke-cpu-big.out" 2>&1 & pa=$!
run --users-file "$S/users-smoke.json" "${B[@]}" --out "$S/cpu/smoke-cpu-rows-st.jsonl" > "$S/cpu/smoke-cpu-st.out" 2>&1 & pb=$!
wait $pa; ra=$?; wait $pb; rb=$?
[ $ra -ne 0 ] || [ $rb -ne 0 ] && { echo "smoke rows failed ($ra, $rb)"; exit 1; }
cat "$S/cpu/smoke-cpu-rows-big.jsonl" "$S/cpu/smoke-cpu-rows-st.jsonl" > "$S/cpu/smoke-cpu-rows.jsonl"
rm -f "$S/cpu/smoke-cpu-rows-big.jsonl" "$S/cpu/smoke-cpu-rows-st.jsonl"
"$PY" -X utf8 -W ignore "$S/sm158_check.py" cpu
rc=$?
echo "[sm158 cpu-proof] end $(date '+%m-%d %H:%M:%S') rc=$rc"
exit $rc
