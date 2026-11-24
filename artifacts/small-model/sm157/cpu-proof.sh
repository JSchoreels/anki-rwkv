#!/usr/bin/env bash
# Output-aware low-rank truncation (SM157): CPU proofs before the probe run (design.md "Proofs").
#   1 the 3 smoke users x the 6 release models x REF, Q2, T24 (labels.sh SMOKE_LABELS): must equal SM148's committed CPU
#     smoke rows (the new code leaves every existing label unchanged), two processes (Big; Small + Tiny)
#   2 pipeline check: user 2388 x the 6 models x PIPE_LABELS with the test statistics (2 training users,
#     work/sm157-stats-test.json, never reported): runs on the fork and expertium paths, differs from plain
#   3 sm157_check.py cpu -> check-cpu-sm157.json
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm157
S=$WT/artifacts/small-model/sm157
. "$S/models.sh"; . "$S/labels.sh"
mkdir -p "$S/cpu"
run() { env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS -u HELDOUT_GPU_MEMORY_FRACTION \
  PYTORCH_JIT=0 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PY" -X utf8 -W ignore \
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
rm -f "$S/cpu/smoke-cpu-rows"*.jsonl "$S/cpu/pipe-cpu-rows.jsonl"
mapfile -t A < <(grid "big" "$SMOKE_LABELS"); mapfile -t B < <(grid "small tiny" "$SMOKE_LABELS")
run --users-file "$S/users-smoke.json" "${A[@]}" --out "$S/cpu/smoke-cpu-rows-big.jsonl" > "$S/cpu/smoke-cpu-big.out" 2>&1 & pa=$!
run --users-file "$S/users-smoke.json" "${B[@]}" --out "$S/cpu/smoke-cpu-rows-st.jsonl" > "$S/cpu/smoke-cpu-st.out" 2>&1 & pb=$!
wait $pa; ra=$?; wait $pb; rb=$?
[ $ra -ne 0 ] || [ $rb -ne 0 ] && { echo "smoke rows failed ($ra, $rb)"; exit 1; }
cat "$S/cpu/smoke-cpu-rows-big.jsonl" "$S/cpu/smoke-cpu-rows-st.jsonl" > "$S/cpu/smoke-cpu-rows.jsonl"
rm -f "$S/cpu/smoke-cpu-rows-big.jsonl" "$S/cpu/smoke-cpu-rows-st.jsonl"
mapfile -t P < <(grid "big small tiny" "$PIPE_LABELS")
SM157_STATS="$WT/work/sm157-stats-test.json" run --users-file "$S/users-pipe.json" "${P[@]}" \
  --out "$S/cpu/pipe-cpu-rows.jsonl" > "$S/cpu/pipe-cpu.out" 2>&1 || { echo "pipeline rows failed"; exit 1; }
"$PY" -X utf8 -W ignore "$S/sm157_check.py" cpu
