#!/usr/bin/env bash
# SM148 coverage: rescore the long-card Big users killed by the memory guard, one fresh CPU process per (user, model),
# all 4 labels, same flags and dump as launch-sm148.sh lc_one. Splitting the labels would not lower the peak (memprobe.py:
# the peak is one label's forward pass), so a killed (user, model) stays missing. Limit stays 5.9 GB (memguard.ps1).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm148; S=$WT/artifacts/small-model/sm148; C=$S/cpu; CV=$C/cov
mkdir -p "$CV"; cd "$WT" || exit 1
. "$S/models.sh"; . "$S/labels.sh"
SC=$WT/scripts/small_model_heldout.py
DUMP=$WT/work/sm148-dump; LOG=$CV/cov.log; WORKERS=${WORKERS:-2}
SCORE_FLAGS=(--inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets "$EDGES")
log() { echo "[cov $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
job() {  # job USER TAG KA PATH
  local u=$1 tag=$2 ka=$3 path=$4 out=$CV/u-$1-$2.jsonl a=() n
  [ -s "$out" ] && return 0
  for n in $LONG_LABELS; do a+=(--only "$tag/${!n}"); done
  echo "{\"users\": [$u]}" > "$CV/users-$u.json"; rm -f "$out.part" "$DUMP/${u}_${tag}_"*.npz
  env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS -u HELDOUT_GPU_MEMORY_FRACTION CUDA_VISIBLE_DEVICES= PYTORCH_JIT=0 "$PY" -X utf8 -W ignore "$SC" \
    --device cpu --workers 1 --pool 732 --users-file "$CV/users-$u.json" --model "$tag=$ka:$path" "${a[@]}" "${SCORE_FLAGS[@]}" --time-limit 1000000 \
    --dump "$DUMP" --dump-scope all --out "$out.part" >> "$CV/score-$u-$tag.out" 2>&1
  local rc=$?
  if [ $rc = 0 ]; then mv "$out.part" "$out"; log "user $u $tag done: $(grep -a '^\[small-model\] 1/1' "$CV/score-$u-$tag.out" | tail -1 | cut -c1-110)"
  else log "user $u $tag FAILED rc $rc (memory guard kill if listed in memguard-kill)"; rm -f "$out.part" "$CV/STOP" "$DUMP/${u}_${tag}_"*.npz; fi
}
pids=()
wait_slot() { local p a=(); for p in "${pids[@]:-}"; do :; done
  while true; do a=(); for p in "${pids[@]:-}"; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && a+=("$p"); done; pids=("${a[@]:-}")
    [ "${#a[@]}" -lt "$1" ] && return 0; sleep 10; done; }
for u in $(cat "$CV/users.txt"); do
  while IFS='|' read -r tag ka path digest role size; do
    [ "$size" = big ] || continue
    wait_slot "$WORKERS"
    job "$u" "$tag" "$ka" "$path" & pids+=($!)
    sleep 15
  done <<< "$(echo "$MODELS" | sed '/^$/d')"
done
wait_slot 1; wait
log "all jobs ended"; touch "$CV/DONE"
