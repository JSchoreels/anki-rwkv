#!/usr/bin/env bash
# Output-aware low-rank truncation (SM157): the CPU pointer chain (design.md "Run"), launched detached (HANDOFF.md).
#   1 fit     sm157_fit.py on the training users, half A then half B, two processes (two-head models; Instant-Only
#             models) -> work/sm157-fit/stats-{a,b}-{two,io}.json; merge -> sm157-stats-a.json, sm157-stats-b.json and
#             sm157-stats.json (all 100 users; the file the weighted labels read, SM157_STATS)
#   2 probe   the 8 probe users (users-probe.json) x the 6 models x PROBE_LABELS, two processes (same split by model)
#             -> cpu/sm157-rows-{two,io}.jsonl -> cpu/sm157-rows.jsonl
#   3 report  sm157_report.py report -> sm157-result.{json,md}
# Progress and the last line ("SM157 RUN DONE" / "SM157 RUN FAILED ...") in cpu/run.out. Memory guard 5.9 GB per
# process (memguard.ps1). Finished models / users are skipped on a rerun (fit: per model; scorer: per user and label).
# stop: touch cpu/STOP (checked between steps); kill the python processes for an immediate stop.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm157
S=$WT/artifacts/small-model/sm157
C=$S/cpu
F=$WT/work/sm157-fit
. "$S/models.sh"; . "$S/labels.sh"
mkdir -p "$C" "$F"
LOG=$C/run.out
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
fail() { log "SM157 RUN FAILED ($*)"; touch "$C/memguard.off"; exit 1; }
stopcheck() { [ -e "$C/STOP" ] && fail "STOP file"; [ -e "$C/memguard-kill" ] && fail "memory guard kill"; return 0; }
ENVX=(env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS -u HELDOUT_GPU_MEMORY_FRACTION PYTORCH_JIT=0
      CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4)
models() {   # models GROUP (two | io) -> --model args
  local out=()
  while IFS='|' read -r tag ka path sha role size; do
    [ -z "$tag" ] && continue
    case "$tag" in *inst*|*-io-*) g=io;; *) g=two;; esac
    [ "$g" = "$1" ] && out+=(--model "$tag=$ka:$path")
  done <<< "$MODELS"
  printf '%s\n' "${out[@]}"
}
only() {   # only GROUP -> --model/--only args of PROBE_LABELS
  local out=()
  while IFS='|' read -r tag ka path sha role size; do
    [ -z "$tag" ] && continue
    case "$tag" in *inst*|*-io-*) g=io;; *) g=two;; esac
    [ "$g" = "$1" ] || continue
    out+=(--model "$tag=$ka:$path"); for n in $PROBE_LABELS; do out+=(--only "$tag/${!n}"); done
  done <<< "$MODELS"
  printf '%s\n' "${out[@]}"
}
cd "$WT" || exit 1
rm -f "$C/memguard.off"
powershell -NoProfile -ExecutionPolicy Bypass -File "$S/memguard.ps1" -Dir "$(cygpath -w "$C")" \
  -Match "sm157_fit.py;small_model_heldout.py&sm157" -LimitGB 5.9 &
log "start $(git rev-parse HEAD) dirty=$(git status --porcelain scripts | wc -l)"
# step 1: fit (two processes; each runs half A then half B for its models)
fitgroup() {
  local g=$1; mapfile -t M < <(models "$g")
  for h in a b; do
    "${ENVX[@]}" "$PY" -X utf8 -W ignore "$S/sm157_fit.py" "${M[@]}" --users-file "$S/users-fit-$h.json" \
      --out "$F/stats-$h-$g.json" >> "$C/fit-$g.out" 2>&1 || return 1
  done
}
if [ ! -e "$S/sm157-stats.json" ]; then
  fitgroup two & p1=$!; fitgroup io & p2=$!
  wait $p1; r1=$?; wait $p2; r2=$?
  [ $r1 -ne 0 ] || [ $r2 -ne 0 ] && fail "fit $r1 $r2"
  stopcheck
  "$PY" -X utf8 "$S/sm157_fit.py" --merge "$F/stats-a-two.json" "$F/stats-a-io.json" --out "$S/sm157-stats-a.json" || fail merge-a
  "$PY" -X utf8 "$S/sm157_fit.py" --merge "$F/stats-b-two.json" "$F/stats-b-io.json" --out "$S/sm157-stats-b.json" || fail merge-b
  "$PY" -X utf8 "$S/sm157_fit.py" --merge "$S/sm157-stats-a.json" "$S/sm157-stats-b.json" --out "$S/sm157-stats.json" || fail merge
  log "fit done"
fi
stopcheck
# step 2: probe scoring
run() { "${ENVX[@]}" SM157_STATS="$S/sm157-stats.json" "$PY" -X utf8 -W ignore scripts/small_model_heldout.py \
  --device cpu --workers 1 --pool 732 --inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows \
  --prior-buckets "$EDGES" --time-limit 100000 --users-file "$S/users-probe.json" "$@"; }
mapfile -t A < <(only two); mapfile -t B < <(only io)
run "${A[@]}" --out "$C/sm157-rows-two.jsonl" > "$C/probe-two.out" 2>&1 & pa=$!
run "${B[@]}" --out "$C/sm157-rows-io.jsonl" > "$C/probe-io.out" 2>&1 & pb=$!
wait $pa; ra=$?; wait $pb; rb=$?
[ $ra -ne 0 ] || [ $rb -ne 0 ] && fail "probe $ra $rb"
stopcheck
cat "$C/sm157-rows-two.jsonl" "$C/sm157-rows-io.jsonl" > "$C/sm157-rows.jsonl"
log "probe done"
touch "$C/memguard.off"
"$PY" -X utf8 -W ignore "$S/sm157_report.py" report >> "$LOG" 2>&1; rr=$?
[ $rr -ne 0 ] && fail "report exit $rr"
log "SM157 RUN DONE"
