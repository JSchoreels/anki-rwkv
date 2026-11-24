#!/usr/bin/env bash
# Energy-triggered rank growth (SM159): the CPU pointer chain (design.md "Run"), launched detached (HANDOFF.md).
#   0 selftests  state_lowrank (SM147 / SM148 / SM157 / SM159), sm159_bytes, sm159_report -> cpu/selftest.out
#   1 calib      sm159_calib.py on the 20 training users (users-calib.json) x the 6 models x TRAIN_GRID (bytes only),
#                three processes (Big two-head; Big Instant-Only; Small + Tiny) -> work/sm159-calib/calib-*.json,
#                merged into cpu/calib-train.json
#   2 select     sm159_report.py select -> sm159-taus.json (matched code per model)
#   3 probe      the 8 probe users x the 6 models x the probe grid (sm159_report.py labels), three processes (same
#                split) -> cpu/sm159-rows-{a,b,c}.jsonl -> cpu/sm159-rows.jsonl
#   4 fallback   sm159_report.py fallback (bytes only): extra probe-matched labels, scored the same way
#   5 report     sm159_report.py report -> sm159-result.{json,md}
# Progress and the last line ("SM159 RUN DONE" / "SM159 RUN FAILED ...") in cpu/run.out. Memory guard 5.9 GB per
# process (memguard.ps1). Finished (model, label) pairs / (user, label) rows are skipped on a rerun.
# stop: touch cpu/STOP (checked between steps); kill the python processes for an immediate stop.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm159
S=$WT/artifacts/small-model/sm159
C=$S/cpu
F=$WT/work/sm159-calib
. "$S/models.sh"
EDGES=0,1,3,5,9,16,32,128,300,1000
TRAIN_GRID="800 1000 1250 1500 1800 2200"
mkdir -p "$C" "$F"
LOG=$C/run.out
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
fail() { log "SM159 RUN FAILED ($*)"; touch "$C/memguard.off"; exit 1; }
stopcheck() { [ -e "$C/STOP" ] && fail "STOP file"; [ -e "$C/memguard-kill" ] && fail "memory guard kill"; return 0; }
ENVX=(env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS -u HELDOUT_GPU_MEMORY_FRACTION PYTORCH_JIT=0
      CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=3 MKL_NUM_THREADS=3)
group_of() { case "$1" in sm91s2*) echo a;; sm99*) echo b;; *) echo c;; esac; }
models() {   # models GROUP (a | b | c) -> --model args
  local out=()
  while IFS='|' read -r tag ka path sha role size; do
    [ -z "$tag" ] && continue
    [ "$(group_of "$tag")" = "$1" ] && out+=(--model "$tag=$ka:$path")
  done <<< "$MODELS"
  printf '%s\n' "${out[@]}"
}
keys() { case "$1" in a) echo sm91s2;; b) echo sm99;; c) echo sm118,sm113,sm130th,sm130io;; esac; }
cd "$WT" || exit 1
rm -f "$C/memguard.off"
powershell -NoProfile -ExecutionPolicy Bypass -File "$S/memguard.ps1" -Dir "$(cygpath -w "$C")" \
  -Match "sm159_calib.py;small_model_heldout.py&sm159" -LimitGB 5.9 &
log "start $(git rev-parse HEAD) dirty=$(git status --porcelain scripts | wc -l)"
# step 0: selftests
"$PY" -X utf8 -c "import sys; sys.path.insert(0, '.'); from scripts import state_lowrank as s; s.selftest(); s.selftest_sm148(); s.selftest_sm157(); s.selftest_sm159()" > "$C/selftest.out" 2>&1 || fail selftest-codec
"$PY" -X utf8 "$S/sm159_bytes.py" >> "$C/selftest.out" 2>&1 || fail selftest-bytes
"$PY" -X utf8 "$S/sm159_report.py" selftest >> "$C/selftest.out" 2>&1 || fail selftest-report
log "selftests OK"
# step 1: calibration on the training users (bytes only)
calib() {
  local g=$1; mapfile -t M < <(models "$g"); local L=()
  for c in $TRAIN_GRID; do L+=(--label "w8+lr2q8x8g4e$c"); done
  "${ENVX[@]}" "$PY" -X utf8 -W ignore "$S/sm159_calib.py" "${M[@]}" "${L[@]}" --users-file "$S/users-calib.json" \
    --training --threads 3 --out "$F/calib-$g.json" >> "$C/calib-$g.out" 2>&1
}
if [ ! -e "$C/calib-train.json" ]; then
  calib a & p1=$!; calib b & p2=$!; calib c & p3=$!
  wait $p1; r1=$?; wait $p2; r2=$?; wait $p3; r3=$?
  [ $r1 -ne 0 ] || [ $r2 -ne 0 ] || [ $r3 -ne 0 ] && fail "calib $r1 $r2 $r3"
  "$PY" -X utf8 -c "
import json, sys
out = {'models': {}, 'provenance': {}, 'note': 'SM159 rank read-out on the 20 training users (aggregates only)'}
for g in 'abc':
    d = json.load(open(sys.argv[1] + '/calib-' + g + '.json'))
    out['models'].update(d['models']); out['provenance'].update(d['provenance'])
json.dump(out, open(sys.argv[2], 'w'), indent=1)" "$F" "$C/calib-train.json" || fail calib-merge
  log "calib done"
fi
stopcheck
# step 2: thresholds
"$PY" -X utf8 "$S/sm159_report.py" select >> "$LOG" 2>&1 || fail select
stopcheck
# step 3: probe scoring
run() { "${ENVX[@]}" "$PY" -X utf8 -W ignore scripts/small_model_heldout.py \
  --device cpu --workers 1 --pool 732 --inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows \
  --prior-buckets "$EDGES" --time-limit 100000 --users-file "$S/users-probe.json" "$@"; }
probe() {   # probe GROUP OUTSUFFIX ONLY-LINES...
  local g=$1 suf=$2; shift 2
  [ $# -eq 0 ] && return 0
  mapfile -t M < <(models "$g"); local O=()
  for x in "$@"; do O+=(--only "$x"); done
  run "${M[@]}" "${O[@]}" --out "$C/sm159-rows-$g$suf.jsonl" >> "$C/probe-$g$suf.out" 2>&1
}
for g in a b c; do mapfile -t "ONLY_$g" < <("$PY" -X utf8 "$S/sm159_report.py" labels --group "$(keys $g)"); done
log "probe grid: a ${#ONLY_a[@]} b ${#ONLY_b[@]} c ${#ONLY_c[@]} (model, label) pairs"
probe a "" "${ONLY_a[@]}" & pa=$!; probe b "" "${ONLY_b[@]}" & pb=$!; probe c "" "${ONLY_c[@]}" & pc=$!
wait $pa; ra=$?; wait $pb; rb=$?; wait $pc; rc=$?
[ $ra -ne 0 ] || [ $rb -ne 0 ] || [ $rc -ne 0 ] && fail "probe $ra $rb $rc"
cat "$C/sm159-rows-a.jsonl" "$C/sm159-rows-b.jsonl" "$C/sm159-rows-c.jsonl" > "$C/sm159-rows.jsonl"
log "probe done"
stopcheck
# step 4: fallback (bytes only)
mapfile -t EXTRA < <("$PY" -X utf8 "$S/sm159_report.py" fallback 2>> "$LOG") || fail fallback
log "fallback: ${#EXTRA[@]} extra (model, label) pairs ${EXTRA[*]:-}"
if [ ${#EXTRA[@]} -gt 0 ]; then
  XA=(); XB=(); XC=()
  for x in "${EXTRA[@]}"; do case "$(group_of "${x%%/*}")" in a) XA+=("$x");; b) XB+=("$x");; c) XC+=("$x");; esac; done
  probe a -x "${XA[@]}" & pa=$!; probe b -x "${XB[@]}" & pb=$!; probe c -x "${XC[@]}" & pc=$!
  wait $pa; ra=$?; wait $pb; rb=$?; wait $pc; rc=$?
  [ $ra -ne 0 ] || [ $rb -ne 0 ] || [ $rc -ne 0 ] && fail "fallback probe $ra $rb $rc"
  cat "$C/sm159-rows-a.jsonl" "$C/sm159-rows-b.jsonl" "$C/sm159-rows-c.jsonl" $(ls "$C"/sm159-rows-*-x.jsonl 2>/dev/null) \
    > "$C/sm159-rows.jsonl"
  log "fallback scoring done"
fi
touch "$C/memguard.off"
# step 5: report
"$PY" -X utf8 -W ignore "$S/sm159_report.py" report >> "$LOG" 2>&1; rr=$?
[ $rr -ne 0 ] && fail "report exit $rr"
log "SM159 RUN DONE"
