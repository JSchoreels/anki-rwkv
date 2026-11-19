#!/usr/bin/env bash
# Again pressed as Hard (SM132): ONE GPU ticket = the steps given, in order (run by launch-sm132.sh through
# scripts/r01_queue.py; chained in one ticket so the six-layer Tiny (SM130)'s ~110-min tickets do not sit between them).
#   bash sm132_ticket.sh CAP STEP...
#   smoke   users 2388, 4676: the show-time two-head cool-down for Small (SM118) at p = 0 and 20%, the show-time big
#           model (SM91s2) at p = 0, on the GPU -> gpu/smoke-gpu-rows.jsonl, then sm132_check.py smoke (GPU vs the
#           committed CPU rows within 1e-5, identical corruption counts, p = 0 vs the published rows within 1e-5);
#           exit 3 on a mismatch
#   sP      SM118 at p = P% (s0 = the reference), Anki card states;  s20g = SM118 at 20%, grade only (states kept)
#   bP      SM91s2 at p = P%
# All: 723 users of the 732 set, show-time inputs, leak-free (neighbour), w8+s8 (the published format), targets true;
# finished users are skipped (resume). One scorer process per step (each p is its own input batch).
# In-ticket retries (this ticket holds the GPU): a step whose label still misses users runs again once in a fresh process
# at CAP (an OOM after fragmentation), then once at cap 0.92 (memory only, same numerics).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
CAP=$1; shift
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm132
S=$WT/artifacts/small-model/sm132
G=$S/gpu
W=C:/Users/jonat/workspace/wt
SC=$WT/scripts/small_model_heldout.py
ROWS=$G/sm132-rows.jsonl
USERS732=$W/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json
TRACK=E89-F65e-F77a
CK118=$W/sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2/expertium/$TRACK/master.pth
CK91=$W/sm91s2/artifacts/small-model/sm91s2-big-showtime/expertium/big-showtime/master.pth
T118=sm118-e26-dec2ep32k-t91s2; T91=sm91s2-big-showtime
M118="$T118=expertium:w64:$CK118"; M91="$T91=fork::$CK91"
TLS=${SM132_TL_SMALL:-1500}; TLB=${SM132_TL_BIG:-2100}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$WT" || exit 1
say() { echo "[sm132-ticket $(date '+%m-%d %H:%M:%S')] $*" >> "$G/score.out"; }
run() { env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS HELDOUT_GPU_MEMORY_FRACTION=$CAP "$PY" -X utf8 "$SC" \
          --pool 732 --inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows "$@" >> "$G/score.out" 2>&1; }
missing() {  # missing LABEL: users of the 723 without a row for LABEL; unparseable lines skipped
  "$PY" -X utf8 -c "
import json, sys
got = set()
try:
    f = open(sys.argv[1], 'rb')
except OSError:
    f = []
for raw in f:
    try:
        r = json.loads(raw.decode('utf-8', errors='replace'))
        if r['label'] == sys.argv[2] and ('benchmark' in r or ('nonfinite' in r and 'error' not in r)):
            got.add(r['user_id'])
    except (ValueError, KeyError, TypeError, AttributeError):
        pass
print(723 - len(got))" "$ROWS" "$1"; }
# step -> (model, tag, p, extra flags, label suffix, time limit)
spec() {
  local st=$1
  case "$st" in
    s20g) echo "$M118 $T118 0.2 keep +aah20g $TLS";;
    s*) local p=${st#s}; echo "$M118 $T118 $(awk "BEGIN{print $p/100}") anki +aah$p $TLS";;
    b*) local p=${st#b}; echo "$M91 $T91 $(awk "BEGIN{print $p/100}") anki +aah$p $TLB";;
  esac
}
one() {  # one STEP: one scorer process
  set -- $(spec "$1")
  run --users-file "$USERS732" --model "$1" --only "$2/w8+s8" --again-as-hard "$3" --again-as-hard-state "$4" \
      --time-limit "$6" --out "$ROWS"
}
stepped() {  # stepped STEP: the step, then the in-ticket retries
  local st=$1 m rc keep label
  set -- $(spec "$st"); label="$2/w8+s8@e2s-query$5"
  one "$st" || return 1
  m=$(missing "$label"); [ "$m" = 0 ] && return 0
  say "$st: $m users missing, retry in a fresh process at cap $CAP"
  one "$st" || return 1
  m=$(missing "$label"); [ "$m" = 0 ] && return 0
  say "$st: $m users missing, retry at cap 0.92 (memory only, same numerics)"
  keep=$CAP; CAP=0.92; one "$st"; rc=$?; CAP=$keep
  say "$st: $(missing "$label") users missing after the retries"
  return $rc
}
for step in "$@"; do
  say "step $step (cap $CAP)"
  case "$step" in
    smoke)
      for a in "$M118 $T118 0" "$M118 $T118 0.2" "$M91 $T91 0"; do
        set -- $a
        run --users-file "$S/smoke-users.json" --model "$1" --only "$2/w8+s8" --again-as-hard "$3" \
          --time-limit 1200 --out "$G/smoke-gpu-rows.jsonl" || exit 1
      done
      "$PY" -X utf8 "$S/sm132_check.py" smoke >> "$G/score.out" 2>&1 || exit 3
      touch "$G/smoke-ok";;
    s0|s5|s10|s20|s30|s50|s20g|b10|b20) stepped "$step" || exit 1;;
    *) say "unknown step $step"; exit 2;;
  esac
done
say "ticket done"
exit 0
