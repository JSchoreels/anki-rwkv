#!/usr/bin/env bash
# Long-card state drift on real users (SM136): ONE GPU ticket = the steps given, in order (run by launch-sm136.sh through
# scripts/r01_queue.py; chained so the six-layer Tiny with longer recovery (SM135)'s tickets do not sit between them).
#   bash sm136_ticket.sh CAP STEP...     STEP = smoke | big | small | tiny | ext
#   smoke: users 1866, 2883, the six models x five formats on the GPU -> gpu/smoke-gpu-rows.jsonl, then
#          sm136_check.py smoke (GPU vs the committed CPU rows); exit 3 on a mismatch
#   big:   the show-time big model (SM91s2) + the show-time instant-only big model (SM99)
#   small: the show-time two-head cool-down for Small (SM118) + the longer Small Instant-Only model (SM113)
#   tiny:  the show-time two-head cool-down for Tiny (SM124) + the longer cool-down for Tiny (SM115)
#   ext:   the six models on the extension pool (users-sm136-ext.json, --pool 1500, report only) -> gpu/sm136-ext-rows.jsonl
# Formats: w8+s8, w8+s4c8n, w8+s4c8nhx8, w8+s8hx8, w8 (float32 state). Main pool: users-sm136.json (66 users of the 732
#   set with a card of >= 128 prior answers). Show-time inputs, leak-free (neighbour), --nonfinite-rows (an overflowing
#   int4 card state is a 'nonfinite' row, done, not retried), --prior-buckets 0,32,128,300,1000 (SM136 flag).
# In-ticket retries (this ticket holds the GPU): a step whose labels still miss users runs again once in a fresh process
#   at CAP, then once at cap 0.92 (memory only, same numerics). Finished users are skipped (resume).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
CAP=$1; shift
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm136
S=$WT/artifacts/small-model/sm136
G=$S/gpu
W=C:/Users/jonat/workspace/wt
SC=$WT/scripts/small_model_heldout.py
ROWS=$G/sm136-rows.jsonl
EXTROWS=$G/sm136-ext-rows.jsonl
UMAIN=$S/users-sm136.json
UEXT=$S/users-sm136-ext.json
TRACK=E89-F65e-F77a
CK91=$W/sm91s2/artifacts/small-model/sm91s2-big-showtime/expertium/big-showtime/master.pth
CK99=$W/sm99/artifacts/small-model/sm99-big-inst-e2/expertium/big-inst/master.pth
CK118=$W/sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2/expertium/$TRACK/master.pth
CK113=$W/sm113/artifacts/small-model/sm113-inst-e35st-st-dec2ep32k-t99-p2/expertium/$TRACK/master.pth
CK124=$W/sm124/artifacts/small-model/sm124-c1-e29st-dec2ep32k-t91s2-p2/expertium/$TRACK/master.pth
CK115=$W/sm115/artifacts/small-model/sm115-c1-e38st-st-dec2ep32k-t99-p2/expertium/$TRACK/master.pth
T91=sm91s2-big-showtime; T99=sm99-big-inst-e2; T118=sm118-e26-dec2ep32k-t91s2; T113=sm113-inst-e35st-st-dec2ep32k-t99
T124=sm124-c1-e29st-dec2ep32k-t91s2; T115=sm115-c1-e38st-st-dec2ep32k-t99
FORMATS="w8+s8 w8+s4c8n w8+s4c8nhx8 w8+s8hx8 w8"
labels() { local t f; for t in "$@"; do for f in $FORMATS; do echo -n "$t/$f "; done; done; }
BIGL=$(labels $T91 $T99); SMALLL=$(labels $T118 $T113); TINYL=$(labels $T124 $T115); ALLL="$BIGL $SMALLL $TINYL"
MB="--model $T91=fork::$CK91 --model $T99=fork::$CK99"
MS="--model $T118=expertium:w64:$CK118 --model $T113=expertium:w64:$CK113"
MT="--model $T124=expertium:w64-c1:$CK124 --model $T115=expertium:w64-c1:$CK115"
TL=${SM136_TL:-5400}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$WT" || exit 1
say() { echo "[sm136-ticket $(date '+%m-%d %H:%M:%S')] $*" >> "$G/score.out"; }
run() { env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS HELDOUT_GPU_MEMORY_FRACTION=$CAP "$PY" -X utf8 -W ignore "$SC" \
          --inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets 0,32,128,300,1000 \
          "$@" >> "$G/score.out" 2>&1; }
only() { local l; for l in "$@"; do echo -n "--only $l "; done; }
missing() {  # missing ROWS USERSFILE LABEL...: users of the file without a row for some label (@e2s-query)
  "$PY" -X utf8 -c "
import json, sys
want = set(json.load(open(sys.argv[2]))['users'])
got = {}
try:
    f = open(sys.argv[1], 'rb')
except OSError:
    f = []
for raw in f:
    try:
        r = json.loads(raw.decode('utf-8', errors='replace'))
        if 'benchmark' in r or ('nonfinite' in r and 'error' not in r):
            got.setdefault(r['label'], set()).add(r['user_id'])
    except (ValueError, KeyError, TypeError, AttributeError):
        pass
print(max(len(want - got.get(l + '@e2s-query', set())) for l in sys.argv[3:]))" "$@"; }
big() { run --pool 732 --users-file "$UMAIN" $MB $(only $BIGL) --time-limit "$TL" --out "$ROWS"; }
small() { run --pool 732 --users-file "$UMAIN" $MS $(only $SMALLL) --time-limit "$TL" --out "$ROWS"; }
tiny() { run --pool 732 --users-file "$UMAIN" $MT $(only $TINYL) --time-limit "$TL" --out "$ROWS"; }
ext() { run --pool 1500 --users-file "$UEXT" $MB $MS $MT $(only $ALLL) --time-limit "$TL" --out "$EXTROWS"; }
stepped() {  # stepped NAME ROWS USERSFILE LABELS...: the step, then the in-ticket retries
  local name=$1 rows=$2 uf=$3 m rc keep; shift 3
  "$name" || return 1
  m=$(missing "$rows" "$uf" "$@"); [ "$m" = 0 ] && return 0
  say "$name: $m users missing, retry in a fresh process at cap $CAP"
  "$name" || return 1
  m=$(missing "$rows" "$uf" "$@"); [ "$m" = 0 ] && return 0
  say "$name: $m users missing, retry at cap 0.92 (memory only, same numerics)"
  keep=$CAP; CAP=0.92; "$name"; rc=$?; CAP=$keep
  say "$name: $(missing "$rows" "$uf" "$@") users missing after the retries"
  return $rc
}
for step in "$@"; do
  say "step $step (cap $CAP)"
  case "$step" in
    smoke)
      run --pool 732 --users-file "$S/cpu/smoke-users.json" $MB $MS $MT $(only $ALLL) --time-limit 1800 \
        --out "$G/smoke-gpu-rows.jsonl" || exit 1
      "$PY" -X utf8 "$S/sm136_check.py" smoke >> "$G/score.out" 2>&1 || exit 3
      touch "$G/smoke-ok";;
    big) stepped big "$ROWS" "$UMAIN" $BIGL || exit 1;;
    small) stepped small "$ROWS" "$UMAIN" $SMALLL || exit 1;;
    tiny) stepped tiny "$ROWS" "$UMAIN" $TINYL || exit 1;;
    ext) stepped ext "$EXTROWS" "$UEXT" $ALLL || say "ext: incomplete (report only), going on";;
    *) say "unknown step $step"; exit 2;;
  esac
done
say "ticket done"
exit 0
