#!/usr/bin/env bash
# Compact state for Small and Big (SM131): ONE GPU ticket = the steps given, in order (run by launch-sm131.sh through
# scripts/r01_queue.py; chained in one ticket so the six-layer Tiny (SM130)'s ~110-min tickets do not sit between them).
#   bash sm131_ticket.sh CAP STEP...     STEP = smoke | big | small
#   smoke: the show-time big model (SM91s2), 2 small users (2388, 4676), the four formats, on the GPU -> gpu/smoke-gpu-rows.jsonl,
#          then sm131_check.py smoke (GPU vs the committed CPU rows; w8+s8 also vs the published rows); exit 3 on a mismatch
#   big:   the show-time big model (SM91s2) + the show-time instant-only big model (SM99), fork default / compact / card-only
#   small: the show-time two-head cool-down for Small (SM118) + the longer Small Instant-Only model (SM113), compact / card-only
#          (their fork-default rows come from the compact state for Tiny (SM129), same scorer tree)
# All: 723 users of the 732 set, show-time inputs, leak-free (neighbour), w8 weights; finished users are skipped (resume).
# In-ticket retries (this ticket holds the GPU, so a higher cap is safe): a big / small step whose labels still miss users
#   runs again once in a fresh process at CAP (an OOM after fragmentation, as user 3735 in SM129), then once at cap 0.92.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
CAP=$1; shift
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm131
S=$WT/artifacts/small-model/sm131
G=$S/gpu
W=C:/Users/jonat/workspace/wt
SC=$WT/scripts/small_model_heldout.py
ROWS=$G/sm131-rows.jsonl
USERS732=$W/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json
TRACK=E89-F65e-F77a
CK91=$W/sm91s2/artifacts/small-model/sm91s2-big-showtime/expertium/big-showtime/master.pth
CK99=$W/sm99/artifacts/small-model/sm99-big-inst-e2/expertium/big-inst/master.pth
CK118=$W/sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2/expertium/$TRACK/master.pth
CK113=$W/sm113/artifacts/small-model/sm113-inst-e35st-st-dec2ep32k-t99-p2/expertium/$TRACK/master.pth
T91=sm91s2-big-showtime; T99=sm99-big-inst-e2; T118=sm118-e26-dec2ep32k-t91s2; T113=sm113-inst-e35st-st-dec2ep32k-t99
BIGL="$T91/w8+s4c8n $T91/w8+s4c8nhx8 $T91/w8+s4hgx4c8hx8n $T99/w8+s4c8n $T99/w8+s4c8nhx8 $T99/w8+s4hgx4c8hx8n"
SMALLL="$T118/w8+s4c8nhx8 $T118/w8+s4hgx4c8hx8n $T113/w8+s4c8nhx8 $T113/w8+s4hgx4c8hx8n"
TLB=${SM131_TL_BIG:-5400}; TLS=${SM131_TL_SMALL:-2700}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$WT" || exit 1
say() { echo "[sm131-ticket $(date '+%m-%d %H:%M:%S')] $*" >> "$G/score.out"; }
run() { env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS HELDOUT_GPU_MEMORY_FRACTION=$CAP "$PY" -X utf8 "$SC" \
          --pool 732 --inputs e2s-query --split-existing --deleted-cards neighbour "$@" >> "$G/score.out" 2>&1; }
only() { local l; for l in "$@"; do echo -n "--only $l "; done; }
missing() {  # missing LABEL...: users of the 723 without a row for some label (@e2s-query); unparseable lines skipped
  "$PY" -X utf8 -c "
import json, sys
got = {}
try:
    f = open(sys.argv[1], 'rb')
except OSError:
    f = []
for raw in f:
    try:
        r = json.loads(raw.decode('utf-8', errors='replace'))
        if 'benchmark' in r:
            got.setdefault(r['label'], set()).add(r['user_id'])
    except (ValueError, KeyError, TypeError, AttributeError):
        pass
print(723 - min(len(got.get(l + '@e2s-query', ())) for l in sys.argv[2:]))" "$ROWS" "$@"; }
big() { run --users-file "$USERS732" --model "$T91=fork::$CK91" --model "$T99=fork::$CK99" $(only $BIGL) --time-limit "$TLB" --out "$ROWS"; }
small() { run --users-file "$USERS732" --model "$T118=expertium:w64:$CK118" --model "$T113=expertium:w64:$CK113" $(only $SMALLL) \
            --time-limit "$TLS" --out "$ROWS"; }
stepped() {  # stepped NAME LABELS...: the step, then the in-ticket retries
  local name=$1 m rc keep; shift
  "$name" || return 1
  m=$(missing "$@"); [ "$m" = 0 ] && return 0
  say "$name: $m users missing, retry in a fresh process at cap $CAP"
  "$name" || return 1
  m=$(missing "$@"); [ "$m" = 0 ] && return 0
  say "$name: $m users missing, retry at cap 0.92 (memory only, same numerics)"
  keep=$CAP; CAP=0.92; "$name"; rc=$?; CAP=$keep
  say "$name: $(missing "$@") users missing after the retries"
  return $rc
}
for step in "$@"; do
  say "step $step (cap $CAP)"
  case "$step" in
    smoke)
      run --users-file "$S/smoke-users.json" --model "$T91=fork::$CK91" $(only $T91/w8+s8 $T91/w8+s4c8n $T91/w8+s4c8nhx8 $T91/w8+s4hgx4c8hx8n) \
        --time-limit 1200 --out "$G/smoke-gpu-rows.jsonl" || exit 1
      "$PY" -X utf8 "$S/sm131_check.py" smoke >> "$G/score.out" 2>&1 || exit 3
      touch "$G/smoke-ok";;
    big) stepped big $BIGL || exit 1;;
    small) stepped small $SMALLL || exit 1;;
    *) say "unknown step $step"; exit 2;;
  esac
done
say "ticket done"
exit 0
