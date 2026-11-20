#!/usr/bin/env bash
# Six-layer Tiny with the card-only state (SM134): ONE GPU ticket = the steps given, in order (run by launch-sm134.sh through
# scripts/r01_queue.py).   bash sm134_ticket.sh CAP STEP...     STEP = smoke | score
#   smoke: the two-head six-layer Tiny (SM130), users 2388, 4676, 1712, w8+s8 / compact / card-only, on the GPU ->
#          gpu/smoke-gpu-rows.jsonl, then sm134_check.py smoke (GPU vs the committed CPU rows and the published GPU rows); exit 3 on a mismatch
#   score: the two-head six-layer Tiny (SM130) and the Instant-Only six-layer Tiny (SM130), card-only w8+s4hgx4c8hx8n only,
#          723 users of the 732 set, show-time inputs, leak-free (neighbour), w8 weights; finished users are skipped (resume)
# Scorer flags exactly as six-layer Tiny (SM130)'s final scoring (launch-sm130.sh score()), plus --nonfinite-rows (as SM131).
# In-ticket retries (this ticket holds the GPU): a score step whose labels still miss users runs again once in a fresh process at
# CAP, then once at cap 0.92 (memory only, same numerics).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
CAP=$1; shift
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm134
S=$WT/artifacts/small-model/sm134
G=$S/gpu
W=C:/Users/jonat/workspace/wt
SC=$WT/scripts/small_model_heldout.py
ROWS=$G/sm134-rows.jsonl
USERS732=$W/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json
TRACK=E89-F65e-F77a
ARCH=w64-c1-d2p1u1
SM=$W/sm130/artifacts/small-model
CKTH=$SM/sm130-th-dec2ep32k-p2/expertium/$TRACK/master.pth
CKIO=$SM/sm130-io-dec2ep32k-p2/expertium/$TRACK/master.pth
TTH=sm130-th-dec2ep32k-t91s2; TIO=sm130-io-dec2ep32k-t99
FMT=w8+s4hgx4c8hx8n
LABELS="$TTH/$FMT $TIO/$FMT"
TL=${SM134_TL:-2400}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$WT" || exit 1
say() { echo "[sm134-ticket $(date '+%m-%d %H:%M:%S')] $*" >> "$G/score.out"; }
run() { env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS HELDOUT_GPU_MEMORY_FRACTION=$CAP "$PY" -X utf8 "$SC" \
          --pool 732 --inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows "$@" >> "$G/score.out" 2>&1; }
missing() {  # users of the 723 without a row for some label (@e2s-query); unparseable lines skipped
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
        if 'benchmark' in r or ('nonfinite' in r and 'error' not in r):
            got.setdefault(r['label'], set()).add(r['user_id'])
    except (ValueError, KeyError, TypeError, AttributeError):
        pass
print(723 - min(len(got.get(l + '@e2s-query', ())) for l in sys.argv[2:]))" "$ROWS" "$@"; }
score() { run --users-file "$USERS732" --model "$TTH=expertium:$ARCH:$CKTH" --model "$TIO=expertium:$ARCH:$CKIO" \
            --only $TTH/$FMT --only $TIO/$FMT --time-limit "$TL" --out "$ROWS"; }
stepped() {
  local m rc keep
  score || return 1
  m=$(missing $LABELS); [ "$m" = 0 ] && return 0
  say "score: $m users missing, retry in a fresh process at cap $CAP"
  score || return 1
  m=$(missing $LABELS); [ "$m" = 0 ] && return 0
  say "score: $m users missing, retry at cap 0.92 (memory only, same numerics)"
  keep=$CAP; CAP=0.92; score; rc=$?; CAP=$keep
  say "score: $(missing $LABELS) users missing after the retries"
  return $rc
}
for step in "$@"; do
  say "step $step (cap $CAP)"
  case "$step" in
    smoke)
      run --users-file "$S/smoke-users.json" --model "$TTH=expertium:$ARCH:$CKTH" \
        --only $TTH/w8+s8 --only $TTH/w8+s4c8nhx8 --only $TTH/$FMT --time-limit 900 --out "$G/smoke-gpu-rows.jsonl" || exit 1
      "$PY" -X utf8 "$S/sm134_check.py" smoke >> "$G/score.out" 2>&1 || exit 3
      touch "$G/smoke-ok";;
    score) stepped || exit 1;;
    *) say "unknown step $step"; exit 2;;
  esac
done
say "ticket done"
exit 0
