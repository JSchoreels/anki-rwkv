#!/usr/bin/env bash
# Small-model runbook 2026-10-16d (SM22): stage longdecay = the cont stage's decay + 464-user scoring of
# runbook-2026-10-16b.sh, but the 0.25-epoch decay from the ctx8192 epoch EPOCH runs at a longer chunk length (DCHUNK x DBATCH,
# 32,768 reviews per update unchanged). Same users/seed (1)/LR shape (one_minus_sin 7e-4 -> 0)/options as $SM/$SRC_NAME-decay,
# so the two decays are a paired comparison; only chunk_reviews/batch_chunks differ (passed through --extra-options).
#   SRC_NAME=scale-w64kd-gate-cached-ctx8192-cont EPOCH=8 DCHUNK=16384 DBATCH=2 DRYRUN=0 bash $SM/runbook-2026-10-16d.sh longdecay
# DRYRUN=1 stubs the queue (echo), prepares into scratch folders under $SM only (deleted afterwards), asserts the chunk fields and
# that the user list equals the 8192 decay's, and logs "DRYRUN OK".
set -u
STAGE=${1:?stage longdecay}
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
BASH_EXE="C:/Program Files/Git/usr/bin/bash.exe"
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/small-model
DEC_PY=$WT/scripts/expertium_decay.py
SC=$WT/scripts/small_model_heldout.py
SM=$WT/artifacts/small-model
CKPT=expertium/E89-F65e-F77a/master.pth
OUT=$SM/heldout.jsonl
export DRYRUN=${DRYRUN:-0}
log() { echo "[0xx $(date +%H:%M:%S)] $*"; }
q() {
  local minutes=$1 purpose=$2; shift 3
  if [ "$DRYRUN" = 1 ]; then
    log "DRYRUN enqueue: $purpose (expected $minutes min)"; echo "    cmd: $*"; return 0
  fi
  log "enqueue: $purpose (expected $minutes min)"
  "$PY" "$QUEUE" run --owner small-model --purpose "small-model: $purpose" --claim gpu --expected-minutes "$minutes" -- "$@"
  local rc=$?; log "exit $rc: $purpose"; return $rc
}
need() { if [ "$DRYRUN" = 1 ]; then log "DRYRUN warning (would stop): $1"; return 0; fi; log "$1"; return 1; }

longdecay() {
  SRC_NAME=${SRC_NAME:-scale-w64kd-gate-cached-ctx8192-cont}; EPOCH=${EPOCH:-8}; DCHUNK=${DCHUNK:-16384}
  DBATCH=${DBATCH:-$((32768 / DCHUNK))}
  [ $((DCHUNK * DBATCH)) -eq 32768 ] || { log "reviews per update $((DCHUNK * DBATCH)) != 32768 (VRAM/optimizer tuned for it)"; return 2; }
  [[ "$SRC_NAME" =~ ^scale-w64kd-([a-z]+)-cached-ctx([0-9]+)-cont$ ]] || { log "cannot derive label from SRC_NAME=$SRC_NAME"; return 2; }
  TIER=${BASH_REMATCH[1]}; SCHUNK=${BASH_REMATCH[2]}
  LBL=s-w64kd-$TIER-ctx${SCHUNK}c-dec$EPOCH-L$((DCHUNK / 1024))k
  SRC_EXP=$WT/artifacts/accuracy/experiments/$SRC_NAME
  REF=$SM/$SRC_NAME-decay
  DECF=$SM/$SRC_NAME-decay-L$((DCHUNK / 1024))k
  D=$DECF/$CKPT
  EXTRA="{\"chunk_reviews\": $DCHUNK, \"batch_chunks\": $DBATCH}"
  log "longdecay: source $SRC_NAME e$EPOCH, chunk $DCHUNK x $DBATCH ($((DCHUNK * DBATCH)) reviews/update), out $DECF, label $LBL"
  [ -f "$SRC_EXP/run/epoch-$(printf "%02d" "$EPOCH")/$CKPT" ] || { need "no epoch-$EPOCH checkpoint of $SRC_NAME" || return 2; }

  # ---- paired-user check: prepare the new decay (and, if the 8192 reference is missing, what cont would prepare) in scratch
  SCR=$SM/longdecay-check-prepare; REFSCR=$SM/longdecay-check-ref; rm -rf "$SCR" "$REFSCR"
  "$PY" -X utf8 "$DEC_PY" prepare --source-run "$SRC_EXP/run" --epoch "$EPOCH" --out "$SCR" --fraction 0.25 --shape one_minus_sin \
    --evaluate none --seed 1 --extra-options "$EXTRA" > /dev/null || { log "scratch prepare failed"; rm -rf "$SCR"; return 1; }
  REFDIR=$REF
  if [ ! -f "$REF/decay.json" ]; then
    log "reference decay $REF missing: comparing against what cont would prepare (chunk $SCHUNK x $((32768 / SCHUNK)))"
    "$PY" -X utf8 "$DEC_PY" prepare --source-run "$SRC_EXP/run" --epoch "$EPOCH" --out "$REFSCR" --fraction 0.25 --shape one_minus_sin \
      --evaluate none --seed 1 > /dev/null || { log "reference scratch prepare failed"; rm -rf "$SCR" "$REFSCR"; return 1; }
    REFDIR=$REFSCR
  fi
  NEW="$SCR" REFD="$REFDIR" DCHUNK="$DCHUNK" DBATCH="$DBATCH" "$PY" -X utf8 - <<'PYEOF' || { log "paired check FAILED"; rm -rf "$SCR" "$REFSCR"; return 1; }
import json, os
from pathlib import Path
new, ref = Path(os.environ["NEW"]), Path(os.environ["REFD"])
dn, dr = (json.loads((p / "decay.json").read_text()) for p in (new, ref))
pn, pr = (json.loads((p / "protocol.json").read_text()) for p in (new, ref))
chunk, bc = int(os.environ["DCHUNK"]), int(os.environ["DBATCH"])
assert dn["options"]["chunk_reviews"] == chunk and dn["options"]["batch_chunks"] == bc, dn["options"]
assert pn["training"]["chunk_reviews"] == chunk
assert chunk * bc == 32768 and dr["options"]["chunk_reviews"] * dr["options"]["batch_chunks"] == 32768
assert pn["rows"] == pr["rows"], "training user/row set differs from the reference decay"
assert dn["training_users"] == dr["training_users"] and dn["training_reviews"] == dr["training_reviews"]
assert dn["shuffle_seed"] == dr["shuffle_seed"] == 1 and dn["lr"] == dr["lr"]
for k in ("source_epoch", "source_update_offset_end", "fraction_of_epoch_reviews", "track", "evaluation_set", "dropped_width32_options"):
    assert dn[k] == dr[k], (k, dn[k], dr[k])
assert pn["full_run"]["lr_schedule"] == pr["full_run"]["lr_schedule"] and pn["full_run"]["update_offset"] == pr["full_run"]["update_offset"]
strip = lambda o: {k: v for k, v in o.items() if k not in ("chunk_reviews", "batch_chunks")}
assert strip(dn["options"]) == strip(dr["options"]), "options differ beyond chunk fields"
print(f"[longdecay-check] OK: chunk_reviews {chunk} x batch_chunks {bc}; same {dn['training_users']} users / {dn['training_reviews']:,} reviews "
      f"(row sets identical), seed 1, same LR/offset/options as the reference ({os.environ['REFD']}, chunk {dr['options']['chunk_reviews']} x {dr['options']['batch_chunks']})")
PYEOF
  rm -rf "$SCR" "$REFSCR"

  # ---- decay (smoke first); the KD cache is cold at a new chunk length
  if [ "$DRYRUN" = 1 ]; then
    log "DRYRUN: would prepare $DECF (--source-run $SRC_EXP/run --epoch $EPOCH --fraction 0.25 --shape one_minus_sin --evaluate none --seed 1 --extra-options '$EXTRA')"; ok=1
  elif [ -d "$DECF" ]; then ok=1
  else "$PY" -X utf8 "$DEC_PY" prepare --source-run "$SRC_EXP/run" --epoch "$EPOCH" --out "$DECF" --fraction 0.25 --shape one_minus_sin \
      --evaluate none --seed 1 --extra-options "$EXTRA" && ok=1 || ok=0; fi
  [ "$ok" = 1 ] || { log "decay prepare failed"; return 1; }
  [ -f "$D" ] || q 90 "longdecay $SRC_NAME: 0.25-epoch decay from e$EPOCH at chunk $DCHUNK x $DBATCH (smoke first, cold KD cache)" -- "$BASH_EXE" -c \
      "'$PY' -X utf8 '$DEC_PY' smoke --out '$DECF' && '$PY' -X utf8 '$DEC_PY' run --out '$DECF'" || return 1

  # ---- 464-user scoring of the decayed checkpoint only, resume ticket only if users remain
  MODELS="--model $LBL=expertium:w64:$D"
  LABELS="--only $LBL/w8+s8"
  # shellcheck disable=SC2086
  q 30 "longdecay $SRC_NAME: full-history scoring of $LBL (464 users)" -- "$PY" -X utf8 "$SC" $MODELS $LABELS --time-limit 3000 --out "$OUT" || return 1
  remaining=$(OUT="$OUT" L1="$LBL/w8+s8" "$PY" -X utf8 - <<'PYEOF'
import json, os
want = {"fork/w8+s8": set(), os.environ["L1"]: set()}
for line in open(os.environ["OUT"], encoding="utf-8"):
    if '"label"' in line:
        r = json.loads(line)
        if r["label"] in want and "benchmark" in r:
            want[r["label"]].add(r["user_id"])
total = want["fork/w8+s8"]
print(len(total - want[os.environ["L1"]]) if total else 0)
PYEOF
)
  log "users still unscored after the scoring ticket (vs the fork users): ${remaining:-?}"
  if [ "$DRYRUN" = 1 ] || [ "${remaining:-1}" != 0 ]; then
    # shellcheck disable=SC2086
    q 15 "longdecay $SRC_NAME: scoring resume (only if users remain)" -- "$PY" -X utf8 "$SC" $MODELS $LABELS --time-limit 1200 --out "$OUT"
  fi
  [ "$DRYRUN" = 1 ] && log "DRYRUN OK"
  return 0
}

cd "$WT" || exit 1
case "$STAGE" in
longdecay) longdecay; rc=$?; [ $rc -eq 0 ] && log "done: $STAGE" || log "$STAGE ended with exit $rc"; exit $rc ;;
*) echo "unknown stage $STAGE (longdecay)"; exit 2 ;;
esac
