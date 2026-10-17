#!/usr/bin/env bash
# SM32b (docs/SMALL_MODEL.md "State quantization, QAT in the final decay (SM32b)"): the 0.25-epoch decay at 32,768-review
# chunks from the epoch-10 checkpoint of scale-w64kd-gate-cached-ctx32768-cont, exactly dec10's recipe (same users,
# shuffle seed, LR 7e-4 -> 0 one_minus_sin, KD alpha 0.5, KD cache), with the card/note state through the vq378x412
# codec at every stored step (fake-quant, STE, frozen k-means codebooks). Runs from THIS worktree (state-vq): the
# small-model worktree and its runs are only read.
#   (0) preconditions: codebooks sha256, CPU check ALL OK, clean source tree (provenance)
#   (1) prepare the decay folder (CPU) and check it equals dec10's decay.json plus the state_vq key only
#   (2) ONE GPU ticket: CUDA codec check (CPU vs GPU codec, seconds) -> trainer smoke (2 users x 512 reviews) -> decay
#   (3) CPU: 34-user scoring (users34.json) of qat/w8+vq378x412, dec10/w8+vq378x412, dec10/w8+s8, qat/w8+s8
#       (3 processes in parallel), then the table (state_vq_report.py) -> artifacts/small-model/state-vq/qat/summary.md
# GPU rule (user, 2026-10-17): no ticket may overlap the break 2026-10-17 20:00 - 2026-10-18 02:00 (NO_BREAK=1 skips).
# DRYRUN=1: prepare + checks only (decay folder in a scratch path, deleted), echo the ticket and the scoring commands.
# Launch (after review): nohup bash $WT/artifacts/small-model/runbook-2026-10-18-qat.sh > $WT/artifacts/small-model/runbook-2026-10-18-qat.log 2>&1 &
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
BASH_EXE="C:/Program Files/Git/usr/bin/bash.exe"
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/state-vq
SMALL=C:/Users/jonat/workspace/wt/small-model
DEC_PY=$WT/scripts/expertium_decay.py
CHK=$WT/scripts/check_state_vq_qat.py
SC=$WT/scripts/small_model_heldout.py
CKPT=expertium/E89-F65e-F77a/master.pth
SRC_RUN=$SMALL/artifacts/accuracy/experiments/scale-w64kd-gate-cached-ctx32768-cont/run
DEC10_DIR=$SMALL/artifacts/small-model/scale-w64kd-gate-cached-ctx32768-cont-decay
DEC10=$DEC10_DIR/$CKPT
VQ=$WT/artifacts/small-model/state-vq
QD=$VQ/qat
BOOKS=$VQ/codebooks.npz
BOOKS_SHA=eab6db585da4f2944b0abf26788a93da49bb81a8b8a44ba2f62819e11a73aee0
RECIPE=vq378x412
D=$WT/artifacts/small-model/scale-w64kd-gate-cached-ctx32768-cont-decay-qat378x412$([ "${FACTOR:-float64}" = float32 ] && echo f32)
QAT=$D/$CKPT
USERS34=$VQ/users34.json
MINUTES=${MINUTES:-420}   # GPU ticket estimate: CPU per-step measurement 257 min unpadded x 1.6 (docs SM32b); scheduling only
MAX_MINUTES=${MAX_MINUTES:-720}   # the ticket's first command re-estimates from CUDA per-step times and stops above this
FACTOR=${FACTOR:-float64}  # float64 = the CPU scorer's factorization; float32 only if the float64 estimate is too long
export DRYRUN=${DRYRUN:-0}
log() { echo "[qat $(date '+%m-%d %H:%M:%S')] $*"; }
wait_slot() {  # the user's GPU break 2026-10-17 20:00 - 2026-10-18 02:00: no ticket that would overlap it
  local m=$1 said=0 bs be
  [ "${NO_BREAK:-0}" = 1 ] && return 0
  bs=$(date -d "2026-10-17 20:00" +%s); be=$(date -d "2026-10-18 02:00" +%s)
  while :; do
    local now; now=$(date +%s)
    if [ "$now" -ge "$be" ] || [ $((now + m * 60)) -le "$bs" ]; then return 0; fi
    [ "$said" = 0 ] && log "break window: a ${m}-min ticket would overlap 2026-10-17 20:00 - 02:00; waiting for 02:00" && said=1
    sleep 120
  done
}
cd "$WT" || exit 1
log "state-vq $(git -C $WT log --oneline -1)"

# ---- (0) preconditions
[ -f "$BOOKS" ] || { log "no codebooks $BOOKS (regenerate: docs SM32)"; exit 2; }
sha=$("$PY" -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$BOOKS")
[ "$sha" = "$BOOKS_SHA" ] || { log "codebooks sha256 $sha != pinned $BOOKS_SHA (the check and the PTQ table used the pinned file)"; exit 2; }
grep -q "^ALL OK" "$QD/check.log" || { log "CPU check $QD/check.log has no ALL OK; run: $PY -X utf8 $CHK all > $QD/check.log 2>&1"; exit 2; }
if [ -n "$(git -C $WT status --porcelain -- scripts expertium rwkv)" ]; then
  log "uncommitted source changes in $WT (scripts/expertium/rwkv); commit first so the run's provenance is clean"; exit 2
fi
[ -f "$SRC_RUN/epoch-10/$CKPT" ] && [ -f "$DEC10" ] || { log "epoch-10 checkpoint or dec10 missing"; exit 2; }

# ---- (1) prepare (CPU) and compare with dec10's decay.json
OPT="{\"state_vq\": {\"recipe\": \"$RECIPE\", \"codebooks\": \"$BOOKS\", \"sha256\": \"$BOOKS_SHA\", \"factor_dtype\": \"$FACTOR\"}}"
PREP=$D; [ "$DRYRUN" = 1 ] && PREP=$WT/artifacts/small-model/qat-dryrun-prepare
[ "$DRYRUN" = 1 ] && rm -rf "$PREP"
if [ ! -d "$PREP" ]; then
  "$PY" -X utf8 "$DEC_PY" prepare --source-run "$SRC_RUN" --epoch 10 --out "$PREP" --fraction 0.25 --shape one_minus_sin \
    --evaluate none --seed 1 --extra-options "$OPT" || { log "prepare failed"; exit 1; }
fi
A="$DEC10_DIR" B="$PREP" "$PY" -X utf8 - <<'PYEOF' || { log "prepared decay differs from dec10 beyond state_vq"; exit 1; }
import json, os, sys
a = json.load(open(os.path.join(os.environ["A"], "decay.json"))); b = json.load(open(os.path.join(os.environ["B"], "decay.json")))
pa = json.load(open(os.path.join(os.environ["A"], "protocol.json"))); pb = json.load(open(os.path.join(os.environ["B"], "protocol.json")))
fail = []
if {k: v for k, v in b["options"].items() if k != "state_vq"} != a["options"]: fail.append("options")
if "state_vq" not in b["options"]: fail.append("no state_vq")
for k in ("source_epoch", "source_update_offset_end", "fraction_of_epoch_reviews", "training_users", "training_reviews", "shuffle_seed", "lr"):
    if a[k] != b[k]: fail.append(f"{k}: {a[k]} vs {b[k]}")
for k in ("rows", "training", "dataset_path", "dataset_revision"):
    if pa[k] != pb[k]: fail.append(f"protocol {k}")
fa = {k: v for k, v in pa["full_run"].items()}; fb = {k: v for k, v in pb["full_run"].items()}
if fa != fb: fail.append(f"full_run {fa} vs {fb}")
print("[qat-prepare] " + ("OK: same users, rows, LR schedule, init checkpoint/optimizer and options as dec10, plus state_vq "
      + json.dumps(b["options"]["state_vq"]) if not fail else "REFUSE " + "; ".join(fail)))
sys.exit(1 if fail else 0)
PYEOF
[ "$DRYRUN" = 1 ] && rm -rf "$PREP"

# ---- (2) GPU ticket: CUDA codec check -> smoke -> decay
if [ ! -f "$QAT" ]; then
  [ "$DRYRUN" = 1 ] || wait_slot "$MINUTES"
  CMD="'$PY' -X utf8 '$CHK' cuda-codec --factor-dtype $FACTOR --max-minutes $MAX_MINUTES --out artifacts/small-model/state-vq/qat/cuda-codec.json && '$PY' -X utf8 '$DEC_PY' smoke --out '$D' && '$PY' -X utf8 '$DEC_PY' run --out '$D'"
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue (expected $MINUTES min): $CMD"
  else
    log "enqueue: SM32b QAT decay (expected $MINUTES min)"
    "$PY" "$QUEUE" run --owner small-model --purpose "small-model: SM32b state-VQ QAT ($RECIPE) 0.25-epoch decay at 32k from ctx32768c e10 (codec check + smoke first)" \
      --claim gpu --expected-minutes "$MINUTES" -- "$BASH_EXE" -c "$CMD"
    rc=$?; log "exit $rc: SM32b QAT decay"
    [ -f "$QAT" ] || { log "no QAT checkpoint ($QAT); stop"; exit 1; }
  fi
fi
if [ -f "$D/expertium/E89-F65e-F77a/result.json" ]; then
  R="$D/expertium/E89-F65e-F77a/result.json" "$PY" -X utf8 - <<'PYEOF'
import json, os
r = json.load(open(os.environ["R"]))
s = r.get("state_vq") or {}
print(f"[qat-result] status {r.get('status')}, {r.get('updates')} updates, {r.get('review_presentations'):,} reviews in "
      f"{r.get('training_seconds', 0):.0f} s = {r.get('reviews_per_second', 0):,.0f} reviews/s (dec10: 23,908 reviews/s, 3,865 s), "
      f"peak allocated {r.get('peak_allocated_mib', 0):,.0f} MiB, KD cache {r.get('kd_cache')}, state_vq {s.get('recipe')} sha {str(s.get('sha256'))[:12]}")
PYEOF
fi

# ---- (3) CPU scoring on users34 (3 processes), then the table
mkdir -p "$QD/logs"
score() {  # score NAME ARGS...
  local name=$1; shift
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN score $name: $*"; return 0; fi
  "$PY" -X utf8 "$SC" --device cpu --users-file "$USERS34" --time-limit 100000 --out "$QD/$name.jsonl" "$@" > "$QD/logs/$name.log" 2>&1
  log "scoring $name exit $?"
}
score qat-vq --model "qat=expertium:w64:$QAT" --only "qat/w8+$RECIPE" &
score dec10-vq --model "dec10=expertium:w64:$DEC10" --only "dec10/w8+$RECIPE" &
score s8 --model "dec10=expertium:w64:$DEC10" --model "qat=expertium:w64:$QAT" --only dec10/w8+s8 --only qat/w8+s8 &
wait
if [ "$DRYRUN" != 1 ]; then
  "$PY" -X utf8 -m scripts.state_vq_report --jsonl "$VQ/fork.jsonl" --jsonl "$QD/qat-vq.jsonl" --jsonl "$QD/dec10-vq.jsonl" \
    --jsonl "$QD/s8.jsonl" --labels "dec10/w8+s8,dec10/w8+$RECIPE,qat/w8+s8,qat/w8+$RECIPE" --student dec10/w8+s8 \
    --out artifacts/small-model/state-vq/qat/summary > "$QD/logs/report.log" 2>&1 && cat "$QD/summary.md" || log "report failed (see $QD/logs/report.log)"
fi
log "done"
