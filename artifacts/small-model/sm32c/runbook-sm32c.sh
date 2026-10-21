#!/usr/bin/env bash
# SM32c (2026-10-20): state compression (~400 bits per card/note, vq378x412p) trained in with LEARNED catalogs, during the
# one-epoch 32k decay of the best model, with carried state. ONE change from SM42 (the current best, ties SM40v2):
# the "state_vq" option {"recipe": "vq378x412p", "learn": true, impl cuda, float32 factorization, catalogs initialised from
# the SM32 k-means file}. Same source (plain 32k run, epoch 10, AdamW state continued), same users/order/LR (7e-4 -> 0
# one_minus_sin over 369,477,375 reviews), same replay KD targets, strided-lane carry, 32,768 x 1, gpu_memory_fraction 0.8.
#   1. GPU check ticket (~10 min): scripts/check_state_vq_learn.py --gpu (CUDA kernels vs torch codec, dstored backward,
#      timing of the learned path vs the SM32b kernel path). Starts as soon as the break guard lifts HOLD-chain (01:00),
#      i.e. normally just before the Muon decay (SM36) is queued. Stop if it fails.
#   2. CPU: prepare the decay; refuse unless decay.json/protocol.json equal SM42's except the state_vq option.
#   3. Wait for the SM36 arm to end (its launcher log), so SM36's scoring is not queued behind this decay.
#   4. GPU: smoke + decay in one ticket (expected ~420 min: SM42's decay ~275 min x ~1.35 QAT + learned-path overhead).
#   5. CPU (68 users of the SM32b re-check, 2 processes): sm32c/w8+vq378x412 with the LEARNED catalogs exported next to
#      the checkpoint, and sm32c/w8+s8. The references (sm42/w8+vq378x412 with the k-means file, sm42/w8+s8) are scored by
#      score-sm42-ptq.sh (started 10-20 00:14).
# Launch: nohup bash C:/Users/jonat/workspace/wt/sm32c/artifacts/small-model/sm32c/runbook-sm32c.sh \
#           > C:/Users/jonat/workspace/wt/sm32c/artifacts/small-model/sm32c/runbook-sm32c.log 2>&1 & disown
# DRYRUN=1: steps 2 (into a scratch folder, removed) and the command echo only; no queue, no waiting.
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
BASH_EXE="C:/Program Files/Git/usr/bin/bash.exe"
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm32c
SRC_WT=C:/Users/jonat/workspace/wt/small-model
SM42_WT=C:/Users/jonat/workspace/wt/sm42
SM=$WT/artifacts/small-model
O=$SM/sm32c
DEC_PY=$WT/scripts/expertium_decay.py
SC=$WT/scripts/small_model_heldout.py
CKPT=expertium/E89-F65e-F77a/master.pth
SRC_EXP=$SRC_WT/artifacts/accuracy/experiments/scale-w64kd-gate-cached-ctx32768-cont; FROM=10
SM42_DECAY=$SM42_WT/artifacts/small-model/scale-w64kd-gate-cached-ctx32768-cont-decay1ep32k-lanes
D=$SM/scale-w64kd-gate-cached-ctx32768-cont-decay1ep32k-lanes-vqlearn
L=sm32c
HOLD=$SRC_WT/artifacts/small-model/HOLD-chain
SM36_LOG=$SM42_WT/artifacts/small-model/launch-sm42-sm36-after-prune.log
BOOKS=$SM/state-vq/codebooks.npz
BOOKS_SHA=eab6db585da4f2944b0abf26788a93da49bb81a8b8a44ba2f62819e11a73aee0
SPEC='{"lanes": "strided"}'
DIST='{"teacher": "C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth", "alpha": 0.9, "decay_alpha": 0.5, "kd_source": "replay", "replay": "C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/fork-fp32-gate5k"}'
VQ="{\"recipe\": \"vq378x412p\", \"codebooks\": \"$BOOKS\", \"sha256\": \"$BOOKS_SHA\", \"factor_dtype\": \"float32\", \"impl\": \"cuda\", \"learn\": true}"
EXTRA="{\"distillation\": $DIST, \"state_carry\": $SPEC, \"chunk_reviews\": 32768, \"batch_chunks\": 1, \"state_vq\": $VQ}"
MIN_DECAY=${MIN_DECAY:-420}
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION
log() { echo "[sm32c $(date '+%m-%d %H:%M:%S')] $*"; }
q() {  # q MINUTES PURPOSE -- CMD...
  local minutes=$1 purpose=$2; shift 3
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $purpose (expected $minutes min)"; echo "    cmd: $*"; return 0; fi
  log "enqueue: $purpose (expected $minutes min)"
  "$PY" "$QUEUE" run --owner small-model --purpose "$purpose" --claim gpu --expected-minutes "$minutes" -- "$@"
  local rc=$?; log "exit $rc: $purpose"; return $rc
}
cd "$WT" || exit 1

# ---- 0. preconditions
if [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then log "STOP: uncommitted source changes"; exit 2; fi
"$PY" -c "import json,sys; r=json.load(open(r'$SM/check-state-vq-learn.json')); sys.exit(0 if r.get('passed') and not r.get('fails') else 1)" \
  || { log "STOP: CPU check report check-state-vq-learn.json has not passed"; exit 2; }
[ -f "$SRC_EXP/run/epoch-$FROM/$CKPT" ] && [ -f "$SM42_DECAY/decay.json" ] && [ -f "$BOOKS" ] || { log "STOP: source checkpoint, SM42 decay files or codebooks missing"; exit 2; }
log "start ($(git -C "$WT" log --oneline -1))"

# ---- 1. GPU check
if [ "$DRYRUN" != 1 ]; then
  until [ ! -e "$HOLD" ] && [ "$(date +%H)" -ge 1 ] && [ "$(date +%H)" -lt 21 ]; do sleep 30; done
fi
if [ "$DRYRUN" = 1 ] || [ ! -f "$SM/check-state-vq-learn-gpu.json" ]; then
  q 10 "SM32c GPU check: learned-codebook kernels (record + dstored) vs torch codec, timing" -- "$PY" -X utf8 "$WT/scripts/check_state_vq_learn.py" \
    --gpu --output artifacts/small-model/check-state-vq-learn-gpu.json || { log "STOP: GPU check failed"; exit 1; }
fi
"$PY" -c "import json,sys; r=json.load(open(r'$SM/check-state-vq-learn-gpu.json')); print('[gpu-check]', json.dumps({k: {kk: v.get(kk) for kk in ('ok','learned_over_sm32b','ms_fwd_bwd_64x2048x4','pick_mismatch','dstored_rel_diff')} if isinstance(v, dict) else v for k, v in r['checks']['GPU'].items()})); sys.exit(0 if r.get('passed') else 1)" \
  2>/dev/null || { [ "$DRYRUN" = 1 ] || { log "STOP: GPU check report not passed"; exit 1; }; }

# ---- 2. prepare + compare with SM42's decay
if [ "$DRYRUN" = 1 ]; then DUSE=$O/check-prepare; rm -rf "$DUSE"; else DUSE=$D; fi
if [ ! -d "$DUSE" ]; then
  "$PY" -X utf8 "$DEC_PY" prepare --source-run "$SRC_EXP/run" --epoch "$FROM" --out "$DUSE" --fraction 1.0 --shape one_minus_sin \
    --evaluate none --seed 1 --extra-options "$EXTRA" || { log "STOP: prepare failed"; exit 1; }
fi
A="$SM42_DECAY" B="$DUSE" WTR="$WT" "$PY" -X utf8 - <<'PYEOF' || { log "STOP: decay differs from SM42 beyond state_vq"; exit 1; }
import json, os, re, sys
from pathlib import Path
sys.path.insert(0, os.environ["WTR"])
a_dir, b_dir = Path(os.environ["A"]), Path(os.environ["B"])
a = json.loads((a_dir / "decay.json").read_text()); b = json.loads((b_dir / "decay.json").read_text())
pa = json.loads((a_dir / "protocol.json").read_text()); pb = json.loads((b_dir / "protocol.json").read_text())
fail = []
for k in ("source_run", "source_epoch", "source_update_offset_end", "fraction_of_epoch_reviews", "training_users", "training_reviews",
          "shuffle_seed", "lr", "track", "model"):
    if a.get(k) != b.get(k):
        fail.append(f"{k}: {a.get(k)} vs {b.get(k)}")
oa, ob = dict(a["options"]), dict(b["options"])
vq = ob.pop("state_vq", None)
if oa != ob:
    fail.append(f"options differ beyond state_vq: { {k: (oa.get(k), ob.get(k)) for k in set(oa) | set(ob) if oa.get(k) != ob.get(k)} }")
from scripts.state_vq import training_spec
spec = training_spec(vq)
if not (spec.get("learn") and spec.get("impl") == "cuda" and spec["recipe"] == "vq378x412p"):
    fail.append(f"state_vq {spec}")
if pa["rows"] != pb["rows"]:
    fail.append("protocol rows (users and order) differ from SM42")
for k in ("training", "dataset_path", "dataset_revision"):
    if pa.get(k) != pb.get(k):
        fail.append(f"protocol {k} differs")
fa, fb = dict(pa["full_run"]), dict(pb["full_run"])
for k in ("lr_schedule", "init_optimizer", "init_checkpoint", "update_offset"):
    if fa.get(k) != fb.get(k):
        fail.append(f"full_run {k}: {fa.get(k)} vs {fb.get(k)}")
text = (Path(os.environ["WTR"]) / "scripts" / "gpu_r01_prefetch.py").read_text(encoding="utf-8")
allow = set(re.findall(r'"([a-z_0-9]+)"', text[text.index("set(options) - {"):text.index("or bool(options.get(\"preprocessing_cache\"))")]))
if set(b["options"]) - allow:
    fail.append(f"options outside the prefetch allowlist: {sorted(set(b['options']) - allow)}")
from scripts.state_carry_train import validate_options
validate_options(b["options"])
if fail:
    print("[sm32c-check] REFUSE:\n  " + "\n  ".join(fail)); raise SystemExit(1)
print(f"[sm32c-check] OK: equal to SM42's decay except state_vq {spec}; {b['training_users']} users, {b['training_reviews']:,} reviews, {b['lr']}")
PYEOF
if [ "$DRYRUN" = 1 ]; then rm -rf "$DUSE"; fi

# ---- 3. wait for the SM36 arm (Muon decay + its scoring) to end
if [ "$DRYRUN" != 1 ]; then
  log "waiting for the SM36 arm to end ($SM36_LOG)"
  until grep -aEq 'ARM=sm36 (finished OK|ended with exit)' "$SM36_LOG" 2>/dev/null; do sleep 60; done
  log "SM36: $(grep -aE 'ARM=sm36 (finished OK|ended with exit)' "$SM36_LOG" | tail -1)"
  until [ ! -e "$HOLD" ]; do sleep 120; done
fi

# ---- 4. GPU: smoke + decay
if [ "$DRYRUN" = 1 ] || [ ! -f "$D/$CKPT" ]; then
  q $MIN_DECAY "SM32c 1-epoch decay at 32k, lanes carry, learned state compression ~400 bits (smoke first)" -- "$BASH_EXE" -c \
    "'$PY' -X utf8 '$DEC_PY' smoke --out '$D' && '$PY' -X utf8 '$DEC_PY' run --out '$D'" || { log "STOP: decay ticket failed"; exit 1; }
fi
[ "$DRYRUN" = 1 ] || [ -f "$D/expertium/E89-F65e-F77a/state-vq-books.npz" ] || { log "STOP: no checkpoint / learned catalogs"; exit 1; }

# ---- 5. CPU scoring on the 68 users
BOOKS_L=$D/expertium/E89-F65e-F77a/state-vq-books.npz
if [ "$DRYRUN" = 1 ]; then log "DRYRUN score: $L/w8+vq378x412 (--vq-codebooks $BOOKS_L) and $L/w8+s8 on $O/users68.json"; log "DRYRUN OK"; exit 0; fi
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
log "CPU scoring, 68 users"
"$PY" -X utf8 "$SC" --device cpu --users-file "$O/users68.json" --time-limit 100000 --out "$O/sm32c-vq.jsonl" --vq-codebooks "$BOOKS_L" \
  --model "$L=expertium:w64:$D/$CKPT" --only "$L/w8+vq378x412" > "$O/logs/sm32c-vq.log" 2>&1 &
"$PY" -X utf8 "$SC" --device cpu --users-file "$O/users68.json" --time-limit 100000 --out "$O/sm32c-s8.jsonl" \
  --model "$L=expertium:w64:$D/$CKPT" --only "$L/w8+s8" > "$O/logs/sm32c-s8.log" 2>&1 &
wait
log "done"
