#!/usr/bin/env bash
# SM15 first pruning step (2026-10-19): depth cut of the context streams of the w64 student (docs/SMALL_MODEL_PRUNING.md,
# option b2), adapted to the sc-lanes parent. Prepared, not run. Needs the prune worktree (branch prune, from sm21v2).
#
#   P_dec (reference, decayed) = s-w64kd-gate-sc-lanes-dec12-1ep32k/w8+s8  (+0.80% / +0.73% vs the fork, 464 users)
#   P_e   (pruning source, undecayed) = scale-w64kd-gate-sc-lanes-cont epoch 12 (2,048 x 16, strided-lane carry, replay KD)
#
# Stages (first argument or STAGE=):
#   screen  CPU, not queued (design 2.1): zero-shot score the cuts of P_dec on the 68 feature-screen users, write
#           $SM/prune-screen/report.{md,json} with the chosen ARCH (pre-registered rule). Two processes of 2 threads.
#   e1      GPU (design 2.2, adapted): make-run -> fine-tune 0.5 epoch at 2,048 x 16 (strided lanes carry, replay targets,
#           constant LR 7e-4 after warmup, KD alpha 0.9) -> promote -> the SAME decay as the parent (1 epoch at 32,768 x 1,
#           seed 1, lanes carry, replay targets, gpu_memory_fraction 0.8, 7e-4 -> 0 one_minus_sin) -> 464-user scoring
#           (w8+s8 and fp32) -> 732-pool scoring (w8+s8) -> decision helper (design 2.3) vs P_dec on 464 users, 732 reported.
#           delta = pruning cost + 0.5 epoch of extra training (the parent decayed straight from epoch 12).
#   ARCH=w64-d2p1u1 (D7) unless the screen picks a D8 (w64-d3p1u1 / w64-d2p2u1 / w64-d2p1u2); env ARCH is required for e1.
#
#   dry run  DRYRUN=1 ARCH=w64-d2p1u1 bash /c/Users/jonat/workspace/wt/prune/artifacts/small-model/runbook-prune-depth.sh e1
#   launch   see launch-prune-after-chain.sh (waits for the small-model chain), or
#            ARCH=<arch> nohup bash <this script> e1 > <log> 2>&1 & disown
#   Resume: rerun the same command (finished tickets/rows are skipped). SUMMARY_ONLY=1 recomputes the decision only.
# `accuracy_loop train` is not used: the fine-tune and the decay are expertium_decay.py runs (one process, no "more epochs"
# exit code 3 to handle); a failed ticket stops the runbook and a rerun continues from the checkpoints already on disk.
# Does not touch the GPU outside the queue: every GPU step is a r01_queue.py ticket (claim gpu, owner small-model, purpose SM15 ...).
# Never edit scripts/ expertium/ rwkv/ of the prune worktree while a ticket runs (SourceWatch stops the epoch).
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
BASH_EXE="C:/Program Files/Git/usr/bin/bash.exe"
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/prune
PAR_WT=C:/Users/jonat/workspace/wt/sm21v2                  # read only: parent experiment, checkpoint, 464 / pool rows
SRC_WT=C:/Users/jonat/workspace/wt/small-model             # read only: fork rows, pool rows
SM=$WT/artifacts/small-model
PRUNE=$WT/scripts/small_model_prune.py
DEC_PY=$WT/scripts/expertium_decay.py
SC=$WT/scripts/small_model_heldout.py                      # ported --pool support (small-model's copy cannot build the pruned presets)
CKPT=expertium/E89-F65e-F77a/master.pth
PARENT_NAME=scale-w64kd-gate-sc-lanes-cont
PARENT_EXP=${PARENT_EXP:-$PAR_WT/artifacts/accuracy/experiments/$PARENT_NAME}
PARENT_EPOCH=${PARENT_EPOCH:-12}
PARENT_LBL=${PARENT_LBL:-s-w64kd-gate-sc-lanes-dec12-1ep32k/w8+s8}
PARENT_DEC=$PAR_WT/artifacts/small-model/$PARENT_NAME-decay1ep32k
FT_FRACTION=${FT_FRACTION:-0.5}
P_DEC_464=$PAR_WT/artifacts/small-model/heldout-sm40v2.jsonl
P_DEC_POOL=$PAR_WT/artifacts/small-model/heldout-sm40v2-pools.jsonl
FORK_464=$SRC_WT/artifacts/small-model/heldout.jsonl
FORK_POOL=$SRC_WT/artifacts/small-model/heldout-pools.jsonl
SPEC='{"lanes": "strided"}'
STAGE=${1:-${STAGE:-}}
ARCH=${ARCH:-}
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
log() { echo "[prune $(date '+%m-%d %H:%M:%S')] $*"; }
q() {  # q MINUTES PURPOSE -- CMD...
  local minutes=$1 purpose=$2; shift 3
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $purpose (expected $minutes min)"; echo "    cmd: $*"; return 0; fi
  log "enqueue: $purpose (expected $minutes min)"
  "$PY" "$QUEUE" run --owner small-model --purpose "$purpose" --claim gpu --expected-minutes "$minutes" -- "$@"
  local rc=$?; log "exit $rc: $purpose"; return $rc
}
need() { if [ "$DRYRUN" = 1 ]; then log "DRYRUN warning (would stop): $1"; return 0; fi; log "STOP: $1"; return 1; }
label_users() {  # label_users FILE LABEL -> number of users with a benchmark row for LABEL
  F="$1" LB="$2" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
p = Path(os.environ['F'])
print(len({r['user_id'] for r in map(json.loads, open(p, encoding='utf-8')) if r.get('label') == os.environ['LB'] and 'benchmark' in r}) if p.exists() else 0)"; }

# ------------------------------------------------------------------------------------------------ stage screen (CPU)
stage_screen() {
  local S=$SM/prune-screen
  mkdir -p "$S"
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: would run run-screen.sh for shards 0 and 1 (2 processes), merge, screen-report"; return 0; fi
  for s in 0 1; do bash "$S/run-screen.sh" $s > "$S/screen-s$s.log" 2>&1 & done
  wait
  cat "$S"/screen-s0.jsonl "$S"/screen-s1.jsonl > "$S/screen.jsonl"
  "$PY" -X utf8 "$PRUNE" screen-report --jsonl "$S/screen.jsonl" --parent P_dec/w8+s8 --out "$S/report"
}

# ------------------------------------------------------------------------------------------------ stage e1 (GPU)
preconditions() {
  [ -n "$ARCH" ] || { need "ARCH is not set (the screen's choice, e.g. w64-d2p1u1)" || return 2; }
  "$PY" -X utf8 -c "
import sys; sys.path.insert(0, r'$WT')
from expertium.variants import variant; variant('$ARCH')" 2>/dev/null || { need "ARCH $ARCH is not a valid preset" || return 2; }
  [ -f "$PARENT_EXP/run/epoch-$PARENT_EPOCH/$CKPT" ] && [ -f "$PARENT_EXP/run/epoch-$PARENT_EPOCH/${CKPT%master.pth}optimizer.pth" ] \
    || { need "epoch $PARENT_EPOCH checkpoint of $PARENT_NAME missing" || return 2; }
  if [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    need "uncommitted source changes in scripts/ expertium/ rwkv/ (SourceWatch refuses them)" || return 2; fi
  "$PY" -c "
import json, sys
r = json.load(open(r'$SM/check-prune.json'))
sys.exit(0 if r.get('passed') and not r.get('fails') else 1)" 2>/dev/null || { need "check-prune.json has not passed (or is missing)" || return 2; }
  [ -f "$P_DEC_464" ] && [ -f "$P_DEC_POOL" ] && [ -f "$FORK_464" ] || { need "reference rows missing" || return 2; }
  [ -f "$PARENT_DEC/$CKPT" ] || { need "P_dec checkpoint missing" || return 2; }
  return 0
}

main_run() {
  local SUF=""; [ "$DRYRUN" = 1 ] && SUF="-dry"
  NAME=prune-$ARCH-sc-lanes-e$PARENT_EPOCH$SUF
  EXP=$WT/artifacts/accuracy/experiments/$NAME
  RUN=$EXP/run
  FT=$SM/$NAME-ft
  D=$SM/$NAME-decay1ep32k
  L=s-w64kd-gate-prune-$ARCH-dec
  OUT=$SM/heldout-prune-$ARCH.jsonl                         # 464-user rows (this worktree)
  OUT_POOL=$SM/heldout-prune-$ARCH-pools.jsonl              # 732-pool rows (this worktree)
  local FT_EPOCH=$((PARENT_EPOCH + 1))
  if [ "$DRYRUN" = 1 ]; then rm -rf "$EXP" "$FT" "$D"; fi

  # ---- CPU: synthetic experiment (pruned P_e + fresh AdamW) and the constant-LR fine-tune protocol
  if [ ! -d "$EXP" ]; then
    "$PY" -X utf8 "$PRUNE" make-run --parent-exp "$PARENT_EXP" --epoch "$PARENT_EPOCH" --arch "$ARCH" --name "$NAME" || { log "make-run failed"; return 1; }
  fi
  if [ ! -d "$FT" ]; then
    "$PY" -X utf8 "$PRUNE" ft-prepare --run "$RUN" --epoch "$PARENT_EPOCH" --out "$FT" --fraction "$FT_FRACTION" --seed 2 || { log "ft-prepare failed"; return 1; }
  fi
  FT="$FT" ARCH="$ARCH" SPEC="$SPEC" "$PY" -X utf8 - <<'PYEOF' || { log "fine-tune check failed"; return 1; }
import json, os
from pathlib import Path
d = json.loads((Path(os.environ["FT"]) / "decay.json").read_text()); p = json.loads((Path(os.environ["FT"]) / "protocol.json").read_text())
o, fail = d["options"], []
if "lr_schedule" in p["full_run"]:
    fail.append("lr_schedule present (must be constant LR)")
if p["full_run"].get("update_offset") != 0:
    fail.append(f"update_offset {p['full_run'].get('update_offset')}")
if o.get("architecture") != os.environ["ARCH"]:
    fail.append(f"architecture {o.get('architecture')}")
if o.get("state_carry") != json.loads(os.environ["SPEC"]):
    fail.append(f"state_carry {o.get('state_carry')}")
if (o.get("chunk_reviews"), o.get("batch_chunks"), o.get("gpu_memory_fraction")) != (2048, 16, 0.8):
    fail.append(f"chunk/batch/gpu {o.get('chunk_reviews')}/{o.get('batch_chunks')}/{o.get('gpu_memory_fraction')}")
k = o.get("distillation", {})
if k.get("kd_source") != "replay" or "cache" in k or k.get("alpha") != 0.9:
    fail.append(f"distillation {k}")
if d["fraction_of_epoch_reviews"] != 0.5 or d["shuffle_seed"] != 2:
    fail.append("fraction/seed")
if fail:
    print("[prune-check] REFUSE: " + "; ".join(fail)); raise SystemExit(1)
print(f"[prune-check] fine-tune OK: {d['training_users']} users, {d['training_reviews']:,} reviews, {d['lr']}, 2048 x 16, lanes carry, replay targets")
PYEOF

  # ---- GPU ticket 1: fine-tune (smoke first, inside the ticket)
  if [ "$DRYRUN" = 1 ] || [ ! -f "$FT/$CKPT" ]; then
    q 60 "SM15 fine-tune $ARCH 0.5 epoch at 2048x16 lanes carry from sc-lanes e$PARENT_EPOCH (smoke first)" -- "$BASH_EXE" -c \
      "'$PY' -X utf8 '$DEC_PY' smoke --out '$FT' && '$PY' -X utf8 '$DEC_PY' run --out '$FT'" || return 1
  fi
  [ "$DRYRUN" = 1 ] || [ -f "$FT/$CKPT" ] || { log "no fine-tune checkpoint; stop"; return 1; }
  if [ "$DRYRUN" != 1 ]; then
    "$PY" -X utf8 -c "
import json
r = json.load(open(r'$FT/$CKPT'.replace('master.pth', 'result.json')))
rps = r.get('reviews_per_second') or 0
print(f'[prune] fine-tune: {rps:,.0f} reviews/s, {r.get(\"updates\")} updates, {r.get(\"review_presentations\"):,} reviews (parent epoch 12: 88,701 reviews/s with 13 layers)')
if rps < 60000: print('[prune] WARNING: below 60k reviews/s; the 32k decay estimate below is optimistic')"
    if [ ! -d "$RUN/epoch-$FT_EPOCH" ]; then
      "$PY" -X utf8 "$PRUNE" promote --ft "$FT" --run "$RUN" --epoch "$FT_EPOCH" || { log "promote failed"; return 1; }
    fi
  fi

  # ---- CPU: the decay prepared exactly like the parent's (SM40v2): same users / order / schedule, options differ only in the architecture
  local SRC_EPOCH=$FT_EPOCH; [ "$DRYRUN" = 1 ] && SRC_EPOCH=$PARENT_EPOCH
  if [ "$DRYRUN" = 1 ]; then DUSE=$SM/prune-check-decay; rm -rf "$DUSE"; else DUSE=$D; fi
  if [ ! -d "$DUSE" ]; then
    "$PY" -X utf8 "$DEC_PY" prepare --source-run "$RUN" --epoch "$SRC_EPOCH" --out "$DUSE" --fraction 1.0 --shape one_minus_sin \
      --evaluate none --seed 1 --extra-options '{"chunk_reviews": 32768, "batch_chunks": 1}' || { log "decay prepare failed"; return 1; }
  fi
  REF=$SM/prune-check-parent-ref; rm -rf "$REF"
  "$PY" -X utf8 "$DEC_PY" prepare --source-run "$PARENT_EXP/run" --epoch "$PARENT_EPOCH" --out "$REF" --fraction 1.0 --shape one_minus_sin \
    --evaluate none --seed 1 --extra-options '{"chunk_reviews": 32768, "batch_chunks": 1}' > /dev/null || { log "reference prepare failed"; return 1; }
  A="$REF" B="$DUSE" SPEC="$SPEC" ARCH="$ARCH" "$PY" -X utf8 - <<'PYEOF' || { rm -rf "$REF"; return 1; }
import json, os
from pathlib import Path
a = json.loads((Path(os.environ["A"]) / "decay.json").read_text()); b = json.loads((Path(os.environ["B"]) / "decay.json").read_text())
pa = json.loads((Path(os.environ["A"]) / "protocol.json").read_text()); pb = json.loads((Path(os.environ["B"]) / "protocol.json").read_text())
fail = []
for k in ("fraction_of_epoch_reviews", "training_users", "training_reviews", "shuffle_seed", "lr"):
    if a[k] != b[k]:
        fail.append(f"{k}: {a[k]} vs {b[k]}")
if pa["rows"] != pb["rows"]:
    fail.append("protocol rows (users and order) differ from the parent's decay (SM40v2)")
if pa["full_run"]["lr_schedule"] != pb["full_run"]["lr_schedule"]:
    fail.append("lr_schedule differs")
oa = dict(a["options"]); ob = dict(b["options"])
if oa.get("architecture") == ob.get("architecture") or ob.get("architecture") != os.environ["ARCH"]:
    fail.append(f"architecture {ob.get('architecture')}")
oa.pop("architecture", None); ob.pop("architecture", None)
if oa != ob:
    fail.append("options differ from the parent's decay beyond the architecture: "
                f"{ {k: (oa.get(k), ob.get(k)) for k in set(oa) | set(ob) if oa.get(k) != ob.get(k)} }")
o = b["options"]
if o.get("state_carry") != json.loads(os.environ["SPEC"]):
    fail.append(f"state_carry {o.get('state_carry')}")
if (o.get("chunk_reviews"), o.get("batch_chunks"), o.get("gpu_memory_fraction")) != (32768, 1, 0.8):
    fail.append(f"chunk/batch/gpu fraction {o.get('chunk_reviews')}/{o.get('batch_chunks')}/{o.get('gpu_memory_fraction')}")
d = o.get("distillation", {})
if d.get("kd_source") != "replay" or "cache" in d or d.get("alpha") != 0.9 or d.get("decay_alpha") != 0.5:
    fail.append(f"distillation {d}")
if pb["full_run"]["lr_schedule"] != {"decay_start": 0.0, "decay_end": 1.0, "shape": "one_minus_sin", "final": 0.0}:
    fail.append("schedule is not 7e-4 -> 0 one_minus_sin over the whole epoch")
if fail:
    print("[prune-check] REFUSE:\n  " + "\n  ".join(fail)); raise SystemExit(1)
print(f"[prune-check] decay OK: {b['training_users']} users, {b['training_reviews']:,} reviews (= the parent's SM40v2 decay), {b['lr']}, "
      f"32768 x 1, state_carry {o['state_carry']}, replay targets, gpu_memory_fraction 0.8, architecture {o['architecture']}")
PYEOF
  rm -rf "$REF"
  # options must be accepted by the prefetch allowlist and the carry / replay validators (CPU)
  DEC="$DUSE" ROOT_WT="$WT" "$PY" -X utf8 - <<'PYEOF' || { log "allowlist/replay check failed"; return 1; }
import json, os, re, sys
from pathlib import Path
sys.path.insert(0, os.environ["ROOT_WT"])
o = json.loads((Path(os.environ["DEC"]) / "decay.json").read_text())["options"]
text = (Path(os.environ["ROOT_WT"]) / "scripts" / "gpu_r01_prefetch.py").read_text(encoding="utf-8")
allow = set(re.findall(r'"([a-z_0-9]+)"', text[text.index("set(options) - {"):text.index("or bool(options.get(\"preprocessing_cache\"))")]))
extra = sorted(set(o) - allow)
if extra:
    print("options outside the prefetch allowlist:", extra); sys.exit(1)
from scripts.state_carry_train import validate_options
validate_options(o)
from scripts.expertium_options import ReplayTargets
rt = ReplayTargets(o["distillation"])
print(f"[prune-check] state_carry options valid, replay spec valid (teacher sha256 {str(rt.teacher_sha256)[:12]}), {len(o)} option keys allowlisted")
PYEOF
  if [ "$DRYRUN" = 1 ]; then rm -rf "$DUSE" "$EXP" "$FT"; fi  # scratch of the dry run

  # ---- GPU ticket 2: the decay (smoke first, inside the ticket)
  if [ "$DRYRUN" = 1 ] || [ ! -f "$D/$CKPT" ]; then
    q 240 "SM15 1-epoch decay $ARCH at 32k with strided-lane carry from the fine-tuned e$FT_EPOCH (smoke first)" -- "$BASH_EXE" -c \
      "'$PY' -X utf8 '$DEC_PY' smoke --out '$D' && '$PY' -X utf8 '$DEC_PY' run --out '$D'" || return 1
  fi
  [ "$DRYRUN" = 1 ] || [ -f "$D/$CKPT" ] || { log "no decay checkpoint; stop"; return 1; }

  # ---- GPU tickets 3-6: full-history scoring, 464 users (w8+s8 and fp32) then the 732 pool (w8+s8); rows resume per user
  for pass in 1 2; do
    if [ "$DRYRUN" = 1 ] || [ "$(label_users "$OUT" "$L/w8+s8")" -lt 464 ] || [ "$(label_users "$OUT" "$L/fp32")" -lt 464 ]; then
      q $((pass == 1 ? 25 : 15)) "SM15 scoring $L, 464 users w8+s8 + fp32 (pass $pass)" -- "$PY" -X utf8 "$SC" --model "$L=expertium:$ARCH:$D/$CKPT" \
        --only "$L/w8+s8" --only "$L/fp32" --time-limit 3000 --out "$OUT" || return 1
    fi
  done
  log "464 users scored: w8+s8 $(label_users "$OUT" "$L/w8+s8"), fp32 $(label_users "$OUT" "$L/fp32")"
  for pass in 1 2; do
    if [ "$DRYRUN" = 1 ] || [ "$(label_users "$OUT_POOL" "$L/w8+s8")" -lt 732 ]; then
      q $((pass == 1 ? 40 : 20)) "SM15 scoring $L, 732 pool (pass $pass)" -- "$PY" -X utf8 "$SC" --pool 732 --model "$L=expertium:$ARCH:$D/$CKPT" \
        --only "$L/w8+s8" --time-limit 4800 --out "$OUT_POOL" || return 1
    fi
  done
  log "732-pool users scored: $(label_users "$OUT_POOL" "$L/w8+s8")"
  return 0
}

summary() {  # design 2.3 on the 464 users (decisive), 732 pool reported; vs P_dec, paired
  L=s-w64kd-gate-prune-$ARCH-dec
  OUT=$SM/heldout-prune-$ARCH.jsonl; OUT_POOL=$SM/heldout-prune-$ARCH-pools.jsonl
  "$PY" -X utf8 "$PRUNE" decide --jsonl "$OUT" --label "$L/w8+s8" --fp32-label "$L/fp32" --ref-jsonl "$P_DEC_464" --ref-label "$PARENT_LBL" \
    --fork-jsonl "$FORK_464" --min-users 464 --out "$SM/prune-decision-$ARCH.json"
  cat "$P_DEC_POOL" "$FORK_POOL" > "$SM/prune-ref-pools.jsonl"
  "$PY" -X utf8 "$PRUNE" decide --jsonl "$OUT_POOL" --label "$L/w8+s8" --ref-jsonl "$SM/prune-ref-pools.jsonl" --ref-label "$PARENT_LBL" \
    --fork-jsonl "$SM/prune-ref-pools.jsonl" --min-users 600 --out "$SM/prune-decision-$ARCH-pools.json"
}

cd "$WT" || exit 1
log "started: stage '$STAGE' $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN, ARCH=$ARCH)"
rc=0
case "$STAGE" in
  screen) stage_screen || rc=$?;;
  e1)
    if [ "${SUMMARY_ONLY:-0}" = 1 ]; then summary; exit 0; fi
    preconditions || rc=$?
    [ $rc -eq 0 ] && { main_run || rc=$?; }
    if [ $rc -eq 0 ]; then
      if [ "$DRYRUN" = 1 ]; then
        # decision helper smoke: P_dec against itself must be a PASS with delta 0 (CPU, rows only)
        "$PY" -X utf8 "$PRUNE" decide --jsonl "$P_DEC_464" --label "$PARENT_LBL" --ref-jsonl "$P_DEC_464" --ref-label "$PARENT_LBL" \
          --fork-jsonl "$FORK_464" --min-users 464 --out "$SM/prune-decision-dryrun.json" || rc=$?
      else summary; fi
    fi;;
  *) echo "usage: [ARCH=w64-d2p1u1] [DRYRUN=1] bash $0 screen|e1"; rc=2;;
esac
if [ $rc -eq 0 ]; then
  log "done"
  [ "$DRYRUN" = 1 ] && echo "DRYRUN OK"
else
  log "ended with exit $rc"
fi
exit $rc
