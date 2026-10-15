#!/usr/bin/env bash
# Small-model runbook 2026-10-16 (docs/SMALL_MODEL.md "Queue 2026-10-16"). runbook-next.sh stays untouched; helpers are copies.
#
# Stage `extend` (Q4, branch E): extend scale-w64kd-<TIER>-cached-<RUN_TAG> from 4 to EPOCHS (6) epochs, one ticket per
# epoch, a 100-user score after epoch 5 with the stop rule, a 0.25-epoch decay from e6 into its own folder, 464-user scoring
# of <LBL>-e6 and <LBL>-dec6.
#   TIER=gate EPOCHS=6 RUN_TAG=ctx2048 nohup bash $SM/runbook-2026-10-16.sh extend > $SM/runbook-2026-10-16-extend.log 2>&1 & disown
#
# Why the epoch count can be raised in place (checked on CPU every launch, refuses otherwise):
#   - the LR is constant 7e-4 after a 5-update warmup (expertium_options.default_learning_rate); it depends on the global update
#     offset only, not on the total epoch count. The run has no final_lr_schedule / lr_schedule / optimizer option, and its
#     built-in cool-down (final_decay_epochs) is absent, so epochs 5-6 get exactly the LR a 6-epoch run would have had.
#   - the epoch count appears in two files, both set: <exp>/experiment.json (accuracy_loop train loop) and <exp>/run/run.json
#     (run_full_training.write_protocol: `final = epoch == epochs` only picks the evaluation set, switched off by accuracy_loop
#     (evaluate=False), and final_lr_schedule, which is absent). The per-epoch user order is seeded by split_seed*1000+epoch.
#   - epoch N+1 resumes from epoch-N master.pth/optimizer.pth with update_offset = result.update_offset_end; epochs that are
#     complete are skipped by accuracy_loop.
# Preconditions refused (exit 2): epoch FROM (4) not complete, the old runbook (scale) still running, the e4 decay missing
# (that stage reads run.json and must finish before the epoch count changes). DRYRUN=1 stubs the queue (echo), never writes
# experiment.json/run.json and turns missing-artifact exits into warnings; FROM=<n> lets a dry run use an earlier epoch.
# Environment: TIER, EPOCHS (default 6), RUN_TAG, FROM (default 4), FORCE=1 (continue past the e5 stop rule), DRYRUN=1.
set -u
STAGE=${1:?stage extend}
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
BASH_EXE="C:/Program Files/Git/usr/bin/bash.exe"
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/small-model
AL=$WT/scripts/accuracy_loop.py
DEC_PY=$WT/scripts/expertium_decay.py
SC=$WT/scripts/small_model_heldout.py
SM=$WT/artifacts/small-model
CKPT=expertium/E89-F65e-F77a/master.pth
TRACK=E89-F65e-F77a
OUT=$SM/heldout.jsonl
GATE_USERS=$SM/gate-users.json
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
# missing ARTIFACT-DESCRIPTION: real run = fail (return 1), dry run = warn and continue
need() { if [ "$DRYRUN" = 1 ]; then log "DRYRUN warning (would stop): $1"; return 0; fi; log "$1"; return 1; }

# score LABEL=PATH... -- ONLY users: generic scoring ticket body used for gate (100 users) and finals (464).
extend() {
  TIER=${TIER:?gate or final}; EPOCHS=${EPOCHS:-6}; FROM=${FROM:-4}
  SUFFIX=${RUN_TAG:+-$RUN_TAG}
  NAME=scale-w64kd-$TIER-cached$SUFFIX
  EXP=$WT/artifacts/accuracy/experiments/$NAME
  LBL=s-w64kd-$TIER$SUFFIX
  DEC4=$SM/$NAME-decay          # the existing e4 decay (old runbook)
  DEC6=$SM/$NAME-e6-decay       # new folder
  EP() { echo "$EXP/run/epoch-$(printf "%02d" "$1")/$CKPT"; }
  [ -f "$EXP/experiment.json" ] || { log "no experiment $EXP"; return 2; }
  [ "$EPOCHS" -gt "$FROM" ] || { log "EPOCHS ($EPOCHS) must exceed FROM ($FROM)"; return 2; }

  # ---- preconditions
  cur=$("$PY" -c "import json; print(json.load(open(r'$EXP/experiment.json'))['epochs'])") || return 2
  if [ "$cur" != 4 ] && [ "$cur" != "$EPOCHS" ] && [ "$DRYRUN" != 1 ]; then log "experiment.json epochs is $cur (expect 4 or $EPOCHS)"; return 2; fi
  "$PY" -c "
import json, sys
r = json.load(open(r'$EXP/run/epoch-$(printf "%02d" "$FROM")/expertium/$TRACK/result.json'))
sys.exit(0 if r.get('status') == 'complete' and not r.get('smoke_only') else 1)" 2>/dev/null \
    && [ -f "$(EP "$FROM")" ] || { need "epoch-$(printf "%02d" "$FROM") of $NAME is not complete"; [ "$DRYRUN" = 1 ] || return 2; }
  if ps -ef 2>/dev/null | grep -v grep | grep -q "runbook-next.sh"; then
    need "runbook-next.sh is still running" || return 2
  fi
  [ -f "$DEC4/$CKPT" ] || { need "e4 decay $DEC4/$CKPT missing: the old runbook has not finished"; [ "$DRYRUN" = 1 ] || return 2; }

  # ---- step 1: CPU check that epoch FROM+1 continues epoch FROM unchanged, then the epoch-count edit
  scratch=$SM/ext-check-$NAME
  rm -rf "$scratch"
  EXP="$EXP" SCRATCH="$scratch" FROM="$FROM" EPOCHS="$EPOCHS" ROOT_WT="$WT" TRACK="$TRACK" "$PY" -X utf8 - <<'PYEOF' || return 2
import json, os, shutil, sys
from pathlib import Path
exp, scratch = Path(os.environ["EXP"]), Path(os.environ["SCRATCH"])
frm, epochs, track = int(os.environ["FROM"]), int(os.environ["EPOCHS"]), os.environ["TRACK"]
sys.path.insert(0, os.environ["ROOT_WT"])
from scripts import run_full_training as rft
fail = []
run = json.loads((exp / "run" / "run.json").read_text())
experiment = json.loads((exp / "experiment.json").read_text())
# LR/schedule independence of the total epoch count
for key in ("final_lr_schedule", "final_decay_epochs"):
    if run.get(key):
        fail.append(f"run.json has {key}={run[key]!r}: LR would depend on the last epoch")
for where, options in (("run.json options", run["options"]), ("experiment options", experiment["options"]),
                       ("option_overrides", experiment["option_overrides"])):
    for key in ("lr_schedule", "optimizer"):
        if key in options:
            fail.append(f"{where} has {key}: LR/optimizer may depend on the schedule")
if experiment.get("epoch_runner") or experiment.get("init"):
    fail.append("experiment has an epoch_runner/init hook")
if run["epochs"] not in (4, epochs) or experiment["epochs"] not in (4, epochs):
    if os.environ.get("DRYRUN") != "1":
        fail.append(f"epochs run.json={run['epochs']} experiment.json={experiment['epochs']}")
# write the epoch FROM+1 protocol in a scratch run folder under the worktree (write_protocol needs ROOT-relative paths)
sr = scratch / "run"
(sr / f"epoch-{frm:02d}" / "expertium" / track).mkdir(parents=True)
shutil.copy(exp / "run" / f"epoch-{frm:02d}" / "expertium" / track / "result.json", sr / f"epoch-{frm:02d}" / "expertium" / track / "result.json")
new = dict(run, epochs=epochs)
(sr / "run.json").write_text(json.dumps(new))
rft.write_protocol(sr, frm + 1, new)
nxt = json.loads((sr / f"epoch-{frm + 1:02d}" / "protocol.json").read_text())
real = json.loads((exp / "run" / f"epoch-{frm:02d}" / "protocol.json").read_text())
IGN = {"epoch", "evaluate", "benchmark_db", "evaluation_set", "control_dir", "init_master", "init_optimizer", "update_offset", "resumed"}
a = {k: v for k, v in real["full_run"].items() if k not in IGN}
b = {k: v for k, v in nxt["full_run"].items() if k not in IGN}
# evaluation settings are overridden by accuracy_loop for the real run (evaluate False); compare the rest
a.pop("fast_evaluation", None), b.pop("fast_evaluation", None)
if a != b:
    fail.append(f"full_run fields differ: epoch {frm} {a} vs epoch {frm + 1} {b}")
for k in ("training", "execution_mode", "dataset_path", "dataset_revision"):
    if real[k] != nxt[k]:
        fail.append(f"protocol {k} differs: {real[k]} vs {nxt[k]}")
if "lr_schedule" in real["full_run"] or "lr_schedule" in nxt["full_run"]:
    fail.append("lr_schedule present in a protocol")
prev = json.loads((exp / "run" / f"epoch-{frm:02d}" / "expertium" / track / "result.json").read_text())
fr = nxt["full_run"]
if not fr["init_master"].endswith(f"epoch-{frm:02d}/expertium/{track}/master.pth"):
    fail.append(f"init_master is {fr['init_master']}, not epoch-{frm:02d}")
if not fr["init_optimizer"].endswith(f"epoch-{frm:02d}/expertium/{track}/optimizer.pth"):
    fail.append(f"init_optimizer is {fr['init_optimizer']}")
if fr["update_offset"] != prev["update_offset_end"]:
    fail.append(f"update_offset {fr['update_offset']} != {prev['update_offset_end']}")
if frm > 1:
    pf = real["full_run"]
    if pf.get("update_offset") is None or not pf["init_master"].endswith(f"epoch-{frm - 1:02d}/expertium/{track}/master.pth"):
        fail.append("real protocol of epoch FROM does not resume from epoch FROM-1 (unexpected chain)")
# default LR for the first update of the new epoch (warmup finished: offset >= 5)
sys.path.insert(0, os.environ["ROOT_WT"])
from scripts.expertium_options import default_learning_rate
lr_a, lr_b = default_learning_rate(0, prev["update_offset_end"]), default_learning_rate(10 ** 6, prev["update_offset_end"])
if not (lr_a == lr_b == 7e-4):
    fail.append(f"LR not constant 7e-4 at the epoch boundary: {lr_a} {lr_b}")
shutil.rmtree(scratch, ignore_errors=True)
if fail:
    print("[extend-check] REFUSE:\n  " + "\n  ".join(fail)); sys.exit(1)
print(f"[extend-check] OK: epoch {frm + 1} protocol = epoch {frm} protocol except resume from epoch-{frm:02d} "
      f"(update_offset {prev['update_offset_end']}), LR constant 7e-4, no lr_schedule/optimizer/final_lr_schedule/final_decay_epochs, "
      f"order seed {run['split_seed']}*1000+epoch")
PYEOF
  if [ "$cur" != "$EPOCHS" ]; then
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN: would set epochs $cur -> $EPOCHS in experiment.json and run/run.json (backups *.bak-4ep); NOT done"
    else
      for f in "$EXP/experiment.json" "$EXP/run/run.json"; do
        [ -f "$f.bak-4ep" ] || cp "$f" "$f.bak-4ep"
        "$PY" -c "
import json, os
p = r'$f'; d = json.load(open(p)); d['epochs'] = $EPOCHS
d['extended_from_epochs'] = 4
tmp = p + '.tmp'; json.dump(d, open(tmp, 'w'), indent=1); os.replace(tmp, p)" || return 2
      done
      log "epochs set to $EPOCHS in experiment.json and run/run.json (backups *.bak-4ep)"
    fi
  else log "epochs already $EPOCHS; resuming"; fi

  # ---- step 2: one ticket per epoch (smoke first), the 100-user score and the stop rule after e5
  OPTS=$("$PY" -c "import json; print(json.dumps(json.load(open(r'$EXP/experiment.json'))['option_overrides']))") || return 2
  SMOKE=$SM/smoke-prep-$NAME-ext
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: would prepare smoke scratch $SMOKE (decay prepare of scale-w64kd-gate epoch 3, 1%, run options)"
  else
    rm -rf "$SMOKE"
    "$PY" -X utf8 "$DEC_PY" prepare --source-run "$WT/artifacts/accuracy/experiments/scale-w64kd-gate/run" --epoch 3 --out "$SMOKE" \
      --fraction 0.01 --evaluate none --seed 1 --extra-options "$OPTS" > /dev/null || return 1
  fi
  for epoch in $(seq $((FROM + 1)) "$EPOCHS"); do
    if [ ! -f "$(EP "$epoch")" ]; then
      q 66 "extend $NAME epoch $epoch/$EPOCHS ($TIER, smoke first)" -- "$BASH_EXE" -c \
        "'$PY' -X utf8 '$DEC_PY' smoke --out '$SMOKE' && '$PY' -X utf8 '$AL' train '$NAME' --epochs-per-job 1"
      rc=$?
      [ $rc -eq 0 ] || [ $rc -eq 3 ] || { log "training failed (exit $rc), stop"; return 1; }
      [ -f "$(EP "$epoch")" ] || { need "no epoch-$epoch checkpoint after training" || return 1; }
    fi
    if [ "$epoch" -eq $((FROM + 1)) ] && [ "$epoch" -lt "$EPOCHS" ]; then
      decision=$SM/gate-$NAME-e$epoch-extend.json
      q 6 "extend gate: score $LBL-e$FROM + $LBL-e$epoch on the 100 gate users" -- "$PY" -X utf8 "$SC" \
        --users-file "$GATE_USERS" --model "$LBL-e$FROM=expertium:w64:$(EP "$FROM")" --model "$LBL-e$epoch=expertium:w64:$(EP "$epoch")" \
        --only "$LBL-e$FROM/w8+s8" --only "$LBL-e$epoch/w8+s8" --time-limit 600 --out "$OUT" || return 1
      OUT="$OUT" USERS="$GATE_USERS" A="$LBL-e$FROM/w8+s8" B="$LBL-e$epoch/w8+s8" DEC="$decision" WTR="$WT" "$PY" -X utf8 - <<'PYEOF'
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.environ["WTR"])
from scripts import small_model_gate as g
users = json.loads(Path(os.environ["USERS"]).read_text())["users"]
a, b = os.environ["A"], os.environ["B"]
per = g.load(os.environ["OUT"], [a, b], set(users))
missing = {l: len([u for u in users if u not in per[l]]) for l in (a, b)}
if any(missing.values()):
    json.dump({"verdict": "incomplete", "missing_users": missing}, open(os.environ["DEC"], "w"), indent=1)
    print(f"[extend-gate] incomplete rows {missing}"); sys.exit(1)
res, gain = {}, {}
for head, name in g.HEADS.items():
    ea, eb = g.rw(*g.weighted(per, a, users, head)), g.rw(*g.weighted(per, b, users, head))
    res[name] = {"before": ea, "after": eb, "gain": ea - eb}
stop = all(v["gain"] <= 0 for v in res.values())
json.dump({"verdict": "stop" if stop else "go", "rule": "stop if before - after <= 0 on both heads", "heads": res}, open(os.environ["DEC"], "w"), indent=1)
print("[extend-gate] " + ("STOP" if stop else "GO") + " " + json.dumps(res))
sys.exit(10 if stop else 0)
PYEOF
      rc=$?
      if [ $rc -eq 10 ]; then
        if [ "${FORCE:-0}" = 1 ]; then log "e$epoch flattened ($decision), FORCE=1: continuing"
        else log "e$epoch flattened vs e$FROM on both heads ($decision): stop after epoch $epoch. Re-launch with FORCE=1 to continue."; return 0; fi
      elif [ $rc -ne 0 ]; then log "extend gate failed (exit $rc)"; [ "$DRYRUN" = 1 ] && log "DRYRUN: no rows, continuing" || return 1
      fi
    fi
  done
  EN=$(EP "$EPOCHS")

  # ---- step 3: decay from e6 into its own folder (same recipe as the run's existing decay stage)
  D=$DEC6/$CKPT
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: would prepare $DEC6 (--source-run $EXP/run --epoch $EPOCHS --fraction 0.25 --shape one_minus_sin --evaluate none --seed 1)"; ok=1
  elif [ -d "$DEC6" ]; then ok=1
  else "$PY" -X utf8 "$DEC_PY" prepare --source-run "$EXP/run" --epoch "$EPOCHS" --out "$DEC6" --fraction 0.25 --shape one_minus_sin \
      --evaluate none --seed 1 && ok=1 || ok=0; fi
  [ "$ok" = 1 ] || { log "decay prepare failed"; return 1; }
  [ -f "$D" ] || q 27 "extend $NAME: 0.25-epoch decay from e$EPOCHS (smoke first)" -- "$BASH_EXE" -c \
      "'$PY' -X utf8 '$DEC_PY' smoke --out '$DEC6' && '$PY' -X utf8 '$DEC_PY' run --out '$DEC6'" || return 1

  # ---- step 4: 464-user scoring of e6 and dec6, and a resume ticket only if users remain
  MODELS="--model $LBL-e$EPOCHS=expertium:w64:$EN --model $LBL-dec$EPOCHS=expertium:w64:$D"
  LABELS="--only $LBL-e$EPOCHS/w8+s8 --only $LBL-dec$EPOCHS/w8+s8"
  # shellcheck disable=SC2086
  q 30 "extend $NAME: full-history scoring of e$EPOCHS and dec$EPOCHS (464 users)" -- "$PY" -X utf8 "$SC" $MODELS $LABELS --time-limit 1900 --out "$OUT" || return 1
  remaining=$(OUT="$OUT" L1="$LBL-e$EPOCHS/w8+s8" L2="$LBL-dec$EPOCHS/w8+s8" "$PY" -X utf8 - <<'PYEOF'
import json, os
want = {"fork/w8+s8": set(), os.environ["L1"]: set(), os.environ["L2"]: set()}
for line in open(os.environ["OUT"], encoding="utf-8"):
    if '"label"' in line:
        r = json.loads(line)
        if r["label"] in want and "benchmark" in r:
            want[r["label"]].add(r["user_id"])
total = want["fork/w8+s8"]
print(max(len(total - want[os.environ["L1"]]), len(total - want[os.environ["L2"]])) if total else 0)
PYEOF
)
  log "users still unscored after the scoring ticket (vs the fork's users): ${remaining:-?}"
  if [ "$DRYRUN" = 1 ] || [ "${remaining:-1}" != 0 ]; then
    # shellcheck disable=SC2086
    q 15 "extend $NAME: scoring resume (only if users remain)" -- "$PY" -X utf8 "$SC" $MODELS $LABELS --time-limit 1200 --out "$OUT"
  fi
  return 0
}

cd "$WT" || exit 1
case "$STAGE" in
extend) extend; rc=$?; [ $rc -eq 0 ] && log "done: $STAGE" || log "$STAGE ended with exit $rc"; exit $rc ;;
*) echo "unknown stage $STAGE (extend)"; exit 2 ;;
esac
