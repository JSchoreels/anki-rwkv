#!/usr/bin/env bash
# Small-model runbook 2026-10-16b (docs/SMALL_MODEL.md "Re-assessment 2026-10-16 ~03:30 (G and Q7)").
# runbook-2026-10-16.sh (stage extend, Q4) and runbook-next.sh stay untouched; helpers below are copies.
#
# Stage `cont` (Q7, SM19): continue scale-w64kd-<TIER>-cached-<RUN_TAG> from epoch FROM to epoch TO in a NEW experiment
# scale-w64kd-<TIER>-cached-ctx<CHUNK>-cont at CHUNK-review chunks x BATCH_CHUNKS (default 32768/CHUNK: 32,768 reviews per
# update, unchanged, so the LR schedule, warmup offset and optimizer state carry over). The source run is only READ.
#   primary    TIER=gate FROM=6 TO=8 CHUNK=4096 nohup bash $SM/runbook-2026-10-16b.sh cont > $SM/runbook-2026-10-16b-cont.log 2>&1 & disown
#   fallback 1 TIER=gate FROM=6 TO=8 CHUNK=2048 nohup bash $SM/runbook-2026-10-16b.sh cont > $SM/runbook-2026-10-16b-cont.log 2>&1 & disown
#              (epochs 7-8 at 2,048 = the same stage with CHUNK=2048; new experiment ...-ctx2048-cont, labels ...-ctx2048c-e<N>)
# Steps: (0) refuse unless the source epoch FROM is complete, the Q4 `extend` runbook has exited and its 464-user rows of
#   <LBL>-e<FROM> and <LBL>-dec<FROM> cover the 100 gate users; (1) copy experiment.json, run/run.json, run/epoch-NN stubs
#   (result.json + protocol.json) and the epoch FROM checkpoint (master.pth, optimizer.pth) into the new experiment and set
#   chunk_reviews / batch_chunks / epochs / name / note; (2) CPU check that epoch FROM+1 resumes from epoch FROM with
#   update_offset = its update_offset_end, the same LR/optimizer fields and only the chunk fields changed, option keys inside the
#   gpu_r01_prefetch allowlist and identical to the source's; (3) one ticket per epoch (smoke first) with a 100-user score of
#   <NEW_LBL>-e<N> after each epoch and the stop rule after FROM+1 (e<FROM> - e<FROM+1> <= 0 on both heads, FORCE=1 continues);
#   (4) 0.25-epoch decay from the last epoch at CHUNK into $SM/<NEW>-decay; (5) 464-user scoring of e<last> and dec<last>,
#   resume ticket only if users remain.
# Environment: TIER (required), FROM (6), TO (8), CHUNK (4096), BATCH_CHUNKS (32768/CHUNK), RUN_TAG (ctx2048), SRC_LBL
# (s-w64kd-TIER-RUN_TAG), FIRST_MIN (queue estimate of the first epoch), FORCE=1, DRYRUN=1.
# DRYRUN=1 stubs the queue (echo), builds the new experiment, the smoke scratch and the decay only in scratch folders under
# $SM (deleted afterwards), never writes into the source run or the real new experiment, and turns missing-artifact
# exits into warnings; use FROM=4 TO=6 to dry-run before epoch 6 exists.
set -u
STAGE=${1:?stage cont}
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

# rows_ok LABEL: 0 when heldout.jsonl has w8+s8 rows (both heads) for all 100 gate users under LABEL
rows_ok() {
  OUT="$OUT" USERS="$GATE_USERS" L="$1" WTR="$WT" "$PY" -X utf8 - <<'PYEOF'
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.environ["WTR"])
from scripts import small_model_gate as g
users = json.loads(Path(os.environ["USERS"]).read_text())["users"]
per = g.load(os.environ["OUT"], [os.environ["L"]], set(users))
sys.exit(0 if all(u in per[os.environ["L"]] for u in users) else 1)
PYEOF
}

cont() {
  TIER=${TIER:?gate or final}; FROM=${FROM:-6}; TO=${TO:-8}; CHUNK=${CHUNK:-4096}; RUN_TAG=${RUN_TAG:-ctx2048}
  BATCH_CHUNKS=${BATCH_CHUNKS:-$((32768 / CHUNK))}
  [ "$TO" -gt "$FROM" ] || { log "TO ($TO) must exceed FROM ($FROM)"; return 2; }
  SRC=scale-w64kd-$TIER-cached-$RUN_TAG
  SRC_EXP=$WT/artifacts/accuracy/experiments/$SRC
  SRC_LBL=${SRC_LBL:-s-w64kd-$TIER-$RUN_TAG}   # override when the source labels differ (ctx8192-cont scored as s-w64kd-gate-ctx8192c)
  NAME=scale-w64kd-$TIER-cached-ctx${CHUNK}-cont
  EXP=$WT/artifacts/accuracy/experiments/$NAME
  LBL=s-w64kd-$TIER-ctx${CHUNK}c
  [ "$NAME" != "$SRC" ] || { log "new experiment equals the source"; return 2; }
  EP() { echo "$EXP/run/epoch-$(printf "%02d" "$1")/$CKPT"; }
  FIRST_MIN=${FIRST_MIN:-66}; [ "$CHUNK" -gt 2048 ] && [ "$FIRST_MIN" = 66 ] && FIRST_MIN=80   # cold KD cache at a new chunk length for the first epoch
  [ -f "$SRC_EXP/experiment.json" ] || { log "no source experiment $SRC_EXP"; return 2; }

  # ---- step 0: preconditions on the source (read-only)
  "$PY" -c "
import json, sys
r = json.load(open(r'$SRC_EXP/run/epoch-$(printf "%02d" "$FROM")/expertium/$TRACK/result.json'))
sys.exit(0 if r.get('status') == 'complete' and not r.get('smoke_only') else 1)" 2>/dev/null \
    && [ -f "$SRC_EXP/run/epoch-$(printf "%02d" "$FROM")/$CKPT" ] \
    || { need "epoch-$(printf "%02d" "$FROM") of $SRC is not complete"; [ "$DRYRUN" = 1 ] || return 2; }
  if ps -ef 2>/dev/null | grep -v grep | grep -E "runbook-2026-10-16\.sh|runbook-next\.sh" | grep -q .; then
    need "runbook-2026-10-16.sh (Q4 extend) or runbook-next.sh is still running" || return 2
  fi
  for l in "$SRC_LBL-e$FROM" "$SRC_LBL-dec$FROM"; do
    rows_ok "$l/w8+s8" || { need "Q4 scoring incomplete: $l/w8+s8 lacks rows for some of the 100 gate users in $OUT"; [ "$DRYRUN" = 1 ] || return 2; }
  done
  if [ -f "$EXP/experiment.json" ] && [ "$DRYRUN" != 1 ]; then log "$NAME exists: resuming (no copy)"; BUILD=0; DEST=$EXP
  elif [ "$DRYRUN" = 1 ]; then BUILD=1; DEST=$SM/cont-check-$NAME
  else BUILD=1; DEST=$EXP.partial; fi

  # ---- step 1: copy source -> DEST (read-only on the source), set chunk fields
  if [ "$BUILD" = 1 ]; then
    rm -rf "$DEST"
    SRC_EXP="$SRC_EXP" DEST="$DEST" FROM="$FROM" TO="$TO" CHUNK="$CHUNK" BATCH_CHUNKS="$BATCH_CHUNKS" NAME="$NAME" SRC="$SRC" TRACK="$TRACK" \
      "$PY" -X utf8 - <<'PYEOF' || return 2
import json, os, shutil
from datetime import datetime, timezone
from pathlib import Path
src, dest = Path(os.environ["SRC_EXP"]), Path(os.environ["DEST"])
frm, to, chunk, bc = (int(os.environ[k]) for k in ("FROM", "TO", "CHUNK", "BATCH_CHUNKS"))
track, name = os.environ["TRACK"], os.environ["NAME"]
(dest / "run").mkdir(parents=True)
for n in range(1, frm + 1):
    e = f"epoch-{n:02d}"
    t = Path(e) / "expertium" / track
    (dest / "run" / t).mkdir(parents=True)
    shutil.copy2(src / "run" / e / "protocol.json", dest / "run" / e / "protocol.json")
    shutil.copy2(src / "run" / t / "result.json", dest / "run" / t / "result.json")
    if n == frm:
        for f in ("master.pth", "optimizer.pth"):
            shutil.copy2(src / "run" / t / f, dest / "run" / t / f)
note = (f"{name}: continuation of {os.environ['SRC']} from epoch {frm} to {to} at {chunk}-review chunks x {bc} "
        f"({chunk * bc} reviews/update), docs/SMALL_MODEL.md SM19")
exp = json.loads((src / "experiment.json").read_text())
run = json.loads((src / "run" / "run.json").read_text())
exp.update(name=name, note=note, epochs=to, created_utc=datetime.now(timezone.utc).isoformat(),
           continued_from={"experiment": os.environ["SRC"], "epoch": frm})
for opts in (exp["options"], exp["option_overrides"], run["options"]):
    opts["chunk_reviews"], opts["batch_chunks"] = chunk, bc
run.update(epochs=to, continued_from={"experiment": os.environ["SRC"], "epoch": frm})
(dest / "experiment.json").write_text(json.dumps(exp, indent=1) + "\n", encoding="utf-8")
(dest / "run" / "run.json").write_text(json.dumps(run, indent=1) + "\n", encoding="utf-8")
print(f"[cont-copy] {dest}: epochs 1..{frm} stubs, epoch {frm} checkpoint, chunk {chunk} x {bc}, epochs {to}")
PYEOF
  fi

  # ---- step 2: CPU check of the resume (DEST run folder; protocol of epoch FROM+1 is written in a scratch folder)
  scratch=$SM/cont-protocol-$NAME
  rm -rf "$scratch"
  DEST="$DEST" SRC_EXP="$SRC_EXP" SCRATCH="$scratch" FROM="$FROM" TO="$TO" CHUNK="$CHUNK" BATCH_CHUNKS="$BATCH_CHUNKS" \
    ROOT_WT="$WT" TRACK="$TRACK" "$PY" -X utf8 - <<'PYEOF' || { rm -rf "$scratch"; [ "$BUILD" = 1 ] && [ "$DEST" != "$EXP" ] && rm -rf "$DEST"; return 2; }
import json, os, re, shutil, sys
from pathlib import Path
dest, src, scratch = Path(os.environ["DEST"]), Path(os.environ["SRC_EXP"]), Path(os.environ["SCRATCH"])
frm, to, chunk, bc = (int(os.environ[k]) for k in ("FROM", "TO", "CHUNK", "BATCH_CHUNKS"))
track = os.environ["TRACK"]
sys.path.insert(0, os.environ["ROOT_WT"])
from scripts import run_full_training as rft
from scripts.expertium_options import default_learning_rate
fail = []
run = json.loads((dest / "run" / "run.json").read_text())
sr_ = json.loads((src / "run" / "run.json").read_text())
exp = json.loads((dest / "experiment.json").read_text())
# 1. chunk fields and epochs in all three places; nothing else changed in the options
for where, o in (("run.json", run["options"]), ("experiment options", exp["options"]), ("option_overrides", exp["option_overrides"])):
    if (o.get("chunk_reviews"), o.get("batch_chunks")) != (chunk, bc):
        fail.append(f"{where}: chunk_reviews/batch_chunks = {o.get('chunk_reviews')}/{o.get('batch_chunks')}, want {chunk}/{bc}")
a = {k: v for k, v in sr_["options"].items() if k not in ("chunk_reviews", "batch_chunks")}
b = {k: v for k, v in run["options"].items() if k not in ("chunk_reviews", "batch_chunks")}
if a != b:
    fail.append("run.json options differ from the source beyond chunk_reviews/batch_chunks")
if run["epochs"] != to or exp["epochs"] != to:
    fail.append(f"epochs run.json={run['epochs']} experiment.json={exp['epochs']}, want {to}")
# 2. LR/optimizer independence of the epoch count and chunk length
for key in ("final_lr_schedule", "final_decay_epochs"):
    if run.get(key):
        fail.append(f"run.json has {key}")
for where, o in (("run.json options", run["options"]), ("option_overrides", exp["option_overrides"])):
    for key in ("lr_schedule", "optimizer"):
        if key in o:
            fail.append(f"{where} has {key}")
if exp.get("epoch_runner") or exp.get("init"):
    fail.append("experiment has an epoch_runner/init hook")
if abs(run["options"]["batch_chunks"] * run["options"]["chunk_reviews"] - 32768) > 0 and os.environ.get("ALLOW_UPDATE_SIZE") != "1":
    fail.append(f"reviews per update {run['options']['batch_chunks'] * run['options']['chunk_reviews']} != 32768 (the optimizer state/LR were tuned for it)")
# 3. option keys inside the prefetch allowlist (same key set as the source, whose training ran)
text = (Path(os.environ["ROOT_WT"]) / "scripts" / "gpu_r01_prefetch.py").read_text(encoding="utf-8")
allow = set(re.findall(r'"([a-z_0-9]+)"', text[text.index("set(options) - {"):text.index("or bool(options.get(\"preprocessing_cache\"))")]))
extra = sorted(set(run["options"]) - allow)
if extra:
    fail.append(f"options outside the gpu_r01_prefetch allowlist: {extra}")
if set(run["options"]) != set(sr_["options"]):
    fail.append("option key set differs from the source run")
cache = run["options"].get("distillation", {}).get("cache")
if cache and not Path(cache).is_dir():
    fail.append(f"KD cache {cache} not found")
# 4. protocol of epoch FROM+1 written from the copied run
prev_p = dest / "run" / f"epoch-{frm:02d}" / "expertium" / track / "result.json"
prev = json.loads(prev_p.read_text())
if prev.get("status") != "complete" or not (prev_p.parent / "master.pth").exists() or not (prev_p.parent / "optimizer.pth").exists():
    fail.append("copied epoch FROM is not complete / lacks checkpoint")
sr = scratch / "run"
(sr / f"epoch-{frm:02d}" / "expertium" / track).mkdir(parents=True)
shutil.copy(prev_p, sr / f"epoch-{frm:02d}" / "expertium" / track / "result.json")
(sr / "run.json").write_text(json.dumps(run))
rft.write_protocol(sr, frm + 1, run)
nxt = json.loads((sr / f"epoch-{frm + 1:02d}" / "protocol.json").read_text())
real = json.loads((src / "run" / f"epoch-{frm:02d}" / "protocol.json").read_text())
IGN = {"epoch", "evaluate", "benchmark_db", "evaluation_set", "control_dir", "init_master", "init_optimizer", "update_offset", "resumed"}
fa = {k: v for k, v in real["full_run"].items() if k not in IGN}
fb = {k: v for k, v in nxt["full_run"].items() if k not in IGN}
fa.pop("fast_evaluation", None), fb.pop("fast_evaluation", None)
if fa != fb:
    fail.append(f"full_run fields differ: {fa} vs {fb}")
for k in ("training", "execution_mode", "dataset_path", "dataset_revision"):
    if real[k] != nxt[k]:
        fail.append(f"protocol {k} differs: {real[k]} vs {nxt[k]}")
if "lr_schedule" in real["full_run"] or "lr_schedule" in nxt["full_run"]:
    fail.append("lr_schedule present in a protocol")
fr = nxt["full_run"]
if not fr["init_master"].endswith(f"epoch-{frm:02d}/expertium/{track}/master.pth"):
    fail.append(f"init_master {fr['init_master']}")
if not fr["init_optimizer"].endswith(f"epoch-{frm:02d}/expertium/{track}/optimizer.pth"):
    fail.append(f"init_optimizer {fr['init_optimizer']}")
if fr["update_offset"] != prev["update_offset_end"]:
    fail.append(f"update_offset {fr['update_offset']} != {prev['update_offset_end']}")
# the per-epoch order is seeded by split_seed*1000+epoch: the same rows as the source would have used for this epoch number
if not (default_learning_rate(0, prev["update_offset_end"]) == default_learning_rate(10 ** 6, prev["update_offset_end"]) == 7e-4):
    fail.append("LR not constant 7e-4 at the epoch boundary")
shutil.rmtree(scratch, ignore_errors=True)
if fail:
    print("[cont-check] REFUSE:\n  " + "\n  ".join(fail)); sys.exit(1)
print(f"[cont-check] OK: epoch {frm + 1} resumes from epoch-{frm:02d} (update_offset {prev['update_offset_end']}, LR constant 7e-4, "
      f"AdamW state copied), chunk_reviews {chunk} x batch_chunks {bc} = {chunk * bc} reviews/update, "
      f"{len(run['options'])} option keys all in the prefetch allowlist, only chunk fields differ from the source")
PYEOF
  if [ "$BUILD" = 1 ] && [ "$DRYRUN" != 1 ]; then rm -rf "$EXP"; mv "$DEST" "$EXP" || return 2; DEST=$EXP; log "created $EXP"; fi

  # ---- decay prepare check in dry run (scratch, source = the DEST run), then clean up
  OPTS=$("$PY" -c "import json; print(json.dumps(json.load(open(r'$DEST/experiment.json'))['option_overrides']))") || return 2
  if [ "$DRYRUN" = 1 ]; then
    DCHK=$SM/cont-check-decay-$NAME; rm -rf "$DCHK"
    "$PY" -X utf8 "$DEC_PY" prepare --source-run "$DEST/run" --epoch "$FROM" --out "$DCHK" --fraction 0.25 --shape one_minus_sin \
      --evaluate none --seed 1 > /dev/null || { log "DRYRUN: decay prepare check failed"; rm -rf "$DCHK" "$DEST"; return 1; }
    "$PY" -c "
import json; d = json.load(open(r'$DCHK/decay.json'))['options']
print('[cont-check] decay prepare options: chunk_reviews', d['chunk_reviews'], 'batch_chunks', d['batch_chunks'], 'decay_alpha', d['distillation']['decay_alpha'])
assert d['chunk_reviews'] == $CHUNK and d['batch_chunks'] == $BATCH_CHUNKS" || { rm -rf "$DCHK" "$DEST"; return 1; }
    rm -rf "$DCHK"
  fi

  # ---- step 3: one ticket per epoch (smoke first), 100-user score after each, stop rule after FROM+1
  SMOKE=$SM/smoke-prep-$NAME
  rm -rf "$SMOKE"
  "$PY" -X utf8 "$DEC_PY" prepare --source-run "$WT/artifacts/accuracy/experiments/scale-w64kd-gate/run" --epoch 3 --out "$SMOKE" \
    --fraction 0.01 --evaluate none --seed 1 --extra-options "$OPTS" > /dev/null || return 1
  LAST=$TO
  for epoch in $(seq $((FROM + 1)) "$TO"); do
    minutes=66; [ "$epoch" -eq $((FROM + 1)) ] && minutes=$FIRST_MIN
    if [ "$DRYRUN" = 1 ] || [ ! -f "$(EP "$epoch")" ]; then
      q "$minutes" "cont $NAME epoch $epoch/$TO ($TIER, chunk $CHUNK x $BATCH_CHUNKS, smoke first)" -- "$BASH_EXE" -c \
        "'$PY' -X utf8 '$DEC_PY' smoke --out '$SMOKE' && '$PY' -X utf8 '$AL' train '$NAME' --epochs-per-job 1"
      rc=$?
      [ $rc -eq 0 ] || [ $rc -eq 3 ] || { log "training failed (exit $rc), stop"; return 1; }
      [ -f "$(EP "$epoch")" ] || { need "no epoch-$epoch checkpoint after training" || return 1; }
    fi
    if [ "$DRYRUN" = 1 ] || ! rows_ok "$LBL-e$epoch/w8+s8"; then
      q 4 "cont gate: score $LBL-e$epoch on the 100 gate users" -- "$PY" -X utf8 "$SC" \
        --users-file "$GATE_USERS" --model "$LBL-e$epoch=expertium:w64:$(EP "$epoch")" --only "$LBL-e$epoch/w8+s8" --time-limit 600 --out "$OUT" || return 1
    fi
    if [ "$epoch" -eq $((FROM + 1)) ] && [ "$epoch" -lt "$TO" ]; then
      decision=$SM/gate-$NAME-e$epoch.json
      OUT="$OUT" USERS="$GATE_USERS" A="$SRC_LBL-e$FROM/w8+s8" B="$LBL-e$epoch/w8+s8" DEC="$decision" WTR="$WT" "$PY" -X utf8 - <<'PYEOF'
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
    print(f"[cont-gate] incomplete rows {missing}"); sys.exit(1)
res = {}
for head, name in g.HEADS.items():
    ea, eb = g.rw(*g.weighted(per, a, users, head)), g.rw(*g.weighted(per, b, users, head))
    res[name] = {"before": ea, "after": eb, "gain": ea - eb}
stop = all(v["gain"] <= 0 for v in res.values())
json.dump({"verdict": "stop" if stop else "go", "rule": "stop if before - after <= 0 on both heads", "heads": res}, open(os.environ["DEC"], "w"), indent=1)
print("[cont-gate] " + ("STOP" if stop else "GO") + " " + json.dumps(res))
sys.exit(10 if stop else 0)
PYEOF
      rc=$?
      if [ $rc -eq 10 ]; then
        if [ "${FORCE:-0}" = 1 ]; then log "e$epoch flattened ($decision), FORCE=1: continuing"
        else log "e$epoch flattened vs e$FROM on both heads ($decision): skipping the remaining epochs, decay from e$epoch"; LAST=$epoch; break; fi
      elif [ $rc -ne 0 ]; then log "cont gate failed (exit $rc)"; [ "$DRYRUN" = 1 ] && log "DRYRUN: no rows, continuing" || return 1
      fi
    fi
  done
  EN=$(EP "$LAST")

  # ---- step 4: 0.25-epoch decay from the last epoch at CHUNK x BATCH_CHUNKS (run options carry them)
  DECF=$SM/$NAME-decay
  D=$DECF/$CKPT
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: would prepare $DECF (--source-run $EXP/run --epoch $LAST --fraction 0.25 --shape one_minus_sin --evaluate none --seed 1)"; ok=1
  elif [ -d "$DECF" ]; then ok=1
  else "$PY" -X utf8 "$DEC_PY" prepare --source-run "$EXP/run" --epoch "$LAST" --out "$DECF" --fraction 0.25 --shape one_minus_sin \
      --evaluate none --seed 1 && ok=1 || ok=0; fi
  [ "$ok" = 1 ] || { log "decay prepare failed"; return 1; }
  [ -f "$D" ] || q 27 "cont $NAME: 0.25-epoch decay from e$LAST at chunk $CHUNK (smoke first)" -- "$BASH_EXE" -c \
      "'$PY' -X utf8 '$DEC_PY' smoke --out '$DECF' && '$PY' -X utf8 '$DEC_PY' run --out '$DECF'" || return 1

  # ---- step 5: 464-user scoring and a resume ticket only if users remain
  MODELS="--model $LBL-e$LAST=expertium:w64:$EN --model $LBL-dec$LAST=expertium:w64:$D"
  LABELS="--only $LBL-e$LAST/w8+s8 --only $LBL-dec$LAST/w8+s8"
  # shellcheck disable=SC2086
  q 30 "cont $NAME: full-history scoring of e$LAST and dec$LAST (464 users)" -- "$PY" -X utf8 "$SC" $MODELS $LABELS --time-limit 3000 --out "$OUT" || return 1
  remaining=$(OUT="$OUT" L1="$LBL-e$LAST/w8+s8" L2="$LBL-dec$LAST/w8+s8" "$PY" -X utf8 - <<'PYEOF'
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
    q 15 "cont $NAME: scoring resume (only if users remain)" -- "$PY" -X utf8 "$SC" $MODELS $LABELS --time-limit 1200 --out "$OUT"
  fi
  [ "$DRYRUN" = 1 ] && rm -rf "$DEST" "$SMOKE"
  return 0
}

cd "$WT" || exit 1
case "$STAGE" in
cont) cont; rc=$?; [ $rc -eq 0 ] && log "done: $STAGE" || log "$STAGE ended with exit $rc"; exit $rc ;;
*) echo "unknown stage $STAGE (cont)"; exit 2 ;;
esac
