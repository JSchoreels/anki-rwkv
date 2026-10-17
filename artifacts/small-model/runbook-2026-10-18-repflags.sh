#!/usr/bin/env bash
# Small-model runbook 2026-10-18 (SM30, docs/SMALL_MODEL.md "Review-number flag inputs (SM30)"): the paired GPU test of
# the review-number flag inputs. Same recipe as `runbook-2026-10-16b.sh cont` with FROM=6 TO=8 CHUNK=2048 (which made
# s-w64kd-gate-ctx2048c-e7/-e8/-dec8), plus the track option "rep_flags": true:
#   continue scale-w64kd-gate-cached-ctx2048 (small-model worktree, READ ONLY) from epoch 6 into a NEW experiment
#   scale-w64kd-gate-cached-ctx2048-cont-repflags in THIS worktree, epochs 7-8 at 2,048 x 16 (32,768 reviews/update,
#   same seeds, split, user order, LR and AdamW state), 0.25-epoch one_minus_sin decay from e8 (seed 1), 464-user
#   scoring of e8 and dec8 under labels s-w64kd-gate-ctx2048crf-e8 / -dec8 (ARCH w64-repflags), then a paired summary
#   against s-w64kd-gate-ctx2048c-dec8/w8+s8 (+1.66% / +1.66% vs fork).
# The trainer adds the flag weights at exactly zero when it loads the 92-input epoch-6 checkpoint (the starting
# predictions are the 92-input model's, bit for bit: scripts/check_rep_flags.py C) and gives them fresh AdamW state;
# the teacher keeps its 92 inputs and the KD cache folder (batches-e66b1ec9cc5b) is reused, all hits.
#
#   launch   nohup bash /c/Users/jonat/workspace/wt/rep-flags/artifacts/small-model/runbook-2026-10-18-repflags.sh cont \
#              > /c/Users/jonat/workspace/wt/rep-flags/artifacts/small-model/runbook-2026-10-18-repflags.log 2>&1 & disown
#   dry run  DRYRUN=1 bash /c/Users/jonat/workspace/wt/rep-flags/artifacts/small-model/runbook-2026-10-18-repflags.sh cont
#
# Steps: (0) refuse unless scripts/check_rep_flags.py passed (artifacts/small-model/check-rep-flags.json), the source
#   epoch 6 is complete and the reference rows (fork, ctx2048c-dec8) exist; (1) copy experiment.json, run/run.json, the
#   epoch 1-6 stubs and the epoch-6 checkpoint (unchanged, 92 inputs) and set rep_flags / name / note / epochs;
#   (2) CPU check: epoch 7 resumes from epoch 6 (update_offset, LR 7e-4, AdamW), options = source + rep_flags only,
#   all in the prefetch allowlist, KD cache digest unchanged and its folder present; (3) one ticket per epoch
#   (smoke first, from a 92-input checkpoint, so the smoke also exercises the widening) and a 100-user score after each
#   epoch; NO stop rule (the pair needs e8; e7 vs ctx2048c-e7 is logged only); (4) the 0.25-epoch decay from e8, its
#   users/reviews checked equal to the reference decay; (5) 464-user scoring of e8 and dec8, resume ticket if users
#   remain; (6) CPU summary -> artifacts/small-model/repflags-summary.json.
# Do not edit this worktree while it runs (SourceWatch stops the epoch). Environment: FIRST_MIN (66), DRYRUN=1.
# DRYRUN=1 stubs the queue (echo), builds the experiment, the smoke and the decay only in scratch folders under $SM
# (deleted afterwards), never writes into the source run or the real new experiment, and turns missing artifacts into
# warnings.
set -u
STAGE=${1:?stage cont}
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
BASH_EXE="C:/Program Files/Git/usr/bin/bash.exe"
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/rep-flags
SRC_WT=C:/Users/jonat/workspace/wt/small-model           # read only: source experiment and reference score rows
AL=$WT/scripts/accuracy_loop.py
DEC_PY=$WT/scripts/expertium_decay.py
SC=$WT/scripts/small_model_heldout.py
SM=$WT/artifacts/small-model
CKPT=expertium/E89-F65e-F77a/master.pth
TRACK=E89-F65e-F77a
OUT=$SM/heldout-repflags.jsonl                           # new rows (this worktree)
REF_OUT=$SRC_WT/artifacts/small-model/heldout.jsonl      # fork and ctx2048c rows (read only)
GATE_USERS=$SM/gate-users.json
FROM=6; TO=8; CHUNK=2048; BATCH_CHUNKS=16
SRC=scale-w64kd-gate-cached-ctx2048
SRC_EXP=$SRC_WT/artifacts/accuracy/experiments/$SRC
SRC_LBL=s-w64kd-gate-ctx2048                             # source epoch labels (e6)
REF_LBL=s-w64kd-gate-ctx2048c                            # the paired run without flags (e7, e8, dec8)
REF_DECAY=$SRC_WT/artifacts/small-model/scale-w64kd-gate-cached-ctx2048-cont-decay
NAME=scale-w64kd-gate-cached-ctx2048-cont-repflags
EXP=$WT/artifacts/accuracy/experiments/$NAME
LBL=s-w64kd-gate-ctx2048crf
ARCH=w64-repflags
export DRYRUN=${DRYRUN:-0}
log() { echo "[sm30 $(date +%H:%M:%S)] $*"; }
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

# rows_ok FILE LABEL [USERS_FILE]: 0 when FILE has rows (both heads) under LABEL for all users (100 gate users by default)
rows_ok() {
  F="$1" L="$2" USERS="${3:-$GATE_USERS}" WTR="$WT" "$PY" -X utf8 - <<'PYEOF'
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.environ["WTR"])
from scripts import small_model_gate as g
if not Path(os.environ["F"]).exists():
    sys.exit(1)
users = json.loads(Path(os.environ["USERS"]).read_text())["users"]
per = g.load(os.environ["F"], [os.environ["L"]], set(users))
sys.exit(0 if all(u in per[os.environ["L"]] for u in users) else 1)
PYEOF
}

# compare A_FILE A_LABEL B_FILE B_LABEL OUT_JSON [all]: paired review-weighted LogLoss per head (B - A), % vs fork
compare() {
  FA="$1" A="$2" FB="$3" B="$4" DEC="$5" ALL="${6:-}" USERS="$GATE_USERS" REF="$REF_OUT" WTR="$WT" "$PY" -X utf8 - <<'PYEOF'
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.environ["WTR"])
from scripts import small_model_gate as g
fork = "fork/w8+s8"
a, b = os.environ["A"], os.environ["B"]
class Every:
    def __contains__(self, _):
        return True
pa = g.load(os.environ["FA"], [a], Every())
pb = g.load(os.environ["FB"], [b], Every()) if Path(os.environ["FB"]).exists() else {b: {}}
pf = g.load(os.environ["REF"], [fork], Every())
users = (sorted(set(pa[a]) & set(pb[b]) & set(pf[fork])) if os.environ["ALL"] else
         json.loads(Path(os.environ["USERS"]).read_text())["users"])
per = {a: pa[a], b: pb[b], fork: pf[fork]}
missing = {l: len([u for u in users if u not in per[l]]) for l in per}
if any(missing.values()) or not users:
    print(f"[sm30-compare] incomplete rows {missing}"); sys.exit(1)
res = {"users": len(users), "a": a, "b": b}
for head, name in g.HEADS.items():
    va, sizes = g.weighted(per, a, users, head)
    vb, _ = g.weighted(per, b, users, head)
    vf, _ = g.weighted(per, fork, users, head)
    ea, eb, ef = g.rw(va, sizes), g.rw(vb, sizes), g.rw(vf, sizes)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(sizes), size=(2000, len(sizes)))
    w = sizes[idx]
    deltas = ((vb[idx] - va[idx]) * w).sum(1) / w.sum(1)
    res[name] = {"a": ea, "b": eb, "fork": ef, "delta_b_minus_a": eb - ea,
                 "delta_ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
                 "a_pct_vs_fork": (ea / ef - 1) * 100, "b_pct_vs_fork": (eb / ef - 1) * 100,
                 "users_b_better": int((vb < va).sum())}
json.dump(res, open(os.environ["DEC"], "w"), indent=1)
print("[sm30-compare] " + json.dumps(res))
PYEOF
}

cont() {
  EP() { echo "$EXP/run/epoch-$(printf "%02d" "$1")/$CKPT"; }
  FIRST_MIN=${FIRST_MIN:-66}   # KD cache warm (same digest), 2,048 epochs measured 58-60 min
  [ -f "$SRC_EXP/experiment.json" ] || { log "no source experiment $SRC_EXP"; return 2; }

  # ---- step 0: preconditions (read only on the source)
  "$PY" -c "
import json, sys
r = json.load(open(r'$SM/check-rep-flags.json'))
sys.exit(0 if r.get('failed') == [] else 1)" 2>/dev/null || { need "scripts/check_rep_flags.py has not passed ($SM/check-rep-flags.json)" || return 2; }
  "$PY" -c "
import json, sys
r = json.load(open(r'$SRC_EXP/run/epoch-$(printf "%02d" "$FROM")/expertium/$TRACK/result.json'))
sys.exit(0 if r.get('status') == 'complete' and not r.get('smoke_only') else 1)" 2>/dev/null \
    && [ -f "$SRC_EXP/run/epoch-$(printf "%02d" "$FROM")/$CKPT" ] \
    || { need "epoch-$(printf "%02d" "$FROM") of $SRC is not complete" || return 2; }
  for l in "$SRC_LBL-e$FROM/w8+s8" "$REF_LBL-e7/w8+s8" "$REF_LBL-e8/w8+s8"; do
    rows_ok "$REF_OUT" "$l" || { need "reference rows $l missing for the 100 gate users in $REF_OUT" || return 2; }
  done
  if [ -f "$EXP/experiment.json" ] && [ "$DRYRUN" != 1 ]; then log "$NAME exists: resuming (no copy)"; BUILD=0; DEST=$EXP
  elif [ "$DRYRUN" = 1 ]; then BUILD=1; DEST=$SM/cont-check-$NAME
  else BUILD=1; DEST=$EXP.partial; fi

  # ---- step 1: copy source -> DEST (read only on the source), set rep_flags and the chunk fields
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
        f"({chunk * bc} reviews/update) with rep_flags (3 review-number flag inputs, zero-initialised), "
        f"docs/SMALL_MODEL.md SM30; pairs with scale-w64kd-gate-cached-ctx2048-cont")
exp = json.loads((src / "experiment.json").read_text())
run = json.loads((src / "run" / "run.json").read_text())
exp.update(name=name, note=note, epochs=to, created_utc=datetime.now(timezone.utc).isoformat(),
           continued_from={"experiment": os.environ["SRC"], "epoch": frm})
for opts in (exp["options"], exp["option_overrides"], run["options"]):
    opts["chunk_reviews"], opts["batch_chunks"], opts["rep_flags"] = chunk, bc, True
run.update(epochs=to, continued_from={"experiment": os.environ["SRC"], "epoch": frm})
(dest / "experiment.json").write_text(json.dumps(exp, indent=1) + "\n", encoding="utf-8")
(dest / "run" / "run.json").write_text(json.dumps(run, indent=1) + "\n", encoding="utf-8")
print(f"[sm30-copy] {dest}: epochs 1..{frm} stubs, epoch {frm} checkpoint (92 inputs), chunk {chunk} x {bc}, rep_flags, epochs {to}")
PYEOF
  fi

  # ---- step 2: CPU check of the resume (protocol of epoch FROM+1 written in a scratch folder)
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
from scripts.expertium_options import cache_digest, default_learning_rate
fail = []
run = json.loads((dest / "run" / "run.json").read_text())
sr_ = json.loads((src / "run" / "run.json").read_text())
exp = json.loads((dest / "experiment.json").read_text())
for where, o in (("run.json", run["options"]), ("experiment options", exp["options"]), ("option_overrides", exp["option_overrides"])):
    if (o.get("chunk_reviews"), o.get("batch_chunks"), o.get("rep_flags")) != (chunk, bc, True):
        fail.append(f"{where}: chunk_reviews/batch_chunks/rep_flags = {o.get('chunk_reviews')}/{o.get('batch_chunks')}/{o.get('rep_flags')}")
ignore = ("chunk_reviews", "batch_chunks", "rep_flags")
if {k: v for k, v in sr_["options"].items() if k not in ignore} != {k: v for k, v in run["options"].items() if k not in ignore}:
    fail.append("run.json options differ from the source beyond chunk_reviews/batch_chunks/rep_flags")
if (sr_["options"].get("chunk_reviews"), sr_["options"].get("batch_chunks")) != (chunk, bc):
    fail.append("the source did not train at the same chunk x batch (the pair would change two things)")
if set(run["options"]) != set(sr_["options"]) | {"rep_flags"}:
    fail.append("option key set is not the source's plus rep_flags")
for key in ("split_seed", "train_users", "review_counts"):
    if run.get(key) != sr_.get(key):
        fail.append(f"run.json {key} differs from the source")
if run["epochs"] != to or exp["epochs"] != to:
    fail.append(f"epochs run.json={run['epochs']} experiment.json={exp['epochs']}, want {to}")
for key in ("final_lr_schedule", "final_decay_epochs"):
    if run.get(key):
        fail.append(f"run.json has {key}")
for where, o in (("run.json options", run["options"]), ("option_overrides", exp["option_overrides"])):
    for key in ("lr_schedule", "optimizer"):
        if key in o:
            fail.append(f"{where} has {key}")
if exp.get("epoch_runner") or exp.get("init"):
    fail.append("experiment has an epoch_runner/init hook")
if run["options"]["batch_chunks"] * run["options"]["chunk_reviews"] != 32768:
    fail.append("reviews per update != 32768")
text = (Path(os.environ["ROOT_WT"]) / "scripts" / "gpu_r01_prefetch.py").read_text(encoding="utf-8")
allow = set(re.findall(r'"([a-z_0-9]+)"', text[text.index("set(options) - {"):text.index("or bool(options.get(\"preprocessing_cache\"))")]))
extra = sorted(set(run["options"]) - allow)
if extra:
    fail.append(f"options outside the gpu_r01_prefetch allowlist: {extra}")
digest_src, digest_new = cache_digest(sr_["options"])[0], cache_digest(run["options"])[0]
cache = run["options"].get("distillation", {}).get("cache")
if digest_src != digest_new:
    fail.append(f"KD cache digest changes with rep_flags: {digest_src} -> {digest_new}")
if not cache or not (Path(cache) / f"batches-{digest_new}").is_dir():
    fail.append(f"KD cache folder {cache}/batches-{digest_new} not found (epochs would recompute the teacher)")
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
fr = nxt["full_run"]
if real["full_run"].get("evaluate", True):
    # accuracy_loop.run_epoch writes every epoch protocol with evaluate=false (the trainer refuses rep_flags otherwise)
    fail.append("the source's epoch protocols evaluate in training (rep_flags needs evaluate=false)")
if not fr["init_master"].endswith(f"epoch-{frm:02d}/expertium/{track}/master.pth"):
    fail.append(f"init_master {fr['init_master']}")
if not fr["init_optimizer"].endswith(f"epoch-{frm:02d}/expertium/{track}/optimizer.pth"):
    fail.append(f"init_optimizer {fr['init_optimizer']}")
if fr["update_offset"] != prev["update_offset_end"]:
    fail.append(f"update_offset {fr['update_offset']} != {prev['update_offset_end']}")
if not (default_learning_rate(0, prev["update_offset_end"]) == default_learning_rate(10 ** 6, prev["update_offset_end"]) == 7e-4):
    fail.append("LR not constant 7e-4 at the epoch boundary")
shutil.rmtree(scratch, ignore_errors=True)
if fail:
    print("[sm30-check] REFUSE:\n  " + "\n  ".join(fail)); sys.exit(1)
print(f"[sm30-check] OK: epoch {frm + 1} resumes from epoch-{frm:02d} (update_offset {prev['update_offset_end']}, LR 7e-4, AdamW "
      f"state copied; the trainer adds the flag weights at zero), {chunk} x {bc}, options = source + rep_flags "
      f"({len(run['options'])} keys, all in the prefetch allowlist), KD cache batches-{digest_new} reused")
PYEOF
  if [ "$BUILD" = 1 ] && [ "$DRYRUN" != 1 ]; then rm -rf "$EXP"; mv "$DEST" "$EXP" || return 2; DEST=$EXP; log "created $EXP"; fi

  OPTS=$("$PY" -c "import json; print(json.dumps(json.load(open(r'$DEST/experiment.json'))['option_overrides']))") || return 2
  if [ "$DRYRUN" = 1 ]; then
    DCHK=$SM/cont-check-decay-$NAME; rm -rf "$DCHK"
    "$PY" -X utf8 "$DEC_PY" prepare --source-run "$DEST/run" --epoch "$FROM" --out "$DCHK" --fraction 0.25 --shape one_minus_sin \
      --evaluate none --seed 1 > /dev/null || { log "DRYRUN: decay prepare check failed"; rm -rf "$DCHK" "$DEST"; return 1; }
    "$PY" -c "
import json; d = json.load(open(r'$DCHK/decay.json'))['options']
print('[sm30-check] decay prepare options: chunk_reviews', d['chunk_reviews'], 'batch_chunks', d['batch_chunks'], 'rep_flags', d.get('rep_flags'))
assert d['chunk_reviews'] == $CHUNK and d['batch_chunks'] == $BATCH_CHUNKS and d.get('rep_flags') is True" || { rm -rf "$DCHK" "$DEST"; return 1; }
    rm -rf "$DCHK"
  fi

  # ---- step 3: one ticket per epoch (smoke first), 100-user score after each; no stop rule (the pair needs e8)
  SMOKE=$SM/smoke-prep-$NAME
  rm -rf "$SMOKE"
  "$PY" -X utf8 "$DEC_PY" prepare --source-run "$SRC_WT/artifacts/accuracy/experiments/scale-w64kd-gate/run" --epoch 3 --out "$SMOKE" \
    --fraction 0.01 --evaluate none --seed 1 --extra-options "$OPTS" > /dev/null || return 1
  for epoch in $(seq $((FROM + 1)) "$TO"); do
    minutes=66; [ "$epoch" -eq $((FROM + 1)) ] && minutes=$FIRST_MIN
    if [ "$DRYRUN" = 1 ] || [ ! -f "$(EP "$epoch")" ]; then
      q "$minutes" "SM30 $NAME epoch $epoch/$TO (rep_flags, chunk $CHUNK x $BATCH_CHUNKS, smoke first)" -- "$BASH_EXE" -c \
        "'$PY' -X utf8 '$DEC_PY' smoke --out '$SMOKE' && '$PY' -X utf8 '$AL' train '$NAME' --epochs-per-job 1"
      rc=$?
      [ $rc -eq 0 ] || [ $rc -eq 3 ] || { log "training failed (exit $rc), stop"; return 1; }
      [ -f "$(EP "$epoch")" ] || { need "no epoch-$epoch checkpoint after training" || return 1; }
    fi
    if [ "$DRYRUN" = 1 ] || ! rows_ok "$OUT" "$LBL-e$epoch/w8+s8"; then
      q 4 "SM30 gate: score $LBL-e$epoch on the 100 gate users" -- "$PY" -X utf8 "$SC" \
        --users-file "$GATE_USERS" --model "$LBL-e$epoch=expertium:$ARCH:$(EP "$epoch")" --only "$LBL-e$epoch/w8+s8" --time-limit 600 --out "$OUT" || return 1
    fi
    compare "$REF_OUT" "$REF_LBL-e$epoch/w8+s8" "$OUT" "$LBL-e$epoch/w8+s8" "$SM/repflags-gate-e$epoch.json" \
      || { [ "$DRYRUN" = 1 ] && log "DRYRUN: no rows for the e$epoch comparison" || log "e$epoch comparison incomplete (continuing)"; }
  done
  EN=$(EP "$TO")

  # ---- step 4: 0.25-epoch decay from e8 at 2,048 x 16 (seed 1: the reference decay's users and reviews)
  DECF=$SM/$NAME-decay
  D=$DECF/$CKPT
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: would prepare $DECF (--source-run $EXP/run --epoch $TO --fraction 0.25 --shape one_minus_sin --evaluate none --seed 1)"; ok=1
  elif [ -d "$DECF" ]; then ok=1
  else "$PY" -X utf8 "$DEC_PY" prepare --source-run "$EXP/run" --epoch "$TO" --out "$DECF" --fraction 0.25 --shape one_minus_sin \
      --evaluate none --seed 1 && ok=1 || ok=0; fi
  [ "$ok" = 1 ] || { log "decay prepare failed"; return 1; }
  if [ "$DRYRUN" != 1 ]; then
    "$PY" -c "
import json, sys
a = json.load(open(r'$REF_DECAY/decay.json')); b = json.load(open(r'$DECF/decay.json'))
keys = ('source_epoch', 'training_users', 'training_reviews', 'shuffle_seed', 'lr', 'fraction_of_epoch_reviews')
diff = {k: (a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)}
oa = {k: v for k, v in a['options'].items() if k != 'rep_flags'}; ob = {k: v for k, v in b['options'].items() if k != 'rep_flags'}
print('[sm30-decay] vs reference decay:', 'same users/reviews/schedule/options' if not diff and oa == ob and b['options'].get('rep_flags') else f'DIFFERS {diff}')
sys.exit(0 if not diff and oa == ob and b['options'].get('rep_flags') is True else 1)" || return 1
  fi
  [ -f "$D" ] || q 27 "SM30 $NAME: 0.25-epoch decay from e$TO at chunk $CHUNK (smoke first)" -- "$BASH_EXE" -c \
      "'$PY' -X utf8 '$DEC_PY' smoke --out '$DECF' && '$PY' -X utf8 '$DEC_PY' run --out '$DECF'" || return 1

  # ---- step 5: 464-user scoring and a resume ticket only if users remain
  MODELS="--model $LBL-e$TO=expertium:$ARCH:$EN --model $LBL-dec$TO=expertium:$ARCH:$D"
  LABELS="--only $LBL-e$TO/w8+s8 --only $LBL-dec$TO/w8+s8"
  # shellcheck disable=SC2086
  q 30 "SM30 $NAME: full-history scoring of e$TO and dec$TO (464 users)" -- "$PY" -X utf8 "$SC" $MODELS $LABELS --time-limit 3000 --out "$OUT" || return 1
  remaining=$(OUT="$OUT" REF="$REF_OUT" L1="$LBL-e$TO/w8+s8" L2="$LBL-dec$TO/w8+s8" "$PY" -X utf8 - <<'PYEOF'
import json, os
from pathlib import Path
def users(path, label):
    found = set()
    if Path(path).exists():
        for line in open(path, encoding="utf-8"):
            if '"label"' in line:
                r = json.loads(line)
                if r["label"] == label and "benchmark" in r:
                    found.add(r["user_id"])
    return found
total = users(os.environ["REF"], "fork/w8+s8")
print(max(len(total - users(os.environ["OUT"], os.environ["L1"])), len(total - users(os.environ["OUT"], os.environ["L2"]))) if total else 0)
PYEOF
)
  log "users still unscored after the scoring ticket (vs the fork's users): ${remaining:-?}"
  if [ "$DRYRUN" = 1 ] || [ "${remaining:-1}" != 0 ]; then
    # shellcheck disable=SC2086
    q 15 "SM30 $NAME: scoring resume (only if users remain)" -- "$PY" -X utf8 "$SC" $MODELS $LABELS --time-limit 1200 --out "$OUT"
  fi

  # ---- step 6: paired 464-user summary (CPU)
  compare "$REF_OUT" "$REF_LBL-e$TO/w8+s8" "$OUT" "$LBL-e$TO/w8+s8" "$SM/repflags-summary-e$TO.json" all \
    || { [ "$DRYRUN" = 1 ] || log "e$TO summary incomplete"; }
  compare "$REF_OUT" "$REF_LBL-dec$TO/w8+s8" "$OUT" "$LBL-dec$TO/w8+s8" "$SM/repflags-summary.json" all \
    || { [ "$DRYRUN" = 1 ] && log "DRYRUN: no rows for the summary" || { log "dec$TO summary incomplete"; return 1; }; }
  [ "$DRYRUN" = 1 ] && rm -rf "$DEST" "$SMOKE"
  return 0
}

cd "$WT" || exit 1
case "$STAGE" in
cont) cont; rc=$?; [ $rc -eq 0 ] && log "done: $STAGE" || log "$STAGE ended with exit $rc"; exit $rc ;;
*) echo "unknown stage $STAGE (cont)"; exit 2 ;;
esac
