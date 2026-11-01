#!/usr/bin/env bash
# Small-model runbook 2026-10-18 (SM21 v2, strided lanes; docs/SMALL_MODEL.md "Design 2026-10-16: training with carried
# state (SM21)" and its v2 status note; HANDOFF-SM21.md in the sm21v2 worktree): the v2 arm of the SM21 test, adapted from
# wt/state-carry/artifacts/small-model/runbook-2026-10-18-sm21.sh (arms A and B, run from the state-carry worktree).
#   Continue scale-w64kd-gate-cached-ctx32768-cont (small-model worktree, READ ONLY) from epoch 10 into one NEW experiment
#   in THIS worktree (sm21v2), epochs 11-12 at 2,048 x 16 (32,768 reviews/update, same seeds, split, user order, LR and
#   AdamW state as the source), then a 0.25-epoch one_minus_sin decay from e12 (seed 1), then 464-user scoring of e12 and
#   dec12:
#     arm C  lanes     scale-w64kd-gate-sc-lanes-cont  labels s-w64kd-gate-sc-lanes-e12 / -dec12   "state_carry": {"lanes": "strided"}
#   Same SM11 replay targets as arms A and B (distillation = source spec without "cache" plus "kd_source": "replay").
#   v2 layout (scripts/state_carry_train.py): update u of a user holds window u of each of its 16 lanes, so every window
#   continues its lane predecessor exactly; lane starts (update 0) and entities first seen in a lane start from zero.
#   Paired against (rows read only from wt/state-carry/artifacts/small-model/heldout-sm21.jsonl):
#     arm B  control   s-w64kd-gate-sc-zero-*    "state_carry": {"carry": false}   (same inputs, zero state)
#     arm A  v1 carry  s-w64kd-gate-sc-carry-*   "state_carry": true               (update-granularity carry)
#   Decision rule (pre-registered, as Q6 and the A/B pair): C - B <= -0.001 on the curve head with upper95 < 0 AND instant
#   <= +0.0005 (review-weighted LogLoss over the 464 test users at w8+s8, dec12; also computed for e12). Also reported:
#   C - A (v2 vs v1, same rule shape, informational), C vs the source dec10 (s-w64kd-gate-ctx32768c-dec10) and vs fork.
#
#   launch   nohup bash /c/Users/jonat/workspace/wt/sm21v2/artifacts/small-model/runbook-2026-10-18-sm21v2.sh \
#              > /c/Users/jonat/workspace/wt/sm21v2/artifacts/small-model/runbook-2026-10-18-sm21v2.log 2>&1 & disown
#   dry run  DRYRUN=1 bash /c/Users/jonat/workspace/wt/sm21v2/artifacts/small-model/runbook-2026-10-18-sm21v2.sh
#   Resumes: rerun the same command; finished epochs / decays / scoring rows are skipped. SUMMARY_ONLY=1 recomputes the
#   summary only (e.g. once arm B's rows exist in the state-carry worktree).
#
# Steps: (1) copy experiment.json, run/run.json, the epoch 1-10 stubs and the epoch-10 checkpoint, set chunk 2048 x 16,
#   the state_carry option and the replay distillation spec; (2) CPU check: epoch 11 resumes from epoch 10 (update_offset, LR
#   7e-4, AdamW), options = source + state_carry (+ chunk fields + replay spec), allowlisted in gpu_r01_prefetch, replay
#   manifest valid for the teacher and dataset, replay users cover the run's train users; (3) one ticket per epoch (smoke first)
#   and a 100-user score after each epoch (logged vs the source e10 and vs arms A/B when their rows exist); exit 3 of
#   accuracy_loop train = more epochs to do, not a failure; (4) 0.25-epoch decay from e12, its users/reviews checked equal to
#   the source decay (dec10); (5) 464-user scoring, resume ticket if users remain.
#   Final: paired summary -> artifacts/small-model/sm21v2-summary.json (+ .md).
# Preconditions (checked first): replay export complete (sm11_replay_export.py status exit 0), the v1 merged CPU checks and
#   GPU parity reports (inherited from state-carry) ALL OK, the v2 CPU check (check-state-carry-lanes.json) and the v2 GPU
#   check (check-state-carry-lanes-gpu.json, artifacts/small-model/jobs/sm21v2-gpu-check.sh) passed, no uncommitted changes
#   in scripts/ expertium/ rwkv/, replay folder present.
# Do not edit this worktree while it runs (SourceWatch stops the epoch). Environment: FIRST_MIN (80), DRYRUN=1, SUMMARY_ONLY=1.
# GPU budget: 2 x 80 (epochs) + 2 x 4 (gate) + 25 (decay) + 25 (+15 resume) = 218 min (3.6 h); the v2 per-update cost is
# measured by the GPU check (T), v1 was +9% over stock.
# DRYRUN=1 stubs the queue (echo), builds the experiment, the smoke and the decay only in scratch folders under $SM (deleted
# afterwards), never writes into the source run or the real new experiment, and turns missing artifacts (incl. the
# incomplete replay export and the not yet run GPU check) into warnings. Torch threads capped at 4 in every CPU step.
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
BASH_EXE="C:/Program Files/Git/usr/bin/bash.exe"
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm21v2
SC_WT=C:/Users/jonat/workspace/wt/state-carry            # read only: arms A (v1 carry) and B (zero) score rows
SRC_WT=C:/Users/jonat/workspace/wt/small-model           # read only: source experiment and reference score rows
AL=$WT/scripts/accuracy_loop.py
DEC_PY=$WT/scripts/expertium_decay.py
SC=$WT/scripts/small_model_heldout.py
EXPORT=$WT/scripts/sm11_replay_export.py
SM=$WT/artifacts/small-model
CKPT=expertium/E89-F65e-F77a/master.pth
TRACK=E89-F65e-F77a
OUT=$SM/heldout-sm21v2.jsonl                             # new rows (this worktree)
AB_OUT=$SC_WT/artifacts/small-model/heldout-sm21.jsonl   # arm A / arm B rows (state-carry worktree, read only)
CARRY=s-w64kd-gate-sc-carry; ZERO=s-w64kd-gate-sc-zero; LANES=s-w64kd-gate-sc-lanes
REF_OUT=$SRC_WT/artifacts/small-model/heldout.jsonl      # fork and source rows (read only)
GATE_USERS=$SM/gate-users.json
FROM=10; TO=12; CHUNK=2048; BATCH_CHUNKS=16
SRC=scale-w64kd-gate-cached-ctx32768-cont
SRC_EXP=$SRC_WT/artifacts/accuracy/experiments/$SRC
SRC_LBL=s-w64kd-gate-ctx32768c                           # source labels: -e10, -dec10
REF_DECAY=$SRC_WT/artifacts/small-model/$SRC-decay
REPLAY=C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/fork-fp32-gate5k
REPLAY_RUN=C:/Users/jonat/workspace/wt/sm11/artifacts/accuracy/experiments/scale-w64kd-gate/run/run.json   # run.json the export used
SMOKE_RUN=$SRC_WT/artifacts/accuracy/experiments/scale-w64kd-gate/run                                      # smoke source (epoch 3)
ARCH=w64
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
log() { echo "[sm21v2 $(date +%H:%M:%S)] $*"; }
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

# compare A_FILE A_LABEL B_FILE B_LABEL OUT_JSON [all]: paired review-weighted LogLoss per head (B - A), % vs fork;
# 95% interval = user-level bootstrap (2,000 resamples, seed 0, the gate script's boot_upper)
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
pa = g.load(os.environ["FA"], [a], Every()) if Path(os.environ["FA"]).exists() else {a: {}}
pb = g.load(os.environ["FB"], [b], Every()) if Path(os.environ["FB"]).exists() else {b: {}}
pf = g.load(os.environ["REF"], [fork], Every())
users = (sorted(set(pa[a]) & set(pb[b]) & set(pf[fork])) if os.environ["ALL"] else
         json.loads(Path(os.environ["USERS"]).read_text())["users"])
per = {a: pa[a], b: pb[b], fork: pf[fork]}
missing = {l: len([u for u in users if u not in per[l]]) for l in per}
if any(missing.values()) or not users:
    print(f"[sm21-compare] incomplete rows {missing}"); sys.exit(1)
res = {"users": len(users), "a": a, "b": b, "delta": "b - a"}
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
print("[sm21-compare] " + json.dumps(res))
PYEOF
}

# ---- preconditions common to both arms
preconditions() {
  [ -f "$SRC_EXP/experiment.json" ] || { log "no source experiment $SRC_EXP"; return 2; }
  if [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    need "uncommitted source changes in scripts/ expertium/ rwkv/ (SourceWatch / the orchestrator refuse them)" || return 2
  fi
  for f in check-state-carry-trainer-merged.txt check-sm11-replay-targets-merged.txt; do
    grep -q "ALL OK" "$SM/$f" 2>/dev/null || { need "$SM/$f does not contain ALL OK" || return 2; }
  done
  for f in check-state-carry-gpu.json check-state-carry-kernels-gpu.json check-state-carry-lanes.json check-state-carry-lanes-gpu.json; do
    "$PY" -c "
import json, sys
r = json.load(open(r'$SM/$f'))
sys.exit(0 if r.get('fails') in ([], {}, 0, None) and r.get('passed') else 1)" 2>/dev/null || { need "check report $f has not passed (or is missing)" || return 2; }
  done
  "$PY" -X utf8 "$EXPORT" status --out "$REPLAY" --run "$REPLAY_RUN"
  rc=$?
  if [ "$rc" != 0 ]; then need "replay export incomplete or invalid (status exit $rc)" || return 2; else log "replay export complete"; fi
  # best-effort: the source epoch FROM and the reference rows
  "$PY" -c "
import json, sys
r = json.load(open(r'$SRC_EXP/run/epoch-$(printf "%02d" "$FROM")/expertium/$TRACK/result.json'))
sys.exit(0 if r.get('status') == 'complete' and not r.get('smoke_only') else 1)" 2>/dev/null \
    && [ -f "$SRC_EXP/run/epoch-$(printf "%02d" "$FROM")/$CKPT" ] \
    || { need "epoch-$(printf "%02d" "$FROM") of $SRC is not complete" || return 2; }
  for l in "$SRC_LBL-e$FROM/w8+s8" "$SRC_LBL-dec$FROM/w8+s8"; do
    rows_ok "$REF_OUT" "$l" || { need "reference rows $l missing for the 100 gate users in $REF_OUT" || return 2; }
  done
  return 0
}

# arm NAME LABEL STATE_CARRY_JSON TAG
arm() {
  NAME=$1; LBL=$2; SCJSON=$3; TAG=$4
  EXP=$WT/artifacts/accuracy/experiments/$NAME
  EP() { echo "$EXP/run/epoch-$(printf "%02d" "$1")/$CKPT"; }
  FIRST_MIN=${FIRST_MIN:-80}
  log "=== arm $TAG: $NAME (state_carry $SCJSON) ==="
  if [ -f "$EXP/experiment.json" ] && [ "$DRYRUN" != 1 ]; then log "$NAME exists: resuming (no copy)"; BUILD=0; DEST=$EXP
  elif [ "$DRYRUN" = 1 ]; then BUILD=1; DEST=$SM/cont-check-$NAME
  else BUILD=1; DEST=$EXP.partial; fi

  # ---- step 1: copy source -> DEST (read only on the source)
  if [ "$BUILD" = 1 ]; then
    rm -rf "$DEST"
    SRC_EXP="$SRC_EXP" DEST="$DEST" FROM="$FROM" TO="$TO" CHUNK="$CHUNK" BATCH_CHUNKS="$BATCH_CHUNKS" NAME="$NAME" SRC="$SRC" \
      TRACK="$TRACK" SCJSON="$SCJSON" REPLAY="$REPLAY" TAG="$TAG" "$PY" -X utf8 - <<'PYEOF' || return 2
import json, os, shutil
from datetime import datetime, timezone
from pathlib import Path
src, dest = Path(os.environ["SRC_EXP"]), Path(os.environ["DEST"])
frm, to, chunk, bc = (int(os.environ[k]) for k in ("FROM", "TO", "CHUNK", "BATCH_CHUNKS"))
track, name = os.environ["TRACK"], os.environ["NAME"]
carry = json.loads(os.environ["SCJSON"])
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
note = (f"{name}: SM21 arm {os.environ['TAG']}: continuation of {os.environ['SRC']} from epoch {frm} to {to} at {chunk}-review "
        f"chunks x {bc} ({chunk * bc} reviews/update) with state_carry={os.environ['SCJSON']} and the SM11 full-history replay "
        f"targets (distillation kd_source replay); docs/SMALL_MODEL.md SM21 v2, HANDOFF-SM21.md (sm21v2); pairs with arms "
        f"A (scale-w64kd-gate-sc-carry-cont, v1) and B (scale-w64kd-gate-sc-zero-cont, zero) of the state-carry worktree")
exp = json.loads((src / "experiment.json").read_text())
run = json.loads((src / "run" / "run.json").read_text())
exp.update(name=name, note=note, epochs=to, created_utc=datetime.now(timezone.utc).isoformat(),
           continued_from={"experiment": os.environ["SRC"], "epoch": frm})
for opts in (exp["options"], exp["option_overrides"], run["options"]):
    opts["chunk_reviews"], opts["batch_chunks"], opts["state_carry"] = chunk, bc, carry
    d = {k: v for k, v in opts["distillation"].items() if k != "cache"}   # replay refuses a "cache" key
    d.update(kd_source="replay", replay=os.environ["REPLAY"])
    opts["distillation"] = d
run.update(epochs=to, continued_from={"experiment": os.environ["SRC"], "epoch": frm})
(dest / "experiment.json").write_text(json.dumps(exp, indent=1) + "\n", encoding="utf-8")
(dest / "run" / "run.json").write_text(json.dumps(run, indent=1) + "\n", encoding="utf-8")
print(f"[sm21-copy] {dest}: epochs 1..{frm} stubs, epoch {frm} checkpoint, chunk {chunk} x {bc}, state_carry {carry}, "
      f"replay distillation {d}, epochs {to}")
PYEOF
  fi

  # ---- step 2: CPU check of the resume (protocol of epoch FROM+1 written in a scratch folder)
  scratch=$SM/cont-protocol-$NAME
  rm -rf "$scratch"
  DEST="$DEST" SRC_EXP="$SRC_EXP" SCRATCH="$scratch" FROM="$FROM" TO="$TO" CHUNK="$CHUNK" BATCH_CHUNKS="$BATCH_CHUNKS" \
    ROOT_WT="$WT" TRACK="$TRACK" SCJSON="$SCJSON" REPLAY="$REPLAY" REPLAY_RUN="$REPLAY_RUN" DRYRUN="$DRYRUN" \
    "$PY" -X utf8 - <<'PYEOF' || { rm -rf "$scratch"; [ "$BUILD" = 1 ] && [ "$DEST" != "$EXP" ] && rm -rf "$DEST"; return 2; }
import json, os, re, shutil, sys
from pathlib import Path
dest, src, scratch = Path(os.environ["DEST"]), Path(os.environ["SRC_EXP"]), Path(os.environ["SCRATCH"])
frm, to, chunk, bc = (int(os.environ[k]) for k in ("FROM", "TO", "CHUNK", "BATCH_CHUNKS"))
track = os.environ["TRACK"]
carry = json.loads(os.environ["SCJSON"])
sys.path.insert(0, os.environ["ROOT_WT"])
import torch
torch.set_num_threads(4)
from scripts import run_full_training as rft
from scripts.expertium_options import default_learning_rate
fail = []
run = json.loads((dest / "run" / "run.json").read_text())
sr_ = json.loads((src / "run" / "run.json").read_text())
exp = json.loads((dest / "experiment.json").read_text())
want = {k: v for k, v in sr_["options"]["distillation"].items() if k != "cache"}
want.update(kd_source="replay", replay=os.environ["REPLAY"])
for where, o in (("run.json", run["options"]), ("experiment options", exp["options"]), ("option_overrides", exp["option_overrides"])):
    if (o.get("chunk_reviews"), o.get("batch_chunks"), o.get("state_carry")) != (chunk, bc, carry):
        fail.append(f"{where}: chunk_reviews/batch_chunks/state_carry = {o.get('chunk_reviews')}/{o.get('batch_chunks')}/{o.get('state_carry')}")
    if o.get("distillation") != want:
        fail.append(f"{where}: distillation {o.get('distillation')} != {want}")
ignore = ("chunk_reviews", "batch_chunks", "state_carry", "distillation")
if {k: v for k, v in sr_["options"].items() if k not in ignore} != {k: v for k, v in run["options"].items() if k not in ignore}:
    fail.append("run.json options differ from the source beyond chunk_reviews/batch_chunks/state_carry/distillation")
if set(run["options"]) != set(sr_["options"]) | {"state_carry"}:
    fail.append("option key set is not the source's plus state_carry")
if run["options"]["distillation"].get("alpha") != sr_["options"]["distillation"].get("alpha") or \
        run["options"]["distillation"].get("decay_alpha") != sr_["options"]["distillation"].get("decay_alpha") or \
        run["options"]["distillation"].get("teacher") != sr_["options"]["distillation"].get("teacher"):
    fail.append("teacher / alpha / decay_alpha differ from the source")
if sr_["options"].get("chunk_reviews") * sr_["options"].get("batch_chunks") != 32768:
    fail.append("the source did not train at 32,768 reviews/update (the pair would change the update size)")
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
# the SM11 replay spec: constructed for real (manifest format / teacher sha256 / dataset checks), CPU only
try:
    from scripts.expertium_options import ReplayTargets
    rt = ReplayTargets(run["options"]["distillation"])
    print(f"[sm21-check] replay spec valid: teacher sha256 {str(rt.teacher_sha256)[:12]}")
except Exception as exc:  # noqa: BLE001
    fail.append(f"replay spec refused: {type(exc).__name__}: {exc}")
# the export must cover the run's training users (the export used REPLAY_RUN)
replay_run = Path(os.environ["REPLAY_RUN"])
if replay_run.exists():
    ru = json.loads(replay_run.read_text())
    if sorted(ru["train_users"]) != sorted(run["train_users"]):
        fail.append("the replay export's run has other training users than this run")
    if ru["dataset_revision"] != run["dataset_revision"]:
        fail.append("the replay export's run has another dataset revision")
elif os.environ["DRYRUN"] != "1":
    fail.append(f"replay export run {replay_run} missing")
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
if real["training"] != nxt["training"]:
    fail.append(f"protocol training differs: {real['training']} vs {nxt['training']}")
for k in ("execution_mode", "dataset_path", "dataset_revision"):
    if real[k] != nxt[k]:
        fail.append(f"protocol {k} differs: {real[k]} vs {nxt[k]}")
fr = nxt["full_run"]
if real["full_run"].get("evaluate", True):
    fail.append("the source's epoch protocols evaluate in training (the carry trainer needs evaluate=false)")
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
    print("[sm21-check] REFUSE:\n  " + "\n  ".join(fail)); sys.exit(1)
print(f"[sm21-check] OK: epoch {frm + 1} resumes from epoch-{frm:02d} (update_offset {prev['update_offset_end']}, LR 7e-4, AdamW "
      f"state copied), {chunk} x {bc}, options = source + state_carry {carry} + replay spec "
      f"({len(run['options'])} keys, all in the prefetch allowlist)")
PYEOF
  if [ "$BUILD" = 1 ] && [ "$DRYRUN" != 1 ]; then rm -rf "$EXP"; mv "$DEST" "$EXP" || return 2; DEST=$EXP; log "created $EXP"; fi

  OPTS=$("$PY" -c "import json; print(json.dumps(json.load(open(r'$DEST/experiment.json'))['option_overrides']))") || return 2
  if [ "$DRYRUN" = 1 ]; then
    DCHK=$SM/cont-check-decay-$NAME; rm -rf "$DCHK"
    "$PY" -X utf8 "$DEC_PY" prepare --source-run "$DEST/run" --epoch "$FROM" --out "$DCHK" --fraction 0.25 --shape one_minus_sin \
      --evaluate none --seed 1 > /dev/null || { log "DRYRUN: decay prepare check failed"; rm -rf "$DCHK" "$DEST"; return 1; }
    SCJ="$SCJSON" "$PY" -c "
import json, os; d = json.load(open(r'$DCHK/decay.json'))['options']
print('[sm21-check] decay prepare options: chunk_reviews', d['chunk_reviews'], 'batch_chunks', d['batch_chunks'], 'state_carry', d.get('state_carry'), 'kd_source', d['distillation'].get('kd_source'))
assert d['chunk_reviews'] == $CHUNK and d['batch_chunks'] == $BATCH_CHUNKS and d.get('state_carry') == json.loads(os.environ['SCJ'])
assert d['distillation'].get('kd_source') == 'replay' and 'cache' not in d['distillation']" || { rm -rf "$DCHK" "$DEST"; return 1; }
    rm -rf "$DCHK"
  fi

  # ---- step 3: one ticket per epoch (smoke first), 100-user score after each; no stop rule (the pair needs e12)
  SMOKE=$SM/smoke-prep-$NAME
  rm -rf "$SMOKE"
  "$PY" -X utf8 "$DEC_PY" prepare --source-run "$SMOKE_RUN" --epoch 3 --out "$SMOKE" \
    --fraction 0.01 --evaluate none --seed 1 --extra-options "$OPTS" > /dev/null || return 1
  for epoch in $(seq $((FROM + 1)) "$TO"); do
    minutes=80; [ "$epoch" -eq $((FROM + 1)) ] && minutes=$FIRST_MIN
    if [ "$DRYRUN" = 1 ] || [ ! -f "$(EP "$epoch")" ]; then
      q "$minutes" "SM21 $NAME epoch $epoch/$TO ($TAG, chunk $CHUNK x $BATCH_CHUNKS, smoke first)" -- "$BASH_EXE" -c \
        "'$PY' -X utf8 '$DEC_PY' smoke --out '$SMOKE' && '$PY' -X utf8 '$AL' train '$NAME' --epochs-per-job 1"
      rc=$?
      [ $rc -eq 0 ] || [ $rc -eq 3 ] || { log "training failed (exit $rc), stop"; return 1; }
      [ -f "$(EP "$epoch")" ] || { need "no epoch-$epoch checkpoint after training" || return 1; }
    fi
    if [ "$DRYRUN" = 1 ] || ! rows_ok "$OUT" "$LBL-e$epoch/w8+s8"; then
      q 4 "SM21 gate: score $LBL-e$epoch on the 100 gate users" -- "$PY" -X utf8 "$SC" \
        --users-file "$GATE_USERS" --model "$LBL-e$epoch=expertium:$ARCH:$(EP "$epoch")" --only "$LBL-e$epoch/w8+s8" --time-limit 600 --out "$OUT" || return 1
    fi
    # log only: this arm vs the source e10 (100 gate users), and vs arms B (zero) and A (v1 carry) when their rows exist
    compare "$REF_OUT" "$SRC_LBL-e$FROM/w8+s8" "$OUT" "$LBL-e$epoch/w8+s8" "$SM/sm21v2-gate-$TAG-e$epoch-vs-src-e$FROM.json" \
      || { [ "$DRYRUN" = 1 ] && log "DRYRUN: no rows for the e$epoch gate comparison" || log "e$epoch gate comparison incomplete (continuing)"; }
    for other in zero carry; do
      ol=$ZERO; [ "$other" = carry ] && ol=$CARRY
      compare "$AB_OUT" "$ol-e$epoch/w8+s8" "$OUT" "$LBL-e$epoch/w8+s8" "$SM/sm21v2-gate-e$epoch-lanes-vs-$other.json" \
        || log "e$epoch lanes-vs-$other gate comparison incomplete (arm rows missing; continuing)"
    done
  done
  EN=$(EP "$TO")

  # ---- step 4: 0.25-epoch decay from e12 at 2,048 x 16 (seed 1: the source decay's users and reviews)
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
keys = ('training_users', 'training_reviews', 'shuffle_seed', 'lr', 'fraction_of_epoch_reviews')
diff = {k: (a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)}
ign = ('chunk_reviews', 'batch_chunks', 'state_carry', 'distillation')
oa = {k: v for k, v in a['options'].items() if k not in ign}; ob = {k: v for k, v in b['options'].items() if k not in ign}
ok = not diff and oa == ob and b['options'].get('state_carry') == json.loads(r'''$SCJSON''') and b['options']['distillation'].get('kd_source') == 'replay'
print('[sm21-decay] vs source decay:', 'same users/reviews/schedule/options' if ok else f'DIFFERS {diff}')
sys.exit(0 if ok else 1)" || return 1
  fi
  [ -f "$D" ] || q 25 "SM21 $NAME: 0.25-epoch decay from e$TO ($TAG, chunk $CHUNK, smoke first)" -- "$BASH_EXE" -c \
      "'$PY' -X utf8 '$DEC_PY' smoke --out '$DECF' && '$PY' -X utf8 '$DEC_PY' run --out '$DECF'" || return 1

  # ---- step 5: 464-user scoring and a resume ticket only if users remain
  MODELS="--model $LBL-e$TO=expertium:$ARCH:$EN --model $LBL-dec$TO=expertium:$ARCH:$D"
  LABELS="--only $LBL-e$TO/w8+s8 --only $LBL-dec$TO/w8+s8"
  # shellcheck disable=SC2086
  q 25 "SM21 $NAME: full-history scoring of e$TO and dec$TO (464 users, $TAG)" -- "$PY" -X utf8 "$SC" $MODELS $LABELS --time-limit 3000 --out "$OUT" || return 1
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
    q 15 "SM21 $NAME: scoring resume (only if users remain, $TAG)" -- "$PY" -X utf8 "$SC" $MODELS $LABELS --time-limit 1200 --out "$OUT"
  fi
  [ "$DRYRUN" = 1 ] && rm -rf "$DEST" "$SMOKE"
  return 0
}

summary() {
  # C (lanes) vs B (zero) and vs A (v1 carry): compare with a = the other arm, b = C, so delta_b_minus_a = C - other
  for h in dec$TO e$TO; do
    compare "$AB_OUT" "$ZERO-$h/w8+s8" "$OUT" "$LANES-$h/w8+s8" "$SM/sm21v2-lanes-vs-zero-$h.json" all \
      || log "lanes-vs-zero $h incomplete (arm B rows missing?)"
    compare "$AB_OUT" "$CARRY-$h/w8+s8" "$OUT" "$LANES-$h/w8+s8" "$SM/sm21v2-lanes-vs-carry-$h.json" all \
      || log "lanes-vs-carry $h incomplete (arm A rows missing?)"
    compare "$REF_OUT" "$SRC_LBL-${h/$TO/$FROM}/w8+s8" "$OUT" "$LANES-$h/w8+s8" "$SM/sm21v2-lanes-vs-src-$h.json" all \
      || log "lanes-vs-source $h incomplete"
  done
  SM="$SM" "$PY" -X utf8 - <<'PYEOF'
import json, os
from pathlib import Path
sm = Path(os.environ["SM"])
def load(n):
    p = sm / n
    return json.loads(p.read_text()) if p.exists() else None
rule = "C(lanes) - B(zero): curve <= -0.001 with upper95 < 0 AND instant <= +0.0005, 464 users, review-weighted, w8+s8"
out, md = {"rule": rule}, []
for h in ("dec12", "e12"):
    out[h] = {}
    for other, name in (("zero", "B (zero)"), ("carry", "A (v1 carry)")):
        cz = load(f"sm21v2-lanes-vs-{other}-{h}.json")
        if cz is None:
            out[h][f"lanes_vs_{other}"] = "not available (rows missing)"
            md.append(f"{h}: lanes vs {name}: rows missing")
            continue
        cur, ins = cz["curve"], cz["instant"]
        ok = cur["delta_b_minus_a"] <= -0.001 and cur["delta_ci95"][1] < 0 and ins["delta_b_minus_a"] <= 0.0005
        out[h][f"lanes_vs_{other}"] = {
            "users": cz["users"],
            "lanes": {"curve": cur["b"], "instant": ins["b"], "curve_pct_vs_fork": cur["b_pct_vs_fork"], "instant_pct_vs_fork": ins["b_pct_vs_fork"]},
            other: {"curve": cur["a"], "instant": ins["a"], "curve_pct_vs_fork": cur["a_pct_vs_fork"], "instant_pct_vs_fork": ins["a_pct_vs_fork"]},
            "lanes_minus_other": {"curve": cur["delta_b_minus_a"], "curve_ci95": cur["delta_ci95"], "instant": ins["delta_b_minus_a"], "instant_ci95": ins["delta_ci95"]},
            "fork": {"curve": cur["fork"], "instant": ins["fork"]},
            "users_lanes_better": {"curve": cur["users_b_better"], "instant": ins["users_b_better"]},
            "decision" if other == "zero" else "rule_shape_vs_v1": ("PASS: lanes win" if ok else "FAIL: lanes do not meet the rule")}
        md.append(f"{h}: lanes curve {cur['b']:.5f} ({cur['b_pct_vs_fork']:+.2f}% vs fork), instant {ins['b']:.5f} ({ins['b_pct_vs_fork']:+.2f}%); "
                  f"{name} curve {cur['a']:.5f} ({cur['a_pct_vs_fork']:+.2f}%), instant {ins['a']:.5f} ({ins['a_pct_vs_fork']:+.2f}%); "
                  f"C - {name[0]} curve {cur['delta_b_minus_a']:+.5f} [{cur['delta_ci95'][0]:+.5f}, {cur['delta_ci95'][1]:+.5f}], "
                  f"instant {ins['delta_b_minus_a']:+.5f} -> {'PASS' if ok else 'FAIL'}" + ("" if other == "zero" else " (informational)"))
    s = load(f"sm21v2-lanes-vs-src-{h}.json")
    if s:
        out[h][f"lanes_vs_source_{h.replace('12', '10')}"] = {k: {"delta": s[k]["delta_b_minus_a"], "ci95": s[k]["delta_ci95"]} for k in ("curve", "instant")}
(sm / "sm21v2-summary.json").write_text(json.dumps(out, indent=1) + "\n")
(sm / "sm21v2-summary.md").write_text("# SM21 v2 lanes vs control and v1 (464 users, w8+s8)\n\n" + "\n\n".join(md) + "\n\nRule: " + rule + "\n")
print("[sm21v2-summary] " + " | ".join(md))
PYEOF
}

main() {
  cd "$WT" || exit 1
  log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN)"
  if [ "${SUMMARY_ONLY:-0}" = 1 ]; then summary; return 0; fi
  preconditions || return 2
  arm scale-w64kd-gate-sc-lanes-cont "$LANES" '{"lanes": "strided"}' lanes || return 1
  summary  # incomplete pairs are logged; rerun with SUMMARY_ONLY=1 once arm B's rows exist
  return 0
}

main; rc=$?
if [ $rc -eq 0 ]; then
  log "done"
  [ "$DRYRUN" = 1 ] && echo "DRYRUN OK"
else
  log "ended with exit $rc"
fi
exit $rc
