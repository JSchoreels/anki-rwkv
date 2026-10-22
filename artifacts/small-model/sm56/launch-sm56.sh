#!/usr/bin/env bash
# SM56 (user approved 2026-10-22 ~10:30): full retrain on the 8,000 final-tier users FROM SCRATCH with an adaptive number
#   of constant-LR epochs, then SM55's one-epoch decay. Question: does having all 8,000 users in EVERY epoch gain more than
#   SM55 (5,000 users for epochs 1-12, 8,000 only in the decay; +0.49% / +0.53% vs the big model on 723 users)?
# Recipe (HANDOFF-sm56.md "Recipe and lineage"): the lineage's constant phase = scale-w64kd-gate-cached-ctx2048 e1-6
#   (2,048 x 16, cached bf16 KD, no carry) -> ctx8192 x 4 e7-8 -> ctx16384 x 2 e9 -> ctx32768 x 1 e10 (no carry) ->
#   sc-lanes-cont e11-12 (2,048 x 16, strided lanes, replay KD). SM56 uses sc-lanes-cont's epoch recipe for EVERY epoch:
#   w64, training seed 2025, user-order seed 20261013 (per-epoch shuffle split_seed x 1000 + epoch), AdamW (R01 schedule:
#   peak 7e-4, 5-update warmup, constant), chunk 2,048 x 16, "state_carry": {"lanes": "strided"}, KD alpha 0.9 /
#   decay_alpha 0.5 from the full-history replay targets fork-fp32-final8k (all 8,000 users), gpu_memory_fraction 0.8.
#   Options = sc-lanes-cont's options with distillation.replay -> fork-fp32-final8k (checked below), on the final tier.
#   Label windows: user 5599's last update (its final 14-review window, first reviews only) has no label in this layout
#   too; it is cut from its train_range by the epoch hook scripts/sm56_epochs.py (as SM55's decay did).
# Per epoch: one queue ticket (accuracy_loop train --epochs-per-job 1; exit 3 = more epochs to do), then a 464-set scoring
#   ticket (458 fit users, full history, w8+s8, scorer cap 0.8), losses in sm56/epochs.jsonl, decision line in this log.
# Stopping rule (user 2026-10-22, plateau on 2 consecutive epochs; sm56_rule.py): gain_k = L_{k-1} - L_k,
#   total_k = L_1 - L_k per head; stop after epoch k >= 5 when gain_k < 0.05 total_k AND gain_{k-1} < 0.05 total_{k-1}
#   on both heads; min 5, cap 14. Then the decay from epoch k: SM55's setup exactly (1 epoch at 32,768 x 1, Muon fresh
#   state, LR 7e-4 -> 0 one_minus_sin, lanes carry, replay fork-fp32-final8k, the same 8,000 users in the same seed-1
#   order, the same 5599 tail trim), checked field by field against SM55's decay.json/protocol.json; then the 732 pool
#   (723 fit users; the 464 set is a subset) and the summary vs SM55 and SM36 (paired user bootstrap 2,000, seed 0).
# Scheduling: epoch 1 waits until wt/sm59's launch-sm59.log shows check D finished (check D passed / start arm / STOP /
#   all arms finished): check D needs ~17 GB RAM, an 8k training leaves ~12 GB. Every ticket checks quiet.json and the
#   daily 21:00-01:00 break itself (a ticket that would end inside a break sleeps to its end; --max-wait-minutes makes a
#   queue wait give up with exit 75 and re-plan); CPU steps also stay out of breaks.
#
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm56/artifacts/small-model/sm56/launch-sm56.sh
#   launch   nohup bash C:/Users/jonat/workspace/wt/sm56/artifacts/small-model/sm56/launch-sm56.sh >> C:/Users/jonat/workspace/wt/sm56/artifacts/small-model/sm56/launch-sm56.log 2>&1 & disown
#   resume   the launch command again (finished epochs, scored epochs, the finished decay and scored users are skipped;
#            decisions are recomputed from epochs.jsonl)
#   stop     touch C:/Users/jonat/workspace/wt/sm56/artifacts/small-model/sm56/STOP (checked before every enqueue)
#   summary  SUMMARY_ONLY=1 bash <this file>
# Do not edit scripts/ expertium/ rwkv/ of the sm56 worktree while it runs (SourceWatch stops the epoch).
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
BASH_EXE="C:/Program Files/Git/usr/bin/bash.exe"
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm56
SM=$WT/artifacts/small-model
S=$SM/sm56
AL=$WT/scripts/accuracy_loop.py
DEC_PY=$WT/scripts/expertium_decay.py
SC=$WT/scripts/small_model_heldout.py
CHK=$WT/scripts/sm56_epochs.py
RULE=$S/sm56_rule.py
NAME=scale-w64kd-final-sc-lanes
EXP=$WT/artifacts/accuracy/experiments/$NAME
RUN=$EXP/run
TRACK=E89-F65e-F77a
CKPT=expertium/$TRACK/master.pth
V2EXP=C:/Users/jonat/workspace/wt/sm21v2/artifacts/accuracy/experiments/scale-w64kd-gate-sc-lanes-cont   # read only
export S55=C:/Users/jonat/workspace/wt/sm55/artifacts/small-model                                       # read only
SM55_DECAY=$S55/scale-w64kd-gate-sc-lanes-cont-decay1ep32k-muon-final8k
export S42=C:/Users/jonat/workspace/wt/sm42/artifacts/small-model                                       # read only
export SRC_WT=C:/Users/jonat/workspace/wt/small-model                                                   # read only (fork rows)
SM59LOG=C:/Users/jonat/workspace/wt/sm59/artifacts/small-model/sm59/launch-sm59.log                     # read only
REPLAY5K=C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/fork-fp32-gate5k
REPLAY8K=C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/fork-fp32-final8k
USERS464=$S/eval-users-464-fit.json    # 464 set minus 6 never-fit = 458
USERS732=$S/eval-users-732-fit.json    # 732 pool minus 9 never-fit = 723 (copy of sm55-eval-users-732-fit.json)
OUT464=$SM/heldout-sm56.jsonl
OUT_POOL=$SM/heldout-sm56-pools.jsonl
LBL=s-w64kd-final-sc-lanes
SPEC='{"lanes": "strided"}'
DIST="{\"teacher\": \"C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth\", \"alpha\": 0.9, \"decay_alpha\": 0.5, \"kd_source\": \"replay\", \"replay\": \"$REPLAY8K\"}"
EXTRA="{\"distillation\": $DIST, \"state_carry\": $SPEC, \"chunk_reviews\": 32768, \"batch_chunks\": 1, \"optimizer\": \"expertium-iter53\"}"  # = SM55's decay
SMOKE_EXTRA="{\"distillation\": $DIST, \"state_carry\": $SPEC, \"chunk_reviews\": 2048, \"batch_chunks\": 16}"
MAX_EPOCHS=14
MIN_EPOCH=${MIN_EPOCH:-120}   # sc-lanes-cont: 4,184 s per 5k epoch (88.5k reviews/s) x 1.575 = 110 min + smoke/setup; then measured
MIN_SCORE464=12; MIN_SCORE464B=8   # 458 users; SM55's 723-user scoring took 11 min
MIN_DECAY=510                      # SM55 measured 495 min training (19,578 reviews/s) + smoke/setup
MIN_SCORE1=20; MIN_SCORE2=12
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION   # both default to a 0.8 cap
log() { echo "[sm56 $(date '+%m-%d %H:%M:%S')] $*"; }
export EXP S RUN
window() {  # window MINUTES -> "GO <max wait minutes>" or "WAIT <seconds> <reason>"
  M="$1" Q="$QUIET" "$PY" -X utf8 - <<'PYEOF'
import json, os
from datetime import datetime, timedelta
now, need = datetime.now(), timedelta(minutes=float(os.environ["M"]))
import re
blocks = []
try:
    text = open(os.environ["Q"], encoding="utf-8").read()
except OSError as error:
    print(f"WAIT 600 quiet.json unreadable: {error}"); raise SystemExit
try:
    windows = [(w["start"], w["end"], w.get("scope", "all")) for w in json.loads(text)]
except (ValueError, KeyError, TypeError):
    # malformed JSON (e.g. unescaped quotes in a reason): take every start/end pair textually, scope assumed "all"
    windows = [(s, e, "all") for s, e in re.findall(r'"start"\s*:\s*"([^"]+)"[^{}]*?"end"\s*:\s*"([^"]+)"', text, re.S)]
for s, e, scope in windows:
    if scope in ("all", "gpu"):
        blocks.append((datetime.fromisoformat(s), datetime.fromisoformat(e), "quiet.json"))
day = now.replace(hour=0, minute=0, second=0, microsecond=0)
for d in range(-1, 9):  # the brief's daily break 21:00-01:00
    start = day + timedelta(days=d, hours=21)
    blocks.append((start, start + timedelta(hours=4), "daily 21:00-01:00"))
blocks = sorted(b for b in blocks if b[1] > now)
merged = []
for s, e, why in blocks:
    if merged and s <= merged[-1][1]:
        merged[-1] = (merged[-1][0], max(merged[-1][1], e), merged[-1][2])
    else:
        merged.append((s, e, why))
for s, e, why in merged:
    if now >= s or now + need > s:
        print(f"WAIT {int((e - now).total_seconds()) + 60} break {s:%m-%d %H:%M}-{e:%m-%d %H:%M} ({why}); "
              f"a {need.total_seconds() / 60:.0f}-min ticket would end {now + need:%m-%d %H:%M}"); raise SystemExit
    print(f"GO {max(1, int((s - need - now).total_seconds() // 60))}"); raise SystemExit
print("GO 100000")
PYEOF
}

wait_window() {  # wait_window MINUTES WHAT: sleep until a MINUTES-long step ends before the next break; sets MAXWAIT
  while true; do
    [ -e "$S/STOP" ] && { log "STOP: STOP file present"; return 1; }
    local w; w=$(window "$1")
    case "$w" in
      GO*) MAXWAIT=${w#GO }; return 0;;
      WAIT*) set -- "$1" "$2" "$w"; local s; s=$(echo "$3" | cut -d' ' -f2)
             if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $2 would wait: ${3#WAIT }"; MAXWAIT=1; return 0; fi
             log "$2 waits: ${3#WAIT * }"; sleep "$s";;
      *) log "window check failed: $w"; sleep 600;;
    esac
  done
}

q() {  # q MINUTES PURPOSE -- CMD...   (window-aware; retries after a break if the queue wait would run into it)
  local minutes=$1 purpose=$2 rc; shift 3
  while true; do
    wait_window "$minutes" "$purpose" || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $purpose (expected $minutes min, max wait $MAXWAIT min)"; echo "    cmd: $*"; return 0; fi
    log "enqueue: $purpose (expected $minutes min, max wait $MAXWAIT min)"
    "$PY" "$QUEUE" run --owner small-model --purpose "small-model: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$MAXWAIT" --launched-by "detached launcher launch-sm56.sh" -- "$@"
    rc=$?; log "exit $rc: $purpose"
    [ "$rc" = 75 ] && { log "queue wait would reach the next break; re-planning"; sleep 60; continue; }
    return $rc
  done
}
cpu_step() { wait_window "$1" "$2"; }   # CPU work also stays out of the breaks
need() { if [ "$DRYRUN" = 1 ]; then log "DRYRUN warning (would stop): $1"; return 0; fi; log "STOP: $1"; return 1; }
label_users() { F="$1" LB="$2" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
p = Path(os.environ['F'])
print(len({r['user_id'] for r in map(json.loads, open(p, encoding='utf-8')) if r.get('label') == os.environ['LB'] and 'benchmark' in r}) if p.exists() else 0)"; }
ep_dir() { echo "$RUN/epoch-$(printf "%02d" "$1")"; }
ep_done() {  # epoch k trained to completion (result.json complete + checkpoint)
  [ -f "$(ep_dir "$1")/$CKPT" ] && R="$(ep_dir "$1")/expertium/$TRACK/result.json" "$PY" -X utf8 -c "
import json, os, sys
r = json.load(open(os.environ['R']))
sys.exit(0 if r.get('status') == 'complete' and not r.get('smoke_only') else 1)" 2>/dev/null; }

preconditions() {
  for f in "$V2EXP/experiment.json" "$V2EXP/run/run.json" "$V2EXP/run/epoch-12/$CKPT" "$SM55_DECAY/decay.json" "$SM55_DECAY/protocol.json" \
           "$SM55_DECAY/protocol.untrimmed.json" "$S55/heldout-sm55-pools.jsonl" "$S42/heldout-sm36.jsonl" "$S42/heldout-sm36-pools.jsonl" \
           "$SRC_WT/artifacts/small-model/heldout.jsonl" "$SRC_WT/artifacts/small-model/heldout-pools.jsonl" "$USERS464" "$USERS732" \
           "$REPLAY8K/manifest.json"; do
    [ -f "$f" ] || { need "missing $f" || return 2; }
  done
  if [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    need "uncommitted source changes in scripts/ expertium/ rwkv/ (SourceWatch refuses them)" || return 2; fi
  grep -q "ALL OK" "$S55/check-sm11-replay-targets-sm55.txt" 2>/dev/null || { need "SM11 CPU check (sm55 check-sm11-replay-targets-sm55.txt) not ALL OK" || return 2; }
  "$PY" -X utf8 "$WT/scripts/sm11_replay_export.py" status --out "$REPLAY8K" --run "$S55/sm55-final8k-run.json" >/dev/null \
    || { need "replay targets fork-fp32-final8k incomplete" || return 2; }
  "$PY" -X utf8 "$RULE" selftest || { need "stopping-rule self-test failed" || return 2; }
  return 0
}

prepare_experiment() {
  if [ ! -f "$EXP/experiment.json" ]; then
    local opts
    opts=$(V2="$V2EXP" R8="$REPLAY8K" "$PY" -X utf8 -c "
import json, os
o = json.load(open(os.environ['V2'] + '/experiment.json'))['option_overrides']
o['distillation']['replay'] = os.environ['R8']
print(json.dumps(o))") || return 1
    "$PY" -X utf8 "$AL" prepare "$NAME" --tier final --seed 2025 --order-seed 20261013 --epochs $MAX_EPOCHS --options "$opts" \
      --epoch-runner scripts.sm56_epochs:run_epoch \
      --note "SM56: full retrain on the 8,000 final-tier users from scratch, sc-lanes-cont's epoch recipe (2048 x 16, strided lanes, replay KD fork-fp32-final8k, AdamW 7e-4 constant) every epoch, adaptive epochs (min 5, cap 14), then SM55's decay; artifacts/small-model/sm56/launch-sm56.sh" \
      || { log "STOP: accuracy_loop prepare failed"; return 1; }
  fi
  V2="$V2EXP" R8="$REPLAY8K" R5="$REPLAY5K" WTR="$WT" "$PY" -X utf8 - <<'PYEOF' || { log "STOP: experiment check failed"; return 1; }
import json, os, re, sys
from pathlib import Path
sys.path.insert(0, os.environ["WTR"])
from scripts.eval_pools import pool_users, split_info
exp, v2 = Path(os.environ["EXP"]), Path(os.environ["V2"])
e, v = json.loads((exp / "experiment.json").read_text()), json.loads((v2 / "experiment.json").read_text())
run, vrun = json.loads((exp / "run/run.json").read_text()), json.loads((v2 / "run/run.json").read_text())
_, split, tiers = split_info()
final = tiers["tiers"]["final"]
fail = []
oe, ov = json.loads(json.dumps(e["options"])), json.loads(json.dumps(v["options"]))
re_, rv = oe["distillation"].pop("replay"), ov["distillation"].pop("replay")
if Path(re_) != Path(os.environ["R8"]) or Path(rv) != Path(os.environ["R5"]):
    fail.append(f"replay folders {re_} / {rv}")
if oe != ov:
    fail.append(f"options differ from sc-lanes-cont beyond the replay folder: { {k: (oe.get(k), ov.get(k)) for k in set(oe) | set(ov) if oe.get(k) != ov.get(k)} }")
if run["options"] != e["options"] or vrun["options"] != v["options"]:
    fail.append("run.json options differ from experiment.json options")
for k in ("seed", "order_seed", "model", "track", "dataset_path", "dataset_revision", "split", "split_sha256", "label_db"):
    if e.get(k) != v.get(k):
        fail.append(f"{k}: {e.get(k)} vs {v.get(k)}")
for k in ("split_seed", "dataset_path", "dataset_revision", "snapshot_users", "lr", "slice_percent", "review_counts"):
    if run.get(k) != vrun.get(k):
        fail.append(f"run.json {k} differs")
o = e["options"]
want = {"architecture": "w64", "chunk_reviews": 2048, "batch_chunks": 16, "state_carry": {"lanes": "strided"}, "training_seed": 2025,
        "gpu_memory_fraction": 0.8}
fail += [f"option {k} = {o.get(k)}, want {w}" for k, w in want.items() if o.get(k) != w]
if o.get("optimizer") or o["distillation"].get("kd_source") != "replay" or "cache" in o["distillation"]:
    fail.append(f"optimizer {o.get('optimizer')} / distillation {o['distillation']}")
if (e["tier"], e["kind"], e["epochs"], e["init"], e["epoch_runner"]) != ("final", "train", 14, None, "scripts.sm56_epochs:run_epoch"):
    fail.append(f"experiment tier/kind/epochs/init/runner {e['tier']}/{e['kind']}/{e['epochs']}/{e['init']}/{e['epoch_runner']}")
if run["train_users"] != final["train_users"] or len(run["train_users"]) != 8000 or run["training_reviews_per_epoch"] != final["train_reviews"]:
    fail.append("run.json train users are not the 8,000 final-tier users")
if run["final_test_users"]:
    fail.append("run.json lists sealed final-test users")
users = set(run["train_users"])
for p in (464, 732):
    if users & set(pool_users(p)):
        fail.append(f"training users overlap the {p} pool")
if users & set(split["final_test_users"]):
    fail.append("training users overlap the sealed final-test users")
text = (Path(os.environ["WTR"]) / "scripts" / "gpu_r01_prefetch.py").read_text(encoding="utf-8")
allow = set(re.findall(r'"([a-z_0-9]+)"', text[text.index("set(options) - {"):text.index("or bool(options.get(\"preprocessing_cache\"))")]))
if set(o) - allow:
    fail.append(f"options outside the prefetch allowlist: {sorted(set(o) - allow)}")
from scripts.state_carry_train import validate_options
validate_options(o)
from scripts.expertium_options import ReplayTargets
rt = ReplayTargets(o["distillation"])
if fail:
    print("[sm56-exp] REFUSE:\n  " + "\n  ".join(fail)); raise SystemExit(1)
print(f"[sm56-exp] OK: {e['name']} = sc-lanes-cont's options except distillation.replay ({Path(re_).name}); seed {e['seed']}, order seed "
      f"{e['order_seed']}, AdamW R01 schedule ({run['lr']}), {len(users):,} final-tier users / {run['training_reviews_per_epoch']:,} reviews "
      f"per epoch, disjoint from the 464/732 pools and the sealed users; options allowlisted, carry options valid, replay spec valid "
      f"({rt.folder.name}); {e['epochs']} epochs max, from scratch (no init)")
PYEOF
  "$PY" -X utf8 "$CHK" check --experiment "$EXP" --out "$S/check-label-windows-sm56.json" || { log "STOP: label-window check failed (see sm56/check-label-windows-sm56.json)"; return 1; }
  # epoch-1 build (CPU): accuracy_loop's protocol writer + the hook on a scratch copy; epoch 2 chain with a fake epoch-1 result
  rm -rf "$S/check-exp"
  AL_NAME="$NAME" WTR="$WT" SM56_PROTOCOL_ONLY=1 "$PY" -X utf8 - <<'PYEOF' || { log "STOP: epoch-build check failed"; rm -rf "$S/check-exp"; return 1; }
import json, os, random, shutil, sys
from argparse import Namespace
from pathlib import Path
sys.path.insert(0, os.environ["WTR"])
from scripts import accuracy_loop as al
from scripts import run_full_training as rft
stock_writer = rft.write_protocol  # each real epoch is its own process; restore the stock writer between the two calls here
exp, scratch = Path(os.environ["EXP"]), Path(os.environ["S"]) / "check-exp"
(scratch / "run").mkdir(parents=True)
for f in ("experiment.json", "run/run.json", "label-window-trims.json"):
    shutil.copy(exp / f, scratch / f)
al.exp_dir = lambda name: scratch
assert al.run_epoch(Namespace(name=os.environ["AL_NAME"], epoch=1)) == 0
p1 = json.loads((scratch / "run/epoch-01/protocol.json").read_text())
run = json.loads((scratch / "run/run.json").read_text())
trims = {t["user_id"]: t["to"] for t in json.loads((scratch / "label-window-trims.json").read_text())["trims"]}
order = run["train_users"][:]; random.Random(run["split_seed"] * 1000 + 1).shuffle(order)
fail = []
if [r["user_id"] for r in p1["rows"]] != order:
    fail.append("epoch 1 rows are not the split_seed x 1000 + 1 shuffle of the 8,000 users")
counts = run["review_counts"]
for r in p1["rows"]:
    if r["train_range"] != [0, trims.get(r["user_id"], counts[str(r["user_id"])])]:
        fail.append(f"user {r['user_id']} train_range {r['train_range']}")
fr = p1["full_run"]
if any(k in fr for k in ("init_master", "init_optimizer")) or fr.get("update_offset", 0) != 0 or fr["evaluate"] or p1["evaluation_rows"]:
    fail.append(f"epoch 1 does not start from scratch with evaluation off: {sorted(fr)}")
if len(p1["label_window_trims"]) != len(trims) or "accuracy_loop" not in p1:
    fail.append("trims or accuracy_loop record missing")
# epoch 2: chain from a (fake) complete epoch 1
case = scratch / "run/epoch-01/expertium/E89-F65e-F77a"; case.mkdir(parents=True)
(case / "result.json").write_text(json.dumps({"status": "complete", "update_offset_end": 12345}))
rft.write_protocol = stock_writer
assert al.run_epoch(Namespace(name=os.environ["AL_NAME"], epoch=2)) == 0
p2 = json.loads((scratch / "run/epoch-02/protocol.json").read_text())
f2 = p2["full_run"]
order2 = run["train_users"][:]; random.Random(run["split_seed"] * 1000 + 2).shuffle(order2)
if not str(f2.get("init_master", "")).endswith("epoch-01/expertium/E89-F65e-F77a/master.pth") or not str(f2.get("init_optimizer", "")).endswith("optimizer.pth") \
        or f2.get("update_offset") != 12345 or [r["user_id"] for r in p2["rows"]] != order2 or len(p2["label_window_trims"]) != len(trims):
    fail.append(f"epoch 2 chain wrong: {f2}")
shutil.rmtree(scratch)
if fail:
    print("[sm56-build] REFUSE:\n  " + "\n  ".join(fail[:20])); raise SystemExit(1)
print(f"[sm56-build] OK: epoch 1 = scratch init (no init_master/optimizer, update_offset 0, evaluation off), 8,000 rows in the seed "
      f"{run['split_seed']}x1000+1 order, trims {trims} applied; epoch 2 resumes epoch 1's master + AdamW state at its update_offset_end")
PYEOF
  # smoke inputs for epoch 1: sc-lanes-cont e12 (read only) with the epoch options; must equal the experiment's options
  rm -rf "$S/smoke-prep"
  "$PY" -X utf8 "$DEC_PY" prepare --source-run "$V2EXP/run" --epoch 12 --out "$S/smoke-prep" --fraction 0.01 --evaluate none --seed 1 \
    --extra-options "$SMOKE_EXTRA" >/dev/null || { log "STOP: smoke prepare failed"; return 1; }
  WTR="$WT" "$PY" -X utf8 - <<'PYEOF' || { log "STOP: smoke options differ from the experiment's"; return 1; }
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.environ["WTR"])
from scripts.expertium_options import compatible_options
mine, _ = compatible_options(json.loads((Path(os.environ["EXP"]) / "run/run.json").read_text())["options"])
smoke = json.loads((Path(os.environ["S"]) / "smoke-prep/decay.json").read_text())["options"]
if smoke != mine:
    print("[sm56-smoke] differ:", {k: (smoke.get(k), mine.get(k)) for k in set(smoke) | set(mine) if smoke.get(k) != mine.get(k)}); raise SystemExit(1)
print("[sm56-smoke] OK: smoke-prep options == the experiment's epoch options")
PYEOF
  return 0
}

check_decay() {  # check_decay DIR: SM55's decay exactly, except the source checkpoint (run, epoch, update offset)
  A="$SM55_DECAY" B="$1" WTR="$WT" "$PY" -X utf8 - <<'PYEOF'
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.environ["WTR"])
a_dir, b_dir = Path(os.environ["A"]), Path(os.environ["B"])
a, b = json.loads((a_dir / "decay.json").read_text()), json.loads((b_dir / "decay.json").read_text())
fail = []
for k in ("options", "dropped_width32_options", "fraction_of_epoch_reviews", "training_users", "training_reviews", "train_users_from",
          "shuffle_seed", "lr", "evaluation_set", "evaluation_users", "model", "track", "trimmed_unlabelled_tails", "training_reviews_after_trims"):
    if a.get(k) != b.get(k):
        fail.append(f"{k}: {str(a.get(k))[:200]} vs {str(b.get(k))[:200]}")
for f in ("protocol.json", "protocol.untrimmed.json"):
    pa, pb = json.loads((a_dir / f).read_text()), json.loads((b_dir / f).read_text())
    for k in set(pa) | set(pb):
        if k == "full_run":
            fa, fb = dict(pa[k]), dict(pb[k])
            for x in ("init_master", "update_offset", "benchmark_db", "epoch"):
                fa.pop(x, None); fb.pop(x, None)
            if fa != fb:
                fail.append(f"{f} full_run differs: {fa} vs {fb}")
        elif pa.get(k) != pb.get(k):
            fail.append(f"{f} {k} differs")
pb = json.loads((b_dir / "protocol.json").read_text())["full_run"]
if pb.get("update_offset") != b["source_update_offset_end"] or not Path(pb["init_master"]).as_posix().endswith(f"epoch-{b['source_epoch']:02d}/expertium/E89-F65e-F77a/master.pth"):
    fail.append(f"decay does not start from the source epoch: {pb}")
if fail:
    print("[sm56-decay-check] REFUSE:\n  " + "\n  ".join(fail)); raise SystemExit(1)
print(f"[sm56-decay-check] OK vs SM55's decay: identical options (incl. replay {Path(b['options']['distillation']['replay']).name}, Muon "
      f"{b['options'].get('optimizer')}, {b['options']['chunk_reviews']} x {b['options']['batch_chunks']}), users and seed-1 order, "
      f"trims {[t['user_id'] for t in b.get('trimmed_unlabelled_tails', [])]}, LR {b['lr']}, fresh optimizer; source "
      f"{Path(b['source_run']).parent.name} e{b['source_epoch']} (update offset {b['source_update_offset_end']})")
PYEOF
}

prepare_decay() {  # prepare_decay SOURCE_RUN EPOCH OUT: SM55's prepare command with the source changed; trim; checks
  "$PY" -X utf8 "$DEC_PY" prepare --source-run "$1" --epoch "$2" --out "$3" --fraction 1.0 --shape one_minus_sin \
    --evaluate none --seed 1 --fresh-optimizer --train-tier final --extra-options "$EXTRA" >/dev/null || { log "decay prepare failed"; return 1; }
  "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$3" --trim-unlabelled-tail >/dev/null || { log "decay label windows (trim pass) failed"; return 1; }
  "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$3" --out "$4" >/dev/null || { log "decay label windows check failed (see $4)"; return 1; }
  check_decay "$3" || return 1
  CUDA_VISIBLE_DEVICES= "$PY" -X utf8 "$WT/scripts/check_muon_fresh_decay.py" --decay "$3" --out "$5" | grep -c PASS | sed 's/^/[sm56-muon] PASS lines: /'
  [ "${PIPESTATUS[0]}" = 0 ] || { log "Muon fresh-state CPU check failed (see $5)"; return 1; }
  return 0
}

dry_decay_check() {  # DRYRUN: SM55's exact prepare from sc-lanes-cont e12 (read only) + our run's options + EXTRA == SM55's
  rm -rf "$S/check-decay"
  prepare_decay "$V2EXP/run" 12 "$S/check-decay" "$S/check-decay-label-windows.json" "$S/check-muon-fresh-decay-dry.json" || { rm -rf "$S/check-decay"; return 1; }
  WTR="$WT" EXTRA="$EXTRA" A="$SM55_DECAY" "$PY" -X utf8 - <<'PYEOF' || return 1
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.environ["WTR"])
from scripts.expertium_options import compatible_options
mine, dropped = compatible_options(dict(json.loads((Path(os.environ["EXP"]) / "run/run.json").read_text())["options"], **json.loads(os.environ["EXTRA"])))
a = json.loads((Path(os.environ["A"]) / "decay.json").read_text())
if (mine, dropped) != (a["options"], a["dropped_width32_options"]):
    print("[sm56-decay-check] REFUSE: SM56 run options + EXTRA differ from SM55's decay options:",
          {k: (mine.get(k), a["options"].get(k)) for k in set(mine) | set(a["options"]) if mine.get(k) != a["options"].get(k)}); raise SystemExit(1)
print("[sm56-decay-check] OK: the SM56 run's options + the decay EXTRA == SM55's decay options (so the real decay from SM56 e_k gets them)")
PYEOF
  rm -rf "$S/check-decay"
}

wait_sm59() {
  local n=0
  while true; do
    [ -e "$S/STOP" ] && { log "STOP: STOP file present"; return 1; }
    if grep -qE '\[sm59-launch [^]]*\] (check D passed|start arm |STOP|all arms finished)' "$SM59LOG" 2>/dev/null; then
      log "SM59 check D finished: $(grep -E '\[sm59-launch [^]]*\] (check D passed|start arm |STOP|all arms finished)' "$SM59LOG" | head -1 | cut -c1-160)"; return 0; fi
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN: epoch 1 would wait for SM59 check D (last line: $(tail -1 "$SM59LOG" 2>/dev/null | cut -c1-140))"; return 0; fi
    [ $((n % 6)) -eq 0 ] && log "epoch 1 waits for SM59 check D (last line: $(tail -1 "$SM59LOG" 2>/dev/null | cut -c1-140))"
    n=$((n + 1)); sleep 300
  done
}

score() {  # score USERS_FILE POOL OUT LABEL CKPT MIN1 MIN2 WANT WHAT
  local pass
  for pass in 1 2; do
    if [ "$DRYRUN" = 1 ] || [ "$(label_users "$3" "$4/w8+s8")" -lt "$8" ]; then
      q $([ $pass = 1 ] && echo "$6" || echo "$7") "SM56 scoring $4 on $9 (pass $pass)" -- "$PY" -X utf8 "$SC" --pool "$2" \
        --users-file "$1" --model "$4=expertium:w64:$5" --only "$4/w8+s8" --time-limit 2400 --out "$3" || { log "STOP: scoring ticket failed ($4)"; return 1; }
      [ "$DRYRUN" = 1 ] && return 0
    fi
  done
  local got; got=$(label_users "$3" "$4/w8+s8")
  [ "$got" -ge "$8" ] || { log "STOP: $4 scored on $got of $8 users after 2 passes"; return 1; }
  return 0
}

epochs() {  # trains, scores and decides epoch by epoch; sets STOP_EPOCH
  local k rc decision out minutes wall
  STOP_EPOCH=0
  for k in $(seq 1 $MAX_EPOCHS); do
    if ! ep_done "$k"; then
      if [ "$k" -gt 1 ] && ! ep_done $((k - 1)) && [ "$DRYRUN" != 1 ]; then log "STOP: epoch $((k - 1)) is not complete"; return 1; fi
      [ "$k" = 1 ] && { wait_sm59 || return 1; }
      minutes=$MIN_EPOCH
      wall=$(PH="$EXP/phases.jsonl" "$PY" -X utf8 -c "
import json, math, os
from pathlib import Path
p = Path(os.environ['PH'])
w = [json.loads(l)['wall_seconds'] for l in p.read_text().splitlines() if json.loads(l).get('status') == 'complete'] if p.exists() else []
print(math.ceil(max(w[-3:]) / 60 * 1.05) + 3 if w else '')")
      [ -n "$wall" ] && minutes=$wall
      local cmd="'$PY' -X utf8 '$AL' train '$NAME' --epochs-per-job 1"
      [ "$k" = 1 ] && cmd="'$PY' -X utf8 '$DEC_PY' smoke --out '$S/smoke-prep' && $cmd"
      q "$minutes" "SM56 $NAME epoch $k/$MAX_EPOCHS (8k final tier, 2048 x 16 lanes carry, replay KD, AdamW 7e-4$([ "$k" = 1 ] && echo ', from scratch, smoke first'))" -- \
        "$BASH_EXE" -c "$cmd"
      rc=$?
      [ $rc -eq 0 ] || [ $rc -eq 3 ] || { log "STOP: epoch $k ticket failed (exit $rc)"; return 1; }
      if [ "$DRYRUN" = 1 ]; then
        log "DRYRUN: would score $LBL-e$k on the 464 set and decide; dry run stops the epoch loop after epoch 1"
        score "$USERS464" 464 "$OUT464" "$LBL-e$k" "$(ep_dir "$k")/$CKPT" $MIN_SCORE464 $MIN_SCORE464B 458 "the 464 set (458 fit users)" || return 1
        STOP_EPOCH=5; return 0
      fi
      ep_done "$k" || { log "STOP: no complete epoch-$k checkpoint after training"; return 1; }
      log "epoch $k trained: $(tail -1 "$EXP/phases.jsonl" | "$PY" -X utf8 -c "import json,sys; r=json.loads(sys.stdin.read()); print(f\"{r['review_presentations']:,} reviews, {r['training_seconds']/60:.1f} min training, {r['wall_seconds']/60:.1f} min wall, {r['reviews_per_second']:,.0f} reviews/s\")")"
    fi
    score "$USERS464" 464 "$OUT464" "$LBL-e$k" "$(ep_dir "$k")/$CKPT" $MIN_SCORE464 $MIN_SCORE464B 458 "the 464 set (458 fit users)" || return 1
    cpu_step 2 "epoch $k decision" || return 1
    out=$("$PY" -X utf8 "$RULE" record "$k" "$LBL-e$k/w8+s8"); rc=$?
    decision=$(echo "$out" | tail -1)
    [ $rc -eq 0 ] || { log "STOP: epoch $k decision failed: $(echo "$out" | head -1)"; return 1; }
    log "$(echo "$out" | head -1)"
    if [ "$decision" = stop ]; then STOP_EPOCH=$k; return 0; fi
  done
  log "STOP: no decision after $MAX_EPOCHS epochs"; return 1
}

decay_and_score() {
  local K=$1 D DL
  D=$SM/$NAME-dec$K-1ep32k-muon; DL=$LBL-dec$K-1ep32k-muon
  export D
  if [ "$DRYRUN" = 1 ]; then
    dry_decay_check || { log "STOP: decay dry check failed"; return 1; }
    log "DRYRUN: would prepare $D from $RUN epoch $K (the same command as the check above)"
  elif [ ! -d "$D" ]; then
    cpu_step 10 "decay prepare" || return 1
    prepare_decay "$RUN" "$K" "$D" "$S/check-replay-label-windows-dec$K.json" "$S/check-muon-fresh-decay-dec$K.json" \
      || { log "STOP: decay preparation or checks failed"; rm -rf "$D"; return 1; }
  else
    check_decay "$D" || { log "STOP: existing $D does not match SM55's decay"; return 1; }
  fi
  if [ "$DRYRUN" = 1 ] || [ ! -f "$D/$CKPT" ]; then
    log "decay start: 1 epoch at 32,768 x 1 from epoch $K, Muon fresh, 8,000 users (expected $MIN_DECAY min)"
    q $MIN_DECAY "SM56 1-epoch decay at 32k on the 8k final tier from $NAME e$K (581.9M reviews), lanes carry, Muon fresh (smoke first)" -- \
      "$BASH_EXE" -c "'$PY' -X utf8 '$DEC_PY' smoke --out '$D' && '$PY' -X utf8 '$DEC_PY' run --out '$D'" || { log "STOP: decay ticket failed"; return 1; }
  fi
  [ "$DRYRUN" = 1 ] || [ -f "$D/$CKPT" ] || { log "STOP: no decay checkpoint"; return 1; }
  score "$USERS732" 732 "$OUT_POOL" "$DL" "$D/$CKPT" $MIN_SCORE1 $MIN_SCORE2 723 "the 732 pool minus 9 never-fit (723; includes the 464 set)" || return 1
  [ "$DRYRUN" = 1 ] && return 0
  cpu_step 5 "summary" || return 1
  "$PY" -X utf8 "$RULE" summary "$DL/w8+s8" || { log "summary failed"; return 1; }
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN)"
if [ "${SUMMARY_ONLY:-0}" = 1 ]; then
  K=$(tail -1 "$S/epochs.jsonl" | "$PY" -X utf8 -c "import json,sys; print(json.loads(sys.stdin.read())['epoch'])")
  D=$SM/$NAME-dec$K-1ep32k-muon "$PY" -X utf8 "$RULE" summary "$LBL-dec$K-1ep32k-muon/w8+s8"; exit $?
fi
rc=0
[ "$DRYRUN" = 1 ] || cpu_step 10 "preparation checks" || exit 1
preconditions || rc=$?
[ $rc -eq 0 ] && { prepare_experiment || rc=$?; }
[ $rc -eq 0 ] && { epochs || rc=$?; }
[ $rc -eq 0 ] && { decay_and_score "$STOP_EPOCH" || rc=$?; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then echo "DRYRUN OK"; else log "done"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
