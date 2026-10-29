#!/usr/bin/env bash
# SM72 step 3 (plan approved by the user 10-24 ~20:40, "faisons comme ça, et on verra"; recipe fixed by the coordinator
#   10-26): the small model RETRAINED leak-free. Recipe = the full retrain (SM56, launch-sm56.sh) exactly, except:
#   (1) every training process gets RWKV_DELETED_CARDS=neighbour (rwkv/data_processing.get_rwkv_data: flags 0, deleted
#       cards moved to the nearest existing card's deck and preset, the SM71 transform), also recorded in the experiment's
#       "env" so accuracy_loop sets it for every child;
#   (2) distillation teacher = the big model's leak-free epoch (SM72 step 1, wt/sm72 .../big-leakfree-1ep/master.pth) and
#       replay targets = big-leakfree-1ep-final8k-neighbour (SM72 step 2, exported with the same transform); the trainer
#       refuses a teacher hash or deleted_cards mismatch (scripts/expertium_options.ReplayTargets);
#   (3) 15 fixed epochs (no plateau rule), then SM56's one-epoch 32k Muon cool-down from epoch 15 (main result), then
#       the same cool-down from epoch 10 as a side branch (comparison with the released version), after the main result;
#   (4) scoring leak-free: --deleted-cards neighbour --split-existing, the scorer never gets RWKV_DELETED_CARDS.
#   Options = SM56's experiment options with distillation.teacher / distillation.replay replaced (checked key by key);
#   seeds, users (8,000 final tier), order, AdamW 7e-4 constant, 2,048 x 16 strided-lane carry, cap 0.8 unchanged.
# Breaks: ONLY from quiet.json (scope all/gpu), re-read at every check (sleeps <= 10 min, then re-check), so a window
#   the user moves or lifts takes effect within 10 min. An epoch (~110 min, not resumable) starts only if it ends
#   before the next window; cool-downs run in segments: a watcher requests a snapshot stop (expertium_decay.py stop)
#   when a window is <= 2 min away, the next segment resumes from the snapshot (SM69 resumable decay).
#   CPU steps also stay out of windows.
# Start: waits until wt/sm72's launcher logged "verify exit 0" and "done" for the teacher export (step 2).
#
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/launch-sm72s.sh
#   launch   WMI hidden (HANDOFF-sm72s.md); log artifacts/small-model/sm72s/launch-sm72s.log
#   resume   the launch command again (finished/scored epochs, finished cool-downs, scored users are skipped;
#            an interrupted cool-down resumes from its last snapshot; an interrupted epoch restarts that epoch)
#   stop     touch C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/STOP (checked before every ticket;
#            a running cool-down segment gets a snapshot stop)
#   no side  touch .../sm72s/NO-DEC10 (skips the epoch-10 side cool-down)
# Do not edit scripts/ expertium/ rwkv/ of the sm72s worktree while it runs (SourceWatch stops the epoch).
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
BASH_EXE="C:/Program Files/Git/usr/bin/bash.exe"
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm72s
SM=$WT/artifacts/small-model
S=$SM/sm72s
G=$S/gpu
LOG=$S/launch-sm72s.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$S/launch-sm72s-dryrun.log
AL=$WT/scripts/accuracy_loop.py
DEC_PY=$WT/scripts/expertium_decay.py
SC=$WT/scripts/small_model_heldout.py
CHK=$WT/scripts/sm56_epochs.py
EXPORT=$WT/scripts/sm11_replay_export.py
RULE=$S/sm72s_rule.py
NAME=scale-w64kd-final-sc-lanes-leakfree
[ "${DRYRUN:-0}" = 1 ] && NAME=$NAME-dryrun   # the dry run prepares a scratch experiment and deletes it
EXP=$WT/artifacts/accuracy/experiments/$NAME
RUN=$EXP/run
TRACK=E89-F65e-F77a
CKPT=expertium/$TRACK/master.pth
EXP56=C:/Users/jonat/workspace/wt/sm56/artifacts/accuracy/experiments/scale-w64kd-final-sc-lanes                     # read only
S56=C:/Users/jonat/workspace/wt/sm56/artifacts/small-model                                                           # read only
DEC56=$S56/scale-w64kd-final-sc-lanes-dec15-1ep32k-muon                                                             # read only: the reference cool-down
CK56_15=$DEC56/$CKPT
V2EXP=C:/Users/jonat/workspace/wt/sm21v2/artifacts/accuracy/experiments/scale-w64kd-gate-sc-lanes-cont               # read only (smoke inputs)
SM72G=C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/gpu-big                                           # read only
SM72LOG=$SM72G/launch-sm72-big.log
BIGCK=C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72-big-leakfree-1ep/expertium/big-leakfree-1ep/master.pth
REPLAY=C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/big-leakfree-1ep-final8k-neighbour
RUN8K=C:/Users/jonat/workspace/wt/sm55/artifacts/small-model/sm55-final8k-run.json                                   # read only
REPLAY8K_OLD=C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/fork-fp32-final8k
FORK_PTH=C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth
USERS464=$S/eval-users-464-fit.json    # 464 set minus 6 never-fit = 458
USERS732=$S/eval-users-732-fit.json    # 732 pool minus 9 never-fit = 723
ULONG=$S/users-732-long.json           # the 3 long users of the 732 pool, scored at the 0.92 cap (as SM56 / SM71): 726 in all
OUT464=$G/small-lf-epochs-464.jsonl
OUTDEC=$G/small-lf-neighbour.jsonl
OUTREF=$G/small-dec15-neighbour.jsonl
LBL=s-w64kd-final-sc-lanes-leakfree
MODE=neighbour
LF="env RWKV_DELETED_CARDS=$MODE"
SPEC='{"lanes": "strided"}'
DIST="{\"teacher\": \"$BIGCK\", \"alpha\": 0.9, \"decay_alpha\": 0.5, \"kd_source\": \"replay\", \"replay\": \"$REPLAY\"}"
EXTRA="{\"distillation\": $DIST, \"state_carry\": $SPEC, \"chunk_reviews\": 32768, \"batch_chunks\": 1, \"optimizer\": \"expertium-iter53\"}"  # = SM56's cool-down
SMOKE_EXTRA="{\"distillation\": $DIST, \"state_carry\": $SPEC, \"chunk_reviews\": 2048, \"batch_chunks\": 16}"
MAX_EPOCHS=25   # user 10-28 ~08:30: 20 (was 15; cool-down from e15 scored); 10-29: 25 (cool-down from e20 scored, released); experiment.json / run.json raised with it; NO-DEC10 set
SIDE_EPOCH=10
MIN_EPOCH=${MIN_EPOCH:-125}   # SM56: 104.7-117.9 min wall per epoch (82-93k reviews/s); then 1.05 x the slowest of the last 3 + 3
MIN_SCORE464=12; MIN_SCORE464B=8
MIN_SCORE732=12; MIN_LONG=6
MIN_DECAY=${MIN_DECAY:-520}   # SM56 cool-downs: ~495-505 min training (19.5k reviews/s) + smoke/setup; segmented
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS   # scorer cap 0.8; the launcher itself runs without the transform
log() { echo "[sm72s $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
export EXP S RUN
mkdir -p "$G"

block() {  # -> "NOW <seconds to its end>" inside a scope all/gpu quiet.json window, else "NEXT <minutes to the next start>"
  Q="$QUIET" "$PY" -X utf8 - <<'PYEOF'
import json, os, re
from datetime import datetime
now = datetime.now()
try:
    text = open(os.environ["Q"], encoding="utf-8").read()
except OSError as error:
    print(f"ERR quiet.json unreadable: {error}"); raise SystemExit
try:
    wins = [(w["start"], w["end"]) for w in json.loads(text) if w.get("scope", "all") in ("all", "gpu")]
except (ValueError, KeyError, TypeError):  # malformed JSON: take every start/end pair textually, scope assumed "all"
    wins = re.findall(r'"start"\s*:\s*"([^"]+)"[^{}]*?"end"\s*:\s*"([^"]+)"', text, re.S)
blocks = sorted((datetime.fromisoformat(s), datetime.fromisoformat(e)) for s, e in wins)
merged = []
for s, e in blocks:
    if merged and s <= merged[-1][1]:
        merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    else:
        merged.append((s, e))
for s, e in merged:
    if s <= now < e:
        print(f"NOW {int((e - now).total_seconds()) + 60}"); raise SystemExit
    if s > now:
        print(f"NEXT {int((s - now).total_seconds() // 60)}"); raise SystemExit
print("NEXT 100000")
PYEOF
}

stopped() { [ -e "$S/STOP" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }   # re-read quiet.json every <= 10 min

wait_free() {  # wait_free MINUTES WHAT: returns once MINUTES fit before the next window (sets FREE = minutes to it)
  local n=0
  while true; do
    stopped && return 1
    local b; b=$(block)
    case "$b" in
      NOW*) local s=${b#NOW }
            if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $2 would wait inside a quiet window (${s}s left)"; FREE=100000; return 0; fi
            [ $((n % 6)) = 0 ] && log "$2: quiet window, ${s}s left"; n=$((n + 1)); nap "$s";;
      NEXT*) FREE=${b#NEXT }
             [ "$FREE" -ge $(( $1 + 3 )) ] && return 0
             if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $2 ($1 min) would wait: $FREE min to the next window"; FREE=100000; return 0; fi
             [ $((n % 6)) = 0 ] && log "$2 needs $1 min, $FREE min before the next window; waiting"; n=$((n + 1)); nap $(( FREE * 60 + 60 ));;
      *) log "window check failed: $b"; nap 600;;
    esac
  done
}

q() {  # q MINUTES PURPOSE -- CMD...   one ticket that must end before the next window
  local minutes=$1 purpose=$2 rc; shift 3
  while true; do
    wait_free "$minutes" "$purpose" || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $purpose (expected $minutes min)"; echo "    cmd: $*"; return 0; fi
    local maxwait=$(( FREE - minutes - 3 )); [ "$maxwait" -lt 1 ] && maxwait=1
    log "enqueue: $purpose (expected $minutes min, max wait $maxwait min, $FREE min to the next window)"
    "$PY" "$QUEUE" run --owner small-model --purpose "small-model: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm72s/launch-sm72s.sh" -- "$@"
    rc=$?; log "exit $rc: $purpose"
    [ "$rc" = 75 ] && { log "queue wait would reach the next window; re-planning"; sleep 60; continue; }
    return $rc
  done
}
cpu_step() { wait_free "$1" "$2"; }   # CPU work also stays out of the windows
need() { if [ "$DRYRUN" = 1 ]; then log "DRYRUN warning (would stop): $1"; return 0; fi; log "STOP: $1"; return 1; }
label_users() { F="$1" LB="$2" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
p = Path(os.environ['F'])
print(len({r['user_id'] for r in map(json.loads, open(p, encoding='utf-8')) if r.get('label') == os.environ['LB'] and 'benchmark_existing' in r}) if p.exists() else 0)"; }
ep_dir() { echo "$RUN/epoch-$(printf "%02d" "$1")"; }
ep_done() {  # epoch k trained to completion (result.json complete + checkpoint)
  [ -f "$(ep_dir "$1")/$CKPT" ] && R="$(ep_dir "$1")/expertium/$TRACK/result.json" "$PY" -X utf8 -c "
import json, os, sys
r = json.load(open(os.environ['R']))
sys.exit(0 if r.get('status') == 'complete' and not r.get('smoke_only') else 1)" 2>/dev/null; }
dec_done() { [ -f "$1/$CKPT" ] && R="$1/expertium/$TRACK/result.json" "$PY" -X utf8 -c "
import json, os, sys
sys.exit(0 if json.load(open(os.environ['R'])).get('status') == 'complete' else 1)" 2>/dev/null; }

wait_export() {  # step 2 (wt/sm72's launcher): the export's CPU verify passed and the launcher logged "done"
  local n=0
  while true; do
    stopped && return 1
    if grep -q "\] verify exit 0;" "$SM72LOG" 2>/dev/null && tail -1 "$SM72LOG" | grep -q "\] done$"; then
      log "teacher export verified: $(grep "\] verify exit" "$SM72LOG" | tail -1 | cut -c1-200)"; return 0; fi
    if grep -q "\] verify exit [1-9]" "$SM72LOG" 2>/dev/null; then log "STOP: the teacher export's CPU verify failed ($SM72LOG)"; return 1; fi
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN: would wait for the teacher export (last line: $(tail -1 "$SM72LOG" | cut -c1-160))"; return 0; fi
    [ $((n % 6)) = 0 ] && log "waiting for the teacher export (step 2; last line: $(tail -1 "$SM72LOG" | cut -c1-160))"
    n=$((n + 1)); sleep 300
  done
}

preconditions() {
  for f in "$EXP56/experiment.json" "$EXP56/run/run.json" "$V2EXP/run/epoch-12/$CKPT" "$DEC56/decay.json" "$DEC56/protocol.json" \
           "$CK56_15" "$BIGCK" "$RUN8K" "$REPLAY/manifest.json" "$USERS464" "$USERS732" "$ULONG" "$SM72G/bigleak-neighbour.jsonl"; do
    [ -f "$f" ] || { need "missing $f" || return 2; }
  done
  if [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    need "uncommitted source changes in scripts/ expertium/ rwkv/ (SourceWatch refuses them)" || return 2; fi
  $LF "$PY" -X utf8 "$EXPORT" status --out "$REPLAY" --run "$RUN8K" >/dev/null || { need "replay targets $REPLAY incomplete" || return 2; }
  M="$REPLAY/manifest.json" T="$BIGCK" OLD="$REPLAY8K_OLD/manifest.json" "$PY" -X utf8 - <<'PYEOF' || { need "replay manifest check failed" || return 2; }
import hashlib, json, os
from pathlib import Path
m, old = json.loads(Path(os.environ["M"]).read_text()), json.loads(Path(os.environ["OLD"]).read_text())
sha = hashlib.sha256(Path(os.environ["T"]).read_bytes()).hexdigest()
fail = []
if m.get("deleted_cards") != "neighbour":
    fail.append(f"deleted_cards {m.get('deleted_cards')}")
if m["teacher_sha256"] != sha or Path(m["teacher"]) != Path(os.environ["T"]):
    fail.append(f"teacher {m['teacher']} / {m['teacher_sha256'][:12]} vs {sha[:12]}")
for k in ("format", "kind", "dataset_path", "dataset_revision", "layout", "encoding_seed", "max_pass_reviews", "burn_in_reviews"):
    if m.get(k) != old.get(k):
        fail.append(f"{k} differs from fork-fp32-final8k's manifest")
if fail:
    print("[sm72s-replay] REFUSE: " + "; ".join(fail)); raise SystemExit(1)
print(f"[sm72s-replay] OK: deleted_cards neighbour, teacher = the big model's leak-free epoch ({sha[:12]}), same dataset/layout/passes as fork-fp32-final8k")
PYEOF
  return 0
}

prepare_experiment() {
  if [ ! -f "$EXP/experiment.json" ]; then
    local opts
    opts=$(E56="$EXP56" T="$BIGCK" R="$REPLAY" "$PY" -X utf8 -c "
import json, os
o = json.load(open(os.environ['E56'] + '/experiment.json'))['option_overrides']
o['distillation']['teacher'] = os.environ['T']
o['distillation']['replay'] = os.environ['R']
print(json.dumps(o))") || return 1
    $LF "$PY" -X utf8 "$AL" prepare "$NAME" --tier final --seed 2025 --order-seed 20261013 --epochs $MAX_EPOCHS --options "$opts" \
      --env "{\"RWKV_DELETED_CARDS\": \"$MODE\"}" --epoch-runner scripts.sm56_epochs:run_epoch \
      --note "SM72 step 3: the full retrain (SM56) leak-free: RWKV_DELETED_CARDS=neighbour, teacher = the big model's leak-free epoch, replay big-leakfree-1ep-final8k-neighbour; 15 fixed epochs, then the 32k Muon cool-down from e15 (and e10); artifacts/small-model/sm72s/launch-sm72s.sh" \
      || { log "STOP: accuracy_loop prepare failed"; return 1; }
  fi
  E56="$EXP56" T="$BIGCK" R="$REPLAY" WTR="$WT" $LF "$PY" -X utf8 - <<'PYEOF' || { log "STOP: experiment check failed"; return 1; }
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.environ["WTR"])
from scripts.eval_pools import pool_users, split_info
exp, e56d = Path(os.environ["EXP"]), Path(os.environ["E56"])
e, v = json.loads((exp / "experiment.json").read_text()), json.loads((e56d / "experiment.json").read_text())
run, vrun = json.loads((exp / "run/run.json").read_text()), json.loads((e56d / "run/run.json").read_text())
_, split, tiers = split_info()
final = tiers["tiers"]["final"]
fail = []
oe, ov = json.loads(json.dumps(e["options"])), json.loads(json.dumps(v["options"]))
de, dv = oe.pop("distillation"), ov.pop("distillation")
if Path(de.pop("teacher")) != Path(os.environ["T"]) or Path(de.pop("replay")) != Path(os.environ["R"]):
    fail.append("distillation teacher/replay are not the leak-free ones")
dv.pop("teacher"); dv.pop("replay")
if de != dv:
    fail.append(f"distillation differs beyond teacher/replay: {de} vs {dv}")
if oe != ov:
    fail.append(f"options differ from SM56's: { {k: (oe.get(k), ov.get(k)) for k in set(oe) | set(ov) if oe.get(k) != ov.get(k)} }")
if run["options"] != e["options"]:
    fail.append("run.json options differ from experiment.json options")
if e.get("env") != {"RWKV_DELETED_CARDS": "neighbour"}:
    fail.append(f"experiment env {e.get('env')}")
for k in ("seed", "order_seed", "model", "track", "dataset_path", "dataset_revision", "split", "split_sha256", "label_db", "tier", "kind",
          "epochs", "init", "epoch_runner", "train_users", "eval_users"):
    if k == "epochs" and e.get(k) in (20, 25) and v.get(k) == 15:  # user 10-28/29: epochs 16-25 added after SM56's 15
        continue
    if e.get(k) != v.get(k):
        fail.append(f"{k}: {str(e.get(k))[:80]} vs SM56 {str(v.get(k))[:80]}")
for k in ("split_seed", "dataset_path", "dataset_revision", "snapshot_users", "lr", "slice_percent", "review_counts", "train_users",
          "training_reviews_per_epoch", "final_test_users"):
    if run.get(k) != vrun.get(k):
        fail.append(f"run.json {k} differs from SM56's")
if run["train_users"] != final["train_users"] or len(run["train_users"]) != 8000:
    fail.append("run.json train users are not the 8,000 final-tier users")
users = set(run["train_users"])
for p in (464, 732):
    if users & set(pool_users(p)):
        fail.append(f"training users overlap the {p} pool")
if users & set(split["final_test_users"]):
    fail.append("training users overlap the sealed final-test users")
from scripts.state_carry_train import validate_options
validate_options(e["options"])
from scripts.expertium_options import ReplayTargets
rt = ReplayTargets(e["options"]["distillation"])  # checks the teacher hash and deleted_cards against the env
if fail:
    print("[sm72s-exp] REFUSE:\n  " + "\n  ".join(fail)); raise SystemExit(1)
print(f"[sm72s-exp] OK: {e['name']} = SM56's experiment (options, seeds {e['seed']}/{e['order_seed']}, AdamW {run['lr']}, "
      f"{len(users):,} final-tier users / {run['training_reviews_per_epoch']:,} reviews per epoch, {e['epochs']} epochs, from scratch, "
      f"epoch hook) except distillation teacher/replay = leak-free ({rt.folder.name}) and env RWKV_DELETED_CARDS=neighbour; "
      f"disjoint from the 464/732 pools and the sealed users")
PYEOF
  if [ "$DRYRUN" = 1 ] && ! $LF "$PY" -X utf8 "$EXPORT" status --out "$REPLAY" --run "$RUN8K" >/dev/null 2>&1; then
    log "DRYRUN: replay targets not complete yet: label-window check (and the trims comparison) skipped"
  else
  $LF "$PY" -X utf8 "$CHK" check --experiment "$EXP" --gate-replay "$REPLAY" --out "$S/check-label-windows-sm72s.json" >> "$LOG" 2>&1 \
    || { log "STOP: label-window check failed (see sm72s/check-label-windows-sm72s.json)"; return 1; }
  log "label windows: $("$PY" -X utf8 -c "import json; r=json.load(open(r'$S/check-label-windows-sm72s.json')); print(r['users'], 'users,', format(r['reviews'], ','), 'reviews, trims', [t['user_id'] for t in r['trims']], 'ok', r['ok'])")"
  T56="$EXP56/label-window-trims.json" T="$EXP/label-window-trims.json" "$PY" -X utf8 -c "
import json, os, sys
a, b = json.load(open(os.environ['T56'])), json.load(open(os.environ['T']))
sys.exit(0 if a['trims'] == b['trims'] else 1)" || { log "STOP: label-window trims differ from SM56's"; return 1; }
  fi
  # smoke inputs for epoch 1: sc-lanes-cont e12 (read only) with the epoch options; must equal the experiment's options
  rm -rf "$S/smoke-prep"
  $LF "$PY" -X utf8 "$DEC_PY" prepare --source-run "$V2EXP/run" --epoch 12 --out "$S/smoke-prep" --fraction 0.01 --evaluate none --seed 1 \
    --extra-options "$SMOKE_EXTRA" >/dev/null || { log "STOP: smoke prepare failed"; return 1; }
  WTR="$WT" "$PY" -X utf8 - <<'PYEOF' || { log "STOP: smoke options differ from the experiment's"; return 1; }
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.environ["WTR"])
from scripts.expertium_options import compatible_options
mine, _ = compatible_options(json.loads((Path(os.environ["EXP"]) / "run/run.json").read_text())["options"])
smoke = json.loads((Path(os.environ["S"]) / "smoke-prep/decay.json").read_text())["options"]
if smoke != mine:
    print("[sm72s-smoke] differ:", {k: (smoke.get(k), mine.get(k)) for k in set(smoke) | set(mine) if smoke.get(k) != mine.get(k)}); raise SystemExit(1)
print("[sm72s-smoke] OK: smoke-prep options == the experiment's epoch options")
PYEOF
  return 0
}

check_decay() {  # check_decay DIR: SM56's cool-down from e15 exactly, except the source epoch and the leak-free teacher/replay
  A="$DEC56" B="$1" T="$BIGCK" R="$REPLAY" "$PY" -X utf8 - <<'PYEOF'
import json, os
from pathlib import Path
a_dir, b_dir = Path(os.environ["A"]), Path(os.environ["B"])
a, b = json.loads((a_dir / "decay.json").read_text()), json.loads((b_dir / "decay.json").read_text())
fail = []
oa, ob = json.loads(json.dumps(a["options"])), json.loads(json.dumps(b["options"]))
da, db = oa.pop("distillation"), ob.pop("distillation")
if Path(db.pop("teacher")) != Path(os.environ["T"]) or Path(db.pop("replay")) != Path(os.environ["R"]):
    fail.append("cool-down distillation teacher/replay are not the leak-free ones")
da.pop("teacher"); da.pop("replay")
if oa != ob or da != db:
    fail.append(f"options differ: { {k: (oa.get(k), ob.get(k)) for k in set(oa) | set(ob) if oa.get(k) != ob.get(k)} } / {da} vs {db}")
for k in ("dropped_width32_options", "fraction_of_epoch_reviews", "training_users", "training_reviews", "train_users_from",
          "shuffle_seed", "lr", "evaluation_set", "evaluation_users", "model", "track", "trimmed_unlabelled_tails", "training_reviews_after_trims"):
    if a.get(k) != b.get(k):
        fail.append(f"{k}: {str(a.get(k))[:200]} vs {str(b.get(k))[:200]}")
a_prot = "protocol.prepared.json" if (a_dir / "protocol.prepared.json").exists() else "protocol.json"
b_prot = "protocol.prepared.json" if (b_dir / "protocol.prepared.json").exists() else "protocol.json"
pa, pb = json.loads((a_dir / a_prot).read_text()), json.loads((b_dir / b_prot).read_text())
for k in set(pa) | set(pb):
    if k == "full_run":
        fa, fb = dict(pa[k]), dict(pb[k])
        for x in ("init_master", "update_offset", "benchmark_db", "epoch", "snapshot_users", "control_dir", "options"):
            fa.pop(x, None); fb.pop(x, None)
        if fa != fb:
            fail.append(f"protocol full_run differs: { {x: (fa.get(x), fb.get(x)) for x in set(fa) | set(fb) if fa.get(x) != fb.get(x)} }")
    elif k != "options" and pa.get(k) != pb.get(k):
        fail.append(f"protocol {k} differs")
fb = pb["full_run"]
if fb.get("update_offset") != b["source_update_offset_end"] or not Path(fb["init_master"]).as_posix().endswith(f"epoch-{b['source_epoch']:02d}/expertium/E89-F65e-F77a/master.pth") \
        or "leakfree" not in Path(b["source_run"]).as_posix():
    fail.append(f"cool-down does not start from the leak-free run's source epoch: {fb.get('init_master')}")
if fail:
    print("[sm72s-decay-check] REFUSE:\n  " + "\n  ".join(fail)); raise SystemExit(1)
print(f"[sm72s-decay-check] OK vs SM56's cool-down from e15: identical options except the leak-free teacher/replay (Muon "
      f"{b['options'].get('optimizer')}, {b['options']['chunk_reviews']} x {b['options']['batch_chunks']}), users and seed-1 order, "
      f"trims {[t['user_id'] for t in b.get('trimmed_unlabelled_tails', [])]}, LR {b['lr']}, fresh optimizer; source e{b['source_epoch']} "
      f"(update offset {b['source_update_offset_end']})")
PYEOF
}

prepare_decay() {  # prepare_decay SOURCE_RUN EPOCH OUT LW_JSON MUON_JSON: SM56's prepare command; trim; checks
  $LF "$PY" -X utf8 "$DEC_PY" prepare --source-run "$1" --epoch "$2" --out "$3" --fraction 1.0 --shape one_minus_sin \
    --evaluate none --seed 1 --fresh-optimizer --train-tier final --extra-options "$EXTRA" >/dev/null || { log "decay prepare failed"; return 1; }
  $LF "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$3" --trim-unlabelled-tail >/dev/null || { log "decay label windows (trim pass) failed"; return 1; }
  $LF "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$3" --out "$4" >/dev/null || { log "decay label windows check failed (see $4)"; return 1; }
  check_decay "$3" >> "$LOG" 2>&1 || { log "decay check failed (see the log)"; return 1; }
  CUDA_VISIBLE_DEVICES= $LF "$PY" -X utf8 "$WT/scripts/check_muon_fresh_decay.py" --decay "$3" --out "$5" | grep -c PASS | sed 's/^/[sm72s-muon] PASS lines: /' >> "$LOG"
  [ "${PIPESTATUS[0]}" = 0 ] || { log "Muon fresh-state CPU check failed (see $5)"; return 1; }
  return 0
}

score() {  # score USERS_FILE POOL OUT LABEL MODEL_SPEC MIN1 MIN2 WANT WHAT [ENV]: leak-free scoring, 2 passes
  local pass
  for pass in 1 2; do
    if [ "$DRYRUN" = 1 ] || [ "$(label_users "$3" "$4/w8+s8")" -lt "$8" ]; then
      q $([ $pass = 1 ] && echo "$6" || echo "$7") "SM72 leak-free scoring $4 on $9 (pass $pass)" -- \
        env -u RWKV_DELETED_CARDS ${10:-} "$PY" -X utf8 "$SC" --pool "$2" --users-file "$1" --model "$4=$5" --only "$4/w8+s8" \
        --time-limit 2400 --split-existing --deleted-cards $MODE --out "$3" >> "$G/score.out" 2>&1 || { log "STOP: scoring ticket failed ($4)"; return 1; }
      [ "$DRYRUN" = 1 ] && return 0
    fi
  done
  local got; got=$(label_users "$3" "$4/w8+s8")
  [ "$got" -ge "$8" ] || { log "STOP: $4 scored on $got of $8 users after 2 passes"; return 1; }
  return 0
}
score732() {  # score732 OUT LABEL MODEL_SPEC: 723 users at the 0.8 cap, then the 3 long users at 0.92 (726 in all)
  score "$USERS732" 732 "$1" "$2" "$3" $MIN_SCORE732 8 723 "the 732 pool minus 9 never-fit (723)" || return 1
  local want=$(( 723 + 3 ))
  if [ "$DRYRUN" = 1 ] || [ "$(label_users "$1" "$2/w8+s8")" -lt "$want" ]; then
    q $MIN_LONG "SM72 leak-free scoring $2 on the 3 long users of the 732 pool (cap 0.92)" -- \
      env -u RWKV_DELETED_CARDS HELDOUT_GPU_MEMORY_FRACTION=0.92 "$PY" -X utf8 "$SC" --pool 732 --users-file "$ULONG" --model "$2=$3" \
      --only "$2/w8+s8" --time-limit 2400 --split-existing --deleted-cards $MODE --out "$1" >> "$G/score.out" 2>&1 || log "long users ticket failed ($2); continuing with $(label_users "$1" "$2/w8+s8") users"
  fi
  return 0
}

epochs() {
  local k rc out minutes wall
  for k in $(seq 1 $MAX_EPOCHS); do
    if ! ep_done "$k"; then
      if [ "$k" -gt 1 ] && ! ep_done $((k - 1)) && [ "$DRYRUN" != 1 ]; then log "STOP: epoch $((k - 1)) is not complete"; return 1; fi
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
      q "$minutes" "SM72 small model leak-free (sm72s) epoch $k/$MAX_EPOCHS (8k final tier, 2048 x 16 lanes carry, leak-free replay KD, AdamW 7e-4$([ "$k" = 1 ] && echo ', from scratch, smoke first'))" -- \
        $LF "$BASH_EXE" -c "$cmd"
      rc=$?
      [ $rc -eq 0 ] || [ $rc -eq 3 ] || { log "STOP: epoch $k ticket failed (exit $rc)"; return 1; }
      if [ "$DRYRUN" = 1 ]; then
        score "$USERS464" 464 "$OUT464" "$LBL-e$k" "expertium:w64:$(ep_dir "$k")/$CKPT" $MIN_SCORE464 $MIN_SCORE464B 458 "the 464 set (458 fit users)" || return 1
        log "DRYRUN: epochs 2-$MAX_EPOCHS the same (one ticket + one 464 scoring each)"; return 0
      fi
      ep_done "$k" || { log "STOP: no complete epoch-$k checkpoint after training"; return 1; }
      log "epoch $k trained: $(tail -1 "$EXP/phases.jsonl" | "$PY" -X utf8 -c "import json,sys; r=json.loads(sys.stdin.read()); print(f\"{r['review_presentations']:,} reviews, {r['training_seconds']/60:.1f} min training, {r['wall_seconds']/60:.1f} min wall, {r['reviews_per_second']:,.0f} reviews/s\")")"
    fi
    score "$USERS464" 464 "$OUT464" "$LBL-e$k" "expertium:w64:$(ep_dir "$k")/$CKPT" $MIN_SCORE464 $MIN_SCORE464B 458 "the 464 set (458 fit users)" || return 1
    cpu_step 2 "epoch $k record" || return 1
    out=$("$PY" -X utf8 "$RULE" record "$k" "$LBL-e$k/w8+s8") || { log "STOP: epoch $k record failed: $out"; return 1; }
    log "$out"
  done
  return 0
}

run_decay() {  # run_decay DIR WHAT: segments to the next window or the end; snapshot stop before a window; resumable
  local D=$1 what=$2 rc qpid stop_sent b
  while ! dec_done "$D"; do
    stopped && return 1
    wait_free 30 "$what" || return 1
    local seg=$(( FREE - 5 )); [ "$seg" -gt "$MIN_DECAY" ] && seg=$MIN_DECAY
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $what (segment to the next window or the end, expected $seg min of ~$MIN_DECAY)"
      echo "    cmd: $LF $PY -X utf8 $DEC_PY smoke --out $D && $LF $PY -X utf8 $DEC_PY run --out $D"; return 0; fi
    # a stop flag left by the previous segment's stop request would stop the smoke at once: decay.json's control_dir
    # points the smoke copy at this folder's control/, and only `run` (not `smoke`) clears it (10-28 00:21 failure)
    rm -f "$D/control/STOP"
    log "enqueue: $what (expected $seg min of ~$MIN_DECAY, $FREE min to the next window)"
    "$PY" "$QUEUE" run --owner small-model --purpose "small-model: $what (segment to the next window or the end)" --claim gpu \
      --expected-minutes "$seg" --max-wait-minutes $(( FREE - 20 > 1 ? FREE - 20 : 1 )) --launched-by "detached launcher sm72s/launch-sm72s.sh" -- \
      $LF "$BASH_EXE" -c "'$PY' -X utf8 '$DEC_PY' smoke --out '$D' && '$PY' -X utf8 '$DEC_PY' run --out '$D'" >> "$G/decay.out" 2>&1 &
    qpid=$!
    stop_sent=0
    while kill -0 "$qpid" 2>/dev/null; do
      b=$(block)
      if [ -e "$S/STOP" ] || [[ "$b" == NOW* ]] || { [[ "$b" == NEXT* ]] && [ "${b#NEXT }" -le 2 ]; }; then
        # re-requested every 30 s: `run` clears control/STOP when it starts, so a request made while queued is repeated
        [ "$stop_sent" = 0 ] && log "requesting a snapshot stop ($b)"
        "$PY" -X utf8 "$DEC_PY" stop --out "$D" > /dev/null 2>&1; stop_sent=1
      fi
      sleep 30
    done
    wait "$qpid"; rc=$?; log "exit $rc: $what segment"
    [ "$rc" = 75 ] && { sleep 60; continue; }
    dec_done "$D" && break
    [ "$stop_sent" = 0 ] && { log "STOP: $what segment ended without completing and without a stop request (exit $rc); see $G/decay.out"; return 1; }
  done
  return 0
}

decay_and_score() {  # decay_and_score K
  local K=$1 D DL
  D=$SM/$NAME-dec$K-1ep32k-muon; DL=$LBL-dec$K-1ep32k-muon
  if [ "$DRYRUN" = 1 ]; then
    log "DRYRUN: would prepare $D from $RUN epoch $K (SM56's prepare command + EXTRA, trims, check vs SM56's e15 cool-down, Muon CPU check)"
  elif [ ! -d "$D" ]; then
    cpu_step 15 "cool-down e$K prepare" || return 1
    prepare_decay "$RUN" "$K" "$D" "$S/check-replay-label-windows-dec$K.json" "$S/check-muon-fresh-decay-dec$K.json" \
      || { log "STOP: cool-down e$K preparation or checks failed"; rm -rf "$D"; return 1; }
    log "cool-down e$K prepared: $(tail -1 "$LOG" | cut -c1-40) $(grep -h "sm72s-decay-check" "$LOG" | tail -1 | cut -c1-200)"
  else
    check_decay "$D" >> "$LOG" 2>&1 || { log "STOP: existing $D does not match SM56's cool-down"; return 1; }
  fi
  run_decay "$D" "SM72 small model leak-free (sm72s): 1-epoch 32k Muon cool-down from e$K (8k final tier, 581.9M reviews, lanes carry, leak-free replay KD)" || return 1
  [ "$DRYRUN" = 1 ] || dec_done "$D" || { log "STOP: no cool-down e$K checkpoint"; return 1; }
  score732 "$OUTDEC" "$DL" "expertium:w64:$D/$CKPT" || return 1
  [ "$DRYRUN" = 1 ] && return 0
  cpu_step 5 "summary" || return 1
  EXP="$EXP" "$PY" -X utf8 "$RULE" summary >> "$LOG" 2>&1 || { log "summary failed"; return 1; }
  log "summary written: $S/sm72s-summary.md (cool-down from e$K)"
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN)"
rc=0
wait_export || rc=1
[ $rc -eq 0 ] && { [ "$DRYRUN" = 1 ] || cpu_step 15 "preparation checks" || rc=1; }
[ $rc -eq 0 ] && { preconditions >> "$LOG" 2>&1 || rc=$?; }
[ $rc -eq 0 ] && { prepare_experiment >> "$LOG" 2>&1 || rc=$?; }
# reference row: the small model's cool-down from epoch 15 (SM56, leak inputs) scored leak-free (~15 min)
[ $rc -eq 0 ] && { score732 "$OUTREF" sm56dec15 "expertium:w64:$CK56_15" || rc=$?; }
[ $rc -eq 0 ] && { epochs || rc=$?; }
[ $rc -eq 0 ] && { decay_and_score $MAX_EPOCHS || rc=$?; }
if [ $rc -eq 0 ] && [ ! -e "$S/NO-DEC10" ]; then decay_and_score $SIDE_EPOCH || rc=$?; fi
[ "$DRYRUN" = 1 ] && rm -rf "$EXP"
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
