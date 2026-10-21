#!/usr/bin/env bash
# SM39 arm 1 + SM51 (2026-10-21): SM36's one-epoch Muon decay with ONE change, the Muon learning rate x0.5, plus weight
# snapshots along the decay (SM51 input), then uniform averages of the last 2/3/4 points, scored on the 732 pool.
#   SM36 = wt/sm42 runbook-2026-10-19-sm42-sm36.sh ARM=sm36: one-epoch decay at 32,768 x 1, strided-lane carry, replay KD
#          targets, gpu_memory_fraction 0.8, "optimizer": "expertium-iter53", fresh optimizer, from v2 epoch 12 (wt/sm21v2).
#   Arm 1 "optimizer" = the dict form of expertium-iter53 with lr_scale x0.5: Muon effective peak 0.0025 -> 0.00125.
#          The AdamW part (21,623 parameters) is not tied to lr_scale and stays as SM36. The Muon decoupled weight decay
#          is written as wd / lr_scale (wd_at_adamw_rate, Expertium's "decay at the AdamW rate"), so the decay per step
#          (lr * lr_scale * wd / lr_scale) is unchanged; only the orthogonalised step is halved.
#   Snapshots: protocol full_run.snapshot_users 1250 (the trainer's existing full-run snapshots, master.pth + optimizer.pth
#          at user boundaries) -> users 1250 / 2500 / 3750 (26.2% / 50.3% / 77.2% of the reviews) + the final weights.
#   SM51:  scripts/average_checkpoints.py decay -> avg2 (77% + 100%), avg3 (50% + 77% + 100%), avg4 (all four).
# Checks: CPU (scripts/check_decay_snapshots.py cpu): options/protocol vs SM36's own decay.json/protocol.json, optimizer
#   groups, snapshots on vs off bitwise on a short real CPU run. GPU (same script, gpu mode, first thing in the decay
#   ticket): three trainer smokes (off, off, on) must agree bitwise, else the ticket stops before the decay.
# Scoring: one ticket, the 732 pool minus the 9 never-fit users (users-732-fit.json, 723 users) for the final model and
#   the three averages (w8+s8); the 464 readout is the 458-user subset of the same rows (the 464 pool is inside the 732
#   pool and the rows are identical: SM36's 464 and pool rows agree to 0 on 458 common users). Pairs vs SM36 w8+s8
#   (wt/sm42 heldout-sm36-pools.jsonl, read only), fork w8+s8 for % (this worktree's heldout.jsonl + heldout-pools.jsonl).
#
# Expected GPU time: SM36 measured 306 min for smoke + decay (10-20 01:01 -> 06:07); here decay ticket 335 (3 smokes ~2 min,
#   margin); scoring pass 1: 4 models x 42.8 M reviews at ~61k review-evals/s (SM48 numpy) ~47 min -> 70; pass 2 only if
#   users are missing (25).
#
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm39/artifacts/small-model/sm39/runbook-sm39.sh
#   launch   via launch-sm39-after-sm55.sh (waits for SM55, respects quiet windows); resume: rerun the same command
#            (finished decay / averages / scored users are skipped). SUMMARY_ONLY=1 recomputes the summary.
# Do not edit scripts/ expertium/ rwkv/ of wt/sm39 while it runs (SourceWatch stops the epoch).
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
BASH_EXE="C:/Program Files/Git/usr/bin/bash.exe"
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm39
V2W=C:/Users/jonat/workspace/wt/sm21v2                  # read only: v2 experiment (e12)
SM42W=C:/Users/jonat/workspace/wt/sm42                  # read only: SM36 decay.json/protocol.json and rows
SM=$WT/artifacts/small-model
O=$SM/sm39
DEC_PY=$WT/scripts/expertium_decay.py
CHECK=$WT/scripts/check_decay_snapshots.py
AVG=$WT/scripts/average_checkpoints.py
SC=$WT/scripts/small_model_heldout.py
CKPT=expertium/E89-F65e-F77a/master.pth
SRC_EXP=$V2W/artifacts/accuracy/experiments/scale-w64kd-gate-sc-lanes-cont; FROM=12
REF=$SM42W/artifacts/small-model/scale-w64kd-gate-sc-lanes-cont-decay1ep32k-muon
SM36_POOL=$SM42W/artifacts/small-model/heldout-sm36-pools.jsonl; SM36_LBL=s-w64kd-gate-sc-lanes-dec12-1ep32k-muon/w8+s8
FORK_FILES="$SM/heldout.jsonl,$SM/heldout-pools.jsonl"
D=$SM/scale-w64kd-gate-sc-lanes-cont-decay1ep32k-muon0p5
CASE=$D/expertium/E89-F65e-F77a
L=s-w64kd-gate-sc-lanes-dec12-1ep32k-muon0p5
USERS=$O/users-732-fit.json; NUSERS=723
OUT=$SM/heldout-sm39-pools.jsonl
SNAP_USERS=1250
OPT='{"kind": "muon", "lr_scale": 1.7857142857142858, "momentum": 0.95, "adjust_lr_fn": "original", "groups": "matrices", "include_lora": true, "lora_weight_decay": 0.0, "wd_at_adamw_rate": true}'
DIST='{"teacher": "C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth", "alpha": 0.9, "decay_alpha": 0.5, "kd_source": "replay", "replay": "C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/fork-fp32-gate5k"}'
EXTRA="{\"distillation\": $DIST, \"state_carry\": {\"lanes\": \"strided\"}, \"chunk_reviews\": 32768, \"batch_chunks\": 1, \"optimizer\": $OPT}"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION                       # scoring uses its default cap of 0.8
log() { echo "[sm39 $(date '+%m-%d %H:%M:%S')] $*"; }
quiet_wait() {  # quiet_wait MINUTES: sleep until a job of MINUTES neither starts in nor reaches a quiet window
  local minutes=$1 out secs
  while :; do
    out=$("$PY" "$O/quiet_wait.py" "$minutes") || { log "quiet_wait helper failed; waiting 10 min"; sleep 600; continue; }
    secs=${out%% *}
    [ "$secs" -le 0 ] 2>/dev/null && return 0
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN: would wait ${secs}s for a $minutes-min job: ${out#* }"; return 0; fi
    log "waiting ${secs}s before a $minutes-min job: ${out#* }"
    sleep $(( secs < 1800 ? secs : 1800 ))
  done
}
q() {  # q MINUTES PURPOSE -- CMD...
  local minutes=$1 purpose=$2; shift 3
  quiet_wait "$minutes"
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $purpose (expected $minutes min)"; echo "    cmd: $*"; return 0; fi
  log "enqueue: $purpose (expected $minutes min)"
  "$PY" "$QUEUE" run --owner small-model --purpose "$purpose" --claim gpu --expected-minutes "$minutes" -- "$@"
  local rc=$?; log "exit $rc: $purpose"; return $rc
}
need() { if [ "$DRYRUN" = 1 ]; then log "DRYRUN warning (would stop): $1"; return 0; fi; log "STOP: $1"; return 1; }
label_users() { F="$1" LB="$2" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
p = Path(os.environ['F'])
print(len({r['user_id'] for r in map(json.loads, open(p, encoding='utf-8')) if r.get('label') == os.environ['LB'] and 'benchmark' in r}) if p.exists() else 0)"; }
LABELS=("$L/w8+s8" "$L-avg2/w8+s8" "$L-avg3/w8+s8" "$L-avg4/w8+s8")
min_users() { local m=999999 n; for lb in "${LABELS[@]}"; do n=$(label_users "$OUT" "$lb"); [ "$n" -lt "$m" ] && m=$n; done; echo "$m"; }

preconditions() {
  [ -f "$SRC_EXP/run/epoch-$FROM/$CKPT" ] || { need "epoch $FROM checkpoint of $SRC_EXP missing" || return 2; }
  if [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    need "uncommitted source changes in scripts/ expertium/ rwkv/ (SourceWatch refuses them)" || return 2; fi
  [ -f "$REF/decay.json" ] && [ -f "$REF/protocol.json" ] && [ -f "$SM36_POOL" ] && [ -f "$USERS" ] \
    && [ -f "$SM/heldout.jsonl" ] && [ -f "$SM/heldout-pools.jsonl" ] || { need "SM36 decay files, SM36 rows, fork rows or users file missing" || return 2; }
  return 0
}

main_run() {
  quiet_wait 15   # CPU work below (prepare + CPU check, a few minutes) stays out of the breaks too
  if [ "$DRYRUN" = 1 ]; then DUSE=$O/dry-prep-decay; rm -rf "$DUSE"; CPU_OUT=$O/check-decay-snapshots-cpu.json
  else DUSE=$D; CPU_OUT=$O/check-decay-snapshots-cpu-run.json; fi
  if [ ! -d "$DUSE" ]; then
    "$PY" -X utf8 "$DEC_PY" prepare --source-run "$SRC_EXP/run" --epoch "$FROM" --out "$DUSE" --fraction 1.0 --shape one_minus_sin \
      --evaluate none --seed 1 --fresh-optimizer --snapshot-users $SNAP_USERS --extra-options "$EXTRA" || { log "prepare failed"; return 1; }
  fi
  # options vs SM36's own files, optimizer groups, snapshots on/off bitwise on a short real CPU run
  if [ "$DRYRUN" = 1 ] || [ ! -f "$CASE/master.pth" ]; then
    CUDA_VISIBLE_DEVICES= "$PY" -X utf8 "$CHECK" cpu --decay "$DUSE" --reference "$REF" --out "$CPU_OUT" --snapshot-users $SNAP_USERS \
      || { log "STOP: CPU check failed ($CPU_OUT)"; return 1; }
  fi
  # options must be accepted by the prefetch allowlist and the carry/replay validators (CPU)
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
print(f"[check] state_carry options valid, replay spec valid (teacher sha256 {str(rt.teacher_sha256)[:12]}), {len(o)} option keys allowlisted")
PYEOF
  if [ "$DRYRUN" = 1 ]; then rm -rf "$DUSE"; fi

  # ---- GPU: snapshot smokes (off, off, on: bitwise) then the decay, one ticket
  if [ "$DRYRUN" = 1 ] || [ ! -f "$CASE/master.pth" ]; then
    q 335 "SM39 arm 1: SM36 Muon decay with Muon LR x0.5 + 4 weight snapshots (snapshot smokes first)" -- "$BASH_EXE" -c \
      "'$PY' -X utf8 '$CHECK' gpu --decay '$D' --out '$O/check-decay-snapshots-gpu.json' && '$PY' -X utf8 '$DEC_PY' run --out '$D'" || return 1
  fi
  [ "$DRYRUN" = 1 ] || [ -f "$CASE/master.pth" ] || { log "STOP: no decay checkpoint"; return 1; }
  if [ "$DRYRUN" != 1 ]; then
    n=$(ls -d "$CASE"/snapshots/users-* 2>/dev/null | wc -l)
    [ "$n" -eq 3 ] || { log "STOP: expected 3 snapshots (users 1250/2500/3750), found $n"; return 1; }
  fi

  # ---- CPU: SM51 averages (seconds)
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $PY $AVG decay --case $CASE --out-dir $O/avg --last 2,3,4"
  elif [ ! -f "$O/avg/avg4/master.pth" ]; then
    CUDA_VISIBLE_DEVICES= "$PY" -X utf8 "$AVG" decay --case "$CASE" --out-dir "$O/avg" --last 2,3,4 || { log "STOP: averaging failed"; return 1; }
  fi

  # ---- GPU: scoring, 732 pool minus never-fit (723 users), final + 3 averages in one pass (rows resume per user)
  for pass in 1 2; do
    if [ "$DRYRUN" = 1 ] || [ "$(min_users)" -lt $NUSERS ]; then
      q $((pass == 1 ? 70 : 25)) "SM39 scoring arm 1 final + avg2/3/4 (SM51), 732 pool minus never-fit (pass $pass)" -- \
        "$PY" -X utf8 "$SC" --pool 732 --users-file "$USERS" --variants w8+s8 --time-limit 5400 --out "$OUT" \
        --model "$L=expertium:w64:$CASE/master.pth" --model "$L-avg2=expertium:w64:$O/avg/avg2/master.pth" \
        --model "$L-avg3=expertium:w64:$O/avg/avg3/master.pth" --model "$L-avg4=expertium:w64:$O/avg/avg4/master.pth" || return 1
    fi
  done
  [ "$DRYRUN" = 1 ] || log "scored users per label (min): $(min_users) of $NUSERS"
  return 0
}

summary() {
  WTR="$WT" O="$O" OUT="$OUT" L="$L" USERS="$USERS" SM36_POOL="$SM36_POOL" SM36_LBL="$SM36_LBL" FORK_FILES="$FORK_FILES" "$PY" -X utf8 - <<'PYEOF'
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.environ["WTR"])
from scripts import small_model_gate as g

class Every:
    def __contains__(self, _):
        return True

def load(files, label):
    merged = {}
    for f in files:
        if Path(f).exists():
            merged.update(g.load(f, [label], Every())[label])
    return merged

e = os.environ
spec = json.loads(Path(e["USERS"]).read_text())
pools = {"732": set(spec["users"]), "464": set(spec["users_464_subset"])}
fork = load(e["FORK_FILES"].split(","), "fork/w8+s8")
ref = load([e["SM36_POOL"]], e["SM36_LBL"])
out, lines = {}, []
for name in ("", "-avg2", "-avg3", "-avg4"):
    label = f"{e['L']}{name}/w8+s8"
    new = load([e["OUT"]], label)
    for pool, wanted in pools.items():
        users = sorted(set(new) & set(ref) & set(fork) & wanted)
        if not users:
            lines.append(f"{label} {pool}: no common users"); continue
        per = {"a": ref, "b": new, "f": fork}
        res = {"users": len(users), "pool_users": len(wanted), "a": e["SM36_LBL"], "b": label, "delta": "b - a"}
        for head, hname in g.HEADS.items():
            va, sizes = g.weighted(per, "a", users, head)
            vb, _ = g.weighted(per, "b", users, head)
            vf, _ = g.weighted(per, "f", users, head)
            ea, eb, ef = g.rw(va, sizes), g.rw(vb, sizes), g.rw(vf, sizes)
            idx = np.random.default_rng(0).integers(0, len(sizes), size=(2000, len(sizes)))
            w = sizes[idx]
            d = ((vb[idx] - va[idx]) * w).sum(1) / w.sum(1)
            res[hname] = {"sm36": ea, "new": eb, "fork": ef, "delta": eb - ea, "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                          "sm36_pct_vs_fork": (ea / ef - 1) * 100, "new_pct_vs_fork": (eb / ef - 1) * 100, "users_new_better": int((vb < va).sum())}
        out[f"{pool}{name or '-final'}"] = res
        c, i = res["curve"], res["instant"]
        lines.append(f"| {name.lstrip('-') or 'final'} | {pool} ({len(users)}) | {c['new']:.5f} ({c['new_pct_vs_fork']:+.2f}%) | {i['new']:.5f} ({i['new_pct_vs_fork']:+.2f}%) | "
                     f"{c['sm36']:.5f} ({c['sm36_pct_vs_fork']:+.2f}%) | {i['sm36']:.5f} ({i['sm36_pct_vs_fork']:+.2f}%) | "
                     f"{c['delta']:+.5f} [{c['ci95'][0]:+.5f}, {c['ci95'][1]:+.5f}] | {i['delta']:+.5f} [{i['ci95'][0]:+.5f}, {i['ci95'][1]:+.5f}] |")
o = Path(e["O"])
(o / "sm39-summary.json").write_text(json.dumps(out, indent=1) + "\n")
head = ("# SM39 arm 1 (Muon LR x0.5) and SM51 averages vs SM36, w8+s8, review-weighted benchmark LogLoss, user bootstrap 95% CI (2,000, seed 0)\n\n"
        "| Model | Pool (users) | Curve | Instant | SM36 curve | SM36 instant | Curve delta (new - SM36) | Instant delta |\n|---|---|---|---|---|---|---|---|\n")
(o / "sm39-summary.md").write_text(head + "\n".join(lines) + "\n")
print("[sm39-summary]\n" + "\n".join(lines))
PYEOF
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN)"
rc=0
if [ "${SUMMARY_ONLY:-0}" = 1 ]; then summary; exit 0; fi
preconditions || rc=$?
[ $rc -eq 0 ] && { main_run || rc=$?; }
if [ $rc -eq 0 ]; then
  [ "$DRYRUN" = 1 ] || summary
  log "done"
  [ "$DRYRUN" = 1 ] && echo "DRYRUN OK"
else
  log "ended with exit $rc"
fi
exit $rc
