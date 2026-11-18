#!/usr/bin/env bash
# SM40v2 (2026-10-18): ONE-EPOCH final decay at 32k windows from the carried-state v2 model's epoch 12, then full-history
# scoring (464 users, and the 732 pool). Prepared, not run. Copy of the SM40 recipe (small-model worktree,
# runbook-2026-10-18-dec1ep32k.sh.held: expertium_decay.py prepare --fraction 1.0 from e10 at 32,768 x 1) applied to
#   scale-w64kd-gate-sc-lanes-cont (this worktree) epoch 12 (trained at 2,048 x 16 with "state_carry": {"lanes": "strided"}).
# Decay chunk: 32,768 x 1 (one window per update, same reviews/update as the e11-e12 updates), LR 7e-4 -> 0 one_minus_sin over
#   all reviews of one epoch (--fraction 1.0, seed 1, same users and order as SM40 / dec10-1ep), the replay distillation targets
#   of e11-e12 (kd_source replay), "state_carry": {"lanes": "strided"} kept, gpu_memory_fraction 0.8.
# With batch_chunks 1 the strided layout has ONE lane: update u = window u of the user, so each window starts from its
#   predecessor's exact end state (the predecessor being the previous update); the user's first window and entities first
#   seen later start from zero. Proven on CPU by scripts/check_state_carry_lane1.py (artifacts/small-model/check-state-carry-lane1.json,
#   needs "passed": true; no trainer change was necessary).
# Pairs (464 users, review-weighted LogLoss at w8+s8, user-level bootstrap 2,000 resamples seed 0; delta = new - other):
#   vs s-w64kd-gate-ctx32768c-dec10 (plain 32k quarter decay, rows in wt/state-carry heldout.jsonl; 732 pool: wt/small-model heldout-pools.jsonl)
#   vs s-w64kd-gate-sc-lanes-dec12 (v2 quarter decay at 2,048 x 16, this worktree heldout-sm21v2.jsonl)
#   (informational: vs the fork). Labels: s-w64kd-gate-sc-lanes-dec12-1ep32k/w8+s8.
#
# Expected GPU time: one epoch = 369.5 M reviews. The 32k quarter decay dec10 ran 23,908 reviews/s (3,865 s for 92.4 M) -> a
#   plain 1-epoch decay ~258 min; the 32k epoch 10 (ticket 000420, 07:27-14:26 UTC on 10-17) took ~7 h wall clock; carry costs
#   ~+10% per update -> ~285 min of pure training, up to ~7.5 h wall clock in the epoch-10 conditions. Queued as 450 min (smoke
#   ~5 min included). Scoring: 464 users 25 (+15 resume), 732 pool 45 (+20 resume). Total ~ 8.5 h worst case.
#   Memory: largest user 6810 at 32,768 x 1 holds ONE lane table (<= 1 x its entities, smaller than the 16-lane table of the
#   2,048 x 16 check, 10.5 GiB peak); activations equal the plain 32k decay dec10 (same window size, same 0.8 cap).
#   The smoke ticket runs the real options on a scratch copy and is the only memory test; if it OOMs on 6810 stop and report.
#
#   dry run  DRYRUN=1 bash /c/Users/jonat/workspace/wt/sm21v2/artifacts/small-model/runbook-2026-10-18-sm40v2.sh
#   launch   nohup bash /c/Users/jonat/workspace/wt/sm21v2/artifacts/small-model/runbook-2026-10-18-sm40v2.sh \
#              > /c/Users/jonat/workspace/wt/sm21v2/artifacts/small-model/runbook-2026-10-18-sm40v2.log 2>&1 & disown
#   Resume: rerun the same command (finished decay / scoring rows are skipped). SUMMARY_ONLY=1 recomputes the summary only.
# Do not edit scripts/ expertium/ rwkv/ of this worktree while it runs (SourceWatch stops the epoch).
# Does not touch the GPU outside the queue; every GPU step is a r01_queue.py ticket (claim gpu, owner small-model).
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
BASH_EXE="C:/Program Files/Git/usr/bin/bash.exe"
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm21v2
SC_WT=C:/Users/jonat/workspace/wt/state-carry           # read only: dec10 464 rows
SRC_WT=C:/Users/jonat/workspace/wt/small-model          # read only: plain 32k source run, pool rows, pool scorer
SM=$WT/artifacts/small-model
DEC_PY=$WT/scripts/expertium_decay.py
SC=$WT/scripts/small_model_heldout.py                   # 464 users
SC_POOL=$SRC_WT/scripts/small_model_heldout.py          # has --pool (this worktree's copy does not)
CKPT=expertium/E89-F65e-F77a/master.pth
NAME=scale-w64kd-gate-sc-lanes-cont
EXP=$WT/artifacts/accuracy/experiments/$NAME
FROM=12
D=$SM/$NAME-decay1ep32k                                 # decay folder
L=s-w64kd-gate-sc-lanes-dec12-1ep32k
OUT=$SM/heldout-sm40v2.jsonl                            # 464-user rows (this worktree)
OUT_POOL=$SM/heldout-sm40v2-pools.jsonl                 # 732-pool rows (this worktree)
DEC10_464=$SC_WT/artifacts/small-model/heldout.jsonl; DEC10_LBL=s-w64kd-gate-ctx32768c-dec10/w8+s8
DEC10_POOL=$SRC_WT/artifacts/small-model/heldout-pools.jsonl
Q_464=$SM/heldout-sm21v2.jsonl; Q_LBL=s-w64kd-gate-sc-lanes-dec12/w8+s8
FORK_464=$SRC_WT/artifacts/small-model/heldout.jsonl
SPEC='{"lanes": "strided"}'
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
log() { echo "[sm40v2 $(date '+%m-%d %H:%M:%S')] $*"; }
q() {  # q MINUTES PURPOSE -- CMD...
  local minutes=$1 purpose=$2; shift 3
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

preconditions() {
  [ -f "$EXP/run/epoch-$FROM/$CKPT" ] && [ -f "$EXP/run/epoch-$FROM/${CKPT%master.pth}optimizer.pth" ] \
    || { need "epoch $FROM checkpoint/optimizer of $NAME missing" || return 2; }
  if [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    need "uncommitted source changes in scripts/ expertium/ rwkv/ (SourceWatch refuses them)" || return 2; fi
  for f in check-state-carry-lanes.json check-state-carry-lanes-gpu.json check-state-carry-lane1.json; do
    "$PY" -c "
import json, sys
r = json.load(open(r'$SM/$f'))
sys.exit(0 if r.get('fails') in ([], {}, 0, None) and r.get('passed') else 1)" 2>/dev/null || { need "check report $f has not passed (or is missing)" || return 2; }
  done
  [ -f "$DEC10_464" ] && [ -f "$Q_464" ] && [ -f "$DEC10_POOL" ] || { need "reference rows missing" || return 2; }
  return 0
}

main_run() {
  # ---- prepare the decay: options of the e12 run (2,048 x 16 + strided lanes + replay) with the window set to 32,768 x 1
  if [ "$DRYRUN" = 1 ]; then DUSE=$SM/cont-check-sm40v2-decay; rm -rf "$DUSE"; else DUSE=$D; fi
  if [ ! -d "$DUSE" ]; then
    "$PY" -X utf8 "$DEC_PY" prepare --source-run "$EXP/run" --epoch "$FROM" --out "$DUSE" --fraction 1.0 --shape one_minus_sin \
      --evaluate none --seed 1 --extra-options '{"chunk_reviews": 32768, "batch_chunks": 1}' || { log "prepare failed"; return 1; }
  fi
  # same users / reviews / LR shape as the plain SM40 recipe (prepared in scratch from the plain 32k source's e10 run)
  REF=$SM/cont-check-sm40v2-plain-ref; rm -rf "$REF"
  "$PY" -X utf8 "$DEC_PY" prepare --source-run "$SRC_WT/artifacts/accuracy/experiments/scale-w64kd-gate-cached-ctx32768-cont/run" \
    --epoch 10 --out "$REF" --fraction 1.0 --shape one_minus_sin --evaluate none --seed 1 > /dev/null || { log "reference prepare failed"; return 1; }
  A="$REF" B="$DUSE" SPEC="$SPEC" "$PY" -X utf8 - <<'PYEOF' || { rm -rf "$REF"; return 1; }
import json, os
from pathlib import Path
a = json.loads((Path(os.environ["A"]) / "decay.json").read_text()); b = json.loads((Path(os.environ["B"]) / "decay.json").read_text())
pa = json.loads((Path(os.environ["A"]) / "protocol.json").read_text()); pb = json.loads((Path(os.environ["B"]) / "protocol.json").read_text())
fail = []
for k in ("fraction_of_epoch_reviews", "training_users", "training_reviews", "shuffle_seed", "lr"):
    if a[k] != b[k]:
        fail.append(f"{k}: {a[k]} vs {b[k]}")
if pa["rows"] != pb["rows"]:
    fail.append("protocol rows (users and order) differ from the plain SM40 recipe")
if pa["full_run"]["lr_schedule"] != pb["full_run"]["lr_schedule"]:
    fail.append("lr_schedule differs")
ign = ("distillation", "state_carry")
oa = {k: v for k, v in a["options"].items() if k not in ign}; ob = {k: v for k, v in b["options"].items() if k not in ign}
if oa != ob:
    fail.append("options differ from the plain recipe beyond distillation/state_carry: "
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
    print("[sm40v2-check] REFUSE:\n  " + "\n  ".join(fail)); raise SystemExit(1)
print(f"[sm40v2-check] OK: {b['training_users']} users, {b['training_reviews']:,} reviews (= plain SM40 recipe), {b['lr']}, "
      f"32768 x 1, state_carry {o['state_carry']}, replay targets, gpu_memory_fraction 0.8")
PYEOF
  rm -rf "$REF"
  # options must be accepted by the prefetch allowlist (CPU)
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
print(f"[sm40v2-check] state_carry options valid, replay spec valid (teacher sha256 {str(rt.teacher_sha256)[:12]}), {len(o)} option keys allowlisted")
PYEOF
  if [ "$DRYRUN" = 1 ]; then rm -rf "$DUSE"; fi

  # ---- GPU: the decay (smoke first, inside one ticket)
  if [ "$DRYRUN" = 1 ] || [ ! -f "$D/$CKPT" ]; then
    q 450 "SM40v2 1-epoch decay at 32k with strided-lane carry from sc-lanes e12 (smoke first)" -- "$BASH_EXE" -c \
      "'$PY' -X utf8 '$DEC_PY' smoke --out '$D' && '$PY' -X utf8 '$DEC_PY' run --out '$D'" || return 1
  fi
  [ "$DRYRUN" = 1 ] || [ -f "$D/$CKPT" ] || { log "no decay checkpoint; stop"; return 1; }

  # ---- GPU: full-history scoring, 464 users then the 732 pool (up to two passes each; rows resume per user)
  for pass in 1 2; do
    if [ "$DRYRUN" = 1 ] || [ "$(label_users "$OUT" "$L/w8+s8")" -lt 464 ]; then
      q $((pass == 1 ? 25 : 15)) "SM40v2 scoring $L, 464 users (pass $pass)" -- "$PY" -X utf8 "$SC" --model "$L=expertium:w64:$D/$CKPT" \
        --only "$L/w8+s8" --time-limit 3000 --out "$OUT" || return 1
    fi
  done
  log "464 users scored: $(label_users "$OUT" "$L/w8+s8")"
  for pass in 1 2; do
    if [ "$DRYRUN" = 1 ] || [ "$(label_users "$OUT_POOL" "$L/w8+s8")" -lt 732 ]; then
      q $((pass == 1 ? 45 : 20)) "SM40v2 scoring $L, 732 pool (pass $pass)" -- "$PY" -X utf8 "$SC_POOL" --pool 732 --model "$L=expertium:w64:$D/$CKPT" \
        --only "$L/w8+s8" --time-limit 4800 --out "$OUT_POOL" || return 1
    fi
  done
  log "732-pool users scored: $(label_users "$OUT_POOL" "$L/w8+s8")"
  return 0
}

summary() {
  WTR="$WT" SM="$SM" OUT="$OUT" OUT_POOL="$OUT_POOL" L="$L/w8+s8" DEC10_464="$DEC10_464" DEC10_LBL="$DEC10_LBL" DEC10_POOL="$DEC10_POOL" \
    Q_464="$Q_464" Q_LBL="$Q_LBL" FORK_464="$FORK_464" "$PY" -X utf8 - <<'PYEOF'
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.environ["WTR"])
from scripts import small_model_gate as g

class Every:
    def __contains__(self, _):
        return True

def load(path, label):
    return g.load(path, [label], Every())[label] if Path(path).exists() else {}

def pair(name, file_a, label_a, file_b, label_b, file_fork):
    """b - a, review-weighted, per head, users present in a, b and the fork; user-level bootstrap (2,000, seed 0)."""
    fork = "fork/w8+s8"
    per = {label_a: load(file_a, label_a), label_b: load(file_b, label_b), fork: load(file_fork, fork)}
    users = sorted(set(per[label_a]) & set(per[label_b]) & set(per[fork]))
    if not users:
        return name, None, f"{name}: no common users"
    res = {"users": len(users), "a": label_a, "b": label_b, "delta": "b - a"}
    for head, hname in g.HEADS.items():
        va, sizes = g.weighted(per, label_a, users, head)
        vb, _ = g.weighted(per, label_b, users, head)
        vf, _ = g.weighted(per, fork, users, head)
        ea, eb, ef = g.rw(va, sizes), g.rw(vb, sizes), g.rw(vf, sizes)
        idx = np.random.default_rng(0).integers(0, len(sizes), size=(2000, len(sizes)))
        w = sizes[idx]
        deltas = ((vb[idx] - va[idx]) * w).sum(1) / w.sum(1)
        res[hname] = {"a": ea, "b": eb, "fork": ef, "delta": eb - ea, "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
                      "a_pct_vs_fork": (ea / ef - 1) * 100, "b_pct_vs_fork": (eb / ef - 1) * 100, "users_b_better": int((vb < va).sum())}
    c, i = res["curve"], res["instant"]
    line = (f"{name} ({len(users)} users): new curve {c['b']:.5f} ({c['b_pct_vs_fork']:+.2f}% vs fork) instant {i['b']:.5f} ({i['b_pct_vs_fork']:+.2f}%); "
            f"other curve {c['a']:.5f} ({c['a_pct_vs_fork']:+.2f}%) instant {i['a']:.5f} ({i['a_pct_vs_fork']:+.2f}%); "
            f"new - other curve {c['delta']:+.5f} [{c['ci95'][0]:+.5f}, {c['ci95'][1]:+.5f}] instant {i['delta']:+.5f} [{i['ci95'][0]:+.5f}, {i['ci95'][1]:+.5f}]")
    return name, res, line

e = os.environ
pairs = [
    pair("464 vs dec10 (plain 32k quarter decay)", e["DEC10_464"], e["DEC10_LBL"], e["OUT"], e["L"], e["FORK_464"]),
    pair("464 vs sc-lanes dec12 (v2 quarter decay at 2048x16)", e["Q_464"], e["Q_LBL"], e["OUT"], e["L"], e["FORK_464"]),
    pair("732 pool vs dec10", e["DEC10_POOL"], e["DEC10_LBL"], e["OUT_POOL"], e["L"], e["DEC10_POOL"]),
]
sm = Path(e["SM"])
(sm / "sm40v2-summary.json").write_text(json.dumps({n: r for n, r, _ in pairs}, indent=1) + "\n")
(sm / "sm40v2-summary.md").write_text("# SM40v2: 1-epoch 32k decay with strided-lane carry (w8+s8, review-weighted, user bootstrap 95% CI)\n\n"
                                      + "\n\n".join(l for _, _, l in pairs) + "\n")
print("[sm40v2-summary]\n" + "\n".join(l for _, _, l in pairs))
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
