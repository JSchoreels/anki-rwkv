#!/usr/bin/env bash
# SM59 (2026-10-22): "better distillation, step 1": fine-tune SM36 (both heads trained, no head_only) with a larger teacher weight
# in the KD mix (target = w * teacher + (1 - w) * label, w = decay_alpha, the weight used while the LR decays). Arms: kd-control
# (0.5, as SM59 / SM36), kd-090 (0.9), kd-100 (1.0). Everything else = SM59 (prepare-sm59.sh, same code as branch sm59).
#   STAGE=arm ARM=kd-control|kd-090|kd-100 MIN=N bash runbook-sm59.sh    one queue ticket: smoke + fine-tune + score the
#                                             732 pool minus 9 never-fit users (one lane); then the paired summary (CPU, seconds)
#   SUMMARY_ONLY=1 STAGE=arm ARM=... bash runbook-sm59.sh               recompute the summary only
#   DRYRUN=1 ...                              print the ticket command, enqueue nothing
# Resume: rerun (a finished fine-tune is skipped; scoring resumes per user). Do not edit scripts/ expertium/ rwkv/ of
# wt/sm59 while a ticket runs (SourceWatch).
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
BASH_EXE="C:/Program Files/Git/usr/bin/bash.exe"
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm59
S=$WT/artifacts/small-model/sm59
DEC_PY=$WT/scripts/expertium_decay.py
SC=$WT/scripts/small_model_heldout.py
CKPT=expertium/E89-F65e-F77a/master.pth
SM36_POOL=C:/Users/jonat/workspace/wt/sm42/artifacts/small-model/heldout-sm36-pools.jsonl
SM36_LBL=s-w64kd-gate-sc-lanes-dec12-1ep32k-muon/w8+s8
FORK_464=C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/heldout.jsonl
FORK_POOL=C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/heldout-pools.jsonl
USERS=$S/users-732-fit.json
STAGE=${STAGE:-}; ARM=${ARM:-}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION                       # scoring uses its default cap of 0.8
log() { echo "[sm59${ARM:+-$ARM} $(date '+%m-%d %H:%M:%S')] $*"; }
q() {  # q MINUTES PURPOSE -- CMD...
  local minutes=$1 purpose=$2; shift 3
  log "enqueue: $purpose (expected $minutes min)"
  if [ "${DRYRUN:-0}" = 1 ]; then log "DRYRUN command: $*"; return 0; fi
  "$PY" "$QUEUE" run --owner small-model --purpose "$purpose" --claim gpu --expected-minutes "$minutes" -- "$@"
  local rc=$?; log "queue exit $rc: $purpose"; return $rc
}
cd "$WT" || exit 1
if [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then log "STOP: uncommitted source changes"; exit 2; fi
[ "${DRYRUN:-0}" = 1 ] || "$PY" -c "import json,sys; r=json.load(open(r'$S/check-head-only-d.json')); sys.exit(0 if r.get('passed') else 1)" \
  || { log "STOP: check D (check-head-only-d.json) has not passed"; exit 2; }
log "started: $(git -C "$WT" log --oneline -1)"

[ "$STAGE" = arm ] && { case "$ARM" in kd-control|kd-090|kd-100) true;; *) false;; esac; } || { echo "usage: STAGE=arm ARM=kd-control|kd-090|kd-100 MIN=N"; exit 2; }
D=$S/$ARM
L=s-w64kd-sm36-$ARM
OUT=$S/heldout-sm59-$ARM.jsonl

summary() {
  WTR="$WT" S="$S" D="$D" ARM="$ARM" OUT="$OUT" L="$L/w8+s8" SM36_POOL="$SM36_POOL" SM36_LBL="$SM36_LBL" FORK_464="$FORK_464" \
    FORK_POOL="$FORK_POOL" CKPT="$CKPT" "$PY" -X utf8 - <<'PYEOF'
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.environ["WTR"])
from scripts import small_model_gate as g
from scripts.eval_pools import pool_users

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
fork_lbl, a_lbl, b_lbl = "fork/w8+s8", e["SM36_LBL"], e["L"]
per = {a_lbl: load([e["SM36_POOL"]], a_lbl), b_lbl: load([e["OUT"]], b_lbl), fork_lbl: load([e["FORK_464"], e["FORK_POOL"]], fork_lbl)}
common = set(per[a_lbl]) & set(per[b_lbl]) & set(per[fork_lbl])
out, lines = {}, []
for name, pool in (("732", set(pool_users(732))), ("464", set(pool_users(464)))):
    users = sorted(common & pool)
    res = {"users": len(users), "a": a_lbl, "b": b_lbl, "delta": "b - a (arm - SM36)"}
    for head, hname in g.HEADS.items():
        va, sizes = g.weighted(per, a_lbl, users, head)
        vb, _ = g.weighted(per, b_lbl, users, head)
        vf, _ = g.weighted(per, fork_lbl, users, head)
        ea, eb, ef = g.rw(va, sizes), g.rw(vb, sizes), g.rw(vf, sizes)
        idx = np.random.default_rng(0).integers(0, len(sizes), size=(2000, len(sizes)))
        w = sizes[idx]
        deltas = ((vb[idx] - va[idx]) * w).sum(1) / w.sum(1)
        res[hname] = {"sm36": ea, "arm": eb, "fork": ef, "delta": eb - ea,
                      "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
                      "delta_pct": (eb / ea - 1) * 100, "sm36_pct_vs_fork": (ea / ef - 1) * 100, "arm_pct_vs_fork": (eb / ef - 1) * 100,
                      "users_arm_better": int((vb < va).sum())}
    out[name] = res
    c, i = res["curve"], res["instant"]
    lines.append(f"{name} pool ({len(users)} users): arm curve {c['arm']:.5f} ({c['arm_pct_vs_fork']:+.2f}% vs fork) instant {i['arm']:.5f} "
                 f"({i['arm_pct_vs_fork']:+.2f}%); SM36 curve {c['sm36']:.5f} ({c['sm36_pct_vs_fork']:+.2f}%) instant {i['sm36']:.5f} "
                 f"({i['sm36_pct_vs_fork']:+.2f}%); arm - SM36 curve {c['delta']:+.5f} [{c['ci95'][0]:+.5f}, {c['ci95'][1]:+.5f}] "
                 f"({c['delta_pct']:+.3f}%, {c['users_arm_better']} users better) instant {i['delta']:+.5f} [{i['ci95'][0]:+.5f}, {i['ci95'][1]:+.5f}] "
                 f"({i['delta_pct']:+.3f}%, {i['users_arm_better']} users better)")
r = json.loads((Path(e["D"]) / Path(e["CKPT"]).parent / "result.json").read_text())
train = {k: r.get(k) for k in ("review_presentations", "updates", "training_seconds", "reviews_per_second", "case_elapsed_seconds", "head_only")}
out["training"] = train
lines.append(f"training: {train['review_presentations']:,} reviews, {train['updates']} updates, {train['reviews_per_second']:,.0f} reviews/s, "
             f"{train['training_seconds'] / 60:.1f} min training ({train['case_elapsed_seconds'] / 60:.1f} min case)")
s = Path(e["S"])
(s / f"summary-{e['ARM']}.json").write_text(json.dumps(out, indent=1) + "\n")
(s / f"summary-{e['ARM']}.md").write_text(f"# SM59 {e['ARM']} fine-tune from SM36: {b_lbl} vs {a_lbl}\n\nw8+s8, review-weighted, "
                                          "paired user bootstrap 2,000 (seed 0) 95% CI; delta = arm - SM36; 9 never-fit users excluded.\n\n"
                                          + "\n\n".join(lines) + "\n")
print(f"[sm59-{e['ARM']} summary]\n" + "\n".join(lines))
PYEOF
}

if [ "${SUMMARY_ONLY:-0}" = 1 ]; then summary; exit $?; fi
[ -f "$D/decay.json" ] || { log "STOP: $D not prepared (prepare-sm59.sh)"; exit 2; }
MIN=${MIN:?expected minutes}
TRAIN="'$PY' -X utf8 '$DEC_PY' smoke --out '$D' && '$PY' -X utf8 '$DEC_PY' run --out '$D'"
[ -f "$D/$CKPT" ] && TRAIN="true"
SCORE="'$PY' -X utf8 '$SC' --pool 732 --users-file '$USERS' --model '$L=expertium:w64:$D/$CKPT' --only '$L/w8+s8' --time-limit 2700 --out '$OUT'"
q "$MIN" "SM59 $ARM: SM36 fine-tune, both heads, KD weight (smoke, train, score 723 users)" -- "$BASH_EXE" -c "$TRAIN && $SCORE"
rc=$?
[ "${DRYRUN:-0}" = 1 ] && { log "DRYRUN end (decay.json weights: $("$PY" -c "import json;d=json.load(open(r'$D/decay.json'))['options'];print(d['distillation']['alpha'],d['distillation']['decay_alpha'],d.get('head_only'))"))"; exit 0; }
[ -f "$D/$CKPT" ] || { log "STOP: no fine-tune checkpoint (exit $rc)"; exit 1; }
n=$(L="$L/w8+s8" OUT="$OUT" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
p = Path(os.environ['OUT'])
print(len({r['user_id'] for r in map(json.loads, open(p, encoding='utf-8')) if r.get('label') == os.environ['L'] and 'benchmark' in r}) if p.exists() else 0)")
log "scored users with a benchmark row: $n of 723"
summary || rc=1
[ $rc -eq 0 ] && log "done" || log "ended with exit $rc"
exit $rc
