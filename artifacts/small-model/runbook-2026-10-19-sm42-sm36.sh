#!/usr/bin/env bash
# SM42 / SM36 (2026-10-19): two one-epoch 32k final decays, each ONE change away from SM40v2 (prepared, not run).
# SM40v2 = runbook-2026-10-18-sm40v2.sh (wt/sm21v2): one-epoch decay at 32,768 x 1 with "state_carry": {"lanes": "strided"},
#   replay (full-history fork) KD targets, gpu_memory_fraction 0.8, from scale-w64kd-gate-sc-lanes-cont epoch 12 (v2).
#   ARM=sm42   CONTROL "nocarry-src". Same decay options as SM40v2 (lanes carry, replay targets, 32768 x 1, 0.8) but from the
#              PLAIN 32k run's epoch 10 (scale-w64kd-gate-cached-ctx32768-cont, trained at 32,768 x 1 WITHOUT carry, cached
#              targets) instead of v2's epoch 12. Label s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes.
#              Question: did the 2 carried-state epochs at 2,048 (e11-e12) matter, or does the long 32k decay alone give
#              SM40v2's gain?  Pairs: vs dec10 (plain quarter decay of the same e10) and vs SM40v2, on 464 and 732.
#              The source run used "cache": .../kd-teacher/fork-bf16-s2025 (window-teacher targets); the decay uses the replay
#              spec of SM40v2, passed via --extra-options (replaces the whole distillation dict), plus state_carry.
#   ARM=sm36   MUON DECAY. Same as SM40v2 (source v2 e12) but "optimizer": "expertium-iter53" (Muon effective LR 0.0025 on
#              309,888 parameters, AdamW on 21,623; scripts/muon_optimizer.py). The decay starts from a FRESH optimizer state
#              (prepare --fresh-optimizer: protocol full_run has NO init_optimizer, so the AdamW optimizer.pth of e12 is never
#              loaded; it could not be: the hybrid state_dict has keys adamw/muon). scripts/check_muon_fresh_decay.py proves it on CPU
#              (artifacts/small-model/check-muon-fresh-decay.json). The first GPU run of this optimizer: the smoke is its first
#              test (stop and report if it fails). Label s-w64kd-gate-sc-lanes-dec12-1ep32k-muon.
#              KD: replay targets do not depend on the optimizer (no cache folder or digest: ReplayTargets); TEACHER_BLIND also
#              excludes "optimizer" from the window-teacher cache key. So there is NO cold cache.
#              Pair: vs SM40v2 (AdamW arm, everything else identical), on 464 and 732.
# Preconditions mirror SM40v2 (clean scripts/ expertium/ rwkv/ tree of THIS worktree, carry checks passed, reference rows exist).
# The decay options are checked on CPU against SM40v2's own decay.json / protocol.json (read only, wt/sm21v2): identical except
# the one change of the arm; same users, order, reviews and LR schedule.
#
# Expected GPU time (SM40v2 measured, 10-18 00:31 -> 05:21 = 290 min incl. smoke, ~24,500 reviews/s over 369.5 M reviews;
# 464 scoring 15 min, 732 pool 18 min + 1.4 min resume). Queued per arm: decay 330 (smoke ~5 min included, margin; Muon 360:
# unmeasured Muon step cost), 464 scoring 25 (+15 resume), 732 pool 45 (+20 resume). Nominal per arm: SM42 400 min, SM36 430 min.
#
#   dry run  ARM=sm42 DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm42/artifacts/small-model/runbook-2026-10-19-sm42-sm36.sh
#            ARM=sm36 DRYRUN=1 bash (same file)
#   launch   normally through launch-sm42-sm36-after-prune.sh; by hand:
#            ARM=sm42 nohup bash <this file> > <wt>/artifacts/small-model/runbook-2026-10-19-sm42.log 2>&1 & disown
#   Resume: rerun the same command (finished decay / scoring rows are skipped). SUMMARY_ONLY=1 recomputes the summary only.
# Do not edit scripts/ expertium/ rwkv/ of the sm42 worktree while it runs (SourceWatch stops the epoch).
# Does not touch the GPU outside the queue; every GPU step is a r01_queue.py ticket (claim gpu, owner small-model).
set -u
ARM=${ARM:-}
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
BASH_EXE="C:/Program Files/Git/usr/bin/bash.exe"
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm42
V2W=C:/Users/jonat/workspace/wt/sm21v2                  # read only: v2 experiment (e12), SM40v2 decay.json/protocol.json
SC_WT=C:/Users/jonat/workspace/wt/state-carry           # read only: dec10 464 rows
SRC_WT=C:/Users/jonat/workspace/wt/small-model          # read only: plain 32k source run, pool rows, pool scorer
SM=$WT/artifacts/small-model
DEC_PY=$WT/scripts/expertium_decay.py
SC=$WT/scripts/small_model_heldout.py                   # 464 users
SC_POOL=$SRC_WT/scripts/small_model_heldout.py          # has --pool (this worktree's copy does not)
CKPT=expertium/E89-F65e-F77a/master.pth
V2_DECAY=$V2W/artifacts/small-model/scale-w64kd-gate-sc-lanes-cont-decay1ep32k   # SM40v2's decay folder (reference options)
SM40V2_464=$SM/heldout-sm40v2.jsonl; SM40V2_POOL=$SM/heldout-sm40v2-pools.jsonl; SM40V2_LBL=s-w64kd-gate-sc-lanes-dec12-1ep32k/w8+s8
DEC10_464=$SC_WT/artifacts/small-model/heldout.jsonl; DEC10_LBL=s-w64kd-gate-ctx32768c-dec10/w8+s8
DEC10_POOL=$SRC_WT/artifacts/small-model/heldout-pools.jsonl    # 732 = the 464 rows above + these 268 extra users (fork rows too)
FORK_464=$SRC_WT/artifacts/small-model/heldout.jsonl
SPEC='{"lanes": "strided"}'
DIST='{"teacher": "C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth", "alpha": 0.9, "decay_alpha": 0.5, "kd_source": "replay", "replay": "C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/fork-fp32-gate5k"}'
case "$ARM" in
  sm42)
    TAG=SM42; MIN_DECAY=330
    SRC_EXP=$SRC_WT/artifacts/accuracy/experiments/scale-w64kd-gate-cached-ctx32768-cont; FROM=10
    D=$SM/scale-w64kd-gate-cached-ctx32768-cont-decay1ep32k-lanes
    L=s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes
    OUT=$SM/heldout-sm42.jsonl; OUT_POOL=$SM/heldout-sm42-pools.jsonl
    EXTRA="{\"distillation\": $DIST, \"state_carry\": $SPEC, \"chunk_reviews\": 32768, \"batch_chunks\": 1}"
    FRESH=""; WHAT="from the plain 32k e10 (no carry in the source)";;
  sm36)
    TAG=SM36; MIN_DECAY=360
    SRC_EXP=$V2W/artifacts/accuracy/experiments/scale-w64kd-gate-sc-lanes-cont; FROM=12
    D=$SM/scale-w64kd-gate-sc-lanes-cont-decay1ep32k-muon
    L=s-w64kd-gate-sc-lanes-dec12-1ep32k-muon
    OUT=$SM/heldout-sm36.jsonl; OUT_POOL=$SM/heldout-sm36-pools.jsonl
    EXTRA="{\"distillation\": $DIST, \"state_carry\": $SPEC, \"chunk_reviews\": 32768, \"batch_chunks\": 1, \"optimizer\": \"expertium-iter53\"}"
    FRESH="--fresh-optimizer"; WHAT="Muon expertium-iter53 with a fresh state from sc-lanes e12";;
  *) echo "usage: ARM=sm42|sm36 [DRYRUN=1] [SUMMARY_ONLY=1] bash $0"; exit 2;;
esac
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ARM
unset HELDOUT_GPU_MEMORY_FRACTION                       # scoring uses its default cap of 0.8
log() { echo "[$ARM $(date '+%m-%d %H:%M:%S')] $*"; }
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
  [ -f "$SRC_EXP/run/epoch-$FROM/$CKPT" ] && [ -f "$SRC_EXP/run/epoch-$FROM/${CKPT%master.pth}optimizer.pth" ] \
    || { need "epoch $FROM checkpoint/optimizer of $SRC_EXP missing" || return 2; }
  if [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    need "uncommitted source changes in scripts/ expertium/ rwkv/ (SourceWatch refuses them)" || return 2; fi
  for f in check-state-carry-lanes.json check-state-carry-lanes-gpu.json check-state-carry-lane1.json; do
    "$PY" -c "
import json, sys
r = json.load(open(r'$SM/$f'))
sys.exit(0 if r.get('fails') in ([], {}, 0, None) and r.get('passed') else 1)" 2>/dev/null || { need "check report $f has not passed (or is missing)" || return 2; }
  done
  [ -f "$DEC10_464" ] && [ -f "$SM40V2_464" ] && [ -f "$SM40V2_POOL" ] && [ -f "$DEC10_POOL" ] && [ -f "$FORK_464" ] \
    && [ -f "$V2_DECAY/decay.json" ] && [ -f "$V2_DECAY/protocol.json" ] || { need "reference rows or SM40v2 decay files missing" || return 2; }
  return 0
}

main_run() {
  # ---- prepare the decay (options of SM40v2's decay: 32,768 x 1 + strided lanes + replay), checked against SM40v2's own files
  if [ "$DRYRUN" = 1 ]; then DUSE=$SM/cont-check-$ARM-decay; rm -rf "$DUSE"; else DUSE=$D; fi
  if [ ! -d "$DUSE" ]; then
    # shellcheck disable=SC2086
    "$PY" -X utf8 "$DEC_PY" prepare --source-run "$SRC_EXP/run" --epoch "$FROM" --out "$DUSE" --fraction 1.0 --shape one_minus_sin \
      --evaluate none --seed 1 $FRESH --extra-options "$EXTRA" || { log "prepare failed"; return 1; }
  fi
  A="$V2_DECAY" B="$DUSE" SRC_EXP="$SRC_EXP" FROM=$FROM SPEC="$SPEC" "$PY" -X utf8 - <<'PYEOF' || return 1
import json, os
from pathlib import Path
arm, a_dir, b_dir = os.environ["ARM"], Path(os.environ["A"]), Path(os.environ["B"])
a = json.loads((a_dir / "decay.json").read_text()); b = json.loads((b_dir / "decay.json").read_text())
pa = json.loads((a_dir / "protocol.json").read_text()); pb = json.loads((b_dir / "protocol.json").read_text())
fail = []
for k in ("fraction_of_epoch_reviews", "training_users", "training_reviews", "shuffle_seed", "lr"):
    if a[k] != b[k]:
        fail.append(f"{k}: {a[k]} vs {b[k]}")
if pa["rows"] != pb["rows"]:
    fail.append("protocol rows (users and order) differ from SM40v2")
if pa["full_run"]["lr_schedule"] != pb["full_run"]["lr_schedule"]:
    fail.append("lr_schedule differs from SM40v2")
oa, ob = dict(a["options"]), dict(b["options"])
if "optimizer" in oa:
    fail.append("SM40v2 reference unexpectedly has an optimizer option")
if arm == "sm36" and ob.pop("optimizer", None) != "expertium-iter53":
    fail.append("optimizer is not expertium-iter53")
if oa != ob:
    fail.append("options differ from SM40v2 beyond the arm's one change: "
                f"{ {k: (oa.get(k), ob.get(k)) for k in set(oa) | set(ob) if oa.get(k) != ob.get(k)} }")
if pa["training"] != pb["training"] or pa["dataset_path"] != pb["dataset_path"] or pa["dataset_revision"] != pb["dataset_revision"]:
    fail.append("protocol training/dataset differ from SM40v2")
src, epoch = Path(os.environ["SRC_EXP"]) / "run", int(os.environ["FROM"])
if Path(b["source_run"]).resolve() != src.resolve() or b["source_epoch"] != epoch:
    fail.append(f"source {b['source_run']} epoch {b['source_epoch']}")
fr = pb["full_run"]
if arm == "sm42":
    if Path(fr.get("init_optimizer", "")).resolve() != (src / f"epoch-{epoch:02d}/expertium/E89-F65e-F77a/optimizer.pth").resolve():
        fail.append("SM42 must continue from the source epoch's AdamW state (init_optimizer)")
elif "init_optimizer" in fr or fr.get("optimizer_state") != "fresh":
    fail.append("SM36 must start from a fresh optimizer state (no init_optimizer)")
o = b["options"]
if o.get("state_carry") != json.loads(os.environ["SPEC"]):
    fail.append(f"state_carry {o.get('state_carry')}")
if (o.get("chunk_reviews"), o.get("batch_chunks"), o.get("gpu_memory_fraction")) != (32768, 1, 0.8):
    fail.append(f"chunk/batch/gpu fraction {o.get('chunk_reviews')}/{o.get('batch_chunks')}/{o.get('gpu_memory_fraction')}")
d = o.get("distillation", {})
if d.get("kd_source") != "replay" or "cache" in d or d.get("alpha") != 0.9 or d.get("decay_alpha") != 0.5:
    fail.append(f"distillation {d}")
if fr["lr_schedule"] != {"decay_start": 0.0, "decay_end": 1.0, "shape": "one_minus_sin", "final": 0.0}:
    fail.append("schedule is not 7e-4 -> 0 one_minus_sin over the whole epoch")
if fail:
    print(f"[{arm}-check] REFUSE:\n  " + "\n  ".join(fail)); raise SystemExit(1)
print(f"[{arm}-check] OK vs SM40v2's decay.json/protocol.json: {b['training_users']} users, {b['training_reviews']:,} reviews, {b['lr']}, "
      f"32768 x 1, state_carry {o['state_carry']}, replay targets, gpu_memory_fraction 0.8, source epoch {b['source_epoch']}, "
      f"optimizer {'expertium-iter53 (fresh state)' if arm == 'sm36' else 'AdamW (source state continued)'}")
PYEOF
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
print(f"[check] state_carry options valid, replay spec valid (teacher sha256 {str(rt.teacher_sha256)[:12]}; no cache folder or digest, "
      f"so the optimizer cannot change the KD targets), {len(o)} option keys allowlisted")
PYEOF
  if [ "$ARM" = sm36 ]; then
    CUDA_VISIBLE_DEVICES= "$PY" -X utf8 "$WT/scripts/check_muon_fresh_decay.py" --decay "$DUSE" --out "$SM/check-muon-fresh-decay.json" \
      || { log "Muon fresh-state CPU check failed"; return 1; }
  fi
  if [ "$DRYRUN" = 1 ]; then rm -rf "$DUSE"; fi

  # ---- GPU: the decay (smoke first, inside one ticket)
  if [ "$DRYRUN" = 1 ] || [ ! -f "$D/$CKPT" ]; then
    q $MIN_DECAY "$TAG 1-epoch decay at 32k, lanes carry, $WHAT (smoke first)" -- "$BASH_EXE" -c \
      "'$PY' -X utf8 '$DEC_PY' smoke --out '$D' && '$PY' -X utf8 '$DEC_PY' run --out '$D'" || return 1
  fi
  [ "$DRYRUN" = 1 ] || [ -f "$D/$CKPT" ] || { log "no decay checkpoint; stop"; return 1; }

  # ---- GPU: full-history scoring, 464 users then the 732 pool (up to two passes each; rows resume per user)
  for pass in 1 2; do
    if [ "$DRYRUN" = 1 ] || [ "$(label_users "$OUT" "$L/w8+s8")" -lt 464 ]; then
      q $((pass == 1 ? 25 : 15)) "$TAG scoring $L, 464 users (pass $pass)" -- "$PY" -X utf8 "$SC" --model "$L=expertium:w64:$D/$CKPT" \
        --only "$L/w8+s8" --time-limit 3000 --out "$OUT" || return 1
    fi
  done
  log "464 users scored: $(label_users "$OUT" "$L/w8+s8")"
  for pass in 1 2; do
    if [ "$DRYRUN" = 1 ] || [ "$(label_users "$OUT_POOL" "$L/w8+s8")" -lt 732 ]; then
      q $((pass == 1 ? 45 : 20)) "$TAG scoring $L, 732 pool (pass $pass)" -- "$PY" -X utf8 "$SC_POOL" --pool 732 --model "$L=expertium:w64:$D/$CKPT" \
        --only "$L/w8+s8" --time-limit 4800 --out "$OUT_POOL" || return 1
    fi
  done
  log "732-pool users scored: $(label_users "$OUT_POOL" "$L/w8+s8")"
  return 0
}

summary() {
  WTR="$WT" SM="$SM" OUT="$OUT" OUT_POOL="$OUT_POOL" L="$L/w8+s8" DEC10_464="$DEC10_464" DEC10_LBL="$DEC10_LBL" DEC10_POOL="$DEC10_POOL" \
    SM40V2_464="$SM40V2_464" SM40V2_POOL="$SM40V2_POOL" SM40V2_LBL="$SM40V2_LBL" FORK_464="$FORK_464" "$PY" -X utf8 - <<'PYEOF'
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.environ["WTR"])
from scripts import small_model_gate as g

class Every:
    def __contains__(self, _):
        return True

def load(files, label):
    """Merge the per-user rows of one label over several files (732 = the 464 rows file + the 268-extra pool file)."""
    merged = {}
    for f in files:
        if Path(f).exists():
            merged.update(g.load(f, [label], Every())[label])
    return merged

def pair(name, files_a, label_a, files_b, label_b, files_fork):
    """b - a, review-weighted, per head, users present in a, b and the fork; user-level bootstrap (2,000, seed 0)."""
    fork = "fork/w8+s8"
    per = {label_a: load(files_a, label_a), label_b: load(files_b, label_b), fork: load(files_fork, fork)}
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
arm = e["ARM"]
fork_464, fork_732 = [e["FORK_464"]], [e["FORK_464"], e["DEC10_POOL"]]
pairs = []
if arm == "sm42":
    pairs.append(pair("464 vs dec10 (plain 32k quarter decay of the same e10)", [e["DEC10_464"]], e["DEC10_LBL"], [e["OUT"]], e["L"], fork_464))
    pairs.append(pair("732 vs dec10", [e["DEC10_464"], e["DEC10_POOL"]], e["DEC10_LBL"], [e["OUT_POOL"]], e["L"], fork_732))
pairs.append(pair("464 vs SM40v2 (v2 e12 + 1ep32k lanes)", [e["SM40V2_464"]], e["SM40V2_LBL"], [e["OUT"]], e["L"], fork_464))
pairs.append(pair("732 vs SM40v2", [e["SM40V2_POOL"]], e["SM40V2_LBL"], [e["OUT_POOL"]], e["L"], fork_732))
sm = Path(e["SM"])
(sm / f"{arm}-summary.json").write_text(json.dumps({n: r for n, r, _ in pairs}, indent=1) + "\n")
(sm / f"{arm}-summary.md").write_text(f"# {arm.upper()}: {e['L']} (w8+s8, review-weighted, user bootstrap 95% CI; delta = new - other)\n\n"
                                      + "\n\n".join(l for _, _, l in pairs) + "\n")
print(f"[{arm}-summary]\n" + "\n".join(l for _, _, l in pairs))
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
