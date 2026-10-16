#!/usr/bin/env bash
# Small-model runbook 2026-10-16c (docs/SMALL_MODEL.md "Re-assessment 2026-10-16 ~03:30 (G and Q7)", fallback 2 / "Q8").
# runbook-2026-10-16.sh, runbook-2026-10-16b.sh and runbook-next.sh stay untouched; helpers below are copies of the b script.
#
# Stage `arch`: a from-scratch run of scale-w64kd-<TIER>-cached-ctx2048 with another OPTIMIZER (Muon: "expertium-iter53"), in a NEW
# experiment scale-<ARCH>kd-<TIER>-cached-ctx<CHUNK>-<tag> (tag muon for expertium-iter53). The options are exactly the source run's
# option_overrides (w64, KD with the same teacher/alpha/cache path, chunk 2048 x 16, the gate-tier users, seed 2025, order seed from the
# split) plus "optimizer". The source run (RUN_TAG=ctx2048) is only READ.
#   launch  cd C:/Users/jonat/workspace/wt/small-model && SM=$PWD/artifacts/small-model && TIER=gate ARCH=w64 OPTIMIZER=expertium-iter53 CHUNK=2048 \
#             nohup bash $SM/runbook-2026-10-16c.sh arch > $SM/runbook-2026-10-16c-arch.log 2>&1 & disown
#   dry run DRYRUN=1 TIER=gate ARCH=w64 OPTIMIZER=expertium-iter53 CHUNK=2048 bash $SM/runbook-2026-10-16c.sh arch
# Steps: (0) refuse (exit 2) while runbook-2026-10-16.sh / -16b.sh / runbook-next.sh runs, or if the source run lacks epochs 1..EPOCHS-1
#   checkpoints for the gate references; (1) CPU preflight: Muon grouping of the w64 master (parameter counts, effective LRs), optimizer
#   state save/load round trip, KD cache key of the source vs the new run, option keys inside the gpu_r01_prefetch allowlist;
#   (2) create the experiment with accuracy_loop.py prepare (once; relaunch resumes) and check it against the source;
#   (3) score the missing AdamW reference epochs on the 100 gate users (the ctx2048 run has no e3 gate row; e1,e2,e4,e5,e6 exist);
#   (4) one ticket per epoch (smoke first, without the AdamW optimizer state), a 100-user score after each epoch and the gate against
#   the AdamW run's same-epoch label <SRC_LBL>-e<N>: test A (Δ < 0 and upper95 < 0 on both heads) is reported, the STOP rule is
#   "upper95 >= 0 on both heads" for N < EPOCHS (FORCE=1 continues); (5) 0.25-epoch decay from the last epoch into $SM/<NAME>-decay;
#   (6) 464-user scoring of e<last> and dec<last>, resume ticket only if users remain.
# Environment: TIER (required), ARCH (w64), OPTIMIZER (required: expertium-iter53, or a JSON object {"kind":"muon",...}), CHUNK (2048),
#   BATCH_CHUNKS (32768/CHUNK), EPOCHS (4, at most 6: the AdamW references exist up to e6), RUN_TAG (ctx2048 = the source run and
#   reference labels), EPOCH1_MIN (85: cold KD cache, see below), EPOCH_MIN (70), FORCE=1, DRYRUN=1.
# KD cache: the teacher-target cache key hashes the trainer options except "distillation", so adding "optimizer" gives a NEW key
#   (batches-2b2206de733f instead of batches-e66b1ec9cc5b, +6 GB): epoch 1 recomputes the teacher targets online (like the AdamW run's
#   epoch 1: 80.8k reviews/s, 76 min), epochs 2+ hit the new cache. The preflight prints both keys.
# DRYRUN=1 stubs the queue (echo), builds the experiment only as dryrun-<NAME> (deleted afterwards), never writes the real experiment,
#   and turns missing-artifact exits into warnings.
set -u
STAGE=${1:?stage arch}
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

arch() {
  TIER=${TIER:?gate or final}; ARCH=${ARCH:-w64}; OPTIMIZER=${OPTIMIZER:?expertium-iter53 or a JSON object}
  CHUNK=${CHUNK:-2048}; EPOCHS=${EPOCHS:-4}; RUN_TAG=${RUN_TAG:-ctx2048}
  BATCH_CHUNKS=${BATCH_CHUNKS:-$((32768 / CHUNK))}
  EPOCH1_MIN=${EPOCH1_MIN:-85}; EPOCH_MIN=${EPOCH_MIN:-70}
  [ "$ARCH" = w64 ] || { log "ARCH=$ARCH: only w64 is wired (the w48 preset is not written)"; return 2; }
  [ "$EPOCHS" -ge 2 ] && [ "$EPOCHS" -le 6 ] || { log "EPOCHS must be 2..6 (AdamW reference labels exist up to e6)"; return 2; }
  [ "$CHUNK" = 2048 ] || log "WARNING: CHUNK=$CHUNK, the AdamW reference labels are 2,048-review chunks (the comparison is then confounded)"
  case "$OPTIMIZER" in expertium-iter53) TAG=muon ;; \{*) TAG=muonx ;; *) log "unknown OPTIMIZER $OPTIMIZER"; return 2 ;; esac
  SRC=scale-w64kd-$TIER-cached-$RUN_TAG
  SRC_EXP=$WT/artifacts/accuracy/experiments/$SRC
  SRC_LBL=s-w64kd-$TIER-$RUN_TAG
  NAME=scale-${ARCH}kd-$TIER-cached-ctx${CHUNK}-$TAG
  EXP=$WT/artifacts/accuracy/experiments/$NAME
  LBL=s-${ARCH}kd-$TIER-ctx${CHUNK}-$TAG
  [ "$NAME" != "$SRC" ] || { log "new experiment equals the source"; return 2; }
  EP() { echo "$EXP/run/epoch-$(printf "%02d" "$1")/$CKPT"; }
  SRCEP() { echo "$SRC_EXP/run/epoch-$(printf "%02d" "$1")/$CKPT"; }
  [ -f "$SRC_EXP/experiment.json" ] || { log "no source experiment $SRC_EXP"; return 2; }
  if [ "$DRYRUN" = 1 ]; then CHECKNAME=dryrun-$NAME; trap 'rm -rf "$WT/artifacts/accuracy/experiments/dryrun-$NAME"' EXIT
  else CHECKNAME=$NAME; fi

  # ---- step 0: preconditions (read-only on the source)
  if ps -ef 2>/dev/null | grep -v grep | grep -E "runbook-2026-10-16\.sh|runbook-2026-10-16b\.sh|runbook-next\.sh" | grep -q .; then
    need "runbook-2026-10-16.sh, -16b.sh (Q4/Q7 stages) or runbook-next.sh is still running" || return 2
  fi
  for n in $(seq 1 "$EPOCHS"); do
    [ -f "$(SRCEP "$n")" ] || { need "no source checkpoint $(SRCEP "$n")" || return 2; }
  done

  # ---- step 1: CPU preflight (Muon grouping, optimizer state round trip, KD cache key, allowlist)
  OPTS=$(SRC_EXP="$SRC_EXP" OPTIMIZER="$OPTIMIZER" CHUNK="$CHUNK" BATCH_CHUNKS="$BATCH_CHUNKS" ARCH="$ARCH" "$PY" -X utf8 -c "
import json, os
o = json.load(open(os.environ['SRC_EXP'] + '/experiment.json'))['option_overrides']
spec = os.environ['OPTIMIZER']
o['optimizer'] = json.loads(spec) if spec.startswith('{') else spec
o['architecture'] = os.environ['ARCH']
o['chunk_reviews'], o['batch_chunks'] = int(os.environ['CHUNK']), int(os.environ['BATCH_CHUNKS'])
print(json.dumps(o))") || return 2
  log "options: $OPTS"
  WTR="$WT" SRC_EXP="$SRC_EXP" OPTS="$OPTS" ARCH="$ARCH" "$PY" -X utf8 - <<'PYEOF' || { log "preflight REFUSED"; return 2; }
import hashlib, io, json, os, re, sys
from pathlib import Path
sys.path.insert(0, os.environ["WTR"])
import torch
from scripts.expertium_options import PEAK_LR, WIDTH32_ONLY, make_optimizer
from expertium.variants import variant
from expertium.model.variant_model import VariantSrsRWKV
opts = json.loads(os.environ["OPTS"])
src = json.loads((Path(os.environ["SRC_EXP"]) / "run" / "run.json").read_text())["options"]
full = {**src, **opts}   # what prepare stores: split base options + overrides
fail = []
# a. grouping: Muon on the 2-D matrices (+LoRA group), AdamW on the rest; effective Muon LR = lr x lr_scale
master = VariantSrsRWKV(variant(os.environ["ARCH"])).float()
opt = make_optimizer(opts["optimizer"], master)
total = sum(p.numel() for p in master.parameters())
muon_groups, adamw_groups = opt.muon.param_groups, opt.adamw.param_groups
mu = sum(p.numel() for g in muon_groups for p in g["params"])
lora = sum(p.numel() for p in muon_groups[-1]["params"]) if len(muon_groups) > 4 else 0
ad = sum(p.numel() for g in adamw_groups for p in g["params"])
print(f"[preflight] {os.environ['ARCH']}: {total:,} parameters; Muon {mu:,} (LoRA group {lora:,}), AdamW {ad:,}; "
      f"{type(opt.muon).__name__}/{type(opt.adamw).__name__}; Muon groups {len(muon_groups)}, AdamW groups {len(adamw_groups)}")
if type(opt).__name__ != "HybridOptimizer" or not isinstance(opt.muon, torch.optim.Muon) or mu + ad != total or mu == 0:
    fail.append("optimizer is not a Muon+AdamW hybrid covering all parameters")
if any(p.dim() != 2 for g in muon_groups for p in g["params"]):
    fail.append("Muon group holds a non-2-D parameter")
for g in muon_groups:
    print(f"[preflight]   Muon group: {sum(p.numel() for p in g['params']):,} params, lr {g['lr']:.1e} x lr_scale {g['lr_scale']:.4f} = "
          f"{g['lr'] * g['lr_scale']:.4f}, weight_decay {g['weight_decay']:.5f}")
    if abs(g["lr"] - PEAK_LR) > 1e-12 or abs(g["lr"] * g["lr_scale"] - 0.0025) > 1e-9:
        fail.append(f"Muon group LR {g['lr']} x {g['lr_scale']} != 7e-4 x 0.0025/7e-4")
if any(abs(g["lr"] - PEAK_LR) > 1e-12 for g in adamw_groups):
    fail.append("AdamW group LR is not the 7e-4 peak")
# b. optimizer state round trip (what the epoch resume and the decay do with optimizer.pth)
for p in master.parameters():
    p.grad = torch.randn_like(p) * 1e-2
opt.step()
buf = io.BytesIO(); torch.save(opt.state_dict(), buf); buf.seek(0)
state = torch.load(buf, map_location="cpu", weights_only=True)
other = make_optimizer(opts["optimizer"], VariantSrsRWKV(variant(os.environ["ARCH"])).float())
other.load_state_dict(state)
if set(state) != {"adamw", "muon"}:
    fail.append(f"optimizer state keys {sorted(state)}")
print("[preflight] optimizer state_dict save/load round trip OK (keys adamw, muon)")
# c. KD cache key (scripts/expertium_options.py Teacher): sha256 of the options except 'distillation' and width-32-only keys
def key(o):
    b = {k: v for k, v in o.items() if k != "distillation" and k not in WIDTH32_ONLY}
    return hashlib.sha256(json.dumps(b, sort_keys=True).encode()).hexdigest()[:12]
cache = Path(opts["distillation"]["cache"])
ks, kn = key(src), key(full)
print(f"[preflight] KD cache key: source run batches-{ks} ({'exists' if (cache / ('batches-' + ks)).is_dir() else 'MISSING'}), "
      f"this run batches-{kn} ({'exists, warm' if (cache / ('batches-' + kn)).is_dir() else 'does not exist yet: epoch 1 computes the teacher targets online (cold)'}); "
      f"'optimizer' is part of the key: {ks != kn}")
if not (cache / f"batches-{ks}").is_dir():
    fail.append("source KD cache folder missing")
# d. every option key is accepted by the prefetch allowlist
text = (Path(os.environ["WTR"]) / "scripts" / "gpu_r01_prefetch.py").read_text(encoding="utf-8")
allow = set(re.findall(r'"([a-z_0-9]+)"', text[text.index("set(options) - {"):text.index("or bool(options.get(\"preprocessing_cache\"))")]))
extra = sorted(set(full) - allow - set(src))
if extra or "optimizer" not in allow:
    fail.append(f"options outside the gpu_r01_prefetch allowlist: {extra} (optimizer allowed: {'optimizer' in allow})")
print("[preflight] option keys vs prefetch allowlist: " + ("OK (optimizer allowed)" if not extra and "optimizer" in allow else "FAIL"))
# e. only the intended fields differ from the source run
diff = {k for k in set(full) | set(src) if full.get(k) != src.get(k)}
print(f"[preflight] option keys differing from the source run: {sorted(diff)}")
if diff - {"optimizer"}:
    fail.append(f"unintended option differences {sorted(diff - {'optimizer'})}")
if fail:
    print("[preflight] REFUSE:\n  " + "\n  ".join(fail)); sys.exit(1)
PYEOF

  # ---- step 2: the experiment (real: once, relaunch resumes; dry run: dryrun-<NAME>, deleted at exit)
  if [ -f "$EXP/experiment.json" ] && [ "$DRYRUN" != 1 ]; then log "resuming existing $NAME"
  else
    rm -rf "$WT/artifacts/accuracy/experiments/$CHECKNAME"
    "$PY" -X utf8 "$AL" prepare "$CHECKNAME" --tier "$TIER" --seed 2025 --epochs "$EPOCHS" --options "$OPTS" \
      --note "small-model Muon run (fallback 2): w64kd, $TIER tier, $EPOCHS epochs at $CHUNK x $BATCH_CHUNKS, optimizer $OPTIMIZER, from scratch; gated per epoch against $SRC (docs/SMALL_MODEL.md)" \
      > /dev/null || return 2
  fi
  SRC_EXP="$SRC_EXP" NEW="$WT/artifacts/accuracy/experiments/$CHECKNAME" EPOCHS="$EPOCHS" "$PY" -X utf8 - <<'PYEOF' || { [ "$DRYRUN" = 1 ] || log "experiment check failed"; return 2; }
import json, os, sys
from pathlib import Path
s = json.loads((Path(os.environ["SRC_EXP"]) / "experiment.json").read_text())
n = json.loads((Path(os.environ["NEW"]) / "experiment.json").read_text())
r = json.loads((Path(os.environ["NEW"]) / "run" / "run.json").read_text())
fail = []
for k in ("tier", "kind", "seed", "order_seed", "model", "track", "train_users", "eval_users", "split_sha256", "dataset_revision", "dataset_path", "label_db"):
    if s[k] != n[k]:
        fail.append(f"experiment field {k} differs from the source")
if n["epochs"] != int(os.environ["EPOCHS"]) or r["epochs"] != n["epochs"]:
    fail.append(f"epochs {n['epochs']}/{r['epochs']}")
if {k for k in set(s["options"]) | set(n["options"]) if s["options"].get(k) != n["options"].get(k)} != {"optimizer"}:
    fail.append("options differ from the source in more than 'optimizer'")
if n["options"] != r["options"] or n["init"] or n["epoch_runner"] or n["checkpoint"]:
    fail.append("run.json options differ from experiment options, or init/epoch_runner/checkpoint set")
if fail:
    print("[arch-check] REFUSE:\n  " + "\n  ".join(fail)); sys.exit(1)
print(f"[arch-check] OK: {n['name']} = source {s['name']} (tier {n['tier']}, seed {n['seed']}, order_seed {n['order_seed']}, {len(n['eval_users'])} eval users, "
      f"same users/split/KD) + optimizer {json.dumps(n['options']['optimizer'])}; epochs {n['epochs']}")
PYEOF

  # ---- step 3: AdamW reference rows on the 100 gate users (the ctx2048 run has no e3 gate row), one ticket for all missing
  models=""; only=""
  for n in $(seq 1 "$EPOCHS"); do
    rl=$SRC_LBL-e$n
    if rows_ok "$rl/w8+s8"; then log "reference $rl/w8+s8 present"
    else models="$models --model $rl=expertium:w64:$(SRCEP "$n")"; only="$only --only $rl/w8+s8"; log "reference $rl/w8+s8 missing: will be scored"; fi
  done
  if [ -n "$models" ]; then
    # shellcheck disable=SC2086
    q 6 "arch gate: score the missing AdamW references of $SRC on the 100 gate users" -- "$PY" -X utf8 "$SC" \
      --users-file "$GATE_USERS" $models $only --time-limit 900 --out "$OUT" || return 1
  fi

  # ---- step 4: smoke source = a scratch decay of the AdamW epoch 3 with this run's options, without its AdamW optimizer state
  SMOKE=$SM/smoke-prep-$NAME
  rm -rf "$SMOKE"
  "$PY" -X utf8 "$DEC_PY" prepare --source-run "$SRC_EXP/run" --epoch 3 --out "$SMOKE" --fraction 0.01 --evaluate none --seed 1 \
    --extra-options "$OPTS" > /dev/null || return 1
  "$PY" -X utf8 -c "
import json
p = r'$SMOKE/protocol.json'; d = json.load(open(p))
d['full_run'].pop('init_optimizer', None)
json.dump(d, open(p, 'w'), indent=1)
o = json.load(open(r'$SMOKE/decay.json'))['options']
assert o['optimizer'] == json.loads('''$OPTS''')['optimizer'], 'smoke options lack the optimizer'
print('[smoke-prep] smoke options carry optimizer', json.dumps(o['optimizer']), '; AdamW init_optimizer removed (fresh Muon state)')" || return 1
  LAST=$EPOCHS
  for epoch in $(seq 1 "$EPOCHS"); do
    minutes=$EPOCH_MIN; [ "$epoch" -eq 1 ] && minutes=$EPOCH1_MIN
    if [ "$DRYRUN" = 1 ] || [ ! -f "$(EP "$epoch")" ]; then
      q "$minutes" "arch $NAME epoch $epoch/$EPOCHS ($TIER, $OPTIMIZER, chunk $CHUNK x $BATCH_CHUNKS, smoke first)" -- "$BASH_EXE" -c \
        "'$PY' -X utf8 '$DEC_PY' smoke --out '$SMOKE' && '$PY' -X utf8 '$AL' train '$NAME' --epochs-per-job 1"
      rc=$?
      [ $rc -eq 0 ] || [ $rc -eq 3 ] || { log "training failed (exit $rc), stop"; return 1; }
      [ -f "$(EP "$epoch")" ] || { need "no epoch-$epoch checkpoint after training" || return 1; }
    fi
    if [ "$DRYRUN" = 1 ] || ! rows_ok "$LBL-e$epoch/w8+s8"; then
      q 4 "arch gate: score $LBL-e$epoch on the 100 gate users" -- "$PY" -X utf8 "$SC" \
        --users-file "$GATE_USERS" --model "$LBL-e$epoch=expertium:$ARCH:$(EP "$epoch")" --only "$LBL-e$epoch/w8+s8" --time-limit 600 --out "$OUT" || return 1
    fi
    decision=$SM/gate-$NAME-e$epoch.json; [ "$DRYRUN" = 1 ] && decision=$SM/dryrun-gate-$NAME-e$epoch.json
    OUT="$OUT" USERS="$GATE_USERS" A="$SRC_LBL-e$epoch/w8+s8" B="$LBL-e$epoch/w8+s8" DEC="$decision" EPOCH="$epoch" EPOCHS="$EPOCHS" WTR="$WT" \
      "$PY" -X utf8 - <<'PYEOF'
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.environ["WTR"])
from scripts import small_model_gate as g
users = json.loads(Path(os.environ["USERS"]).read_text())["users"]
a, b, epoch, epochs = os.environ["A"], os.environ["B"], int(os.environ["EPOCH"]), int(os.environ["EPOCHS"])
per = g.load(os.environ["OUT"], [a, b], set(users))
missing = {l: len([u for u in users if u not in per[l]]) for l in (a, b)}
if any(missing.values()):
    json.dump({"verdict": "incomplete", "missing_users": missing}, open(os.environ["DEC"], "w"), indent=1)
    print(f"[arch-gate] incomplete rows {missing}"); sys.exit(1)
res = {}
for head, name in g.HEADS.items():
    base, sizes = g.weighted(per, a, users, head)
    cand, _ = g.weighted(per, b, users, head)
    delta = g.rw(cand, sizes) - g.rw(base, sizes)
    upper = g.boot_upper(cand, base, sizes)
    res[name] = {"adamw": g.rw(base, sizes), "muon": g.rw(cand, sizes), "delta": delta, "upper95": upper,
                 "pct_vs_adamw": delta / g.rw(base, sizes) * 100}
test_a = all(v["delta"] < 0 and v["upper95"] < 0 for v in res.values())
stop = all(v["upper95"] >= 0 for v in res.values()) and epoch < epochs
json.dump({"verdict": "stop" if stop else "go", "epoch": epoch, "test_A_beats_adamw_same_epoch": test_a,
           "rule": "stop (epoch < EPOCHS) if upper95 >= 0 on both heads", "heads": res}, open(os.environ["DEC"], "w"), indent=1)
print(f"[arch-gate] e{epoch}: " + ("STOP" if stop else "GO") + f", test A {'passed' if test_a else 'not passed'} " + json.dumps(res))
sys.exit(10 if stop else 0)
PYEOF
    rc=$?
    if [ $rc -eq 10 ]; then
      if [ "${FORCE:-0}" = 1 ]; then log "e$epoch not better than AdamW ($decision), FORCE=1: continuing"
      else log "e$epoch: Muon not better than AdamW on either head ($decision): skipping the remaining epochs, decay from e$epoch"; LAST=$epoch; break; fi
    elif [ $rc -ne 0 ]; then log "arch gate failed (exit $rc)"; [ "$DRYRUN" = 1 ] && log "DRYRUN: no rows, continuing" || return 1
    fi
  done
  EN=$(EP "$LAST")

  # ---- step 5: 0.25-epoch decay from the last epoch (the run options carry the optimizer; the epoch's Muon state is its init_optimizer)
  DECF=$SM/$NAME-decay
  D=$DECF/$CKPT
  if [ "$DRYRUN" = 1 ]; then
    DCHK=$SM/arch-check-decay-$NAME; rm -rf "$DCHK"
    "$PY" -X utf8 "$DEC_PY" prepare --source-run "$SRC_EXP/run" --epoch "$EPOCHS" --out "$DCHK" --fraction 0.25 --shape one_minus_sin \
      --evaluate none --seed 1 --extra-options "$OPTS" > /dev/null || { log "DRYRUN: decay prepare check failed"; rm -rf "$DCHK"; return 1; }
    "$PY" -c "
import json; d = json.load(open(r'$DCHK/decay.json'))['options']
print('[arch-check] decay prepare options: optimizer', json.dumps(d['optimizer']), 'chunk_reviews', d['chunk_reviews'], 'batch_chunks', d['batch_chunks'])
assert d['chunk_reviews'] == $CHUNK and d['batch_chunks'] == $BATCH_CHUNKS and 'optimizer' in d" || { rm -rf "$DCHK"; return 1; }
    rm -rf "$DCHK"
    log "DRYRUN: would prepare $DECF (--source-run $EXP/run --epoch $LAST --fraction 0.25 --shape one_minus_sin --evaluate none --seed 1)"; ok=1
  elif [ -d "$DECF" ]; then ok=1
  else "$PY" -X utf8 "$DEC_PY" prepare --source-run "$EXP/run" --epoch "$LAST" --out "$DECF" --fraction 0.25 --shape one_minus_sin \
      --evaluate none --seed 1 && ok=1 || ok=0; fi
  [ "$ok" = 1 ] || { log "decay prepare failed"; return 1; }
  [ -f "$D" ] || q 30 "arch $NAME: 0.25-epoch decay from e$LAST (smoke first)" -- "$BASH_EXE" -c \
      "'$PY' -X utf8 '$DEC_PY' smoke --out '$DECF' && '$PY' -X utf8 '$DEC_PY' run --out '$DECF'" || return 1

  # ---- step 6: 464-user scoring and a resume ticket only if users remain
  MODELS="--model $LBL-e$LAST=expertium:$ARCH:$EN --model $LBL-dec$LAST=expertium:$ARCH:$D"
  LABELS="--only $LBL-e$LAST/w8+s8 --only $LBL-dec$LAST/w8+s8"
  # shellcheck disable=SC2086
  q 30 "arch $NAME: full-history scoring of e$LAST and dec$LAST (464 users)" -- "$PY" -X utf8 "$SC" $MODELS $LABELS --time-limit 3000 --out "$OUT" || return 1
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
    q 15 "arch $NAME: scoring resume (only if users remain)" -- "$PY" -X utf8 "$SC" $MODELS $LABELS --time-limit 1200 --out "$OUT"
  fi
  [ "$DRYRUN" = 1 ] && rm -rf "$SMOKE" "$SM"/dryrun-gate-"$NAME"-e*.json
  return 0
}

cd "$WT" || exit 1
case "$STAGE" in
arch) arch; rc=$?; [ $rc -eq 0 ] && log "done: $STAGE" || log "$STAGE ended with exit $rc"; exit $rc ;;
*) echo "unknown stage $STAGE (arch)"; exit 2 ;;
esac
