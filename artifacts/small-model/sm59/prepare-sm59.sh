#!/usr/bin/env bash
# SM59 (2026-10-22): prepare a fine-tune (both heads trained) from SM36 with another teacher weight (CPU only, seconds).
#   DECAY_ALPHA=0.5|0.9|1.0 FRACTION=0.22 NAME=kd-control bash prepare-sm59.sh
# Same as SM50's prepare (copy) but no "head_only" and decay_alpha (the weight used while the LR decays) = DECAY_ALPHA; users = a fresh shuffle
# (seed 2) of the same 5,000 training users (v2 run), until FRACTION of an epoch's reviews; SM36's master.pth,
# optimizer.pth (Muon/AdamW state continued) and update offset; LR peak 1e-4 -> 0, one_minus_sin, over those reviews.
set -eu
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm59
SM36_DIR=C:/Users/jonat/workspace/wt/sm42/artifacts/small-model/scale-w64kd-gate-sc-lanes-cont-decay1ep32k-muon
SM36_CASE=$SM36_DIR/expertium/E89-F65e-F77a
V2_RUN=C:/Users/jonat/workspace/wt/sm21v2/artifacts/accuracy/experiments/scale-w64kd-gate-sc-lanes-cont/run
OUT=$WT/artifacts/small-model/sm59/${NAME:?}
DIST="{\"teacher\": \"C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth\", \"alpha\": 0.9, \"decay_alpha\": ${DECAY_ALPHA:?}, \"kd_source\": \"replay\", \"replay\": \"C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/fork-fp32-gate5k\"}"
EXTRA="{\"distillation\": $DIST, \"state_carry\": {\"lanes\": \"strided\"}, \"chunk_reviews\": 32768, \"batch_chunks\": 1, \"optimizer\": \"expertium-iter53\"}"
cd "$WT"
"$PY" -X utf8 scripts/expertium_decay.py prepare --source-run "$V2_RUN" --epoch 12 --init-case "$SM36_CASE" --peak 1e-4 \
  --out "artifacts/small-model/sm59/$NAME" --fraction "${FRACTION:?}" --shape one_minus_sin --evaluate none --seed 2 --extra-options "$EXTRA"
A="$SM36_DIR" B="$OUT" DECAY_ALPHA="$DECAY_ALPHA" SM36_CASE="$SM36_CASE" "$PY" -X utf8 - <<'PYEOF'
import json, os
from pathlib import Path
a_dir, b_dir = Path(os.environ["A"]), Path(os.environ["B"])
a = json.loads((a_dir / "decay.json").read_text()); b = json.loads((b_dir / "decay.json").read_text())
pa = json.loads((a_dir / "protocol.json").read_text()); pb = json.loads((b_dir / "protocol.json").read_text())
fail = []
oa, ob = dict(a["options"]), dict(b["options"])
da = ob["distillation"].pop("decay_alpha"); oa["distillation"] = dict(oa["distillation"]); oa["distillation"].pop("decay_alpha")
if da != float(os.environ["DECAY_ALPHA"]):
    fail.append(f"decay_alpha {da}")
if "head_only" in ob:
    fail.append("head_only present")
if oa != ob:
    fail.append(f"options differ from SM36 beyond decay_alpha: { {k: (oa.get(k), ob.get(k)) for k in set(oa) | set(ob) if oa.get(k) != ob.get(k)} }")
for k in ("training", "dataset_path", "dataset_revision", "execution_mode"):
    if pa[k] != pb[k]:
        fail.append(f"protocol {k} differs from SM36")
fr = pb["full_run"]; case = Path(os.environ["SM36_CASE"])
if Path(fr["init_master"]).resolve() != (case / "master.pth").resolve() or Path(fr["init_optimizer"]).resolve() != (case / "optimizer.pth").resolve():
    fail.append("does not start from SM36's master/optimizer")
res = json.loads((case / "result.json").read_text())
if fr["update_offset"] != res["update_offset_end"]:
    fail.append("update offset is not SM36's end")
if fr["lr_schedule"] != {"decay_start": 0.0, "decay_end": 1.0, "shape": "one_minus_sin", "final": 0.0, "peak": 1e-4}:
    fail.append(f"schedule {fr['lr_schedule']}")
users_a = {r["user_id"] for r in pa["rows"]}; users_b = [r["user_id"] for r in pb["rows"]]
if not set(users_b) <= users_a:
    fail.append("users outside SM36's training users")
if fail:
    print("[sm59-prepare] REFUSE:\n  " + "\n  ".join(fail)); raise SystemExit(1)
print(f"[sm59-prepare] OK {b_dir.name}: options = SM36's, decay_alpha {da}; {b['training_users']} users, "
      f"{b['training_reviews']:,} reviews ({b['training_reviews'] / a['training_reviews']:.3f} of SM36's epoch), {b['lr']}, "
      f"from SM36 (update offset {fr['update_offset']}, optimizer state continued)")
PYEOF
