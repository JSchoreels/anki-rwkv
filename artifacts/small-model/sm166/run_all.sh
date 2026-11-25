#!/usr/bin/env bash
# Weekend RWKV family release (SM166): rebuild every asset from scratch, CPU only. Publishes nothing:
# no gh, no tag, no upload. Run from the repository root:  bash artifacts/small-model/sm166/run_all.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/Scripts/python.exe"
OUT="$ROOT/work/release-2026-11-28"            # never committed (work/.gitignore = *)
SCRATCH="${SCRATCH:-${TMPDIR:-/tmp}/sm166}"     # synthetic users + clean extracts
FORK="C:/Users/jonat/workspace/anki"; FC=eda153ddbff7847de8bf2591fe5e0935a8dfe836; INF=qt/aqt/rwkv_inference
WT="C:/Users/jonat/workspace/wt"
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 RWKV_DELETED_CARDS=neighbour

mkdir -p "$OUT/fork-main" "$SCRATCH"; printf '*\n' > "$ROOT/work/.gitignore"
# 1. fork main files (git objects, read only)
for m in sm118 sm113 sm130th sm130io sm99; do
  mkdir -p "$OUT/fork-main/$m"
  for f in time_head.f32.bin time_head.int8.bin time_head.json; do git -C "$FORK" cat-file blob "$FC:$INF/$m/$f" > "$OUT/fork-main/$m/$f"; done
  git -C "$FORK" cat-file blob "$FC:rslib/src/rwkv/testdata/$m-time-vectors.json.gz" > "$OUT/fork-main/$m/fixture-time-vectors.json.gz"
done
mkdir -p "$OUT/fork-main/sm91s2"
for e in f32.bin int8.bin json; do git -C "$FORK" cat-file blob "$FC:$INF/RWKV_trained_on_5000_10000.time_head.$e" > "$OUT/fork-main/sm91s2/time_head.$e"; done
git -C "$FORK" cat-file blob "$FC:rslib/src/rwkv/testdata/legacy-time-vectors.json.gz" > "$OUT/fork-main/sm91s2/fixture-time-vectors.json.gz"
# 2. synthetic users (fixed seed, no dataset data)
rm -rf "$SCRATCH/synth"; "$PY" -X utf8 scripts/synthetic_testvector_users.py "$SCRATCH/synth" > /dev/null
# 3. answer-time vectors (PyTorch on the CPU; model code read from wt/fork-rc-export)
declare -A P=([sm91s2]=$WT/big-package/artifacts/models/sm91s2 [sm99]=$WT/big-package/artifacts/models/sm99
              [sm118]=$WT/fork-rc-export/artifacts/models/sm118 [sm113]=$WT/time-heads-package/artifacts/models/sm113
              [sm130th]=$WT/fork-rc-export/artifacts/models/sm130th [sm130io]=$WT/fork-rc-export/artifacts/models/sm130io)
for k in sm91s2 sm99 sm118 sm113 sm130th sm130io; do
  "$PY" -X utf8 artifacts/small-model/sm166/time_vectors.py "$k" --package "${P[$k]}" --head "$OUT/fork-main/$k" \
    --fixture "$OUT/fork-main/$k/fixture-time-vectors.json.gz" --data "$SCRATCH/synth" --out "$OUT/time-vectors/$k"
done 2>&1 | grep sm166 | tee artifacts/small-model/sm166/logs/time-vectors.log
# 4. packages, zips, INTEGRATION.md, WEIGHTS-NOTICE.md, SHA256SUMS (+ release-manifest.json, SHA256SUMS copy here)
"$PY" -X utf8 -I artifacts/small-model/sm166/build_release.py --out "$OUT" | tee artifacts/small-model/sm166/logs/build.log
# 5. clean-extract verification
rm -rf "$SCRATCH/verify"; mkdir -p "$SCRATCH/verify"
"$PY" -X utf8 -I artifacts/small-model/sm166/verify_release.py --dist "$OUT/dist" --scratch "$SCRATCH/verify"
# Publishing (NOT run here; only after the user's explicit OK, by the main session):
# gh release create rwkv-family-2026-11-28 --repo JSchoreels/anki-rwkv --title "..." \
#   --notes-file artifacts/small-model/sm166/RELEASE_NOTES.md "$OUT"/dist/*
