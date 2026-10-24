#!/bin/bash
# SM70: features + probes on the dumped users (CPU). Outputs in artifacts/small-model/sm70/.
cd /c/Users/jonat/workspace/wt/sm70
P=/c/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
O=artifacts/small-model/sm70
FORK=C:/Users/jonat/workspace/wt/small-model/artifacts/small-model/sm24-dump
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
set -e
$P scripts/sm70_inventory.py --users @$O/dump --out $O/inventory.json
$P scripts/sm70_features.py --dump $O/dump --fork-dump $FORK --out $O/features.parquet --workers 6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $P scripts/sm70_within.py --features $O/features.parquet --model sm56 --out $O/within-sm56.json --workers 6 > $O/within-sm56.log 2>&1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $P scripts/sm70_within.py --features $O/features.parquet --model fork --out $O/within-fork.json --workers 6 > $O/within-fork.log 2>&1
$P scripts/sm70_probe.py --features $O/features.parquet --model sm56 --bins 6 --fit-weight user --out $O/pooled-sm56.json > $O/pooled-sm56.log 2>&1
$P scripts/sm70_probe.py --features $O/features.parquet --model sm56 --bins 6 --fit-weight user --no-interact --no-tree --out $O/pooled-sm56-shift.json > $O/pooled-sm56-shift.log 2>&1
$P scripts/sm70_probe.py --features $O/features.parquet --model fork --bins 6 --fit-weight user --no-tree --out $O/pooled-fork.json > $O/pooled-fork.log 2>&1
$P scripts/sm70_gap.py --features $O/features.parquet --out $O/gap.json > $O/gap.log 2>&1
echo "[sm70 probes] done"
