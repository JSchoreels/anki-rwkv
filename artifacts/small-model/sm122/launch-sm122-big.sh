#!/usr/bin/env bash
# Retrieval-effort probe (SM122): the CPU check of all three models (-> check-cpu.json), then, only if every proof
# passes, the Big pass (MODELS=sm91s2 launch-sm122.sh, which also reruns the report at its end).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm122
S=$WT/artifacts/small-model/sm122
R=$S/run
mkdir -p "$R"
LOG=$R/launch-sm122.log
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=-1
log() { echo "[sm122 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
cd "$WT" || exit 1
log "check start (sm92, sm106, sm91s2)"
"$PY" -X utf8 "$WT/scripts/sm122_effort.py" check --models sm92,sm106,sm91s2 > "$R/check.out" 2>&1; rc=$?
log "check exit $rc"
[ "$rc" = 0 ] || { log "STOP: check failed, Big not started"; exit 1; }
MODELS=sm91s2 bash "$S/launch-sm122.sh"
