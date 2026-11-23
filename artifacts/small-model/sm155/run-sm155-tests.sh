#!/usr/bin/env bash
# Bounded-memory long-card scorer (SM155): the pre-registered exactness matrix (design.md), memory and speed probes.
# CPU only, 2 lanes (2 scorer processes at a time), memory guard 5.9 GB per CPU process. Each cell runs the stock
# engine and then --engine stream with the same command; dumps in work/sm155/ (never committed).
# Launch (hidden): see HANDOFF.md. Ends with $C/done (or $C/failed); summary in sm155-exactness.{json,txt}.
set -u
WT=C:/Users/jonat/workspace/wt/sm155
S=$WT/artifacts/small-model/sm155
C=$S/cpu
WK=$WT/work/sm155
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
SC=$WT/scripts/small_model_heldout.py
mkdir -p "$C" "$WK"
rm -f "$C/done" "$C/failed"
LOG=$C/run.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
source "$WT/artifacts/small-model/sm148/models.sh"   # MODELS: TAG|KIND:ARCH|PATH|SHA256|ROLE|SIZE
FLAGS=(--inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets 0,1,3,5,9,16,32,128,300,1000)
MAIN="w8,w8+s8hx8,w8+lr2.16.4q8x8,w8+lr2q8x8"
FAM="fp32,w4,w8+s8,w8+s4c8n,w8+s4hgx4,w8+t4.128.8c8nhx8,w8+lr4q8x8,w8+lr2.16.4q6x4,w8+lrRq8x8,w8+lr4fx0,w8+lr4"
echo '{"users": [2388, 4676, 1866]}' > "$C/users-smoke.json"
echo '{"users": [2883, 3882, 1215, 280, 4407]}' > "$C/users-long.json"
echo '{"users": [4407]}' > "$C/users-4407.json"
echo '{"users": [1458]}' > "$C/users-1458.json"

model_arg() {  # model_arg TAG -> TAG=KIND:ARCH:PATH
  echo "$MODELS" | awk -F'|' -v t="$1" '$1 == t {print $1 "=" $2 ":" $3}'
}

cell() {  # cell NAME TAG LABELS USERSFILE CHUNK [extra args...]: stock then stream into $WK/NAME
  local name=$1 tag=$2 labels=$3 users=$4 chunk=$5; shift 5
  local d=$WK/$name eng t0 rc
  mkdir -p "$d"
  for eng in parallel stream; do
    [ -f "$d/$eng.ok" ] && continue
    t0=$(date +%s)
    env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS CUDA_VISIBLE_DEVICES= PYTORCH_JIT=0 \
      "$PY" -X utf8 -W ignore "$SC" --device cpu --pool 732 --users-file "$users" --model "$(model_arg "$tag")" \
      --variants "$labels" "${FLAGS[@]}" --time-limit 1000000 --out "$d/$eng.jsonl" --dump "$d/dump-$eng" \
      --dump-scope all --engine "$eng" --chunk-rows "$chunk" "$@" > "$d/$eng.out" 2>&1
    rc=$?
    if [ $rc = 0 ]; then touch "$d/$eng.ok"; else echo "$name $eng exit $rc" >> "$C/cell-failures.txt"; fi
    log "$name $eng exit $rc in $(( $(date +%s) - t0 )) s"
  done
}

lane_a() {   # Big models (the slowest), then the default chunk size and the memory probe
  local tag
  for tag in sm91s2-big-showtime sm99-big-inst-e2; do
    cell "main-$tag-smoke" "$tag" "$MAIN" "$C/users-smoke.json" 2048
    cell "main-$tag-long" "$tag" "$MAIN" "$C/users-long.json" 2048
  done
  cell "chunk16k-sm91s2" sm91s2-big-showtime "w8+s8hx8,w8+lr2.16.4q8x8" "$C/users-4407.json" 16384
  local d=$WK/mem-1458 eng
  mkdir -p "$d"
  for eng in parallel stream; do
    [ -f "$d/$eng.ok" ] && continue
    env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS CUDA_VISIBLE_DEVICES= PYTORCH_JIT=0 \
      "$PY" -X utf8 "$WT/artifacts/small-model/sm136/peakrun.py" "$PY" -X utf8 -W ignore "$SC" --device cpu --pool 732 \
      --users-file "$C/users-1458.json" --model "$(model_arg sm91s2-big-showtime)" --variants w8+s8hx8 "${FLAGS[@]}" \
      --time-limit 1000000 --out "$d/$eng.jsonl" --dump "$d/dump-$eng" --dump-scope all --engine "$eng" \
      --chunk-rows 16384 > "$d/$eng.out" 2>&1 && touch "$d/$eng.ok"
    log "mem-1458 $eng: $(grep -a PEAK "$d/$eng.out" | cut -c1-60)"
  done
}

lane_b() {   # Small and Tiny models, then the format families
  local tag
  for tag in sm118-e26-dec2ep32k-t91s2 sm113-inst-e35st-st-dec2ep32k-t99 sm130-th-dec2ep32k-t91s2 sm130-io-dec2ep32k-t99; do
    cell "main-$tag-smoke" "$tag" "$MAIN" "$C/users-smoke.json" 2048
    cell "main-$tag-long" "$tag" "$MAIN" "$C/users-long.json" 2048
  done
  cell "fam-sm91s2" sm91s2-big-showtime "$FAM" "$C/users-smoke.json" 1024
  cell "fam-sm118" sm118-e26-dec2ep32k-t91s2 "$FAM" "$C/users-smoke.json" 1024
  cell "fam-sm118-vq" sm118-e26-dec2ep32k-t91s2 "w8+vq403x214" "$C/users-smoke.json" 1024 \
    --vq-codebooks artifacts/small-model/state-vq/codebooks-dec10.npz
}

log "start (lanes a and b, 2 CPU processes)"
wp=$(cat /proc/$$/winpid 2>/dev/null || echo 0)
rm -f "$C/memguard.off"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$S/memguard.ps1")" -Dir "$(cygpath -w "$C")" \
  -ParentPid "$wp" -Match "small_model_heldout.py&--device cpu&sm155" > /dev/null 2>&1 &
MG=$!
lane_a & A=$!
lane_b & B=$!
wait $A $B
touch "$C/memguard.off"; sleep 12; kill $MG 2>/dev/null
"$PY" -X utf8 "$S/sm155_compare.py" $(ls -d "$WK"/main-* "$WK"/fam-* "$WK"/chunk16k-* "$WK"/mem-* 2>/dev/null) \
  --json "$S/sm155-exactness.json" > "$S/sm155-exactness.txt" 2>&1
log "compare: $(tail -1 "$S/sm155-exactness.txt")"
if [ -s "$C/cell-failures.txt" ] || [ -e "$C/memguard-kill" ]; then touch "$C/failed"; log "done with failures"; else touch "$C/done"; log "done"; fi
