#!/usr/bin/env bash
# Output-aware low-rank truncation (SM157): GPU follow-up on the 732 set (design.md "GPU follow-up"). PREPARED, NOT
# STARTED: launch only after the CPU pointer passed (sm157-result.json verdict.gpu_recommended) and the main session
# agreed. A narrow copy of the GPU chain of the rank-1 int8 states test (SM150)'s launcher; no CPU long-card chain.
#   0 preconditions  files, clean source, report selftest, the pointer verdict (KIND = verdict.gpu_kind unless KIND is
#                    set in the environment; FORCE=1 runs even without a pass)
#   1 CPU smoke rows the 3 smoke users x 6 models x Q2<KIND>, T24<KIND> on the CPU (if missing) -> cpu/smoke-w-cpu-rows.jsonl
#   2 GPU smoke      ticket: 3 smoke users x 6 models x REF, Q2<KIND>, T24<KIND> -> gpu/smoke-gpu-rows.jsonl;
#                    sm157_report.py smoke vs the CPU rows (REF 1e-5; low-rank labels SM150's discrete-codec rule)
#   3 scoring        one ticket per size (Tiny, Small, Big), the 732 set (USERS732, 723 users; Big 718), the 2 models x
#                    Q2<KIND>, T24<KIND>, --prior-buckets; passes at cap 0.8, then 0.92 after a pass without progress
#   REF, Q2 and T24 rows on the 732 set are SM148's (wt/sm148/artifacts/small-model/sm148/gpu/sm148-rows-732.jsonl);
#   a fresh agent writes the 732 read-out against them (11-23 budgets) when the chain is done.
#   All scoring: show-time inputs (e2s-query), leak-free (--deleted-cards neighbour), --nonfinite-rows, --split-existing,
#   SM157_STATS = sm157-stats.json (the CPU pointer's fit; its sha256 is in every weighted row).
# ETA: about 3.5 to 6 GPU hours. SM150's 2-label passes took Tiny 50, Small 75, Big 130 minutes with the fused CUDA
#   kernel; the weighted labels run the torch step loop on the GPU (the fused kernel is for plain labels only), not yet
#   timed; the smoke ticket measures it (gpu/score.out review-evals/s).
# Breaks: quiet.json windows only (scope all / gpu, sm135_block.py).
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm157/artifacts/small-model/sm157/launch-sm157-gpu.sh
#   launch   WMI hidden (HANDOFF.md); log artifacts/small-model/sm157/gpu/launch-sm157-gpu.log
#   resume   remove gpu/STOP, then the launch command again (finished users and gpu/smoke-ok are skipped)
#   stop     touch C:/Users/jonat/workspace/wt/sm157/artifacts/small-model/sm157/gpu/STOP
# Do not edit scripts/ expertium/ rwkv/ of the sm157 worktree while it runs.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm157
SM=$WT/artifacts/small-model
S=$SM/sm157
G=$S/gpu
C=$S/cpu
mkdir -p "$G" "$C"
LOG=$G/launch-sm157-gpu.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm157-gpu-dryrun.log
SC=$WT/scripts/small_model_heldout.py
RULE=$S/sm157_report.py
BLOCK=$SM/sm135/sm135_block.py
. "$S/models.sh"; . "$S/labels.sh"
USERS732=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json
USMOKE=$S/users-smoke.json
STATS=$S/sm157-stats.json
CPUSMOKE0=$C/smoke-cpu-rows.jsonl          # REF, Q2, T24 (cpu-proof.sh)
CPUSMOKEW=$C/smoke-w-cpu-rows.jsonl        # the weighted labels (step 1)
CPUSMOKE=$C/smoke-all-cpu-rows.jsonl
GPUSMOKE=$G/smoke-gpu-rows.jsonl
ROWS732=$G/sm157-rows-732.jsonl
TICKETS=$G/tickets.jsonl
STOPF=$G/STOP
NAME="Output-aware low-rank truncation (SM157)"
declare -A MIN=([tiny]=70 [small]=105 [big]=180)   # pass 1 per size, 2 labels (SM150: 50 / 75 / 130 min with the fused kernel)
declare -A WANT=([tiny]=723 [small]=723 [big]=718)  # Big: the 5 largest users never fit the scorer (SM148)
MIN_SMOKE=25; MIN_RETRY=40
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START SM11_INPUTS
log() { echo "[sm157 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$STOPF" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }
free_mib() { awk '/MemFree/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 99999; }
block() { "$PY" -X utf8 "$BLOCK" | tr -d '\r'; }
jline() { "$PY" -X utf8 -c "import json, sys; print(json.dumps(dict(zip(sys.argv[1::2], [int(v) if v.lstrip('-').isdigit() else v for v in sys.argv[2::2]]))))" "$@" | tr -d '\r'; }

wait_free() {  # wait_free MINUTES WHAT: returns once MINUTES fit before the next quiet window (FREE = minutes to it)
  local n=0 b
  while true; do
    stopped && return 1
    b=$(block)
    case "$b" in
      NOW*) local s=${b#NOW }
            if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $2 would wait inside a quiet window (${s}s left)"; FREE=100000; return 0; fi
            [ $((n % 6)) = 0 ] && log "$2: quiet window, ${s}s left"; n=$((n + 1)); nap "$s";;
      NEXT*) FREE=${b#NEXT }
             [ "$FREE" -ge $(( $1 + 3 )) ] && return 0
             if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $2 ($1 min) would wait: $FREE min to the next quiet window"; FREE=100000; return 0; fi
             [ $((n % 6)) = 0 ] && log "$2 needs $1 min, $FREE min before the next quiet window; waiting"; n=$((n + 1)); nap $(( FREE * 60 + 60 ));;
      *) log "window check failed: $b"; nap 600;;
    esac
  done
}

ram_gate() {  # ram_gate WHAT: hold while < 3 GB RAM is free, at most 30 min, then go on
  local i=0
  while [ "$DRYRUN" != 1 ] && [ "$(free_mib)" -lt 3072 ]; do
    if [ $i -ge 6 ]; then log "$1: still $(free_mib) MiB RAM free after 30 min; going on"; return 0; fi
    [ $i = 0 ] && log "$1: $(free_mib) MiB RAM free (< 3 GB), holding"; i=$((i + 1)); sleep 300
  done
  return 0
}

q() {  # q MINUTES KIND PURPOSE -- CMD...   one ticket that must end before the next quiet window
  local minutes=$1 kind=$2 purpose=$3 rc t0 mx base wpid; shift 4
  while true; do
    wait_free "$minutes" "$purpose" || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $NAME: $purpose (expected $minutes min)"; echo "    cmd: $*" >> "$LOG"; return 0; fi
    local maxwait=$(( FREE - minutes - 3 )); [ "$maxwait" -lt 1 ] && maxwait=1
    log "enqueue: $purpose (expected $minutes min, max wait $maxwait min, $FREE min to the next quiet window)"
    t0=$(date +%s); base=$(gpu_used); rm -f "$G/.gpu-max"
    ( m=0; while true; do u=$(gpu_used); [ -n "$u" ] && [ "$u" -gt "$m" ] && { m=$u; echo "$m" > "$G/.gpu-max"; }; sleep 30; done ) &
    wpid=$!
    "$PY" "$QUEUE" run --owner sm157 --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm157/launch-sm157-gpu.sh" -- "$@"
    rc=$?; kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null; mx=$(cat "$G/.gpu-max" 2>/dev/null | tr -dc '0-9'); mx=${mx:-0}
    log "exit $rc: $purpose (nvidia-smi max $mx MiB, baseline $base MiB)"
    [ "$rc" = 75 ] && { log "queue wait would reach the next quiet window; re-planning"; sleep 60; continue; }
    jline kind "$kind" purpose "$purpose" exit "$rc" minutes "$(( ($(date +%s) - t0 + 30) / 60 ))" nvidia_smi_max_mib "$mx" nvidia_smi_baseline_mib "${base:-0}" >> "$TICKETS"
    return $rc
  done
}

min_users() {  # min_users ROWS LABEL...: users with a row (benchmark, or a 'nonfinite' row) for every label (the minimum)
  F="$1" "$PY" -X utf8 -c "
import json, os, sys
from pathlib import Path
p = Path(os.environ['F'])
got = {}
if p.exists():
    for raw in open(p, 'rb'):
        try:
            r = json.loads(raw.decode('utf-8', errors='replace'))
        except ValueError:
            continue
        if 'benchmark' in r or ('nonfinite' in r and 'error' not in r):
            got.setdefault(r.get('label'), set()).add(r['user_id'])
print(min(len(got.get(l, ())) for l in sys.argv[1:]))" "${@:2}" | tr -d '\r'; }

group_args() {  # group_args SIZE "NAMES": GARGS (--model / --only) and GLBLS (labels) of one size (all = every model)
  GARGS=(); GLBLS=()
  local tag ka path digest role size n
  while IFS='|' read -r tag ka path digest role size; do
    [ -z "$tag" ] && continue
    [ "$1" = all ] || [ "$1" = "$size" ] || continue
    GARGS+=(--model "$tag=$ka:$path")
    for n in $2; do GARGS+=(--only "$tag/${!n}"); GLBLS+=("$tag/${!n}@e2s-query"); done
  done <<< "$MODELS"
}

SCORE_FLAGS=(--inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets "$EDGES")

cpu_smoke() {  # the weighted labels on the CPU, 3 smoke users (the GPU smoke's reference), about 10 minutes
  [ -s "$CPUSMOKE" ] && return 0
  group_args all "$WNAMES"
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: CPU smoke rows of $WNAMES -> $CPUSMOKEW"; return 0; fi
  log "CPU smoke rows: 3 smoke users x 6 models x $WNAMES"
  env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS CUDA_VISIBLE_DEVICES= PYTORCH_JIT=0 SM157_STATS="$STATS" \
    "$PY" -X utf8 -W ignore "$SC" --device cpu --workers 1 --pool 732 --users-file "$USMOKE" "${GARGS[@]}" "${SCORE_FLAGS[@]}" \
    --time-limit 100000 --out "$CPUSMOKEW" >> "$C/smoke-w-cpu.out" 2>&1 || { log "STOP: CPU smoke rows failed"; return 1; }
  cat "$CPUSMOKE0" "$CPUSMOKEW" > "$CPUSMOKE"
}

smoke() {
  [ -e "$G/smoke-ok" ] && return 0
  group_args all "REF $WNAMES"
  q $MIN_SMOKE smoke "GPU smoke: 3 smoke users, 6 models x REF / Q2 and T24 weighted ($KIND), vs the CPU rows" -- \
    env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS HELDOUT_GPU_MEMORY_FRACTION=0.8 PYTORCH_JIT=0 SM157_STATS="$STATS" \
    "$PY" -X utf8 -W ignore "$SC" --pool 732 --users-file "$USMOKE" "${GARGS[@]}" "${SCORE_FLAGS[@]}" --time-limit 1800 \
    --out "$GPUSMOKE" >> "$G/score.out" 2>&1 || { log "STOP: smoke ticket failed (see $G/score.out)"; return 1; }
  [ "$DRYRUN" = 1 ] && return 0
  local out; out=$("$PY" -X utf8 "$RULE" smoke --gpu "$GPUSMOKE" --cpu "$CPUSMOKE" --names "REF $WNAMES" 2>&1 | tr -d '\r')
  echo "$out" > "$G/smoke-check.txt"; log "$out"
  case "$out" in *"smoke OK"*) touch "$G/smoke-ok"; return 0;; esac
  log "STOP: GPU smoke mismatch (see $G/smoke-check.txt, $G/smoke-check.json)"; return 1
}

score() {  # score SIZE: one size on the 732 set, passes with the cap rule
  local size=$1 cap=0.8 pass have prev=-1 minutes tl want=${WANT[$1]}
  group_args "$size" "$WNAMES"
  for pass in 1 2 3 4 5; do
    have=$(min_users "$ROWS732" "${GLBLS[@]}")
    [ "$have" -ge "$want" ] && break
    if [ "$have" = "$prev" ]; then
      if [ "$cap" = 0.8 ]; then cap=0.92; log "no progress on $size ($have of $want users): next pass at cap 0.92 (memory only, same numerics)"
      else log "no progress at cap 0.92 on $size ($have of $want users); going on"; break; fi
    fi
    prev=$have
    minutes=${MIN[$size]}; [ "$pass" -gt 1 ] && minutes=$MIN_RETRY
    tl=$(( minutes * 60 * 6 / 5 ))
    ram_gate "scoring $size"
    q $minutes score "scoring $size (2 models x rank 2 and rank 2-then-4, weighted $KIND, int8 factors) on e2s-query inputs, the 732 set (pass $pass, $have done, cap $cap)" -- \
      env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS HELDOUT_GPU_MEMORY_FRACTION=$cap PYTORCH_JIT=0 SM157_STATS="$STATS" \
      "$PY" -X utf8 "$SC" --pool 732 --users-file "$USERS732" "${GARGS[@]}" "${SCORE_FLAGS[@]}" --time-limit $tl \
      --out "$ROWS732" >> "$G/score.out" 2>&1 || { log "STOP: scoring ticket failed ($size)"; return 1; }
    [ "$DRYRUN" = 1 ] && break
  done
  [ "$DRYRUN" = 1 ] && return 0
  have=$(min_users "$ROWS732" "${GLBLS[@]}")
  log "$size: $have of $want users scored for every label (${#GLBLS[@]} labels)"
  return 0
}

preconditions() {
  for f in "$BLOCK" "$RULE" "$S/models.sh" "$S/labels.sh" "$S/design.md" "$STATS" "$USERS732" "$USMOKE" "$CPUSMOKE0" "$SC" \
           "$S/sm157-result.json"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/"; return 1; fi
  "$PY" -X utf8 "$RULE" selftest >> "$LOG" 2>&1 || { log "STOP: report selftest failed"; return 1; }
  local v; v=$("$PY" -X utf8 -c "import json; v = json.load(open(r'$S/sm157-result.json'))['verdict']; print(int(bool(v.get('gpu_recommended'))), v.get('gpu_kind') or '-')" | tr -d '\r')
  KIND=${KIND:-${v#* }}
  if [ "${v%% *}" != 1 ] && [ "${FORCE:-0}" != 1 ]; then log "STOP: the CPU pointer did not pass (FORCE=1 to run anyway)"; return 1; fi
  case "$KIND" in r) WNAMES="Q2R T24R";; rk) WNAMES="Q2RK T24RK";; g) WNAMES="Q2G T24G";; *) log "STOP: no weight kind ($KIND)"; return 1;; esac
  log "weight kind $KIND: labels $WNAMES"
  return 0
}

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1 | cut -c1-90) (DRYRUN=$DRYRUN; dirty source files: $(git -C "$WT" status --porcelain -- scripts expertium rwkv | wc -l))"
rc=0
stopped && rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
[ $rc -eq 0 ] && { cpu_smoke || rc=1; }
[ $rc -eq 0 ] && { smoke || rc=1; }
for step in tiny small big; do
  [ $rc -eq 0 ] || break
  stopped && { rc=1; break; }
  score "$step" || rc=1
done
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done: rows in $ROWS732 (the 732 read-out vs SM148's REF / Q2 / T24 rows is the next agent's step)"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
