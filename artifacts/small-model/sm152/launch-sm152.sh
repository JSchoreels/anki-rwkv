#!/usr/bin/env bash
# Smaller shift vectors and factors on T24 states (SM152) (artifacts/small-model/sm152/design.md, pre-registered 11-23
#   before any scoring of the new labels). No training. A narrow copy of the rank-1 int8 states (SM150)'s launcher: new
#   labels S4 (T24 + int4 shift vectors), F6 (T24 + int6 factors), F6S4 (both) only; REF and T24 rows are SM148's (REF,
#   T24 and Q2 rescored on the 3 smoke users only, as the integrity check). Two parts, one per launch (PART=gpu default):
#   PART=gpu (scripts/r01_queue.py tickets, owner sm152, claim gpu, cap 0.8; PYTORCH_JIT=0 for the fork's eager modules):
#     1 smoke   sm152_check.py gpu-selftest + 3 smoke users x 6 models x REF, T24, S4, F6, F6S4; sm152_report.py smoke vs
#               the CPU rows and (REF, T24) vs SM148's GPU smoke rows
#     2 scoring one ticket per size (Tiny, Small, Big), the 732 set (723 users), the 2 models of the size x GPU_LABELS
#               (S4, F6, F6S4), --prior-buckets; passes at cap 0.8, then 0.92 after a pass without progress (memory only)
#     3 report  sm152_report.py report (CPU) -> sm152-result.{json,md} (long-card warnings "not run yet"), done
#   PART=longcard (LATER, separate launch; NOT started with the GPU run; no ticket; memory guard 5.9 GB per process,
#     quiet.json windows of scope all / cpu):
#     1 long-card users (users-longcard.json, 50 users), SMALL AND TINY ONLY, one fresh CPU process per (size, user), the
#       2 models x LONG_LABELS (S4, F6, F6S4), --prior-buckets, --dump work/sm152-dump --dump-scope all (never committed),
#       LC_WORKERS (default 2) processes at a time
#     2 dR      SM148's REF dumps are copied next to the new dumps, sm152_dr.py aggregate (Small, Tiny) -> sm152-dr.json
#     3 report  sm152_report.py report again (with the long-card warnings), longcard done
#   All scoring: show-time inputs (e2s-query), leak-free (--deleted-cards neighbour), --nonfinite-rows, --split-existing.
# Breaks: quiet.json windows only (GPU: scope all / gpu, sm135_block.py; CPU: scope all / cpu, sm136_block.py).
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm152/artifacts/small-model/sm152/launch-sm152.sh   (PART=longcard too)
#   launch   WMI hidden (HANDOFF.md); log artifacts/small-model/sm152/gpu/launch-sm152.log (longcard: cpu/launch-sm152-longcard.log)
#   resume   remove gpu/STOP (and cpu/STOP), then the launch command again (finished users and steps are skipped)
#   stop     touch C:/Users/jonat/workspace/wt/sm152/artifacts/small-model/sm152/gpu/STOP   (both parts)
# Do not edit scripts/ expertium/ rwkv/ of the sm152 worktree while it runs.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm152
SM=$WT/artifacts/small-model
S=$SM/sm152
G=$S/gpu
C=$S/cpu
mkdir -p "$G" "$C/longcard"
PART=${PART:-gpu}
case "$PART" in gpu|longcard) ;; *) echo "PART must be gpu or longcard"; exit 1;; esac
LOG=$G/launch-sm152.log
[ "$PART" = longcard ] && LOG=$C/launch-sm152-longcard.log
[ "${DRYRUN:-0}" = 1 ] && LOG=${LOG%.log}-dryrun.log
SC=$WT/scripts/small_model_heldout.py
RULE=$S/sm152_report.py
CHECK=$S/sm152_check.py
BLOCK=$SM/sm135/sm135_block.py
CBLOCK=$SM/sm136/sm136_block.py
. "$S/models.sh"; . "$S/labels.sh"
USERS732=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json
ULONG=$S/users-longcard.json
USMOKE=$S/users-smoke.json
CPUSMOKE=$C/smoke-cpu-rows.jsonl
GPUSMOKE=$G/smoke-gpu-rows.jsonl
ROWS732=$G/sm152-rows-732.jsonl
LONGROWS=$C/sm152-longcard-rows.jsonl
DUMP=$WT/work/sm152-dump
WT148=C:/Users/jonat/workspace/wt/sm148
REFN=REF
DRJ=$S/sm152-dr.json
DRPY=$S/sm152_dr.py
TICKETS=$G/tickets.jsonl
STOPF=$G/STOP
NAME="Smaller shift vectors and factors on T24 states (SM152)"
declare -A MIN=([tiny]=70 [small]=100 [big]=165)   # pass 1 per size, 3 labels (SM150: 41 / 57 / 94 min for 2 labels)
declare -A WANT=([tiny]=723 [small]=723 [big]=718)   # Big: the 5 largest users never fit the scorer (SM148)
MIN_SMOKE=15; MIN_RETRY=30
LC_WORKERS=${LC_WORKERS:-2}; LC_WORKERS_AFTER=${LC_WORKERS_AFTER:-$LC_WORKERS}   # CPU processes (longcard part)
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START SM11_INPUTS
log() { echo "[sm152 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$STOPF" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }
free_mib() { awk '/MemFree/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 99999; }
block() { "$PY" -X utf8 "$BLOCK" | tr -d '\r'; }
cblock() { "$PY" -X utf8 "$CBLOCK" --scopes all,cpu | tr -d '\r'; }
jline() { "$PY" -X utf8 -c "import json, sys; print(json.dumps(dict(zip(sys.argv[1::2], [int(v) if v.lstrip('-').isdigit() else v for v in sys.argv[2::2]]))))" "$@" | tr -d '\r'; }

wait_free() {  # wait_free MINUTES WHAT [BLOCKFN]: returns once MINUTES fit before the next quiet window (FREE = minutes to it)
  local n=0 b fn=${3:-block}
  while true; do
    stopped && return 1
    b=$($fn)
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

ram_gate() {  # ram_gate WHAT: hold while < 3 GB RAM is free, at most 30 min, then go on (GPU tickets)
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
    "$PY" "$QUEUE" run --owner sm152 --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm152/launch-sm152.sh" -- "$@"
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

smoke() {
  [ -e "$G/smoke-ok" ] && return 0
  group_args all "$GPU_SMOKE_LABELS"
  q $MIN_SMOKE smoke "GPU smoke: low-rank kernel self-test + 3 users (2 small, 1 long-card), 6 models x REF / T24 / T24 + int4 shifts / int6 factors / both, vs the CPU rows and SM148" -- \
    env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS HELDOUT_GPU_MEMORY_FRACTION=0.8 PYTORCH_JIT=0 bash -c \
    '"$0" -X utf8 -W ignore "$1" gpu-selftest --out "$2" && shift 2 && exec "$0" -X utf8 "$@"' \
    "$PY" "$CHECK" "$G/gpu-selftest.json" "$SC" --pool 732 --users-file "$USMOKE" "${GARGS[@]}" "${SCORE_FLAGS[@]}" --time-limit 900 \
    --out "$GPUSMOKE" >> "$G/score.out" 2>&1 || { log "STOP: smoke ticket failed (see $G/score.out)"; return 1; }
  [ "$DRYRUN" = 1 ] && return 0
  local out; out=$("$PY" -X utf8 "$RULE" smoke --gpu "$GPUSMOKE" --cpu "$CPUSMOKE" 2>&1 | tr -d '\r')
  echo "$out" > "$G/smoke-check.txt"; log "$out"
  case "$out" in *"smoke OK"*) touch "$G/smoke-ok"; return 0;; esac
  log "STOP: GPU smoke mismatch (see $G/smoke-check.txt, $G/smoke-check.json)"; return 1
}

score() {  # score SIZE: one size on the 732 set, passes with the cap rule
  local size=$1 cap=0.8 pass have prev=-1 minutes tl want=${WANT[$1]}
  group_args "$size" "$GPU_LABELS"
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
    q $minutes score "scoring $size (2 models x T24 + int4 shift vectors / int6 factors / both) on e2s-query inputs, the 732 set minus 9 never-fit (723 users) (pass $pass, $have done, cap $cap)" -- \
      env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS HELDOUT_GPU_MEMORY_FRACTION=$cap PYTORCH_JIT=0 "$PY" -X utf8 "$SC" --pool 732 \
      --users-file "$USERS732" "${GARGS[@]}" "${SCORE_FLAGS[@]}" --time-limit $tl \
      --out "$ROWS732" >> "$G/score.out" 2>&1 || { log "STOP: scoring ticket failed ($size)"; return 1; }
    [ "$DRYRUN" = 1 ] && break
  done
  [ "$DRYRUN" = 1 ] && return 0
  have=$(min_users "$ROWS732" "${GLBLS[@]}")
  log "$size: $have of $want users scored for every label (${#GLBLS[@]} labels)"
  return 0
}

gpu_chain() {
  local rc=0 step
  smoke || rc=1
  for step in tiny small big; do
    [ $rc -eq 0 ] || break
    stopped && { rc=1; break; }
    score "$step" || rc=1
  done
  echo $rc > "$G/.gpu-chain-rc"
  log "GPU chain ended (exit $rc)"
}

# ---------------------------------------------------------------- CPU chain (long-card users)
memguard_on() {  # CPU long-card processes of this run only; kill above 5.9 GB (writes cpu/memguard-kill and cpu/STOP)
  [ "$DRYRUN" = 1 ] && return 0
  rm -f "$C/memguard.off"
  local wp; wp=$(cat /proc/$$/winpid 2>/dev/null || echo 0)
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$S/memguard.ps1")" -Dir "$(cygpath -w "$C")" -ParentPid "$wp" \
    -Match "small_model_heldout.py&--device cpu&sm152" > /dev/null 2>&1 &
  MG=$!
  log "memory guard on (launcher Windows PID $wp, limit 5.9 GB per CPU process)"
}
memguard_off() { [ "$DRYRUN" = 1 ] && return 0; touch "$C/memguard.off"; sleep 12; kill "$MG" 2>/dev/null; return 0; }

lc_users() {  # lc_users SIZE: "user reviews" of users-longcard.json without complete rows for SIZE, smallest first
  local size=$1
  group_args "$size" "$LONG_LABELS"
  U="$ULONG" D="$C/longcard/$size" WT="$WT" "$PY" -X utf8 -c "
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.environ['WT'])
from scripts import compress_common as cc
counts = cc.run_info()['review_counts']
want = set(sys.argv[1:])
killed = set()
kf = Path(os.environ['D']) / 'killed.txt'
if kf.exists():
    killed = {int(x.split()[0]) for x in kf.read_text().split(chr(10)) if x.strip()}
for u in sorted(json.load(open(os.environ['U']))['users'], key=lambda u: (counts[str(u)], u)):
    p, got = Path(os.environ['D']) / f'u-{u}.jsonl', set()
    if p.exists():
        for raw in open(p, 'rb'):
            try:
                r = json.loads(raw.decode('utf-8', errors='replace'))
            except ValueError:
                continue
            if 'error' not in r and ('benchmark' in r or 'nonfinite' in r):
                got.add(r.get('label'))
    if not want <= got and u not in killed:
        print(u, counts[str(u)])" "${GLBLS[@]}" | tr -d '\r'
}

lc_one() {  # lc_one SIZE USER: the 2 models of SIZE x LONG_LABELS with --prior-buckets, CPU, one fresh process
  local size=$1 u=$2 D=$C/longcard/$1
  group_args "$size" "$LONG_LABELS"
  echo "{\"users\": [$u]}" > "$D/users-$u.json"
  rm -f "$D/u-$u.jsonl" "$D/u-$u.part.jsonl"
  local tag tags=""; for tag in $(echo "$MODELS" | awk -F'|' -v s="$size" '$6 == s {print $1}'); do tags="$tags $tag"; done
  dump_rm() { local t; for t in $tags; do rm -f "$DUMP/${u}_${t}_"*.npz; done; }   # a user's dumps exist only with its complete rows
  dump_rm
  local t0; t0=$(date +%s)
  env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START -u SM11_INPUTS -u HELDOUT_GPU_MEMORY_FRACTION CUDA_VISIBLE_DEVICES= PYTORCH_JIT=0 "$PY" -X utf8 -W ignore "$SC" \
    --device cpu --workers 1 --pool 732 --users-file "$D/users-$u.json" "${GARGS[@]}" "${SCORE_FLAGS[@]}" --time-limit 1000000 \
    --dump "$DUMP" --dump-scope all --out "$D/u-$u.part.jsonl" >> "$D/score-$u.out" 2>&1
  local rc=$?
  if [ $rc = 0 ]; then mv "$D/u-$u.part.jsonl" "$D/u-$u.jsonl"
    log "long-card $size user $u done in $(( ($(date +%s) - t0) / 60 )) min: $(grep -a '^\[small-model\] 1/1' "$D/score-$u.out" | tail -1 | cut -c1-110)"
  elif grep -q "u-$u.part.jsonl" "$C/memguard-kill" 2>/dev/null; then
    echo "$u killed by the memory guard (exit $rc)" >> "$D/killed.txt"; rm -f "$C/STOP"
    log "long-card $size user $u killed by the memory guard (exit $rc); counted as missing, going on"; dump_rm
  else echo "$size $u exit $rc" >> "$C/longcard/failed.txt"; log "long-card $size user $u failed (exit $rc, see $D/score-$u.out)"; dump_rm; fi
  return 0
}

lc_slots() {  # lc_slots N: wait until fewer than N long-card processes run (N = 1: all done)
  local p alive
  while true; do
    alive=()
    for p in "${LC_PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive+=("$p"); done
    LC_PIDS=("${alive[@]}")
    [ ${#LC_PIDS[@]} -lt "$1" ] && return 0
    sleep 10
  done
}

mem_fits() {  # mem_fits REVIEWS: free RAM >= this user's expected peak + 2 GB (SM136: ~3.2 GB at 103k, 5.6 GB at 150k reviews)
  local need=$(( $1 * 38 / 1000 + 800 + 2048 ))
  [ "$(free_mib)" -ge "$need" ]
}

cpu_chain() {
  local rc=0 size u n todo waited
  memguard_on
  for size in tiny small; do
    mkdir -p "$C/longcard/$size"
    if [ "$DRYRUN" = 1 ]; then
      group_args "$size" "$LONG_LABELS"
      log "DRYRUN: long-card CPU rows ($size): $(lc_users $size | grep -c .) of 50 users to score, $LC_WORKERS at a time, e.g."
      echo "    cmd: $SC --device cpu --workers 1 --pool 732 --users-file $C/longcard/$size/users-U.json ${GARGS[*]} ${SCORE_FLAGS[*]} --dump $DUMP --dump-scope all --out $C/longcard/$size/u-U.part.jsonl" >> "$LOG"
      continue
    fi
    LC_PIDS=()
    todo=$(lc_users "$size")
    log "long-card CPU rows ($size): $(echo "$todo" | grep -c .) of 50 users to score ($LC_WORKERS processes at a time, $(free_mib) MiB RAM free)"
    while read -r u n; do
      [ -z "$u" ] && continue
      if [ -e "$G/.gpu-chain-rc" ]; then lc_slots $LC_WORKERS_AFTER; else lc_slots $LC_WORKERS; fi
      wait_free $(( n * 2 * 6 / 60000 + 5 )) "long-card user $u ($size)" cblock || { lc_slots 1; rc=1; break 2; }
      waited=0
      until mem_fits "$n"; do   # never start a user whose peak would not fit; with nothing running, start anyway (guard)
        [ ${#LC_PIDS[@]} = 0 ] && break
        [ $waited = 0 ] && log "long-card user $u ($size, $n reviews): $(free_mib) MiB RAM free, waiting for a slot to end"
        waited=1; sleep 30; lc_slots $(( ${#LC_PIDS[@]} ))
      done
      stopped && { lc_slots 1; rc=1; break 2; }
      lc_one "$size" "$u" &
      LC_PIDS+=($!)
      sleep 20
    done <<< "$todo"
    lc_slots 1
  done
  memguard_off
  if [ $rc = 0 ] && [ "$DRYRUN" != 1 ]; then
    cat "$C"/longcard/*/u-*.jsonl > "$LONGROWS.tmp" && mv "$LONGROWS.tmp" "$LONGROWS"
    log "long-card rows merged: $(wc -l < "$LONGROWS") rows in $LONGROWS (killed: $(cat "$C"/longcard/*/killed.txt 2>/dev/null | wc -l), failed: $(cat "$C/longcard/failed.txt" 2>/dev/null | wc -l))"
  fi
  echo $rc > "$C/.cpu-chain-rc"
  log "CPU chain ended (exit $rc)"
}

preconditions() {
  for f in "$BLOCK" "$CBLOCK" "$RULE" "$CHECK" "$DRPY" "$S/models.sh" "$S/labels.sh" "$S/design.md" "$S/memguard.ps1" "$S/sm152-bytes.json" \
           "$USERS732" "$ULONG" "$USMOKE" "$CPUSMOKE" "$SC" "$S/../sm148/gpu/sm148-rows-732.jsonl" "$S/../sm148/gpu/smoke-gpu-rows.jsonl"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/"; return 1; fi
  "$PY" -X utf8 -c "import json, sys; sys.exit(0 if json.load(open(r'$S/check-cpu-sm152.json'))['ok'] else 1)" \
    || { log "STOP: the CPU proofs (check-cpu-sm152.json) are not ALL OK"; return 1; }
  "$PY" -X utf8 "$BLOCK" selftest >> "$LOG" 2>&1 || { log "STOP: block selftest failed"; return 1; }
  "$PY" -X utf8 "$RULE" check >> "$LOG" 2>&1 || { log "STOP: pins changed"; return 1; }
  "$PY" -X utf8 "$RULE" selftest >> "$LOG" 2>&1 || { log "STOP: report selftest failed"; return 1; }
  "$PY" -X utf8 "$DRPY" selftest >> "$LOG" 2>&1 || { log "STOP: dR selftest failed"; return 1; }
  return 0
}

final_check() {  # every expected output of the part exists (non-empty) before done
  local f
  if [ "$PART" = gpu ]; then
    [ -e "$G/smoke-ok" ] || { log "STOP: expected output missing: $G/smoke-ok"; return 1; }
    for f in "$G/gpu-selftest.json" "$GPUSMOKE" "$G/smoke-check.txt" "$ROWS732" "$S/sm152-result.json" "$S/sm152-result.md" "$TICKETS"; do
      [ -s "$f" ] || { log "STOP: expected output missing or empty: $f"; return 1; }
    done
  else
    for f in "$LONGROWS" "$DRJ" "$S/sm152-result.json" "$S/sm152-result.md"; do
      [ -s "$f" ] || { log "STOP: expected output missing or empty: $f"; return 1; }
    done
  fi
  return 0
}

cd "$WT" || exit 1
log "started ($PART part): $(git -C "$WT" log --oneline -1 | cut -c1-90) (DRYRUN=$DRYRUN; dirty source files: $(git -C "$WT" status --porcelain -- scripts expertium rwkv | wc -l))"
rc=0
stopped && rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
if [ $rc -eq 0 ] && [ "$PART" = gpu ]; then
  rm -f "$G/.gpu-chain-rc"
  gpu_chain
  [ "$(cat "$G/.gpu-chain-rc" 2>/dev/null | tr -dc '0-9')" = 0 ] || rc=1
fi
if [ $rc -eq 0 ] && [ "$PART" = longcard ]; then
  [ -s "$ROWS732" ] || log "note: the GPU rows ($ROWS732) are missing; the report will be pending"
  rm -f "$C/.cpu-chain-rc"
  cpu_chain
  [ "$(cat "$C/.cpu-chain-rc" 2>/dev/null | tr -dc '0-9')" = 0 ] || rc=1
  if [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ]; then   # the per-review |dR| read-out from the dumps (aggregates only)
    n=0; for tag in $(echo "$MODELS" | awk -F'|' '$6 != "big" && NF > 5 {print $1}'); do   # SM148's REF dumps (same code, same inputs) next to the new ones
      for f in "$WT148/work/sm148-dump/"*"_${tag}_${!REFN}@e2s-query.npz"; do [ -e "$f" ] && { cp -n "$f" "$DUMP/"; n=$((n + 1)); }; done
    done; log "REF dumps copied from SM148: $n files"
    out=$(set -o pipefail; "$PY" -X utf8 "$DRPY" aggregate --dump "$DUMP" --users "$ULONG" --out "$DRJ" --sizes Small,Tiny 2>&1 | tr -d '\r'); r=$?
    log "dR read-out exit $r: $(echo "$out" | tail -1 | cut -c1-300)"
    [ $r -eq 0 ] || { log "STOP: dR read-out failed"; rc=1; }
  elif [ "$DRYRUN" = 1 ]; then log "DRYRUN: $PY -X utf8 $DRPY aggregate --dump $DUMP --users $ULONG --out $DRJ --sizes Small,Tiny"; fi
fi
STATUS="not run"
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $PY -X utf8 $RULE report"
  else
    out=$(set -o pipefail; "$PY" -X utf8 "$RULE" report 2>&1 | tr -d '\r'); r=$?   # pipefail: the report's own exit code
    log "report exit $r: $(echo "$out" | tail -1 | cut -c1-600)"
    if [ $r -eq 2 ]; then log "report pending (coverage): the main session decides (sm152-result.json 'pending')"; STATUS=pending
    elif [ $r -ne 0 ]; then log "STOP: report exit $r"; rc=1
    else STATUS=$("$PY" -X utf8 -c "import json; d = json.load(open(r'$S/sm152-result.json')); print(d['status'], {s: g['pick'] for s, g in d['go'].items()})" | tr -d '\r'); fi
  fi
fi
[ $rc -eq 0 ] && [ "$DRYRUN" != 1 ] && { final_check || rc=1; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK ($PART part)"
  elif [ "$PART" = gpu ]; then log "done (status $STATUS)"
  else log "longcard done (status $STATUS)"; fi
else
  log "STOP: ended with exit $rc ($PART part)"
fi
exit $rc
