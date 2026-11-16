#!/usr/bin/env bash
# Timed recall on your own collection (SM121): CPU only, no GPU, no queue ticket (design: artifacts/small-model/sm121/design.md).
#   Answer time = front + back, question shown to grade press, capped at the max answer seconds ("within S s" includes
#   reading the back). Own collection: aggregate outputs only; per-review arrays stay in sm121/work/ (gitignored).
# Steps: start not before Tue 11-17 01:05; convert the repro copy (work/repro); the tiny CPU check (first 800 reviews);
#   main trunks fp32 (SM92 with its SM117 and SM83 heads, SM98, SM106, SM91s2, SM99); the reproduction control trunks
#   (SM92 + SM98, int4 card / int8 note state, on the repro copy) and sm121-repro.{md,json}; the main report
#   sm121-result.{md,json}; then wait (until Tue 18:30 at the latest) for the SM119 heads of SM113 / SM115, score them and
#   rewrite the report.
# Quiet windows: quiet.json (scope all/cpu) plus the daily 20:00-01:00 break and Sunday from 16:00, re-read at every check;
#   every step starts only when its budget fits before the next window. Below-normal priority, 4 threads, CUDA hidden.
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm121/artifacts/small-model/sm121/launch-sm121.sh
#   launch   WMI hidden (artifacts/small-model/sm121/HANDOFF.md); log artifacts/small-model/sm121/run/launch-sm121.log
#   resume   the launch command again (finished trunk files are skipped)
#   stop     touch C:/Users/jonat/workspace/wt/sm121/artifacts/small-model/sm121/run/STOP (checked between steps)
# Read only: wt/sm117 + wt/sm119 heads, wt/sm92 (SM83 head), the 2026-10-14 collection copy, the fork's shipped files.
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
WT=C:/Users/jonat/workspace/wt/sm121
S=$WT/artifacts/small-model/sm121
R=$S/run
W=$S/work
mkdir -p "$R" "$W"
LOG=$R/launch-sm121.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$R/launch-sm121-dryrun.log
X="$PY -X utf8 $WT/scripts/sm121_own.py"
SM119HEADS=C:/Users/jonat/workspace/wt/sm119/artifacts/small-model
START_AT=${START_AT:-"2026-11-17 01:05"}
SM119_DEADLINE=${SM119_DEADLINE:-"2026-11-17 18:30"}
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=-1
log() { echo "[sm121 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$R/STOP" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }

block() {  # -> "NOW <seconds to its end>" inside a quiet window (quiet.json scope all/cpu, or the daily break), else "NEXT <minutes>"
  Q="$QUIET" "$PY" -X utf8 - <<'PYEOF'
import json, os, re
from datetime import datetime, timedelta
now = datetime.now()
try:
    text = open(os.environ["Q"], encoding="utf-8").read()
except OSError as error:
    print(f"ERR quiet.json unreadable: {error}"); raise SystemExit
try:
    wins = [(w["start"], w["end"]) for w in json.loads(text) if w.get("scope", "all") in ("all", "cpu")]
except (ValueError, KeyError, TypeError):  # malformed JSON: take every start/end pair textually, scope assumed "all"
    wins = re.findall(r'"start"\s*:\s*"([^"]+)"[^{}]*?"end"\s*:\s*"([^"]+)"', text, re.S)
blocks = [(datetime.fromisoformat(s), datetime.fromisoformat(e)) for s, e in wins]
day = now.replace(hour=0, minute=0, second=0, microsecond=0)
blocks += [(day + timedelta(days=d, hours=20), day + timedelta(days=d + 1, hours=1)) for d in range(-1, 9)]  # daily break
blocks += [(day + timedelta(days=d, hours=16), day + timedelta(days=d, hours=20))                          # Sunday from 16:00
           for d in range(-1, 9) if (day + timedelta(days=d)).weekday() == 6]
blocks.sort()
merged = []
for s, e in blocks:
    if merged and s <= merged[-1][1]:
        merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    else:
        merged.append((s, e))
for s, e in merged:
    if s <= now < e:
        print(f"NOW {int((e - now).total_seconds()) + 60}"); raise SystemExit
    if s > now:
        print(f"NEXT {int((s - now).total_seconds() // 60)}"); raise SystemExit
print("NEXT 100000")
PYEOF
}

wait_free() {  # wait_free MINUTES WHAT: returns once MINUTES fit before the next quiet window
  local n=0
  while true; do
    stopped && return 1
    local b; b=$(block)
    case "$b" in
      NOW*) local s=${b#NOW }
            if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $2 would wait inside a quiet window (${s}s left)"; return 0; fi
            [ $((n % 6)) = 0 ] && log "$2: quiet window, ${s}s left"; n=$((n + 1)); nap "$s";;
      NEXT*) local free=${b#NEXT }
             [ "$free" -ge $(( $1 + 3 )) ] && return 0
             if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $2 ($1 min) would wait: $free min to the next window"; return 0; fi
             [ $((n % 6)) = 0 ] && log "$2 needs $1 min, $free min before the next window; waiting"; n=$((n + 1)); nap $(( free * 60 + 60 ));;
      *) log "window check failed: $b"; nap 600;;
    esac
  done
}

step() {  # step MINUTES TIMEOUT_MIN WHAT OUTFILE -- CMD...   one CPU step, killed after TIMEOUT_MIN
  local minutes=$1 tmo=$2 what=$3 outf=$4 rc; shift 5
  wait_free "$minutes" "$what" || return 1
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN step: $what (budget $minutes min, timeout $tmo min)"; echo "    cmd: $*" >> "$LOG"; return 0; fi
  log "start: $what (budget $minutes min, timeout $tmo min)"
  timeout "${tmo}m" "$@" >> "$outf" 2>&1; rc=$?
  log "exit $rc: $what$( [ "$rc" = 124 ] && echo ' (timeout)')$( [ -f "$outf" ] && echo ": $(tail -1 "$outf" | cut -c1-200)")"
  return $rc
}

trunk_step() {  # trunk_step KEY SOURCE STATE PREFIX BUDGET TIMEOUT
  local key=$1 source=$2 state=$3 prefix=$4 budget=$5 tmo=$6
  local f=$W/$prefix-$key.npz
  if [ "$DRYRUN" != 1 ] && [ -f "$f" ] && [ -f "${f%.npz}.json" ]; then log "$prefix $key: already scored, skipped"; return 0; fi
  step "$budget" "$tmo" "$prefix $key ($source, $state)" "$R/$prefix-$key.out" -- $X trunk --model "$key" --source "$source" --state "$state" --out "$f"
}

head_ready() { [ -f "$1/head.pth" ] && [ -f "$1/master.pth" ] && F="$1/train.json" "$PY" -X utf8 -c "
import json, os, sys
sys.exit(0 if json.load(open(os.environ['F'])).get('status') == 'complete' else 1)" 2>/dev/null; }

cd "$WT" || exit 1
log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN)"
if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts)" ]; then log "STOP: uncommitted changes in scripts/"; exit 1; fi
if [ "$DRYRUN" != 1 ]; then
  now=$(date +%s); at=$(date -d "$START_AT" +%s)
  if [ "$now" -lt "$at" ]; then log "sleeping until $START_AT"; while [ "$(date +%s)" -lt "$at" ]; do stopped && exit 1; sleep 300; done; fi
fi
rc=0
# 1. convert the repro copy (stdout to work/: the converter prints first/last review dates)
if [ "$DRYRUN" = 1 ] || [ ! -f "$W/repro/copy-info.json" ]; then
  step 10 20 "convert the repro copy" "$W/convert.out" -- $X convert || rc=1
fi
# 2. the tiny CPU check
if [ $rc -eq 0 ]; then
  step 25 60 "CPU check (first 800 reviews)" "$R/check.out" -- $X check --cap 800 || { log "STOP: the CPU check failed ($S/check-cpu.json, $R/check.out)"; rc=1; }
fi
# 3. main trunks, fp32 state (Big last; killed and skipped after 3 h)
if [ $rc -eq 0 ]; then
  for key in sm92 sm98 sm106; do trunk_step "$key" main fp32 trunk 60 120 || { log "STOP: trunk $key failed"; rc=1; break; }; done
fi
if [ $rc -eq 0 ]; then
  for key in sm99 sm91s2; do trunk_step "$key" main fp32 trunk 90 180 || log "trunk $key failed or took over 3 h: skipped (Big)"; done
fi
# 4. reproduction control: int4 card / int8 note state on the repro copy, then its report
if [ $rc -eq 0 ]; then
  for key in sm92 sm98; do trunk_step "$key" repro s4c8n repro-trunk 60 120 || { log "STOP: repro trunk $key failed"; rc=1; break; }; done
fi
[ $rc -eq 0 ] && { step 20 60 "reproduction control report" "$R/repro.out" -- $X repro || { log "STOP: repro report failed"; rc=1; }; }
# 5. main report (without the SM119 heads if they are not there yet)
[ $rc -eq 0 ] && { step 60 150 "main report" "$R/report.out" -- $X report || { log "STOP: main report failed"; rc=1; }; }
# 6. the SM119 heads (SM113, SM115): wait until the deadline, then score and rewrite the report
if [ $rc -eq 0 ]; then
  dl=$(date -d "$SM119_DEADLINE" +%s); n=0; added=0
  while true; do
    stopped && { rc=1; break; }
    ready=""
    for key in sm113 sm115; do head_ready "$SM119HEADS/sm119-$key-head" && ready="$ready $key"; done
    if [ "$ready" = " sm113 sm115" ] || [ "$DRYRUN" = 1 ]; then break; fi
    if [ "$(date +%s)" -ge "$dl" ]; then log "SM119 deadline $SM119_DEADLINE reached; ready:${ready:- none}"; break; fi
    [ $((n % 6)) = 0 ] && log "waiting for the SM119 heads (ready:${ready:- none})"; n=$((n + 1)); sleep 600
  done
  for key in ${ready:-}; do trunk_step "$key" main fp32 trunk 60 120 && added=1 || log "trunk $key failed"; done
  if [ "$added" = 1 ]; then step 60 150 "main report with the SM119 heads" "$R/report.out" -- $X report || { log "STOP: main report failed"; rc=1; }; fi
fi
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
