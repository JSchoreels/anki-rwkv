#!/usr/bin/env bash
# The big instant-only teacher (SM110) (artifacts/small-model/sm110/design.md), designed 2026-11-12:
#   the Small Instant-Only model (SM98)'s cool-down rerun with ONLY its KD replay targets swapped:
#   targets  the show-time instant-only big model (SM99, final epoch 2, read only) replays the full histories of the
#            8,000 cool-down users on show-time inputs (scripts/sm11_replay_export.py --inputs e2s-query,
#            RWKV_DELETED_CARDS=neighbour) into a NEW cache folder (never the old one)
#   start    SM98's constant-LR epoch 35 (wt/sm98, read only)
#   cool-down the release's one-epoch 32k Muon cool-down on show-time inputs, curve loss weight 0, == SM98's cool-down
#            except the teacher + replay folder + paths (sm110_pair.py; prepared after the export, trims must match)
#   mid-check after 4,000 of the 8,000 users: a snapshot stop, SM110's snapshot users-04000 vs SM98's own snapshot
#            users-04000 (scored first, before the export), show-time instant; sm110_rule.py mid: continue if < 0
# Final rule (sm110_rule.py decide, pre-registered): SM110 - SM98, instant, show-time inputs, 722 users of the 732 set:
#   go if the 95% CI is below 0 and the gain >= 0.05% (% of the big model (1 leak-free epoch) normal instant LogLoss).
# CPU proofs (sm110_check.py -> check-cpu-sm110.json): ALL OK required.
# Breaks: sm110_block.py = quiet.json (scope all/gpu) + the daily 20:00-01:00 (Sunday from 16:00) unless quiet.json overlaps
#   that night; re-read at every check, CPU steps too. Tickets must end before the next break; the export ticket stops
#   starting users before it; the cool-down starts when 30 min fit, snapshot-stops <= 2 min before a break, resumes after.
#   dry run  DRYRUN=1 bash C:/Users/jonat/workspace/wt/sm110/artifacts/small-model/sm110/launch-sm110.sh
#   launch   WMI hidden (HANDOFF.md); log artifacts/small-model/sm110/gpu/launch-sm110.log
#   resume   the launch command again (finished steps are skipped; the export skips finished users; the cool-down
#            resumes from its snapshot)
#   stop     touch C:/Users/jonat/workspace/wt/sm110/artifacts/small-model/sm110/gpu/STOP (between tickets; a running
#            cool-down gets a snapshot stop at the next user boundary; a running export stops at its time limit)
# Do not edit scripts/ expertium/ rwkv/ of the sm110 worktree while it runs (SourceWatch stops the training).
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py
WT=C:/Users/jonat/workspace/wt/sm110
SM=$WT/artifacts/small-model
S=$SM/sm110
G=$S/gpu
mkdir -p "$G"
LOG=$G/launch-sm110.log
[ "${DRYRUN:-0}" = 1 ] && LOG=$G/launch-sm110-dryrun.log
DEC_PY=$WT/scripts/expertium_decay.py
SC=$WT/scripts/small_model_heldout.py
EXPORT=$WT/scripts/sm11_replay_export.py
RULE=$S/sm110_rule.py
PAIR=$S/sm110_pair.py
CHECK=$S/sm110_check.py
WARM="$PY -X utf8 $WT/scripts/sm77_warm.py"
TRACK=E89-F65e-F77a
ARCH=w64
K=25                          # the release's cool-down recipe (sm77_warm.py cooldown --epoch 25: users / order / seeds)
TEACHER=C:/Users/jonat/workspace/wt/sm99/artifacts/small-model/sm99-big-inst-e2/expertium/big-inst/master.pth   # read only
TEACHER_SHA=c49e45d698194da033caa1231f598f875922a4a2a3e921714a3fd433d37e5768
RUN8K=C:/Users/jonat/workspace/wt/sm55/artifacts/small-model/sm55-final8k-run.json                            # read only
REPLAY=C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/sm99-big-inst-final8k-neighbour-e2s       # NEW
S98=C:/Users/jonat/workspace/wt/sm98/artifacts/small-model                                                    # read only
E35=$S98/sm98-inst-e35-st
E35_SHA=f7577d14ba102ab9ce1a4e995f9176854f1aa719a627209244d54ec02536ebd4
DEC98=$S98/sm98-inst-e35st-st-dec-1ep32k-muon
SNAP98=$DEC98/expertium/$TRACK/snapshots/users-04000
S72=C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s                                             # read only
USERS732=$S72/eval-users-732-fit.json; ULONG=$S72/users-732-long.json
USERS1500=C:/Users/jonat/workspace/wt/sm57/artifacts/small-model/sm57/stage4/users-sm93-score.json
RECIPE=$SM/sm110-recipe-e35st-dec-1ep32k-muon-t99    # the cool-down recipe proof (never trained)
DEC=$SM/sm110-inst-e35st-st-dec-1ep32k-muon-t99      # the real cool-down, prepared after the export
CLBL=sm110-inst-e35st-st-dec1ep32k-t99
SPEC="expertium:$ARCH:$DEC/expertium/$TRACK/master.pth"
MIDSNAP=$DEC/expertium/$TRACK/snapshots/users-04000
MID_NEW=sm110-dec-u4000; MID_REF=sm98-dec-u4000
OUTROWS=$G/sm110-rows.jsonl
ROWS1500=$G/sm110-rows-1500.jsonl
MIDROWS=$G/sm110-mid-rows.jsonl
SEGS=$G/segments.jsonl
TICKETS=$G/tickets.jsonl
STOPF=$G/STOP
MODE=neighbour
LF="env RWKV_DELETED_CARDS=$MODE"
EXPORT_RATE=38000             # reviews/s for planning (big-model exports: 40,577 for the 8k targets of 10-26, 38,174 for SM91 step 3)
MIN_DECAY=${MIN_DECAY:-560}   # SM98's cool-down: 479.6 min wall at 20,232 reviews/s + setup
MIN_START=30                  # the cool-down starts when 30 min fit (snapshot stop before the break, resume after)
MIN_SMOKE=15; MIN_SCORE732=25; MIN_SCORE_RETRY=20; MIN_LONG=15; MIN_POOL=35
TL732=1800; TLLONG=1380; TLPOOL=2400
NAME="the big instant-only teacher (SM110)"
export DRYRUN=${DRYRUN:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START SM11_INPUTS   # caps 0.8
log() { echo "[sm110 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
stopped() { [ -e "$STOPF" ] && { log "STOP: STOP file present"; return 0; }; return 1; }
nap() { [ "$DRYRUN" = 1 ] && return 0; local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }
free_mib() { awk '/MemFree/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 99999; }
block() { "$PY" -X utf8 "$S/sm110_block.py"; }
jline() { "$PY" -X utf8 -c "import json, sys; print(json.dumps(dict(zip(sys.argv[1::2], [int(v) if v.lstrip('-').isdigit() else v for v in sys.argv[2::2]]))))" "$@"; }

wait_free() {  # wait_free MINUTES WHAT: returns once MINUTES fit before the next break (sets FREE = minutes to it)
  local n=0
  while true; do
    stopped && return 1
    local b; b=$(block)
    case "$b" in
      NOW*) local s=${b#NOW }
            if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $2 would wait inside a break (${s}s left)"; FREE=100000; return 0; fi
            [ $((n % 6)) = 0 ] && log "$2: break, ${s}s left"; n=$((n + 1)); nap "$s";;
      NEXT*) FREE=${b#NEXT }
             [ "$FREE" -ge $(( $1 + 3 )) ] && return 0
             if [ "$DRYRUN" = 1 ]; then log "DRYRUN: $2 ($1 min) would wait: $FREE min to the next break"; FREE=100000; return 0; fi
             [ $((n % 6)) = 0 ] && log "$2 needs $1 min, $FREE min before the next break; waiting"; n=$((n + 1)); nap $(( FREE * 60 + 60 ));;
      *) log "window check failed: $b"; nap 600;;
    esac
  done
}

ram_gate() {  # ram_gate WHAT [strict]: hold while < 3 GB RAM is free, at most 30 min (strict: then STOP instead of going on)
  local i=0
  while [ "$DRYRUN" != 1 ] && [ "$(free_mib)" -lt 3072 ]; do
    if [ $i -ge 6 ]; then
      [ "${2:-}" = strict ] && { log "STOP: $1: $(free_mib) MiB RAM free (< 3 GB) for 30 min"; return 1; }
      log "$1: still $(free_mib) MiB RAM free after 30 min; going on"; return 0
    fi
    [ $i = 0 ] && log "$1: $(free_mib) MiB RAM free (< 3 GB), holding"; i=$((i + 1)); sleep 300
  done
  return 0
}
cpu_step() { wait_free "$1" "$2" || return 1; ram_gate "$2"; }

q() {  # q MINUTES KIND PURPOSE -- CMD...   one ticket that must end before the next break; timed in $TICKETS (+ nvidia-smi max)
  local minutes=$1 kind=$2 purpose=$3 rc t0 mx base wpid; shift 4
  while true; do
    wait_free "$minutes" "$purpose" || return 1
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $NAME: $purpose (expected $minutes min)"; echo "    cmd: $*" >> "$LOG"; return 0; fi
    local maxwait=$(( FREE - minutes - 3 )); [ "$maxwait" -lt 1 ] && maxwait=1
    log "enqueue: $purpose (expected $minutes min, max wait $maxwait min, $FREE min to the next break)"
    t0=$(date +%s); base=$(gpu_used); rm -f "$G/.gpu-max"
    ( m=0; while true; do u=$(gpu_used); [ -n "$u" ] && [ "$u" -gt "$m" ] && { m=$u; echo "$m" > "$G/.gpu-max"; }; sleep 30; done ) &
    wpid=$!
    "$PY" "$QUEUE" run --owner sm110 --purpose "$NAME: $purpose" --claim gpu --expected-minutes "$minutes" \
      --max-wait-minutes "$maxwait" --launched-by "detached launcher sm110/launch-sm110.sh" -- "$@"
    rc=$?; kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null; mx=$(cat "$G/.gpu-max" 2>/dev/null || echo 0)
    log "exit $rc: $purpose (nvidia-smi max $mx MiB, baseline $base MiB)"
    [ "$rc" = 75 ] && { log "queue wait would reach the next break; re-planning"; sleep 60; continue; }
    jline kind "$kind" purpose "$purpose" exit "$rc" minutes "$(( ($(date +%s) - t0 + 30) / 60 ))" nvidia_smi_max_mib "${mx:-0}" nvidia_smi_baseline_mib "${base:-0}" >> "$TICKETS"
    return $rc
  done
}

label_users() { F="$1" LB="$2" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
p = Path(os.environ['F'])
print(len({r['user_id'] for r in map(json.loads, open(p, encoding='utf-8')) if r.get('label') == os.environ['LB'] and 'benchmark' in r}) if p.exists() else 0)"; }
done_in() { [ -f "$1/expertium/$TRACK/master.pth" ] && R="$1/expertium/$TRACK/result.json" "$PY" -X utf8 -c "
import json, os, sys
r = json.load(open(os.environ['R']))
sys.exit(0 if r.get('status') == 'complete' and not r.get('smoke_only') else 1)" 2>/dev/null; }
mid_pending() { [ -f "$MIDSNAP/snapshot.json" ] && [ -f "$MIDSNAP/master.pth" ] && [ ! -e "$G/mid-done" ]; }
export_left() {  # reviews still to export (0 = complete)
  R="$REPLAY" J="$RUN8K" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
run = json.load(open(os.environ['J']))
users = Path(os.environ['R']) / 'users'
print(sum(run['review_counts'][str(u)] for u in run['train_users'] if not (users / f'{u}.npz').exists()))"; }

segment_line() {  # segment_line FOLDER STAGE START_EPOCH END_EPOCH EXIT BASE_MIB MAX_MIB STOP_SENT
  "$PY" -X utf8 -c "
import json, sys
f, stage, a, b, rc, base, mx, stop = sys.argv[1:9]
print(json.dumps({'folder': f, 'stage': stage, 'start_epoch_s': int(a), 'end_epoch_s': int(b), 'minutes': round((int(b) - int(a)) / 60, 1),
                  'exit': int(rc), 'nvidia_smi_baseline_mib': int(base or 0), 'nvidia_smi_max_used_mib': int(mx or 0), 'stop_requested': stop == '1'}))" \
    "$@" >> "$SEGS"
}

run_folder() {  # run_folder DIR WHAT SEG_MAX: expertium_decay.py run in segments; resumable from snapshots.
                # Returns 10 when the mid-check snapshot (users-04000) exists and the mid-check has not run yet.
  local D=$1 what=$2 segmax=$3 rc qpid stop_sent b need t0 base mx u why
  while ! done_in "$D"; do
    stopped && return 1
    mid_pending && return 10
    need=15; [ ! -d "$D/expertium" ] && [ ! -d "$D/attempts" ] && need=$MIN_START
    wait_free "$need" "$what" || return 1
    ram_gate "$what" || return 1
    wait_free "$need" "$what" || return 1
    local seg=$(( FREE - 5 )); [ "$seg" -gt "$segmax" ] && seg=$segmax
    if [ "$DRYRUN" = 1 ]; then log "DRYRUN enqueue: $NAME: $what (expected $seg min of ~$segmax; snapshot stop at users-04000 for the mid-check)"
      echo "    cmd: $LF $PY -X utf8 $DEC_PY run --out $D" >> "$LOG"; return 0; fi
    rm -f "$D/control/STOP"
    log "enqueue: $what (expected $seg min of ~$segmax, $FREE min to the next break, $(free_mib) MiB RAM free)"
    t0=$(date +%s); base=$(gpu_used); mx=0
    "$PY" "$QUEUE" run --owner sm110 --purpose "$NAME: $what (segment to the next break, the mid-check or the end)" --claim gpu \
      --expected-minutes "$seg" --max-wait-minutes $(( FREE - 20 > 1 ? FREE - 20 : 1 )) --launched-by "detached launcher sm110/launch-sm110.sh" -- \
      $LF "$PY" -X utf8 "$DEC_PY" run --out "$D" >> "$G/train.out" 2>&1 &
    qpid=$!
    stop_sent=0
    while kill -0 "$qpid" 2>/dev/null; do
      u=$(gpu_used); [ -n "$u" ] && [ "$u" -gt "$mx" ] && mx=$u
      b=$(block); why=""
      [ -e "$STOPF" ] && why="STOP file"
      { [[ "$b" == NOW* ]] || { [[ "$b" == NEXT* ]] && [ "${b#NEXT }" -le 2 ]; }; } && why="break ($b)"
      mid_pending && why="mid-check: snapshot users-04000 written"
      if [ -n "$why" ]; then
        [ "$stop_sent" = 0 ] && log "requesting a snapshot stop ($why)"
        "$PY" -X utf8 "$DEC_PY" stop --out "$D" > /dev/null 2>&1; stop_sent=1
      fi
      sleep 30
    done
    wait "$qpid"; rc=$?; log "exit $rc: $what segment (nvidia-smi max $mx MiB, baseline $base MiB)"
    segment_line "$D" "$what" "$t0" "$(date +%s)" "$rc" "$base" "$mx" "$stop_sent"
    [ "$rc" = 75 ] && { sleep 60; continue; }
    done_in "$D" && break
    mid_pending && return 10
    [ "$stop_sent" = 0 ] && { log "STOP: $what segment ended without completing and without a stop request (exit $rc); see $G/train.out"; return 1; }
  done
  [ "$DRYRUN" = 1 ] || log "$what complete: $("$PY" -X utf8 -c "import json; r=json.load(open(r'$D/expertium/$TRACK/result.json')); print(f\"{r.get('review_presentations', 0):,} reviews, {r.get('training_seconds', 0)/60:.1f} min training, {r.get('reviews_per_second', 0):,.0f} reviews/s, peak allocated {r.get('peak_allocated_mib', 0):,.0f} MiB, reserved {r.get('peak_reserved_mib', 0):,.0f} MiB, process tree RSS {r.get('peak_process_tree_rss_mib', 0):,.0f} MiB, curve_loss_weight {r.get('curve_loss_weight')}, init {str(r.get('initial_parameter_sha256'))[:12]}, trained {str(r.get('trained_parameter_sha256'))[:12]}, update offset end {r.get('update_offset_end')}\")")"
  return 0
}

score() {  # score INPUTS LABEL CKPT ROWS POOL USERS_FILE EXPECT STRICT MIN1 TL [CAP0]: passes at CAP0 (default 0.8; a pass
           # without progress -> cap 0.92; no progress at 0.92: strict -> STOP, else go on with what fits)
  local v=$1 name=$2 ck=$3 rows=$4 pool=$5 uf=$6 expect=$7 strict=$8 min1=$9 tl=${10} cap=${11:-0.8} pass have prev=-1 minutes lb
  lb="$name/w8+s8"; [ "$v" != normal ] && lb="$lb@$v"
  for pass in 1 2 3 4 5; do
    have=$(label_users "$rows" "$lb")
    [ "$have" -ge "$expect" ] && break
    if [ "$have" = "$prev" ]; then
      if [ "$cap" = 0.8 ]; then cap=0.92; log "no progress on $lb ($have of $expect users): next pass at cap 0.92 (memory only, same numerics)"
      elif [ "$strict" = 1 ]; then log "STOP: no progress on $lb ($have of $expect users)"; return 1
      else log "no progress at cap 0.92 on $lb ($have of $expect users): the rest do not fit; going on"; break; fi
    fi
    prev=$have
    minutes=$min1; [ "$pass" -gt 1 ] && minutes=$MIN_SCORE_RETRY
    q $minutes score "scoring $name on $v inputs, ${pool#* } (pass $pass, $have done, cap $cap)" -- \
      env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START HELDOUT_GPU_MEMORY_FRACTION=$cap "$PY" -X utf8 "$SC" --pool "${pool%% *}" --users-file "$uf" \
      --model "$name=expertium:$ARCH:$ck" --only "$name/w8+s8" --inputs "$v" --time-limit $tl --split-existing \
      --deleted-cards $MODE --out "$rows" >> "$G/score.out" 2>&1 || { log "STOP: scoring ticket failed ($lb)"; return 1; }
    [ "$DRYRUN" = 1 ] && break
  done
  [ "$DRYRUN" = 1 ] || [ "$strict" != 1 ] || [ "$(label_users "$rows" "$lb")" -ge "$expect" ] || { log "STOP: $lb scored on $(label_users "$rows" "$lb") of $expect users"; return 1; }
  [ "$DRYRUN" = 1 ] || log "$lb: $(label_users "$rows" "$lb") users scored in $(basename "$rows")"
  return 0
}
score732() {  # score732 INPUTS: the final model, 723 users of the 732 set (strict), then the 3 long users (cap 0.92, not needed by the rule)
  local ck=$DEC/expertium/$TRACK/master.pth
  score "$1" "$CLBL" "$ck" "$OUTROWS" "732 the 732 set minus 9 never-fit (723 users)" "$USERS732" 723 1 $MIN_SCORE732 $TL732 || return 1
  score "$1" "$CLBL" "$ck" "$OUTROWS" "732 the 3 long users of the 732 set" "$ULONG" 726 0 $MIN_LONG $TLLONG 0.92 \
    || log "long users scoring failed ($1); going on (the rule pool does not need them)"
  return 0
}

rule() {  # rule CMD WHEN
  [ "$DRYRUN" = 1 ] && { log "DRYRUN: then $PY -X utf8 $RULE $1 ($2)"; return 0; }
  cpu_step 3 "rule" || return 1
  local out r; out=$("$PY" -X utf8 "$RULE" "$1" 2>&1); r=$?
  log "rule $1 ($2) exit $r: $(echo "$out" | tail -1 | cut -c1-900)"
  [ "$r" = 0 ]
}

run_export() {  # the new targets, tickets cut by --time-limit before the next break; per-user resumable
  local left tl minutes rc
  while true; do
    stopped && return 1
    if [ "$DRYRUN" = 1 ]; then left=581942738; else left=$(export_left); fi
    [ "$left" = 0 ] && { log "export complete: $REPLAY"; return 0; }
    minutes=$(( left / EXPORT_RATE / 60 + 1 ))
    wait_free 30 "export ($left reviews left, ~$minutes min)" || return 1
    tl=$(( (FREE - 20) * 60 )); [ "$tl" -gt $(( (minutes + 45) * 60 )) ] && tl=$(( (minutes + 45) * 60 ))
    [ $(( tl / 60 )) -lt "$minutes" ] && log "export: $FREE min before the next break, ~$minutes min of export left: this ticket stops starting users after $(( tl / 60 )) min"
    q $(( (tl / 60 < minutes ? tl / 60 : minutes) + 10 )) export "teacher targets from the show-time instant-only big model (SM99) on show-time inputs, the 8,000 cool-down users (${left} reviews left, ~$minutes min; stops starting users after $(( tl / 60 )) min)" -- \
      env RWKV_DELETED_CARDS=$MODE "$PY" -X utf8 "$EXPORT" export --out "$REPLAY" --run "$RUN8K" --teacher "$TEACHER" --inputs e2s-query \
      --device cuda --time-limit "$tl" >> "$G/export.out" 2>&1
    rc=$?
    [ "$DRYRUN" = 1 ] && return 0
    [ "$rc" = 0 ] || [ "$rc" = 3 ] || { log "STOP: export ticket exit $rc (see $G/export.out)"; return 1; }
    [ "$(export_left)" = "$left" ] && { log "STOP: the export made no progress (see $G/export.out)"; return 1; }
    log "export ticket: $(grep 'pass finished\|reviews/s' "$G/export.out" | tail -1 | cut -c1-200)"
  done
}

after_export() {  # CPU: status, verify (8 users recomputed on CPU), all 8,000 files vs the old targets, prepare the real cool-down
  [ -e "$G/after-export-ok" ] && return 0
  if [ "$DRYRUN" = 1 ]; then
    log "DRYRUN: export status / verify --sample 8 / $CHECK targets / prepare_decay $E35 $DEC (sm77_warm cooldown + label-window trim + check + Muon fresh) / $PAIR --arm $DEC --ref $DEC98 --targets"
    return 0
  fi
  cpu_step 25 "export checks and cool-down prepare" || return 1
  $LF "$PY" -X utf8 "$EXPORT" status --out "$REPLAY" --run "$RUN8K" --inputs e2s-query >> "$G/export.out" 2>&1 || { log "STOP: export status not complete"; return 1; }
  CUDA_VISIBLE_DEVICES= $LF "$PY" -X utf8 "$EXPORT" verify --out "$REPLAY" --run "$RUN8K" --teacher "$TEACHER" --inputs e2s-query --sample 8 >> "$G/export.out" 2>&1 \
    || { log "STOP: export verify failed (see $G/export.out, $REPLAY/verify.json)"; return 1; }
  log "export verify OK: $(tail -1 "$G/export.out" | cut -c1-200)"
  "$PY" -X utf8 "$CHECK" targets --out "$G/check-targets-sm110.json" >> "$LOG" 2>&1 || { log "STOP: the new targets are not the old targets' shape / NaN pattern / finite (see $G/check-targets-sm110.json)"; return 1; }
  cp "$REPLAY/manifest.json" "$G/replay-manifest.json"; cp "$REPLAY/status.json" "$G/replay-status.json" 2>/dev/null; cp "$REPLAY/verify.json" "$G/replay-verify.json"
  if [ ! -f "$DEC/decay.json" ]; then
    local X; X=$(cat "$S/extra-options.json")
    { $LF $WARM cooldown --epoch $K --start "$E35" --out "${DEC#$WT/}" --arch $ARCH --fraction 1.0 --extra-options "$X" >> "$G/prepare.out" 2>&1 \
      && $LF "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$DEC" --trim-unlabelled-tail > /dev/null \
      && $LF "$PY" -X utf8 "$WT/scripts/check_replay_label_windows.py" --decay "$DEC" --out "$G/check-replay-label-windows-$(basename "$DEC").json" > /dev/null; } \
      || { log "STOP: cool-down prepare / label windows failed"; rm -rf "$DEC"; return 1; }
  fi
  CUDA_VISIBLE_DEVICES= $LF "$PY" -X utf8 "$WT/scripts/check_muon_fresh_decay.py" --decay "$DEC" --out "$G/check-muon-fresh-$(basename "$DEC").json" \
    | grep -c PASS | sed "s/^/[sm110-muon] $(basename "$DEC") PASS lines: /" >> "$LOG"
  [ "${PIPESTATUS[0]}" = 0 ] || { log "STOP: Muon fresh-state CPU check failed"; return 1; }
  "$PY" -X utf8 "$PAIR" --arm "$DEC" --ref "$DEC98" --targets >> "$LOG" 2>&1 \
    || { log "STOP: the cool-down differs from SM98's cool-down beyond the targets (or the trims differ)"; return 1; }
  touch "$G/after-export-ok"
  return 0
}

mid_check() {  # score SM110's snapshot users-04000 (show-time, 723 users), then the mid rule
  [ "$DRYRUN" = 1 ] && { log "DRYRUN: mid-check: score $MID_NEW = $MIDSNAP/master.pth on e2s-query (723 users), then $PY -X utf8 $RULE mid"; return 0; }
  log "mid-check: SM110's snapshot users-04000 ($(head -c 300 "$MIDSNAP/snapshot.json" | tr -d '\n' | cut -c1-160))"
  score e2s-query "$MID_NEW" "$MIDSNAP/master.pth" "$MIDROWS" "732 the 732 set minus 9 never-fit (723 users)" "$USERS732" 723 1 $MIN_SCORE732 $TL732 || return 1
  rule mid "after 4,000 of 8,000 cool-down users" || return 1
  local d; d=$("$PY" -X utf8 -c "import json; print(json.load(open(r'$S/sm110-mid.json'))['decision'])")
  touch "$G/mid-done"
  [ "$d" = stop ] && { touch "$G/mid-stop"; log "mid-check says stop: the cool-down is not finished; verdict no go (stopped at the mid-check)"; }
  [ "$d" = continue ] && log "mid-check says continue: resuming the cool-down"
  return 0
}

cd "$WT" || exit 1

preconditions() {
  for f in "$S/check-cpu-sm110.json" "$S/extra-options.json" "$S/sm110_block.py" "$RULE" "$PAIR" "$CHECK" "$TEACHER" "$RUN8K" \
           "$E35/expertium/$TRACK/master.pth" "$DEC98/decay.json" "$SNAP98/master.pth" "$SNAP98/snapshot.json" "$USERS732" "$ULONG" \
           "$USERS1500" "$RECIPE/decay.json"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  grep -q '"all_ok": true' "$S/check-cpu-sm110.json" || { log "STOP: the CPU proofs are not ALL OK ($S/check-cpu-sm110.json)"; return 1; }
  done_in "$DEC98" || { log "STOP: SM98's cool-down is not complete ($DEC98)"; return 1; }
  if [ "$DRYRUN" != 1 ] && [ -n "$(git -C "$WT" status --porcelain -- scripts expertium rwkv)" ]; then
    log "STOP: uncommitted source changes in scripts/ expertium/ rwkv/ (SourceWatch refuses them)"; return 1; fi
  "$PY" -X utf8 "$PAIR" --arm "$RECIPE" --ref "$DEC98" --untrimmed >> "$LOG" 2>&1 || { log "STOP: the cool-down recipe is not SM98's cool-down except the targets"; return 1; }
  "$PY" -X utf8 "$RULE" selftest >> "$LOG" 2>&1 || { log "STOP: rule selftest failed"; return 1; }
  local sha
  sha=$(sha256sum "$TEACHER" | cut -d' ' -f1); [ "$sha" = "$TEACHER_SHA" ] || { log "STOP: the teacher changed (sha256 $sha)"; return 1; }
  sha=$(sha256sum "$E35/expertium/$TRACK/master.pth" | cut -d' ' -f1); [ "$sha" = "$E35_SHA" ] || { log "STOP: SM98's epoch 35 changed (sha256 $sha)"; return 1; }
  return 0
}

log "started: $(git -C "$WT" log --oneline -1) (DRYRUN=$DRYRUN; SM99 show-time targets for the 8,000 users -> $REPLAY; SM98's cool-down from its epoch 35 with only the targets swapped -> $DEC; mid-check at users-04000)"
rc=0
cpu_step 5 "preparation" || rc=1
[ $rc -eq 0 ] && { preconditions || rc=1; }
# 1. SM98's own cool-down snapshot after 4,000 users, show-time inputs (the mid-check's reference; also proves a snapshot scores)
[ $rc -eq 0 ] && { score e2s-query "$MID_REF" "$SNAP98/master.pth" "$MIDROWS" "732 the 732 set minus 9 never-fit (723 users)" "$USERS732" 723 1 $MIN_SCORE732 $TL732 || rc=1; }
# 2. the teacher targets (GPU), then the CPU checks and the real cool-down folder
[ $rc -eq 0 ] && { run_export || rc=1; }
[ $rc -eq 0 ] && { after_export || rc=1; }
# 3. smoke, then the cool-down with the mid-check
if [ $rc -eq 0 ] && [ ! -e "$G/smoke-dec-ok" ] && ! done_in "$DEC"; then
  ram_gate "smoke" strict || rc=1
  [ $rc -eq 0 ] && { q $MIN_SMOKE check "trainer smoke of the cool-down with SM99's show-time targets (fresh Muon, curve loss weight 0, show-time inputs, 2 users x 512 reviews)" -- \
    $LF "$PY" -X utf8 "$DEC_PY" smoke --out "$DEC" >> "$G/check.out" 2>&1 || { log "STOP: cool-down smoke failed (see $G/check.out)"; rc=1; }; }
  [ $rc -eq 0 ] && [ "$DRYRUN" != 1 ] && { touch "$G/smoke-dec-ok"; log "smoke: $(grep '"smoke"' "$G/check.out" | tail -1 | cut -c1-200)"; }
fi
WHAT="the release's one-epoch 32k Muon cool-down on show-time inputs from SM98's epoch 35 with SM99's show-time targets, curve loss weight 0"
while [ $rc -eq 0 ] && [ ! -e "$G/mid-stop" ]; do
  run_folder "$DEC" "$WHAT" $MIN_DECAY; f=$?
  if [ $f = 10 ]; then mid_check || rc=1; continue; fi
  [ $f = 0 ] || rc=1
  [ "$DRYRUN" = 1 ] && mid_check
  break
done
if [ $rc -eq 0 ] && [ -e "$G/mid-stop" ]; then
  log "done (verdict no go: stopped at the mid-check, $(cut -c1-300 "$S/sm110-mid.md" 2>/dev/null | grep -m1 Decision | cut -c1-200))"
  exit 0
fi
# 4. final scoring and the rule
[ $rc -eq 0 ] && { score732 e2s-query || rc=1; }
[ $rc -eq 0 ] && { rule decide "show-time 732" || rc=1; }
[ $rc -eq 0 ] && { score732 normal || rc=1; }
[ $rc -eq 0 ] && { score e2s-query "$CLBL" "$DEC/expertium/$TRACK/master.pth" "$ROWS1500" "1500 the 1,500 validation users outside the 732 set + 6 check users" "$USERS1500" 771 0 $MIN_POOL $TLPOOL \
  || log "the 1,500-pool scoring failed; going on (secondary only)"; }
[ $rc -eq 0 ] && { rule decide "final" || rc=1; }
if [ $rc -eq 0 ]; then
  if [ "$DRYRUN" = 1 ]; then log "DRYRUN OK"; else log "done (verdict $("$PY" -X utf8 -c "import json; print(json.load(open(r'$S/sm110-decision.json')).get('verdict'))" 2>/dev/null))"; fi
else
  log "STOP: ended with exit $rc"
fi
exit $rc
