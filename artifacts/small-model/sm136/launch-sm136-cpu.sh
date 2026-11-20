#!/usr/bin/env bash
# Long-card state drift on real users (SM136), CPU scoring (design.md amendment 11-20 03:00): the GPU smoke disagreed with
# the CPU rows on long-card buckets (amplified kernel rounding), so the same design, rule, pool, models and formats are
# scored on the CPU, which mirrors the fork. Same scorer command line as sm136_ticket.sh, plus --device cpu.
# Every user is scored in its own fresh process (one stage = 2 models x 5 formats), 2 workers in parallel (SM135's
# training job holds ~11 GB RAM), users from the smallest to the largest review count (cpu/order.json). Memory guard 5.9 GB
# per scorer process (memguard.ps1); a user it kills goes to cpu/oom.txt (the run goes on, then STOPs before the rule).
# Stages run in the order of cpu/STAGES (default "tiny small big ext"; edit the file while running to cut, read at every
# stage start). ext = the extension pool, report only. Quiet windows: quiet.json scope all/cpu/gpu ONLY (sm136_block.py
# --scopes all,cpu,gpu); no daily or Sunday break. The GPU queue is never used.
#   launch  WMI hidden (HANDOFF.md); log artifacts/small-model/sm136/cpu/launch-sm136-cpu.log
#   resume  the launch command again (complete users are skipped; claims are cleared at start)
#   stop    touch cpu/STOP2 (workers end after their current user)
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
WT=C:/Users/jonat/workspace/wt/sm136
S=$WT/artifacts/small-model/sm136
C=$S/cpu
G=$S/gpu
R=$C/rows
LOG=$C/launch-sm136-cpu.log
SC=$WT/scripts/small_model_heldout.py
RULE=$S/sm136_rule.py
W=C:/Users/jonat/workspace/wt
TRACK=E89-F65e-F77a
CK91=$W/sm91s2/artifacts/small-model/sm91s2-big-showtime/expertium/big-showtime/master.pth
CK99=$W/sm99/artifacts/small-model/sm99-big-inst-e2/expertium/big-inst/master.pth
CK118=$W/sm118/artifacts/small-model/sm118-e26-dec2ep32k-t91s2-p2/expertium/$TRACK/master.pth
CK113=$W/sm113/artifacts/small-model/sm113-inst-e35st-st-dec2ep32k-t99-p2/expertium/$TRACK/master.pth
CK124=$W/sm124/artifacts/small-model/sm124-c1-e29st-dec2ep32k-t91s2-p2/expertium/$TRACK/master.pth
CK115=$W/sm115/artifacts/small-model/sm115-c1-e38st-st-dec2ep32k-t99-p2/expertium/$TRACK/master.pth
SHA91=67ee8e9b0d0169733242c40646e0d2e5637845a25775762002ffc4e6396f9590
SHA99=c49e45d698194da033caa1231f598f875922a4a2a3e921714a3fd433d37e5768
SHA118=b4fe1175d25a7e6abff70e87ac0af1f4d76db9622d54f7cb53196fbf35117eb6
SHA113=e4574e2aa8b2cfe6360dbd86f2a75b71d23ea81677b00ffcc9c3abce6b4cb736
SHA124=c78b24d8ee187b58fd8c23c90476fb68a1aab88a137cc1f6a64c03530b7dcb3c
SHA115=c59d70495f7a81541d3a70329313354b5f0ec07c9fdd98ced58f849c5890fb0a
T91=sm91s2-big-showtime; T99=sm99-big-inst-e2; T118=sm118-e26-dec2ep32k-t91s2; T113=sm113-inst-e35st-st-dec2ep32k-t99
T124=sm124-c1-e29st-dec2ep32k-t91s2; T115=sm115-c1-e38st-st-dec2ep32k-t99
FORMATS="w8+s8 w8+s4c8n w8+s4c8nhx8 w8+s8hx8 w8"
NW=2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=
unset HELDOUT_GPU_MEMORY_FRACTION SM11_GPU_MEMORY_FRACTION RWKV_DELETED_CARDS EXPERTIUM_END_TO_START SM11_INPUTS
mkdir -p "$R"
log() { echo "[sm136 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
free_mib() { awk '/MemFree/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 99999; }
block() { "$PY" -X utf8 "$S/sm136_block.py" --scopes all,cpu,gpu; }

stage_models() {  # models of a stage as --model args, and its label tags
  case $1 in
    big)   MODELS="--model $T91=fork::$CK91 --model $T99=fork::$CK99"; TAGS="$T91 $T99";;
    small) MODELS="--model $T118=expertium:w64:$CK118 --model $T113=expertium:w64:$CK113"; TAGS="$T118 $T113";;
    tiny)  MODELS="--model $T124=expertium:w64-c1:$CK124 --model $T115=expertium:w64-c1:$CK115"; TAGS="$T124 $T115";;
    ext)   MODELS="--model $T91=fork::$CK91 --model $T99=fork::$CK99 --model $T118=expertium:w64:$CK118 --model $T113=expertium:w64:$CK113 --model $T124=expertium:w64-c1:$CK124 --model $T115=expertium:w64-c1:$CK115"
           TAGS="$T91 $T99 $T118 $T113 $T124 $T115";;
  esac
  ONLY=""; for t in $TAGS; do for f in $FORMATS; do ONLY="$ONLY --only $t/$f"; done; done
}
est_ms() { case $1 in tiny) echo 9;; small) echo 10;; big) echo 16;; ext) echo 40;; esac; }  # ms per review incl. margin

remaining() {  # remaining STAGE: "user reviews" lines of the stage's users that are not complete, smallest first
  local st=$1; stage_models "$st"
  ST=$st D="$R/$st" ORDER="$C/order.json" TAGS="$TAGS" FORMATS="$FORMATS" OOM="$C/oom.txt" "$PY" -X utf8 -c "
import json, os
from pathlib import Path
st, d = os.environ['ST'], Path(os.environ['D'])
order = json.load(open(os.environ['ORDER']))['ext' if st == 'ext' else 'main']
want = {t + '/' + f + '@e2s-query' for t in os.environ['TAGS'].split() for f in os.environ['FORMATS'].split()}
oom = set()
try:
    oom = {l.split()[0] + ' ' + l.split()[1] for l in open(os.environ['OOM']) if l.strip()}
except OSError:
    pass
for n, u in order:
    got = set()
    p = d / f'u-{u}.jsonl'
    if p.exists():
        for raw in open(p, 'rb'):
            try:
                r = json.loads(raw.decode('utf-8', errors='replace'))
            except ValueError:
                continue
            if 'error' not in r and ('benchmark' in r or 'nonfinite' in r):
                got.add(r.get('label'))
    if want <= got or f'{st} {u}' in oom:
        continue
    print(u, n)
"
}

one() {  # one STAGE USER: score it in a fresh process (own users file and output file)
  local st=$1 u=$2 pool=732 d="$R/$1"
  [ "$st" = ext ] && pool=1500
  stage_models "$st"
  mkdir -p "$d"
  echo "{\"users\": [$u]}" > "$d/users-$u.json"
  rm -f "$d/u-$u.jsonl"
  "$PY" -X utf8 -W ignore "$SC" --device cpu --inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows \
    --prior-buckets 0,32,128,300,1000 --pool $pool --users-file "$d/users-$u.json" $MODELS $ONLY --time-limit 1000000 \
    --out "$d/u-$u.jsonl" >> "$d/score-$u.out" 2>&1
  local rc=$?
  if [ $rc != 0 ]; then
    if grep -q "u-$u.jsonl" "$C/memguard-kill" 2>/dev/null; then echo "$st $u killed by the memory guard" >> "$C/oom.txt"; log "$st user $u killed by the memory guard"
    else echo "$st $u exit $rc" >> "$C/failed.txt"; log "$st user $u failed (exit $rc)"; fi
    rm -f "$d/u-$u.jsonl"
  else
    grep -a "^\[small-model\] 1/1" "$d/score-$u.out" | tail -1 | sed "s/^/$st /" >> "$C/progress.out"
    log "$st user $u done: $(grep -a '^\[small-model\] 1/1' "$d/score-$u.out" | tail -1 | cut -c1-120)"
  fi
}

worker() {  # worker STAGE USER:REVIEWS...: take the next unclaimed user in order
  local st=$1; shift; local ms; ms=$(est_ms "$st")
  for ur in "$@"; do
    local u=${ur%%:*} n=${ur##*:}
    [ -e "$C/STOP2" ] && return
    mkdir "$C/claim/$st-$u" 2>/dev/null || continue
    local need=$(( n * ms / 60000 + 5 )) k=0
    while true; do   # not inside a quiet window, and the user must fit before the next one
      [ -e "$C/STOP2" ] && return
      local b; b=$(block)
      case "$b" in
        NOW*) [ $((k % 6)) = 0 ] && log "worker: quiet window, ${b#NOW } s left"; k=$((k + 1)); sleep 300;;
        NEXT*) [ "${b#NEXT }" -ge "$need" ] && break; [ $((k % 6)) = 0 ] && log "worker: user $u needs ~$need min, ${b#NEXT } min to the next quiet window; waiting"; k=$((k + 1)); sleep 300;;
        *) sleep 300;;
      esac
    done
    k=0
    while [ "$(free_mib)" -lt 3072 ] && [ $k -lt 30 ]; do [ $k = 0 ] && log "$(free_mib) MiB RAM free (< 3 GB), holding $st user $u"; k=$((k + 1)); sleep 60; done
    one "$st" "$u"
  done
}

merge() {
  : > "$G/sm136-rows.jsonl.tmp"; : > "$G/sm136-ext-rows.jsonl.tmp"
  for st in tiny small big; do cat "$R/$st"/u-*.jsonl >> "$G/sm136-rows.jsonl.tmp" 2>/dev/null; done
  cat "$R/ext"/u-*.jsonl >> "$G/sm136-ext-rows.jsonl.tmp" 2>/dev/null
  mv "$G/sm136-rows.jsonl.tmp" "$G/sm136-rows.jsonl"; mv "$G/sm136-ext-rows.jsonl.tmp" "$G/sm136-ext-rows.jsonl"
}

preconditions() {
  for f in "$S/sm136_block.py" "$RULE" "$S/sm136_pool.py" "$S/users-sm136.json" "$S/users-sm136-ext.json" "$C/order.json" \
           "$CK91" "$CK99" "$CK118" "$CK113" "$CK124" "$CK115"; do
    [ -f "$f" ] || { log "STOP: missing $f"; return 1; }
  done
  local sha pair
  for pair in "$CK91 $SHA91" "$CK99 $SHA99" "$CK118 $SHA118" "$CK113 $SHA113" "$CK124 $SHA124" "$CK115 $SHA115"; do
    set -- $pair
    sha=$(sha256sum "$1" | cut -d' ' -f1); [ "$sha" = "$2" ] || { log "STOP: $1 changed (sha256 $sha)"; return 1; }
  done
  "$PY" -X utf8 "$RULE" selftest >> "$LOG" 2>&1 || { log "STOP: rule selftest failed"; return 1; }
}

cd "$WT" || exit 1
log "started CPU scoring: $(git -C "$WT" log --oneline -1); $NW workers, free RAM $(free_mib) MiB; quiet.json windows only; formats $FORMATS"
preconditions || { log "STOP: preconditions failed"; exit 1; }
rm -rf "$C/claim"; mkdir -p "$C/claim"
rm -f "$C/STOP" "$C/memguard-kill" "$C/memguard.off"
[ -f "$C/STAGES" ] || echo "tiny small big ext" > "$C/STAGES"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$S/memguard.ps1")" -Dir "$(cygpath -w "$C")" \
  -Match "small_model_heldout.py&--device&cpu" > /dev/null 2>&1 &
MG=$!
for st in $(cat "$C/STAGES"); do
  [ -e "$C/STOP2" ] && break
  users=$(remaining "$st" | awk '{print $1 ":" $2}')
  n=$(echo $users | wc -w)
  log "stage $st: $n users left (smallest first), $NW workers"
  rm -rf "$C/claim"; mkdir -p "$C/claim"
  pids=""
  for i in $(seq 1 $NW); do worker "$st" $users & pids="$pids $!"; sleep 2; done
  wait $pids
  merge
  log "stage $st ended: $(remaining "$st" | wc -l) users left"
done
touch "$C/memguard.off"; sleep 12; kill $MG 2>/dev/null; rm -f "$C/STOP"
merge
[ -e "$C/STOP2" ] && { log "STOP: STOP2 file present"; exit 1; }
oom=$(cat "$C/oom.txt" 2>/dev/null | wc -l); failed=$(cat "$C/failed.txt" 2>/dev/null | wc -l)
[ "$oom" != 0 ] && { log "STOP: memory guard killed fresh single-user processes: $(tr '\n' ';' < "$C/oom.txt")"; exit 1; }
[ "$failed" != 0 ] && { log "STOP: scorer failures: $(tr '\n' ';' < "$C/failed.txt")"; exit 1; }
out=$("$PY" -X utf8 "$RULE" decide 2>&1); r=$?
log "rule decide exit $r: $(echo "$out" | tail -1 | cut -c1-700)"
if [ "$r" = 0 ]; then
  log "done (verdict $("$PY" -X utf8 -c "import json; v=json.load(open(r'$S/sm136-decision.json'))['verdict']; print('; '.join(k + ': ' + x for k, x in v.items()))"))"
elif [ "$r" = 2 ]; then
  log "STOP: the rule is pending (fewer than 64 of 66 main-pool users for a model and format; see the line above)"; exit 1
else
  log "STOP: the rule did not decide (exit $r)"; exit 1
fi
exit 0
