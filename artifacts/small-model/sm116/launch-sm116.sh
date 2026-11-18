#!/bin/bash
# Training-free state check (SM116): the one pre-registered score (design.md). CPU only, 3 processes (2 torch threads
# each, < 6 GB RAM each), the users of users-score.json (629: the 722-user rule pool with <= 125k reviews) split into 3 fixed slices of the scorer's stratified order.
# Each pass stops starting new users 25 min before the next quiet window (quiet.json scope all/gpu, the daily
# 20:00-01:00 break, Sunday from 16:00); the launcher sleeps through the window and resumes (scored users are kept).
# When all 722 users have rows: sm116_rule.py decide -> sm116-decision.json / .md, log line "done (verdict ...)".
# Memory guard (memguard.ps1): a scorer above 5.9 GB is killed and the launcher stops (STOP).
# Stop: touch rows/STOP (checked between passes; a running pass ends at its time limit).
cd /c/Users/jonat/workspace/wt/sm116 || exit 1
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
M=C:/Users/jonat/workspace/wt/sm92/artifacts/small-model/sm92-e26-dec-1ep32k-muon/expertium/E89-F65e-F77a/master.pth
D=artifacts/small-model/sm116
G=$D/rows
LOG=$G/launch-sm116.log
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
mkdir -p $G
log() { echo "[sm116 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
nap() { local s=$1; [ "$s" -gt 600 ] && s=600; sleep "$s"; }

block() {  # "NOW <seconds to its end>" inside a quiet window, else "NEXT <minutes>" (the SM128 launcher's check)
  Q="$QUIET" "$PY" -X utf8 - <<'PYEOF'
import json, os, re
from datetime import datetime, timedelta
now = datetime.now()
try:
    text = open(os.environ["Q"], encoding="utf-8").read()
except OSError as error:
    print(f"ERR quiet.json unreadable: {error}"); raise SystemExit
try:
    wins = [(w["start"], w["end"]) for w in json.loads(text) if w.get("scope", "all") in ("all", "gpu")]
except (ValueError, KeyError, TypeError):
    wins = re.findall(r'"start"\s*:\s*"([^"]+)"[^{}]*?"end"\s*:\s*"([^"]+)"', text, re.S)
blocks = [(datetime.fromisoformat(s), datetime.fromisoformat(e)) for s, e in wins]
day = now.replace(hour=0, minute=0, second=0, microsecond=0)
blocks += [(day + timedelta(days=d, hours=20), day + timedelta(days=d + 1, hours=1)) for d in range(-1, 9)]
blocks += [(day + timedelta(days=d, hours=16), day + timedelta(days=d, hours=20))
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

missing() { "$PY" -X utf8 -c "
import json
users = json.load(open('$D/users-score.json'))['users']
done = set()
try:
    for l in open('$G/sm116-k70.jsonl'):
        r = json.loads(l)
        if r.get('label') == 'sm116k70/w8+s8@e2s-query' and 'error' not in r: done.add(r['user_id'])
except FileNotFoundError: pass
print(len([u for u in users if u not in done]))"; }

part() {  # part NAME START COUNT LIMIT_SECONDS
  SM116_BASIS=$D/work/basis.npz SM116_KEEP=0.7 env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START \
    "$PY" -X utf8 $D/sm116_state.py score --device cpu --pool 732 --users-file $D/users-score.json --start $2 --count $3 \
    --model sm116k70=expertium:w64:$M --only sm116k70/w8+s8 --inputs e2s-query --split-existing --deleted-cards neighbour \
    --time-limit $4 --out $G/sm116-k70-$1.jsonl >> $G/score-$1.out 2>&1
  echo "$?" > $G/score-$1.rc
}

N=$("$PY" -X utf8 -c "import json; print(len(json.load(open('$D/users-score.json'))['users']))")
log "start (commit $(git rev-parse --short HEAD), $(missing) of $N users to score)"
[ -f $D/work/basis.npz ] || { log "STOP: $D/work/basis.npz missing"; exit 1; }
pass=0
while true; do
  [ -e $G/STOP ] && { log "STOP: STOP file present"; exit 1; }
  left=$(missing)
  [ "$left" = 0 ] && break
  b=$(block)
  case "$b" in
    NOW*) s=${b#NOW }; log "quiet window, ${s}s left; $left users to score"; nap "$s"; continue;;
    NEXT*) free=${b#NEXT }
           if [ "$free" -lt 40 ]; then log "$free min to the next window; waiting"; nap $(( free * 60 + 60 )); continue; fi;;
    *) log "window check failed: $b"; nap 600; continue;;
  esac
  limit=$(( (free - 25) * 60 ))
  pass=$((pass + 1))
  log "pass $pass: $left users to score, new users start for ${limit}s"
  rm -f $G/memguard.off
  powershell -NoProfile -ExecutionPolicy Bypass -File C:/Users/jonat/workspace/wt/sm116/$D/memguard.ps1 &
  part a 0 210 $limit & pa=$!; part b 210 210 $limit & pb=$!; part c 420 209 $limit & pc=$!
  wait $pa $pb $pc
  touch $G/memguard.off
  cat $G/sm116-k70-a.jsonl $G/sm116-k70-b.jsonl $G/sm116-k70-c.jsonl > $G/sm116-k70.jsonl 2>/dev/null
  log "pass $pass ended: exit codes $(cat $G/score-a.rc) $(cat $G/score-b.rc) $(cat $G/score-c.rc); $(missing) users left"
  [ -e $G/memguard-kill ] && { log "STOP: memory guard killed a scorer: $(cat $G/memguard-kill)"; exit 1; }
  if [ "$(cat $G/score-a.rc)$(cat $G/score-b.rc)$(cat $G/score-c.rc)" != "000" ]; then
    log "STOP: a scorer part failed (see $G/score-{a,b,c}.out)"; exit 1
  fi
  errs=$(grep -c '"error"' $G/sm116-k70.jsonl)
  [ "$errs" != 0 ] && { log "STOP: $errs error rows in $G/sm116-k70.jsonl"; exit 1; }
done
"$PY" -X utf8 $D/sm116_rule.py decide >> $G/decide.out 2>&1; rc=$?
[ $rc = 0 ] || { log "STOP: sm116_rule.py decide exit $rc"; exit 1; }
log "done (verdict $("$PY" -X utf8 -c "import json; print(json.load(open('$D/sm116-decision.json'))['verdict'])"))"
