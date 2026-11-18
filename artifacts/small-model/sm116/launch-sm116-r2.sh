#!/bin/bash
# Training-free state check (SM116), pass 2 (added 11-18 17:30 after pass 1's memory-guard stop): the scorer's memory
# only grows within a process (high-water mark of the largest user so far), so two of the three long-lived scorers
# crossed 5.9 GB. Same score, same pre-registered users, same command line; only the process layout changes:
# every remaining user is scored in its own fresh process, 3 workers in parallel, memory guard unchanged (5.9 GB).
# A user killed by the guard in a fresh process is listed in rows/r2/oom.txt and the run stops before the rule.
# Stops starting users 25 min before a quiet window and resumes after it. Stop: touch rows/STOP2.
cd /c/Users/jonat/workspace/wt/sm116 || exit 1
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
M=C:/Users/jonat/workspace/wt/sm92/artifacts/small-model/sm92-e26-dec-1ep32k-muon/expertium/E89-F65e-F77a/master.pth
D=artifacts/small-model/sm116
G=$D/rows
R=$G/r2
LOG=$G/launch-sm116.log
QUIET=C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json
mkdir -p $R
log() { echo "[sm116 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

block() {  # "NOW <seconds>" inside a quiet window, else "NEXT <minutes>" (same check as launch-sm116.sh)
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
for s, e in blocks:
    if s <= now < e:
        print(f"NOW {int((e - now).total_seconds()) + 60}"); raise SystemExit
    if s > now:
        print(f"NEXT {int((s - now).total_seconds() // 60)}"); raise SystemExit
print("NEXT 100000")
PYEOF
}

merge() { cat $G/sm116-k70-a.jsonl $G/sm116-k70-b.jsonl $G/sm116-k70-c.jsonl $R/u-*.jsonl > $G/sm116-k70.jsonl 2>/dev/null; }

remaining() { "$PY" -X utf8 -c "
import json
users = json.load(open('$D/users-score.json'))['users']
done = set()
for l in open('$G/sm116-k70.jsonl'):
    r = json.loads(l)
    if r.get('label') == 'sm116k70/w8+s8@e2s-query' and 'error' not in r: done.add(r['user_id'])
oom = set()
try: oom = {int(x.split()[0]) for x in open('$R/oom.txt') if x.strip()}
except FileNotFoundError: pass
print(' '.join(str(u) for u in users if u not in done and u not in oom))"; }

one() {  # one USER: score it in a fresh process
  local u=$1
  echo "{\"users\": [$u]}" > $R/users-$u.json
  rm -f $R/u-$u.jsonl
  SM116_BASIS=$D/work/basis.npz SM116_KEEP=0.7 env -u RWKV_DELETED_CARDS -u EXPERTIUM_END_TO_START \
    "$PY" -X utf8 $D/sm116_state.py score --device cpu --pool 732 --users-file $R/users-$u.json --start 0 --count 1 \
    --model sm116k70=expertium:w64:$M --only sm116k70/w8+s8 --inputs e2s-query --split-existing --deleted-cards neighbour \
    --time-limit 100000 --out $R/u-$u.jsonl >> $R/score-$u.out 2>&1
  local rc=$?
  if [ $rc != 0 ]; then
    if grep -q "u-$u.jsonl" $G/memguard-kill 2>/dev/null; then echo "$u killed by the memory guard" >> $R/oom.txt
    else echo "$u exit $rc" >> $R/failed.txt; fi
    rm -f $R/u-$u.jsonl
  fi
  grep "^\[small-model\] 1/1" $R/score-$u.out | tail -1 >> $R/progress.out
}

worker() {  # worker INDEX USERS...: users i with i % 3 == INDEX
  local w=$1; shift; local i=0
  for u in "$@"; do
    if [ $((i % 3)) = $w ]; then
      while true; do
        [ -e $G/STOP2 ] && return
        b=$(block)
        case "$b" in
          NOW*) sleep 600;;
          NEXT*) [ "${b#NEXT }" -lt 25 ] && sleep 600 || break;;
          *) sleep 600;;
        esac
      done
      one $u
    fi
    i=$((i + 1))
  done
}

rm -f $G/STOP $G/memguard-kill $G/memguard.off
merge
users=$(remaining)
n=$(echo $users | wc -w)
log "pass 2 (launch-sm116-r2.sh): $n users left, each in its own fresh process, 3 at a time (pass 1's long-lived scorers grew past 5.9 GB)"
powershell -NoProfile -ExecutionPolicy Bypass -File C:/Users/jonat/workspace/wt/sm116/$D/memguard.ps1 &
worker 0 $users & p0=$!; worker 1 $users & p1=$!; worker 2 $users & p2=$!
wait $p0 $p1 $p2
touch $G/memguard.off
rm -f $G/STOP
merge
[ -e $G/STOP2 ] && { log "STOP: STOP2 file present"; exit 1; }
left=$(remaining | wc -w)
oom=$(cat $R/oom.txt 2>/dev/null | wc -l); failed=$(cat $R/failed.txt 2>/dev/null | wc -l)
log "pass 2 ended: $left users left, $oom killed by the memory guard in a fresh process, $failed failed otherwise"
[ "$oom" != 0 ] && { log "STOP: memory guard killed fresh single-user processes: $(tr '\n' ';' < $R/oom.txt)"; exit 1; }
[ "$failed" != 0 ] || [ "$left" != 0 ] && { log "STOP: scorer failures ($(tr '\n' ';' < $R/failed.txt 2>/dev/null)), $left left"; exit 1; }
errs=$(grep -c '"error"' $G/sm116-k70.jsonl)
[ "$errs" != 0 ] && { log "STOP: $errs error rows in $G/sm116-k70.jsonl"; exit 1; }
"$PY" -X utf8 $D/sm116_rule.py decide >> $G/decide.out 2>&1; rc=$?
[ $rc = 0 ] || { log "STOP: sm116_rule.py decide exit $rc"; exit 1; }
log "done (verdict $("$PY" -X utf8 -c "import json; print(json.load(open('$D/sm116-decision.json'))['verdict'])"))"
