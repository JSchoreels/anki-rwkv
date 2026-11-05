#!/usr/bin/env bash
set -u
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
cd /c/Users/jonat/workspace/wt/sm41
S=/c/Users/jonat/AppData/Local/Temp/claude/eefeeb6d2df69e2b/41a2947b-8da0-4ee6-a033-bc96eb42a697/scratchpad/prof/fuse
F=artifacts/small-model/sm41/fullrun/fits/3992.jsonl
B="ihr-0.90#0 fsrs7-0.90#0 ihr-0.90-s2#0 fsrs7-0.90-free#0"
sub() { local g=$1 out=tf; for l in $B; do n=${l%%#*}; s=${l#*#}; if [ -n "$g" ]; then case $n in *-free) n="${n%-free}-$g-free";; *) n="$n-$g";; esac; fi; out="$out,$n#$s"; done; echo "$out"; }
L=$(sub "")
FUSED="$L,$(sub g0.5_0.2 | sed 's/^tf,//'),$(sub gm0.5_m0.2 | sed 's/^tf,//')"
echo "$FUSED" > $S/fused_lanes.txt
job() { local name=$1 eng=$2; shift 2; mkdir -p $S/$name; "$PY" -X utf8 scripts/$eng.py run --user 3992 --split 840 --days 60 --ivl-hist --fsrs7-fits $F --min-free-gb 1 --log-dir $S/$name --out $S/$name/row.jsonl "$@" > $S/$name/out.log 2>&1; echo "$name exit $? $(date +%T)" >> $S/status; }
job ref_imp sm41_engine --lanes "$L" --drift 0.5,0.2 & job dev_imp sm41_engine_dev --lanes "$L" --drift 0.5,0.2 & wait
job ref_dec sm41_engine --lanes "$L" --drift=-0.5,-0.2 & job ref_none sm41_engine --lanes "$L" & wait
job fused sm41_engine_dev --lanes "$FUSED"
echo "all done $(date +%T)" >> $S/status
