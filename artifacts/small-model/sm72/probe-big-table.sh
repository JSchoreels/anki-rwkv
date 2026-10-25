#!/usr/bin/env bash
# SM72 probe after the first launch's OOM (user 9974): budgeted carry table (state_carry table_gpu_mib).
# Ticket 1: the 4-user 131k probe at 2 x 8,192 three ways (no budget = before; budget 0 = every table in host memory;
#           budget 768 MiB): per-update loss and gradient norm must match bit for bit.
# Ticket 2: the worst users at budget 768: 9974 (OOM'd), 6433 (2.75 GB table), 8431 cut to 500k, 8902 cut to 700k.
export PATH=/usr/bin:/mingw64/bin:$PATH RWKV_DELETED_CARDS=neighbour
W=C:/Users/jonat/workspace/wt/sm72; A=$W/artifacts/small-model; PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
QUEUE=C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py; LOG=$A/sm72/probe-big-table.log
"$PY" "$QUEUE" run --owner small-model --purpose "small-model: SM72 big-model carry-table probe 1/2: 4 users x 131k, no budget vs host vs 768 MiB (bit-for-bit)" \
  --claim gpu --expected-minutes 7 --max-wait-minutes 60 --launched-by "sm72/probe-big-table.sh" -- \
  bash -c "for n in ctl host gpu; do $PY -X utf8 $W/scripts/sm72_big_epoch.py run --out $A/sm72-big-probe-tbl-\$n; echo \"probe \$n exit \$?\"; done" >> "$LOG" 2>&1
echo "ticket 1 exit $?" >> "$LOG"
"$PY" "$QUEUE" run --owner small-model --purpose "small-model: SM72 big-model carry-table probe 2/2: worst users (9974, 6433, 8431 500k, 8902 700k) at 768 MiB" \
  --claim gpu --expected-minutes 9 --max-wait-minutes 60 --launched-by "sm72/probe-big-table.sh" -- \
  $PY -X utf8 $W/scripts/sm72_big_epoch.py run --out $A/sm72-big-probe-worst >> "$LOG" 2>&1
echo "ticket 2 exit $?" >> "$LOG"
