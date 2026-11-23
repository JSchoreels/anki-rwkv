#!/usr/bin/env bash
# Rank-2 states with a tail-energy penalty (SM149b): GPU smoke, one queue ticket (< 10 min, cap 0.8 from the options).
# 1. the trainer's smoke of the prepared arm sm149b-sm118-tail1 (2 users x 512 reviews, scratch copy);
# 2. real-data throughput: a scratch copy of the prepared folder (sm149b-timing-tail1 in the run tree) trains
#    ${SECONDS_TRAIN:-270} s, then a snapshot stop; GPU memory sampled by nvidia-smi every 5 s. The first updates are the
#    same users and windows as SM149's QAT arm (its updates.jsonl), so the read-out (smoke_readout.py) is paired.
set -u
R=C:/Users/jonat/workspace/wt/sm149-run
S=C:/Users/jonat/workspace/wt/sm149b/artifacts/small-model/sm149b
G=$S/gpu
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
cd $R
env RWKV_DELETED_CARDS=neighbour SM149_CODE=$R $PY -X utf8 $S/sm149b_run.py smoke --out artifacts/small-model/sm149b-sm118-tail1 \
  > $G/smoke-trainer.out 2>&1
echo "trainer smoke exit $?"
T=artifacts/small-model/sm149b-timing-tail1
rm -rf $T && mkdir -p $T && cp artifacts/small-model/sm149b-sm118-tail1/decay.json artifacts/small-model/sm149b-sm118-tail1/protocol.json $T/
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 5 > $G/timing-nvsmi.csv 2>&1 &
smi=$!
env RWKV_DELETED_CARDS=neighbour SM149_CODE=$R $PY -X utf8 $S/sm149b_run.py run --out $T > $G/timing-tail1.out 2>&1 &
pid=$!
$PY -c "import time; time.sleep(${SECONDS_TRAIN:-270})"
env SM149_CODE=$R $PY -X utf8 $S/sm149b_run.py stop --out $T
wait $pid
echo "timing exit $?"
kill $smi 2>/dev/null
exit 0
