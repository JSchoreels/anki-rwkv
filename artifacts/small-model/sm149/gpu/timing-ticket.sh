#!/usr/bin/env bash
# SM149 design: real-data throughput of both arms of Small two-head (SM118) on scratch copies of the prepared folders
# (sm149-timing-qat / -ctl in the run tree, deleted after the read-out). Each arm trains ~150 s, then a snapshot stop.
R=C:/Users/jonat/workspace/wt/sm149-run
S=C:/Users/jonat/workspace/wt/sm149/artifacts/small-model/sm149
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe
cd $R
for arm in ${ARMS:-qat ctl}; do
  env RWKV_DELETED_CARDS=neighbour SM149_CODE=$R $PY -X utf8 $S/sm149_run.py run --out artifacts/small-model/sm149-timing-$arm &
  pid=$!
  $PY -c "import time; time.sleep(${SECONDS_PER_ARM:-150})"
  env SM149_CODE=$R $PY -X utf8 $S/sm149_run.py stop --out artifacts/small-model/sm149-timing-$arm
  wait $pid
  echo "arm $arm exit $?"
done
