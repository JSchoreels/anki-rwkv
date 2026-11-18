#!/usr/bin/env bash
# Starts the SM26 runbook once the SM32b QAT GPU ticket has exited ("exit N: SM32b QAT decay" or a runbook stop), so
# SM26 trains on the GPU while the QAT run's CPU scoring (users34) goes on.
QLOG=/c/Users/jonat/workspace/wt/state-vq/artifacts/small-model/runbook-2026-10-18-qat.log
LOG=/c/Users/jonat/workspace/wt/small-model/artifacts/small-model/runbook-2026-10-18-sm26.log
until grep -qE "exit [0-9]+: SM32b QAT decay|; stop|failed|refuse|REFUSE" $QLOG 2>/dev/null; do sleep 120; done
echo "[launcher $(date '+%m-%d %H:%M:%S')] QAT GPU ticket over: $(grep -hE 'exit [0-9]+: SM32b|; stop|failed|REFUSE' $QLOG | tail -1); starting SM26" >> $LOG
bash /c/Users/jonat/workspace/wt/small-model/artifacts/small-model/runbook-2026-10-18-sm26.sh >> $LOG 2>&1
echo "[launcher $(date '+%m-%d %H:%M:%S')] SM26 runbook exit $?" >> $LOG
