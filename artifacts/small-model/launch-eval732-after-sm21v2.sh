#!/usr/bin/env bash
# 2026-10-18 (user: 732 users as the standard readout): score the 268 new users of the 732 pool after the SM21 v2 runbook ends
# (its lanes-dec12 checkpoint then exists). The 2225 pool is run later, when the GPU is otherwise free.
V2LOG=/c/Users/jonat/workspace/wt/sm21v2/artifacts/small-model/runbook-2026-10-18-sm21v2.log
SM=/c/Users/jonat/workspace/wt/small-model/artifacts/small-model
until grep -aqE "^\[sm21v2 [0-9:]+\] done" $V2LOG 2>/dev/null; do sleep 120; done
echo "[launcher $(date '+%m-%d %H:%M:%S')] SM21 v2 runbook done; POOL=732" >> $SM/runbook-eval-pools-732.log
cd /c/Users/jonat/workspace/wt/small-model && POOL=732 bash $SM/runbook-eval-pools.sh >> $SM/runbook-eval-pools-732.log 2>&1
echo "[launcher $(date '+%m-%d %H:%M:%S')] eval pools 732 runbook exit $?" >> $SM/runbook-eval-pools-732.log
