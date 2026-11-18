#!/usr/bin/env bash
# Starts the SM32b QAT runbook once the morning SM30 rep-flags runbook has finished (keeps the queue order).
# Waits for its last line "[sm30 HH:MM:SS] done: cont" (or "... ended with exit N", which also frees the GPU) in the
# log the user named (small-model worktree) or where launch-after-night.sh writes it (rep-flags worktree).
# Launch (after review): nohup bash /c/Users/jonat/workspace/wt/state-vq/artifacts/small-model/launch-after-repflags-qat.sh \
#   > /c/Users/jonat/workspace/wt/state-vq/artifacts/small-model/launch-after-repflags-qat.log 2>&1 &
LOGS="/c/Users/jonat/workspace/wt/small-model/artifacts/small-model/runbook-2026-10-18-repflags.log /c/Users/jonat/workspace/wt/rep-flags/artifacts/small-model/runbook-2026-10-18-repflags.log"
until grep -qE "\] done|ended with exit" $LOGS 2>/dev/null; do sleep 120; done
echo "[launcher $(date '+%m-%d %H:%M:%S')] rep-flags runbook finished: $(grep -hE '\] done|ended with exit' $LOGS 2>/dev/null | tail -1), starting SM32b"
bash /c/Users/jonat/workspace/wt/state-vq/artifacts/small-model/runbook-2026-10-18-qat.sh > /c/Users/jonat/workspace/wt/state-vq/artifacts/small-model/runbook-2026-10-18-qat.log 2>&1
echo "[launcher $(date '+%m-%d %H:%M:%S')] SM32b runbook exit $?"
