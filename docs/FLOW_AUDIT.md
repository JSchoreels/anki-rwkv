# Flow audit (one per wake)

Whenever a queue ticket finishes and the main session wakes, a small agent (Haiku, or Sonnet at low effort) checks whether that job followed the coordination flow, before or alongside the fresh agent that processes its results. It reads, reports, and appends one entry to [FLOW_AUDIT_LOG.md](FLOW_AUDIT_LOG.md). It changes nothing else: it reports violations, and the main session decides on fixes.

Input: the finished ticket ids (printed by `scripts/r01_queue_wait.py`). Each ticket's record is the last line with that ticket in `artifacts/coordination/queue/history.jsonl`.

## Checklist

Mark each item `ok`, `VIOLATION` (with evidence) or `n/a`.

1. **Lane.** If the job measured speed (reviews/s, timing, benchmark in its purpose, command or outputs), its record has `"timing": true`. Accuracy jobs need no flag.
2. **Quiet window.** A timing job's `started` (UTC; convert to local time) and its finish time fall outside every window in `artifacts/coordination/queue/quiet.json`. If not, its reviews/s must not be quoted.
3. **Expected minutes.** The record has `expected_minutes`, and the real run time (finish − `started`) is at most about 1.5× that. Flag large overruns, since they break the quiet-window planning.
4. **Exit code.** The job exited 0. If not, check that the follow-up handles it (re-queue or explanation).
5. **Handoff.** The owning track has a committed `HANDOFF.md` that names this ticket and its output paths.
6. **Pushed.** `git status -sb` in the track's worktree (`C:\Users\jonat\workspace\wt\<track>`) and in the main checkout shows no `ahead` commits.
7. **Docs.** For timing jobs, the result is recorded in `docs/GPU_OPTIMIZATIONS.md` on the track's branch once processed. Note it as pending if the result hasn't been processed yet.
8. **No waiting agent.** The owner did not keep an agent alive for the job: the command is detached and no agent is listed as waiting on it. Note this when you can't verify it.
9. **Outside work.** Other GPU users while a timing job ran: note `nvidia-smi` processes if checked now, and any non-queue GPU jobs mentioned in handoffs.

## Log entry format

Append to `docs/FLOW_AUDIT_LOG.md`, then commit and push:

```
## <local date time> — <ticket(s)>
- verdict: clean | N violations
- 1 lane: ok · 2 quiet: n/a · 3 expected: VIOLATION (60 → 97 min) · ...
- notes: <one or two lines>
```

Keep each entry under 10 lines. Recurring violations go to the main session as a proposed rule or code change, not a silent fix.
