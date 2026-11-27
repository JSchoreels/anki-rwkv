# Flow audit (one per wake)

Whenever a queue ticket finishes and the main session wakes, a small agent (Haiku, or Sonnet at low effort) checks whether that job followed the coordination flow, before or alongside the fresh agent that processes its results. It reads, reports, and appends one entry to [FLOW_AUDIT_LOG.md](FLOW_AUDIT_LOG.md). It changes nothing else: it reports violations, and the main session decides on fixes.

Input: the finished ticket ids (printed by `scripts/r01_queue_wait.py`). Each ticket's record is the last line with that ticket in `artifacts/coordination/queue/history.jsonl`.

## Checklist

Mark each item `ok`, `VIOLATION` (with evidence) or `n/a`.

1. **Lane.** If the job measured speed (reviews/s, timing, benchmark in its purpose, command or outputs), its record has `"timing": true`. Accuracy jobs need no flag, even when they log throughput in passing (for example a QAT or training run's reviews/s). That figure is then not a speed result: mark the lane `ok` and check that no handoff or doc quotes it as one.
2. **Quiet window.** A timing job's `started` (UTC; convert to local time) and its finish time fall outside every window in `artifacts/coordination/queue/quiet.json`. If not, its reviews/s must not be quoted.
3. **Expected minutes.** The record has `expected_minutes`, and the real run time (finish − `started`) is at most about 1.5× that. Flag large overruns, since they break the quiet-window planning.
4. **Exit code.** The job exited 0. If not, check that the follow-up handles it (re-queue or explanation).
5. **Handoff.** The owning track has a committed `HANDOFF.md` that names this ticket and its output paths. A runbook that queues its steps one by one cannot know their ticket ids in advance: a committed handoff that names the runbook, its log (which records each step) and the output paths counts, with the id taken from the log.
6. **Pushed.** `git status -sb` in the track's worktree (`C:\Users\jonat\workspace\wt\<track>`) and in the main checkout shows no `ahead` commits. Also run `.venv/Scripts/python.exe scripts/check_stray_outputs.py` (report only): it lists git-ignored copies in the main checkout of files committed on a track branch (jobs run from the main checkout write them). `different` ones are a violation to report; `identical`/`subset` ones are cleaned by the main session with `--delete-identical-or-subset`.
7. **Docs.** For timing jobs, the result is recorded in `docs/GPU_OPTIMIZATIONS.md` on the track's branch once processed. Note it as pending if the result hasn't been processed yet.
8. **No waiting agent.** The owner did not keep an agent alive for the job: the command is detached and no agent is listed as waiting on it. Read the `launcher` field of the ticket record: `launched_by` (from `--launched-by` / `R01_LAUNCHED_BY`), `stdin_tty`/`stdout_tty`, and `parents` (parent process chain with pid and command line). A detached runbook, or a `launched_by` naming one or \"agent, detached\", is `ok`; an interactive terminal parent (`stdout_tty` true) or an agent process in `parents` not marked detached is a violation. Records written before this field existed (tickets up to 000354) have no `launcher`: mark them `n/a (not verifiable)`.
9. **Outside work.** Other GPU users while a timing job ran: note `nvidia-smi` processes if checked now, and any non-queue GPU jobs mentioned in handoffs. CPU-heavy accuracy steps (collection scoring/noise with threads) must have run through the queue with `--claim cpu-heavy --expected-minutes N` so they never overlapped a `--timing` ticket (it claims `cpu-heavy` automatically); flag any that ran unwrapped or overlapped a timing run, and any ticket without `--expected-minutes`.

## Handoff-rule overrides

The main session may override a planning agent's handoff rule when the user's own instruction governs (for example a user-set cap or epoch count). Before launching, it must record the override, the reason and the user instruction in that track's HANDOFF. Item 5 also checks that this record exists whenever the job's command or limits differ from the handoff's original rule. Example: the small-model scale-up (cap 57 to 66 min per epoch, 3 epochs instead of 4), recorded in small-model HANDOFF commit 0b3fb7f.

## Log entry format

Append to `docs/FLOW_AUDIT_LOG.md`, then commit and push:

```
## <local date time> — <ticket(s)>
- verdict: clean | N violations
- 1 lane: ok · 2 quiet: n/a · 3 expected: VIOLATION (60 → 97 min) · ...
- notes: <one or two lines>
```

Keep each entry under 10 lines. Recurring violations go to the main session as a proposed rule or code change, not a silent fix.
