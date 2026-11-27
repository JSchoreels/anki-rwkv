# Rules for agents working on the small-model week plan (2026-10-21)

Read C:/Users/jonat/workspace/anki-rwkv/AGENT.md first (repository rules: checks before GPU, tolerances, docs).

## Git
- Work only in the worktree named in your brief. Create it with
  `git -C C:/Users/jonat/workspace/wt/small-model worktree add C:/Users/jonat/workspace/wt/<name> -b <name> small-model`.
- Commit as Jonathan Schoreels <jonathan.schoreels@gmail.com> (`git -c user.name="Jonathan Schoreels" -c user.email=jonathan.schoreels@gmail.com commit`).
  Every message ends with a blank line, then `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Push after every commit (`git push -u origin <branch>` the first time). Never bypass the pre-push hook (no --no-verify).
- Stage files by explicit path; `git add -f` for artifacts/. Never commit docs/GPU_OPTIMIZATIONS_LOG.md or anything under .claude/.
- Read-only: C:/Users/jonat/workspace/anki (fork), and every other worktree (read their files, never edit them).
- Never write to C:/Users/jonat/AppData/Roaming/Anki2/User 1. Own-collection results are aggregate only (no per-card/per-user rows
  from the user's own collection); benchmark users' per-user files may be committed. Collection: 13,973 cards / 13,198 notes.
- Published pages never name "Expertium" (say "outside research"). Do not edit pages/ (the main session does).

## GPU
- All GPU work goes through the queue: `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py`
  (see its --help; absolute script paths only; give honest --expected-minutes; speed claims need --timing tickets).
- Cap new GPU jobs at ~0.8 GPU memory (gpu_memory_fraction 0.8 or equivalent).
- Breaks: the queue has quiet windows every day 21:00-01:00 and Sunday 10-25 from 16:00 to 01:00
  (artifacts/coordination/queue/quiet.json). A ticket whose expected end reaches a window waits for 01:00.
  Chained launchers must also check the window themselves before enqueuing (respect quiet.json).
  Breaks are soft (user, 10-21): a ticket may run into a break when a cancel at the break would lose at most ~1-2 h
  (short scorings/checks, or resumable work). Long non-resumable training must end before the break or wait for 01:00.
  Don't start a ~2 h ticket less than ~15 min before a break. Judge by expected minutes; say in your report what you chose.
  During breaks the CPU must be quiet too: no heavy CPU work (checks, builds, scoring) between 21:00 and 01:00.
- Never change watched source in a worktree while a GPU ticket or runbook from it runs. Never edit a runbook .sh while it runs.
- Do not kill or cancel processes or tickets you did not start.

## Token budget (hand off before waiting)
- Do not wait on long GPU jobs. Once a job is queued: launch it detached (no background children of yours), make it write
  outputs to files, commit a short HANDOFF-<id>.md (ticket IDs, log paths, exact follow-up commands, what "done" looks like
  in the log, e.g. a `[<id> ...] done` / `STOP` line), push, and stop.
- Short CPU checks (minutes) are fine to wait on.

## Report (final message, under 250 words)
What was built, commits (hash + one line), checks run and their result, what is queued (ticket IDs, expected minutes),
the log line the main session should watch for, and anything that needs the user's decision. Name things with a short
name first and the ID in parentheses. Do not claim results you did not measure.
