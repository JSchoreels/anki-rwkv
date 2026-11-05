# The small model in the review simulator (SM41): handoff, the changing-learner simulation (SM41g)

**Status: done** (runner finished 2026-11-05 19:21; check and report run 19:25; result in `learner-drift-result.md`).

Branch `sm41`, worktree `C:/Users/jonat/workspace/wt/sm41`. Always `git fetch` + `git merge --ff-only` first. If SSH
to GitHub fails:
`git -c credential.helper= -c credential.helper='!gh auth git-credential' fetch|push https://github.com/JSchoreels/anki-rwkv.git sm41`.

Read first: the brief [drift-brief.md](drift-brief.md) and the plan, metrics and decision rule (fixed before the
runs) in [learner-drift-result.md](learner-drift-result.md). The brief's `drift-result.md` already holds the stage 2
drift study, so this study writes `learner-drift-result.md` and `learner-drift/`.

Do not touch the real-data check of the time-aware target (SM41r): `scripts/sm41_realcheck.py`,
`artifacts/small-model/sm41/realcheck/`, `realcheck-result.md` (uncommitted, another agent's). Never stage, edit or
revert them.

## What runs (CPU only)

- **Runner:** `artifacts/small-model/sm41/learner-drift/run-learner-drift.sh`, launched hidden via WMI (Windows
  pid 78040; MSYS pid in `learner-drift/runner.log`). Engine code: the commit that adds this file.
- **Jobs** (`learner-drift/jobs.txt`, 60): the 20 users of part A x scenarios `improve` (`--drift 0.5,0.2`) and
  `decline` (`--drift=-0.5,-0.2`) first, largest users first, then `none` (no drift). All with `--ivl-hist`, the 57
  lanes of `learner-drift/lanes.txt`, 60 days, the full run's pre-split FSRS-7 refits.
- **Outputs:** `learner-drift/<scenario>/rows/<user>.jsonl` and `learner-drift/<scenario>/logs/<user>.log`.
- **Load:** 2 workers at once (the real-data check holds 4 slots until about 16:15-16:45; the main session allowed
  1-2 more); 2 more workers start when `realcheck/runner.log` has a line starting with `done `. 1 thread each,
  below-normal priority, a split is held while less than 3 GB of RAM is free.
- **Break:** a worker claims no new job from 19:20 (40 min before the 20:00-01:00 window in `quiet.json`) and
  resumes after 01:00 by itself; the engine also holds per split from 19:50.
- **Expected end:** `improve` + `decline` about 17:30-18:00; `none` about 18:30-19:20. Jobs still unclaimed at
  19:20 run after 01:00.
- **Resumable.** If the runner died (no `done` in `runner.log`, no `sm41_engine.py run` process with
  `learner-drift` in its command line, e.g. after a Claude app restart): delete `learner-drift/locks/` and relaunch.
  Finished splits are skipped.
  ```
  $si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash /c/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/learner-drift/run-learner-drift.sh >> /c/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/learner-drift/runner.out 2>&1"'; ProcessStartupInformation=$si}
  ```
  Keep all `sm41_` python processes at 4 or fewer once the real-data check's own count is known (check with the
  command below); if the real-data check is still running with 4, relaunch with the 2 initial workers only (as is).

## Check commands

```
tail -n 5 /c/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/learner-drift/runner.log
for s in improve decline none; do ls /c/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/learner-drift/$s/rows 2>/dev/null | wc -l; done   # 20 each when done
grep -l Traceback /c/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/learner-drift/*/logs/*.log
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? { $_.CommandLine -like '*sm41_*' } | select ProcessId,@{n='MB';e={[int]($_.WorkingSetSize/1MB)}},CommandLine"
```

## Follow-up (fresh agent, Sonnet medium)

1. **Checks** (when `improve` and `decline` have 20 row files each; `none` may still be running):
   `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 scripts/sm41_learner_drift.py check`
   (from the worktree). It writes `learner-drift/check.out`: row counts, Tracebacks, and the `none` rerun against
   the existing full-run / screen 3 rows (largest relative gap should be about 1e-6 or less; part A's sample check
   gave 1.0e-6). If rows are missing, relaunch (above) and stop.
2. **Report:** `... scripts/sm41_learner_drift.py report` -> `learner-drift/tables.md` + `tables.json`. The
   interval columns need `none`; if it is not done, write the result without the interval change and say so.
3. **Fill "Results" in `learner-drift-result.md`** (short; plain names with the ID in parentheses, never a bare
   ID, never "Expertium"):
   - the bit-identity line is already in the plan (IDENTICAL, `learner-drift/bitcheck/compare.out`);
   - per scenario, for the instant head and the time-aware rule (beta = 2): the gap vs FSRS-7 at matched realised
     retention and at matched memorised cards (days 30-59 and 60 days) with CIs, and the change vs no drift;
   - "does it schedule less": reviews/day, median later-day interval, seconds per review vs FSRS-7, change vs no
     drift; plus the per-lane table's key rows (targets 0.85 / 0.90);
   - the re-anchoring check (`-free` lanes; the no-drift `-free` gap has an outlier, +577 ± 1198%: give the median
     from `tables.json` if needed, or say the free lanes are noisy);
   - FSRS-7 stayed frozen at the pre-split refit: one sentence on a mid-run refit (it would see the drifted passes
     and lengthen / shorten its intervals for every card at once; not run);
   - **verdict by the plan's decision rule** on the user's hypothesis ("RWKV would schedule less when the learner
     improves"), and whether a scenario meets the confirmation rule (ranking change, or a change vs no drift with a
     CI excluding 0);
   - speed: lane reviews/s per process and CPU process-hours, from the tables.
4. **If the confirmation rule fires:** do not run it before the break. Write a short "Next" line in the result: the
   confirmation on all 117 users (304 splits) of that scenario, only the lanes it needs (`tf`, `ihr-0.80/0.85/0.90`,
   `ihr-*-s2`, `fsrs7-*` x 4 seeds = 49 lanes), is about 30-40 process-hours (about 8-10 h on 4 processes): it needs
   the main session's go-ahead and its own handoff.
5. **Commit as Jonathan Schoreels:**
   `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, the message ending
   with a blank line + `Co-Authored-By: Claude Sonnet <noreply@anthropic.com>` (your own model name). Explicit
   paths, `git add -f`: `learner-drift/` (`rows/*.jsonl`, `logs/*.log`, `runner.log`, `runner.out`, `check.out`,
   `tables.md`, `tables.json`, but not `locks/`), `learner-drift-result.md`, this file with the status set to
   "done". Then push. Never stage `docs/GPU_OPTIMIZATIONS_LOG.md`, `.claude/`, `pages/`,
   `artifacts/coordination/queue/quiet.json` or the real-data check's files. Never bypass hooks. The main session
   updates the Factory page.
6. Reply in under 300 words: commit hash, bit-identity check, per scenario the gaps vs FSRS-7 (both metrics) for
   the instant head and the time-aware rule with CIs and the change vs no drift, intervals and reviews/day, the
   verdict on the user's hypothesis, speed (lane reviews/s per process, CPU process-hours).
