# The small model in the review simulator (SM41): handoff, the changing-learner confirmation on all users (SM41g)

**Status: done** (finished 2026-11-08 02:37; results in learner-drift-result.md, "Results of the confirmation on all users").

Branch `sm41`, worktree `C:/Users/jonat/workspace/wt/sm41`. Always `git fetch` + `git merge --ff-only` first. If SSH fails:
`git -c credential.helper= -c credential.helper='!gh auth git-credential' fetch|push https://github.com/JSchoreels/anki-rwkv.git sm41`.
Plan, decision bar (written before the results): "Confirmation on all users (plan ...)" in
[learner-drift-result.md](learner-drift-result.md). Do not touch `scripts/sm41_engine.py` or any file the runner imports
while it runs; never touch the real-data check's files (`scripts/sm41_realcheck.py`, `realcheck/`).

## What runs (CPU only)

- **Runner:** `learner-drift-all/run-learner-drift-all.sh`, launched hidden via WMI (Windows pid 36744; MSYS pid in
  `learner-drift-all/runner.log`). 4 workers, 1 thread, below-normal priority; a split waits while < 3 GB RAM is free
  (the GPU scoring of another experiment starts ~01:01); no new job from 40 min before a quiet.json window (scope all
  / cpu, incl. the daily 20:00-01:00 break), the engine holds per split from 10 min before.
- **Jobs** (`learner-drift-all/jobs.txt`): 117 users (one job per user, all its splits `user d1,d2,d3`, largest first),
  304 splits. **145 lanes** per split (`lanes.txt`): `tf` + 48 policy lanes x 3 scenarios fused (improving `-g0.5_0.2`,
  declining `-gm0.5_m0.2`, no drift = no tag). 60 days, `--ivl-hist`, pre-split FSRS-7 refits `fullrun/fits/<user>.jsonl`.
- **Outputs:** `learner-drift-all/rows/<user>.jsonl` (one row per split), `learner-drift-all/logs/<user>.log`.
- **Expected end:** about 4.5-5.5 h of 4 processes: ~06:00-07:00 on 11-07, plus any hold for the 20:00 break (jobs
  claimed after 19:20 wait until 01:00).
- **Spot-check (done 11-07 00:50, before launch):** users 1606 split 660 and 2755 split 495, full 145 lanes
  (`learner-drift-all/spotcheck/`, `spotcheck.py`): no-drift lanes vs the existing full-run / screen 3 rows: 96 lanes,
  largest relative gap 1.95e-06; improving and declining views vs the 20-user separate runs: 96 lanes each, 1.0e-06
  and 2.6e-06; no Tracebacks. Report code verified on these rows.
- **Resumable.** If the runner died (no `done` in `runner.log`, no `sm41_engine.py run` process with `learner-drift-all`
  in its command line, e.g. after a Claude app restart): delete `learner-drift-all/locks/` and relaunch (PowerShell tool):
  ```
  $si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash /c/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/learner-drift-all/run-learner-drift-all.sh >> /c/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/learner-drift-all/runner.out 2>&1"'; ProcessStartupInformation=$si}
  ```
  Finished splits are skipped. Check first that no other sm41 job pushes the process count over 4.

## Check commands

```
cd /c/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/learner-drift-all
tail -n 5 runner.log; ls rows | wc -l   # 117 when done
grep -l Traceback logs/*.log
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? { $_.CommandLine -like '*sm41_*' } | select ProcessId,@{n='MB';e={[int]($_.WorkingSetSize/1MB)}},CommandLine"
```

## Follow-up (fresh agent, Sonnet medium; after `done` in `runner.log`)

1. `fetch` + `ff`. From the worktree:
   `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 scripts/sm41_learner_drift.py check --dir learner-drift-all`
   -> `learner-drift-all/check.out`: 117 row files, 304 split rows, no Tracebacks, no-drift lanes vs the existing rows
   (a few single-draw flips like the 20-user run are expected; list lanes with gap > 1e-6). Missing splits: relaunch and stop.
2. `... scripts/sm41_learner_drift.py report --dir learner-drift-all` -> `learner-drift-all/tables.md`, `tables.json`
   (scenario read from the lane's `g` tag; no-drift baseline = the fused no-drift lanes; CI over users, splits of a user
   averaged; no `-free` lanes so no re-anchoring section).
3. Fill a **"Results of the confirmation on all users"** subsection under "Confirmation on all users" in
   `learner-drift-result.md` (plain names with the ID in parentheses, never a bare ID, never "Expertium"): per scenario
   the instant head and the time-aware rule (beta = 2): gap vs FSRS-7 at matched realised retention and at matched
   memorised cards (days 30-59 and 60 days) with 95% CIs and the change vs no drift; reviews/day, median later-day
   interval, seconds per review vs FSRS-7 (change vs no drift); side by side with the 20-user numbers; key rows
   (targets 0.85 / 0.90); apply the decision bar (CI excludes 0 AND |change| >= 2.9 points) and say which 20-user
   findings are confirmed or retracted; **verdict on the user's hypothesis** ("RWKV would be more efficient since it
   would schedule less" when the learner improves) by the plan's two conditions; speed (lane reviews/s per process, CPU
   process-hours from the Speed table); FSRS-7 frozen caveat in one sentence.
4. Commit as Jonathan Schoreels (`git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`,
   message ending with a blank line + `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`), explicit paths,
   `git add -f` for `learner-drift-all/` (`rows`, `logs`, `runner.log`, `runner.out`, `check.out`, `tables.*`; not
   `locks/`), the result file, this file with status "done"; push right after. Never stage `.claude/`,
   `docs/GPU_OPTIMIZATIONS_LOG.md`, `pages/`, `artifacts/coordination/queue/quiet.json`.
5. Reply in under 300 words: commit hash, check result, the gaps and changes with CIs, verdict, speed.
