# The small model in the review simulator (SM41): handoff, stage 2 part 1 (the drift study)

**Status: done 2026-11-03 (drift study finished; result in [drift-result.md](drift-result.md)).** The Follow-up section below was carried out.

Branch `sm41`, worktree `C:/Users/jonat/workspace/wt/sm41`. Stage 1: [prototype-result.md](prototype-result.md). Design incl. stage 2 and the interval convention (duration leak): [DESIGN.md](DESIGN.md) section 8. Always `git fetch` + `git merge --ff-only` first.

## What runs (CPU only)

- Runner: `artifacts/small-model/sm41/drift/run-drift.sh`, launched 2026-11-03 16:27:26 hidden via WMI (bash pid 77892, script pid in `drift/runner.log`), 4 processes x 1 thread, 0.6-1.6 GB each, ~100% of one core each. Code commit: 2c85e5de (a first launch at 16:12 on b2a2b271 was stopped and its rows deleted: negative commit intervals gave NaN in some free-running lanes; fixed in 2c85e5de).
- Jobs: 50 users x 2-3 splits x 60 days (133 splits) + 10 users x 1 split x 120 days, 32 lanes per split (2 teacher-forced + 10 free-running arms x 3 seeds), one job per (user, horizon) -> `drift/rows/<user>-<days>.jsonl`, log `drift/logs/<user>-<days>.log`; at the end the runner writes the report (`drift/drift-tables.md`, `drift/drift-result.json`, `drift/report.out`) and "done" in `drift/runner.log`.
- Breaks: every job re-reads `artifacts/coordination/queue/quiet.json` (main checkout) before each user and each split and holds from 10 min before a window (scope all/cpu) until its end. Tonight: hold 17:20-19:30, then free through the night; from 11-04 the 20:00-01:00 windows.
- Expected: ~3.5 process-hours of lanes + warm-ups on 4 processes (measured 2,456 lane reviews/s per process), about 55 min of running. Started 16:27, so jobs run until the 17:20 hold, wait, resume at 19:30 and should be done by about 19:50-20:00 (report written right after).
- First read (11 splits, before the hold): plain free running `fr` has a first-touch state drift of -4.0 points at days 30-59 (mean abs 6.2) and Easy share of passes +10.6 points. Daily shared-stream re-anchoring `fr-shared1` is best so far (mean abs 2.7, Easy +3.0, seconds +0.1). Grade temperature 0.8 makes it worse. Duration leak: teacher-forced same-day grade LogLoss 0.6312 answer-to-answer vs 0.6354 show-time; later-day identical. Not final.
- Resumable: a (user, split, days) row already in `rows/` is skipped. If the runner died (e.g. a Claude app restart), relaunch the same command hidden via WMI. It redoes only the unfinished splits (a job's warm-up reruns):
  ```
  $si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash /c/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/drift/run-drift.sh >> /c/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/drift/runner.out 2>&1"'; ProcessStartupInformation=$si}
  ```

## Follow-up (fresh agent, Sonnet medium)

1. Check that it finished: `drift/runner.log` ends with "done", there are 60 files in `drift/rows/` (50 with `-60`, 10 with `-120`), 133 rows with days 60 and 10 with days 120, and `grep -l Traceback drift/logs/*` is empty.
2. Rebuild the report (the runner already does this; repeat after any relaunch):
   `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 C:/Users/jonat/workspace/wt/sm41/scripts/sm41_drift.py report`
3. Write `artifacts/small-model/sm41/drift-result.md` (short; plain names with IDs in parentheses, never "Expertium"):
   - the design (users, splits, horizons, lanes, arms: DESIGN.md 8.2);
   - the CPU proof (`drift/proof-cpu.json`);
   - the primary metric fixed before the run: first-touch STATE drift on days 30-59 at the 60-day horizon. Answer which arm removes most of it (the "Mitigation vs plain free running" table, paired per user) and what is left: the remaining state drift, Easy share and seconds drift of the best arm;
   - how the drift grows with the horizon (the 120-day tables, by 30-day block);
   - the duration-leak table (teacher forcing, show-time vs answer-to-answer query rows) and fr vs fr-answer;
   - speed: lane reviews/s per process (report footer) and CPU (4 processes x 1 thread).
   Caveat to keep: `fr-texp` feeds a geometric-mean time, so compare its workload with the "seconds expected" column, not "seconds fed".
4. Commit as Jonathan Schoreels (`git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending with a blank line + `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`). Stage with `git add -f`:
   - `artifacts/small-model/sm41/drift/rows/*.jsonl`, `drift/logs/*.log`, `drift/drift-tables.md`, `drift/drift-result.json`, `drift/runner.log`, `drift/report.out`;
   - `artifacts/small-model/sm41/drift-result.md`, this file (status "done").

   Push. Never stage `docs/GPU_OPTIMIZATIONS_LOG.md` or `.claude/`. Do not edit `pages/data/factory-curated.json`; the main session updates the Factory page.

## Still open after this (stage 2)

- Done 2026-11-04 (stage 2 part 2, [engine-result.md](engine-result.md), DESIGN.md section 9): intervals chosen with the instant head (K = 16 candidate query rows), the general vectorised lane engine for scheduler lanes (`scripts/sm41_engine.py`), the FSRS-7 baseline policy; CPU proofs PASS; validation on 8 users.
- The time-aware objective on the same candidate grid: deferred (needs a lapse value or a daily time budget, DESIGN.md 9.2).
- The full run: proposal and cost in [engine-result.md](engine-result.md); not started, the main session decides.
- GPU batched-step benchmark (stage 1 follow-up 1, skipped: no GPU without the user's go).
- Retraining on show-time intervals is the separate show-time interval retrain (SM91); not part of this track.
