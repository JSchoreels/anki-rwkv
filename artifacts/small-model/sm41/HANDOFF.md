# The small model in the review simulator (SM41): handoff, stage 2 part 3 (the full run)

**Status: done (finished 2026-11-04 19:39:40; result in fullrun-result.md).** The follow-up below is for a fresh agent (Sonnet, medium effort).

Branch `sm41`, worktree `C:/Users/jonat/workspace/wt/sm41`. Always `git fetch` + `git merge --ff-only` first. Design, metrics and the primary metric, all fixed before launch: [DESIGN.md](DESIGN.md) section 10. Earlier parts: [drift-result.md](drift-result.md) (part 1; its handoff is in the git history of this file) and [engine-result.md](engine-result.md) (part 2).

## What runs (CPU only, no GPU)

- **Runner:** `artifacts/small-model/sm41/fullrun/run-fullrun.sh`, launched hidden via WMI (Win32_Process.Create, ShowWindow=0). Code commit: e4a4b877.
  - bash pid 65144 (Windows); the script logged pid 1385326 (MSYS) in `fullrun/runner.log`. At 08:00 the first jobs held for RAM (1.7 GB available) while the show-time retrain of the release (SM92) head training ran (started 07:47, about 45-90 min); they start on their own when RAM frees.
  - 4 processes x 1 thread, below-normal priority.
- **Jobs:** 117 users x 2-3 splits = 304 split points x 60 days, 59 lanes per split. One job per user, largest first:
  1. the pre-split FSRS-7 refits of its splits (srs-benchmark venv) -> `fullrun/fits/<user>.jsonl`;
  2. every split's lanes, one warm-up pass -> `fullrun/rows/<user>.jsonl`.

  Log: `fullrun/logs/<user>.log`; runner output: `fullrun/runner.out`. At the end the runner writes `fullrun/check.out`, `fullrun/report.out`, `fullrun/fullrun-tables.md` and `fullrun/fullrun-result.json`, then "done" in `fullrun/runner.log`.
- **Holds.** Every job re-reads `C:/Users/jonat/workspace/anki-rwkv/artifacts/coordination/queue/quiet.json` (scope all / cpu) before each user and each split, and holds from 10 min before a window until its end (today 20:00-01:00). It also holds a new split while less than 3 GB of RAM is available.
- **Expected:** about 21 CPU process-hours (smoke: 0.046 engine-seconds per window review, 1.39M window reviews, plus warm-ups and refits), so about 5-6 h on 4 processes. **Expected end: about 14:30-15:30 today** (after the RAM hold), before the 19:50 hold. If it runs into the hold, it resumes at 01:00 and ends in the night.
- **Resumable.** Finished (user, split) fits and rows are skipped. If the runner died (e.g. a Claude app restart: check that the pid is gone and `runner.log` has no "done"), relaunch the same command hidden. Only a user's unfinished splits are redone; its warm-up reruns.
  ```
  $si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash /c/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/fullrun/run-fullrun.sh >> /c/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/fullrun/runner.out 2>&1"'; ProcessStartupInformation=$si}
  ```

## Check commands

```
tail -n 3 /c/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/fullrun/runner.log
ls /c/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/fullrun/rows | wc -l       # 117 when done
cd /c/Users/jonat/workspace/wt/sm41 && /c/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 scripts/sm41_fullrun.py check
grep -l Traceback /c/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/fullrun/logs/*.log
grep -h holding /c/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/fullrun/logs/*.log | tail -n 3
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? { $_.CommandLine -like '*sm41_engine.py run*' } | select ProcessId,@{n='MB';e={[int]($_.WorkingSetSize/1MB)}}"
```

## Follow-up (fresh agent, Sonnet medium)

1. **Checks.**
   - `fullrun/runner.log` ends with "done".
   - `scripts/sm41_fullrun.py check` reports 304 splits planned, 304 finished, 0 missing, 0 duplicates, lanes per row [59], 304 fits, 0 logs with Traceback, and only "pre-split FSRS-7 refit" as the FSRS-7 source.
   - If anything is missing, relaunch (above) and stop. Do not wait on the run.
2. **Rebuild the report.** The runner does this already; repeat after any relaunch:
   `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 C:/Users/jonat/workspace/wt/sm41/scripts/sm41_fullrun.py report > C:/Users/jonat/workspace/wt/sm41/artifacts/small-model/sm41/fullrun/report.out 2>&1`
3. **Write `artifacts/small-model/sm41/fullrun-result.md`.** Keep it short. Plain names with the ID in parentheses, never a bare ID, never "Expertium"; say FSRS7-Instant if it comes up, never FSRS-Instant. Sections:
   - **design in 3 lines:** 117 users x 304 splits, 60 days, 59 lanes; FSRS-7 refit before each split day; DESIGN.md 10.
   - **the FSRS-7 fit check:** the srs-benchmark fit had seen the split day in N of 304 splits (report footer). Refit proof: `fullrun/fsrs7-proof-cpu.json`, numbers in DESIGN.md 10.1. The small model's training users do not overlap the 732 set.
   - **the primary table:** the "Primary" table with its 95% CI, and the verdict line, verbatim. Then one sentence applying the decision rule fixed before launch: the instant head moves on to the time-aware objective only if the common-range gap is < 0 with the CI excluding 0; otherwise FSRS-7 stays the scheduler baseline.
   - **the policy table:** realised retention, aim, min/day, memorised, cost per policy.
   - **each user's best desired retention:** the table at P = 1 / 3 / 10, and whether a per-user target matters (the gain column).
   - **re-anchoring's effect:** the anchored vs free gaps, free minus anchored with CI, the sign flips, and the free lanes' drift.
   - **the drift check:** fr-shared1 state drift vs the drift study's -0.6 ± 1.0.
   - **speed and CPU hours:** lane reviews/s per process, CPU process-hours (report footer), 4 processes x 1 thread, wall time from `runner.log` (start to done, minus any hold).
4. **Commit as Jonathan Schoreels:** `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, with the message ending in a blank line + `Co-Authored-By: Claude Sonnet <noreply@anthropic.com>` (your own model name). Use explicit paths and `git add -f`:
   - `artifacts/small-model/sm41/fullrun/rows/*.jsonl`, `fullrun/fits/*.jsonl`, `fullrun/logs/*.log`;
   - `fullrun/fullrun-tables.md`, `fullrun/fullrun-result.json`, `fullrun/check.out`, `fullrun/report.out`, `fullrun/runner.log`;
   - `artifacts/small-model/sm41/fullrun-result.md`;
   - this file, with the status set to "done".

   Then push (`git push`). Never stage `docs/GPU_OPTIMIZATIONS_LOG.md`, `.claude/` or `pages/data/factory-curated.json`. Never bypass hooks. The main session updates the Factory page.

## Still open after this

- The time-aware objective (DESIGN.md 9.2): only if the primary result says the instant head wins.
- GPU batched-step benchmark (stage 1 follow-up 1): no GPU without the user's go.
- Retraining on show-time intervals is the separate show-time interval retrain (SM91), not part of this track.
