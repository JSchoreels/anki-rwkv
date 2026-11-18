# Handoff v2: follow-ups on the pilot of fluency vs desired retention (SM127)

**Status: Task 1 done, Task 2 (time-drift diagnostic) running on CPU.** Branch `sm127`, worktree
`C:/Users/jonat/workspace/wt/sm127`. Pre-registration: [design.md](design.md) section 12 (committed in 57869061 before
the launch).

## Task 1 (done, commit 57869061)

- The report's primary statistic is now the ratio of population totals with a paired user bootstrap (10,000 resamples,
  seed 127), plus the median per-user ratio; the old mean of per-user ratios is a secondary table (numbers unchanged).
- Outputs: `report-pilot-v2.out`, `sm127-tables-v2.md`, `sm127-result-v2.json`.
- Verdict under the new statistic: inconclusive (fluency at 0.80 vs desired retention 0.90: fluent -14.83%
  [-31.31, +20.20], memorised -24.85% [-36.10, -13.49]). Without feedback: "a high desired retention is enough"
  (fluent -35.22% [-49.78, -21.37]).

## Task 2: what runs

- **Launcher:** `timedrift/run/launch-td.sh`, launched hidden via WMI at 14:01 Wed 11-18 from commit 80e14f50 (a first launch at 13:56 failed at the save of user 384, int64 not JSON serialisable; stopped, fixed, relaunched).
- **Work:** the 12 pilot users x days 0-149 x 8 lanes (`scripts/sm127_timedrift.py run`), 3 processes x 1 thread,
  below-normal priority, largest first. Expected about 1-1.5 CPU process-hours, about 20-30 min wall.
- Each user writes `timedrift/rows/<user>.json` (aggregates only) and `timedrift/logs/<user>.log`; the launcher then runs
  `report` -> `timedrift/timedrift-tables.md`, `timedrift/timedrift-result.json`, `timedrift/report-td.out` and logs
  `done` in `timedrift/run/launch-td.log`.
- **Alive check:**

  ```
  tail -3 /c/Users/jonat/workspace/wt/sm127/artifacts/small-model/sm127/timedrift/run/launch-td.log
  ls /c/Users/jonat/workspace/wt/sm127/artifacts/small-model/sm127/timedrift/rows/*.json | wc -l    # 12 when done
  grep -l Traceback /c/Users/jonat/workspace/wt/sm127/artifacts/small-model/sm127/timedrift/logs/*.log
  ```
- **Stop:** `touch .../sm127/timedrift/STOP` (remove before relaunching). **Relaunch** (done users are skipped), PowerShell:

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm127/artifacts/small-model/sm127/timedrift/run/launch-td.sh >> C:/Users/jonat/workspace/wt/sm127/artifacts/small-model/sm127/timedrift/run/launch-td.nohup 2>&1"'; ProcessStartupInformation=$si}
```

## Follow-up after `done` (fresh agent, Sonnet medium is enough)

1. Check: 12 rows, no Traceback, the report's identity line says OK (lanes also in the pilot reproduce its buckets).
   If not, do not read the diagnostic.
2. Read `timedrift/timedrift-tables.md` against design.md 12.2: the reading line is computed by the report from the
   pre-registered thresholds (within-cell share of the desired retention 0.90 gap, persistence excess on the real
   schedule).
3. If the reading is (a) drift: propose the minimal fix with its cost (design.md 12.2 last bullet); do not run a full
   re-simulation.
4. Commit with `git add -f` the `timedrift/` rows, logs, run log/nohup, tables, result, report-td.out and this file
   (status done); never `work/`, `.npz`, `.pth`. Commit as Jonathan Schoreels, fetch + merge, push `sm127`.
5. Text rules: every model as "plain name (ID)", "%" never "points", "answered within 8 s, reading the back
   included", plain policy names in anything the user reads.
