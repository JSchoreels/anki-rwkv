# Handoff v3: full run of fluency vs desired retention (SM127), 150 users

**Status: done 15:46:56 Sat 11-21 (150 of 150 rows; 3 users rerun after commit 083e5ae9). Result: sm127-result-v3.md.** Launched 01:28 Sat 11-21.
`C:/Users/jonat/workspace/wt/sm127`. Pre-registration: [design.md](design.md) section 13 (13.0-13.6 committed in
4b4852e2 before any v3 simulation; 13.7 amendment + CPU proof result committed in 9ff4e930, before the launch).
Earlier: pilot ([HANDOFF.md](HANDOFF.md)), time-drift diagnostic ([HANDOFF-v2.md](HANDOFF-v2.md)).

## What changed against the pilot (all in design.md 13)

- **Answer-time level anchor** (`--fix`): sampled later-day times scaled by a per-user, per-day factor tc calibrated on
  the real schedule (60-day window, cap kept); the fluent CDF reads S / tc. Fixes the +8% level bias of the diagnostic.
- **Per-user fluency threshold** (`--per-user-s`): fluent = recalled and answered within S_u, the 90th percentile of the
  user's own pre-split passed-review time (clipped 3-45 s). Targets 0.80 / 0.70 / 0.50 keep their names.
- **Code:** `scripts/sm127_fluency.py` (+ one identity hook `time_hook` in `scripts/sm41_engine.py`);
  `scripts/sm127_v3_proof.py` = the proof comparison. With the two flags off the code reproduces the pilot rows exactly.
- **CPU proof (2 users, 365 days, 22 lanes): all bars pass** (`proof-v3/proof-1901.json`, `proof-4752.json`; table in
  design.md 13.7): pilot rows reproduced exactly with the flags off; with the fix `tf` unchanged, budget violations 0,
  `fr-shared1#0` over `tf` seconds per review 0.905 -> 1.042 (user 1901) and 1.157 -> 1.029 (user 4752).

## What runs

- **Launcher:** `artifacts/small-model/sm127/run/launch-v3.sh`, launched hidden via WMI (Win32_Process.Create,
  ShowWindow=0) at 01:28:51 Sat 11-21, wrapper PID 34660 (launcher bash logged pid 74841 in MSYS numbering), from
  commit 9ff4e930.
- **Work:** all 150 users of `users.json` x 365 days x 22 lanes, 3 processes x 1 thread, below-normal priority, largest
  first, each under 6 GB (a user holds while < 6 GB RAM is free). Rows (aggregates only) `rows-v3/<user>.json`, logs
  `logs-v3/<user>.log`. No GPU, no queue ticket. No planned daily break; only `quiet.json` windows (scope all / cpu) hold it.
- **At the end:** the launcher checks each row (budget violations 0 on policy lanes, `nofb_rows` > 0, anchor on), then
  runs `report --set full` -> `sm127-tables-v3.md`, `sm127-result-v3.json`, `report-v3.out`, and logs `done`.
- **ETA:** about 58.6M lane reviews (2,645,757 window reviews x 22.2, the pilot's ratio) at about 400 lane reviews/s per
  process (proof: 405/s with 3 processes), plus a warm-up per user (140 s for small users, 357-463 s for the three
  largest first users; about 200 s on average): about 48 CPU process-hours, **about 16.5 h wall on 3 processes: expected
  end about 18:00 Sat 11-21 (17:00-20:00)**. First users started 01:28; the first per-user finish timings are in `logs-v3`. This is above the design's 8 h figure (that
  was written for 4 processes at 650 lane reviews/s; the pilot measured about 380/s).

## Logs, alive check, stop, relaunch

- **Launcher log:** `run-v3/launch-v3.log`, lines `[sm127v3 MM-DD HH:MM:SS] ...`; stdout/stderr in
  `run/launch-v3.nohup`. Done = last line `done`. `STOP:` lines: `STOP file present`, `uncommitted changes in scripts/`,
  `Traceback in <logs>`, `incomplete (n of 150 rows)`, `sanity check failed for users ...`.
- **Per user:** `logs-v3/<user>.log`: `warm-up ... S ... time anchor True`, `N lane reviews in T s (R/s)`, `exit 0`,
  `holding (quiet window ...)` / `resumed`, `STOP file: exiting without output`, any `Traceback` = failure (read it).
- **Alive check:**

  ```
  tail -3 /c/Users/jonat/workspace/wt/sm127/artifacts/small-model/sm127/run-v3/launch-v3.log
  ls /c/Users/jonat/workspace/wt/sm127/artifacts/small-model/sm127/rows-v3/*.json | wc -l      # 150 when done
  grep -l Traceback /c/Users/jonat/workspace/wt/sm127/artifacts/small-model/sm127/logs-v3/*.log
  powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | ? { $_.CommandLine -like '*sm127_fluency.py run*' } | select ProcessId,@{n='MB';e={[int]($_.WorkingSetSize/1MB)}},CommandLine"
  ```

  3 python processes while at least 3 users remain. After a Claude app restart or a Windows power event, check this
  (detached jobs died at the 11-17 power event): if no python process is left and the log has no `done`, relaunch.
- **Stop:** `touch C:/Users/jonat/workspace/wt/sm127/artifacts/small-model/sm127/run-v3/STOP` (read before every user
  and every simulated day; remove it before relaunching).
- **Relaunch** (finished users are skipped; an interrupted user restarts from its warm-up), PowerShell:

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm127/artifacts/small-model/sm127/run/launch-v3.sh >> C:/Users/jonat/workspace/wt/sm127/artifacts/small-model/sm127/run/launch-v3.nohup 2>&1"'; ProcessStartupInformation=$si}
```

- **No source edits** in `wt/sm127` `scripts/` while it runs (the launcher refuses uncommitted `scripts/` at start; edits
  also change running behaviour only for users not yet started). Other worktrees read only.

## Follow-up after `done` (fresh agent, Sonnet medium is enough; do not wait on the run)

1. **Checks:** `launch-v3.log` ends with `done`; 150 files in `rows-v3/`; no Traceback in `logs-v3/`; if anything
   is missing, relaunch (above) and stop.
2. **Read the verdict as pre-registered, design.md 13.4 (= section 7 with the ratio-of-totals statistic):**
   `sm127-tables-v3.md` / `report-v3.out`. DR\* = best desired-retention arm on fluent cards; G = `flu-0.80` over DR\*
   with its paired bootstrap CI; "a high desired retention is enough" if the CI upper bound is below +2%; "a fluency
   scheduler is needed" if G >= +2% with the CI lower bound above 0 and the memorised gap no worse than -2% (else a
   trade-off); otherwise inconclusive. Same rule on the `-nofb` arms (feedback check). Also read: the level check
   (accepted band [0.95, 1.05]), the floor shares, the per-user S_u line, the secondary arms (`flu-0.70`, `flu-0.50`,
   `flu-0.80-fb95`), the cost line. Do not change any threshold.
3. **Write `sm127-result-v3.md`** (short, full plain sentences): design in 3 lines (150 users, 365 days, 22 lanes,
   equal daily time, anchor + per-user S_u), the policy table, the comparisons table and the verdict line verbatim, the
   feedback check, the level check, floor shares, caveat of 13.2 (S_u lenient for slow users; the global 8 s pilot
   stays the reference for "within 8 s"), pilot verdict next to it, cost.
4. **Commit as Jonathan Schoreels** on `sm127`, `git add -f` explicit paths (never `work/`, `.npz`, `.pth`):
   `rows-v3/*.json`, `logs-v3/*.log`, `run-v3/launch-v3.log`, `run/launch-v3.nohup`, `sm127-tables-v3.md`,
   `sm127-result-v3.json`, `report-v3.out`, `sm127-result-v3.md`, this file (status done). Message ends with a blank
   line and `Co-Authored-By: Claude <your model> <noreply@anthropic.com>`; `git fetch origin`, merge `origin/sm127`,
   push. Never force-push, never bypass hooks.
5. **Factory page** (main checkout; fetch + ff first; another agent may be editing it): close the plan row (state done,
   `ended_at`), add a results row with `cpu_minutes`, times from `launch-v3.log`, `detail.notes` as a list of strings;
   bare-ID check, `scripts/publish_factory_page.py`, `--check`; commit only the two `pages/data` json files.
6. **Text rules:** every model as "plain name (ID)", "%" never "points", never "Expertium" (say "outside research"),
   "answered within the user's own S_u, reading the back included" for the v3 arms and "answered within 8 s, reading the
   back included" for the pilot, plain policy names.
