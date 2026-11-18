# Handoff: fluency vs desired retention in the simulator with answer-time feedback (SM127), the pilot

**Status: pilot running (CPU only).** Design and every rule, fixed before launch: [design.md](design.md). CPU proof:
[check-cpu.json](check-cpu.json) (pass: K1-K8 all OK, 12:09 Wed 11-18). Branch `sm127` (from `origin/sm41`), worktree
`C:/Users/jonat/workspace/wt/sm127`, code `scripts/sm127_fluency.py`. It subclasses the lane engine of the review
simulator hosting the small model (SM41); `scripts/sm41_engine.py` is unchanged.

- **Model:** the show-time retrain (SM92) with its answer-time head from the time heads for the family (SM117).
- **Answer time** = front + back: from the moment the question is shown to the grade press, capped at the preset's
  maximum answer seconds. Every write-up says "answered within 8 s, reading the back included".

## What runs

- **Launcher:** `artifacts/small-model/sm127/run/launch-sm127.sh`, launched hidden via WMI at 12:10:27 Wed 11-18 from commit
  cd2f5526. Wrapper PID 26340 (launcher bash logged pid 17876 in MSYS numbering) (Windows).
- **Work:** the 12 pilot users of `users.json` (`pilot`) x 365 days x 22 lanes.
  - 4 processes x 1 thread at below-normal priority, largest windows first. No GPU, no queue ticket.
  - Each user writes `rows/<user>.json` (aggregate metrics only) and `logs/<user>.log`.
- **At the end:** `report --set pilot` writes `sm127-tables.md`, `sm127-result.json` and `report-pilot.out`, then the
  launcher logs `done`.
- **Holds.**
  - Each user process checks quiet.json (scope all / cpu) and the daily 20:00-01:00 break (Sunday from 16:00) before
    it starts. It does not start a user within 60 min of a window.
  - A running user pauses at the next simulated day inside a window and resumes after it. The process stays alive
    holding its RAM, using no CPU.
  - A user holds while less than 6 GB of RAM is available.
- **Expected:** about 5.7M lane reviews at about 600-650 lane reviews/s per process (proof K8: 649/s on 22 lanes), plus warm-ups: about 3-4 CPU process-hours, about 1.5-2 h wall. **Expected end: about 14:00 (13:30-14:30) Wed 11-18**, well before the 20:00 break. If a user is still
  running at 20:00, it pauses and finishes after 01:00.

## Log lines

- **Launcher log:** `C:/Users/jonat/workspace/wt/sm127/artifacts/small-model/sm127/run/launch-sm127.log`, lines
  `[sm127 MM-DD HH:MM:SS] ...`. Launcher stdout and stderr: `run/launch-sm127.nohup`.
- **Done:** the last line is `[sm127 ...] done`, after `user processes ended: 12 of 12 rows` and `report exit 0`.
- **STOP:** any `[sm127 ...] STOP: ...` line (`STOP file present`, `incomplete (n of 12 rows); relaunch to finish`,
  `uncommitted changes in scripts/`).
- **Per user:** `logs/<user>.log`.
  - `[sm127] user U split D: warm-up ...` (start);
  - `[sm127] user U: N lane reviews in T s (R/s) ...` (finished);
  - `exit 0 <time>`;
  - `holding (quiet window until ...)` / `resumed at ...`;
  - `STOP file: exiting without output`;
  - any `Traceback` = a failure: read it, do not relaunch blindly.

## Alive check, stop, relaunch

- **Alive:**

  ```
  tail -3 /c/Users/jonat/workspace/wt/sm127/artifacts/small-model/sm127/run/launch-sm127.log
  ls /c/Users/jonat/workspace/wt/sm127/artifacts/small-model/sm127/rows/*.json | wc -l      # 12 when done
  grep -l Traceback /c/Users/jonat/workspace/wt/sm127/artifacts/small-model/sm127/logs/*.log
  powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? { $_.CommandLine -like '*sm127_fluency.py run*' } | select ProcessId,@{n='MB';e={[int]($_.WorkingSetSize/1MB)}},CommandLine"
  ```

  4 python processes while at least 4 users remain. After a Claude app restart or a Windows power event, check this:
  detached jobs died at the 11-17 power event. If no python process is left and the log has no `done`, relaunch.
- **Stop:** `touch C:/Users/jonat/workspace/wt/sm127/artifacts/small-model/sm127/run/STOP`. It is read before every
  user and every simulated day, so a running user exits within about a minute. The STOP file must be removed before
  relaunching.
- **Relaunch** (finished users are skipped; an interrupted user restarts from its warm-up), in PowerShell:

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm127/artifacts/small-model/sm127/run/launch-sm127.sh >> C:/Users/jonat/workspace/wt/sm127/artifacts/small-model/sm127/run/launch-sm127.nohup 2>&1"'; ProcessStartupInformation=$si}
```

- **No source edits** in `wt/sm127` (`scripts/`, `expertium/`, `rwkv/`) while it runs. The launcher refuses
  uncommitted `scripts/` changes at its start. Other worktrees (`wt/sm41`, `wt/sm92`, `wt/sm117`) are read only.

## Follow-up after done (fresh agent, Sonnet medium is enough)

1. **Checks.**
   - `run/launch-sm127.log` ends with `done`.
   - 12 files in `rows/`; no `Traceback` in `logs/`.
   - In every row: `budget_violations` is 0 on every policy lane and `nofb_rows` > 0.
   - If anything is missing, relaunch (above) and stop. Do not wait on the run.
2. **Report** (the launcher already ran it; repeat it after any relaunch):
   `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 C:/Users/jonat/workspace/wt/sm127/scripts/sm127_fluency.py report --set pilot > C:/Users/jonat/workspace/wt/sm127/artifacts/small-model/sm127/report-pilot.out 2>&1`
3. **Write `artifacts/small-model/sm127/sm127-pilot-result.md`.** Keep it short, full plain sentences, every model as
   "plain name (ID)", never a bare ID, "%" never "points", never "Expertium" (say "outside research"), FSRS7-Instant
   if it comes up (never FSRS-Instant), always "answered within 8 s, reading the back included". Sections:
   - **Design in 3 lines:** 12 pilot users of the 732 set, 365 days, 22 lanes, the equal daily time budget, the model
     (the show-time retrain (SM92) with its head from the time heads for the family (SM117)); design.md.
   - **The policy table** from `sm127-tables.md`: min/day, memorised, fluent, new cards shown, retention, passed within
     8 s, floor share, budget-hit days.
   - **The comparisons table** and the verdict line, verbatim. Say plainly that the pilot is not decisive (design.md 7):
     it only decides go or no-go for the full run.
   - **Feedback:** fluency vs desired retention with and without answer-time feedback (the `-nofb` lines), and
     whether the verdict differs.
   - **The `flu-0.80` floor share** (the share of decisions where P(recalled and answered within 8 s) was already below
     0.80 at 1 day). If it is above 50%, write the open decision of design.md 9: the user picks the full run's fluency
     definition (keep it / fluency where reachable (fb95) / a lower target / a per-user S).
   - **Drift check:** the `fr-shared1` one-year first-touch drift, against the -5% flag of design.md 8.
   - **Cost:** lane reviews/s per process, CPU process-hours, wall time from `launch-sm127.log`. Then the full-run
     estimate (150 users: 2.65M window reviews, about 10x the pilot's 259k), which needs the user's go.
4. **Commit as Jonathan Schoreels** on `sm127` (artifacts are gitignored: `git add -f` with explicit paths; never
   `work/`, `.npz`, `.pth`): `rows/*.json`, `logs/*.log`, `run/launch-sm127.log`, `run/launch-sm127.nohup`,
   `sm127-tables.md`, `sm127-result.json`, `report-pilot.out`, `sm127-pilot-result.md`, and this file with the status
   set to done.
   - Commit with `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`; the
     message ends with a blank line and `Co-Authored-By: Claude <your model> <noreply@anthropic.com>`.
   - Then `git fetch origin`, merge `origin/sm127` if it moved, and `git push`. Never force-push, never bypass hooks.
5. **Factory page** (main checkout `C:/Users/jonat/workspace/anki-rwkv`; run
   `git fetch origin && git merge --ff-only origin/main` first).
   - Close the SM127 plan row: `state` done, `ended_at`.
   - Add a results row, with the times from `run/launch-sm127.log`, `cpu_minutes` instead of GPU minutes, and
     `detail.notes` as a JSON list of strings. It covers: the verdict (pilot, not decisive), the fluent and memorised
     gaps of `flu-0.80` against the best desired retention with CIs, the feedback effect, the floor share, the drift
     check, and the cost plus the full-run go question.
   - Then run the bare-ID check, `.venv/Scripts/python.exe scripts/publish_factory_page.py`, and `--check`. Commit
     only the two `pages/data` json files, then fetch + merge and push. Never stage `docs/GPU_OPTIMIZATIONS_LOG.md` or
     `.claude/`.
6. **A plain-language summary to the user**, with the same text rules, ending with the open decisions:
   - the go for the full run (about 30 CPU process-hours, about 8 h on 4 processes);
   - the fluency definition, if the floor share asks for it.

## Plan row to insert now (by the agent that owns `pages/data/factory-curated.json`; this agent did not edit it)

```json
{
 "id": "SM127",
 "track": "small model",
 "since": "11-18T12:10",
 "kind": "cpu",
 "minutes": 110,
 "match": [
  "fluency vs desired retention in the simulator with answer-time feedback (SM127): "
 ],
 "what": "Fluency vs desired retention in the simulator with answer-time feedback (SM127): the user's question of 11-15, is a high desired retention enough for mastery or do we need a scheduler that keeps answer time? In the review simulator hosting the small model (SM41), with the show-time retrain (SM92) and its answer-time head from the time heads for the family (SM117), one simulated year per user at the same daily review time (the user's real mean minutes per day; an expensive policy shows fewer new cards): desired retention 0.90 / 0.95 / 0.98 against a fluency scheduler (the interval where P(recalled and answered within 8 s, reading the back included) falls to 0.80; also 0.70, 0.50 and a 'fluency where reachable' variant). Sampled answer times are fed back into the model (the models read a slower success as weaker memory, the retrieval-effort probe (SM122)); a sensitivity arm feeds the user's median time instead. Read-out at day 365: expected memorised cards and expected fluent cards, user-bootstrap CIs. This is the pre-registered pilot: 12 users of the 732 set, 22 lanes, CPU only (CPU proof passed); the full run (150 users, about 30 CPU process-hours) needs the user's go",
 "when": "Started 12:10 Wed 11-18 (CPU, 4 processes, no GPU); pilot result about 14:00 Wed 11-18 (13:30-14:30; a user still running at 20:00 pauses until 01:00)",
 "decides": "Whether a high desired retention already gives the most fluent cards (answered within 8 s, reading the back included) at equal daily time, or a fluency scheduler adds at least 2% more; for the pilot only: go or no-go for the full run, and whether the 8 s / 0.80 target is feasible enough (share of cards already below it at a 1-day interval) or the user must pick another fluency definition",
 "state": "running",
 "started_at": "2026-11-18T12:10+01:00"
}
```
