# Big long cards with 6-bit factors (SM165): handoff

Status: DONE (launched 11-25 04:44, finished 05:45; read-out in RESULT.md). The building agent stopped after the launch check; a fresh agent (Sonnet,
medium effort) does the read-out. Branch `sm165`, worktree `C:/Users/jonat/workspace/wt/sm165`, cut from
`origin/sm155` (cdb38316; it already holds the `q6` factor label of SM152, so no merge and no source change).
Design (pre-registered in 03ac35ec before any row): `design.md`.

## What runs

- CPU only (as the long-card read-outs of SM148 and SM155): F6 `w8+lr2.16.4q6x8` for Big two-head (SM91s2)
  `sm91s2-big-showtime` and Big Instant-Only (SM99) `sm99-big-inst-e2`, the 50 long-card users of
  `artifacts/small-model/sm148/users-longcard.json`, `--engine stream --chunk-rows 16384`, SM155 first-use flags,
  3 processes at a time, memory guard 5.9 GB, quiet.json windows (scope all / cpu) respected between users.
- Queue (`scripts/r01_queue.py`, owner `sm165`, claim `cpu-heavy`, no GPU): ticket 1 `001605-sm165` = bitwise gate
  (users 2388 + 2883, Big two-head (SM91s2), F6 and T24, stock vs stream, plus T24 rows vs the reused rows);
  ticket 2 (next free id, see `cpu/queue.out`) = the 50 users. Purpose prefix `Big long cards with 6-bit factors (SM165): `.
- REF and T24 rows are reused (SM148 `cpu/sm148-longcard-rows.jsonl` in `wt/sm148`, 37 users; SM155
  `artifacts/small-model/sm155/firstuse/sm155-big-longcard-rows.jsonl`, 13 users); only F6 is scored.

## Process, log, patterns

- WMI PID 24708 (outer bash), started 04:44:17. Launcher log `cpu/launch-sm165.log` (raw stdout
  `cpu/launch-sm165.nohup`, queue output `cpu/queue.out`, per-user scorer output `cpu/users/score-<user>.out`, memory
  guard `cpu/memguard.log`, kills `cpu/memguard-kill`).
- Done: line `SM165 DONE (50 of 50 users)`, file `cpu/done`, rows merged into `cpu/sm165-f6-rows.jsonl`.
- Stopped or failed: line `SM165 STOP`, cause on an earlier line starting `STOP:`; files `cpu/failed` or `cpu/stopped`
  (`cpu/users/failed.txt`, `cpu/users/killed.txt`).
- Wait without polling (Bash, run_in_background):
  `until grep -qE 'SM165 (DONE|STOP)' C:/Users/jonat/workspace/wt/sm165/artifacts/small-model/sm165/cpu/launch-sm165.log; do sleep 120; done; tail -5 C:/Users/jonat/workspace/wt/sm165/artifacts/small-model/sm165/cpu/launch-sm165.log`
- Alive (PowerShell): `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*sm165*' } | Select ProcessId, Name`.
- ETA: 3,505,515 reviews x 2 models = 7.0M review-evaluations; SM155 measured about 620 to 660 review-evals/s per
  process for T24-like labels, so with 3 processes about 60 to 70 min after the gate (range 50 to 100 min):
  `SM165 DONE` around 05:50 to 06:00 on 11-25.
- Stop between users: `touch C:/Users/jonat/workspace/wt/sm165/artifacts/small-model/sm165/cpu/STOP` (running users
  finish; then `SM165 STOP`). To also kill running scorers, stop the python processes whose command line holds `sm165`.
- Resume (finished users `cpu/users/u-<user>.jsonl` and `cpu/gate/ok` are kept; remove `cpu/STOP` first; check no
  sm165 process is alive; the launcher refuses to start if `scripts/`, `rwkv/`, `expertium/` or the sm165 scripts are
  uncommitted):

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm165/artifacts/small-model/sm165/launch-sm165.sh >> C:/Users/jonat/workspace/wt/sm165/artifacts/small-model/sm165/cpu/launch-sm165.nohup 2>&1"'; ProcessStartupInformation=$si}
```

- Do not edit `scripts/`, `rwkv/`, `expertium/` or the sm165 files of this worktree while it runs.

## Launch check

LAUNCH_CHECK_PLACEHOLDER

## Read-out (fresh agent, Sonnet, medium effort)

1. `git -C C:/Users/jonat/workspace/wt/sm165 fetch` (merge `origin/sm165` if it moved). Wait for `SM165 DONE` or
   `SM165 STOP` (command above). On a STOP: read the `STOP:` line, `cpu/gate/compare.txt`, `cpu/gate/reuse.txt`,
   `cpu/users/failed.txt`, `cpu/memguard-kill`; if a few users failed for a transient reason, resume once with the
   command above; fix nothing in source; otherwise report to the main session.
2. Read-out (CPU, under a minute, under 1 GB):
   `cd C:/Users/jonat/workspace/wt/sm165/artifacts/small-model/sm165 && OMP_NUM_THREADS=2 C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 sm165_report.py report`
   -> `sm165-result.json`, `sm165-result.md`. Exit 0 = decided; 2 = pending (fewer than 48 users with F6 rows for a
   model); 3 = invalid (T24 vs REF no longer reproduces SM155, i.e. the reused rows changed). It prints one
   `[sm165] ... F6 vs REF ...: inside the tolerance | WARNING` line per model and head.
3. Write `RESULT.md` (short, full plain sentences), like `../sm155/RESULT.md`'s long-card section with F6 added:
   - one sentence on the question and the method (CPU stream engine, 50 long-card users, F6 only, REF / T24 reused);
   - the gate (`cpu/gate/compare.txt` last line, `cpu/gate/reuse.txt`);
   - a table on 128+ cards (reviews of cards with 128 or more earlier answers), from `sm165-result.json`
     `cost[...]["<pair>"]["128-"]`:

     | model, head | T24 vs REF (SM155) | F6 vs REF | F6 vs T24 |
     |---|---|---|---|
     | Big two-head (SM91s2), curve | -0.016% [-0.092, +0.086] | ... | ... |
     | Big two-head (SM91s2), instant | +0.026% [-0.084, +0.110] | ... | ... |
     | Big Instant-Only (SM99), instant | +0.006% [-0.105, +0.061] | ... | ... |

     plus coverage (users per label of 50) and the 128+ review count; point the other buckets to `sm165-result.md`;
   - one sentence on the tolerance: the user's 11-22 long-card tolerance is about 0.5% (point) and 1% (95% CI upper
     bound) vs int8 on 128+ cards; say whether each F6 vs REF cell is inside it; it is a warning only, never a
     blocker, and does not change the Big F6 go of the release-candidate re-check (SM164);
   - speed: review-evals/s per process from the rows' `model_seconds` (2 rows per user, 3 processes side by side).
   Naming: every model as "plain name (ID)" in every sentence; "%" never "points"; never the outside researcher's name.
4. Set the status line at the top of this file to DONE. Commit on `sm165` with `git add -f`: `RESULT.md`, `HANDOFF.md`,
   `sm165-result.json`, `sm165-result.md`, `cpu/sm165-f6-rows.jsonl`, `cpu/users/*.jsonl` (not `*.part.jsonl`),
   `cpu/users/*.out`, `cpu/gate/*.jsonl`, `cpu/gate/*.txt`, `cpu/gate/*.out`, `cpu/gate/ok`, `cpu/launch-sm165.log`,
   `cpu/launch-sm165.nohup`, `cpu/queue.out`, `cpu/memguard.log`, `cpu/done` (and `cpu/memguard-kill`,
   `cpu/users/failed.txt`, `cpu/users/killed.txt` if present). Never `work/`, `.npz`, `.pth`, `quiet.json`,
   `docs/GPU_OPTIMIZATIONS_LOG.md`, `.claude/`, `__pycache__/`. Identity
   `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending with a
   blank line and the Co-Authored-By line of the committing model; `git fetch origin && git merge origin/sm165` right
   before `git push origin sm165`. No force-push, no hook bypass.
5. Do not edit `pages/` (the main session does that). Report to the main session: the three 128+ F6 vs REF cells with
   CI, F6 vs T24, the tolerance sentence, coverage, and the commit.
