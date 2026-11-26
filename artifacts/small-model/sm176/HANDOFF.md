# How many users for an MIT Small (SM176): hand-off

Design: `design.md` (pre-registered; a diff vs How many users for MIT weights (SM174)). Branch `sm176`, worktree
`C:/Users/jonat/workspace/wt/sm176`; run tree `C:/Users/jonat/workspace/wt/sm174-run` (SM174's, detached at cea0f931,
reused, **never edit it**: SourceWatch kills a running epoch). Model folders:
`C:/Users/jonat/workspace/wt/sm174-run/artifacts/small-model/sm176/<arm>/e01 ... dNN`.

## Status: RUNNING (launched Thu 2026-11-26 19:44)

## Launch

- Launcher `launch-sm176.sh` (commit c6270c0d, dry run OK), started hidden via WMI (`Win32_Process.Create`,
  ShowWindow 0) at 19:44:44: WMI PID 47108 (outer bash). Every GPU step is one queue ticket, owner `sm176`, purpose
  prefix `How many users for an MIT Small (SM176): `, cap 0.8; ticket ids in `r01_queue.py status` and the queue folder.
- Smoke ticket **001734-sm176** OK (both arms, 2 updates each), integrity ticket **001735-sm176** OK (SM118 REF on
  the 3 smoke users, max |LogLoss difference| vs SM148 rows 0), first training ticket **001736-sm176
  `train n8000-kd e01`** started 19:45:25: after 2 min ~98,300 reviews/s (the 87k basis was conservative; epoch ETA
  ~99 min), GPU 11.3 GB used, 91% busy. One ticket per epoch / slice eval / scoring after that.
- Log: `gpu/launch-sm176.log` (raw stdout `gpu/launch-sm176.nohup`; trainer `gpu/train.out`; scorers `gpu/score.out`;
  smoke `gpu/smoke.out`; checks `gpu/check.out`; report `gpu/report.out`). Dry-run log `gpu/launch-sm176-dryrun.log`.
- Rows: `gpu/sm176-rows-732.jsonl` (final models, 732 set), `gpu/sm176-slice-rows.jsonl` (early-stop slice).
  Per-arm decisions: `gpu/<arm>-state.json`.
- Alive (PowerShell): `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*sm176*' } | Select ProcessId, Name, CommandLine`.
  Queue: `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status`.

## Log patterns (`C:/Users/jonat/workspace/wt/sm176/artifacts/small-model/sm176/gpu/launch-sm176.log`)

| event | pattern |
|---|---|
| an arm finished (trained, scored, report rewritten) | `SM176 n8000-kd DONE`, `SM176 n1000 DONE`, `SM176 n2000 DONE`, `SM176 n4000 DONE` |
| everything finished | `SM176 DONE` (a line alone, after `final report exit`) |
| stopped or failed | `SM176 STOP` (cause on an earlier line starting `STOP:`) |
| progress | `train: <arm> eNN`, `<arm> eNN complete: ... reviews/s`, `<arm> slice: {...}`, `<arm> decision: ...` |

Wait helper (one wake per arm, Bash `run_in_background`):
`until grep -qE "SM176 n8000-kd DONE|SM176 STOP" C:/Users/jonat/workspace/wt/sm176/artifacts/small-model/sm176/gpu/launch-sm176.log; do sleep 60; done`
(next arms: replace `n8000-kd` by `n1000`, `n2000`, `n4000`; at the end `^\[.*\] SM176 DONE$`).

## ETAs (87k reviews/s basis; expected = SM174's early-stop ratios, upper bound = no early stop)

| arm | GPU expected (upper) | ends about, expected (upper bound) |
|---|---:|---|
| smoke + integrity | 0.3 h | 20:05 Thu 26 Nov |
| 8,000 users, with the usual teacher (SM176 n8000-kd) | ~7 h (8.0 h) | 03:30 Fri (04:20 Fri) |
| 1,000 users, no teacher (SM176 n1000) | ~6 h (8.5 h) | 09:30 Fri (12:50 Fri) |
| 2,000 users, no teacher (SM176 n2000) | ~4.5 h (8.3 h) | 14:00 Fri (21:10 Fri) |
| 4,000 users, no teacher (SM176 n4000) | ~4 h (8.1 h) | **18:00 Fri 27 Nov (05:15 Sat 28 Nov), all done** |

## Stop / resume

- Stop: `touch C:/Users/jonat/workspace/wt/sm176/artifacts/small-model/sm176/gpu/STOP` (a training folder snapshots
  at the next user boundary; the launcher logs `SM176 STOP`). Quiet windows (quiet.json) are respected as in SM174.
- Resume (after a STOP, a crash or a session restart that killed the launcher): `rm -f .../gpu/STOP`, check no sm176
  process is alive (above), then start the same WMI line again; the launcher re-derives the next step from disk:
  `$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm176/artifacts/small-model/sm176/launch-sm176.sh >> C:/Users/jonat/workspace/wt/sm176/artifacts/small-model/sm176/gpu/launch-sm176.nohup 2>&1"'; ProcessStartupInformation=$si}`
- If `train.out` shows a crash that is not a stop, read the traceback before resuming.

## Read-out (fresh Sonnet agent, medium effort; one after each `SM176 <arm> DONE`, or one at the end)

1. `git -C C:/Users/jonat/workspace/wt/sm176 fetch` and merge `origin/sm176` if it moved.
2. `grep -E "STOP|DONE|decision|slice|complete:" C:/Users/jonat/workspace/wt/sm176/artifacts/small-model/sm176/gpu/launch-sm176.log | tail -40`
3. The launcher already rewrote `sm176-result.md` / `sm176-result.json` after the arm. By hand:
   `cd C:/Users/jonat/workspace/wt/sm176/artifacts/small-model/sm176 && OMP_NUM_THREADS=2 C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 sm176_report.py report` (exit 0 all arms, 2 partial).
4. Report to the main session: per arm the curve / instant gap vs the Small two-head release model (SM118) and vs the
   same-budget teacher control (SM176 n8000-kd) with 95% CI; the "users needed" table (+1.0% / +0.5%; the
   equal-compute column is the headline, as in SM174); the slice early-stop line; reviews/s + GPU hours per arm; and
   the side-by-side with SM174's Tiny numbers (`../sm174/RESULT.md`). Always "plain name (ID)", "%" never "points",
   never name the outside research source.
5. Update the status line here. Commit with `git add -f` (never `work/`, `.pth`, `.npz`, run-tree folders, `quiet.json`,
   `docs/GPU_OPTIMIZATIONS_LOG.md`, `.claude/`): `HANDOFF.md`, `sm176-result.md`, `sm176-result.json`,
   `gpu/*-state.json`, `gpu/sm176-rows-732.jsonl`, `gpu/sm176-slice-rows.jsonl`, `gpu/integrity-orig-smoke.jsonl`,
   `gpu/smoke-ok`, `gpu/launch-sm176.log`, `gpu/launch-sm176.nohup`, `gpu/score.out`, `gpu/smoke.out`,
   `gpu/check.out`, `gpu/report.out` (and `gpu/train.out` if under ~20 MB). Identity
   `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending with a
   blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; `git fetch` + merge `origin/sm176` right
   before `git push origin sm176`. No force-push, no hook bypass. Do not edit the scripts while the launcher runs.
6. At the end (`SM176 DONE`): write `RESULT.md` (as SM174's), add the result line to `docs/decisions/DECISIONS.md`
   (+ full entry) on `sm176`, and give the main session the users-needed numbers for Small next to Tiny's.
