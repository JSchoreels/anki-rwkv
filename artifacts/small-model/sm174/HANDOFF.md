# How many users for MIT weights (SM174): hand-off

Design: `design.md` (pre-registered). Branch `sm174`, worktree `C:/Users/jonat/workspace/wt/sm174`; run tree
`C:/Users/jonat/workspace/wt/sm174-run` (detached at cea0f931, **never edit it**: SourceWatch kills a running epoch).

## Status (launch Thu 2026-11-26 00:44)

- Launcher `launch-sm174.sh` (commit 8cbc821e), started hidden via WMI (`Win32_Process.Create`, ShowWindow 0) at
  00:44:32: WMI PID 10720 (outer bash). Every GPU step is one queue ticket, owner `sm174`, purpose prefix
  `How many users for MIT weights (SM174): `; the ticket ids appear in `r01_queue.py status` and in the queue
  folder. Smoke ticket 001636 OK (both arms), integrity ticket 001637 OK (max |LogLoss difference| vs SM148 rows 0), first
  training ticket **001638-sm174 `train n0500 e01`** ran 00:45-00:49: 38.7M reviews in 3.7 min, **172,700 reviews/s**, GPU
  reserved 9.2 GB, process tree RSS 9.5 GB; e02 started 00:49 (one ticket per epoch / slice eval / scoring).
- Log: `gpu/launch-sm174.log` (raw stdout `gpu/launch-sm174.nohup`; trainer output `gpu/train.out`; scorers
  `gpu/score.out`; smoke `gpu/smoke.out`; checks `gpu/check.out`; report `gpu/report.out`).
- Rows: `gpu/sm174-rows-732.jsonl` (final models, 732 set), `gpu/sm174-slice-rows.jsonl` (early-stop slice).
  Per-arm decisions: `gpu/<arm>-state.json`. Model folders: `C:/Users/jonat/workspace/wt/sm174-run/artifacts/small-model/sm174/<arm>/e01 ... dNN`.
- Alive (PowerShell): `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*sm174*' } | Select ProcessId, Name, CommandLine`.
  Queue: `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status`.

## Log patterns

| event | pattern in `gpu/launch-sm174.log` |
|---|---|
| an arm finished (trained, scored, report rewritten) | `SM174 n0500 DONE`, `SM174 n1000 DONE`, `SM174 n2000 DONE`, `SM174 n4000 DONE`, `SM174 n8000 DONE`, `SM174 n8000-kd DONE` |
| everything finished | `SM174 DONE` (a line alone, after `final report exit`) |
| stopped or failed | `SM174 STOP` (cause on an earlier line starting `STOP:`) |
| progress | `train: <arm> eNN`, `<arm> eNN complete: ... reviews/s`, `<arm> slice: {...}`, `<arm> decision: ...` |

Wait helper (one wake per arm): `until grep -qE "SM174 n0500 DONE|SM174 STOP" C:/Users/jonat/workspace/wt/sm174/artifacts/small-model/sm174/gpu/launch-sm174.log; do sleep 60; done` (Bash `run_in_background`).

## ETAs (upper bound, no early stop; measured speed replaces the 178k reviews/s guess after epoch 1)

| arm | GPU | ends about |
|---|---:|---|
| smoke + integrity | 0.2 h | 01:00 Thu 26 Nov |
| 500 users, no teacher (SM174 n0500) | 4.9 h | 06:00 Thu |
| 1,000 users, no teacher (SM174 n1000) | 4.6 h | 10:40 Thu |
| 2,000 users, no teacher (SM174 n2000) | 4.5 h | 15:10 Thu |
| 4,000 users, no teacher (SM174 n4000) | 4.4 h | 19:35 Thu |
| 8,000 users, no teacher (SM174 n8000) | 4.2 h | 23:45 Thu |
| 8,000 users, with the usual teacher (SM174 n8000-kd) | 4.2 h | **~04:00 Fri 27 Nov (all done)** |

## Stop / resume

- Stop: `touch C:/Users/jonat/workspace/wt/sm174/artifacts/small-model/sm174/gpu/STOP` (a training folder snapshots
  at the next user boundary; the launcher logs `SM174 STOP`). Quiet windows (quiet.json, scope all / gpu) are
  respected: no ticket starts into one, a training folder snapshot-stops <= 2 min before one and resumes after.
- Resume (after a STOP, a crash or a session restart that killed the launcher): `rm -f .../gpu/STOP`, check no
  sm174 process is alive (above), then start the same WMI line again. The launcher re-derives the next step from disk
  (finished folders skipped, a folder in flight resumes from its last snapshot, scored users skipped):
  `$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm174/artifacts/small-model/sm174/launch-sm174.sh >> C:/Users/jonat/workspace/wt/sm174/artifacts/small-model/sm174/gpu/launch-sm174.nohup 2>&1"'; ProcessStartupInformation=$si}`
- A decay segment interrupted mid-way resumes exactly (lr_resume segment). If `train.out` shows a crash that is not
  a stop, read the traceback before resuming.

## Read-out (fresh Sonnet agent, medium effort; one after each `SM174 <arm> DONE`, or one at the end)

1. `git -C C:/Users/jonat/workspace/wt/sm174 fetch` and merge `origin/sm174` if it moved.
2. `grep -E "STOP|DONE|decision|slice|complete:" C:/Users/jonat/workspace/wt/sm174/artifacts/small-model/sm174/gpu/launch-sm174.log | tail -40`
3. The launcher already rewrote `sm174-result.md` / `sm174-result.json` after the arm. To recompute by hand:
   `cd C:/Users/jonat/workspace/wt/sm174/artifacts/small-model/sm174 && OMP_NUM_THREADS=2 C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 sm174_report.py report` (exit 0 all arms, 2 partial).
4. Report to the main session: per arm the curve / instant gap vs the six-layer Tiny two-head release model (SM130)
   with 95% CI, the "users needed" table (+1.0% / +0.5%, point and CI upper, and vs the equal-compute control once
   it exists), the slice early-stop line, and reviews/s + GPU hours per arm. Always "plain name (ID)", "%" never
   "points", never name the outside research source.
5. Update the status line at the top of this file. Commit with `git add -f` (never `work/`, `.pth`, `.npz`, the
   run-tree folders, `quiet.json`, `docs/GPU_OPTIMIZATIONS_LOG.md`, `.claude/`): `HANDOFF.md`, `sm174-result.md`,
   `sm174-result.json`, `gpu/*-state.json`, `gpu/sm174-rows-732.jsonl`, `gpu/sm174-slice-rows.jsonl`,
   `gpu/integrity-orig-smoke.jsonl`, `gpu/smoke-ok`, `gpu/launch-sm174.log`, `gpu/launch-sm174.nohup`,
   `gpu/score.out`, `gpu/smoke.out`, `gpu/check.out`, `gpu/report.out` (and `gpu/train.out` if under ~20 MB). Identity
   `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending with a
   blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; `git fetch` + merge `origin/sm174` right
   before `git push origin sm174`. No force-push, no hook bypass. Do not edit the scripts while the launcher runs
   (it checks they are committed only at start, but keep them as launched).
6. At the end (`SM174 DONE`): add the result line to `docs/decisions/DECISIONS.md` (+ full entry) on `sm174`, and
   tell the main session the campaign-goal number (users for +1% and +0.5%, and how much of the gap is budget vs data
   vs teacher).
