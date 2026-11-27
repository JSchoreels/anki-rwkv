# Session inputs confirmation (SM181): handoff

Status: RUNNING since Sat 2026-11-28 00:44 (launcher started hidden via WMI, PID 51148, outer bash). First ticket 001831-sm181 (trainer smoke) registered at 00:44:34, waiting behind SM180's 001830 (`train sm180-sm118-ctl`, started about 00:36, about 40 min).

## Done

- Worktree `C:/Users/jonat/workspace/wt/sm181`, branch `sm181` from `origin/sm179` (`8aab3a56`). Design pre-registered in `design.md` (two axes, rules, AUC as a reported secondary metric) before any scoring row existed.
- Run tree `C:/Users/jonat/workspace/wt/sm179-run` reused unedited (`cd220759`, clean). New arm folder there: `artifacts/small-model/sm181-small-ses` (ignored by git; never commit it: `.pth`, snapshots, `start/`). `sm181_prepare.py pair`: ok (rows, schedule, options, seed equal SM149's control).
- CPU proofs `check-cpu-sm181.json`: ALL OK (K1 show-time data of the 794 Small training users and the 765 decisive users aligned, 0 inversions; K2 Small flag off = stock bitwise, flag on step 0 = release; K3 scorer padded start = release for Small (732) and Tiny (`--pool 1500`), max |difference| 0.0).

## Arms and tickets (one launcher, owner sm181, purpose prefix `Session inputs confirmation (SM181): `, cap 0.8, 0.92 for memory retry passes)

| order | ticket | expected |
|---|---|---|
| 1 | trainer smoke, Small session arm | 3 min |
| 2 | axis 1: score six-layer Tiny two-head (SM130) control `sm171-tiny-ctl` (flag unset) on the 765 decisive users, `--pool 1500`, with the dump | 15 min + retries |
| 3 | axis 1: score SM179's session arm `sm179-tiny-ses` (flag `session`), same | 15 min + retries |
| 4 | axis 2: train `sm181-small-ses` (Small two-head (SM118), 794 users, 67.3M reviews; control took 38.8 min) | 40 min |
| 5 | axis 2: score SM149's control `sm149-sm118-ctl` on the 732 set with the dump | 15 min + retries |
| 6 | axis 2: score `sm181-small-ses` on the 732 set with the dump | 15 min + retries |
| 7-8 | report only: rescore both Tiny models on the 732 set with the dump (overall AUC there) | 10 min each |
| 9 | read-out `sm181_report.py report` (CPU, about a minute) | |

ETA: about 2.4 GPU hours after SM180's running ticket ends (about 01:15), so `SM181 DONE` around 03:40 on 11-28 if no other ticket interleaves; around 04:30 to 05:00 if SM180 queues its scoring tickets in between (the queue serialises them). Budget 5 GPU hours.

## Paths and patterns

- Log: `C:/Users/jonat/workspace/wt/sm181/artifacts/small-model/sm181/gpu/launch-sm181-gpu.log` (raw stdout `gpu/launch-sm181-gpu.nohup`; trainer `gpu/train-sm181-small-ses.out`; scorer `gpu/score.out`; smoke `gpu/smoke.out`).
- Rows: `gpu/sm181-rows-decisive.jsonl`, `gpu/sm181-rows-732.jsonl`, `gpu/sm181-rows-tiny732.jsonl`. Dumps: `C:/Users/jonat/workspace/wt/sm181/work/sm181-dump/{decisive,small732,tiny732}/` (never committed; `work/` is NOT git-ignored in this worktree, so never `git add -A`).
- Patterns in the log: done `SM181 DONE` (the line before it gives the report's exit code and the verdict line); stopped or failed `SM181 STOP` (the cause is on an earlier line starting `STOP:`).
- Alive (PowerShell): `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*sm181*' } | Select ProcessId, Name`. Queue: `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status`.
- Wait without polling (Bash, run_in_background): `until grep -qE 'SM181 (DONE|STOP)' C:/Users/jonat/workspace/wt/sm181/artifacts/small-model/sm181/gpu/launch-sm181-gpu.log; do sleep 120; done; tail -6 C:/Users/jonat/workspace/wt/sm181/artifacts/small-model/sm181/gpu/launch-sm181-gpu.log`
- Stop: `touch C:/Users/jonat/workspace/wt/sm181/artifacts/small-model/sm181/gpu/STOP` (a running training snapshots at the next user boundary and exits; no new ticket starts). A queue cancel does not stop the launcher: touch STOP first.
- Resume after a stop or a machine restart (finished scoring users, the trained arm or its snapshots, and `gpu/smoke-ok` are kept; remove `gpu/STOP` first; check that no sm181 process is still alive):
  `$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm181/artifacts/small-model/sm181/launch-sm181-gpu.sh >> C:/Users/jonat/workspace/wt/sm181/artifacts/small-model/sm181/gpu/launch-sm181-gpu.nohup 2>&1"'; ProcessStartupInformation=$si}`
  (Without the `>> ... 2>&1` redirect the hidden bash hangs before its first line.)
- The report-only Tiny 732 rescoring (steps 7-8) can be skipped if the GPU is needed: touch STOP after step 6 and run the report by hand (below); the tiny732 rows are then simply missing.

## Read-out steps (fresh agent, Sonnet, medium effort)

1. `git -C C:/Users/jonat/workspace/wt/sm181 fetch` (merge `origin/sm181` if it moved). Wait for `SM181 DONE` or `SM181 STOP` (command above). On a STOP, read the `STOP:` line and the tail of the matching `gpu/*.out`; fix nothing in source while unsure; report to the main session.
2. If the report line says exit 2 (pending): read `sm181-result.json` (`axes.<axis>.integrity`, `rule_users`, `users_per_label`, `nonfinite_users`); for missing users rerun the launcher once (resume command) and report. Otherwise the tables are written: `sm181-result.json`, `sm181-result.md`. To recompute by hand: `cd C:/Users/jonat/workspace/wt/sm181/artifacts/small-model/sm181 && OMP_NUM_THREADS=2 C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 sm181_report.py report` (about a minute, about 1 GB; `--no-auc` skips the dumps).
3. Write `RESULT.md` from `sm181-result.json`: the question; the rules (`design.md` "Rules", in plain sentences); the verdict (CONFIRMED / PARTIAL / NOT CONFIRMED) and per axis (decisive Tiny, Small 732) the change vs the control with CI on both heads and whether it is confirmed; each control's LogLoss; the answer-count buckets (report only); Tiny on the 732 set and the decisive users together (report only); **the secondary AUC table** (`auc_secondary` per axis and `report_only.tiny732`): per head the overall AUC and the within-user AUC of control and arm and the change with its 95% CI (absolute difference and % of the control), the users with dumps and with both outcomes, stated as reported only, not part of the go rule; Small training reviews/s and minutes vs the control's 28,883 reviews/s / 38.8 min (quote reviews/s), GPU minutes per ticket from the log; the integrity lines. Every model as "plain name (ID)", e.g. six-layer Tiny two-head (SM130), Small two-head (SM118); "%" never "points"; full plain sentences; never name the outside research's author ("outside research").
4. Update the status at the top of this HANDOFF.md and the A18 row of `docs/ACCURACY_IDEAS.md` (confirmed or not, with the numbers). Commit with `git add -f`: `RESULT.md`, `HANDOFF.md`, `sm181-result.json`, `sm181-result.md`, `gpu/sm181-rows-*.jsonl`, `gpu/smoke-ok`, `gpu/launch-sm181-gpu.log`, `gpu/launch-sm181-gpu.nohup`, `gpu/score.out`, `gpu/smoke.out`, `gpu/train-*.out`, `docs/ACCURACY_IDEAS.md`. Never `work/` (the dumps), `data/`, `.npz`, `.pth`, the run-tree folders, `quiet.json`, `docs/GPU_OPTIMIZATIONS_LOG.md`, `.claude/`. Identity `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; `git fetch` + merge `origin/sm181` right before the push; push `sm181`. No force-push, no hook bypass.
5. Do NOT edit `pages/` or the factory page (the main session does that). Report to the main session: the verdict, per axis and head the change vs the control with CI, the AUC changes (secondary), Small reviews/s, and the commit.
