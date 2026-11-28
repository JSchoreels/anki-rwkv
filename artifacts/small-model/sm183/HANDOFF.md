# Tiny cool-down with session inputs (SM183): handoff

Status: RUNNING since 2026-11-28 05:42 (launcher pid 49564, hidden via WMI). First ticket 001852 (two-head trainer smoke) started 05:42:52.

## Done

- Worktree `C:/Users/jonat/workspace/wt/sm183`, branch `sm183` from `origin/sm181` (`8d406426`). Design pre-registered in `design.md` before any GPU ticket.
- Run tree `C:/Users/jonat/workspace/wt/sm179-run` reused unedited (`cd220759`, clean). New folders there (git-ignored; never commit them): `artifacts/small-model/sm183-th-ses-p1`, `sm183-th-ctlcheck`, `sm183-io-ses-p1` (pass-2 folders are prepared by the launcher). `sm183_prepare.py pair`: ok (rows, order, schedule, options, teacher equal SM130's pass-1 folders; only the start differs).
- Recipe = six-layer Tiny (SM130)'s release cool-down "dec2ep32k": pass 1 from recovery epoch 32 (two-head) / 41 (Instant-Only), fresh Muon, first half of one 7e-4 -> 0 one_minus_sin decay over 2 x 581,942,724 reviews (8,000 users), the show-time teachers' replay targets (SM91s2 / SM99); pass 2 from pass 1's end, `random.Random(2)` order, second half. Session arm = the same with 6 zero columns at input 24.
- CPU proofs `check-cpu-sm183.json`: ALL OK (K1 8,000 training users + 723 scored users aligned, 0 inversions; K2 flag off run tree = SM130's tree bitwise, flag on step 0 = flag off, both chains). Report selftest on SM181's rows relabelled reproduces SM181 exactly.

## Control choice

The release models are the controls if G0 passes: `sm183-th-ctlcheck` (SM130's pass 1 unchanged, flag unset, run tree) is trained to its 400-user snapshot and compared bit for bit with SM130's pass 1 (`check-ctl-sm183.json`). If G0 fails, the launcher trains a two-head control arm `sm183-th-ctl` and drops Instant-Only. The log line `G0 control reproduction: pass|fail` says which.

## Tickets (one launcher, owner sm183, purpose prefix `Tiny cool-down with session inputs (SM183): `, cap 0.8, 0.92 for scorer memory retries)

| order | step | expected |
|---|---|---|
| 1 | trainer smoke `sm183-th-ses-p1` | 3-8 min |
| 2 | G0: train `sm183-th-ctlcheck` to 400 users, snapshot stop, `sm183_check.py ctl` | 12-15 min |
| 3 | train `sm183-th-ses-p1` (two-head pass 1) | ~150 min |
| 4 | prepare + pair + train `sm183-th-ses-p2` | ~151 min |
| 5 | score `sm130-th-release` and `sm183-th-ses` on the 732 set, then on the decisive users (with dumps) | ~80 min |
| 6 | read-out, log `SM183 TH` | 1 min CPU |
| 7 | only if two-head used <= 400 GPU min: smoke, `sm183-io-ses-p1`, `-p2`, score `sm130-io-release` / `sm183-io-ses` on the 732 set (decisive only if <= 680 min used, normally skipped) | ~330 min |
| 8 | read-out, `SM183 DONE` | |

ETA: `SM183 TH` about 13:00 Sat 11-28 (12:30-14:30); `SM183 DONE` about 19:00 Sat 11-28 (18:00-21:00); later if other tickets interleave. About 12 GPU hours in total.

## Paths and patterns

- Log: `C:/Users/jonat/workspace/wt/sm183/artifacts/small-model/sm183/gpu/launch-sm183-gpu.log` (raw stdout `gpu/launch-sm183-gpu.nohup`; trainers `gpu/train-<folder>.out`; scorer `gpu/score.out`; smoke `gpu/smoke.out`; GPU minutes `gpu/gpu-minutes.jsonl`; report `gpu/report.out`).
- Rows: `gpu/sm183-rows-732.jsonl`, `gpu/sm183-rows-decisive.jsonl`. Dumps: `C:/Users/jonat/workspace/wt/sm183/work/sm183-dump/{732,decisive}/` (never committed; `work/` is NOT git-ignored, so never `git add -A`).
- Patterns: two-head read-out written `SM183 TH`; done `SM183 DONE`; stopped or failed `SM183 STOP` (cause on an earlier line starting `STOP:`); `G0 control reproduction:` gives the control choice.
- Alive (PowerShell): `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*sm183*' } | Select ProcessId, Name`. Queue: `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status`.
- Wait without polling (Bash, run_in_background): `until grep -qE 'SM183 (TH|DONE|STOP)' C:/Users/jonat/workspace/wt/sm183/artifacts/small-model/sm183/gpu/launch-sm183-gpu.log; do sleep 120; done; tail -6 C:/Users/jonat/workspace/wt/sm183/artifacts/small-model/sm183/gpu/launch-sm183-gpu.log`
- Stop: `touch C:/Users/jonat/workspace/wt/sm183/artifacts/small-model/sm183/gpu/STOP` (a running training snapshots at the next user boundary and exits; no new ticket starts). A queue cancel does not stop the launcher: touch STOP first.
- Resume after a stop or a machine restart (finished passes, snapshots, scored users, `gpu/smoke-*-ok`, `gpu/g0-done` are kept; remove `gpu/STOP` first; check that no sm183 process is alive):
  `$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm183/artifacts/small-model/sm183/launch-sm183-gpu.sh >> C:/Users/jonat/workspace/wt/sm183/artifacts/small-model/sm183/gpu/launch-sm183-gpu.nohup 2>&1"'; ProcessStartupInformation=$si}`
  (Without the `>> ... 2>&1` redirect the hidden bash hangs before its first line.) The Instant-Only budget gates can be overridden with `BUDGET_IO=<min>` / `BUDGET_IO_DEC=<min>` in that command line (`export BUDGET_IO=...;` before `bash`), only with the main session's OK.
- Never edit `scripts/`, `rwkv/`, `expertium/` of `wt/sm179-run` while this runs (SourceWatch stops the training).

## Read-out steps (fresh agent, Sonnet, medium effort; after `SM183 TH`, again after `SM183 DONE`)

1. `git -C C:/Users/jonat/workspace/wt/sm183 fetch` (merge `origin/sm183` if it moved). On a STOP, read the `STOP:` line and the tail of the matching `gpu/*.out`; fix nothing in source while unsure; report to the main session.
2. The report line in the log gives exit 0 (decided) or 2 (pending) and the verdict lines. Tables: `sm183-result.json`, `sm183-result.md`. By hand: `cd C:/Users/jonat/workspace/wt/sm183/artifacts/small-model/sm183 && OMP_NUM_THREADS=2 C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 sm183_report.py report` (about a minute; `--no-auc` skips the dumps). If pending: read `chains.<c>.sets.<set>.integrity`, `rule_users`, `nonfinite_users`; for missing users rerun the launcher once (resume command) and report.
3. Write `RESULT.md` from `sm183-result.json`: the question; the control choice (G0 result, `check-ctl-sm183.json`); the rules (design.md "Rules", plain sentences); per chain the verdict and per set the change vs the control with CI on both heads, each control's LogLoss; the gain vs the short fine-tune's -0.065% (SM179 / SM181); answer-count buckets (report only); the AUC table (report only, not part of the rule); per pass reviews/s, minutes and peak memory vs SM130's passes (150.3 / 150.8 min, 64,547 / 64,303 reviews/s two-head; 147.0 / 147.6 min Instant-Only), GPU minutes from `gpu/gpu-minutes.jsonl`. Every model as "plain name (ID)", e.g. six-layer Tiny two-head (SM130); "%" never "points"; full plain sentences; never name the outside research's author.
4. Update the status at the top of this file and the A18 row of `docs/ACCURACY_IDEAS.md`. Commit with `git add -f`: `RESULT.md`, `HANDOFF.md`, `sm183-result.json`, `sm183-result.md`, `check-ctl-sm183.json`, `gpu/sm183-rows-*.jsonl`, `gpu/smoke-*-ok`, `gpu/g0-done`, `gpu/gpu-minutes.jsonl`, `gpu/launch-sm183-gpu.log`, `gpu/launch-sm183-gpu.nohup`, `gpu/score.out`, `gpu/smoke.out`, `gpu/train-*.out`, `gpu/report.out`, `docs/ACCURACY_IDEAS.md`. Never `work/`, `data/`, `.npz`, `.pth`, the run-tree folders, `quiet.json`, `.claude/`. Identity `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; `git fetch` + merge `origin/sm183` right before the push; push `sm183`. No force-push, no hook bypass.
5. Do NOT edit `pages/` (the main session does that). Report: the verdict per chain, per set and head the change with CI, AUC (secondary), reviews/s per pass, GPU hours, the commit.
