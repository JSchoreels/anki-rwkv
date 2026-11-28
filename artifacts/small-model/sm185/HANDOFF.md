# Small cool-down with session inputs (SM185): handoff

Status: PREPARED (design, scripts, CPU proofs ALL OK); launch pending.

## Done

- Worktree `C:/Users/jonat/workspace/wt/sm185`, branch `sm185` from `origin/sm183` (`9c75b66a`). Design pre-registered in `design.md` before any GPU ticket.
- Run tree `C:/Users/jonat/workspace/wt/sm179-run` reused unedited (`cd220759`, clean). New folders there (git-ignored; never commit them): `artifacts/small-model/sm185-th-ses-p1`, `sm185-th-ctlcheck`, `sm185-io-ses-p1`, `sm185-io-ctlcheck` (pass-2 folders are prepared by the launcher). `sm185_prepare.py pair`: ok (rows, order, schedule, options, teacher targets, trim equal the release pass-1 folders; only the start differs).
- Recipes = the release cool-downs "dec2ep32k" of Small two-head (SM118) (from the show-time retrain of the release (SM92)'s epoch 26, teacher targets of the show-time big model (SM91s2), both heads) and Small Instant-Only (SM113) (from the Small Instant-Only model (SM98)'s epoch 35, teacher targets of the show-time instant-only big model (SM99), curve loss weight 0): pass 1 fresh Muon, first half of one 7e-4 -> 0 one_minus_sin decay over 2 x 581,942,724 reviews (8,000 users); pass 2 from pass 1's end, `random.Random(2)` order, second half. Session arm = the same with 6 zero columns at input 24.
- CPU proofs `check-cpu-sm185.json`: ALL OK (K1 8,000 training users + 723 scored users aligned, finite; K2 flag off run tree = `wt/sm118`'s tree and = `wt/sm113`'s tree bitwise; flag on step 0 = flag off, both chains). Report selftest on SM183's rows relabelled reproduces SM183's six changes exactly.

## Control choice

The release models are the controls if G0 passes: `sm185-th-ctlcheck` (SM118's pass 1 unchanged, flag unset, run tree) is trained to its 400-user snapshot and compared bit for bit with SM118's pass 1 (`check-ctl-sm185.json`). The log line `G0 control reproduction: pass|fail` says which. **If it fails the launcher stops** (`SM185 STOP`, cause on the `STOP:` line) before the session arm: a control arm `sm185-th-ctl` would add about 11.5 GPU h (relaunch with `export CONTROL_ARM=1;` only with the main session's OK). The Instant-Only chain has its own G0 (`check-ctl-io-sm185.json`, vs SM113's pass 1 attempt 1) at its start.

## Tickets (one launcher, owner sm185, purpose prefix `Small cool-down with session inputs (SM185): `, cap 0.8, 0.92 for scorer memory retries)

| order | step | expected |
|---|---|---|
| 1 | trainer smoke `sm185-th-ses-p1` | 1-5 min |
| 2 | G0: train `sm185-th-ctlcheck` to 400 users, snapshot stop, `sm185_check.py ctl --chain th` | 20-30 min |
| 3 | train `sm185-th-ses-p1` (two-head pass 1) | ~340 min |
| 4 | prepare + pair + train `sm185-th-ses-p2` | ~340 min |
| 5 | score `sm118-th-release` and `sm185-th-ses` on the 732 set, then on the decisive users (with dumps) | ~70 min |
| 6 | read-out, log `SM185 TH` | 1 min CPU |
| 7 | Instant-Only chain only if minutes used + 740 <= `BUDGET_TOTAL` (960): normally skipped, log `Instant-Only chain not run (budget ...)` | (~740 min) |
| 8 | `SM185 DONE` | |

ETA: `SM185 TH` and `SM185 DONE` about 08:00 Sun 11-29 (07:30-09:30) if launched about 18:15 Sat; later if other tickets interleave. About 13.2 GPU hours in total (two-head only).

## Paths and patterns

- Log: `C:/Users/jonat/workspace/wt/sm185/artifacts/small-model/sm185/gpu/launch-sm185-gpu.log` (raw stdout `gpu/launch-sm185-gpu.nohup`; trainers `gpu/train-<folder>.out`; scorer `gpu/score.out`; smoke `gpu/smoke.out`; GPU minutes `gpu/gpu-minutes.jsonl`; report `gpu/report.out`).
- Rows: `gpu/sm185-rows-732.jsonl`, `gpu/sm185-rows-decisive.jsonl`. Dumps: `C:/Users/jonat/workspace/wt/sm185/work/sm185-dump/{732,decisive}/` (never committed; `work/` is NOT git-ignored, so never `git add -A`).
- Patterns: two-head read-out `SM185 TH`; done `SM185 DONE`; stopped or failed `SM185 STOP` (cause on an earlier line starting `STOP:`); `G0 control reproduction:` gives the control choice.
- Alive (PowerShell): `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*sm185*' } | Select ProcessId, Name`. Queue: `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe C:/Users/jonat/workspace/anki-rwkv/scripts/r01_queue.py status`.
- Wait without polling (Bash, run_in_background): `until grep -qE 'SM185 (TH|DONE|STOP)' C:/Users/jonat/workspace/wt/sm185/artifacts/small-model/sm185/gpu/launch-sm185-gpu.log; do sleep 120; done; tail -6 C:/Users/jonat/workspace/wt/sm185/artifacts/small-model/sm185/gpu/launch-sm185-gpu.log`
- Stop: `touch C:/Users/jonat/workspace/wt/sm185/artifacts/small-model/sm185/gpu/STOP` (a running training snapshots at the next user boundary and exits; no new ticket starts). A queue cancel does not stop the launcher: touch STOP first.
- Resume after a stop or a machine restart (finished passes, snapshots, scored users, `gpu/smoke-*-ok`, `gpu/g0-*-done` are kept; remove `gpu/STOP` first; check that no sm185 process is alive):
  `$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm185/artifacts/small-model/sm185/launch-sm185-gpu.sh >> C:/Users/jonat/workspace/wt/sm185/artifacts/small-model/sm185/gpu/launch-sm185-gpu.nohup 2>&1"'; ProcessStartupInformation=$si}`
  (Without the `>> ... 2>&1` redirect the hidden bash hangs before its first line.) Instant-Only follow-up (main session's OK only): the same command with `export BUDGET_TOTAL=1800;` before `bash` (the two-head steps are skipped as done; then smoke, G0 io, two passes, scoring).
- Never edit `scripts/`, `rwkv/`, `expertium/` of `wt/sm179-run` while this runs (SourceWatch stops the training). Do not move or clean `wt/sm118`, `wt/sm113`, `wt/sm92`, `wt/sm98`, `wt/sm91s2`, `wt/sm99`, `wt/sm164`, `wt/sm183`, `wt/sm72s` or the kd-replay caches (read by this run).

## Read-out steps (fresh agent, Sonnet, medium effort; after `SM185 TH` / `SM185 DONE`)

1. `git -C C:/Users/jonat/workspace/wt/sm185 fetch` (merge `origin/sm185` if it moved). On a STOP, read the `STOP:` line and the tail of the matching `gpu/*.out`; fix nothing in source while unsure; report to the main session.
2. The report line in the log gives exit 0 (decided) or 2 (pending) and the verdict lines. Tables: `sm185-result.json`, `sm185-result.md`. By hand: `cd C:/Users/jonat/workspace/wt/sm185/artifacts/small-model/sm185 && OMP_NUM_THREADS=2 C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 sm185_report.py report` (about a minute; `--no-auc` skips the dumps). If pending: read `chains.<c>.sets.<set>.integrity`, `rule_users`, `nonfinite_users`; for missing users rerun the launcher once (resume command) and report.
3. Write `RESULT.md` from `sm185-result.json`: the question; the control choice (G0 result, `check-ctl-sm185.json`); the rules (design.md "Rules", plain sentences); per chain the verdict and per set the change vs the control with CI on both heads, each control's LogLoss; the gain vs the short fine-tune of Small two-head (SM118) in SM181 (-0.065% instant) and vs the Tiny cool-down with session inputs (SM183) (-0.499% / -0.474% instant); answer-count buckets (report only); the AUC table (report only, not part of the rule); per pass reviews/s, minutes and peak memory vs the release passes (SM118: 335.4 min at 28,915 reviews/s, then 338.5 min in two segments at 28,690), GPU minutes from `gpu/gpu-minutes.jsonl`; whether the Instant-Only chain ran (budget). Every model as "plain name (ID)", e.g. Small two-head (SM118); "%" never "points"; full plain sentences; never name the outside research's author.
4. Update the status at the top of this file and the A18 row of `docs/ACCURACY_IDEAS.md`. Commit with `git add -f`: `RESULT.md`, `HANDOFF.md`, `sm185-result.json`, `sm185-result.md`, `check-ctl-sm185.json`, `gpu/sm185-rows-*.jsonl`, `gpu/smoke-*-ok`, `gpu/g0-*-done`, `gpu/gpu-minutes.jsonl`, `gpu/launch-sm185-gpu.log`, `gpu/launch-sm185-gpu.nohup`, `gpu/score.out`, `gpu/smoke.out`, `gpu/train-*.out`, `gpu/report.out`, `docs/ACCURACY_IDEAS.md`. Never `work/`, `data/`, `.npz`, `.pth`, the run-tree folders, `quiet.json`, `.claude/`. Identity `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; `git fetch` + merge `origin/sm185` right before the push; push `sm185`. No force-push, no hook bypass.
5. Do NOT edit `pages/` (the main session does that). Report: the verdict, per set and head the change with CI, AUC (secondary), reviews/s per pass, GPU hours, the commit.
