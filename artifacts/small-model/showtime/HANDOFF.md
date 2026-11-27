# Show-time scoring of the headline models: handoff (2026-11-08)

Goal: show-time (`e2s-query`, what the Anki fork gives at a query) rows for the Factory page's headline table. Leak-free
(`RWKV_DELETED_CARDS=neighbour`), w8+s8, 732 set (723 users + 3 long users) and, for three models, the 1,500 validation users.
Reused, not rescored: the release (SM63, sm72e25), the show-time retrain (SM92) and the big model (1 leak-free epoch) 732-set rows.

## What runs

- Launcher `artifacts/small-model/showtime/launch-showtime.sh`, WMI hidden, bash pid **78036**, started 2026-11-08 12:25.
- Log `artifacts/small-model/showtime/gpu/launch-showtime.log` (last line `[showtime ...] done` when everything finished; a
  `STOP:` line otherwise), scorer output `gpu/score.out`, nohup `launch.nohup`, dry-run log `gpu/launch-showtime-dryrun.log`.
- Rows (not committed, `git add -f` them): `gpu/showtime-rows.jsonl` (732 set + long users), `gpu/showtime-rows-1500.jsonl`.
- Queue owner `small-model`; purposes start with "show-time scoring of the headline models: ". It queues behind the grade
  order by construction (SM95) (`wt/sm95`): do not touch that launcher. The scorer runs from the clean worktree `wt/sm57`
  (1,500-pool option): nobody edits `scripts/ expertium/ rwkv/` there while this runs.
- Breaks: the launcher reads `artifacts/coordination/queue/quiet.json` plus the daily 20:00-01:00 (Sunday from 16:00 to 01:00
  Monday) before every ticket (`showtime_block.py`) and starts no ticket whose expected minutes cross a break. The first ticket
  waits for SM95's segment (to 16:00); the launcher then sleeps through the break and enqueues again at 01:00 (Monday).

## Steps (expected GPU minutes: measured 12 min per 723 users, 1 min for the 3 long users, 17 min for the 771 pool users)

| # | step | checkpoint | expected |
|---|---|---|---|
| 1 | instant-only small model (SM57) stage 1, 732 set + long | `wt/sm57/artifacts/small-model/sm57-inst-e26-dec-1ep32k-muon/expertium/E89-F65e-F77a/master.pth` | ~13 |
| 2 | instant-only small model (SM57) stage 4, 732 set + long | `wt/sm57/artifacts/small-model/sm57s4-inst-e30-dec-1ep32k-muon/expertium/E89-F65e-F77a/master.pth` | ~13 |
| 3 | 1,500 pool (771 users): stage 1, stage 4, the release (SM63) | the two above + `wt/sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/expertium/E89-F65e-F77a/master.pth` | ~3 x 17 |
| 4a | previous release, cool-down from epoch 20 (SM72e20) | `wt/sm72s/.../scale-w64kd-final-sc-lanes-leakfree-dec20-1ep32k-muon/...` | ~13 |
| 4b | earlier release, cool-down from epoch 15 (SM72) | `wt/sm72s/.../scale-w64kd-final-sc-lanes-leakfree-dec15-1ep32k-muon/...` | ~13 |
| 4c | small model, cool-down from epoch 15 (SM56) | `wt/sm56/.../scale-w64kd-final-sc-lanes-dec15-1ep32k-muon/...` | ~13 |
| 4d | previous release, cool-down from epoch 10 (SM56 dec10) | `wt/sm56/.../scale-w64kd-final-sc-lanes-dec10-1ep32k-muon/...` | ~13 |
| 4e | shipped big model (fork RWKV) | `C:/Users/jonat/workspace/anki/qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.pth` (`fork::`) | ~15-20 |

All eight checkpoints found (none missing). Total about 2.5 GPU hours, plus retry passes for users that do not fit at cap 0.8.
Expected end: Monday 2026-11-09 about 04:00 if the GPU is free from 01:00; later if SM95 segments interleave.
The smoke on 2 users was skipped: the GPU was busy (SM95 training, 13.9 of 16.3 GB used). The dry run printed every command and
resolved all checkpoints.

## Relaunch (steps with complete rows are skipped; a pass 2 picks up missing users)

```powershell
$H='C:/Users/jonat/workspace/anki-rwkv/artifacts/small-model/showtime'
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=('"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash ' + $H + '/launch-showtime.sh >> ' + $H + '/launch.nohup 2>&1"'); ProcessStartupInformation=$si}
```

Only relaunch if `tasklist //FI "PID eq 78036"` shows nothing and the log does not end with `done`. Stop: `touch gpu/STOP`.

## Follow-up (fresh agent, Sonnet medium is enough; do not wait on the GPU)

1. `git fetch` + `git merge --ff-only`; `tail -5 gpu/launch-showtime.log`; if `done`, go on; otherwise read the `STOP:` /
   `FAILED` line and the tail of `gpu/score.out`.
2. `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 artifacts/small-model/showtime/showtime_report.py`
   (CPU, seconds) writes `showtime-report.{json,md}`. `--check` re-derives the SM92 numbers from the old rows (passed 11-08).
3. Commit with `git add -f`: `gpu/showtime-rows.jsonl`, `gpu/showtime-rows-1500.jsonl`, `gpu/launch-showtime.log`,
   `showtime-report.json`, `showtime-report.md` (as `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`,
   message ending with a blank line + `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`), explicit paths, push main right
   after. Never stage `docs/GPU_OPTIMIZATIONS_LOG.md`, `.claude/`, `pages/`, `quiet.json`.
4. Reply with the tables of `showtime-report.md` (plain names with the ID in parentheses, never a bare ID; unit stated: points =
   % of the big model (1 leak-free epoch) normal-input LogLoss on 722 users; on the 1,500 pool % of the release's LogLoss; never
   name outside research sources). The main session updates the Factory page (`pages/data/factory-curated.json`), not this agent.
