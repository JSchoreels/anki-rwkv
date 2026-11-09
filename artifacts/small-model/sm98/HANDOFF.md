# The show-time instant-only model, stage 2 (SM98): handoff (2026-11-09 12:15, prepared, NOT launched)

## Question

The show-time instant-only small model (SM96) went GO (11:54 Mon): it beat the show-time retrain of the release (SM92) on
show-time instant LogLoss by -0.214 points [-0.233, -0.195] (722 users of the 732 set; points = % of the big model
(1 leak-free epoch) normal-input instant LogLoss 0.275757). Its chain had one constant-LR instant-only epoch on show-time
inputs (epoch 31) before the cool-down. Do 4 more such epochs (32-35) before the same cool-down help further? Precedent:
in the instant-only small model (SM57), going from 1 to 4 more instant-only epochs (stage 1 -> stage 4) gained -0.071
points of instant.

## Design

| step | the show-time instant-only model, stage 2 (SM98) | reference |
|---|---|---|
| start | the show-time instant-only small model (SM96)'s constant-LR instant-only epoch 31 on show-time inputs (`wt/sm96/artifacts/small-model/sm96-inst-e31-st`, read only, master sha256 f49c36fa5bff, optimizer 845d2cb2aafd, update offset end 684,418) | |
| **epochs 32, 33, 34, 35** | exactly epoch 31's recipe with the epoch index advanced (`sm98_epoch.py`): 8k final tier, the run's order rule `random.Random(split_seed * 1000 + N)` over the 8,000 train users, epoch 25's label-window trims, 2048 x 16 lanes carry, leak-free replay KD, AdamW 7e-4 constant, `"curve_loss_weight": 0.0`, `"show_time_intervals": true`, from the previous epoch's weights + AdamW state + update offset. Options copied unchanged from the previous epoch. Folders `sm98-inst-e{32..35}-st` in the sm98 worktree; epoch 32 prepared now, each later one by the launcher after the previous completes (`prepare --epoch N`, then `pair --epoch N`; a failure stops the launcher) | SM96's epoch 31 step (from SM57 stage 4's epoch 30) |
| cool-down | the release's one-epoch 32k Muon cool-down (8,000 users, 581,942,724 reviews after trims, fresh Muon `expertium-iter53`, LR 7e-4 -> 0 one_minus_sin, KD replay `big-leakfree-1ep-final8k-neighbour` alpha 0.9), `"curve_loss_weight": 0.0` + `"show_time_intervals": true` (`extra-options.json` == SM96's, bytes), **from epoch 35**. Folder `sm98-inst-e35st-st-dec-1ep32k-muon`, prepared by the launcher after epoch 35 | SM96's cool-down `sm96-inst-e31st-st-dec-1ep32k-muon` (only the start differs) |
| label | `sm98-inst-e35st-st-dec1ep32k/w8+s8` (show-time rows `@e2s-query`) | |

## Pre-registered rule (`sm98_rule.py`, fixed before launch)

- **Primary (verdict):** new - the show-time instant-only small model (SM96), instant head, both on show-time inputs
  (`e2s-query`), 722 users of the 732 set, review-weighted, paired user bootstrap 2,000, seed 0, 95% CI, unit points
  (% of the big model (1 leak-free epoch) normal-input instant LogLoss 0.275757). **go** = CI entirely below 0 AND gain
  >= 0.05 points; **no go** otherwise.
- **Secondary (reported):** new - the show-time retrain of the release (SM92) (show-time; normal); new - the instant-only
  small model (SM57) stage 4 (show-time; normal); new - the release (SM63) show-time; new - SM96 normal; the new model's
  show-time cost; the 1,500 pool on show-time inputs, % of the release's own LogLoss: new - the release (SM63), new -
  SM96, new - stage 4, by id part, row check.
- SM96 is never rescored: its rows `artifacts/small-model/sm96/gpu/sm96-rows{,-1500}.jsonl` (carried from branch sm96,
  sha256 pinned, equal to the files in `wt/sm96`). Loaders and `compare` from
  `anki-rwkv/artifacts/small-model/showtime/showtime_report.py` (sha256 pinned, same pin as SM96). Curve head untrained
  (weight 0): never part of the verdict.

## CPU proofs (`sm98_check.py` -> `check-cpu-sm98.json`: ALL OK)

- K1 code: scripts/ expertium/ rwkv/ trees of HEAD == 1e22b62c (the commit SM96's launcher trained epoch 31 and the
  cool-down from, read from its log) == 8b323fb1 (SM57 stage 4's code); no uncommitted source.
- K2 options: the cool-down's `extra-options.json` == SM96's (bytes); epoch 32's options == SM96 epoch 31's == SM92
  epoch 26's + `curve_loss_weight` 0.
- K3 epochs: `sm98_epoch.py pair --epoch 32`: every protocol / decay.json leaf == SM96's epoch 31 except the order rule's
  epoch index (rows), full_run epoch / init_master / init_optimizer / update_offset / control_dir, source_epoch / epoch /
  start / note / created_utc / start hashes (no option differs); epoch 31 itself started from SM57 stage 4's epoch 30;
  epoch 32's start pinned to epoch 31's sha256 and offset 684,418; same 8,000 users and 581,942,724 reviews. The same
  generic rule accepts SM57 stage 4's real epochs 30 vs 29 and 29 vs 28 (`selfcheck`). A simulated chain (fake complete
  epoch 32 holding epoch 31's files, end offset 706,496) builds epoch 33 with the launcher's prepare code and passes the
  same rule (so 33-35 are prepared the same way). Negative controls refused: switch removed, weight 1.0, batch_chunks 8,
  epoch 31's order, start = stage 4's epoch 30, update offset + 1.
- K4 cool-down recipe: the launcher's own `prepare_decay` (`PREPARE_RECIPE=1`, log `gpu/launch-sm98-recipe.log`) wrote
  `sm98-recipe-e31st-dec-1ep32k-muon` from SM96's epoch 31 (never trained): == SM96's real cool-down except paths
  (`sm57_pair.py decay --same-start --flag {}`), == stage 4's except the switch + start, == the release's except both
  flags + start, == SM92's except the weight + start; start sha256 == epoch 31's master; fresh Muon PASS (7 lines);
  label windows OK. After epoch 35 the launcher runs `sm98_check.py cooldown` (-> `check-cpu-sm98-cooldown.json`): the
  same pairs on the real folder (SM96's except the start; the recipe's except the start), start sha256 == epoch 35's
  master.pth, update offset == epoch 35's end, fresh Muon, label windows; a failure stops before the cool-down. (This
  check was exercised on the recipe folder standing in for the real one: all True.)
- K5 rule: `selftest` reproduces SM96's decision from its rows (SM96 - SM92 -0.214 [-0.233, -0.195], 0.276287 vs
  0.276876, go; SM96 - SM96 = 0, no go; 1,500 pool SM96 - the release -0.481 [-0.534, -0.438] %, 1,487 users); `decide`
  run end to end on SM96's rows relabelled as the new model (temp folder): primary 0.000, no go, vs SM92 -0.214, pool
  1,487 users, row check 0.0.
- K6 breaks (below). Dry run: `gpu/launch-sm98-dryrun.log` ends `DRYRUN OK` (12:10).

## Breaks (`sm98_block.py`)

SM96's `sm96_block.py` without the hard hold: quiet.json (scope all/gpu) + daily 20:00-01:00 (Sunday from 16:00) unless a
quiet.json window overlaps that night (then it replaces the default). Checked: 11-09 14:00 -> `NEXT 360`, 19:59 -> `NEXT 1`,
20:30 -> `NOW`, 11-10 01:00 -> `NEXT 1140`, Sunday 11-15 15:00 -> `NEXT 60`; real quiet.json 11-08 20:02 -> `NEXT 58`;
synthetic quiet.json moving 11-10's break to 21:00 -> `NEXT 30` at 20:30. Every epoch and the cool-down start when 30 min
fit, snapshot-stop <= 2 min before a break and resume after it (users in flight are retrained from their start).

## Launch (NOT done; the caller launches after telling the user)

From `C:/Users/jonat/workspace/wt/sm98` at the pushed commit (branch sm98), hidden via WMI:

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash /c/Users/jonat/workspace/wt/sm98/artifacts/small-model/sm98/launch-sm98.sh >> /c/Users/jonat/workspace/wt/sm98/artifacts/small-model/sm98/gpu/launch-sm98.nohup 2>&1"'; ProcessStartupInformation=$si}
```

Record the returned ProcessId (wrapper) and the launcher bash pid here after launch.

Alive check: `tail -5 artifacts/small-model/sm98/gpu/launch-sm98.log`;
`powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | ? { $_.CommandLine -like '*launch-sm98*' } | select ProcessId,CommandLine"`.
Stop: `touch C:/Users/jonat/workspace/wt/sm98/artifacts/small-model/sm98/gpu/STOP` (between tickets; a running epoch /
cool-down gets a snapshot stop). Relaunch (only if the process is gone and the log has no `done`; remove `gpu/STOP`
first if it was a deliberate stop; finished steps are skipped, an epoch / the cool-down resumes from its snapshot): the
same WMI command above.

Log `gpu/launch-sm98.log` (ends `done (verdict ...)` or a `STOP:` line); trainer `gpu/train.out`; scorer `gpu/score.out`;
smokes `gpu/check.out`; cool-down prepare `gpu/prepare.out`; segments `gpu/segments.jsonl`; tickets `gpu/tickets.jsonl`.
Queue owner `sm98`; purposes start "the show-time instant-only model, stage 2 (SM98): ". GPU cap 0.8 (trainer options,
scorer `HELDOUT_GPU_MEMORY_FRACTION=0.8`; 0.92 only for the 3 long users / a no-progress retry pass).
Do not edit `scripts/`, `expertium/` or `rwkv/` in `wt/sm98` while it runs (SourceWatch). Never edit `wt/sm96` or `wt/sm57`.

## Steps and expected times (if launched ~14:00 Mon 2026-11-09, queue empty)

| step | expected | ends about |
|---|---|---|
| preconditions + recipe pair + epoch 32 pair (CPU) | 1 min | 14:01 |
| trainer smoke of epoch 32 (2 users x 512 reviews) | 3 min | 14:04 |
| **epoch 32**, 581.9M reviews at ~78.0k reviews/s (SM96's epoch 31: 124.6 min wall) | ~125 min | ~16:09 |
| epoch 33 prepare + pair (CPU, seconds), **epoch 33** | ~126 min | ~18:15 |
| epoch 34 prepare + pair, **epoch 34** first segment (snapshot stop ~19:58, ~100 min done) | ~103 min | ~19:58 |
| break 20:00-01:00 | | 01:00 Tue |
| **epoch 34** resume (~25 min left + ~5 min restart / in-flight users) | ~30 min | ~01:30 |
| epoch 35 prepare + pair, **epoch 35** | ~126 min | ~03:36 |
| cool-down prepare + label windows + Muon fresh + `sm98_check.py cooldown` (CPU; 19 s for the recipe) | ~2 min | ~03:38 |
| cool-down smoke | 3 min | ~03:41 |
| **cool-down**, 581.9M reviews at ~20.1k reviews/s (SM96's: 483 min wall) | ~483 min | ~11:45 |
| show-time scoring (723 + 3 long users), rule, normal scoring, 1,500 pool scoring (771 users), final rule (SM96: 45 min) | ~45-50 min | ~12:35 |

Total ~17.3 GPU hours (4 x 125 + 483 + ~50 + smokes), ~22.5 h wall with the 20:00-01:00 break. **Expected end Tue
2026-11-10 ~12:30-12:45** (later if other tickets take the GPU; the next break 20:00 Tue is far away). Launched later than
14:00, epoch 34's split moves: a launch after ~15:45 puts the break inside epoch 34 or 35 instead, same total.

Speed and 10-epoch ETA (SM96's measured speeds; 8k tier = 581.94M reviews per epoch, 2k tier = 147.6M): constant-LR
epochs at ~78.0k reviews/s -> 10 epochs ~20.7 h (8k) / ~5.3 h (2k); the cool-down at ~20.1k reviews/s -> 10 epochs at that
speed ~80.5 h (8k) / ~20.4 h (2k).

## Follow-up (fresh Sonnet agent, medium effort; do not wait on the GPU)

1. `git -C C:/Users/jonat/workspace/wt/sm98 fetch` and fast-forward `sm98`. `tail -8 artifacts/small-model/sm98/gpu/launch-sm98.log`.
   If it ends with `done (verdict ...)`, go on. If `STOP:`, read that line and the tails of `gpu/train.out` /
   `gpu/score.out` / `check-cpu-sm98-cooldown.json`, fix only what is clearly a transient (relaunch with the command
   above) and report otherwise. Mid-run progress: `grep -E "complete:|STOP|break" gpu/launch-sm98.log | tail`.
2. If the rule did not run: `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 artifacts/small-model/sm98/sm98_rule.py decide`
   (CPU, seconds; exit 2 = rows missing).
3. Write `artifacts/small-model/sm98/sm98-result.md` in SM96's `sm96-result.md` layout: verdict, primary pair, secondary
   table, 1,500 pool, each epoch's and the cool-down's wall minutes and reviews/s (from the `complete:` log lines and
   `gpu/segments.jsonl`; epoch 34 has 2 segments), nvidia-smi max, and the 10-epoch ETA at those speeds (8k tier = 581.94M
   reviews per epoch; 2k tier = 147.6M).
4. Commit with explicit paths and `git add -f`: `sm98-decision.json`, `sm98-decision.md`, `sm98-result.md`,
   `check-cpu-sm98-cooldown.json`, `gpu/sm98-rows.jsonl`, `gpu/sm98-rows-1500.jsonl`, `gpu/launch-sm98.log`,
   `gpu/launch-sm98.nohup`, `gpu/check.out`, `gpu/prepare.out`, `gpu/segments.jsonl`, `gpu/tickets.jsonl`,
   `gpu/check-muon-fresh-sm98-inst-e35st-st-dec-1ep32k-muon.json`, `gpu/check-replay-label-windows-sm98-inst-e35st-st-dec-1ep32k-muon.json`
   (never checkpoints, snapshots, epoch / cool-down folders, `.claude/`, `docs/GPU_OPTIMIZATIONS_LOG.md`, quiet.json), as
   `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending with a blank line
   + `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Right before the push: `git fetch`, merge `origin/sm98`,
   push `sm98`. Do not merge into main; do not touch the Anki fork or the workload forecast.
5. Report in plain names with IDs in parentheses (e.g. "the show-time instant-only small model (SM96)"), units stated;
   never write the outside research group's name.
