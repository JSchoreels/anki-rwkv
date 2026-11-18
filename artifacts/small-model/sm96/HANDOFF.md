# The show-time instant-only small model (SM96): handoff (2026-11-09 00:35, redesign)

## Question

Does the instant-only small model (SM57) stage 4, given the show-time input switch of the show-time retrain of the release
(SM92), beat the show-time retrain of the release (SM92) on show-time instant LogLoss? On show-time inputs (`e2s-query`,
what RWKV-Instant in the Anki fork sees), 722 users of the 732 set, points = % of the big model (1 leak-free epoch)
normal-input instant LogLoss 0.275757: the show-time retrain of the release (SM92) 0.276876 (+0.406), stage 4 of the
instant-only small model (SM57) 0.277236 (+0.536), stage 4 - SM92 = +0.130 [+0.059, +0.216]. Stage 4 of the instant-only
small model (SM57) loses ~0.45 points of instant on show-time inputs; the show-time retrain of the release (SM92) loses
only 0.04. If the switch removes most of stage 4's show-time cost, the new model lands about 0.3 points below it.

## Design (redesigned 2026-11-09 00:20 by the user, before any GPU ticket)

| step | the show-time instant-only small model (SM96) | reference |
|---|---|---|
| start | stage 4 of the instant-only small model (SM57)'s constant-LR instant-only epoch 30 (`wt/sm57/artifacts/small-model/sm57s4-inst-e30`, read only, master sha256 585c50b6a545, update offset end 662,340) | |
| **epoch 31** | one constant-LR instant-only epoch on show-time inputs: stage 4's epoch recipe (8k final tier, the run's order rule with N = 31, epoch 25's label-window trims, 2048 x 16 lanes carry, leak-free replay KD, AdamW 7e-4 constant, `"curve_loss_weight": 0.0`, from epoch 30's weights + AdamW state + update offset) **+ `"show_time_intervals": true`**. Folder `wt/sm96/artifacts/small-model/sm96-inst-e31-st` (prepared 11-09 00:25 by `sm96_epoch.py prepare`) | stage 4's epoch 30 step (from epoch 29); the switch as in the show-time retrain of the release (SM92)'s epoch 26 |
| cool-down | the release's one-epoch 32k Muon cool-down (8,000 users, 581,942,724 reviews, fresh Muon `expertium-iter53`, LR 7e-4 -> 0 one_minus_sin, KD replay `big-leakfree-1ep-final8k-neighbour` alpha 0.9), `"curve_loss_weight": 0.0` + `"show_time_intervals": true`, **from epoch 31**. Folder `sm96-inst-e31st-st-dec-1ep32k-muon`, prepared by the launcher after epoch 31 | stage 4's cool-down `sm57s4-inst-e30-dec-1ep32k-muon` (start + switch differ) |
| label | `sm96-inst-e31st-st-dec1ep32k/w8+s8` (show-time rows `@e2s-query`) | |

Why (user, 11-09 00:20): if the cool-down alone failed we would need the epoch plus a second cool-down anyway, so do the
epoch first. The chain now matches the show-time retrain of the release (SM92) (epoch 26 + cool-down on show-time inputs),
whose comparison is the primary of the rule. Consequence: new - stage 4 of the instant-only small model (SM57) now reads
epoch 31 + the switch together (one extra constant-LR epoch: in stage 4, epoch 30 - epoch 28 was +0.243 points at constant
LR), not the switch alone. The old folder `sm96-inst-e30-showtime-dec-1ep32k-muon` (cool-down from epoch 30) is never
trained; it stays as the recipe proof of the cool-down prepare command.

## Pre-registered rule (`sm96_rule.py`, unchanged since 6d391cf1 except the label)

- **Primary (verdict):** new - the show-time retrain of the release (SM92), instant head, both on show-time inputs, 722
  users of the 732 set, review-weighted, paired user bootstrap 2,000, seed 0, 95% CI, unit points (% of the big model
  (1 leak-free epoch) normal instant LogLoss). **go** = CI entirely below 0 AND gain >= 0.05 points; **no go** otherwise.
- **Secondary (reported):** new - stage 4 of the instant-only small model (SM57) (show-time; normal); new - the show-time
  retrain of the release (SM92) normal; the new model's show-time cost; new - the release (SM63) show-time; the 1,500 pool
  on show-time inputs, % of the release's own LogLoss (by id part, row check).
- Curve head untrained (weight 0): never part of the verdict. Loaders and `compare` from
  `anki-rwkv/artifacts/small-model/showtime/showtime_report.py` (sha256 pinned).

## CPU proofs (`sm96_check.py` -> `check-cpu-sm96.json`: ALL OK, commit c99330a7)

- K1 code: scripts/ expertium/ rwkv/ trees == sm57 035cf1dd (the code stage 4 trained with), show-time path files
  byte-identical to the branch of the show-time retrain of the release (SM92).
- K2 options: epoch 31's options == stage 4's epoch 30 options + the switch == the epoch 26 options of the show-time
  retrain of the release (SM92) + `curve_loss_weight` 0; the cool-down's `extra-options.json` == the instant options of
  the instant-only small model (SM57) + the switch == those of the show-time retrain of the release (SM92) + weight 0.
- K3 epoch 31 (`sm96_epoch.py pair`): every protocol / decay.json leaf == stage 4's epoch 30 except the order rule's epoch
  index (rows), full_run epoch / init_master / init_optimizer / update_offset / control_dir, source_epoch / start / note /
  start hashes, and `options.show_time_intervals`; epoch 30 itself started from epoch 29 by the same rule; same 8,000
  users and 581,942,724 reviews. Negative controls (no switch, another weight, another batch_chunks) are refused.
- K4 cool-down recipe: the prepare command, run from epoch 30, gives stage 4's cool-down except the switch only
  (`sm57_pair.py decay --same-start`), the release's except both flags + start, the cool-down of the show-time retrain of
  the release (SM92) except weight 0 + start; fresh Muon PASS; label windows. After epoch 31 the launcher runs
  `sm96_check.py cooldown` (-> `check-cpu-sm96-cooldown.json`): the same pairs on the real folder (stage 4's except start,
  switch, paths), start sha256 == epoch 31's master.pth, update offset == epoch 31's end, fresh Muon, label windows. A
  failure stops the launcher before the cool-down.
- K5 rule self-test (+0.130 [+0.059, +0.216] reproduced). K6 breaks (below).

## Breaks (`sm96_block.py`)

quiet.json (scope all/gpu) + daily 20:00-01:00 (Sunday from 16:00) unless a quiet.json window overlaps that night (then it
replaces the default), + a hard hold until 2026-11-09 01:00. Checked: the 11-08 20:02 "break" came from the hard hold, not
from a missing replacement (`--no-hold --now 2026-11-08T20:02` -> `NEXT 58`, quiet.json's 21:00; a synthetic quiet.json
moving 11-10's break to 21:00 gives `NEXT 30` at 20:30). Epoch 31 and the cool-down are snapshot-resumable: each starts
when 30 min fit, snapshot-stops <= 2 min before a break and resumes after it.

## What runs

- Launcher `artifacts/small-model/sm96/launch-sm96.sh`, started 2026-11-09 00:30 hidden via WMI from commit c99330a7:
  wrapper bash pid **68304**, launcher bash pid **69624** (the old launcher pid 55384 was stopped at 00:21 before any
  ticket). It waits for the hold until 01:00.
- Log `gpu/launch-sm96.log` (ends `done (verdict ...)` or a `STOP:` line); trainer `gpu/train.out`; scorer `gpu/score.out`;
  smokes `gpu/check.out`; cool-down prepare `gpu/prepare.out`; segments `gpu/segments.jsonl`; tickets `gpu/tickets.jsonl`;
  dry run `gpu/launch-sm96-dryrun.log` (DRYRUN OK 00:30).
- Queue owner `sm96`; purposes start "the show-time instant-only small model (SM96): ". GPU cap 0.8 (trainer options,
  scorer `HELDOUT_GPU_MEMORY_FRACTION=0.8`; 0.92 only for the 3 long users / a no-progress retry pass).

Steps and expected times (Mon 2026-11-09; queue empty at 00:30):

| step | expected | ends about |
|---|---|---|
| preconditions + epoch 31 pair + recipe pair (CPU) | 1 min | 01:01 |
| trainer smoke of epoch 31 (2 users x 512 reviews) | 2 min | 01:03 |
| **epoch 31**, 581.9M reviews at ~82k reviews/s (stage 4's epochs: 116-123 min) | ~120 min | ~03:05 |
| cool-down prepare + label windows + Muon fresh + `sm96_check.py cooldown` (CPU) | ~10 min | ~03:15 |
| cool-down smoke | 2 min | ~03:17 |
| cool-down, 581.9M reviews at ~20.3k reviews/s (stage 4's: 478 min wall) | ~480 min | ~11:15 |
| show-time scoring, 723 users + 3 long users | ~15 min | ~11:30 |
| rule (CPU) | 1 min | ~11:30 |
| normal scoring, 723 + 3 long users | ~15 min | ~11:45 |
| 1,500 pool show-time scoring (771 users) | ~17 min | ~12:05 |
| final rule | 1 min | ~12:05 |

Total ~10.5-11 GPU hours. Expected end Mon 2026-11-09 ~12:00 (later if other tickets take the GPU; the 20:00 break is
far away). Speed and 10-epoch ETA: constant-LR epochs run at ~82k reviews/s, so 10 epochs take ~5.0 h on the 2k tier
(147.6M reviews per epoch) and ~19.7 h on the 8k tier (581.94M); the cool-down runs at ~20.3k reviews/s, so 10 epochs at
that speed take ~20 h (2k) and ~80 h (8k).

Alive check: `tail -5 gpu/launch-sm96.log`;
`powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | ? { $_.CommandLine -like '*launch-sm96*' } | select ProcessId,CommandLine"`.
Stop: `touch gpu/STOP`. Relaunch (only if the process is gone and the log has no `done`; finished steps are skipped, the
epoch and the cool-down resume from their snapshots):

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash /c/Users/jonat/workspace/wt/sm96/artifacts/small-model/sm96/launch-sm96.sh >> /c/Users/jonat/workspace/wt/sm96/artifacts/small-model/sm96/gpu/launch-sm96.nohup 2>&1"'; ProcessStartupInformation=$si}
```

Do not edit `scripts/`, `expertium/` or `rwkv/` in `wt/sm96` while it runs (SourceWatch).

## Follow-up (fresh Sonnet agent, medium effort; do not wait on the GPU)

1. `git -C C:/Users/jonat/workspace/wt/sm96 fetch` and fast-forward `sm96`. `tail -8 artifacts/small-model/sm96/gpu/launch-sm96.log`.
   If it ends with `done (verdict ...)`, go on. If `STOP:`, read that line and the tails of `gpu/train.out` /
   `gpu/score.out` / `check-cpu-sm96-cooldown.json`, fix only what is clearly a transient (relaunch with the command
   above) and report otherwise.
2. If the rule did not run: `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 artifacts/small-model/sm96/sm96_rule.py decide`
   (CPU, seconds; exit 2 = rows missing).
3. Write `artifacts/small-model/sm96/sm96-result.md`: verdict, primary pair, secondary table, 1,500 pool, epoch 31's and
   the cool-down's wall minutes and reviews/s (from the `complete:` log lines and `gpu/segments.jsonl`), nvidia-smi max,
   and the 10-epoch ETA at those speeds (8k tier = 581.94M reviews per epoch; 2k tier = 147.6M).
4. Commit with explicit paths and `git add -f`: `sm96-decision.json`, `sm96-decision.md`, `sm96-result.md`,
   `check-cpu-sm96-cooldown.json`, `gpu/sm96-rows.jsonl`, `gpu/sm96-rows-1500.jsonl`, `gpu/launch-sm96.log`,
   `gpu/launch-sm96.nohup`, `gpu/check.out`, `gpu/prepare.out`, `gpu/segments.jsonl`, `gpu/tickets.jsonl`,
   `gpu/check-muon-fresh-sm96-inst-e31st-st-dec-1ep32k-muon.json`, `gpu/check-replay-label-windows-sm96-inst-e31st-st-dec-1ep32k-muon.json`
   (never checkpoints, snapshots, `.claude/`, `docs/GPU_OPTIMIZATIONS_LOG.md`, quiet.json), as
   `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending with a blank line
   + `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Right before the push: `git fetch`, merge `origin/sm96`,
   push `sm96`. Do not merge into main; do not touch the Anki fork.
5. Report in plain names with IDs in parentheses (e.g. "the show-time retrain of the release (SM92)"), units stated.
