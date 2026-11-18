# Data-order re-seed (SM80): handoff

Branch `sm80`, worktree `C:/Users/jonat/workspace/wt/sm80` (cut from `sm77`). Launched 10-31 16:19 via WMI (hidden).

## What runs

The data augmentation run (SM67) found that every epoch of the leak-free small model (SM72s) sees bit-identical
inputs. `training_seed` is 2025 in every epoch, and each window's seed is `training_seed + user + offset`. That seed
draws the id-code tables, the day baselines and (through `torch.manual_seed(item.seed)`) the dropout masks. Only the
user order changes from epoch to epoch (`split_seed x 1000 + epoch`).

- **Arm:** SM72s's epoch-21 protocol, started from its epoch 20, with `training_seed` 2025 -> 80021
  (`scripts/sm77_warm.py epoch --data-seed 80021`; the default keeps the run's own seed, bit-identical). Then the
  quarter-epoch 32k Muon cool-down with the same options, users, order and seeds as the reference's cool-down. Only
  epoch 21 is re-seeded. Then the 732 scoring (723 users at the 0.8 cap, plus the 3 long users at 0.92).
- **Reference (read only, already scored):** the tiny architecture additions (SM77) reference. That is SM72s's own
  epoch 21 plus `wt/sm77/artifacts/small-model/sm77-ref-e21-dec-q32k-muon`, label `sm77-ref-e21-q32k/w8+s8` in
  `wt/sm77/artifacts/small-model/sm77/gpu/sm77-neighbour.jsonl`.
- **CPU proof** (`scripts/sm80_reseed.py check`, log line `[sm80-check] PASS`, report `gpu/check-cpu-sm80.json`):
  - The option off gives SM72s's epoch-21 protocol and options exactly.
  - The option on changes only `training_seed`: 8,000 users, 581,942,724 reviews.
  - Batches of 3 users / 4 updates were built with the real options (2048 x 16 strided lanes). The layout, labels,
    review numbers and KD replay targets are bit-identical. Only the id-code tables, the day baselines and the input
    built from them differ.
- **Cool-down pair check** (`[sm80-pair] decay OK` in the log): the arm's protocol.json and decay.json must equal
  the reference's except the start weights and paths. That covers the same users, the same order, the same options
  and the same update offset (463,638, which also proves the re-seeded epoch made the same number of updates).
- **Rule** (`scripts/sm80_reseed.py decide`, the same logic as `scripts/sm77_rule.py`): 722 users, measured in
  points of % of the big model (1 leak-free epoch).
  - go: curve <= -0.20 or instant <= -0.15, with that head's CI clear of zero and the other head <= +0.03;
  - stop: both heads > -0.10;
  - grey: anything else.

  Output: `artifacts/small-model/sm80/sm80-decision.{md,json}`.

## Ticket timeline (10-31, break 20:00-01:00 daily; Sunday 11-01 from 16:00)

| step | expected | when |
|---|---|---|
| CPU preparation + proof (PASS) | 1 min | 16:19-16:20 |
| trainer smoke (2 users x 512 reviews): exit 0, 2 updates, 1,024 reviews | <1 min | 16:20 |
| arm epoch 21 (re-seeded), ~120 min, ticket 000794 | 125 min ticket | started 16:20, ends ~18:25 |
| cool-down prepare + Muon / label-window checks + pair check (CPU) | 1 min | right after the epoch |
| arm quarter-epoch 32k Muon cool-down, ~120 min | 150 min ticket | does not fit before 20:00: waits for 01:00, ~01:00-03:05 (11-01) |
| scoring 723 users (0.8 cap) + 3 long users (0.92) | ~20 min | ~03:05-03:25 |
| rule (CPU) | seconds | ~03:25 |

Expected end: about 03:30 on 11-01, before Sunday's 16:00 break.

## Files

- Launcher log: `C:/Users/jonat/workspace/wt/sm80/artifacts/small-model/sm80/gpu/launch-sm80.log`. The event
  waiter key is `sm80`.
- Trainer output: `gpu/train.out`. Scorer output: `gpu/score.out`. Rows: `gpu/sm80-neighbour.jsonl`. Preparation
  output: `gpu/prepare.out`. Smoke output: `gpu/check.out`.
- Folders: `artifacts/small-model/sm80-arm-e21` (epoch) and `artifacts/small-model/sm80-arm-e21-dec-q32k-muon`
  (cool-down).

## How to read the result

The last log lines are `rule exit 0: [sm80-rule] 722 users: curve ... instant ... (arm - reference): GO|STOP|GREY`
and then `done`. The full table is in `sm80-decision.md`. A negative number means the re-seed helped.

Any `STOP:` line ends the launcher. The line names the step and its output file.

## Relaunch (resumes: finished steps are skipped, an interrupted epoch / cool-down resumes from its snapshot)

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm80/artifacts/small-model/sm80/launch-sm80.sh >> C:/Users/jonat/workspace/wt/sm80/artifacts/small-model/sm80/gpu/launch-sm80.nohup 2>&1"'; ProcessStartupInformation=$si}
```

To stop it: `touch C:/Users/jonat/workspace/wt/sm80/artifacts/small-model/sm80/gpu/STOP`. The epoch or cool-down
takes a snapshot before it stops. Remove the file before you relaunch.

After a Claude app restart, check that the launcher is still alive. Detached jobs died at an app update on 10-22.
If the log has been silent for more than 10 minutes outside a ticket, relaunch.

Do not edit `scripts/`, `expertium/` or `rwkv/` in this worktree while it runs. SourceWatch stops the training.
