# The show-time instant-only small model (SM96): handoff (2026-11-08 20:05)

## Question

Does the instant-only small model (SM57) stage 4, given the show-time input switch of the show-time retrain of the release
(SM92), beat SM92 on show-time instant LogLoss? On show-time inputs (`e2s-query`, what RWKV-Instant in the Anki fork sees),
722 users of the 732 set, points = % of the big model (1 leak-free epoch) normal-input instant LogLoss 0.275757:
SM92 0.276876 (+0.406), stage 4 0.277236 (+0.536), stage 4 - SM92 = +0.130 [+0.059, +0.216]. Stage 4 loses ~0.45 points
of instant on show-time inputs; SM92 loses only 0.04. If the switch removes most of stage 4's show-time cost, the new model
lands about 0.3 points below SM92.

## Design (the cheapest variant that answers it)

| | New (SM96) | Paired reference: stage 4 of the instant-only small model (SM57), read only |
|---|---|---|
| Start | stage 4's constant-LR instant-only epoch 30 (`wt/sm57/artifacts/small-model/sm57s4-inst-e30`, master sha256 585c50b6a545), update offset 662,340 | the same |
| Cool-down | the release's one-epoch 32k Muon cool-down (8,000 users, 581,942,724 reviews, fresh Muon `expertium-iter53`, LR 7e-4 -> 0 one_minus_sin, KD replay `big-leakfree-1ep-final8k-neighbour` alpha 0.9), `"curve_loss_weight": 0.0` **+ `"show_time_intervals": true`** | the same without the inputs flag (`sm57s4-inst-e30-dec-1ep32k-muon`) |
| Folder | `wt/sm96/artifacts/small-model/sm96-inst-e30-showtime-dec-1ep32k-muon` (prepared 11-08, label windows trimmed + checked, Muon fresh PASS) | |
| Label | `sm96-inst-e30-st-dec1ep32k/w8+s8` (show-time rows `@e2s-query`) | `sm57s4-inst-e30-dec1ep32k/w8+s8` |

Why no show-time constant-LR epoch first (SM92 did epoch 26 on show-time inputs before its cool-down):
- The cool-down alone is a full epoch (all 581.9M reviews) on show-time inputs, at the LR range where the model settles;
  the only difference from stage 4 is the inputs flag (`sm57_pair.py decay --same-start` OK), so new - stage 4 is a clean
  paired read of the switch. An extra epoch would add a second change (one more constant-LR epoch: in stage 4, epoch 30 -
  epoch 28 was +0.243 points at constant LR) and confound it.
- Cost: an epoch first adds ~2 GPU hours (stage 4 epochs: 116-123 min at ~82k reviews/s) plus a re-prepared cool-down.
- When to add it: only if the result shows the new model's own show-time cost (show-time - normal) still well above SM92's
  0.04 points (say > 0.15), i.e. the cool-down alone did not adapt. That is the follow-up variant, not this run.

## Pre-registered rule (`sm96_rule.py`, fixed and committed before any GPU run, 05664521)

- **Primary (verdict):** new - SM92, instant head, both on show-time inputs, 722 users of the 732 set, review-weighted,
  paired user bootstrap 2,000, seed 0, 95% CI, unit points (% of the big model (1 leak-free epoch) normal instant LogLoss).
  **go** = CI entirely below 0 AND gain >= 0.05 points; **no go** otherwise.
- **Secondary (reported):** new - stage 4 (show-time; normal); new - SM92 normal; the new model's show-time cost; new - the
  release (SM63) show-time; the 1,500 pool on show-time inputs, % of the release's own LogLoss (the show-time report's
  `pool1500` convention: new - release, new - stage 4, by id part, row check).
- Curve head untrained (weight 0): never part of the verdict.
- Loaders and `compare` are imported from `anki-rwkv/artifacts/small-model/showtime/showtime_report.py` (sha256 pinned;
  the rule refuses if that file changes). `selftest` reproduces the report's stage 4 - SM92 = +0.130 [+0.059, +0.216].

## Code and CPU proofs (`sm96_check.py` -> `check-cpu-sm96.json`: ALL OK)

Branch `sm96` is cut from `sm57` 8b323fb1, the code stage 4 trained with. It already contains the whole `sm92` branch
(merge-base == sm92 tip 29727c6b), so no merge was needed.
- K1: scripts/ expertium/ rwkv/ git trees == 8b323fb1 (bitwise); the source diff to sm92 is exactly the 6 curve-loss-weight /
  1,500-pool files; the show-time path files (`gpu_r01_samples.py`, `sm77_warm.py`, `expertium_decay.py`, the two check
  scripts) are byte-identical to sm92's; the prefetch diff is only the option allowlist line.
- K2: `extra-options.json` == SM57's `extra-options-instant.json` + the flag == SM92's `extra-options.json` + weight 0.
- K3: the cool-down == stage 4's except the flag (same start); == the release's except both flags + start; == SM92's except
  weight 0 + start.
- K4: start sha256 == stage 4's start; fresh Muon; label windows ok.
- K5: rule self-test. K6: break checker (holds until Mon 01:00; 20:00-01:00 nightly after).

## What runs

- Launcher `artifacts/small-model/sm96/launch-sm96.sh`, started 2026-11-08 20:02 hidden via WMI: wrapper bash pid **53528**,
  launcher bash pid **55384**. First log line "started: 05664521".
- Log `gpu/launch-sm96.log` (ends `done (verdict ...)` or a `STOP:` line); trainer `gpu/train.out`; scorer `gpu/score.out`;
  smoke `gpu/check.out`; segments `gpu/segments.jsonl`; tickets `gpu/tickets.jsonl`; dry run `gpu/launch-sm96-dryrun.log`.
- Queue owner `sm96`; purposes start "the show-time instant-only small model (SM96): ". GPU cap 0.8 (trainer options,
  scorer `HELDOUT_GPU_MEMORY_FRACTION=0.8`; 0.92 only for the 3 long users / a no-progress retry pass).
- Breaks: `sm96_block.py` = quiet.json (scope all/gpu) + daily 20:00-01:00 (Sunday from 16:00) unless quiet.json overlaps
  that night, + a hard hold until 2026-11-09 01:00. CPU steps wait too. The cool-down snapshot-stops <= 2 min before a
  break and resumes after it.

Steps and expected times (GPU free at 01:00; the queue was empty at 20:05):

| step | expected | ends about (Mon 11-09) |
|---|---|---|
| preconditions + pair re-check (CPU) | 1 min | 01:00 |
| trainer smoke (2 users x 512 reviews) | 1 min | 01:01 |
| cool-down, 581.9M reviews at ~20.3k reviews/s (stage 4: 478 min wall) | ~480 min | ~09:05 |
| show-time scoring, 723 users + 3 long users | ~15 min | ~09:20 |
| rule (CPU) | 1 min | ~09:20 |
| normal scoring, 723 + 3 long users | ~15 min | ~09:35 |
| 1,500 pool show-time scoring (771 users) | ~17 min | ~09:55 |
| final rule | 1 min | ~10:00 |

Total ~8.5-9 GPU hours. Expected end Mon 2026-11-09 ~10:00 (later if other tickets take the GPU at 01:00; the 20:00
break is far away). Speed: the cool-down runs at ~20.3k reviews/s (stage 4's cool-down: 20,313); at that speed a
10-epoch run would take ~20 h on the 2k tier and ~80 h on the 8k tier (constant-LR epochs run at ~82k reviews/s:
~5.0 h / ~19.7 h).

Alive check: `tail -5 gpu/launch-sm96.log`;
`powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | ? { $_.CommandLine -like '*launch-sm96*' } | select ProcessId,CommandLine"`.
Stop: `touch gpu/STOP`. Relaunch (only if the process is gone and the log has no `done`; finished steps are skipped, the
cool-down resumes from its snapshot):

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash /c/Users/jonat/workspace/wt/sm96/artifacts/small-model/sm96/launch-sm96.sh >> /c/Users/jonat/workspace/wt/sm96/artifacts/small-model/sm96/gpu/launch-sm96.nohup 2>&1"'; ProcessStartupInformation=$si}
```

Do not edit `scripts/`, `expertium/` or `rwkv/` in `wt/sm96` while it runs (SourceWatch).

## Follow-up (fresh Sonnet agent, medium effort; do not wait on the GPU)

1. `git -C C:/Users/jonat/workspace/wt/sm96 fetch` and fast-forward `sm96`. `tail -8 artifacts/small-model/sm96/gpu/launch-sm96.log`.
   If it ends with `done (verdict ...)`, go on. If `STOP:`, read that line and the tails of `gpu/train.out` /
   `gpu/score.out`, fix only what is clearly a transient (relaunch with the command above) and report otherwise.
2. If the rule did not run: `C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 artifacts/small-model/sm96/sm96_rule.py decide`
   (CPU, seconds; exit 2 = rows missing).
3. Write `artifacts/small-model/sm96/sm96-result.md`: verdict, primary pair, secondary table, 1,500 pool, the cool-down's
   wall minutes and reviews/s (from the `complete:` log line and `gpu/segments.jsonl`), nvidia-smi max, and the 10-epoch
   ETA at that speed (8k tier = 581.94M reviews per epoch; 2k tier = 147.6M).
4. Commit with explicit paths and `git add -f`: `sm96-decision.json`, `sm96-decision.md`, `sm96-result.md`, `gpu/sm96-rows.jsonl`,
   `gpu/sm96-rows-1500.jsonl`, `gpu/launch-sm96.log`, `gpu/launch-sm96.nohup`, `gpu/check.out`, `gpu/segments.jsonl`,
   `gpu/tickets.jsonl` (never checkpoints, snapshots, `.claude/`, `docs/GPU_OPTIMIZATIONS_LOG.md`, quiet.json), as
   `git -c user.name="Jonathan Schoreels" -c user.email="jonathan.schoreels@gmail.com" commit`, message ending with a blank line
   + `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Right before the push: `git fetch`, merge `origin/sm96`,
   push `sm96`. Do not merge into main; do not touch the Anki fork.
5. Report in plain names with IDs in parentheses (e.g. "the show-time retrain of the release (SM92)"), units stated.
   If no go and the new model's show-time cost is still > 0.15 points, propose the variant with one show-time
   constant-LR epoch 31 before the cool-down (~2 extra GPU hours).
