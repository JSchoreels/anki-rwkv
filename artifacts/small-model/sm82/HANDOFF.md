# Response-time prediction (SM82) stage 1: handoff

Branch `sm82`, worktree `C:/Users/jonat/workspace/wt/sm82` (cut from `sm80` at 365e9327). Launched 11-01 03:40 via WMI (hidden).

## What was built

- **Stage 0 audit** (CPU, done): [stage0-audit.md](stage0-audit.md) / `.json`. 0.09% of times are 0, 5.6% at the 60 s cap, one user with a 40 s cap; per-user median 4.0 / 7.8 / 19.4 s (p10 / p50 / p90); 38% of the within-user spread is card to card. The interval input is answer-to-answer, so same-day reviews carry their own answer time (77% of back-to-back intervals are within 1.5 s of it). Decision: the head trains and scores only reviews with a recorded time on a later day than the card's previous review (70% of reviews of seen cards). Also `docs/DATA_QUALITY.md` DQ07.
- **Head** (preset `w64-sm82`, `expertium/variants.py`; `expertium/model/variant_model.py` `TimeHead`): one Linear 128 -> 64 (4 grades x 16 log-time bins) on the instant head's hidden layer (`head_p` output), zero init, 8,256 values (339,767 parameters in all, was 331,511). Weight on Muon (head group, like `p_linear`), bias on AdamW. Float32 like `p_linear`. Bins: under 0.5 s, 14 bins x1.41 between 0.5 s and 60 s, 60 s and more (the cap); a lower user cap is censored ("that bin or later"). Loss: censored cross-entropy of the true grade's bins at labelled instant rows, weight 0.1, added after the grade terms. Labels: `scripts/expertium_options.py` `time_labels` / `TimeTargets` (revlog duration by `label_review_th`); the teacher has no time head, so only real answers train it. Export (scorer w8 weight bytes): 410,459 -> 419,163 B (+8,704 B); state unchanged. Flags off: bit for bit.
- **CPU proof** (`scripts/sm82_time.py check`, PASS, 59 s in the launcher; report `gpu/check-cpu-sm82.json`, earlier manual run `check-cpu-sm82.json`): T0 flags off == the tiny architecture additions (SM77) tree (loss bits, outputs, every gradient, 8 updates, KD replay + strided carry); T1 zero head: grade loss terms, per-row loss, rating and curve outputs and every old gradient bitwise, the head gets gradients, time loss = ln 16, total = grade + 0.1 x time (gap 6e-8); T2 labels match the revlog (time and grade) on every instant row; T3 no leak: changing one review's time changes one input value (its own row, after its instant row) and no output up to and including that instant row; T4 TorchScript = eager; T5 optimizer split 318,080 Muon / 21,687 AdamW. Scorer check (CPU, 2 users): w8+s8 benchmark rows of the zero-head model == w64, bit for bit.
- **Baselines** (CPU, done): [time-baselines.jsonl](time-baselines.jsonl) (per-user sums), residual CDFs per grade fit on the non-test reviews (`time-baselines.cdf.npz`). 21.6M time-labelled test reviews on the 722 rule users. Card's last time: 2.445 nats, 0.721 mean |ln error|; user's running median per grade: 2.364 nats, 0.642. Uniform = 2.773.

## What runs (`artifacts/small-model/sm82/launch-sm82.sh`)

1. CPU: extend (SM72s epoch 21 master + zero head -> `artifacts/small-model/sm82-ext-e21`), cool-down prepare (`scripts/sm82_time.py cooldown` -> `artifacts/small-model/sm82-arm-e21-dec-q32k-muon`), label-window trims, Muon fresh-state check, pair check vs the reference (`[sm82-pair] decay OK`: same 1,991 users, order, options, seeds, schedule, update offset 463,638; only the start weights (+ zero head), the architecture and paths differ), CPU proof.
2. Ticket 0 (12 min): trainer smoke of the cool-down (2 users x 512 reviews; must print `[sm82-time] updates 1: time loss 2.77...`) + the time scorer on 3 users with the zero head (GPU).
3. Cool-down, ~125 min (reference: 145.6M reviews in 124 min, 19.5k reviews/s), segmented at windows, snapshot-resumable. `train.out` prints `[sm82-time]` every 500 updates.
4. Grade scoring: 723 users (0.8 cap) + 3 long users (0.92), w8+s8, leak-free, label `sm82-arm-e21-q32k/w8+s8` -> `gpu/sm82-neighbour.jsonl` (~20 min).
5. Time scoring: 723 users, fp32, leak-free inputs -> `gpu/sm82-time.jsonl` (~15-25 min).
6. Rule (CPU): `scripts/sm82_time.py decide` -> `sm82-decision.{md,json}`.

## Tickets and timeline (Sunday 11-01, GPU break from 16:00)

| step | expected | when |
|---|---|---|
| CPU preparation + proof (PASS, 59 s) | 2 min | 03:40-03:42 |
| ticket 000807: trainer smoke (exit 0, 2 updates, 1,024 reviews, `[sm82-time] updates 1: time loss 2.7726`) + time-scorer smoke (3 users) | 18 s | 03:41:39-03:41:57 |
| ticket 000808: cool-down (145,578,259 reviews, 1,991 users) | ~125 min | started 03:41:57, ends ~05:50 |
| grade scoring 723 + 3 long | ~20 min | ~05:50-06:15 |
| time scoring | ~20 min | ~06:15-06:35 |
| rule | seconds | ~06:35 |

Expected end: about 06:40, well before the 16:00 break. If anything slips into the break, the cool-down snapshot-stops 2 min before and resumes at 01:00.

## Files

- Launcher log: `C:/Users/jonat/workspace/wt/sm82/artifacts/small-model/sm82/gpu/launch-sm82.log` (nohup: `launch-sm82.nohup`).
- `gpu/prepare.out`, `gpu/check.out` (smoke), `gpu/train.out`, `gpu/score.out`, `gpu/time.out`; rows `gpu/sm82-neighbour.jsonl`, `gpu/sm82-time.jsonl`.
- Folders: `artifacts/small-model/sm82-ext-e21` (start), `artifacts/small-model/sm82-arm-e21-dec-q32k-muon` (cool-down).
- Reference (read only): `C:/Users/jonat/workspace/wt/sm77/artifacts/small-model/sm77-ref-e21-dec-q32k-muon`, label `sm77-ref-e21-q32k/w8+s8` in `C:/Users/jonat/workspace/wt/sm77/artifacts/small-model/sm77/gpu/sm77-neighbour.jsonl`.

## How to read the result

The last log lines are `rule exit 0: [sm82-rule] 722 users: curve ... instant ... (arm - reference); time LogLoss X vs card's last Y / user median Z, |ln err| ...: PASS|FAIL-GRADE|FAIL-TIME` and then `done`. Bars: grade curve and instant each <= +0.03 points of % of the big model (1 leak-free epoch) vs the reference (paired user bootstrap 2,000, seed 0); the time head beats both baselines on the censored time LogLoss and on the mean |ln error| of its median (uncensored reviews), each with the 95% CI of the difference clear of zero. Full tables (per grade, P(under 5.5 s) predicted vs actual, the head with the grade mixed in, size) in `sm82-decision.md`. Any `STOP:` line ends the launcher and names the step and its output file.

Follow-up (CPU, after `done`): commit `gpu/*.json`, `gpu/*.jsonl`, `gpu/*.log`, `gpu/*.out`, `sm82-decision.{md,json}` on `sm82` (`git add -f`), push; the main session updates the Factory page. Re-run the rule by hand:

```
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; WT=C:/Users/jonat/workspace/wt/sm82; G=$WT/artifacts/small-model/sm82/gpu
$PY -X utf8 $WT/scripts/sm82_time.py decide --time-rows $G/sm82-time.jsonl --baselines $WT/artifacts/small-model/sm82/time-baselines.jsonl \
  --arm-rows $G/sm82-neighbour.jsonl --arm-label sm82-arm-e21-q32k/w8+s8 \
  --ref-rows C:/Users/jonat/workspace/wt/sm77/artifacts/small-model/sm77/gpu/sm77-neighbour.jsonl --ref-label sm77-ref-e21-q32k/w8+s8
```

## Relaunch (resumes: finished steps are skipped, an interrupted cool-down resumes from its snapshot, scoring skips scored users)

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm82/artifacts/small-model/sm82/launch-sm82.sh >> C:/Users/jonat/workspace/wt/sm82/artifacts/small-model/sm82/gpu/launch-sm82.nohup 2>&1"'; ProcessStartupInformation=$si}
```

To stop it: `touch C:/Users/jonat/workspace/wt/sm82/artifacts/small-model/sm82/gpu/STOP` (the cool-down takes a snapshot first). Remove the file before relaunching. After a Claude app restart, check the launcher is alive; if the log has been silent for more than 10 minutes outside a ticket, relaunch.

Do not edit `scripts/`, `expertium/` or `rwkv/` in this worktree while it runs (SourceWatch stops the training).

## Not in this task

Stage 2 (answer-time curves vs interval, answer-time-aware workload in the simulator) and same-day times (need intervals as Anki knows them at show time, the duration-leak item (SM31)).
