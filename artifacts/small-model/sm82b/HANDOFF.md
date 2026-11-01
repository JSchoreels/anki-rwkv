# Frozen-model answer-time head (SM82b): handoff

Branch `sm82b`, worktree `C:/Users/jonat/workspace/wt/sm82b` (cut from `sm82` at 94fb0f66). Launched Sunday 11-01 06:27 via WMI (hidden).

## What was built

- **Goal**: the response-time prediction (SM82) answer-time head without touching the grade predictions: the head is trained alone on a frozen copy of the released small model (the leak-free small model (SM72s) 25 epochs + the 1-epoch 32k Muon cool-down, label `s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon`, `C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/expertium/E89-F65e-F77a/master.pth`).
- **How it is frozen** (`scripts/sm82b_head.py train`): the trunk never trains. Each training user's history runs once through the frozen model on the time scorer's own path (full history, fp32, leak-free inputs `--deleted-cards neighbour`, id / day codes seed 2025 + user) under `torch.no_grad`, every parameter `requires_grad False`; a forward hook keeps the instant head's 128-wide hidden layer (`head_p` output) at the instant rows with a time label. Only the head is in the optimizer, so there is no trunk backward. The written `master.pth` = the released master's tensors, loaded again from its file, + the trained head.
- **Head**: same as SM82 (preset `w64-sm82`, `time_heads.0.linear`, Linear 128 -> 64 = 4 grades x 16 log-time bins, zero init, 8,256 values; no hidden layer, so the comparison with SM82's joint head isolates the cost of freezing). Loss: SM82's censored cross-entropy of the true grade's bins (labels by `time_labels`: a recorded time on a later day than the card's previous review).
- **Optimizer**: plain AdamW (betas 0.9 / 0.999, no weight decay), lr 2e-3, 2% warm-up, cosine to 0 (SM82 had the weight on Muon, the bias on AdamW; with a frozen trunk a 128 x 64 matrix does not need Muon).
- **Data** ([plan.json](plan.json)): the 1,991 training users of SM82's quarter-epoch cool-down (none in the 732 pool), in a seeded order (seed 0), each history's first 400k reviews (a full-history forward of a longer user does not fit the 0.8 cap; 59 users cut): 136.1M of 145.6M reviews, 81.9M time-labelled rows, one pass through a 2M-row shuffle buffer, 9,999 steps of 8,192 rows. A quarter epoch is plenty for 8,256 values (about 10k labelled rows per parameter) and it is SM82's data, so the two heads compare directly.
- **CPU proof** (`scripts/sm82b_head.py check`, PASS in the launcher, [gpu/check-cpu-sm82b.json](gpu/check-cpu-sm82b.json); an earlier manual run [check-cpu-sm82b.json](check-cpu-sm82b.json)): F1 after 14 AdamW updates on 3 training users, the trainable parameters and the optimizer hold only the head's 2 tensors, every non-head tensor of the written master is the released master's bit for bit (keys, dtypes, values), the trunk in memory is unchanged, the head moved off zero, first loss = ln 16; F2 a run stopped after one buffer pass and resumed gives the same head bit for bit; F3 on 2 evaluation users' full histories every grade output (ahead logits, w, w_log_p, instant logits, s, d) of the trained frozen model is the released model's bit for bit, with fp32 and with int8 state; F4 SM82's T3 no-leak probe on the trained head passes (changing a review's answer time changes no output up to and including its own instant row) and every training row has a zero duration input; F5 the labels are `time_labels` of the reviews the rows predict, rows per user = the plan; F6 training loss per row = the scorer's censored LogLoss (gap 9e-7); F7 the scorer (CPU, w8+s8, leak-free) gives the same rows and per-review dumps for the released and the frozen-with-head model, and the time scorer scores the trained head.
- **Smoke** (ticket 0, 35 s): head training on 3 small users + the longest user (cut at 400k): 4/4 users, 295 steps, 0 skipped, peak 8.06 GiB, first step `time loss 2.7726` (ln 16), trunk unchanged; time scorer on 3 users; grade check on 2 users: released vs frozen-with-head bitwise True, and equal to the recorded released scores ([gpu/smoke-grade-check.json](gpu/smoke-grade-check.json)).

## What runs (`artifacts/small-model/sm82b/launch-sm82b.sh`)

Ticket purposes, in order (queue owner small-model):

1. `frozen-model answer-time head (SM82b): smoke: head training on 4 users (3 small + the longest, cut at 400k) + time scoring (3 users) + grade check (2 users) on the frozen released small model` (done, exit 0, 06:28:23-06:28:58)
2. `frozen-model answer-time head (SM82b): head training on the frozen released small model (1,991 users of the quarter-epoch cool-down, first 400k reviews each, 10k AdamW steps)` (started 06:29, see the timeline)
3. `frozen-model answer-time head (SM82b): grade check: the released small model vs the frozen model with the head (5 users, scorer rows and dumps bitwise)`
4. `frozen-model answer-time head (SM82b): time scoring of the head on the 732 pool minus 9 never-fit (723, pass 1)` (and `pass 2` only if users are missing)

Then the rule on CPU: `scripts/sm82b_head.py decide` -> [sm82b-decision.md](sm82b-decision.md) / `.json`.

## Timeline (Sunday 11-01, GPU break from 16:00)

| step | expected | when |
|---|---|---|
| CPU preparation + proof (PASS, 65 s) | 2 min | 06:27-06:28 |
| ticket 1: smoke (exit 0) | 35 s | 06:28:23-06:28:58 |
| ticket 2 (queue 000814): head training (136.1M reviews forward, 9,999 steps) | ~50 min at the scorer's ~45-56k reviews/s; ticket budget 80 min | started 06:29, ends ~07:20 (at worst ~07:50) |
| ticket 3: grade check (5 users, both models) | ~5 min | ~07:25 |
| ticket 4: time scoring (723 users; SM82's took 12.7 min) | ~13 min | ~07:30-07:45 |
| rule | seconds | ~07:45 (at worst ~08:15) |

Expected end: well before the 16:00 break. The long run's throughput had not been measured yet at handoff time; the first `train.out` progress line (step 500) gives it.

## Files

- Launcher log: `C:/Users/jonat/workspace/wt/sm82b/artifacts/small-model/sm82b/gpu/launch-sm82b.log` (nohup: `launch-sm82b.nohup`).
- `gpu/check.out` (smoke), `gpu/train.out` (head training, a line every 500 steps), `gpu/grade.out`, `gpu/time.out`; rows `gpu/grade-check-rows.jsonl` + `gpu/grade-dump/`, `gpu/grade-check.json`, `gpu/sm82b-time.jsonl`.
- Folders: `artifacts/small-model/sm82b-ext-released` (the released master + the zero head), `artifacts/small-model/sm82b-head-released` (`master.pth`, `train.json`, `resume.pt`).
- Grade-check users ([grade-users.json](grade-users.json)): 4396, 722, 1286, 3455, 3781 (10/30/50/70/90% review-count quantiles of the 723).
- Read only: SM82's baselines `C:/Users/jonat/workspace/wt/sm82/artifacts/small-model/sm82/time-baselines.jsonl`, SM82's time rows `.../sm82/gpu/sm82-time.jsonl`, the released model's recorded scores `C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/gpu/small-lf-neighbour.jsonl`.

## How to read the result

The last log lines are `rule exit 0: [sm82b-rule] 722 users: grade unchanged True|False; time LogLoss X vs card's last Y / user median Z / joint (SM82) W (+d), |ln err| ...: PASS|FAIL-GRADE|FAIL-TIME` and then `done`. Pass = the head beats the card's last time and the user's running median per grade on the censored time LogLoss and on the mean |ln error| of its median (uncensored reviews), each with the 95% CI of the review-weighted difference clear of zero (paired user bootstrap 2,000, seed 0), AND grade unchanged (released vs frozen-with-head scorer rows, every grade field, and per-review dumps bitwise equal on the 5 users; `gpu/grade-check.json` also says whether they equal the recorded released scores). The row "vs the jointly trained head (SM82)" is the cost of freezing (SM82: 1.9300 / 0.4625). Full tables (grade check, time predictors, per grade with P(under 5.5 s) predicted / actual, size) in `sm82b-decision.md`. Any `STOP:` line ends the launcher and names the step and its output file.

Follow-up (CPU, after `done`): commit `gpu/*.json`, `gpu/*.jsonl`, `gpu/*.log`, `gpu/*.out`, `sm82b-decision.{md,json}` and `../sm82b-head-released/train.json` on `sm82b` (`git add -f`; not `gpu/grade-dump/`, not `resume.pt`), push; the main session updates the Factory page. Re-run the rule by hand:

```
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; WT=C:/Users/jonat/workspace/wt/sm82b; G=$WT/artifacts/small-model/sm82b/gpu
$PY -X utf8 $WT/scripts/sm82b_head.py decide --time-rows $G/sm82b-time.jsonl --grade-check $G/grade-check.json \
  --train-json $WT/artifacts/small-model/sm82b-head-released/train.json
```

## Relaunch (resumes: finished steps are skipped, the head training resumes from its last buffer pass, scoring skips scored users)

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm82b/artifacts/small-model/sm82b/launch-sm82b.sh >> C:/Users/jonat/workspace/wt/sm82b/artifacts/small-model/sm82b/gpu/launch-sm82b.nohup 2>&1"'; ProcessStartupInformation=$si}
```

To stop between tickets: `touch C:/Users/jonat/workspace/wt/sm82b/artifacts/small-model/sm82b/gpu/STOP` (remove it before relaunching). To stop the head training itself, cancel its queue ticket; a relaunch resumes from `resume.pt` (written after every 2M-row buffer pass). After a Claude app restart, check the launcher is alive; if the log has been silent for more than 10 minutes outside a ticket, relaunch.

Do not edit `scripts/`, `expertium/` or `rwkv/` in this worktree while it runs.
