# Answer-time head on richer frozen features (SM82c): handoff

Branch `sm82c`, worktree `C:/Users/jonat/workspace/wt/sm82c` (cut from `sm82b` at a5057f2f; sources at 9cb2e620). Launched Sunday 11-01 07:40 via WMI (hidden).

## What was built

- **Goal**: how much of the jointly trained head's time LogLoss (response-time prediction (SM82): 1.930 / |ln err| 0.4625) a head on the frozen released small model can reach without touching the grade, when it gets richer inputs than the frozen-model answer-time head (SM82b: 2.3385 / 0.6696, only the instant head's 128-wide layer). Released model: the leak-free small model (SM72s) 25 epochs + the 1-epoch 32k Muon cool-down, `C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/expertium/E89-F65e-F77a/master.pth`.
- **Frozen features (448)** at each instant row: the canonical residual stream (d 64) after the input encoder and after each of the 4 interleaved layer rounds (the last one = the final residual that feeds every head), each layer-normalised without affine parameters (5 x 64), plus the instant head's hidden layer (`head_p`, 128). Captured by a `CARRY_HOOK` stand-in (`scripts/sm82c_head.py` `Capture`) that runs the stock non-exact interleaved path and keeps x at the instant rows after each round.
- **Explicit causal features (26)**, from revlog rows before the predicted review only (plus its card id and elapsed days): the card's last 3 recorded ln times + "none" flags, ln(1 + card review count), the card's last grade (one-hot), the user's running median ln time per grade 1-4 exactly as the user-median baseline computes it + ln(1 + count) per grade, the median over all grades, the user's previous recorded ln time + flag, an exponential average (0.1) of the user's ln times, ln(1 + elapsed days) + new-card flag, ln(1 + review index).
- **Head**: preset `w64-sm82c` (`expertium/variants.py`, `expertium/model/variant_model.py` `TimeMLP`, `time_mlps.0`, created last, never called by the model's forward): Linear 474 -> 128, ReLU, Linear 128 -> 64 (4 grades x 16 log-time bins, zero init), 69,056 values. Export: 331,511 -> 400,567 parameters; scorer w8 weight bytes 410,459 -> 480,859 (+70,400 B); state unchanged. **Ablation** (same pass, not exported, `ablation-head.pth`): explicit features only, 26 -> 128 -> 64, 11,712 values. Loss: SM82's censored cross-entropy of the true grade's bins.
- **Training**: the SM82b setup unchanged except the shuffle buffer (1M rows instead of 2M, rows are 474 wide): the same plan ([plan.json](plan.json) = SM82b's: 1,991 users, first 400k reviews each, 136.1M reviews, 81.9M planned labelled rows), one pass, 9,999 planned steps of 8,192 rows, AdamW lr 2e-3 (2% warm-up, cosine), no weight decay, heads only (one optimizer over both heads' disjoint parameters).
- **CPU proof** (PASS, 67 s in the launcher: [gpu/check-cpu-sm82c.json](gpu/check-cpu-sm82c.json); manual run [check-cpu-sm82c.json](check-cpu-sm82c.json)): C1 no model parameter trainable or in the optimizer, the written master's non-head tensors bitwise the released master's, trunk unchanged, both first losses ln 16; C2 resume bitwise (both heads); C3 on 2 evaluation users every grade output of the exported w64-sm82c model = the released w64 model's bitwise (fp32 and int8 state), and the capture path = the stock forward bitwise (all 7 outputs); C4a SM82's T3 perturbation (one review's answer time) changes none of the 448 frozen features of any instant row predicting that review or an earlier one (6 picks; only the later input row, column 8, changes; later features do change); C4b explicit features: changing review j's duration and grade changes no row 0..j (20 picks x 3 users), prefix tables = the full table's first rows, the per-grade median = the user-median baseline point (gap 1e-7) and the card's last time = the card-last baseline's; C5 rows = plan; C6 loss = metric; C7 the grade scorer (CPU, w8+s8, leak-free) gives equal rows and dumps for released vs exported, the time scorer scores both heads; C8 the training batch path's features = the scorer path's bitwise.
- **Smoke** (ticket 0, 37 s, 07:41:19-07:41:56): head training on 4 users (3 small + the longest cut at 400k): 4/4, 295 steps, 0 skipped, peak 8.35 GiB, first step `time loss 2.7726 / ablation 2.7726`; time scorer on 3 users with both heads; grade check on 2 users bitwise True and equal to the recorded released scores ([gpu/smoke-grade-check.json](gpu/smoke-grade-check.json)).

## What runs (`artifacts/small-model/sm82c/launch-sm82c.sh`)

Ticket purposes, in order (queue owner small-model):

1. `answer-time head on richer frozen features (SM82c): smoke: head training on 4 users (3 small + the longest, cut at 400k) + time scoring of both heads (3 users) + grade check (2 users) on the frozen released small model` (done, exit 0)
2. `answer-time head on richer frozen features (SM82c): head training on the frozen released small model (1,991 users of the quarter-epoch cool-down, first 400k reviews each, 10k AdamW steps, head + explicit-only ablation)` (queue 000818, started 07:41:58, first step ln 16)
3. `answer-time head on richer frozen features (SM82c): grade check: the released small model vs the exported model with the head (5 users, scorer rows and dumps bitwise)`
4. `answer-time head on richer frozen features (SM82c): time scoring of the head and the ablation on the 732 pool minus 9 never-fit (723, pass 1)` (and `pass 2` only if users are missing)

Then the rule on CPU: `scripts/sm82c_head.py decide` -> [sm82c-decision.md](sm82c-decision.md) / `.json`.

## Timeline (Sunday 11-01, GPU break from 16:00)

| step | expected | when |
|---|---|---|
| CPU proof (PASS, 67 s) | 2 min | 07:40-07:41 |
| ticket 1: smoke (exit 0) | 37 s | 07:41:19-07:41:56 |
| ticket 2 (queue 000818): head training | ~45-55 min (SM82b: 40.9 min; the explicit tables are built in the 3 loader workers); budget 90 min | 07:42 -> ~08:35 (worst ~09:15) |
| ticket 3: grade check (5 users, both models) | ~5 min | ~08:40 |
| ticket 4: time scoring (723 users, both heads; SM82b's took 12.7 min) | ~15-20 min | ~08:45-09:05 |
| rule | seconds | ~09:05 (worst ~10:00) |

Ends well before the 16:00 break. `gpu/train.out` prints a line every 500 steps (losses of both heads, users, reviews, minutes).

## Files

- Launcher log: `C:/Users/jonat/workspace/wt/sm82c/artifacts/small-model/sm82c/gpu/launch-sm82c.log` (nohup: `launch-sm82c.nohup`).
- `gpu/check.out` (smoke), `gpu/train.out`, `gpu/grade.out`, `gpu/time.out`; rows `gpu/grade-check-rows.jsonl` + `gpu/grade-dump/`, `gpu/grade-check.json`, `gpu/sm82c-time.jsonl` (head), `gpu/sm82c-time-ablation.jsonl` (ablation).
- Head training folder: `artifacts/small-model/sm82c-head-released` (`master.pth` = released tensors + `time_mlps.0.*`, `ablation-head.pth`, `train.json`, `resume.pt`).
- Grade-check users ([grade-users.json](grade-users.json)): 4396, 722, 1286, 3455, 3781 (as SM82b).
- Read only: SM82's baselines `C:/Users/jonat/workspace/wt/sm82/artifacts/small-model/sm82/time-baselines.jsonl`, SM82's rows `.../sm82/gpu/sm82-time.jsonl`, SM82b's rows `C:/Users/jonat/workspace/wt/sm82b/artifacts/small-model/sm82b/gpu/sm82b-time.jsonl`, the released model's recorded scores `C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/gpu/small-lf-neighbour.jsonl`.

## How to read the result

The last log lines are `rule exit 0: [sm82c-rule] 722 users: grade unchanged True|False; time LogLoss X (ablation A) vs card's last / user median / SM82b / SM82, |ln err| ...: PASS|FAIL-GRADE|FAIL-TIME` and then `done`. Pass = the head beats the card's last time and the user's running median per grade on the censored time LogLoss and the mean |ln error| of its median (uncensored), each with the 95% CI of the review-weighted difference clear of zero (paired user bootstrap 2,000, seed 0), AND grade unchanged (released vs exported scorer rows, every grade field, and per-review dumps bitwise equal on the 5 users). `sm82c-decision.md` has the predictor table (head, ablation, SM82b, SM82, both baselines; given the true grade and grade-mixed), the paired comparisons (incl. "head vs the ablation: what the model features add" and "head vs SM82: the remaining cost of freezing"), the per-grade table with P(under 5.5 s) predicted / actual, and the size. Any `STOP:` line ends the launcher and names the step and its output file.

Follow-up (CPU, after `done`): commit `gpu/*.json`, `gpu/*.jsonl`, `gpu/*.log`, `gpu/*.out`, `sm82c-decision.{md,json}` and `../sm82c-head-released/train.json` on `sm82c` (`git add -f`; not `gpu/grade-dump/`, not `resume.pt` or the `.pth` files), push; the main session updates the Factory page. Re-run the rule by hand:

```
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; WT=C:/Users/jonat/workspace/wt/sm82c; G=$WT/artifacts/small-model/sm82c/gpu
$PY -X utf8 $WT/scripts/sm82c_head.py decide --time-rows $G/sm82c-time.jsonl --ablation-rows $G/sm82c-time-ablation.jsonl \
  --grade-check $G/grade-check.json --train-json $WT/artifacts/small-model/sm82c-head-released/train.json
```

## Relaunch (resumes: finished steps are skipped, the head training resumes from its last buffer pass, scoring skips users scored by both heads)

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm82c/artifacts/small-model/sm82c/launch-sm82c.sh >> C:/Users/jonat/workspace/wt/sm82c/artifacts/small-model/sm82c/gpu/launch-sm82c.nohup 2>&1"'; ProcessStartupInformation=$si}
```

To stop between tickets: `touch C:/Users/jonat/workspace/wt/sm82c/artifacts/small-model/sm82c/gpu/STOP` (remove it before relaunching). To stop the head training itself, cancel its queue ticket; a relaunch resumes from `resume.pt` (written after every 1M-row buffer pass). After a Claude app restart, check the launcher is alive; if the log has been silent for more than 10 minutes outside a ticket, relaunch.

Do not edit `scripts/`, `expertium/` or `rwkv/` in this worktree while it runs.
