# Frozen-model grade side head (SM87): handoff

Branch `sm87`, worktree `C:/Users/jonat/workspace/wt/sm87` (cut from `sm82c` at d5d381f3; sources at c1c46556). Launched Sunday 11-01 10:48 via WMI (hidden). Approved by the user Sunday 11-01 ~10:45 as a cheap try.

## What was built (`scripts/sm87_head.py`)

- **Goal**: does a small side head on the frozen released small model (the leak-free small model (SM72s) 25 epochs + the 1-epoch 32k Muon cool-down, `C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/scale-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon/expertium/E89-F65e-F77a/master.pth`, read only) improve the instant (grade) LogLoss when it gets inputs the frozen model was never asked to keep? Released instant 0.27654 on 722 users (+0.283% vs the big model (1 leak-free epoch), 0.27576).
- **Residual design**: corrected instant logits = released instant logits + side head(inputs); zero-init output Linear, so the start is exactly the released prediction. Trunk, curve and the released instant head untouched (the master is not rewritten; the side head is its own file).
- **Inputs (494)** at each instant row: the SM82c frozen states (residual stream after the encoder and after each of the 4 layer rounds, layer-normalised, 5 x 64, plus the instant head's 128-wide `head_p` layer); the released instant prediction (4 log-probabilities / 5, logit P(pass) / 5); 41 explicit causal grade features from earlier reviews only plus the review's own card id / elapsed days / elapsed seconds (the frame columns the released model's interval input uses, so the same elapsed definition incl. the same-day answer-time content of elapsed seconds, the duration-leak item (SM31)): the card's last 3 grades (one-hot + none) and ln times (+ none), ln review count, ln lapses, ln previous interval, ln elapsed days / seconds, same-day flag, user pass rate last 50 / 500 / all, ln review index, grade mix last 500, running median ln time per grade + all, running average ln time.
- **Heads**: side head Linear 494 -> 128, ReLU, Linear 128 -> 4 (63,876 values); **ablation without the model states** in the same pass: released prediction + explicit features, 46 -> 128 -> 4 (6,532 values). Loss = the scorer's instant metric (binary LogLoss of clamp(1 - softmax[Again])).
- **Training**: SM82c's 1,991 plan users (first 400k reviews each), every labelled instant row (117,706,189 = all non-first reviews of a card, [plan.json](plan.json)), one pass, 1M-row shuffle buffer, 14,369 AdamW steps of 8,192 (lr 2e-3, 2% warm-up, cosine), fp32 frozen features, heads only.
- **Scoring**: the scorer's own path (`small_model_heldout.py` model construction, w8+s8, leak-free inputs, its metrics code): one forward per user gives the released rows (bitwise the scorer's) and from the same captured features the corrected and ablation rows (`released/w8+s8`, `sm87/w8+s8`, `sm87-ablation/w8+s8` in `gpu/sm87-rows.jsonl`). On 5 users (`apply-users.json`) a second forward applies the correction inside the model and checks the curve outputs stay bitwise.
- **CPU proof** (PASS, 102 s by hand: [check-cpu-sm87.json](check-cpu-sm87.json); again in the launcher, 91 s: `gpu/check-cpu-sm87.json`): C1 no model parameter trainable or in the optimizer, trunk unchanged, master sha256 unchanged, first step corrected = released bitwise and first loss = released LogLoss; C2 resume bitwise; C3 zero-init heads give rows equal to the released rows on every grade field, the released rows equal `small_model_heldout.py`'s (CPU w8+s8) on every grade field, trained heads keep the curve fields equal and the in-model apply-check keeps the curve bitwise; C4a T3 duration perturbation changes no frozen feature / released logit of rows predicting that review or earlier; C4b explicit features: changing review j's grade, duration and elapsed seconds changes no row before j and nothing but row j's own elapsed-seconds column; prefix tables = full table's first rows; C5 rows = plan, targets = frame grade > 1; C6 loss = sklearn log_loss; C8 training-path inputs = score-path inputs at fp32 bitwise.

## What runs (`artifacts/small-model/sm87/launch-sm87.sh`)

One GPU ticket (so no other ticket lands between smoke, training and scoring), queue owner small-model, cap 0.8:

1. `frozen-model grade side head (SM87): smoke (side head training on 4 users + scoring 3 users), then side head training on the frozen released small model (1,991 users of the quarter-epoch cool-down, first 400k reviews each, 117.7M instant rows, 14.4k AdamW steps, side head + ablation without the model states), then scoring on the 732 pool minus 9 never-fit (723 users, w8+s8 leak-free: released, corrected and ablation rows; correction applied inside the model on 5 users)` (queue 000828, expected 122 min; registered 10:49:57, started 11:03:40; smoke passed 11:04:10, training started 11:04:10, first step = released bitwise)
2. only if users are missing: `frozen-model grade side head (SM87): scoring on the 732 pool minus 9 never-fit (723 users, pass 2: users missing after pass 1)`

Then on CPU: `compare` (apply-check + released = recorded on the 5 users) -> `decide` -> [sm87-decision.md](sm87-decision.md) / `.json`.

## Timeline (Sunday 11-01, GPU break from 16:00)

| step | expected | when |
|---|---|---|
| CPU proof in the launcher (PASS) | 2 min | 10:48-10:50 |
| queue wait | 14 min | 10:50 -> 11:03:40 |
| smoke inside the ticket (PASS: 4/4 users, 383 steps, 0 skipped, peak 8.36 GiB, first corrected = released bitwise; scoring 3 users: curve equal, released = recorded scores, apply-check 3/3) | 30 s | 11:03:40-11:04:10 |
| side head training | ~50-65 min (SM82c: 45 min for 81M rows; here 118M rows, same forward) | 11:04 -> ~12:10 (worst ~12:45) |
| scoring 723 users (one forward each, 3 row sets) | ~20-30 min | ~12:10 -> ~12:40 (worst ~13:20) |
| compare + rule (CPU) | 1 min | ~12:40 (worst ~13:20) |

If the ticket cannot start before ~13:55 (122 min + 3 before 16:00), the launcher waits through the break and starts it after 01:00.

## Files

- Launcher log: `C:/Users/jonat/workspace/wt/sm87/artifacts/small-model/sm87/gpu/launch-sm87.log` (nohup: `launch-sm87.nohup`); the GPU steps log `gpu: ...` lines there.
- `gpu/check.out` (smoke), `gpu/smoke-compare.json`, `gpu/train.out` (a line every 500 steps: side head / ablation / released training LogLoss), `gpu/score.out`, rows `gpu/sm87-rows.jsonl`, `gpu/apply-compare.json`.
- Training folder: `artifacts/small-model/sm87-side-head` (`side-head.pth`, `ablation-head.pth`, `train.json`, `resume.pt`).
- Read only: the released master; its recorded 732 scores `C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/gpu/small-lf-neighbour.jsonl`; the big model's `C:/Users/jonat/workspace/wt/sm72/artifacts/small-model/sm72/gpu-big/bigleak-neighbour.jsonl`.

## How to read the result

The last log lines are `[sm87-compare] ...: PASS|FAIL`, then `rule exit 0: [sm87-rule] 722 users: instant A -> B (P points, CI [...]); ablation ...; model states add ...; curve unchanged True; released = recorded True: GO|GREY|STOP`, then `done`. Points = review-weighted LogLoss change as % of the big model's (1 leak-free epoch) LogLoss, paired user bootstrap 2,000 (seed 0) (scripts/sm77_rule.compare); negative = better. Rule: go if the instant improves by >= 0.15 points with the CI clear of zero; stop if the gain is < 0.05; in between report. `fail-curve` if any curve field differs. Also check "released = recorded True" (my scorer path reproduces the recorded released scores on all 722 users).

Follow-up (CPU, after `done`): commit `gpu/*.json`, `gpu/*.jsonl`, `gpu/*.log`, `gpu/*.out`, `sm87-decision.{md,json}` and `../sm87-side-head/train.json` on `sm87` (`git add -f`; not `resume.pt` or the `.pth` files), push; the main session updates the Factory page. Re-run the rule by hand:

```
PY=C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe; WT=C:/Users/jonat/workspace/wt/sm87; G=$WT/artifacts/small-model/sm87/gpu
$PY -X utf8 $WT/scripts/sm87_head.py decide --rows $G/sm87-rows.jsonl --check $G/apply-compare.json --train-json $WT/artifacts/small-model/sm87-side-head/train.json
```

## Relaunch (resumes: finished steps are skipped, the training resumes from its last buffer pass, scoring skips users with all 3 rows)

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm87/artifacts/small-model/sm87/launch-sm87.sh >> C:/Users/jonat/workspace/wt/sm87/artifacts/small-model/sm87/gpu/launch-sm87.nohup 2>&1"'; ProcessStartupInformation=$si}
```

To stop between steps: `touch C:/Users/jonat/workspace/wt/sm87/artifacts/small-model/sm87/gpu/STOP` (remove it before relaunching). To stop mid-training, cancel queue ticket 000828; a relaunch resumes from `resume.pt`. After a Claude app restart, check the launcher is alive; if the log has been silent for more than 10 minutes outside a ticket, relaunch.

Do not edit `scripts/`, `expertium/` or `rwkv/` in this worktree while it runs.
