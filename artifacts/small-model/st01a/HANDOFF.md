# Curve consistency on real rapid Good pairs (ST01a): handoff (stage 1 running)

Branch `st01a` (cut from `sm80`), worktree `C:/Users/jonat/workspace/wt/st01a`. Stage 1 launched 11-01 10:33 via WMI
(hidden). It is a measurement only: no training.

## What runs

`artifacts/small-model/st01a/launch-st01a-s1.sh` runs `scripts/st01a_pairs.py` once per model, in GPU queue tickets of
about 15-20 min (time limit 1350 s per ticket; up to 3 passes per model until all 723 users have rows; finished users
are skipped).

- **Models.** First the released small model: the leak-free small model (SM72s), epoch-25 cool-down, label
  `s-w64kd-final-sc-lanes-leakfree-dec25-1ep32k-muon`. Then the big model (1 leak-free epoch), label
  `big-leakfree-1ep`.
- **Settings.** Both use w8+s8 and deleted cards neighbour, with the same protocol as `scripts/small_model_heldout.py`.
- **Users.** The 723 fit users of the 732 set (`wt/sm72s/.../eval-users-732-fit.json`). The 3 long users are left
  out. Every review of these held-out users counts.
- **Pairs.** A pair is two consecutive reviews a -> b of the same card, with b answered Good and a gap of 0-7 days.
  For each pair the script stores:
  - curve_a(gap + H) and curve_b(H) for H = 7, 30, 90 and 365 days;
  - recall at b;
  - where the card has a real next review c: curve_b(e) and curve_a(gap + e), plus c's outcome.

Ticket purposes, in order (pass 2/3 only if a pass hits its time limit):
1. `curve consistency on rapid Good pairs (ST01a): stage 1 measure, curves of the released small model (the leak-free small model (SM72s), epoch-25 cool-down) on 723 users of the 732 set (pass 1)`. Ticket 000823 started 10:33.
2. `curve consistency on rapid Good pairs (ST01a): stage 1 measure, curves of the big model (1 leak-free epoch) on 723 users of the 732 set (pass 1)`.

Expected end: about 11:15-11:30 on 11-01. The small model ran at about 80 users per 100 s, so about 15 min. The big
model needs about 1.5x that. Both fit well before Sunday's 16:00 break, and the launcher checks quiet.json itself.

## Files

- Launcher log: `artifacts/small-model/st01a/gpu/launch-st01a-s1.log`. The last line is `done`, or a `STOP:` line.
  WMI output goes to `gpu/launch-st01a-s1.nohup`.
- Per-user rows: `gpu/pairs-small.jsonl` and `gpu/pairs-big.jsonl`. Ticket outputs: `gpu/pairs-{small,big}.out`.
- Pair files: `work/pairs-small/<user>.npz` and `work/pairs-big/<user>.npz`. These are large: do not commit them.

## Checks already done (CPU, 2 users 2388 and 4676, `work/smoke-*`)

- The p_curve recomputed from the head outputs equals `get_loss`'s p_curve exactly (max abs diff 0.0), for both
  models.
- The per-user curve LogLoss matches the scorer's 732 rows to about 1e-6. The script stores `curve_log_loss` per user,
  so the check can be repeated on all users.
- The small model's curve is the 3-term power law of the w64 variant (`gru_forgetting_curve`, no ahead residual). The
  big model's is the stock mixture + residual. `scripts/st01a_pairs.py` handles both.
- Smoke hint (2 users only, not a result): after a rapid Good the 365-day recall goes **down** in about 80-90% of
  pairs, for BOTH models. The mean change is about -0.07, and 20-24% of pairs move by more than 0.1. Most pairs are
  young cards (a = first review or a same-day step). On mature cards the small model moved slightly down (-0.02) and
  the big model stayed flat (+0.01). The real next review (0-7 days) favoured the curve after the Good. At 7-30 days
  it favoured the curve before the Good. So the question is whether the fall is a correction of an over-optimistic
  tail, or noise.

## Follow-up (fresh agent)

1. `git fetch` and `git merge --ff-only` in `wt/st01a`. Confirm the log ends with `done` and both jsonl files have 723
   rows without `"error"`. If the launcher died, relaunch with the command below.
2. Build the stage 1 tables (CPU, about a minute or two):
   ```
   cd C:/Users/jonat/workspace/wt/st01a
   C:/Users/jonat/workspace/anki-rwkv/.venv/Scripts/python.exe -X utf8 scripts/st01a_report.py \
     --model small=artifacts/small-model/st01a/work/pairs-small --model big=artifacts/small-model/st01a/work/pairs-big \
     --users-file C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s/eval-users-732-fit.json \
     --out artifacts/small-model/st01a/st01a-stage1
   ```
   The report writes `st01a-stage1.{md,json}` with four tables:
   - jumps by slice (gap, interval before a, Good expected);
   - the 365-day detail;
   - tail calibration (which curve was closer at the real next review);
   - the stress slice (mature card, then 1-4 rapid Goods, 365-day recall step by step).

   Optional sanity check: compare `curve_log_loss` in `gpu/pairs-*.jsonl` with the scorer rows
   (`wt/sm72s/.../gpu/small-lf-neighbour.jsonl` and `wt/sm72/.../gpu-big/bigleak-neighbour.jsonl`, label `/w8+s8`).
   They should agree to about 1e-5.
3. **Stage 2 decision rule.** Use the slice "rapid (gap 1-3 d, a passed)", horizon 365 days.
   - **Stage 2 needed** if both of these hold:
     - The small model is clearly jumpier than the big model (1 leak-free epoch). That means at least one of:
       - its share of pairs with |jump| > 0.1 is at least 1.5x the big model's;
       - its share of pairs going down by more than 0.1 is at least 5 points above the big model's;
       - its mature slice (interval before a >= 21 d) goes down in more than 50% of pairs while the big model's does
         not.
     - The tail calibration does NOT back the fall. That means curve_b (after the Good) has LogLoss no better than
       curve_a at 30+ days to the next review. If the curve after the Good predicts the real later reviews better,
       the fall is a correction, and a penalty pulling the two curves together would hurt.
   - **Not needed** if the small model is about as stable as the big model (both drop alike), or if curve_b is better
     calibrated in the tail. In that case write up `st01a-decision.md`, commit, and stop.
4. If stage 2 is needed, here is the design to implement:
   - **Arm.** The quarter-epoch 32k Muon cool-down from the leak-free small model's (SM72s) epoch 21. Pair it against
     the tiny architecture additions (SM77) reference: `wt/sm77/artifacts/small-model/sm77-ref-e21-dec-q32k-muon`,
     scored as `sm77-ref-e21-q32k/w8+s8` in `wt/sm77/.../sm77/gpu/sm77-neighbour.jsonl`.
   - **Infrastructure.** Copy the data-order re-seed (SM80) launcher, `wt/sm80/artifacts/small-model/sm80/launch-sm80.sh`,
     but with the default data seed and only the cool-down step.
   - **Penalty.** Add a small term in `_get_loss_variant` (`expertium/model/variant_model.py`), on pairs inside the
     same training window: consecutive positions of one card, b Good, gap 1-3 days, a passed. The term is
     lambda * mean over H in {30, 90, 365} d of (logit curve_b(H) - logit curve_a(gap + H))^2.
     - If the calibration says curve_b is better, use a one-sided version that pulls only curve_a toward curve_b
       (curve_b detached).
     - Start with lambda chosen so the term is about 1-2% of the curve loss.
   - **CPU proof first.** With weight 0 the run must be bit-identical (loss, gradients and protocol vs the
     reference's decay.json).
   - **Rule.** On 722 users, curve and instant must each be within +0.03 points of the reference. The jump metrics
     (re-run `st01a_pairs.py` + `st01a_report.py` on the arm) must be clearly better.
   - **Budget.** About 2-2.5 h GPU. Resumable across the break (Sunday from 16:00, resumes 01:00).

## Relaunch (resumes: finished users are skipped)

```powershell
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/st01a/artifacts/small-model/st01a/launch-st01a-s1.sh >> C:/Users/jonat/workspace/wt/st01a/artifacts/small-model/st01a/gpu/launch-st01a-s1.nohup 2>&1"'; ProcessStartupInformation=$si}
```

To stop it: `touch C:/Users/jonat/workspace/wt/st01a/artifacts/small-model/st01a/gpu/STOP`. It only checks this
between tickets. Remove the file before you relaunch.

Do not edit `scripts/`, `expertium/` or `rwkv/` in this worktree while a ticket runs.
