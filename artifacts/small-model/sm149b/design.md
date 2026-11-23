# Rank-2 states with a tail-energy penalty (SM149b): design

Written Mon 2026-11-23, before the training run. Branch `sm149b` (from `origin/sm149`), worktree
`C:/Users/jonat/workspace/wt/sm149b`; all new code in this folder. Run tree: SM149's `C:/Users/jonat/workspace/wt/sm149-run`
(branch `sm149-run` = `origin/sm148`, cea0f931), reused and **not edited**. Track: small model. Status: pre-registered
(this commit) before the launch; launch and PIDs in `HANDOFF.md`.

## Question

The rank-2 cool-down (SM149) trained Small two-head (SM118) for a short cool-down with every card and note write
stored in the deployed rank-2 int8-factor format, with a straight-through estimator (STE). It had **no effect**: rank 2
(Q2) still cost +0.073% [+0.066, +0.080] on the instant head against the original at int8; against each arm's own
int8 rows the rank-2 cost was +0.052% for the QAT arm and +0.079% for the control, a ratio of 0.66 (the gate's line
was 0.5).

Outside research (a pointer only; we have no license to its code, which was not opened) saw the same weak result with
the STE alone, and reports that adding a **tail-energy penalty** removed the rank-2 tax: the energy of the state
beyond rank r, on the state before truncation, was penalised in the loss, and it fell about 45 times over a 124-step
decay.

**Question: does the rank-2 cool-down of SM149 plus a tail-energy penalty make Small two-head (SM118) robust enough
that rank 2 (Q2) or rank 2 then 4 (T24) passes the rules rank 4 passes today?**

| plain name (ID) | what it is here |
|---|---|
| the rank-2 cool-down (SM149) | the recipe, the STE hook (`sm149_qat.py`), the trainer entry, the QAT and control arms of Small two-head (SM118), their rows: everything is reused, the new arm adds one loss term |
| the low-rank int8 states experiment (SM148) | the codec (`scripts/state_lowrank.py`, the fused CUDA encode), the labels REF / Q2 / T24, the original model's rows, the rules |
| Small two-head (SM118) | the release model trained here (w64, K = 16, 4 heads, 2 card + 1 note WKV layers, Muon, replay KD targets) |

## The penalty

**Definition (per card / note write, per row and head).** S' = the float update of the write (S' = S G + v k^T, the
state before the codec; the stored state is the codec's rank-2 int8-factor reconstruction of S', exactly as in SM149):

    tail = sum_{i > 2} sigma_i(S')^2 / (sum_i sigma_i(S')^2 + eps),   eps = 1e-6

**Loss term.** `loss += lambda x mean(tail)`, the mean over every sampled (row, head) entry of every coded layer and
stream of one update (card layers 0 and 1, note layer 0). lambda = 1.0 (outside research's value; it was about 2.4% of
the training loss at its start; ours: mean tail 0.0188 on Small two-head (SM118)'s states, about 2.0% to 2.8% of the
loss on the proof users, see "Proofs").

**Sampling.** One write step in 8 (stride 8), counted per stream: each stream (card, note) has its own step counter
that runs across its hook calls, and a step t of a call is sampled when (counter + t) mod 8 = 0, so the two streams are
sampled at the same rate. Among the rows written at a sampled step, rows whose write has fewer than 3 answers (n < 3)
are masked: their S' has rank <= 2 and the tail is exactly 0. Zero and non-finite states are masked too.

**Value and gradient without an SVD.** With T = the best rank-2 approximation of S' (the factored solve of SM148 with
float factors, codec `lr2fx8`, the same fused CUDA kernel as the stored codec; computed without gradient),
||S' - T||^2 = sum_{i > 2} sigma_i^2 exactly (Eckart-Young), and by Danskin's theorem its gradient is 2 (S' - T) with T
held fixed, because T minimises ||S' - X||^2 over rank-2 matrices X. So

    d tail / d S' = (2 (S' - T) - 2 tail S') / (||S'||^2 + eps)

It has no 1 / (sigma_i^2 - sigma_j^2) terms: at a tie sigma_2 = sigma_3 any minimiser gives a valid (sub)gradient, and
for rank <= 2 the gradient is 0. A differentiable `svdvals` per step (outside research's route, which blew its memory
at every step) is not needed; it is only the proofs' reference.

**Where the gradient goes.** The full trajectory: the backward adds lambda / N x d tail / d S' (N = the update's
sampled entry count, a device scalar, no host sync) to dL / d S' at each sampled write inside the straight-through WKV
backward, which then carries it through the coded trajectory to every earlier step, exactly as autograd through an STE
loop with the penalty in the loss would (proof B1). The STE path itself is SM149's, unchanged. Implementation: a copy of
`expertium.model.state_vq_hooks.ste_backward` with that one injection. The fused K = 16 backward kernel of SM149 cannot
take the injection, so a hook call with sampled entries runs the torch loop; a call without keeps the kernel.

**lambda = 0 is SM149.** With lambda = 0 the hook runs SM149's code path; proof B0 shows bitwise equal outputs,
gradients, losses and weights after the optimizer steps. With lambda > 0 the forward value is still SM149's bitwise
(the penalty adds gradients only).

**Logging.** The mean tail of every update pass goes to `<run folder>/tail.jsonl` (one JSON line per pass:
`tail_mean`, `entries`, `penalty`; flushed every 50 passes, one host sync per flush), and the training log prints a
summary line every 100 passes (`[sm149b] tail energy, update passes a-b: mean ...`). The first 3 passes are the
trainer's discarded JIT primer.

## Code (no source change in the run tree)

- `sm149b_qat.py`: a copy of SM149's `sm149_qat.py` (hook, shims, install) extended by the penalty (`tail_terms`,
  `ste_backward_tail`, the per-update `_Group`, the log), and the CPU proofs B0 to B4 (`cpu`).
- `sm149b_run.py`: a copy of SM149's `sm149_run.py`: the run tree's `scripts/expertium_decay.py` with the hook
  installed for a `state_vq` option whose recipe is a low-rank label. lambda: env `SM149B_LAMBDA`, else the option's
  `tail_lambda`, else 1.0; stride: env `SM149B_STRIDE`, else `tail_stride`, else 8. The effective values are recorded in
  result.json (`state_vq.tail_penalty`).
- `sm149b_prepare.py`: builds the arm with SM149's own prepare code (only the folder name changed), then sets
  `"state_vq": {"recipe": "lr2q8x8", "tail_lambda": 1.0, "tail_stride": 8}`; `pair` checks that the folder differs from
  SM149's trained QAT arm folder only there.
- `launch-stage1.sh` (the detached launcher, a copy of SM149's), `gpu/smoke-ticket.sh` + `gpu/smoke_readout.py` (the
  GPU smoke), `sm149b_report.py` (dataset-wide read-out and gate, a copy of SM149's report with the new arm).

## Arms

- **New arm** `sm149b-sm118-tail1`: SM149's QAT arm `sm149-sm118-qat` (start = Small two-head (SM118)'s final
  `master.pth` and Muon `optimizer.pth`, sha256 b4fe1175...; the same 794 users and 67,256,032 reviews of
  `random.Random(149)`; the release options, loss and targets; learning rate 0 -> 7e-5 over 64 updates then
  one_minus_sin to 0; the same seed and steps; rank 2 int8 factors on every card and note write with the STE) **plus the
  penalty at lambda = 1.0, stride 8**. `pair` OK: the folders differ only by the two option keys.
- **Reused, not retrained:** SM149's control arm `sm149-sm118-ctl` (the identical cool-down with the codec off) and
  SM149's QAT arm, with their 732-set rows (`wt/sm149/artifacts/small-model/sm149/gpu/sm149-rows-732.jsonl`); the
  original model's rows of SM148 (`wt/sm148/artifacts/small-model/sm148/gpu/sm148-rows-732.jsonl`,
  `cpu/sm148-longcard-rows.jsonl`). Same scorer commit (the run tree); an integrity ticket re-scores SM149's QAT arm at
  REF on the 3 smoke users, which must equal SM149's rows to 1e-6.

## Scoring

As SM149: the run tree's `scripts/small_model_heldout.py`, `--pool 732 --users-file <sm72s eval-users-732-fit.json>
--inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets
0,1,3,5,9,16,32,128,300,1000`, `PYTORCH_JIT=0`, GPU cap 0.8, then 0.92 for a retry pass of missing users (memory
only). Labels for the new arm only: REF (`w8+s8hx8`), Q2 (`w8+lr2q8x8`), T24 (`w8+lr2.16.4q8x8`). The long-card users
on the CPU (`users-longcard.json` of SM148, 50 users, `--device cpu --workers 1 --dump-scope all`, one fresh process
per user, at most 2 at a time while the GPU runs, memory guard 5.9 GB with SM148's fixed `memguard.ps1`), new arm REF,
Q2 and T24 (HANDOFF.md step 2).

## Pre-registered rules (SM149's rules, unchanged; the reference is the original Small two-head (SM118) at int8)

Statistic as SM148: % = 100 x (sum of the label's LogLoss / sum of the reference's LogLoss - 1), review-weighted;
paired user bootstrap, 2,000 draws, seed 0, 95% CI. Heads: curve and instant. "Original" = Small two-head (SM118)
scored by SM148 (its REF rows); "new" = the new arm's final model.

- **R1 finite:** no non-finite row or LogLoss for any label (732 set on the GPU, long-card users on the CPU).
- **R2 dataset-wide:** new model at the label vs the ORIGINAL at int8 (REF), 732 set rule users, CI upper bound
  **<= +0.05%** per head.
- **R3 long cards:** on the 50 long-card users (CPU), reviews whose card had 128 or more earlier answers, new at the
  label vs the original REF: point **<= +0.5%**, CI upper **<= +1.0%** per head.
- **G per-card guard:** share of the long-card users' cards whose worst review has |dR| > 0.10 (dR = p(new, label) -
  p(original, REF)) **<= 1%** per head.
- **R6 the new model at int8:** new REF vs original REF, CI upper bound **<= +0.05%** per head.
- **R5 coverage:** at least 720 of the 722 rule users per label on the GPU, at least 48 of the 50 long-card users on
  the CPU; pairs use only users both sides have. Otherwise the report is pending and the main session decides.

Reported, no verdict: new Q2 / T24 vs the new REF (the rank-2 cost after training), vs SM149's QAT arm at the same
label (the penalty's own effect), vs SM149's control; the cost per answer-count bucket; the tail energy over training
(`tail.jsonl`: first vs last 100 passes, outside research saw about 45 times lower); training reviews/s and GPU memory.

## Stage gate (fixed now)

- **Pass:** Q2 or T24 passes R1, R2, R3, G and R6 (and R5) for the new model. Then the main session decides on the
  other five release models (Small Instant-Only (SM113), Big two-head (SM91s2), Big Instant-Only (SM99), the two
  six-layer Tiny models (SM130)) with this recipe.
- **Partial:** no rank-2 label passes, but C_new <= 0.5 x C_ctl on the instant head, with C_new = new Q2 vs new REF
  and C_ctl = SM149's control Q2 vs control REF = **+0.079%** (point estimates, %), i.e. C_new <= +0.0395%.
  Then the **single follow-up: lambda = 3.0** (`sm149b_prepare.py arm --lam 3.0` -> `sm149b-sm118-tail3`, everything
  else identical, the same rules and gate). A second partial stops too.
- **No effect** (C_new > +0.0395%): stop; every size keeps rank 4 with int8 factors (Q4, the go of SM148).

## Proofs (CPU, `sm149b_qat.py cpu` -> `check-cpu-sm149b.json`, log `cpu-proof.out`; code tree `wt/sm149-run` at cea0f931, clean)

All checks pass (**ALL OK**, 164 s, 4 threads):

- **B0a lambda = 0 is SM149, synthetic:** the new hook without the penalty vs SM149's `LowRankSTEWKV`: outputs, final
  states and every input gradient **bitwise** equal, K = 16 and 32, zero and carried start. With lambda = 1 the
  outputs are still bitwise SM149's, and the gradients differ.
- **B0b lambda = 0 is SM149, real trainer path:** Small two-head (SM118)'s final weights and Muon state, the release
  options (strided-lane carry, show-time inputs, replay KD targets, alpha 0.5), users 4745 and 2308 in 256-review
  windows, 6 updates at LR 7e-5: every loss and every gradient (sha256 over all parameters) bitwise equal to SM149's
  hook, and the weights after the 6 Muon steps bitwise equal. With lambda = 1: the first loss bitwise SM149's, the
  gradients differ, all finite. CPU seconds per update: SM149 1.51, lambda 0 1.43, lambda 1 1.50.
- **B1 the penalty's gradient is right:** against an independent reference (autograd through a straight-through loop
  with the tail from a differentiable float64 `torch.linalg.svdvals` of every sampled S', the same steps and mask):
  the penalty's own gradient (no output gradient) agrees to at most 1.6e-5 relative, the full gradient (output +
  penalty) to 3.1e-7, the mean tail to 7.4e-8 relative; K = 16 and 32, zero and carried start, stride 8 and 3; forward
  bitwise equal to the reference.
- **B2 ties and young states:** the tail and its gradient on crafted 16 x 16 states (random rotations): sigma_2 =
  sigma_3 tie (tail 1/11), triple tie, near tie, rank 2 and rank 1 (tail 0 to 2e-14), zero (masked) and a tiny-norm
  state (gradient bounded by eps): every value equals svdvals' to 1e-5 or better and every gradient is finite. Young
  histories (56% of the writes at n < 3, stride 1): gradients finite; the 6 rows that never reach 3 answers get
  gradients bitwise equal to the no-penalty run (the mask holds), the other rows differ.
- **B3 the tail moves down:** from Small two-head (SM118)'s end on users 4745 and 2308 (7 updates of 256 reviews),
  the same updates at lambda 0, 1 and 3, then the mean tail of every write of the same batches (stride 1, no
  gradient; 8,756 states). Start 0.01885. At LR 7e-5 (the cool-down's peak): lambda 0 ends at 0.01887, lambda 1 at
  0.01853 (-1.8% vs lambda 0), lambda 3 at 0.01831 (-2.9%). At LR 7e-4: 0.01888 / 0.01586 (-16.0%) / 0.01420 (-24.8%).
  The training loss on those updates moves by less than 0.005% (0.96477 / 0.96475 / 0.96473 at 7e-5). The mean tail
  of these states is 0.019, so lambda = 1 adds about 2.0% (these users' loss 0.96) to 2.8% (the training average 0.68)
  of the loss: outside research's 2.4%.
- **B4 CPU cost:** forward + backward of one coded layer on split shapes of the SM32b sample (rows x steps 512 x 16,
  2048 x 4, 128 x 64): lambda 1 / lambda 0 = 1.08, 0.99, 1.06; the penalty keeps 1.6% to 3.4% of the stored-state
  record for the backward (0.52 MiB / 32 MiB and 1.1 MiB / 32 MiB).

## GPU smoke and overhead

Queue ticket **001519** (owner sm149b, 6 min 54 s, cap 0.8, log `gpu/smoke-ticket.out`):

- **Trainer smoke** of the prepared arm through `sm149b_run.py smoke` (2 users x 512 reviews, scratch copy): complete,
  2 updates, the hook installed with lambda 1 and stride 8 from the option, 5 tail lines logged
  (`gpu/smoke-trainer.out`).
- **Real-data throughput** on a scratch copy of the prepared folder, the same users and windows as SM149's QAT arm
  (its `updates.jsonl`), 129 updates, paired over updates 4 to 129 (126 updates, 3.54M reviews; read-out
  `gpu/smoke_readout.py` -> `gpu/smoke-sm149b.json`): **9,401 reviews/s** (GPU time) vs **10,327** for SM149's QAT arm
  on the same updates, **1.10 x** its GPU time. SM149's QAT arm averaged 9,993 reviews/s over its whole run, so about
  **9,100 reviews/s** is expected here (2.9 x 1.10 = about 3.2 x the control's time). Training loss over those
  updates: 0.72056 vs 0.72047 (the penalty's gradient only moves the weights; its value is not in the logged loss).
- **Memory:** peak allocated **8,440 MiB** (SM149's QAT arm: 8,556 MiB over its whole run), reserved 11,626 MiB of the
  13.0 GB cap; nvidia-smi at most 12,759 MiB in use on the card (desktop included); GPU utilisation 54% (host-bound
  per-step loop, as SM149).
- **Hook calls:** 1,481 (1,451 from a carried state), all penalised; 938 backward calls had sampled entries and ran the
  torch loop instead of the K = 16 kernel (the cost above). Tail log: 134 passes, about 29,000 sampled states per
  pass; mean tail 0.0170 over passes 1-100 and 0.0208 over 101-134 (no trend yet at the end of the warm-up; the data
  differ per user).
- One launcher pitfall found: a scratch copy of a prepared folder keeps the original folder's `control_dir`, so the
  snapshot stop has to be sent to the original's control directory (done; the STOP file was removed afterwards and
  the prepared folder holds no snapshot).

## Estimates

- **Training:** 67,256,032 reviews at about 9,100 reviews/s = **about 123 min** (range 115 to 140; SM149's QAT arm
  112 min x the measured 1.10), one ticket, snapshot every 100 users; expected minutes passed to the queue: 135.
- **Integrity** (SM149's QAT arm REF on the 3 smoke users): 0.3 min. **Trainer smoke:** 0.3 min.
- **Scoring** of the new arm on the 732 set (REF, Q2, T24): about 34 min (SM148's per-label minutes for Small: REF 5.5,
  a low-rank label 14), plus a short retry pass at cap 0.92 for the two largest users (SM149: 1 min).
- **GPU total about 2.7 h** (range 2.5 to 3.1 h). **CPU:** the long-card users, new arm x 3 labels, about 1.5 to 2 CPU
  hours (SM148: about 25 min per Small user-size pass for 4 labels on 2 models; SM149 estimated 2.5 h for 5 labels),
  alongside the scoring.
- Follow-up lambda = 3.0 (only after a partial): the same, about 2.7 GPU h + the CPU step.
- No 8k / 2k 10-epoch ETA: a short cool-down, not a training run.

## Files

- `design.md` (this file), `HANDOFF.md`, `sm149b_qat.py`, `sm149b_run.py`, `sm149b_prepare.py`, `sm149b_report.py`,
  `launch-stage1.sh`, `check-cpu-sm149b.json`, `cpu-proof.out`, `gpu/smoke-ticket.sh`, `gpu/smoke_readout.py`,
  `gpu/smoke-sm149b.json` and the smoke logs.
- Prepared (not committed, in the run tree): `wt/sm149-run/artifacts/small-model/sm149b-sm118-tail1`.
