# AUC ceiling (SM182): design

Written Sat 2026-11-28 (approved by the user on 11-28), before any screen run: only the CPU proofs (`check-cpu-sm182.json`) and the report selftest ran first; the proofs' scale measurement (C4) fixes the weight of arm M below. Branch `sm182` (from `origin/sm179`: design, scripts, launcher, hand-off), worktree `C:/Users/jonat/workspace/wt/sm182`. Run tree: `C:/Users/jonat/workspace/wt/sm182-run`, branch `sm182-run` = `cea0f931` (`origin/sm148`, the tree of the state-energy penalty screen (SM171) and of its control) plus one commit, `060878d3`, with the gated ranking term; it is not edited while the screen runs. Track: small model.

## Question

Our models are always trained on LogLoss. **How much discrimination (AUC) is left on the table?** Train for ranking directly, even at the cost of calibration, to find the ceiling of a short cool-down (arm R), and test whether a mix of LogLoss and ranking improves both (arm M).

| plain name (ID) | what it is here |
|---|---|
| six-layer Tiny two-head (SM130) | the model of the screen: `w64-c1-d2p1u1`, release folder `wt/sm130/artifacts/small-model/sm130-th-dec2ep32k-p2` (`master.pth` sha256 48a5e724...) |
| the state-energy penalty screen (SM171) | its control arm `sm171-tiny-ctl` (lambda 0 = the stock trainer, proven bitwise, P4 of SM171) is this screen's control: the same 944 users, 67,310,673 reviews, learning rate, seed, start and options; not retrained |
| the session inputs screen (SM179) | the recipe and launcher pattern this screen copies (arms = SM171's control plus one option) |
| rolling recalibration on the 732 set (SM125) | the causal per-user Platt scheme used in the read-out |

**Why the control can be reused.** The run tree differs from SM171's run tree only by the gated term; with the option absent the trainer is bitwise the stock trainer (proof C1 below), and SM171's control was proven bitwise equal to the stock trainer. So the only difference between an arm and the control is the loss.

## The ranking term

A training batch holds one user's window (up to 32,768 reviews, strided-lane carry), so every pair inside a batch is a pair of reviews of the same user. For each head, take the labelled rows of the batch: the update rows for the curve head (the curve logit at the review's elapsed time, `curve_logits`) and the query rows for the instant head (its pass logit, logit(1 - P(Again)) = logsumexp(Hard, Good, Easy logits) - Again logit). Labels are the hard recall labels (never the distillation targets). With s the head's logit, the logistic pairwise loss of a (passed review i, failed review j) pair is

softplus(s_j - s_i) = -log sigmoid(s_i - s_j),

the RankNet loss; its mean over all pairs of a user is a smooth surrogate of 1 - (that user's AUC), so it targets within-user AUC, and it is unchanged when every score of the user moves by the same constant (it cannot see calibration).

**Pairs.** All pairs of a 32k window would be up to about 10^8, so each head takes 65,536 pairs per batch: pair k is the passed row at quantile a_k of the passed rows and the failed row at quantile b_k of the failed rows (both in batch order), with (a_k, b_k) = ((k + 1/2) (sqrt 5 - 1) / 2, (k + 1/2) (sqrt 2 - 1)) mod 1, an equidistributed two-dimensional Kronecker sequence. No random state (resume-safe and deterministic), fixed shapes only (no host sync, as the trainer's sync-free loss requires); the term is 0 when a class is empty. Every passed / failed row gets its fair share of pairs (within 25%, proof C2); the sampled mean is within 1% of the exact all-pairs mean (C2).

**Loss.** loss = s x (the stock loss) + w x (P_curve + P_instant), with the track option `"pair_rank": {"weight": w, "logloss": s, "pairs": 65536}`. The stock loss has one curve unit (0.5 curve + 0.5 raw curve BCE) and one instant unit (rating cross-entropy), so the pair term also has one unit per head. Option absent (weight 0): the term is never computed and the stock loss is untouched (bitwise, C1).

**Placement.** `expertium/model/variant_model.py`: `pair_logistic`, `pair_rank_terms`, `apply_pair_rank` and a `torch.jit.ignore` method called from the scripted loss only when the weight is above 0; `scripts/gpu_r01_training.py` applies the option; the prefetch allow-list accepts it; it is teacher-blind (the replay KD targets' digest is unchanged).

## Arms (pre-registered)

| arm | folder | option | what it measures |
|---|---|---|---|
| R, pure ranking | `sm182-tiny-r` | weight 1.0, logloss 0.0 | the ceiling: no LogLoss term at all (also no KD, no curve regularizers); calibration is free to drift |
| M, mixed | `sm182-tiny-m` | weight 0.91 (LAMBDA_M), logloss 1.0 | the stock loss plus the pair term at the same starting value |
| RA, fallback of R only | `sm182-tiny-ra` | weight 1.0, logloss 0.02 | trained only if R stops on a non-finite loss; then it replaces R in the read-out (a 2% LogLoss anchor for stability, reported as such) |

**R needs no LogLoss anchor a priori.** Muon fixes the size of each matrix update (the learning rate, peak 7e-5, 0.1 x the release cool-down's peak), so the pure pair gradient cannot make the weights run away within about 2,550 updates; the CPU proof C3 runs 6 real updates of R with optimizer steps, all finite. The scorer still computes the stock loss to check finiteness: if R's probabilities saturate (a non-finite row), those users are left out and counted (coverage rule below). The fallback RA is pre-registered in case the training itself stops.

**Lambda of M (fixed by C4 on 11-28, before any GPU run): LAMBDA_M = 0.91.** At the release weights (no step), on the first 2,048-review window of each of the first 28 users of the training subset (57,344 reviews, CPU, without the teacher: the pair term reads hard labels only), the pair term is P0 = 0.8378 per window, review-weighted (curve 0.4715, instant 0.3663). The stock loss scale is L0 = 0.7582: the review-weighted stock training loss (with the replay distillation targets) that the state-energy penalty screen's (SM171) GPU probe (ticket 001614, `wt/sm171/artifacts/small-model/sm171/gpu/probe-summary.json`) logged over its first 81 updates from the same release weights on the same 28 users at the real 32,768-review windows. So **LAMBDA_M = L0 / P0 = 0.905, rounded to two digits = 0.91**: the ranking term starts at the same size as the LogLoss term (0.91 x 0.8378 = 0.762 vs 0.758). Why 2,048-review windows: 8,192 and 32,768-review windows went over 6 GB on the CPU; the pair term is a mean over pairs of one user, so it does not grow with the window (it measures that window's AUC), and 2,048-review windows stay far under the 6 GB limit (about 1 GB). Reported, not used for lambda: the pair term's gradient norm at weight 1 is 0.37 (median of the 28 windows) vs 0.082 for the stock loss (median of SM171's probe, 32,768-review windows, so not the same windows), about 4.5 x; Muon fixes the update size, so in M the pair term will dominate the update direction at the start even though the two loss values are equal. If M is not useful, the follow-up weight sweep (below) also tries a weight about 4.5 x lower (gradient-matched, about 0.2).

## Proofs (CPU, before the screen)

`sm182_check.py cpu` -> `check-cpu-sm182.json` (stock tree `wt/sm171-run` at `cea0f931`, run tree `060878d3`, both clean).

- **C1** option off = stock, bitwise: 3 real updates on each of users 4745 and 2308 (release weights and Muon state, release options with 256-review windows so that windows are carried): loss, model input and every gradient equal bit for bit between the two trees.
- **C2** the pair term on 12 synthetic batches: pairs are (passed, failed) rows inside the mask and equal a numpy re-implementation of the sampler; fair share within 25%; sampled mean within 1% of the exact all-pairs mean; shift invariance; an empty class gives exactly 0 with zero gradients; gradient signs (<= 0 on passed rows, >= 0 on failed rows, 0 outside the mask); the instant pass logit equals logit(1 - P(Again)); deterministic.
- **C3** the real trainer path (JIT model, carry, replay KD, Muon) with arms R and M, 3 updates with optimizer steps per user: everything finite; at the first update the model input equals stock, R's loss equals P_curve + P_instant, M's loss equals the stock loss + LAMBDA_M x (P_curve + P_instant) (1e-6 relative).
- **C4** the scale measurement above (2,048-review windows, 28 users; ALL OK on 11-28, `check-cpu-sm182.json`).
- **Report selftest** (`sm182_report.py selftest`): AUC with ties equals sklearn's; the histogram AUC equals the exact one; Platt recovers a known miscalibration; Platt is causal (flipping labels from a day on leaves every earlier recalibrated value bitwise unchanged).
- **GPU smoke** (the launcher's first ticket): trainer smoke of R and M (2 users x 512 reviews), then the control's REF with the per-review dump on the 3 smoke users; `sm182_report.py dumpcheck` must reproduce the scorer's per-user LogLoss, AUC and RMSE(bins) from the dump (2e-6) and the rows must equal SM171's rows (1e-6).

## Recipe

Both arms: SM171's control cool-down with only the loss changed. Start = the release model's final `master.pth` and `optimizer.pth` (end of its own cool-down, Muon state), used in place; the release folder's options, loss and targets unchanged (show-time inputs, strided-lane carry, 32,768-review windows, replay KD targets of the show-time Big model (SM91s2) at alpha 0.5, deleted cards neighbour) plus `pair_rank`; no `state_vq` option. Data / schedule: SM171's 944 users, 67,310,673 reviews, 0 -> 7e-5 over 64 updates then one_minus_sin to 0, about 2,550 updates, snapshots every 100 users, training seed 2025. `sm182_prepare.py pair` proves the rows, schedule, start, protocol and options equal the control's (options minus `state_vq`, plus `pair_rank`).

## Scoring

The run tree's scorer (`scripts/small_model_heldout.py`), one ticket per model: `--pool 732 --users-file <sm72s eval-users-732-fit.json> --inputs e2s-query --split-existing --deleted-cards neighbour --nonfinite-rows --prior-buckets 0,1,3,5,9,16,32,128,300,1000 --dump <absolute work/dump>`, `PYTORCH_JIT=0`, label REF `w8+s8hx8` (int8 weights, int8 card / note state, int8 shift vectors; the deployed format, as SM171 / SM179), show-time inputs, curve and instant. GPU cap 0.8 (0.92 for a retry pass, memory only). Models: the control (rescored with the dump; integrity: its rows must equal SM171's rows within 1e-6), R (or RA), M. The per-review dump (`review_th`, bin, y, p_ahead, p_instant per benchmark review) lives under `artifacts/small-model/sm182/work/` and is never committed.

## Metrics (per head, the rule users, `sm182_report.py`)

Paired user bootstrap against the control, 2,000 draws, numpy `default_rng(0)`, 95% CI, the same draws for every statistic. Rule users: a finite row for all three models, equal benchmark sizes, and a dump of that size.

- **within-user AUC**: the mean of the per-user AUCs (ties averaged) over the users with both outcomes; the change = the mean of per-user differences (AUC units, and in % of the control's mean).
- **overall AUC**: all rule users' reviews pooled (exact point estimate; the bootstrap uses per-user histograms of logit(p), 16,384 bins on [-16, 16]).
- **LogLoss**: % = 100 x (sum of LogLoss x reviews of the arm / the same of the control - 1).
- **calibration**: RMSE(bins) per user (the benchmark's bins, as in earlier results), review-weighted, change in %; the pooled ECE (20 equal-mass bins) reported.
- **Platt recalibration (to separate ranking from calibration)**: per user and head, causal rolling Platt as SM125: rows in review order cut into 7-day blocks from the user's first scored review; each block mapped through sigmoid(a logit(p) + b) fitted on the latest 30,000 scored rows of strictly earlier days; raw until 1,000 earlier rows or with fewer than 20 rows of a class; a in [0.05, 20], b in [-15, 15]. For every model: LogLoss, RMSE(bins) and within-user AUC after recalibration; for R (and M), LogLoss after recalibration vs the control raw and vs the control recalibrated, and the **share of R's LogLoss gap that recalibration recovers** = (R raw - R recalibrated) / (R raw - control raw).

## Rules (pre-registered)

- **R: no go / no-go.** It is a ceiling measurement; all numbers above are reported.
- **M is "useful"** if on at least one head the within-user AUC change has its CI lower bound above 0, AND on both heads the LogLoss change has its CI upper bound at most +0.05% (LogLoss not worse by more than 0.05%). Otherwise "not useful". The per-head conditions are reported.
- **Pending** (exit 2, the main session decides): fewer than 720 rule users, a non-finite row of any model (users left out and counted), the integrity check failing, or the dump not reproducing the scorer's rows.
- **Reported, no verdict:** training reviews/s, minutes and GPU memory vs the control; the Platt fit statistics (share of rows recalibrated, clamped blocks).

## Follow-up (not part of this screen)

If M is useful: the same mix in a full Tiny cool-down (all 8,000 users) and on Small, with the weight re-checked. If R's within-user AUC gain is large but M is not useful: a weight sweep of M (lower and higher weights) before giving up on a mix. If R's gain is small: LogLoss training already sits near the short cool-down's ranking ceiling, and ranking losses are not worth pursuing for this model family.

## Estimates and budget

| step | GPU minutes |
|---|---|
| smoke (trainer R, M; control REF with dump on 3 users) | about 3 |
| `sm182-tiny-r`, `sm182-tiny-m` (67.3M reviews at 55k to 63k reviews/s, the pair term is two sorts and 2 x 65,536 gathers per update; the control took 20.1 min) | about 22 each |
| scoring 3 models x REF with the dump on the 732 set (SM179: about 10 min each) | about 12 each, about 36 |
| **total** | **about 1.4 h** (budget 3 GPU hours; RA, if needed, adds about 22 min) |

Memory: cap 0.8 of 16.3 GB for every ticket (the control peaked at 4.2 GB allocated). CPU: the read-out (per-user Platt on 3 models x 2 heads, about 32M reviews each) takes about 20 to 40 minutes at under 4 GB; the CPU proofs stay under 3 GB (C4 uses 2,048-review windows: 8,192 and 32,768 went over 6 GB on the CPU). No 8k / 2k 10-epoch ETA: this is a short cool-down, not a training run.
