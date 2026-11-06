# Fast-recall prediction (SM94): design

Question (user 11-07 ~00:40): keep the time-aware target (from the small model in the review simulator (SM41)) out of Anki for now, but test whether answer speed helps prediction: (a) can we predict the chance of a good answer before N seconds ("fast pass"), and is a directly trained target better than one composed from the heads we already ship; (b) does the speed signal help predict plain recall (R) better than R alone.

Branch `sm94` (from `sm92`), worktree `C:/Users/jonat/workspace/wt/sm94`. Code `scripts/sm94_fastpass.py`, launcher `launch-sm94.sh`, follow-up `HANDOFF.md`.

## What exists

- The released small model (the release from epoch 25 (SM63), the Anki default) and its shipped answer-time head (the answer-time head release (SM83) = the answer-time head on richer frozen features (SM82c) on the frozen release): per review, the instant head gives P(grade) and the answer-time head gives P(time bin | grade) over 16 log-time bins (bin 0 under 0.5 s, bins 1-14 between edges 500 ms x 120^(k/14), bin 15 = 60 s or more; per-user caps are censored).
- So the composed fast-pass probability is available today: C(N) = sum over Hard / Good / Easy of P(grade) x P(time < N | grade).
- No per-review dumps of these predictions exist on the 732 set: the SM82c / SM92 time scorers keep per-user sums only, and the stage 2 curve queries of response-time prediction (SM82) hold 150 users at one threshold (3.9 s). FSRS7-Instant per-review predictions are not stored either (`srs-benchmark/fsrs_instant.py` writes per-user stats).
- Priors: the real-data check of the time-aware target (SM41r) found the instant head already calibrated in answer speed (recalibration slope -0.009 +- 0.015 per unit of ln(last later-day time / user median), 117 users); response-time prediction (SM82) showed that a jointly trained time loss (weight 0.1, quarter-epoch cool-down) costs grade accuracy (+0.12 curve / +0.27 instant points); the frozen head reaches time LogLoss 1.957 vs 1.930 for the jointly trained one.

## Stage 1 (cheap: one ~15-25 min GPU scoring ticket, then ~10-20 min CPU)

The scorer dumps need per-review grade and time predictions, so stage 1 is one GPU scoring pass, not CPU only.

### Data

- Model: the release (SM63) + its shipped head (SM83), `C:/Users/jonat/workspace/wt/sm82c/artifacts/small-model/sm82c-head-released/master.pth` (read only), fp32, leak-free normal inputs (deleted cards "neighbour"), exactly SM82c's time-scorer path (imported unchanged from the sm82c worktree).
- Users: the 732 set minus the 9 never-fit users (723, `eval-users-732-fit.json`); the readout reports 722-723 users.
- Reviews: the time-scored test reviews (a later day than the card's previous review and a recorded time: 21.78M reviews, about 72% of the instant labels). Same-day reviews are left out because their interval input contains the answer time itself (SM82 stage 0 leak audit).
- Per review dumped: P(grade) (4), P(time bin | grade) (4 x 16, f16), the head's 26 explicit causal time features, grade, answer ms, lower bin, censored flag, user cap. One npz per user (~5-6 GB in all, not committed).
- Identity check: per-user censored time LogLoss vs SM82c's recorded rows: same review sets, relative gap <= 1e-5 (CPU smoke on 3 users: <= 4e-8).

### Events (fixed)

- Fast pass under N = grade Hard / Good / Easy and answer time < N, at the bin edges N = 3.9 / 5.5 / 10.9 / 21.5 s (edges 6, 7, 9, 11: no interpolation inside a bin; "5 / 10 / 20 s" in practice).
- User-relative: N_u = the bin edge nearest in ln time to the user's median uncensored pass answer time (an event definition, a constant per user; no predictor reads outcomes through it).
- A user enters a threshold only if their time cap is >= N (otherwise "under N" is unknown at the cap). Censored reviews are not fast.

### Predictors of the fast-pass event (all on the same reviews)

- C, composed: sum over pass grades of P(grade) x P(time < N | grade).
- B0: the user's causal running fast-pass rate (earlier scored reviews only, prior: the pooled rate with 10 pseudo-reviews).
- B1: the model's P(pass) x the user's causal running P(fast | pass) (same prior). This is the in-dump version of "R x an empirical per-user speed rate".
- S0: logistic recalibration of logit C (Platt), cross-fitted: users split in two halves (seed 0), fit on one half (3,000 random reviews per user), scored on the other.
- S: cross-fitted logistic stack of logit C, logit P(pass), logit P(fast | pass), and the 26 explicit time features: a cheap proxy for "a directly trained target" (it can use what the composition misses, but not the model's internal states).
- B2 (stage 1c, CPU, pending): FSRS7-Instant R x the user's running P(fast | pass). Needs per-review FSRS7-Instant predictions, which do not exist: a wrapper around `srs-benchmark/fsrs_instant.py` (read only) that saves them, on the intersection with the small model's scored reviews. Cost measured on 5 users before any full run. Reported only, not a gate.

### Recall question (b), stage 1

- On the same later-day reviews: R = the instant head's P(pass); Rb0 = cross-fitted recalibration of logit R; Rb = cross-fitted stack of logit R, the 26 explicit time features and 3 interactions (the card's last 3 ln times x logit R). The speed signal's value beyond recalibration is Rb - Rb0.
- Only the instant head (the dump holds instant predictions only); the curve head enters in stage 2b.

### Metrics

- Binary LogLoss, review-weighted (main) and user-averaged (reported); paired user bootstrap 2,000, seed 0, 95% percentile intervals.
- Points for (a): % of C's fast-pass LogLoss on the same reviews (no big-model fast-pass LogLoss exists).
- Points for (b): the change of the all-label instant LogLoss if only these later-day rows changed, in % of the big model (1 leak-free epoch) instant LogLoss on the same users (`bigleak-neighbour.jsonl`, label `big-leakfree-1ep/w8+s8`).
- Calibration of C: mean predicted vs observed, ECE over 20 equal-width bins, Platt slope.
- Speed: the dump's scored reviews/s and full-history reviews; the analysis' seconds.

### Decision rules (fixed before the run; minimum effect: 95% CI below zero AND at least 0.05 points)

- A0, sanity: identity check passes (same reviews, gap <= 1e-5) and >= 722 users; otherwise stop and report.
- A1, the composed fast-pass probability is usable: at every threshold C beats B0 and B1 (CI below zero, at least 0.05 points) and is calibrated (|mean predicted - observed| <= 0.005 and ECE <= 0.01). A pass means Anki could show "chance of a good answer within N s" from the shipped files, no training.
- A2, a direct target is worth training (go for stage 2a): S beats C AND S beats S0 (CI below zero, at least 0.05 points each) at the user-relative threshold or at two or more fixed thresholds. If S0 alone carries the gain (S0 beats C by the bar, S - S0 does not), the verdict is "recalibrate C post hoc", no GPU.
- B1-rule, speed helps recall (go for stage 2b): Rb beats Rb0 by at least 0.05 points (big-model scale) with the CI below zero. Expected no-go from the SM41r prior; Rb0 vs R is reported as a calibration note only.
- Limitation stated with the verdict: a no-go on the stacking proxies does not prove a trained head or trunk could not do better; it ranks stage 2 down, it does not close it.

## Stage 2 (proposed only; needs the user's go after stage 1)

- 2a, direct fast-pass target, head only, grade untouched (if A2 passes, or if the user wants it anyway): the SM82c recipe on the frozen release (1,991 training users, first 400k reviews, 10k AdamW steps) with 5 extra binary outputs (the 4 fixed thresholds + user-relative) trained by BCE next to the time loss; then the stage 1 dump and analysis with the new head as a sixth predictor. Cost ~1.2 GPU h (smoke 0.2 h, head training 0.75 h, dump 0.3 h). Rule: beats C by at least 0.05 points with CI below zero at the user-relative threshold or two fixed ones, grade bitwise unchanged (5-user check).
- 2b, speed as an auxiliary target in the trunk (only if the B1-rule passes): screen = the quarter-epoch Muon cool-down from the leak-free small model (SM72s) epoch 21 against the same-recipe reference of the tiny architecture additions run (SM77), as in SM82, with a fast-pass BCE (5.5 s and user-relative) on the instant rows at weights 0.01 and 0.03 (SM82's 0.1 cost +0.12 / +0.27 points). Cost: 2 arms x ~2 h + 732-set grade scoring ~0.4 h per arm = ~4.8 GPU h. Rule: curve or instant LogLoss at least 0.05 points better than the reference with CI below zero and neither head worse by 0.05. A pass opens the release-matched run (the release's 1-epoch 32k cool-down from epoch 25 with the auxiliary loss, ~6 GPU h + 0.5 h scoring) against the release.
- 2c, speed as an extra input: not proposed. The model already sees answer times through its interval inputs and per-review features, and SM41r found it calibrated in speed; revisit only if the B1-rule passes strongly and 2b fails.

## Not in scope

- The time-aware target in Anki (postponed by the user 11-07).
- Show-time inputs (the show-time retrain of the release (SM92) question); this stage scores the Anki default model on normal inputs.
