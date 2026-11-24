# Answer-time spread shrink (SM153): result

Run Mon 11-23 23:30 to Tue 11-24 01:40, CPU only, no GPU, one thread per process. The design in `design.md` was pre-registered before any shrunk score and no rule was changed after seeing results. Numbers come from `sm153-result.json` (written by `report`) and the per-user rows in `rows/`. Every number is on the 361 held-out users (the show-time instant-only big model (SM99): 360), never on the 361 fit users the shrink was fitted on.

## Question and rule

Answer-time expressiveness (SM142) found that the answer-time heads rank one user's cards well but spread the cards about 1.7 times too widely (out-of-sample between-card slope 0.59). The question: does shrinking each card's predicted time distribution toward the user's running per-grade distribution lower the answer-time LogLoss while the within-user AUC and the card-level expressiveness stay intact?

The primary arm is the combined shrink (a location shift by the factor beta plus a mixture with weight alpha on the card's distribution and 1 - alpha on the user's running distribution), fitted per model on the fit users. The rule, per model: go if and only if the 95% paired user-bootstrap CI of the change in answer-time LogLoss on held-out users has an upper bound below 0 and the gain is at least 0.05% of the raw answer-time LogLoss. The expressiveness guard is a warning only (flag "kills expressiveness" if the within-user AUC at 2, 4 or 8 s falls by more than 0.005 with the CI below 0, or the shape-only distinct curves are more than halved).

## Verdict per model

"After recalibration" means the 21 timed cells of 2 to 10 s, recomputed from the shrunk distributions and passed through the existing rolling per-user Platt recalibration (SM125 for the Small models, SM128 for the Big model, SM140 for the six-layer Tiny models), compared with the existing recalibrated arm on the same users. The design says the after-recalibration read-out is reported and not a gate. Because the shrink would change the cells the fork actually shows, I also state a go or no go for it with the same style of rule (a go would need the 21-cell total to improve, with its CI below 0 and by at least 0.05%).

| Model | Fitted alpha, beta | Before recalibration (answer-time LogLoss, primary arm) | Verdict before | After recalibration (21 timed cells, total) | Verdict after |
|---|---|---|---|---|---|
| the show-time instant-only big model (SM99) | 0.94, 1.05 | -0.015% (CI -0.103% to +0.076%) | no go | +0.039% (CI +0.020% to +0.058%), worse | no go |
| the longer Small Instant-Only model (SM113) | 0.93, 1.00 | -0.052% (CI -0.112% to +0.010%) | no go | +0.079% (CI +0.059% to +0.102%), worse | no go |
| the show-time two-head cool-down for Small (SM118) | 0.92, 1.00 | -0.073% (CI -0.156% to +0.007%) | no go | +0.075% (CI +0.055% to +0.099%), worse | no go |
| the six-layer Tiny two-head model (SM130) | 0.90, 1.00 | -0.084% (CI -0.192% to +0.017%) | no go | +0.097% (CI +0.067% to +0.129%), worse | no go |
| the six-layer Tiny Instant-Only model (SM130) | 0.91, 1.00 | -0.083% (CI -0.178% to +0.006%) | no go | +0.073% (CI +0.048% to +0.101%), worse | no go |

All five models are no go on the pre-registered rule: in every case the CI upper bound is above 0. Only the secondary mixture-only arm of the show-time instant-only big model (SM99) passes the numeric rule (alpha 0.95, beta 1.0: -0.066%, CI -0.117% to -0.017%). Secondary arms get no verdict, and the other models do not show this.

The fitted location factor beta is 1.00 for four models and 1.05 for the show-time instant-only big model (SM99). The shift-only arm (alpha 1) is fitted at beta 1.00 everywhere, so a pure location shrink toward the user's centre gains exactly nothing: the location spread of the cards is not what costs LogLoss at review level (the review-level slope is already 0.93). All of the small gain comes from the mixture, which at weight 0.90 to 0.95 on the card means blending in 5% to 10% of the user's own running distribution. That acts as tail widening (a defensive mixture), not as a correction of the between-card spread. The user's distribution alone is far worse (+13.3% to +15.9% answer-time LogLoss), as expected.

## Answer-time LogLoss change (held-out users, shrunk minus raw, combined arm)

| Model | Raw answer-time LogLoss | Change | Relative change, 95% CI | Users better |
|---|---|---|---|---|
| the show-time instant-only big model (SM99) | 1.9261 | -0.00029 | -0.015% (-0.103% to +0.076%) | 181 of 360 |
| the longer Small Instant-Only model (SM113) | 1.9368 | -0.00101 | -0.052% (-0.112% to +0.010%) | 195 of 361 |
| the show-time two-head cool-down for Small (SM118) | 1.9471 | -0.00142 | -0.073% (-0.156% to +0.007%) | 191 of 361 |
| the six-layer Tiny two-head model (SM130) | 1.9701 | -0.00166 | -0.084% (-0.192% to +0.017%) | 186 of 361 |
| the six-layer Tiny Instant-Only model (SM130) | 1.9643 | -0.00163 | -0.083% (-0.178% to +0.006%) | 194 of 361 |

The gains are smaller than on the fit users (-0.12% to -0.30%) and are indistinguishable from 0 on held-out users for every model.

## After the existing rolling recalibration (21 timed cells of 2 to 10 s)

Before recalibration, the shrink makes the 21 timed cells worse in total for every model (+0.12% for the show-time instant-only big model (SM99), +0.33% to +0.52% for the others, all CIs above 0), even though answer-time LogLoss improves slightly. The shrink is fitted for the 16-bin time LogLoss, not for these cells. After recalibration the damage shrinks to +0.04% to +0.10% but stays significantly above 0: the recalibration absorbs most of the harm but not all, and no cell improves in total (0 better cells, 15 to 19 worse cells of 21 per model). The recalibration alone, without any shrink, improves the cells much more (for example "Good or Easy within 4 s": -0.65% to -1.10% LogLoss against raw), so the shrink adds nothing on top of it.

## Expressiveness (held-out users, raw to combined shrink)

| Model | AUC at 2 s | AUC at 4 s | AUC at 8 s | Review-level slope | Between-card slope | Spread ratio | Distinct curves (location included) | Distinct shapes (shape only) | SD of predicted log median |
|---|---|---|---|---|---|---|---|---|---|
| the show-time instant-only big model (SM99) | 0.838 to 0.840 | 0.827 to 0.827 | 0.810 to 0.810 | 0.924 to 0.914 | 0.583 to 0.576 | 1.155 to 1.167 | 813 to 793 | 82.6 to 87.1 | 0.578 to 0.583 |
| the longer Small Instant-Only model (SM113) | 0.841 to 0.840 | 0.826 to 0.826 | 0.809 to 0.809 | 0.929 to 0.961 | 0.587 to 0.607 | 1.142 to 1.103 | 711 to 688 | 75.9 to 81.0 | 0.577 to 0.560 |
| the show-time two-head cool-down for Small (SM118) | 0.839 to 0.839 | 0.823 to 0.823 | 0.807 to 0.807 | 0.926 to 0.963 | 0.585 to 0.608 | 1.138 to 1.093 | 684 to 659 | 74.1 to 78.1 | 0.567 to 0.548 |
| the six-layer Tiny two-head model (SM130) | 0.831 to 0.830 | 0.817 to 0.817 | 0.801 to 0.801 | 0.929 to 0.977 | 0.587 to 0.618 | 1.111 to 1.053 | 725 to 683 | 79.8 to 83.3 | 0.566 to 0.541 |
| the six-layer Tiny Instant-Only model (SM130) | 0.831 to 0.831 | 0.819 to 0.819 | 0.802 to 0.802 | 0.928 to 0.971 | 0.588 to 0.614 | 1.119 to 1.067 | 715 to 686 | 75.5 to 78.6 | 0.567 to 0.542 |

Within-user, within-grade AUC differences (shrunk minus raw, paired, with CI): the show-time instant-only big model (SM99) improves at 2 s by +0.0024 (CI +0.0016 to +0.0031) and at 4 s by +0.0006 (CI +0.0004 to +0.0008). The other four models change by -0.0001 to -0.0004 at 2 s (CIs include 0) and by -0.0001 to -0.0002 at 4 s and 8 s, where the CI upper bound is about 0 or below (the six-layer Tiny two-head model (SM130) at 8 s: -0.0002, CI -0.0003 to -0.0001; the six-layer Tiny Instant-Only model (SM130) at 8 s: -0.0002, CI -0.0002 to -0.0001). All are far inside the 0.005 flag threshold.

Expressiveness flag: none of the five models is flagged "kills expressiveness". The shape-only distinct curves rise slightly rather than falling, and the pooled PCA of the shapes keeps k95 at 7 to 8. The between-card slope moves only from about 0.59 to 0.61 or 0.62 on the Small and Tiny models (and does not move toward 1 on the Big model, 0.583 to 0.576), far from 1, so the shrink does not fix the 1.7-times spread. The rise of the review-level slope (0.93 to 0.96 or 0.98) mostly reflects the mixture adding the user's own distribution.

Warning, stated plainly: on the four Small and Tiny models the shrink lowers LogLoss slightly while reducing card-level expressiveness slightly. The number of distinct location-included curves per user falls by 3% to 6% (for example 725 to 683 for the six-layer Tiny two-head model (SM130)), the SD of the predicted log median falls by 3% to 4.5% (0.566 to 0.541 for the same model), and the within-user AUC at 8 s falls by 0.0001 to 0.0002 with the CI below 0. These changes are small and the labels stay "partly expressive", but they are real and in the opposite direction from the goal, bought for a LogLoss gain that is not significant. The show-time instant-only big model (SM99) is the exception: its AUC improves a little and its distinct curves fall by 2.5% only.

## CPU minutes (single thread each, two processes at a time)

| Step | CPU minutes |
|---|---|
| Grid pass on all 722 users, five models | 43.2 |
| Apply on held-out users (three SM142 analyses per user), five models | 51.2 |
| Recalibration, combined arm, five models | 121.7 |
| Raw recalibration check (3 users) | 0.3 |
| Total | about 216 (3.6 CPU hours); wall time 23:30 to 01:40 (2 h 10 min) |

The fit and report steps took seconds.

## Checks

- C1 (raw arm equals the SM142 dump sums, 1e-6 relative): passed for every user and model (`c1_max_rel` 0.0 in the fit file; `check-cpu.json` also passed).
- C2 (events and recalibration baseline): passed. The raw events rebuilt from the SM142 distributions equal the SM125, SM128 and SM140 dumps on the same rows for all five models (`c2_same_rows` true, largest absolute difference 3.0e-08). The raw-arm recalibration on 3 users (52, 65, 74) of the longer Small Instant-Only model (SM113) reproduces the recorded SM125 per-user sums of all 21 timed cells exactly (largest relative difference 0.0, raw and recalibrated). The three "grade" cells of the recorded file are not computed by this lean run, by design.
- C3 and C4 (shift and mixture identities, sums to 1, causality): passed (`check-cpu.json`; largest sum error 2.5e-07 on the held-out rows).

## What this means for the fork's answer-time estimates in Card Info

Nothing here supports changing the answer-time distribution the fork shows. The raw heads are already within 0.1% of the best shrink on answer-time LogLoss; the only fitted component that helps is a 5% to 10% blend with the user's own running per-grade distribution (the client would need 64 counters per user), and its gain is not significant on held-out users. For the timed cells (the "answered within S s" numbers a user sees), the shrink is worse than doing nothing even after the rolling recalibration, which already delivers the real improvement. A shrunk Card Info time would also tell cards apart slightly less well on the Small and Tiny models, which is the property the answer-time heads were judged on. The wide spread of predicted card medians (between-card slope 0.59) is therefore not something post-hoc shrinking fixes; it would have to be addressed in training, if at all. Whether to ship anything is the user's decision; this result gives no reason to ship.
