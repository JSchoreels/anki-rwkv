# Answer-time spread shrink (SM153): design

Pre-registered Mon 11-23 before any shrunk prediction of this experiment was scored. The only numbers known before this file are the ones quoted under "What is known" (from answer-time expressiveness (SM142) and the rolling recalibration line (SM125, SM128, SM140)). CPU only, no GPU, no training: it rescores the saved per-review time distributions of answer-time expressiveness (SM142).

## Question

Idea 7 of the 11-23 opportunities note (`wt/ideas-1123/artifacts/ideas-1123/OPPORTUNITIES.md`): answer-time expressiveness (SM142) found that the answer-time heads rank one user's cards well (within-user, within-grade AUC about 0.83 at 2 s against 0.69 for the card's own running median) but spread the cards about 1.7 times too widely (out-of-sample between-card slope 0.59, the review-level slope 0.93). Does shrinking each card's predicted time distribution toward the user's running per-grade distribution lower the time LogLoss, while within-user AUC and the card-level expressiveness stay intact?

**Answer time** is the whole time on the card, front and back together, from the moment the question is shown to the moment a grade button is pressed, capped at the preset's maximum answer seconds. "Answered within S s" always means answered within S s, reading the back included.

## What is known (before this file)

- The review-level slope within a (user, grade) is 0.93 for every model of answer-time expressiveness (SM142), and the between-card slope (predictions on a card's first half, actual times on its second half) is 0.59. Per-review time LogLoss is scored at review level, so the prior expectation is that a location shrink helps less than "1.7 times too wide" suggests: a fitted factor near 0.9 is more likely than 0.6.
- The rolling recalibration (SM125 for the Small models, SM128 for the Big models, SM140 for the six-layer Tiny) applies per-user Platt scaling to the 33 timed and grade-only event probabilities, not to the time distribution; it already makes all 21 timed cells of 2 to 10 s reliable and improves "Good or Easy within 4 s" by -0.68% to -1.24% LogLoss against raw.

## Rows and models

The saved dumps of answer-time expressiveness (SM142), `wt/sm142/artifacts/small-model/sm142/work/dump-<key>/<user>.npz` (branch `sm142`, commit `c8764a2e`): every time-labelled test review of the time heads for the family (SM117)'s scorer (duration above 0, not a first review, same-day reviews included), show-time inputs, leak-free, fp32, with the head's full 4 x 16 distribution, P(grade), card id, grade and duration. 722 users (the show-time instant-only big model (SM99): 719).

| key | model | time head |
|---|---|---|
| sm99 | the show-time instant-only big model (SM99), Big | the time heads for the family (SM117) |
| sm113 | the longer Small Instant-Only model (SM113), Small | the time heads for the new candidates (SM119) |
| sm118 | the show-time two-head cool-down for Small (SM118), Small | the time heads for the new candidates (SM119) |
| sm130th | the six-layer Tiny two-head model (SM130), Tiny | the answer-time heads for the six-layer Tiny (SM138) |
| sm130io | the six-layer Tiny Instant-Only model (SM130), Tiny | the answer-time heads for the six-layer Tiny (SM138) |

Time LogLoss is SM117's: the censored 16-bin LogLoss at the true grade (`sm117_timelib.censored_ll`, floor 1e-7), review-weighted. The raw arm must reproduce each user's recorded sum in the SM142 dump index to 1e-6 relative (check C1).

## The user's running per-grade distribution (causal)

For review i of a user, from the user's whole revlog strictly before i (revlog order): the counts n_{g,k} of earlier time-labelled reviews (SM117's label rule: duration above 0, not a first review; a censored review counts in its lower bin) of grade g in bin k, and n_k over all grades.

- all grades: h_k = (n_k + 1) / (n + 16)
- grade g: p_user,g,k = (n_{g,k} + 4 h_k) / (n_g + 4)
- the user's centre per grade: c_g = the median of p_user,g in ln seconds (`median_log_s`, the binned median rule).

Nothing of review i or later enters; a deployed client can keep these 64 counters per user.

## Shrink family (per grade g, applied to all four grade distributions)

- **Location shift**: shift_beta(p_card,g) moves the card's distribution in ln time by delta = (beta - 1)(m_g - c_g), where m_g is the head's predicted median (ln s) for grade g. The distribution is treated as piecewise uniform in ln time per bin (bin 0 from 0.25 s, bin 15 spread over [60 s, 60 s x e^0.342] for the shift only); mass shifted below the first edge stays in bin 0 and mass shifted above 60 s goes to bin 15 (the cap). beta = 1 is the raw head; beta < 1 pulls every card toward the user's centre by the factor beta.
- **Mixture**: p' = alpha x shift_beta(p_card,g) + (1 - alpha) x p_user,g.

Arms (each fitted per model):

- **primary, combined**: (alpha, beta) jointly on the grid alpha in {0.50, 0.55, ..., 0.85, 0.90, 0.91, ..., 1.00}, beta in {0.40, 0.45, ..., 1.10}.
- secondary, mixture only: beta = 1, alpha on the grid.
- secondary, shift only: alpha = 1, beta on the grid.
- reference: the user's distribution alone (alpha = 0), not fitted.

## No leakage: split of users

The sorted user list of the 722 users is permuted with `numpy.random.default_rng(153)`; the first 361 are the **fit users**, the other 361 the **held-out users** (same split for every model; the show-time instant-only big model (SM99) loses the 3 users it lacks wherever they fall). The grid point with the lowest pooled review-weighted time LogLoss on the fit users is chosen; every reported number is on the held-out users only. The fitted values are reported.

## Go rule (pre-registered; per model, primary arm)

On the held-out users, the change of time LogLoss (shrunk minus raw), review-weighted, with the paired user-bootstrap 95% CI (2,000 resamples of the held-out users, seed 0, `sm117_head.bootstrap_idx`): **go** iff the CI upper bound is below 0 **and** the gain is at least 0.05% of the raw time LogLoss. The secondary arms get the same read-out, not a verdict.

**Expressiveness guard (a warning, not part of the go rule):** on the held-out users, with SM142's own per-user analysis (`sm142_expr.analyse_user`, imported, not edited) run on the shrunk distributions and on the raw ones: within-user, within-grade AUC at 2 / 4 / 8 s (paired difference with its CI), the review-level slope, the between-card slope, the spread ratio, the persistent card differences explained, the per-user number of distinct curves (location included and shape only, 10% grid) and the per-user SD of the predicted log median, the pooled PCA k95 of the shapes, and the card-level shape calibration. A shrink that lowers the within-user AUC at 2, 4 or 8 s by more than 0.005 with the CI below 0, or more than halves the shape-only distinct curves, is flagged as "kills expressiveness" even if it is go on LogLoss. The between-card slope should move toward 1.

## After the existing rolling recalibration

The rolling recalibration works on the timed events, so "after" means: the 21 timed cells of 2 to 10 s (Good or Easy, recalled, Easy within S s, S in {2, 3, 4, 5, 6, 8, 10}) recomputed from the shrunk distributions (SM120's event formulas, `sm125_recal.event_matrix`), then recalibrated with exactly the existing scheme (`sm125_recal.recal_user`, imported, not edited: per user, Platt per event refitted every 7 days on the latest 30,000 earlier reviews of earlier days, raw before 1,000 earlier reviews, repair). Compared with the existing recalibrated arm on the same held-out users (recomputed with the same code from the raw distributions, which must match the recorded per-user sums of SM125 / SM128 / SM140 to 1e-5 relative: check C2), per cell: the LogLoss change with its CI, and in total over the 21 cells (sum of cell LogLosses). Also the 21 cells before recalibration (raw events from shrunk vs raw distributions). Reported, not a gate: if the shrink helps before recalibration but not after, the Platt step already absorbs it.

Cost: the recalibration is about 7,200 rows per second per process (SM125), about 35 min per model per arm on the held-out half. If the whole recalibration part needs more than 1 h of CPU, it runs detached with a HANDOFF.md and the agent stops.

## Checks

- C1: raw arm = the dump index sums (time LogLoss per user, 1e-6 relative), for every user.
- C2: the raw event matrix from the SM142 distributions equals the SM125 / SM128 / SM140 dumped event probabilities (same rows, grade and duration equal, events within 1e-5 absolute), and the recalibrated raw arm reproduces their recorded per-user sums.
- C3: the shift with beta = 1 and the mixture with alpha = 1 return the raw distribution exactly; every shrunk distribution sums to 1 within 1e-9 and is non-negative.
- C4: causality: the user's distribution at review i is unchanged when every review from i on is altered (one user).

## Budget

CPU only, at most 2 processes at a time, each under 6 GB, 1 thread each; anything over about 10 min detached via WMI (`ShowWindow=0`).
