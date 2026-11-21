# How expressive the answer-time predictions are (SM142): design

Pre-registered Sat 11-21 before any per-review head output of this measurement was computed (the only numbers seen before this file are the earlier audits and gates quoted under "What is known"). This is a description, not a go/no-go gate: nothing is shipped or blocked by it.

## Question

The user asked (11-21): "I'm not interested that much by the average response time at all. I'm interested to see how 'expressive' our predictions are. If it's just 3-4 different kinds of curve that all say 'at 2s it's half 4s' it's not really an exploitable result even if it's kinda precise enough in 'average terms'."

So SM142 asks whether the answer-time head gives card-specific predicted time distributions, in location and in shape, that track the real differences between the cards of one user, or only a few templates around the user's average. It also asks whether the head knows which card is slow better than the card's own history does.

**Answer time** is the whole time on the card, front and back together: from the moment the question is shown to the moment a grade button is pressed, capped at the preset's maximum answer seconds. "Answered within S s" always means answered within S s, reading the back included. The moment the answer was shown is not stored in the data.

## What is known

- The answer-time head is an MLP giving, for each of the 4 grades, a distribution over 16 log-time bins. It is trained alone on the frozen model with show-time inputs (recipe of the answer-time head on richer frozen features (SM82c), the time heads for the family (SM117)).
- The stage-0 audit of the answer-time head (SM82) found that within a user 38% of the log-time variance lies between cards (median user 44%, p10 to p90 25% to 60%), and that the correlation with the card's previous time is 0.22 / 0.36 / 0.54 (per-user p10 / p50 / p90).
- The heads beat the card's last time and the user's running median per grade by 18% to 21% in time LogLoss and by about 28% to 36% in mean absolute log error; the explicit-features-only ablation head is about 4% worse than the head. There is no direct test of card-level expressiveness and no comparison with the card's running median of its own earlier times.

## Existing dumps (checked 11-21) and why a new one is needed

- The scoring rows of the time heads for the family (SM117), the time heads for the new candidates (SM119) and the answer-time heads for the six-layer Tiny (SM138) store per-user sums and calibration histograms only.
- The per-review dumps of rolling recalibration (SM125, SM128, SM140: `work/dump-<model>/<user>.npz`) store 33 event probabilities per review (P(set) and P(set and answered within S s) for recalled, Good or Easy, and Easy at S in {2, 3, 4, 5, 6, 8, 10, 15, 20, 30} s), the grade, the duration and the day. They hold no card id (it could be recovered from the row order), no Again distribution, and no 16-bin distribution, so they cannot give medians, quantiles, widths or shapes.
- So SM142 runs one new GPU dump per model (below). The rows are exactly the rows of the scorer (SM117's `score`), checked against each model's recorded scoring sums.

## Models (5), all show-time inputs, leak-free (deleted cards: neighbour), fp32

| key | model | head |
|---|---|---|
| sm113 | the longer Small Instant-Only model (SM113) | the time heads for the new candidates (SM119), `wt/sm119/artifacts/small-model/sm119-sm113-head` |
| sm118 | the show-time two-head cool-down for Small (SM118) | the time heads for the new candidates (SM119), `wt/sm119/artifacts/small-model/sm119-sm118-head` |
| sm130th | the six-layer Tiny two-head model (SM130) | the answer-time heads for the six-layer Tiny (SM138), `wt/sm138/artifacts/small-model/sm138-sm130th-head` |
| sm130io | the six-layer Tiny Instant-Only model (SM130) | the answer-time heads for the six-layer Tiny (SM138), `wt/sm138/artifacts/small-model/sm138-sm130io-head` |
| sm99 | the show-time instant-only big model (SM99), the big reference | the time heads for the family (SM117), `wt/sm117/artifacts/small-model/sm117-sm99-head` |

Each model also carries its explicit-features-only ablation head (26 history features: the card's last three times, its review count and last grade, the user's running medians per grade, and so on). It is reported as a fourth comparison: a learned model of the card's own history, without the frozen model.

## Dump (GPU, one ticket per model)

`scripts/sm142_expr.py dump`: SM117's `score` forward pass and row selection (imported, not edited) on the 732 set (the 722 rule users of `sm117_head.pool_users()`, all with time-labelled rows). Per user, `work/dump-<key>/<user>.npz` (gitignored, never committed) holds, per time-labelled test review (duration above 0, not a first review, same-day reviews included; sorted by review index): the review index, the card id, the grade, the duration, the elapsed days, P(grade) (4, float32), the head's 4 x 16 distribution (float32) and the ablation head's 4 x 16 distribution (float16). About 420 bytes per review, about 12 GB per model. The index `gpu/dump-<key>.jsonl` holds per user the review sha256 and the true-grade censored LogLoss and absolute log error sums of the head and the ablation (SM117's `user_sums`). The launcher checks every user against the model's recorded scoring rows: same sha256, same counts, sums within 1e-6 relative. A user that fails is listed and left out.

## Analysis rows (CPU)

- **Scored rows:** the dump rows on a later day than the card's previous review (elapsed days at least 1, the SM82 rule), whose card has at least one earlier timed review (so every baseline exists). The share of rows dropped by the second condition is reported.
- **Actual log time:** y = ln(min(duration, user cap, 60 s) / 1 s). Times at a lower user cap sit at the cap.
- **Predicted quantiles:** the q-quantile (q = 0.10, 0.25, 0.50, 0.75, 0.90) of the head's distribution for the true grade, in ln seconds, linear in ln time inside the crossing bin (bin 0 from 0.25 s; bin 15, 60 s or more, is a point mass at 60 s), the rule of SM117's `median_log_s`. The predicted log median x = q50; the predicted width w = q75 - q25.
- **Baselines** (causal, from all of the user's earlier reviews with a duration above 0, same-day ones included, each time capped as y):
  - (a) the card's running median: the median of the log times of all the card's earlier timed reviews;
  - (b) the card's last time: the log time of the card's most recent earlier timed review;
  - (c) the user's running median for the true grade: the median of the log times of all of the user's earlier timed reviews of that grade (the user's all-grade running median while there is none of that grade; rows with neither are dropped).
- **Strata:** unless stated otherwise every comparison is within a user and within a true grade. The head knows the grade it is conditioned on, and the card baselines do not, so pooling grades would credit the head with the grade (Again answers are slower), not with knowing the card. Grades-pooled versions are secondary.
- **Aggregation:** per user first, then over users. Every CI is a paired user bootstrap: 2,000 resamples of the users, seed 0 (`np.random.default_rng(0).integers(0, n, (2000, n))`, SM117's `bootstrap_idx`), 95% percentile interval. Each metric names its user weighting.

## Metrics

### 1. Within-user discrimination (AUC)

- At S = 2, 3, 4, 6, 8, 10 s, for users whose cap is above S: label = answered within S s.
- Head score: P(answered within S s | the true grade), from the head's true-grade distribution (SM117's `cdf_at`, linear in ln time inside the bin). Ablation score: the same from the ablation head. Baseline scores: minus (a), minus (b), minus (c) (a shorter time ranks faster).
- AUC with midranks for ties, computed within each (user, true grade) stratum that has both labels; a user's AUC is the pairs-weighted mean of its strata AUCs (pairs = positives x negatives). A user counts at S if it has at least 10 positives and 10 negatives in such strata.
- Aggregate: the mean over counted users (each user equal). Paired differences head minus (a), head minus (b), head minus (c), head minus ablation, with CIs.
- Baseline (c) varies within a stratum only through the drift of the running median, so it sits near 0.5 and stands for "the user's template curve per grade".
- Secondary: (i) within user, grades pooled, the head on the true grade against the same baselines; (ii) within user, grades pooled, the head's own grade mix, P(answered within S s) = sum over grades of P(grade) P(answered within S s | grade), against (a), (b) and the user's all-grade running median.

### 2. Range (calibration slope and spread)

- Review level: within each (user, grade) stratum, demean x and y; per user the mean cross-product cov_u and the mean square of x var_u over its scored rows. Pooled slope = sum_u cov_u / sum_u var_u: each user is weighted by its predicted variance, not by its review count. Also the median of per-user slopes over users with at least 200 scored rows. The same for (a), (b) and the ablation (its own q50). Secondary: the head's slope within user with grades pooled (demeaned per user only), the literal version of the question.
- Card level, out of sample: cards with at least 2 scored rows. Values are centred per (user, grade) stratum (x and y each by their own stratum mean). Each card's scored rows split in time: A = the first floor(n/2), B = the rest, so every prediction in A was made before every time in B. Per user with at least 20 such cards, over cards (each card equal): V_p = var(mean x over A), C_yy = cov(mean y over A, mean y over B) (the persistent between-card variance of the actual times, free of within-card noise), C_py = cov(mean x over A, mean y over B).
  - between-card slope = sum_u C_py / sum_u V_p;
  - spread ratio = sqrt(sum_u V_p / sum_u C_yy): the SD of the predicted log medians between cards over the SD of the persistent actual between-card means; also its median over users with C_yy above 0.
  - Reading: a calibrated predictor has between-card slope 1 and spread ratio = sqrt(explained between-card share) (metric 3). A spread ratio near 1 needs a predictor that knows every card perfectly; a slope well below 1 means the spread is too wide for what the prediction knows, a slope well above 1 means it is squeezed toward the average.

### 3. Variance explained, between cards and within cards

- Between cards (same cards, split and centring as metric 2): explained share = sum_u (C_py^2 / V_p) / sum_u C_yy, the share of the persistent between-card variance that the predicted card means explain linearly (reliability-corrected). Head, ablation, (a), (b); paired differences head minus (a) and head minus ablation.
- Within cards: cards with at least 3 scored rows; x and y (centred per stratum) minus the card's mean over all its scored rows; per user C_w = mean product, V_pw = mean square of x, V_yw = mean square of y. Explained share = sum_u (C_w^2 / V_pw) / sum_u V_yw. It is mildly in-sample (a card mean includes later times whose predictions saw the earlier ones) and is a description only; V_yw contains the unpredictable review-to-review noise, so its share is small for every predictor.
- Context: the persistent between-card share of each user's (user, grade)-centred variance, C_yy / (C_yy + V_yw), median over users.

### 4. Shape expressiveness

- Sample for the pooled PCA: up to 200 scored rows per user, uniform without replacement (`np.random.default_rng([0, user])`), about 140,000 rows per model, the head's true-grade distribution.
- Location-normalised shape: the CDF evaluated at q50 + u for u from -2.5 to +2.5 ln-units in steps of 0.05 (101 points; ln time below 0.25 s gives 0, at 60 s or more gives 1). Location-including: the CDF at ln t for t from 0.5 s to 60 s in steps of 0.05 ln-units.
- PCA (covariance over the sample, rows equal): the number of components for 90%, 95% and 99% of the variance and the participation ratio (sum of eigenvalues)^2 / sum of squared eigenvalues. Done for both representations, and for the ablation.
- **Effective number of distinct curves:** each row's quantile vector rounded to a grid of delta = 0.1 ln-units (about 10% in time): location-normalised (q10 - q50, q25 - q50, q75 - q50, q90 - q50) and location-including (q10, q25, q50, q75, q90). Two curves are "the same" iff every quantile rounds to the same cell. Effective number = exp(Shannon entropy of the cell occupancy). Reported per user (all its scored rows; median and p10 / p90 over users, with the median row count) and over the pooled sample. Also delta = 0.2 as a coarser reading. Also per user the SD of x and the SD of w over its scored rows.
- **Shape calibration:**
  - Card level: cards with at least 2 scored rows in A and 3 in B (same split). Predicted width = the mean w over A; realised spread = the SD of the stratum-centred y over B. Within each user with at least 20 such cards, the Spearman correlation over cards; the mean over users (equal) with CI. Comparator: the card's own earlier spread, the SD of the centred y over A. Paired difference head minus comparator.
  - Review level: within user, the Spearman correlation of w with |y - x| over the scored rows; mean over users with CI. Coverage of the predicted interquartile range: the share of rows with y between q25 and q75 in each within-user quintile of w (ideal 50% in every quintile), the mean over users.

### 5. Life of a card

- Answer count = the number of earlier reviews of the card (all, timed or not, any day) before the scored row; buckets 1, 2-3, 4-7, 8-15, 16-31, 32+.
- Per user and bucket with at least 20 scored rows: the mean x minus the user's mean y over all its scored rows (predicted), the mean y minus the same (actual), the mean w (predicted width) and the realised interquartile range of the residuals y - x. Mean over users that have the bucket (equal), with CIs of predicted minus actual; levels shown as % of the user's typical time, exp(value) - 1. All grades (true-grade predictions) and Good only, with the bucket's grade shares.

### 6. Plain-language read-out

Per model: how many genuinely different time curves a user gets (the per-user effective numbers of metric 4, location-including and location-normalised), and whether the head knows which card is slow better than the card's own history (metric 1 and 3 against (a) and against the ablation).

## What counts as expressive (description, pre-registered)

- **E1 tells fast cards from slow cards:** the within-user, within-grade AUC of the head is above the card's running median (a), with the 95% CI of the difference above 0, at every S of 2, 3, 4 and 6 s.
- **E2 spreads cards widely enough:** the review-level pooled slope (metric 2) is at least 0.8, and the out-of-sample between-card slope is between 0.8 and 1.25.
- **E3 explains more of the persistent card differences than the card's own history:** the between-card explained share of the head is above that of (a), CI of the difference above 0.
- **E4 has informative shapes:** the location-normalised shapes need at least 2 PCA components for 95% of the variance, and the card-level shape calibration correlation is above 0 with its CI above 0. Distinct shapes without E4's calibration are shape noise, not expressiveness.
- Label per model: **card-expressive** iff E1, E2 and E3 hold; **shape-expressive** iff E4 also holds; **template-like** iff E1 fails. The 0.05% minimum effect of the LogLoss go rules does not apply (no LogLoss here).

## CPU proof before the GPU tickets

`scripts/sm142_expr.py check`: the dump on CPU for 2 small users with the six-layer Tiny two-head model (SM130) and its SM138 head equals SM138's recorded GPU scoring sums within 1e-5 relative (same sha256, counts exact); the baselines (a), (b), (c) equal a brute-force recomputation on those users; the AUC function equals a brute-force pair count on random data; the quantile function reproduces `median_log_s`; and the analysis runs end to end on those users (a pipeline check, not a result).

## Launch

- `launch-sm142.sh`: one queue ticket per model through `scripts/r01_queue.py` (owner sm142, purpose "Answer-time expressiveness (SM142): ...", cap 0.8; missing users in a second pass at 0.92), order the six-layer Tiny two-head model (SM130), the six-layer Tiny Instant-Only model (SM130), the longer Small Instant-Only model (SM113), the show-time two-head cool-down for Small (SM118), then the show-time instant-only big model (SM99). Expected 15 to 20 min each (SM125 and SM128 dumps: 14 to 19 min), about 90 GPU minutes. Only quiet.json windows of scope all / gpu count. STOP file, resume (dumped users skipped), DRYRUN, CR strip on every Python output the launcher evaluates, and a final check of the outputs before `done`.
- The CPU analysis is not in the launcher: a fresh agent runs `analyse` per model (one process, 1 thread, under 6 GB RAM, at most 2 at a time) and `report` from `HANDOFF.md`.
