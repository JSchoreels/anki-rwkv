# Rank-2 states through a short cool-down (SM149): stage 1 result (Small two-head (SM118))

**Verdict: no effect, stop. Every size keeps rank 4 with int8 factors (Q4), the go of the low-rank int8 states
experiment (SM148): Big &minus;74.4%, Small &minus;52.3%, Tiny &minus;51.1% state bytes vs int8** (Tiny's looser rule still allows rank 2
below 16 answers then rank 4 (T24, &minus;64.7%) from SM148). Stage 1b (4x reviews) and stage 2 (the other five models) are
not run. The long-card CPU step is skipped: it cannot rescue a label that already fails the dataset-wide rule.

What was measured: a short cool-down from the final weights of Small two-head (SM118) (67.3M reviews, 794 users), in
two arms: QAT (every card and note write stored in the deployed rank-2 int8-factor format during training,
straight-through gradient) and a control with the identical cool-down and the codec off. Both scored on the 732 set
(723 users, show-time inputs) at int8 (REF), rank 2 (Q2) and, for the QAT arm, rank 2 then 4 (T24). The original model's rows are
those of SM148 (not rescored; the integrity check on 3 users matched). Statistic as SM148: review-weighted LogLoss
ratio, paired user bootstrap, 2,000 draws, seed 0, 95% CI. Numbers from `sm149_report.py` (reuses the helpers of
`sm148_report.py`), raw output `sm149-stage1.json`.

## Rule R2 (vs the original Small two-head (SM118) at int8; limit: CI upper bound +0.05% per head)

| Model and label | curve, value [95% CI] | instant, value [95% CI] | R2 |
|---|---|---|---|
| QAT arm at int8 (R6: new model at int8) | +0.009% [&minus;0.037, +0.046] | +0.021% [+0.016, +0.026] | passes |
| Control arm at int8 | &minus;0.003% [&minus;0.050, +0.035] | +0.003% [&minus;0.000, +0.007] | passes |
| QAT arm, Q2 (rank 2) | +0.038% [&minus;0.009, +0.074] | +0.073% [+0.066, +0.080] | fails (instant) |
| QAT arm, T24 (rank 2 then 4) | +0.026% [&minus;0.021, +0.061] | +0.051% [+0.045, +0.057] | fails (both heads) |
| Control arm, Q2 | +0.048% [+0.000, +0.089] | +0.082% [+0.074, +0.090] | fails (both heads) |
| Original model, Q2 (SM148, for reference) | +0.047% | +0.079% | fails |
| Original model, T24 (SM148, for reference) | +0.029% | +0.050% | fails |

No rank-2 label passes. (The figures differ in the third decimal from a quick first read because this one uses
exactly the 723 users common to all labels and the SM148 bootstrap helper.)

Against each arm's own int8 rows (reported, no verdict): QAT Q2 +0.029% [+0.020, +0.039] curve / +0.052% [+0.046, +0.058]
instant; QAT T24 +0.016% [+0.010, +0.023] / +0.030% [+0.026, +0.035]; control Q2 +0.051% [+0.041, +0.061] / +0.079%
[+0.072, +0.086]. Same label against the original model's: QAT Q2 &minus;0.008% / &minus;0.006% [&minus;0.011, &minus;0.001]; QAT T24
&minus;0.003% / +0.002%; control Q2 +0.002% / +0.003%. QAT Q2 vs control Q2: &minus;0.010% [&minus;0.022, +0.001] curve /
&minus;0.009% [&minus;0.013, &minus;0.005] instant. QAT int8 vs control int8: +0.012% / +0.018%.

## The gate (fixed before the run)

- C_qat = QAT Q2 vs QAT int8, instant: +0.052%.
- C_ctl = control Q2 vs control int8, instant: +0.079%.
- Ratio C_qat / C_ctl = 0.66, above the 0.5 line: **no effect** (not partial, not pass). Stop: every size keeps Q4.

Meaning: rank-2-aware training with a short cool-down buys about a third of the rank-2 cost on Small (the instant
cost falls from +0.079% to +0.052% against the arm's own int8, and from +0.082% to +0.073% against the original), not
enough to pass the +0.05% rule. Rank 2 then 4 (T24) is within +0.002% of the original's, so QAT does not help there
either. Note T24 against its own int8 rows (upper bound +0.035%) would pass, but the rule is against the original
model at int8, and the QAT arm's int8 rows are themselves +0.021% worse on the instant head.

## Cost by answer count (all-scope reviews, 723 users; instant head, value [95% CI])

| Answers before the review | Reviews | QAT Q2 vs original int8 | Control Q2 vs original int8 | QAT Q2 vs QAT int8 | Control Q2 vs control int8 |
|---|---|---|---|---|---|
| 1-2 | 10.27M | +0.032% [+0.027, +0.038] | +0.031% [+0.025, +0.037] | +0.017% | +0.025% |
| 3-4 | 7.24M | +0.046% [+0.039, +0.054] | +0.052% [+0.044, +0.060] | +0.033% | +0.049% |
| 5-8 | 7.94M | +0.075% [+0.067, +0.084] | +0.085% [+0.076, +0.095] | +0.055% | +0.082% |
| 9-15 | 5.34M | +0.098% [+0.088, +0.109] | +0.109% [+0.097, +0.121] | +0.072% | +0.108% |
| 16-31 | 3.71M | +0.112% [+0.096, +0.128] | +0.134% [+0.119, +0.150] | +0.089% | +0.136% |
| 32-127 | 2.09M | +0.109% [+0.082, +0.138] | +0.142% [+0.112, +0.178] | +0.087% | +0.140% |
| 128+ | 0.09M | +0.123% [+0.034, +0.310] | +0.176% [+0.052, +0.390] | +0.160% | +0.195% |

The cost rises with answer count as in SM148, and QAT trims every bucket from 9 answers up by roughly a fifth to a
third, never to zero. Curve-head buckets are in `sm149-stage1.json`.

## Training and GPU time (Small two-head (SM118), 67,256,032 reviews, cap 0.8)

| Arm | Time | Reviews/s | Peak allocated | Reserved |
|---|---|---|---|---|
| QAT (rank 2, state_vq lr2q8x8) | 112.2 min | 9,993 | 8.6 GB (8,556 MiB) | 13.0 GB |
| Control (codec off) | 38.8 min | 28,883 | 7.3 GB (7,269 MiB) | 13.0 GB |

QAT cost 2.9 times the control's time, in line with the 3.08 times estimated. GPU tickets (`gpu/launch-stage1.log`),
09:52 to 13:21 Mon 11-23, about 209 minutes in 6 tickets: trainer smoke 0.3 min, QAT 112.2, control 38.8, integrity
(original REF on 3 users) 0.2, scoring pass 1 (cap 0.8, 721 users per label) 55.7, scoring pass 2 (cap 0.92, the
missing 2 users, 723 for every label) 1.0. Earlier throughput and OOM-probing tickets not counted. Coverage: 723 of 723 users per label.

## Not run

Stage 1b (Small two-head once more with 4x the reviews), stage 2 (Small Instant-Only (SM113), Big two-head (SM91s2),
Big Instant-Only (SM99), the two six-layer Tiny models (SM130): about 21 GPU hours), the long-card CPU scoring and the
per-card read-out of the new models.
