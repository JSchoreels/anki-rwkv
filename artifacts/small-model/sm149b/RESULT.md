# Rank-2 states with a tail-energy penalty (SM149b): stage 1 result (Small two-head (SM118))

**Verdict: no effect, stop. Every size keeps rank 4 with int8 factors (Q4), the go of the low-rank int8 states
experiment (SM148).** The penalty also made the cooled model itself worse at int8 (about +0.20% on the instant head
against the original Small two-head (SM118)), so even a partial gate result would have been moot. The pre-registered
follow-up with lambda = 3 is **not run** (it was only for a partial result), the other five release models are **not
run**, and the long-card CPU step is skipped: it cannot rescue labels that already fail the dataset-wide rule.

What was measured: the cool-down of the rank-2 cool-down (SM149) QAT arm (every card and note write stored in the
deployed rank-2 int8-factor format, straight-through gradient) plus a tail-energy penalty in the loss (lambda = 1,
one write step in 8), from the final weights of Small two-head (SM118) (67.3M reviews, 794 users). Scored on the 732
set (723 users, show-time inputs) at int8 (REF), rank 2 (Q2) and rank 2 then 4 (T24). Statistic as SM148:
review-weighted LogLoss ratio, paired user bootstrap, 2,000 draws, seed 0, 95% CI. The original model's rows are those
of SM148 and SM149's arms are reused as they are. Integrity check (SM149's QAT arm re-scored at int8 on 3 users):
identical to SM149's rows (max difference 0). Numbers from `sm149b_report.py`, raw output `sm149b-stage1.json`.

## Rule R2 and R6 (vs the original Small two-head (SM118) at int8; limit: CI upper bound +0.05% per head)

| Model and label | curve, value [95% CI] | instant, value [95% CI] | R2 / R6 |
|---|---|---|---|
| New arm at int8 (R6: new model at int8) | +0.142% [+0.107, +0.179] | +0.204% [+0.189, +0.220] | fails (both heads) |
| New arm, Q2 (rank 2) | +0.170% [+0.135, +0.208] | +0.249% [+0.231, +0.270] | fails (both heads) |
| New arm, T24 (rank 2 then 4) | +0.159% [+0.125, +0.196] | +0.232% [+0.215, +0.250] | fails (both heads) |
| Rank-2 cool-down (SM149) QAT arm, Q2 (reference) | +0.038% [&minus;0.009, +0.074] | +0.073% [+0.066, +0.080] | fails (instant) |
| Rank-2 cool-down (SM149) QAT arm, T24 (reference) | +0.026% [&minus;0.021, +0.061] | +0.051% [+0.045, +0.057] | fails (both heads) |
| Rank-2 cool-down (SM149) control arm, Q2 (reference) | +0.048% [+0.000, +0.089] | +0.082% [+0.074, +0.090] | fails (both heads) |

No rank-2 label passes, and the new model at int8 (R6) fails by about 4 times the limit on the instant head.
(The first quick read of the main session, +0.142% / +0.202% at int8, differs in the third decimal because this one
uses exactly the 723 users common to every label and the SM148 bootstrap helper.)

Against each arm's own int8 rows (reported, no verdict): new Q2 +0.028% [+0.022, +0.034] curve / +0.045% [+0.040, +0.051]
instant; new T24 +0.018% [+0.014, +0.021] / +0.028% [+0.025, +0.031]. Same label against the original model's: new Q2
+0.123% / +0.170%; new T24 +0.131% / +0.182%. Against the rank-2 cool-down (SM149) QAT arm at the same label (the
penalty's own effect): int8 +0.133% [+0.101, +0.171] curve / +0.183% [+0.167, +0.200] instant; Q2 +0.132% / +0.176%
[+0.159, +0.195]; T24 +0.134% / +0.180%. The new int8 model is also +0.201% [+0.184, +0.218] worse than SM149's control
on the instant head.

## The gate (fixed before the run)

- C_new = new Q2 vs the new arm's own int8, instant: +0.045% (a first quick read gave about +0.042%).
- C_ctl = SM149's control Q2 vs control int8, instant: +0.079%.
- Ratio C_new / C_ctl = 0.57, above the 0.5 line (partial needed C_new <= +0.0395%): **no effect**. For reference the
  rank-2 cool-down (SM149) QAT arm without the penalty had a ratio of 0.66.

So the penalty trimmed the rank-2 cost a little more (+0.052% to +0.045% against the arm's own int8), but the
model's own int8 accuracy lost about +0.18% on the instant head, which dwarfs that gain.

## Cost by answer count (all-scope reviews, 723 users; instant head, value [95% CI])

| Answers before the review | Reviews | New Q2 vs original int8 | New Q2 vs new int8 | QAT (SM149) Q2 vs QAT int8 | Control (SM149) Q2 vs control int8 |
|---|---|---|---|---|---|
| 1-2 | 10.27M | +0.191% [+0.175, +0.209] | +0.010% | +0.017% | +0.025% |
| 3-4 | 7.24M | +0.219% [+0.200, +0.238] | +0.023% | +0.033% | +0.049% |
| 5-8 | 7.94M | +0.252% [+0.233, +0.272] | +0.044% | +0.055% | +0.082% |
| 9-15 | 5.34M | +0.271% [+0.248, +0.295] | +0.063% | +0.072% | +0.108% |
| 16-31 | 3.71M | +0.315% [+0.287, +0.347] | +0.089% | +0.089% | +0.136% |
| 32-127 | 2.09M | +0.339% [+0.281, +0.408] | +0.091% | +0.087% | +0.140% |
| 128+ | 0.09M | +0.240% [+0.066, +0.663] | +0.086% | +0.160% | +0.195% |

The loss is spread over every bucket, including 1 to 2 answers (+0.19%), where rank 2 costs almost nothing: it is
the model's weights that moved, not the rank-2 truncation. Curve-head buckets are in `sm149b-stage1.json`.

## Tail energy over training (from `tail.jsonl`, 2,474 update passes; the first 3 are the trainer's discarded primer; 47 passes logged no value)

| Window | Mean tail energy |
|---|---|
| first 100 passes | 0.0184 |
| passes 401-500 | 0.0091 |
| passes 801-900 | 0.0050 |
| passes 2001-2100 | 0.0029 |
| last 100 passes | 0.0035 |
| whole run, mean | 0.0061 |

The tail energy fell about 5 times (first 100 passes over last 100 passes), against about 45 times reported by outside
research on its own model. The penalty was about 2% to 3% of the loss at the start and under 0.5% at the end.

## Training and GPU time (Small two-head (SM118), 67,256,032 reviews, cap 0.8)

Training: 125.8 min, 8,907 reviews/s (SM149's QAT arm: 112.2 min, 9,993), peak allocated 8.6 GB (8,592 MiB), reserved
13.0 GB. GPU tickets (`gpu/launch-stage1.log`), 14:00 to 16:42 Mon 11-23, about 162 minutes in 5 tickets: trainer
smoke 0.2 min, training 126.1, integrity 0.2, scoring pass 1 (cap 0.8, 721 users per label) 34.9, scoring pass 2 (cap
0.92, the missing 2 users, 723 for every label) 0.8. The earlier smoke ticket (001519, 6.9 min) is not counted.
Coverage: 723 of 723 users per label.

## Interpretation

On our Small model, lambda = 1 (about 2% to 3% of the loss at the start) pushes the states toward rank 2 (the tail
energy fell about 5 times), but at a large price in accuracy: the model at int8 is about +0.14% worse on the curve head
and +0.20% on the instant head than the original, where the version without the penalty (SM149) was +0.01% and
+0.02%. Outside research reported the opposite on its own model, where the penalty removed the rank-2 tax; its model
is different (d = 32, one head, card 1 layer and note 3 layers; its penalty fell to about 0.26% of the loss).

## Decision and not run

The decision stays rank 4 with int8 factors (Q4) on every size. Not run: the lambda = 3 follow-up, the other five
release models (Small Instant-Only (SM113), Big two-head (SM91s2), Big Instant-Only (SM99), the two six-layer Tiny
models (SM130)), and the long-card CPU scoring (R3 and the per-card guard).
