# Session inputs confirmation (SM181): result

Run finished 2026-11-28 05:13 ("SM181 DONE"). Full tables: `sm181-result.md`, `sm181-result.json`. Design and rules: `design.md`.

## Question

The session inputs screen (SM179) found an instant-head LogLoss gain of -0.065% for six-layer Tiny two-head (SM130) on the 732 set. Does it hold (1) on the decisive users (765 users with id >= 5000 of the 1,500 pool, same two Tiny models, no training) and (2) for Small two-head (SM118) after the same short fine-tune (arm `sm181-small-ses` against SM149's control `sm149-sm118-ctl`, scored on the 732 set)?

## Rules (pre-registered)

An axis is confirmed when the instant-head change is at most -0.05% with the upper end of its 95% confidence interval below 0, and the curve-head point estimate is at most +0.05%. The verdict is CONFIRMED if both axes are confirmed, PARTIAL if one is, NOT CONFIRMED if neither. The result is pending if an axis has too few users (743 on the decisive users, 720 on the 732 set), a row is not finite, or an integrity check fails. The statistic is the percentage change of the review-weighted LogLoss, with a paired user bootstrap of 2,000 draws.

## Verdict: CONFIRMED (both axes)

| axis | users | curve change | instant change | confirmed |
|---|---|---|---|---|
| six-layer Tiny two-head (SM130), decisive users | 761 | -0.000% [-0.003, +0.002] | -0.063% [-0.071, -0.056] | yes |
| Small two-head (SM118), 732 set | 723 | +0.003% [-0.001, +0.007] | -0.065% [-0.073, -0.057] | yes |

Control LogLoss: six-layer Tiny two-head (SM130) on the decisive users, curve 0.30366, instant 0.28118; Small two-head (SM118), curve 0.30197, instant 0.27863. Integrity checks are all OK: the Tiny benchmark sizes equal SM164's rows (662 common users, no mismatch), SM149's control rows are reproduced exactly (max difference 0.0), no non-finite user.

Answer-count buckets (report only): the instant gain is present in every bucket up to 1,000+ answers on the decisive users (-0.077% for 1 answer to -0.022% for 1000); the curve change is within +-0.005% for the buckets with most reviews. For Small the buckets with 300+ answers (2,723 and 5,997 reviews) are positive for the instant head (+0.17%, +3.5%), but they hold under 0.1% of the reviews.

Report only, six-layer Tiny two-head (SM130): 732 set (SM179's models rescored with the dump, 723 users): curve +0.001% [-0.002, +0.003], instant -0.065% [-0.073, -0.058], reproducing SM179. Both sets together (1,484 users): curve +0.000% [-0.002, +0.002], instant -0.064% [-0.069, -0.059].

## Secondary metric, report only (AUC, arm minus control; not part of the rule)

| axis | head | overall AUC control -> arm | overall change [95% CI] | within-user AUC control -> arm | within-user change [95% CI] |
|---|---|---|---|---|---|
| Tiny, decisive (761 / 755 users) | curve | 0.8400 -> 0.8400 | -0.0000 [-0.0000, +0.0000] (-0.000%) | 0.7877 -> 0.7877 | +0.0000 [-0.0000, +0.0001] (+0.005%) |
| Tiny, decisive | instant | 0.8670 -> 0.8672 | +0.0002 [+0.0002, +0.0003] (+0.026%) | 0.8391 -> 0.8395 | +0.0004 [+0.0003, +0.0004] (+0.042%) |
| Small, 732 set (723 / 722) | curve | 0.8467 -> 0.8467 | -0.0000 [-0.0000, +0.0000] (-0.000%) | 0.7978 -> 0.7978 | -0.0000 [-0.0000, +0.0000] (-0.002%) |
| Small, 732 set | instant | 0.8736 -> 0.8738 | +0.0002 [+0.0002, +0.0002] (+0.024%) | 0.8436 -> 0.8440 | +0.0003 [+0.0003, +0.0004] (+0.038%) |
| Tiny, 732 set (723 / 722) | curve | 0.8450 -> 0.8450 | -0.0000 [-0.0000, +0.0000] (-0.000%) | 0.7951 -> 0.7951 | -0.0000 [-0.0000, +0.0000] (-0.001%) |
| Tiny, 732 set | instant | 0.8719 -> 0.8721 | +0.0002 [+0.0002, +0.0002] (+0.025%) | 0.8418 -> 0.8421 | +0.0003 [+0.0002, +0.0003] (+0.034%) |

The curve head's AUC does not move. The instant head's AUC rises by about 0.0002 overall and 0.0003 to 0.0004 within users (+0.02% to +0.04%), with the confidence interval above 0, consistent with the LogLoss gain; the size is small.

## Cost

Small training: `sm181-small-ses` 28,779 reviews/s, 38.9 min, against the control's 28,883 reviews/s, 38.8 min (peak allocated 7,281 vs 7,269 MiB). The input cost is negligible. GPU tickets (from the log): trainer smoke 31 min wall (queue wait behind SM180), Tiny decisive scoring about 2.4 h including retry passes, Small training about 49 min wall, Small scoring about 24 min, Tiny 732 rescoring about 20 min.

Note: the launcher's last log line says "0 decided, 2 pending" next to the CONFIRMED verdict; the report exit code is 0 and `status` is `decided` in the json, so the wording is a stale launcher message.

## Decision

The session inputs go into the next family's recipe: a full Tiny cool-down with the release recipe (all 8,000 users), then Small, then the fork's feature builder for the 6 columns with test vectors. The true time of day and weekday (A19) remain the next inputs to test on donated data.
