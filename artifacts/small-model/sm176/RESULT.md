# How many users for an MIT Small (SM176): read-out

Finished Fri 2026-11-27 13:06 ("SM176 DONE", final report exit 0). Full tables: `sm176-result.md`, `sm176-result.json`.

## Read this first: what the gaps mix

Every arm used the same review budget: 2.33B reviews (four epochs of the 8,000 users). That is much shorter than the training of the Small two-head release model (SM118), which had about 26 constant epochs plus a 32k-window cool-down, with a teacher. So a gap versus the release model (SM118) mixes three effects: fewer users, no teacher, and a shorter budget. The 8,000-user run WITH the teacher (SM176 n8000-kd) has all users and a teacher, so it isolates the budget effect alone: +1.19% curve / +1.34% instant vs the release model. The fair same-budget comparison is therefore each arm vs that control (SM176 n8000-kd). Early stopping lets the no-teacher arms train for fewer reviews than the budget (1.44B / 0.73B / 1.73B), so they are not all at the same compute.

## Gaps in LogLoss (curve / instant), 732 set (723 users), paired bootstrap 95% CI

| arm | vs release model (SM118) curve / instant | vs control (SM176 n8000-kd) curve / instant |
|---|---|---|
| 1,000 users, no teacher (SM176 n1000) | +2.57% / +2.36% | +1.36% / +1.01% |
| 2,000 users, no teacher (SM176 n2000) | +3.29% / +3.30% | +2.08% / +1.93% |
| 4,000 users, no teacher (SM176 n4000) | +1.40% / +1.55% | +0.21% / +0.21% |
| 8,000 users, with teacher (SM176 n8000-kd) | +1.19% / +1.34% | (control) |

CIs (vs control): n1000 [+1.18, +1.60] / [+0.94, +1.09]; n2000 [+1.86, +2.32] / [+1.80, +2.08]; n4000 [+0.10, +0.29] / [+0.18, +0.24].

## Warning: the 2,000-user arm is an outlier

It is worse than the 1,000-user arm, which cannot be a real data effect. Its held-out slice got worse at epoch 4 (0.3167 vs 0.3164 at epoch 2), so early stopping fired after only 2 constant epochs plus 1 decay epoch (0.73B reviews, about 1.9 GPU h), while the 1,000 and 4,000 arms trained 1.44B and 1.73B. The arm was most likely stopped too early by a noisy slice, not data-starved. Treat the n2000 gap as an upper bound, and the interpolated user counts below as conservative.

## What the final report's numbers mean

- "~2,982" (+1.0%) and "~3,590" (+0.5%): last column of the users-needed table. Number of users where the no-teacher arms, interpolated linearly in log2(users) on the worse head, first come within +1.0% (and +0.5%) of the same-budget teacher control (SM176 n8000-kd). Point estimate only. Because the n2000 arm is a bad outlier, the first crossing is found on the 2,000 to 4,000 segment (2.08% down to 0.21%).
- Sensitivity (my own calculation, not in the report): ignoring n2000 and interpolating 1,000 to 4,000 gives about 1,550 users for +1.0% and about 2,830 for +0.5%. The true answer is probably between these and the report's values. Since n4000 is already +0.21%, both thresholds are met at or below 4,000 users.
- "> 4,000 (extrapolated ~16,994 / ~37,287, indicative)" (vs the release model SM118, point; CI upper ~15,883 / ~30,164): a power-law fit through three points (1,000 / 2,000 / 4,000). The extrapolation is not trustworthy: n2000 distorts the fit, and the floor is a budget effect, since even the control with all users is +1.19% / +1.34% vs release. +1.0% / +0.5% of the release model is not reached here by data alone.

## Same-budget teacher run: users needed (point estimate, worse head)

| threshold vs SM176 n8000-kd | report (interpolation 2,000 to 4,000) | sensitivity (1,000 to 4,000) |
|---|---|---|
| within +1.0% | ~2,982 | ~1,550 |
| within +0.5% | ~3,590 | ~2,830 |

## Small vs Tiny (data-scaling test SM174, same recipe and budget rule)

| | within +1.0% of same-budget teacher run | within +0.5% | 4,000 users, no teacher vs control | control vs release |
|---|---|---|---|---|
| Tiny (SM174) | ~1,182 | ~3,194 | +0.45% / +0.25% | +1.25% / +1.45% |
| Small (SM176) | ~2,982 (~1,550 ignoring n2000) | ~3,590 (~2,830) | +0.21% / +0.21% | +1.19% / +1.34% |

Small needs about the same number of users as Tiny for +0.5% (3,590 vs 3,194, within the noise of the n2000 outlier), and probably somewhat more for +1.0%. At 4,000 users Small is actually closer to its control than Tiny (+0.21% vs +0.45% curve). A bigger architecture does not clearly need many more users at this budget. A confirmation run of n2000 with a longer patience (or 8,000 users without a teacher) would settle it.

## Other facts

Early stops (held-out slice): n8000-kd at constant epoch 2 + 1 decay, n1000 at 16 + 3, n2000 at 2 + 1, n4000 at 5 + 1. Speed 106k to 110k reviews/s (the 87k plan basis was conservative); GPU 6.1 / 3.7 / 1.9 / 4.4 h for n8000-kd / n1000 / n2000 / n4000 (16.1 h total). Weights of this test are experiment-only, never shipped.
