# How many users for MIT weights (SM174): read-out

Finished Thu 2026-11-26 18:26 ("SM174 DONE", final report exit 0). Full tables: `sm174-result.md`, `sm174-result.json`.

## Read this first: what the gaps mix

Every arm used the same review budget: 2.33B reviews (about 4 epochs of the 8,000 users). That is much shorter than the training of the six-layer Tiny two-head release model (SM130), which had about 30 epochs plus a 32k-window cool-down with a teacher. So a gap versus the release model (SM130) mixes three effects: fewer users, no teacher, and a shorter budget. The 8,000-user run WITH the teacher (SM174 n8000-kd) has all users and a teacher, so it isolates the budget effect alone: +1.25% curve / +1.45% instant vs the release model. The fair same-budget comparison is therefore each arm vs that control (SM174 n8000-kd).

## Gaps in LogLoss (curve / instant), 732 set (723 users), paired bootstrap 95% CI

| arm | vs release model (SM130) curve / instant | vs control (SM174 n8000-kd) curve / instant |
|---|---|---|
| 500 users, no teacher (SM174 n0500) | +3.97% / +3.78% | +2.69% / +2.30% |
| 1,000 users, no teacher (SM174 n1000) | +2.39% / +2.12% | +1.12% / +0.67% |
| 2,000 users, no teacher (SM174 n2000) | +1.87% / +1.88% | +0.61% / +0.43% |
| 4,000 users, no teacher (SM174 n4000) | +1.70% / +1.70% | +0.45% / +0.25% |
| 8,000 users, no teacher (SM174 n8000) | +1.49% / +1.72% | +0.24% / +0.27% |
| 8,000 users, with teacher (SM174 n8000-kd) | +1.25% / +1.45% | (control) |

Cost of dropping the teacher at 8,000 users: +0.24% curve / +0.27% instant (CI upper +0.32% / +0.31%).

## What the final report's two numbers mean

- "~1,182" (and "~3,194"): the last column of the users-needed table. It is the number of users where the no-teacher arms, interpolated linearly in log2(users) on the worse head, first come within +1.0% (and +0.5%) of the equal-budget teacher control (SM174 n8000-kd). Point estimate only. So about 1,182 users reach +1.0% of the same-budget teacher model and about 3,194 reach +0.5%.
- "fitted floor +1.67% / +1.78%": versus the release model (SM130), the no-teacher gap flattens as users grow. A power-law fit gap(n) = a + b n^-c over the five no-teacher arms gives a floor a of +1.67% (point) / +1.78% (CI upper bound). It is what infinite users would still cost at this short budget without a teacher. It is not a data cost: it is the budget and recipe cost plus the teacher cost. Hence +1.0% / +0.5% of the release model (SM130) is "> 8,000, not reached" at this budget.

## Decomposition at 8,000 users (curve / instant)

Budget and recipe: +1.25% / +1.45% (control vs release). No teacher: +0.24% / +0.27% more. Fewer users (4,000 and below): the rest of the table above.

## Other facts

Early stops (held-out slice) came at best constant epochs 7 / 20 / 8 / 4 / 2 / 2 for 500 / 1k / 2k / 4k / 8k / 8k-kd, then a short decay. Speed 177k to 188k reviews/s; GPU 0.9 / 3.2 / 2.4 / 2.1 / 3.5 / 3.6 h (500 users stopped early at 0.58B reviews, not the full budget). Weights of this test are experiment-only, never shipped.
