# Training-free state check (SM116): result

**Verdict: starting point.** Keeping 70% of the show-time retrain (SM92)'s card and note state (2,373 of 3,392 values, per block PCA, no training) costs
+0.015% [-0.007, +0.032] on the curve head and +0.032% [+0.025, +0.038] on the instant head, in % of the big model (1 leak-free epoch)
show-time LogLoss. Both upper 95% CI bounds are below the pre-registered +0.05%, so the 70% state becomes the starting point of a trained smaller model.

## Heads (629 users, 14,906,845 scored reviews per head, paired user bootstrap 2,000, seed 0)

| head | unmodified | 70% state | difference | % of the big model's LogLoss (CI) | % of own LogLoss (CI) | users better / worse |
|---|---|---|---|---|---|---|
| curve | 0.303053 | 0.303099 | +0.0000462 | +0.015% [-0.007, +0.032] | +0.015% [-0.007, +0.032] | 248 / 380 |
| instant | 0.274177 | 0.274265 | +0.0000878 | +0.032% [+0.025, +0.038] | +0.032% [+0.025, +0.038] | 203 / 426 |

The curve head's CI includes 0; the instant head's cost is small but clearly above 0.

## Direction users (count discrepancy)

`users-directions.json` holds the 100 users drawn; 6 of them (above 125,000 reviews) were dropped for the RAM limit, so the directions come from
**94 users** (3,450,496 reviews, `users_used`). `sm116-decision.md` (generated text) says "100 users"; that is the drawn count, the correct number used is 94,
as in `design.md` and `HANDOFF.md`.

## Directions table, proof, pool

See `design.md` (sections "Directions found" and "CPU proof result"): variance kept at 70% is 98.65% to 99.98% per block; 90% of the variance needs 4 to 74 of 256
directions per WKV head, 99.9% needs 98 to 242. CPU proof PASS: the 100% projection equals the unmodified model bit for bit on 8 users (GPU rows within 2e-6).
Pool caveat: 629 users with <= 125,000 reviews of the 722-user rule pool (21.4M reviews, 51% of the pool's reviews); the 93 heaviest users are not scored
(the CPU scorer needs more than 6 GB for them).

## Run: two passes

The same command line and the same users in both passes; the 629 users are scored once each.

- **Pass 1** (`launch-sm116.sh`, 15:58-17:11 Wed 11-18): 3 long-lived scorer processes over fixed slices, 357 users scored. The memory guard killed
  two scorers (part a at 16:17, 5.93 GB, after 53 users; part b at 16:36, 5.97 GB, after 95 users) because the memory of a long-lived process only grows
  (high-water mark), not because a single user was too big. Part c finished all 209 users.
  Part a: 1,112 s, 1,449 review-evals/s; part b: 2,284 s, 1,527 review-evals/s; part c: 4,365 s (about 73 min), 1,540 review-evals/s.
  Peak RAM 5.97 GB (the killed part b).
- **Pass 2** (`launch-sm116-r2.sh`, 17:22-18:04): the remaining 272 users, each in its own fresh process, 3 at a time; 0 killed, 0 failed.
  Wall time about 43 min (17:22:05 to 18:04:53); 9.62M reviews, 1,457 review-evals/s per process on average (user-weighted over `rows/r2/progress.out`),
  about 3,700 review-evals/s with 3 processes. Peak RAM 5.61 GB (one process, 17:35:48), under the 5.9 GB guard.
- Total: 15:58 to 18:04 (2 h 06 min including the 11-minute gap between the passes).
