# Result: the show-time two-head cool-down for Small (SM118)

Verdict: **go**, held for the next release decision on 11-28 together with the longer big instant-only teacher (SM113) and the longer cool-down for Tiny (SM115). Nothing here is released.

On the 722 users of the 732 set, with show-time inputs, the show-time two-head cool-down for Small (SM118) beats the show-time retrain of the release Small two-head model (SM92) on both heads. The curve head is -0.114% [-0.177, -0.062] and the instant head is -0.100% [-0.110, -0.091] (paired user bootstrap, 95% CI). It closes 0.362 of the curve gap and 0.398 of the instant gap to the show-time big model (SM91s2). It is still +0.201% [+0.051, +0.359] (curve) and +0.151% [+0.126, +0.176] (instant) behind the show-time big model (SM91s2). The 1,500 validation users outside the 732 set agree: -0.102% curve and -0.099% instant versus the show-time retrain of the release (SM92). Full tables: `sm118-decision.md` and `sm118-decision.json`.

## Stages and speed

| stage | minutes | reviews/s |
|---|---|---|
| reference scoring of the show-time retrain of the release (SM92) after 4,056 users | 11 | scoring |
| teacher export from the show-time big model (SM91s2), 8,000 users, 581.9M reviews | 213 (12,769 s) | 45,576 |
| pass 1 of the cool-down (learning rate 7e-4 to 0.29 x peak) | 335.9 | 28,915 |
| pass 2, first segment (stopped for the 20:00 break from a snapshot) | 221.6 | n/a (segment) |
| pass 2, second segment (resumed after the 01:00 break) | 116.9 | 28,690 |
| early-stop scoring plus final scoring (732 set, normal inputs, 1,500 pool, both models) | about 82 in total | scoring |

Pass 2 in total took 338.5 minutes for 581.9M reviews, about 28.7k reviews/s. Both passes ended at the expected learning rates (0.2929 x peak after pass 1, 5.8e-11 x peak after pass 2). The early stop after pass 1 said continue: pass 1 was -0.147% curve and -0.093% instant against the snapshot of the show-time retrain of the release (SM92).

Launcher: started 06:40 Mon 11-16, ended 03:55 Tue 11-17.
