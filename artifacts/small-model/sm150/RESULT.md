# Rank-1 int8 states (SM150): result

Status: decided, 11-23, finished 21:38. Full tables: `sm150-result.md` and `sm150-result.json`; rules: `design.md`.

## Question

Under the user's 11-23 budgets, rank 2 (Q2) already passes every size (low-rank int8 states test, SM148). Does the next step down also pass: rank 1 everywhere (R1), or rank 1 below 16 answers and then rank 2 (R12)? Both are post-training formats with int8 factors, scored on the six release models on the 732 set (GPU), with long-card users (128+ answers) on the CPU for Small and Tiny as warnings. The cost is the 95% CI upper bound per head. Big is judged against its own int8 (at most +0.15%); Small and Tiny are judged by their total cost against the matching Big model at int8 (at most +0.5% and +1.0%).

## Result

Neither R1 nor R12 passes for any size, except that the Small Instant-Only model (SM113) alone passes R12. Because the Small two-head model (SM118) fails R12, the Small size does not pass. The go pick per size is rank 2 (Q2) for Big, Small and Tiny.

| size | R1 state bytes vs int8 | R12 state bytes vs int8 | Q2 state bytes vs int8 | R1 | R12 | Q2 (go) |
|---|---|---|---|---|---|---|
| Big | -88.1% | -87.6% | -82.8% | fails | fails | passes |
| Small | -78.4% | -77.4% | -68.2% | fails | fails (only SM113 passes) | passes |
| Tiny | -76.6% | -75.6% | -66.7% | fails | fails | passes |

Upper bounds of the cost (curve / instant, 95% CI):

- Big two-head (SM91s2) vs its own int8: R1 +0.507% / +0.543%, R12 +0.331% / +0.368%, Q2 +0.101% / +0.122%. Big Instant-Only (SM99): R1 +0.592%, R12 +0.408%, Q2 +0.132%. The limit is +0.15%.
- Small two-head (SM118) vs Big two-head (SM91s2) at int8: R1 +0.666% / +0.659%, R12 +0.581% / +0.525% (fails), Q2 +0.422% / +0.257%. Small Instant-Only (SM113) vs Big Instant-Only (SM99): R1 +0.589%, R12 +0.446% (passes), Q2 +0.127%. The limit is +0.5%.
- six-layer Tiny two-head (SM130) vs Big two-head (SM91s2) at int8: R1 +1.355% / +1.454%, R12 +1.175% / +1.188%, Q2 +0.939% / +0.861%. six-layer Tiny Instant-Only (SM130) vs Big Instant-Only (SM99): R1 +1.434%, R12 +1.168%, Q2 +0.785%. The limit is +1.0%.

Where the cost sits: R1 grows with the answer count (for Big two-head (SM91s2) curve from +0.125% at 1-2 answers to +0.739% at 128+). R12 removes most of it at 16 answers and above (for the same model +0.332% at 16-31) but is highest at 9-15 answers (+0.466%), where rank 1 is still used. Q2 stays at +0.05% to +0.17% on the curve head at every answer count.

## Warnings (report only)

- Long cards (128+ answers, CPU, Small and Tiny): R1 and R12 show a positive cost on the instant head for every model (for example the Small Instant-Only model (SM113) R1 +1.235% point, upper bound +2.201%), and the share of long-card cards with a review off by more than 0.10 is 0.6% to 7.4% for R1 and R12 (Q2 and Q4 not run on this read-out). Q2 long-card costs are +0.27% to +0.55% point, upper bounds up to +1.7%.
- Big was not run on the CPU long-card step.
- Coverage: Big 718 of 718 scorable users (GPU), Small and Tiny 723 of 723 users, long-card CPU 400 rows, 0 users killed, 0 failed.
- Integrity: the reference rescored on the 3 smoke users equals the low-rank test (SM148) rows within 1e-6 (`check-cpu-sm150.json`).

## Earlier low-rank test (SM148)

The mixed format "rank 2 below 16 answers, rank 4 from 16" (T24) also passed every size in the low-rank int8 states test (SM148), with more headroom than Q2 (state bytes -81.7% / -66.3% / -64.7% for Big / Small / Tiny against -82.8% / -68.2% / -66.7% for Q2). The main session's recommendation therefore stays T24. The final format per size is the user's decision.

## Speed (report only)

Review evaluations per second on the GPU (732 set), reference vs R1 / R12 / Q2: Big two-head (SM91s2) 76,714 vs 30,824 / 30,392 / 29,769 (about 0.39x); Small two-head (SM118) 124,598 vs 57,718 / 56,718 / 57,783 (about 0.46x); six-layer Tiny two-head (SM130) 295,619 vs 84,726 / 83,121 / 85,607 (about 0.28x). On the CPU (long cards) 0.86x to 0.96x of int8. The torch step loop is unfused, so the fork needs a fused kernel for any low-rank format.

## Cost of the run

GPU 208 minutes in 11 tickets (smoke 1, six-layer Tiny 43, Small 59, Big 105 including three retry passes); no ticket failed.

## What it means for the fork

Rank 1 is too lossy for every size and rank 1 then 2 is too lossy for all models except Small Instant-Only (SM113), so the state format will not go below rank 2 on any size. The choice for each size is between Q2 (smallest passing, -82.8% / -68.2% / -66.7%), T24 (about 1% to 2% more bytes but more headroom) and Q4 (rank 4, -74.4% / -52.3% / -51.1%).
