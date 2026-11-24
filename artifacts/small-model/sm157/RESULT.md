# Output-aware low-rank truncation (SM157): result

CPU pointer, run finished Tue 11-24 03:19 (`SM157 RUN DONE`, commit f27a3e0f). Design and rule pre-registered in `design.md` (92e1ace3) before any weighted probe row was scored. Full numbers: `sm157-result.json`, `sm157-result.md`; rows: `cpu/sm157-rows.jsonl` (432 rows, 8 users x 6 models x 9 labels).

## Question

The rank-r codec of the low-rank int8 states test (SM148) keeps the Frobenius-optimal truncation of each card and note state, which treats every key direction as equally important. Does a truncation that minimises the expected read error, `||(S - S_hat) M^(1/2)||_F` with a fitted per-head key metric M, cut the cost of rank 2 (Q2) against int8 rows by at least a third, at unchanged stored bytes? Three metrics were tested: (a) receptance `r` (`M = E[r r^T]`), (b) receptance plus removal key `rk` (`M = E[r r^T] + beta E[kd kd^T]`, equal trace), and (c) the discounted Gramian `g` (`M = R + D M D`).

## Rule (pre-registered)

Cost of a label = review-weighted % change of the benchmark LogLoss against int8 rows (REF, `w8+s8hx8`) on the 8 probe users, paired user bootstrap (2,000 draws, seed 0, 95% CI). A weight kind passes on Q2 when at least 5 of the 9 (model, head) cells are informative (plain cost at least 0.01%), the median cost reduction over the informative cells is at least one third, and no cell of the 9 has the weighted label more than +0.01% worse than plain Q2 (direct paired %). T24 (the shipped format, rank 2 below 16 answers then rank 4) is reported outside the rule.

## Verdict: NO GO

No weight kind passes, so the pointer fails its own rule and the GPU follow-up it would have bought is not started from SM157.

- Receptance plus removal key (`rk`) had a median rank-2 cost reduction of 57% over the 8 informative Q2 cells (81% over the 5 informative T24 cells, information only), well above the one-third bar. It failed on a single cell: on Small two-head (SM118) instant, plain Q2 was already below int8 (-0.023%) and Q2 with `rk` was +0.003%, so the weighted label was +0.027% worse than plain (CI -0.027% to +0.047%), above the +0.01% limit.
- Receptance alone (`r`) had a median reduction of 39% but was worse than plain on Big two-head (SM91s2) curve and Small two-head (SM118) instant.
- The discounted Gramian (`g`) made things worse: median reduction -25% on Q2 and -84% on T24, worse than plain on five cells.

Plainly: the pointer fails, but `rk` looks promising on the head that binds the budgets. On Big two-head (SM91s2) instant, T24 costs +0.034% plain against +0.002% with `rk`; on Big Instant-Only (SM99) instant, +0.023% plain against +0.004% with `rk`. These are 8-user readings with wide intervals. Because the hypothesis came from this pointer, it gets its own pre-registered confirmation on the 732 set: output-aware T24 on the 732 set (SM158), with the M fitted here reused unchanged.

## Cost vs int8 rows per model, head and weight (8 users, 72,210 reviews per cell)

| Model | Head | Q2 plain | Q2 r | Q2 rk | Q2 g | T24 plain | T24 r | T24 rk | T24 g |
|---|---|---|---|---|---|---|---|---|---|
| Big two-head (SM91s2) | curve | +0.030% | +0.044% | -0.032% | +0.064% | +0.022% | +0.025% | -0.041% | +0.057% |
| Big two-head (SM91s2) | instant | +0.079% | +0.037% | +0.039% | +0.055% | +0.034% | +0.009% | +0.002% | +0.026% |
| Big Instant-Only (SM99) | instant | +0.063% | +0.047% | +0.043% | +0.058% | +0.023% | +0.007% | +0.004% | +0.018% |
| Small two-head (SM118) | curve | +0.011% | +0.012% | +0.004% | +0.049% | +0.008% | +0.022% | +0.005% | +0.043% |
| Small two-head (SM118) | instant | -0.023% | +0.011% | +0.003% | +0.035% | -0.021% | +0.002% | -0.004% | +0.019% |
| Small Instant-Only (SM113) | instant | +0.015% | -0.010% | +0.001% | +0.011% | -0.010% | -0.002% | -0.001% | -0.003% |
| six-layer Tiny two-head (SM130) | curve | +0.017% | -0.056% | -0.028% | +0.026% | -0.003% | -0.004% | -0.003% | +0.001% |
| six-layer Tiny two-head (SM130) | instant | +0.078% | +0.020% | +0.061% | +0.091% | +0.018% | +0.023% | +0.025% | +0.051% |
| six-layer Tiny Instant-Only (SM130) | instant | +0.066% | +0.066% | +0.066% | +0.087% | +0.032% | +0.027% | +0.045% | +0.060% |

### Receptance plus removal key (`rk`) with 95% CIs

| Model | Head | T24 plain | T24 rk | T24 rk vs T24 plain (paired) | Q2 rk vs Q2 plain (paired) |
|---|---|---|---|---|---|
| Big two-head (SM91s2) | curve | +0.022% [-0.002%, +0.044%] | -0.041% [-0.093%, -0.014%] | -0.063% [-0.129%, -0.024%] | -0.062% [-0.114%, -0.021%] |
| Big two-head (SM91s2) | instant | +0.034% [+0.020%, +0.047%] | +0.002% [-0.037%, +0.019%] | -0.032% [-0.074%, -0.006%] | -0.039% [-0.106%, -0.004%] |
| Big Instant-Only (SM99) | instant | +0.023% [+0.003%, +0.045%] | +0.004% [-0.033%, +0.029%] | -0.018% [-0.057%, +0.019%] | -0.020% [-0.060%, +0.010%] |
| Small two-head (SM118) | curve | +0.008% [-0.005%, +0.027%] | +0.005% [-0.008%, +0.023%] | -0.003% [-0.022%, +0.009%] | -0.007% [-0.035%, +0.010%] |
| Small two-head (SM118) | instant | -0.021% [-0.045%, +0.034%] | -0.004% [-0.028%, +0.036%] | +0.017% [-0.028%, +0.035%] | +0.027% [-0.027%, +0.047%] |
| Small Instant-Only (SM113) | instant | -0.010% [-0.027%, +0.008%] | -0.001% [-0.031%, +0.024%] | +0.009% [-0.015%, +0.025%] | -0.013% [-0.047%, +0.020%] |
| six-layer Tiny two-head (SM130) | curve | -0.003% [-0.042%, +0.047%] | -0.003% [-0.023%, +0.028%] | -0.000% [-0.035%, +0.045%] | -0.045% [-0.124%, +0.054%] |
| six-layer Tiny two-head (SM130) | instant | +0.018% [-0.011%, +0.096%] | +0.025% [-0.024%, +0.121%] | +0.006% [-0.025%, +0.047%] | -0.017% [-0.048%, +0.015%] |
| six-layer Tiny Instant-Only (SM130) | instant | +0.032% [-0.008%, +0.131%] | +0.045% [-0.028%, +0.136%] | +0.013% [-0.034%, +0.031%] | -0.000% [-0.054%, +0.034%] |

### Rule table

| Format | Weight | Informative cells | Median reduction | Cells worse than plain by more than +0.01% | Pass |
|---|---|---|---|---|---|
| Q2 | receptance (a) | 8 | 39% | Big two-head (SM91s2) curve, Small two-head (SM118) instant | no |
| Q2 | receptance + removal key (b) | 8 | 57% | Small two-head (SM118) instant | no |
| Q2 | discounted Gramian (c) | 8 | -25% | five cells | no |
| T24 | receptance (a) | 5 | 17% | Small two-head (SM118) curve and instant | no (information) |
| T24 | receptance + removal key (b) | 5 | 81% | Small two-head (SM118) instant, six-layer Tiny Instant-Only (SM130) instant | no (information) |
| T24 | discounted Gramian (c) | 5 | -84% | five cells | no (information) |

## M bytes and stability

M ships with the model, one K x K matrix per layer and head (stored bytes of the states are unchanged). If the fork ships `M^(1/2)` and `M^(-1/2)` instead of computing them at load, double these numbers.

| Model | Layers | M bytes (float16, full) | M bytes (float16, symmetric) | Top-2 eigen share r / rk / g | Split-half relative difference (median) r / rk / g |
|---|---|---|---|---|---|
| Big two-head (SM91s2) | 5 | 40,960 | 21,120 | 0.63 / 0.51 / 0.59 | 0.134 / 0.123 / 0.131 |
| Big Instant-Only (SM99) | 5 | 40,960 | 21,120 | 0.63 / 0.51 / 0.59 | 0.129 / 0.121 / 0.130 |
| Small two-head (SM118) | 3 | 6,144 | 3,264 | 0.74 / 0.60 / 0.77 | 0.098 / 0.108 / 0.082 |
| Small Instant-Only (SM113) | 3 | 6,144 | 3,264 | 0.77 / 0.63 / 0.78 | 0.073 / 0.082 / 0.071 |
| six-layer Tiny two-head (SM130) | 2 | 4,096 | 2,176 | 0.74 / 0.58 / 0.76 | 0.092 / 0.107 / 0.064 |
| six-layer Tiny Instant-Only (SM130) | 2 | 4,096 | 2,176 | 0.76 / 0.62 / 0.77 | 0.084 / 0.088 / 0.066 |

The fit used 100 training users (627,234 reviews), none in any evaluation pool, split into two halves of 50. The halves agree to about 7% to 13% (median relative difference of M); the minimum overlap of their top-2 eigen-subspaces for `rk` is 0.74 to 0.96 depending on the model.

## CPU time

- Probe scoring: 85.6 minutes of CPU process time (model plus preparation, all processes).
- Fit: 62.3 minutes of process time (two processes of 4 threads, about 31 minutes of wall time; per model and half, Big 477 s, Small 292 to 303 s, six-layer Tiny 135 to 186 s).
- Proofs 4.5 minutes and the 2-user pipeline fit 0.7 minutes.
- Total about 153 minutes of CPU process time.
- Codec cost per write (one thread, rank 2, int8 factors, 4 heads): K = 16 plain 20.4 us, weighted 22.6 us (+11%); K = 32 plain 36.5 us, weighted 54.0 us (+48%). The extra is the dense `S M^(1/2)` product (K^3); the factored form the fork would use needs 3,072 plus 2,048 multiply-adds per head and write for Big at rank 2.

## Recommendation

The pre-registered GPU follow-up of SM157 is not run (NO GO). The `rk` hypothesis is tested separately in output-aware T24 on the 732 set (SM158), pre-registered in `wt/sm158/artifacts/small-model/sm158/design.md` with its own rules (the 11-23 budgets plus a paired weighted against plain test).
