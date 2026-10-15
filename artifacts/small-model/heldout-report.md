# Small-model held-out report (full history, 360 fork-unseen validation users)

Reference for Δ%: `fork/w8+s8`. Review-weighted over benchmark test reviews; 95% user-bootstrap CI.

| Label | Params | Weight bytes | Curve LogLoss | Δ% [CI] | Instant LogLoss | Δ% [CI] | Curve RMSE(bins) | Δ% | Instant RMSE(bins) | Δ% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fork/fp32 | 2,762,884 | 11,051,536 | 0.29870 | +0.00% [-0.00, +0.01] | 0.27475 | -0.01% [-0.01, -0.01] | 0.03844 | +0.0% | 0.02206 | -0.3% |
| fork/w8+s8 | 2,762,884 | 2,995,488 | 0.29869 | +0.00% [+0.00, +0.00] | 0.27478 | +0.00% [+0.00, +0.00] | 0.03842 | +0.0% | 0.02212 | +0.0% |
| decay/fp32 | 192,800 | 771,200 | 0.31414 | +5.17% [+4.47, +6.03] | 0.29276 | +6.55% [+6.10, +7.00] | 0.05297 | +37.9% | 0.03899 | +76.3% |
| decay/w8+s8 | 192,800 | 245,696 | 0.31414 | +5.17% [+4.47, +6.03] | 0.29282 | +6.57% [+6.12, +7.02] | 0.05297 | +37.9% | 0.03919 | +77.2% |
| fullrun-e10/fp32 | 192,800 | 771,200 | 0.31518 | +5.52% [+4.86, +6.30] | 0.29382 | +6.93% [+6.48, +7.41] | 0.05462 | +42.2% | 0.03969 | +79.4% |
| w32-o0/fp32 | 192,800 | 771,200 | 0.32870 | +10.05% [+9.04, +11.23] | 0.31310 | +13.95% [+12.97, +14.95] | 0.06739 | +75.4% | 0.06488 | +193.3% |
| w64-o0/fp32 | 331,511 | 1,326,044 | 0.32814 | +9.86% [+8.75, +11.25] | 0.30539 | +11.14% [+10.18, +12.10] | 0.06746 | +75.6% | 0.05322 | +140.6% |
| w64kd-o0/fp32 | 331,511 | 1,326,044 | 0.32494 | +8.79% [+7.78, +9.93] | 0.29812 | +8.49% [+7.82, +9.18] | 0.06867 | +78.7% | 0.04530 | +104.8% |

## Paired screen: w64-vs-w32 (k = 1, 410 users)

| Head | Deltas (cand - base) | Mean | Band (full-history sigma) | Band (window sigma) | Same sign | Verdict |
|---|---|---:|---:|---:|---|---|
| curve | -0.00063 | -0.00063 | nan | 0.00622 | True | inside band |
| instant | -0.00765 | -0.00765 | nan | 0.00272 | True | inside band |

## Paired screen: w64kd-vs-w32 (k = 1, 410 users)

| Head | Deltas (cand - base) | Mean | Band (full-history sigma) | Band (window sigma) | Same sign | Verdict |
|---|---|---:|---:|---:|---|---|
| curve | -0.00313 | -0.00313 | nan | 0.00622 | True | inside band |
| instant | -0.01467 | -0.01467 | nan | 0.00272 | True | inside band |

## Paired screen: kd-vs-w64 (k = 1, 410 users)

| Head | Deltas (cand - base) | Mean | Band (full-history sigma) | Band (window sigma) | Same sign | Verdict |
|---|---|---:|---:|---:|---|---|
| curve | -0.00250 | -0.00250 | nan | 0.00622 | True | inside band |
| instant | -0.00703 | -0.00703 | nan | 0.00272 | True | inside band |
