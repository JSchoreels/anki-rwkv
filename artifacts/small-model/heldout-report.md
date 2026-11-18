# Small-model held-out report (full history, 464 fork-unseen validation users)

Reference for Δ%: `fork/w8+s8`. Review-weighted over benchmark test reviews; 95% user-bootstrap CI.

| Label | Params | Weight bytes | Curve LogLoss | Δ% [CI] | Instant LogLoss | Δ% [CI] | Curve RMSE(bins) | Δ% | Instant RMSE(bins) | Δ% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fork/w8+s8 | 2,762,884 | 2,995,488 | 0.29766 | +0.00% [+0.00, +0.00] | 0.27216 | +0.00% [+0.00, +0.00] | 0.03867 | +0.0% | 0.02084 | +0.0% |
| s-w64kd-gate-ctx2048-dec6/w8+s8 | 331,511 | 410,459 | 0.30291 | +1.76% [+1.30, +2.37] | 0.27714 | +1.83% [+1.67, +1.98] | 0.04323 | +11.8% | 0.02418 | +16.0% |
| cal-ctx4096-e6/w8+s8 | 331,511 | 410,459 | 0.30265 | +1.68% [+1.22, +2.30] | 0.27677 | +1.69% [+1.55, +1.84] | 0.04280 | +10.7% | 0.02354 | +12.9% |

## Paired screen: pairs (LogLoss, k = 1, 464 users)

| Head | Deltas (cand - base) | Mean | Band (full-history sigma) | Band (window sigma) | Same sign | Verdict |
|---|---|---:|---:|---:|---|---|
| curve | -0.00025 | -0.00025 | nan | 0.00622 | True | inside band |
| instant | -0.00037 | -0.00037 | nan | 0.00272 | True | inside band |
