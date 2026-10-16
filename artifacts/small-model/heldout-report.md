# Small-model held-out report (full history, 464 fork-unseen validation users)

Reference for Δ%: `fork/w8+s8`. Review-weighted over benchmark test reviews; 95% user-bootstrap CI.

| Label | Params | Weight bytes | Curve LogLoss | Δ% [CI] | Instant LogLoss | Δ% [CI] | Curve RMSE(bins) | Δ% | Instant RMSE(bins) | Δ% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fork/w8+s8 | 2,762,884 | 2,995,488 | 0.29766 | +0.00% [+0.00, +0.00] | 0.27216 | +0.00% [+0.00, +0.00] | 0.03867 | +0.0% | 0.02084 | +0.0% |
| s-w64kd-gate-ctx2048-dec/w8+s8 | 331,511 | 410,459 | 0.30408 | +2.16% [+1.62, +2.90] | 0.27804 | +2.16% [+1.97, +2.35] | 0.04441 | +14.8% | 0.02481 | +19.0% |
| s-w64kd-gate-ctx2048-e4/w8+s8 | 331,511 | 410,459 | 0.30542 | +2.61% [+2.05, +3.33] | 0.27891 | +2.48% [+2.29, +2.69] | 0.04648 | +20.2% | 0.02678 | +28.5% |
