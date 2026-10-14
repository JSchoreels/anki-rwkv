# Decay-only from epoch 10 (0.25 epoch, 7e-4 -> 0, shape one_minus_sin, over all 145,578,259 reviews)

300 held-out users (quick); control = the source epoch's scores of the same users.

| | Benchmark curve LogLoss (weighted / user avg) | Benchmark instant LogLoss (weighted / user avg) | R01 curve / instant (label-weighted) |
|---|---|---|---|
| source epoch | 0.36128 / 0.33635 | 0.34473 / 0.31055 | 0.36039 / 0.34492 |
| decayed | 0.35906 / 0.33456 | 0.34327 / 0.30924 | 0.35808 / 0.34327 |

Paired per-user benchmark LogLoss delta (decayed - source): curve -0.00180 (SE 0.00027, 211/300 improved); instant -0.00131 (SE 0.00012, 225/300 improved).
Training: 145,578,259 reviews in 823 s, 176,906 reviews/s, 3460 updates.
