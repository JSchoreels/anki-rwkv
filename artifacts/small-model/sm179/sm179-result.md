# Session inputs (SM179): result tables

**Verdict: GO** (decided; 723 rule users of the 732 set; label REF w8+s8hx8@e2s-query).

Integrity (control rescored here vs SM171's rows): 723 users, max |LogLoss difference| 0.0, OK.

| arm | curve vs control | instant vs control | go |
|---|---|---|---|
| session inputs (6 columns) | +0.001% [-0.002, +0.003] | -0.065% [-0.073, -0.058] | yes (instant) |
| card-state inputs (4 one-hot columns) | -0.003% [-0.006, -0.001] | -0.008% [-0.011, -0.006] | no |

| model | reviews/s | minutes | peak allocated / reserved MiB |
|---|---|---|---|
| control (SM171's control cool-down, no new input) | 55,723 | 20.1 | 4,216 / 8,354 |
| session inputs (6 columns) | 63,423 | 17.7 | 4,223 / 8,368 |
| card-state inputs (4 one-hot columns) | 63,872 | 17.6 | 4,220 / 8,358 |
