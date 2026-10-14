# Held-out evaluation B: interim summary (chunks 1+2)

INTERIM. Chunks 3 and 4 are queued as tickets 000321/000322; the final aggregation follows 000322. Aggregates only.

## Scope

409 paired users, 27,036,501 reviews, 19,284,495 test reviews, 6 variants, 0 errors (tickets 000309/000310).

## Review-weighted delta vs fp32 (95% bootstrap CI over users)

| Variant | LogLoss curve | LogLoss instant | RMSE(bins) curve | RMSE(bins) instant |
|---|---:|---:|---:|---:|
| fp32 | reference 0.28059 | reference 0.25508 | 0.03927 | 0.01985 |
| fp32+s8 | -0.000% | +0.000% | -0.000% | +0.011% |
| w8-pc+s8 | -0.000% | +0.009% | -0.04% | +0.33% |
| w4-g32+s8 | +0.087% [+0.053, +0.119] | +0.461% [+0.410, +0.520] | +0.735% | +21.2% (0.01985 -> 0.02406) |
| w4-g64+s8 | +0.061% | +0.642% | +0.43% | +31.2% |
| w4-g32+s4 | +0.119% | +0.533% | +0.93% | +22.2% |

w4-g32+s8 by history length (LogLoss curve / instant):
- 20k reviews or fewer (153 users): +0.045% / +0.498%
- over 20k reviews (256 users): +0.089% / +0.459%

## Notes

- UM+ and per-prediction max/p99 |dp| are not available: the source data holds per-user aggregates only.
- Int4 is a download/storage format decoded to f32 at load. The runtime default stays f32 weights + int8 card/note state + f32 deck/preset/global state.
- The instant RMSE(bins) increase (+21% for w4-g32+s8) is the main caveat.
