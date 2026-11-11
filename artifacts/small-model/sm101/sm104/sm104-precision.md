# Tiny vs Small: accuracy cost of the compact state formats (SM104), instant head

80 users of the 732 set (the int4 drift check (SM78)'s 80 random users), 1,019,305 benchmark reviews, CPU, f32 weights, leak-free, show-time inputs. Cost = format minus f32 state of the same model, in % of that model's f32-state instant LogLoss; paired user bootstrap 2,000, seed 0, 95% CI.

| model | f32 state LogLoss | int8 cost, % [95% CI] | int4 cards + int8 notes cost, % [95% CI] |
|---|---|---|---|
| the tiny model (the smaller instant-only model, SM101) | 0.294372 | -0.0007 [-0.0025, +0.0006] | +0.0128 [+0.0032, +0.0225] |
| the Small instant model (the show-time instant-only model, stage 2, SM98) | 0.293658 | -0.0004 [-0.0009, +0.0002] | +0.0095 [+0.0010, +0.0178] |
| Tiny's cost minus Small's | | -0.0004 [-0.0022, +0.0011] | +0.0033 [-0.0051, +0.0127] |

Raw numbers, the curve head (untrained for both models, not used) and per-format LogLoss: `sm104-precision.json`.
