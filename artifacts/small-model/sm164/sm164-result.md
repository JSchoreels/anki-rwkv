# Release-candidate re-check of the final state formats (SM164): result

Status: **decided**. Rows: SM148 (REF, T24) and SM152 (F6) on the 732 set, SM160 (REF, T24) on 100 users with id >= 5000, this run for the rest of the ids >= 5000 part of the 1,500 pool. Statistic: % change of the benchmark LogLoss vs the baseline, review-weighted, paired user bootstrap 2,000 draws, 95% CI.

## Verdict per size

| size | final label | verdict |
|---|---|---|
| Big | `F6 (w8+lr2.16.4q6x8)` | **GO** |
| Small | `F6 (w8+lr2.16.4q6x8)` | **GO** |
| Tiny | `T24 (w8+lr2.16.4q8x8)` | **GO** |

## Per model and head (budget = CI upper bound)

732 rule: the final label vs the baseline on the 722 rule users. 1,500 rule: Big vs its own int8 on the 1,500 set; Small / Tiny: upper(own int8 vs Big int8, 732 set) + upper(final label vs own int8, 1,500 set).

| model | head | baseline | ceiling | 732 rule | own int8 vs Big int8 (732) | vs own int8 (1,500) | 1,500 rule upper | verdict |
|---|---|---|---|---|---|---|---|---|
| Big two-head (SM91s2) | curve | own int8 | +0.15% | +0.043% [+0.032, +0.056] | - | +0.048% [+0.041, +0.055] (n=1472) | +0.055% | PASS |
| Big two-head (SM91s2) | instant | own int8 | +0.15% | +0.069% [+0.063, +0.076] | - | +0.073% [+0.068, +0.077] (n=1472) | +0.077% | PASS |
| Big Instant-Only (SM99) | instant | own int8 | +0.15% | +0.074% [+0.067, +0.081] | - | +0.078% [+0.073, +0.084] (n=1471) | +0.084% | PASS |
| Small two-head (SM118) | curve | Big two-head (SM91s2) at int8 | +0.5% | +0.246% [+0.088, +0.409] | +0.213% [+0.057, +0.374] | +0.033% [+0.029, +0.038] (n=1483) | +0.412% | PASS |
| Small two-head (SM118) | instant | Big two-head (SM91s2) at int8 | +0.5% | +0.211% [+0.185, +0.238] | +0.151% [+0.126, +0.176] | +0.062% [+0.058, +0.067] (n=1483) | +0.243% | PASS |
| Small Instant-Only (SM113) | instant | Big Instant-Only (SM99) at int8 | +0.5% | +0.081% [+0.056, +0.106] | +0.016% [-0.010, +0.042] | +0.069% [+0.064, +0.074] (n=1483) | +0.116% | PASS |
| six-layer Tiny two-head (SM130) | curve | Big two-head (SM91s2) at int8 | +1.0% | +0.722% [+0.548, +0.899] | +0.681% [+0.505, +0.861] | +0.042% [+0.036, +0.047] (n=1484) | +0.908% | PASS |
| six-layer Tiny two-head (SM130) | instant | Big two-head (SM91s2) at int8 | +1.0% | +0.746% [+0.693, +0.802] | +0.672% [+0.620, +0.727] | +0.073% [+0.068, +0.079] (n=1484) | +0.806% | PASS |
| six-layer Tiny Instant-Only (SM130) | instant | Big Instant-Only (SM99) at int8 | +1.0% | +0.668% [+0.619, +0.717] | +0.579% [+0.533, +0.625] | +0.089% [+0.083, +0.095] (n=1483) | +0.720% | PASS |

## Report only

| model | head | vs own int8, 732 | vs own int8, ids >= 5000 | direct vs Big int8, 1,500 | direct vs Big int8, ids >= 5000 | own int8 vs Big int8, ids >= 5000 |
|---|---|---|---|---|---|---|
| Big two-head (SM91s2) | curve | +0.043% [+0.032, +0.056] | +0.052% [+0.044, +0.059] | n/a | n/a | n/a |
| Big two-head (SM91s2) | instant | +0.069% [+0.063, +0.076] | +0.075% [+0.069, +0.082] | n/a | n/a | n/a |
| Big Instant-Only (SM99) | instant | +0.074% [+0.067, +0.081] | +0.082% [+0.074, +0.090] | n/a | n/a | n/a |
| Small two-head (SM118) | curve | +0.034% [+0.027, +0.041] | +0.033% [+0.028, +0.039] | +1.005% [+0.835, +1.198] | +1.634% [+1.374, +1.963] | +1.600% [+1.341, +1.924] |
| Small two-head (SM118) | instant | +0.061% [+0.056, +0.067] | +0.063% [+0.057, +0.069] | +0.354% [+0.327, +0.383] | +0.471% [+0.433, +0.511] | +0.406% [+0.370, +0.444] |
| Small Instant-Only (SM113) | instant | +0.067% [+0.060, +0.073] | +0.070% [+0.063, +0.078] | +0.235% [+0.211, +0.261] | +0.362% [+0.330, +0.398] | +0.292% [+0.262, +0.327] |
| six-layer Tiny two-head (SM130) | curve | +0.040% [+0.031, +0.049] | +0.042% [+0.036, +0.049] | +1.479% [+1.296, +1.692] | +2.106% [+1.823, +2.454] | +2.061% [+1.780, +2.408] |
| six-layer Tiny two-head (SM130) | instant | +0.074% [+0.067, +0.082] | +0.073% [+0.066, +0.081] | +0.875% [+0.826, +0.924] | +0.979% [+0.909, +1.055] | +0.904% [+0.836, +0.978] |
| six-layer Tiny Instant-Only (SM130) | instant | +0.089% [+0.081, +0.099] | +0.089% [+0.081, +0.097] | +0.800% [+0.758, +0.844] | +0.908% [+0.846, +0.980] | +0.817% [+0.758, +0.886] |

The users with id >= 5000 are training users of the Big models: a direct Small or Tiny vs Big comparison there mostly measures Big's in-sample advantage (last column), so it is reported, never a verdict.
