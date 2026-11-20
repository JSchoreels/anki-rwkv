# Six-layer Tiny with the card-only state (SM134): decision

Rule (pre-registered): go(chain) iff every trained head's 95% CI upper bound of 100 x (card-only six-layer / Small published w8+s8 - 1) <= +1.00% on all 722 rule users with finite rows; verdict go iff both chains go. Show-time inputs, leak-free, the 732 set, paired user bootstrap 2,000, seed 0. % = 100 x (new / reference - 1) of the review-weighted LogLoss.

**Verdict: go**

| model | head | card-only LogLoss | card-only vs Small published % [CI] | go | card-only vs six-layer compact | card-only vs six-layer published | compact vs Small (SM130) | published vs Small (SM130) |
|---|---|---|---|---|---|---|---|---|
| the two-head six-layer Tiny (SM130) vs the show-time two-head cool-down for Small (SM118) | curve | 0.301994 | +0.659% [+0.585, +0.743] | yes | +0.175% [+0.150, +0.202] | +0.191% [+0.164, +0.222] | +0.483% [+0.415, +0.562] | +0.467% [+0.398, +0.546] |
| the two-head six-layer Tiny (SM130) vs the show-time two-head cool-down for Small (SM118) | instant | 0.278757 | +0.780% [+0.730, +0.833] | yes | +0.233% [+0.214, +0.255] | +0.255% [+0.233, +0.279] | +0.545% [+0.505, +0.589] | +0.524% [+0.484, +0.567] |
| the Instant-Only six-layer Tiny (SM130) vs the longer Small Instant-Only model (SM113) | instant | 0.278382 | +0.888% [+0.832, +0.943] | yes | +0.283% [+0.263, +0.305] | +0.322% [+0.296, +0.349] | +0.603% [+0.560, +0.647] | +0.564% [+0.522, +0.607] |

Tiny state bytes per card / note: w8+s8 1,567 / 1,567; w8+s4c8n 1,055 / 1,567; w8+s4c8nhx8 803 / 1,315; w8+s4hgx4c8hx8n 619 / 1,315

Unparseable row lines skipped: 0
