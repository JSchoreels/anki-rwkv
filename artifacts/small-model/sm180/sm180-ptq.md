# 4-bit weights (SM180): PTQ read-out

Status **decided**. Rule: int4 vs the same model's int8 weights (same state format), CI upper bound <= +0.15% per head. Users: the 723-user list of the 732 set.

| model | head | w4g32 vs int8 | free | w4g16 vs int8 | free |
|---|---|---|---|---|---|
| Big two-head (SM91s2) | curve | +0.092% [+0.051, +0.140] | yes | +0.057% [+0.022, +0.097] | yes |
| Big two-head (SM91s2) | instant | +0.243% [+0.222, +0.265] | no | +0.156% [+0.133, +0.187] | no |
| Big Instant-Only (SM99) | instant | +0.456% [+0.409, +0.506] | no | +0.242% [+0.203, +0.300] | no |
| Small two-head (SM118) | curve | +0.201% [+0.144, +0.265] | no | +0.157% [+0.113, +0.206] | no |
| Small two-head (SM118) | instant | +1.009% [+0.948, +1.073] | no | +1.050% [+0.952, +1.157] | no |
| Small Instant-Only (SM113) | instant | +0.562% [+0.518, +0.615] | no | +0.797% [+0.726, +0.876] | no |
| six-layer Tiny two-head (SM130) | curve | +0.260% [+0.218, +0.304] | no | +0.227% [+0.183, +0.268] | no |
| six-layer Tiny two-head (SM130) | instant | +0.839% [+0.788, +0.890] | no | +0.254% [+0.235, +0.274] | no |
| six-layer Tiny Instant-Only (SM130) | instant | +0.635% [+0.594, +0.682] | no | +0.450% [+0.401, +0.510] | no |

| size | w4g32 | w4g16 | verdict | QAT target |
|---|---|---|---|---|
| Big | NOT FREE | NOT FREE | PTQ fails | - |
| Small | NOT FREE | NOT FREE | PTQ fails | Small two-head (SM118) |
| Tiny | NOT FREE | NOT FREE | PTQ fails | six-layer Tiny two-head (SM130) |

| model | int8 bytes | w4g32 bytes (saved) | w4g16 bytes (saved) |
|---|---|---|---|
| Big two-head (SM91s2) | 2,995,488 | 1,891,088 (36.9%) | 2,216,976 (26.0%) |
| Big Instant-Only (SM99) | 2,995,488 | 1,891,088 (36.9%) | 2,216,976 (26.0%) |
| Small two-head (SM118) | 410,459 | 272,347 (33.6%) | 310,459 (24.4%) |
| Small Instant-Only (SM113) | 410,459 | 272,347 (33.6%) | 310,459 (24.4%) |
| six-layer Tiny two-head (SM130) | 226,191 | 149,287 (34.0%) | 170,807 (24.5%) |
| six-layer Tiny Instant-Only (SM130) | 226,191 | 149,287 (34.0%) | 170,807 (24.5%) |
