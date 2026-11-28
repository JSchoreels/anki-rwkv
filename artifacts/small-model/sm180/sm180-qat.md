# 4-bit weights (SM180): QAT read-out

Status **decided**. Rule: arm at w4g32 vs control at w8, CI upper <= ceiling per head (+0.15%).

| model | head | arm w4g32 vs control int8 (rule) | free | arm w4g32 vs control w4g32 | arm int8 vs control int8 | arm w4g32 vs original int8 | control int4 (PTQ) vs control int8 |
|---|---|---|---|---|---|---|---|
| six-layer Tiny two-head (SM130) | curve | +0.085% [+0.055, +0.114] | yes | -0.203% [-0.241, -0.168] | +0.113% [+0.088, +0.139] | +0.082% [+0.025, +0.127] | +0.289% [+0.241, +0.335] |
| six-layer Tiny two-head (SM130) | instant | +0.190% [+0.177, +0.204] | no | -0.372% [-0.411, -0.338] | +0.267% [+0.247, +0.292] | +0.202% [+0.189, +0.215] | +0.565% [+0.530, +0.602] |
| Small two-head (SM118) | curve | +0.083% [+0.038, +0.131] | yes | -0.094% [-0.130, -0.056] | +0.095% [+0.066, +0.127] | +0.107% [+0.062, +0.166] | +0.177% [+0.115, +0.249] |
| Small two-head (SM118) | instant | +0.103% [+0.094, +0.114] | yes | -0.616% [-0.667, -0.568] | +1.377% [+1.264, +1.504] | +0.108% [+0.097, +0.119] | +0.724% [+0.673, +0.777] |

- six-layer Tiny two-head (SM130): **QAT not free**
- Small two-head (SM118): **QAT free**
