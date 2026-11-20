# Compact state for Small and Big (SM131): decision

Rule (pre-registered): free = every head's 95% CI upper bound of 100 x (compact / fork default - 1) < +0.05% on the rule users with finite rows in both formats (at least 718 of 722); verdict per size: free if both models are free. Show-time inputs, leak-free, the 732 set, paired user bootstrap 2,000, seed 0. % = 100 x (format / reference - 1) of the review-weighted LogLoss.

- **Small: compact state free** (the show-time two-head cool-down for Small (SM118) free, the longer Small Instant-Only model (SM113) free)
- **Big: compact state free** (the show-time big model (SM91s2) free, the show-time instant-only big model (SM99) free)

Unparseable row lines skipped: 0

| model | head | users | fork default LogLoss | compact LogLoss | compact vs fork default, % [CI] | free | compact vs published | fork default vs published | card-only vs fork default (report only) |
|---|---|---|---|---|---|---|---|---|---|
| the show-time two-head cool-down for Small (SM118) | curve | 722 | 0.300059 | 0.300059 | +0.000% [-0.001, +0.001] | yes | +0.015% [+0.011, +0.018] | +0.014% [+0.011, +0.018] | +0.161% [+0.133, +0.190] |
| the show-time two-head cool-down for Small (SM118) | instant | 722 | 0.276651 | 0.276654 | +0.001% [+0.000, +0.002] | yes | +0.020% [+0.016, +0.024] | +0.019% [+0.014, +0.023] | +0.198% [+0.180, +0.218] |
| the longer Small Instant-Only model (SM113) | instant | 722 | 0.275998 | 0.276 | +0.001% [+0.000, +0.002] | yes | +0.025% [+0.020, +0.029] | +0.024% [+0.020, +0.028] | +0.278% [+0.257, +0.301] |
| the show-time big model (SM91s2) | curve | 721 | 0.299475 | 0.299474 | -0.000% [-0.001, +0.000] | yes | +0.015% [+0.010, +0.021] | +0.015% [+0.010, +0.022] | +0.122% [+0.097, +0.151] |
| the show-time big model (SM91s2) | instant | 721 | 0.276262 | 0.276263 | +0.000% [-0.000, +0.001] | yes | +0.016% [+0.012, +0.021] | +0.016% [+0.012, +0.020] | +0.155% [+0.135, +0.177] |
| the show-time instant-only big model (SM99) | instant | 721 | 0.275975 | 0.275976 | +0.000% [-0.000, +0.001] | yes | +0.017% [+0.014, +0.021] | +0.017% [+0.013, +0.021] | +0.160% [+0.140, +0.182] |

Non-finite rows (amendment 11-19; users count as missing): sm91s2-big-showtime/w8+s4c8n@e2s-query users [3876]; sm99-big-inst-e2/w8+s4c8n@e2s-query users [3876]

the show-time big model (SM91s2): rule on 721 users; missing [3876]

the show-time instant-only big model (SM99): rule on 721 users; missing [3876]

## Serialized bytes (fork layout, sm131-bytes.json)

| size | format | card | note | card + note | card vs fork default | card + note vs fork default |
|---|---|---|---|---|---|---|
| Small | w8+s8 | 2,987 | 1,567 | 4,554 | +52.2% | +29.0% |
| Small | w8+s4c8n | 1,963 | 1,567 | 3,530 | +0.0% | +0.0% |
| Small | w8+s4c8nhx8 | 1,521 | 1,315 | 2,836 | -22.5% | -19.7% |
| Small | w8+s4hgx4c8hx8n | 1,185 | 1,315 | 2,500 | -39.6% | -29.2% |
| Big | w8+s8 | 15,415 | 10,283 | 25,698 | +66.3% | +31.4% |
| Big | w8+s4c8n | 9,271 | 10,283 | 19,554 | +0.0% | +0.0% |
| Big | w8+s4c8nhx8 | 7,747 | 9,267 | 17,014 | -16.4% | -13.0% |
| Big | w8+s4hgx4c8hx8n | 6,619 | 9,267 | 15,886 | -28.6% | -18.8% |
