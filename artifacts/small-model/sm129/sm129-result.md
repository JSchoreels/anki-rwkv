# Compact state for Tiny (SM129): result

Run 01:00 to 02:08 Thu 11-19 (launcher started 22:57 Wed 11-18, waited for the break). 4 scoring tickets (8 ticket lines, the second pass of each is a no-op), 64 GPU minutes. No training. Show-time inputs, leak-free, 722 rule users of the 732 set.

## Verdict

- Compact state (compact int state test (SM47) format, `w8+s4c8nhx8`): **go** (two-head and Instant-Only both pass).
- Card-only compact state (second integer-state test (SM49) arm C', `w8+s4hgx4c8hx8n`): **go** (two-head and Instant-Only both pass).

## Takeaway

The compact state format of the compact int state test (SM47) costs only +0.016% curve / +0.021% instant against the published Tiny format for the show-time two-head cool-down for Tiny (SM124), and about 0% against the fork default int4-card format (+0.000% curve, +0.001% instant). It cuts card + note state from 2,622 to 2,118 bytes (-19%). The card-only format reaches 1,934 bytes (-26%) for +0.16% curve / +0.21% instant against the fork default (+0.28% for the longer cool-down for Tiny (SM115), Instant-Only). Every arm stays within +1% of Small on every head (largest CI upper bound +0.542%).

## two-head: the show-time two-head cool-down for Tiny (SM124) vs the show-time two-head cool-down for Small (SM118)

| format | head | Tiny LogLoss | vs Small (published), % [CI] | cost vs Tiny published | cost vs Tiny fork default | vs Small at fork w8+s4c8n | pass |
|---|---|---|---|---|---|---|---|
| published w8+s8 | curve | 0.300369 | +0.118% [+0.057, +0.174] | 0 | | | |
| published w8+s8 | instant | 0.277105 | +0.183% [+0.164, +0.204] | 0 | | | |
| compact state (SM47 format) | curve | 0.300418 | +0.134% [+0.072, +0.191] | +0.016% [+0.012, +0.021] | +0.000% [-0.001, +0.001] | +0.120% [+0.059, +0.174] | yes |
| compact state (SM47 format) | instant | 0.277165 | +0.204% [+0.184, +0.227] | +0.021% [+0.018, +0.026] | +0.001% [+0.000, +0.002] | +0.186% [+0.165, +0.207] | yes |
| card-only compact state (SM49 arm C') | curve | 0.300904 | +0.296% [+0.239, +0.347] | +0.178% [+0.153, +0.205] | +0.162% [+0.140, +0.186] | +0.282% [+0.226, +0.331] | yes |
| card-only compact state (SM49 arm C') | instant | 0.277742 | +0.413% [+0.380, +0.446] | +0.230% [+0.210, +0.251] | +0.210% [+0.192, +0.228] | +0.394% [+0.363, +0.426] | yes |
| fork default (int4 cards) | curve | 0.300417 | +0.134% [+0.072, +0.190] | +0.016% [+0.011, +0.021] |  | +0.119% [+0.059, +0.174] |  |
| fork default (int4 cards) | instant | 0.277161 | +0.203% [+0.183, +0.225] | +0.020% [+0.016, +0.024] |  | +0.184% [+0.164, +0.206] |  |

## instant-only: the longer cool-down for Tiny (SM115) vs the longer Small Instant-Only model (SM113)

| format | head | Tiny LogLoss | vs Small (published), % [CI] | cost vs Tiny published | cost vs Tiny fork default | vs Small at fork w8+s4c8n | pass |
|---|---|---|---|---|---|---|---|
| published w8+s8 | instant | 0.276446 | +0.186% [+0.166, +0.208] | 0 | | | |
| compact state (SM47 format) | instant | 0.276542 | +0.221% [+0.199, +0.245] | +0.035% [+0.030, +0.040] | +0.001% [-0.000, +0.002] | +0.197% [+0.176, +0.220] | yes |
| card-only compact state (SM49 arm C') | instant | 0.277318 | +0.502% [+0.467, +0.542] | +0.316% [+0.292, +0.341] | +0.282% [+0.262, +0.303] | +0.478% [+0.445, +0.515] | yes |
| fork default (int4 cards) | instant | 0.276539 | +0.220% [+0.198, +0.244] | +0.034% [+0.029, +0.039] |  | +0.196% [+0.176, +0.219] |  |

## Serialized bytes, Tiny (fork layout, sm129-bytes.json; the six-layer Tiny has the same card and note streams)

| format | card | note | card + note |
|---|---|---|---|
| w8+s8 | 1,567 | 1,567 | 3,134 |
| w8+s4c8n | 1,055 | 1,567 | 2,622 |
| w8+s4c8nhx8 | 803 | 1,315 | 2,118 |
| w8+s4hgx4c8hx8n | 619 | 1,315 | 1,934 |
