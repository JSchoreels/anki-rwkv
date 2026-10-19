# SM15 CPU screen: zero-shot depth cuts of P_dec (68 feature-screen users, w8+s8, review-weighted)

Parent P_dec/w8+s8: curve 0.31110, instant 0.28041. No fine-tuning: an upper bound of the post-fine-tune cost. Delta % vs the parent, 95% user-bootstrap CI in brackets.

| arm | params | curve LogLoss | curve delta | instant LogLoss | instant delta |
|---|---:|---:|---:|---:|---:|
| D9 (d2p2u2) | 244,232 | 0.31336 | +0.73% [+0.42, +1.02] | 0.28302 | +0.93% [+0.75, +1.18] |
| D7 (d2p1u1) | 204,782 | 0.32565 | +4.68% [+3.59, +6.04] | 0.29014 | +3.47% [+3.00, +3.99] |
| D7-nocmix | 179,645 | 0.33138 | +6.52% [+4.99, +8.61] | 0.29560 | +5.42% [+4.74, +6.14] |
| deck cut d2p3u3 | 283,682 | 0.31268 | +0.51% [+0.28, +0.84] | 0.28099 | +0.21% [+0.17, +0.25] |
| preset cut d4p1u3 | 292,061 | 0.31109 | -0.00% [-0.41, +0.24] | 0.28111 | +0.25% [+0.19, +0.30] |
| user cut d4p3u1 | 292,061 | 0.31515 | +1.30% [+0.92, +1.78] | 0.28275 | +0.83% [+0.68, +1.01] |

Share of D7's zero-shot LogLoss increase carried by each single-stream cut (curve / instant):

- deck: 11% / 6%
- preset: -0% / 7%
- user: 28% / 24%

**Chosen ARCH: w64-d2p1u1** (no single-stream cut accounts for > 60% of D7's zero-shot increase on either head: D7).
