# SM46: codebook state compression at ~6,000 / ~3,000 bits (`vq5902x3028`), SM36, w8 weights

Card 5,902 bits, note 3,028 bits per entity (int8 state: 26,624 / 14,336). Codebooks fitted after training on SM36's own states (k-means, no retraining). Full-history held-out protocol; review-weighted benchmark LogLoss; delta = vq - SM36 w8+s8; [ ] = 95% paired user bootstrap (2,000 resamples, seed 0, 2.5/97.5 percentiles); % vs the big model (fork w8+s8) on the same users; curve = ahead head.

## Pool 732: 458 of 732 users

| Head | Test reviews | vq LogLoss | SM36 w8+s8 | delta vs s8 | % vs s8 | vq % vs fork | s8 % vs fork | fork |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| curve | 19,833,530 | 0.29813 | 0.29804 | +0.00009 [+0.00006, +0.00012] | +0.030% [+0.021, +0.039] | +0.77% [+0.49, +1.15] | +0.74% [+0.46, +1.12] | 0.29584 |
| instant | 19,833,530 | 0.27412 | 0.27397 | +0.00015 [+0.00012, +0.00018] | +0.055% [+0.047, +0.063] | +0.70% [+0.62, +0.77] | +0.64% [+0.57, +0.72] | 0.27223 |

## Pool 464: 458 of 464 users

| Head | Test reviews | vq LogLoss | SM36 w8+s8 | delta vs s8 | % vs s8 | vq % vs fork | s8 % vs fork | fork |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| curve | 19,833,530 | 0.29813 | 0.29804 | +0.00009 [+0.00006, +0.00012] | +0.030% [+0.021, +0.039] | +0.77% [+0.49, +1.15] | +0.74% [+0.46, +1.12] | 0.29584 |
| instant | 19,833,530 | 0.27412 | 0.27397 | +0.00015 [+0.00012, +0.00018] | +0.055% [+0.047, +0.063] | +0.70% [+0.62, +0.77] | +0.64% [+0.57, +0.72] | 0.27223 |

- GPU scorer: 458 users, 1.53 h of model time, 5,005 reviews/s; path {'card': 'torch step loop, eigh float64', 'note': 'torch step loop, eigh float64'}.
- GPU check: passed True, fails [].
- Codebooks sha256 af46e6a253c4f18e349792527c815860006924c301f70c3924a4872c38cf81f7.
