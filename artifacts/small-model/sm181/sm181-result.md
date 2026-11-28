# Session inputs confirmation (SM181): result tables

**Verdict: CONFIRMED** (decided; label REF w8+s8hx8@e2s-query).

| axis | rule users | curve vs control | instant vs control | confirmed | integrity |
|---|---|---|---|---|---|
| six-layer Tiny two-head (SM130), decisive users (ids >= 5000 of the 1,500 pool) | 761 | -0.000% [-0.003, +0.002] | -0.063% [-0.071, -0.056] | yes | OK |
| Small two-head (SM118), 732 set | 723 | +0.003% [-0.001, +0.007] | -0.065% [-0.073, -0.057] | yes | OK |
| six-layer Tiny two-head (SM130), 732 set (SM179's models rescored with the dump; report only) | 723 | +0.001% [-0.002, +0.003] | -0.065% [-0.073, -0.058] | yes | OK |

Report only, six-layer Tiny two-head (SM130) on the 732 set and the decisive users together (1484 users): curve +0.000% [-0.002, +0.002], instant -0.064% [-0.069, -0.059].

Secondary metric, report only (not part of the rule): AUC, arm minus control, paired user bootstrap.

| axis | head | users | overall AUC control -> arm | overall change | within-user AUC control -> arm | within-user change |
|---|---|---|---|---|---|---|
| decisive | curve | 761 (within-user 755) | 0.8400 -> 0.8400 | -0.0000 [-0.0000, +0.0000] (-0.000%) | 0.7877 -> 0.7877 | +0.0000 [-0.0000, +0.0001] (+0.005%) |
| decisive | instant | 761 (within-user 755) | 0.8670 -> 0.8672 | +0.0002 [+0.0002, +0.0003] (+0.026%) | 0.8391 -> 0.8395 | +0.0004 [+0.0003, +0.0004] (+0.042%) |
| small | curve | 723 (within-user 722) | 0.8467 -> 0.8467 | -0.0000 [-0.0000, +0.0000] (-0.000%) | 0.7978 -> 0.7978 | -0.0000 [-0.0000, +0.0000] (-0.002%) |
| small | instant | 723 (within-user 722) | 0.8736 -> 0.8738 | +0.0002 [+0.0002, +0.0002] (+0.024%) | 0.8436 -> 0.8440 | +0.0003 [+0.0003, +0.0004] (+0.038%) |
| tiny732 | curve | 723 (within-user 722) | 0.8450 -> 0.8450 | -0.0000 [-0.0000, +0.0000] (-0.000%) | 0.7951 -> 0.7951 | -0.0000 [-0.0000, +0.0000] (-0.001%) |
| tiny732 | instant | 723 (within-user 722) | 0.8719 -> 0.8721 | +0.0002 [+0.0002, +0.0002] (+0.025%) | 0.8418 -> 0.8421 | +0.0003 [+0.0002, +0.0003] (+0.034%) |

| model | reviews/s | minutes | peak allocated / reserved MiB |
|---|---|---|---|
| sm149-sm118-ctl | 28,883 | 38.8 | 7,269 / 12,960 |
| sm181-small-ses | 28,779 | 38.9 | 7,281 / 12,970 |
