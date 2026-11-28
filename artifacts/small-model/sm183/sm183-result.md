# Tiny cool-down with session inputs (SM183): result tables

Status: decided; label REF w8+s8hx8@e2s-query; control reproduction check (G0): PASS (the release models are the controls).

**six-layer Tiny two-head (SM130): GO**
**six-layer Tiny Instant-Only (SM130): GO (732 set only; decisive users not scored)**

| model | set | control | rule users | curve vs control | instant vs control | passes | integrity |
|---|---|---|---|---|---|---|---|
| six-layer Tiny two-head (SM130) | 732 | sm130-th-release | 723 | -0.029% [-0.047, -0.008] | -0.499% [-0.578, -0.420] | yes | OK |
| six-layer Tiny two-head (SM130) | decisive | sm130-th-release | 761 | -0.040% [-0.051, -0.030] | -0.474% [-0.573, -0.396] | yes | OK |
| six-layer Tiny Instant-Only (SM130) | 732 | sm130-io-release | 723 | +0.001% [+0.001, +0.001] | -0.589% [-0.699, -0.484] | yes | OK |

Secondary metric, report only (not part of the rule): AUC, arm minus control, paired user bootstrap.

| model | set | head | users | overall AUC control -> arm | overall change | within-user AUC control -> arm | within-user change |
|---|---|---|---|---|---|---|---|
| six-layer Tiny two-head (SM130) | 732 | curve | 723 (within-user 722) | 0.8450 -> 0.8451 | +0.0001 [+0.0000, +0.0002] (+0.013%) | 0.7951 -> 0.7952 | +0.0002 [+0.0001, +0.0003] (+0.020%) |
| six-layer Tiny two-head (SM130) | 732 | instant | 723 (within-user 722) | 0.8719 -> 0.8737 | +0.0017 [+0.0014, +0.0020] (+0.197%) | 0.8418 -> 0.8439 | +0.0021 [+0.0018, +0.0024] (+0.250%) |
| six-layer Tiny two-head (SM130) | decisive | curve | 761 (within-user 755) | 0.8401 -> 0.8403 | +0.0002 [+0.0001, +0.0002] (+0.021%) | 0.7877 -> 0.7881 | +0.0003 [+0.0002, +0.0005] (+0.042%) |
| six-layer Tiny two-head (SM130) | decisive | instant | 761 (within-user 755) | 0.8670 -> 0.8688 | +0.0018 [+0.0014, +0.0022] (+0.203%) | 0.8392 -> 0.8415 | +0.0024 [+0.0021, +0.0027] (+0.283%) |
| six-layer Tiny Instant-Only (SM130) | 732 | curve | 723 (within-user 722) | 0.4568 -> 0.4568 | -0.0000 [-0.0000, -0.0000] (-0.001%) | 0.5409 -> 0.5409 | -0.0000 [-0.0000, +0.0000] (-0.000%) |
| six-layer Tiny Instant-Only (SM130) | 732 | instant | 723 (within-user 722) | 0.8726 -> 0.8746 | +0.0020 [+0.0016, +0.0024] (+0.231%) | 0.8427 -> 0.8449 | +0.0023 [+0.0019, +0.0027] (+0.272%) |

| training folder | reviews | reviews/s | minutes | peak allocated / reserved MiB | update offset end | resumed |
|---|---|---|---|---|---|---|
| sm183-th-ses-p1 | 581,942,724 | 64,866 | 149.5 | 4,224 / 9,202 | 728574 | no |
| sm130-th-dec2ep32k-p1 (release) | 581,942,724 | 64,547 | 150.3 | 4,209 / 9,132 | 728574 | no |
| sm183-th-ses-p2 | 581,942,724 | 64,479 | 150.4 | 4,226 / 10,160 | 750652 | no |
| sm130-th-dec2ep32k-p2 (release) | 581,942,724 | 64,303 | 150.8 | 4,211 / 9,996 | 750652 | no |
| sm183-io-ses-p1 | 581,942,724 | 64,833 | 149.6 | 4,224 / 9,202 | 927276 | no |
| sm130-io-dec2ep32k-p1 (release) | 581,942,724 | 65,967 | 147.0 | 4,209 / 9,132 | 927276 | no |
| sm183-io-ses-p2 | 581,942,724 | 64,853 | 149.6 | 4,226 / 10,160 | 949354 | no |
| sm130-io-dec2ep32k-p2 (release) | 581,942,724 | 65,706 | 147.6 | 4,211 / 9,996 | 949354 | no |
