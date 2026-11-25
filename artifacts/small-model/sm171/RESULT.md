# State-energy penalty (SM171): result

Verdict: **NO GO**. Status decided, 723 rule users of the 732 set (721 after pass 1, 723 after the retry pass at cap 0.92), no non-finite row. Model: six-layer Tiny two-head (SM130). Statistic: review-weighted LogLoss ratio, paired user bootstrap, 2,000 draws, seed 0, 95% CI. Full tables: `sm171-result.md` / `sm171-result.json`.

## Question and rules

Does a short cool-down whose gradient also penalises the share of each card and note state's energy that a rank-2 truncation would drop (weight 1.0 and 0.1, against a weight-0 control) make rank 2 (Q2) cheaper against its own int8 (REF)? A penalty arm passes only if, on both heads (curve and instant), its Q2 cost is at least 0.05% lower than the control's with the CI upper bound below 0, and its REF is not worse than the control's REF by more than 0.05% (CI upper bound).

## Verdict per arm and head

| arm | head | Q2 cost change vs control | REF vs control REF | effect rule | REF guard |
|---|---|---|---|---|---|
| weight 1.0 | curve | -0.035% [-0.053, -0.009] | +0.303% [+0.266, +0.343] | fail | fail |
| weight 1.0 | instant | -0.069% [-0.079, -0.060] | +0.329% [+0.303, +0.354] | pass | fail |
| weight 0.1 | curve | -0.023% [-0.029, -0.017] | +0.037% [+0.027, +0.046] | fail | pass |
| weight 0.1 | instant | -0.040% [-0.045, -0.035] | +0.046% [+0.039, +0.052] | fail | fail |

No arm passes on either head, so the verdict is NO GO. The penalty does shrink the rank-2 cost, but the strong arm pays for it with a REF that is about 0.3% worse, and the weak arm shrinks it by less than the 0.05% minimum effect. The rank-2 format stays out; T24 (rank 2 below 16 answers, then rank 4) stays the chosen format.

## Costs against each arm's own REF (curve / instant)

| model | T24 | Q2 |
|---|---|---|
| weight 1.0 | +0.028% / +0.042% | +0.049% / +0.066% |
| weight 0.1 | +0.032% / +0.056% | +0.061% / +0.096% |
| control | +0.044% / +0.078% | +0.084% / +0.135% |
| original six-layer Tiny two-head (SM130), SM148 rows | +0.040% / +0.074% | +0.080% / +0.132% |

The control reproduces the original's costs, so the extra cool-down alone changes nothing. Penalty Q2 against the control's T24: weight 1.0 is +0.308% curve / +0.318% instant worse, weight 0.1 is +0.053% / +0.064% worse. Rank 2 after the penalty is therefore not as cheap as today's T24 once REF is counted. In the instant head the saving grows with the answer count (weight 1.0: -0.026% at 1-2 answers up to about -0.09% from 9 answers on; the 128+ bucket is too wide to read).

## Against the original model at REF (curve / instant)

Weight 1.0: REF +0.295% / +0.338%, T24 +0.324% / +0.380%, Q2 +0.344% / +0.404%. Weight 0.1: REF +0.029% / +0.054%, Q2 +0.089% / +0.150%. Control: REF -0.008% / +0.008%, Q2 +0.076% / +0.144%.

## Dropped share (energy.jsonl, mean of first 10 and last 10 lines; card / note)

| arm | start | end |
|---|---|---|
| weight 1.0 | 0.050 / 0.056 | 0.0076 / 0.0058 |
| weight 0.1 | 0.054 / 0.060 | 0.028 / 0.029 |
| control | 0.057 / 0.068 | 0.062 / 0.073 |

The penalty works as designed (the share falls by 85-90% at weight 1.0), but the loss pays for it at REF.

## Training and GPU time

| arm | reviews/s | minutes | peak allocated / reserved |
|---|---|---|---|
| weight 1.0 | 15,011 | 74.7 | 5,186 / 10,198 MiB |
| weight 0.1 | 15,088 | 74.4 | 5,186 / 10,198 MiB |
| control | 55,723 | 20.1 | 4,216 / 8,354 MiB |

GPU minutes by ticket: training 169.2, integrity ticket 0.2, scoring pass 1 69.7, retry pass 1.0; about 240 in total (plus the 3-minute probe and 2-minute smoke tickets). Integrity: the original REF on the 3 smoke users equals the SM148 rows exactly (maximum difference 0).

## "0 decided, 2 pending"

That text in the launcher log is only the legend of the report's exit codes: exit 0 means decided, exit 2 means pending (coverage below 720 users or a non-finite row). The report exited 0, so the status is decided; nothing is pending.
