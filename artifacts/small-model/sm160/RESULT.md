# Deploy-time ensemble of the two instant predictions (SM160): result

Branch `sm160`, read-out on 11-24 after the GPU chain finished at 15:26 ("SM160 GPU DONE"). Numbers come from `sm160-result.json` and `sm160-result.md` (made by `sm160_report.py report`).

## Question

In the fork's default mode each size runs two models on every review: the two-head model and the Instant-Only model. The instant prediction that is shown comes from the Instant-Only model alone; the two-head model's own instant prediction is thrown away. For each size, does the logit average `sigmoid(w * logit(p_Instant-Only) + (1 - w) * logit(p_two-head))` have a lower instant-head LogLoss than the Instant-Only model alone, at the shipped state format T24 (`w8+lr2.16.4q8x8@e2s-query`)? The averaging would cost nothing extra at deploy time. The sizes and models are: Big, with Big two-head (SM91s2) and Big Instant-Only (SM99); Small, with Small two-head (SM118) and Small Instant-Only (SM113); Tiny, with six-layer Tiny two-head (SM130) and six-layer Tiny Instant-Only (SM130).

## Rules (pre-registered in design.md)

One weight w per size was fitted on a grid of 0.00 to 1.00 in steps of 0.01, on 100 training users only (5,548,901 reviews, none of them in the 732 set), before any row of the 732 set existed. D is the review-weighted LogLoss change in % of the ensemble against the Instant-Only model alone on the instant head, with a paired user bootstrap of 2,000 draws (seed 0) and a 95% CI. A size gets a go only if all hold: (a) the CI upper bound of D is below 0 and D is at most -0.05%; (b) at least 715 rule users and no non-finite row for either model of the size; (c) every per-review dump reproduces its row's instant LogLoss within 2e-6. The decision label is T24; REF (`w8+s8hx8@e2s-query`, int8 card and note rows) is information only.

## Verdict per size (T24, the decision)

| Size | Fitted w (weight of the Instant-Only model) | Rule users | D (CI) | (a) | (b) | (c) | Go |
|---|---|---|---|---|---|---|---|
| Tiny: six-layer Tiny two-head (SM130) with six-layer Tiny Instant-Only (SM130) | 0.69 | 723 | -0.067% [-0.072, -0.061] | yes | yes | yes | **YES** |
| Small: Small two-head (SM118) with Small Instant-Only (SM113) | 0.78 | 723 | -0.022% [-0.024, -0.019] | no (gain below 0.05%) | yes | yes | NO |
| Big: Big two-head (SM91s2) with Big Instant-Only (SM99) | 0.80 | 718 | -0.012% [-0.014, -0.009] | no (gain below 0.05%) | no (see below) | yes | NO |

The Tiny ensemble is a go: it improves the instant head by 0.067%, the CI upper bound is below 0, and the gain clears the 0.05% bar. For Small and Big the ensemble is a real but tiny improvement (the CI excludes 0), too small to reach 0.05%, so the Instant-Only model alone stays.

Note on the script's overall line: `sm160_report.py` prints "pending" and exits with 2 because Big has 5 users (296, 2073, 2267, 3735, 4852) whose row is an out-of-memory error for both Big models in every pass, so Big has 718 usable users but 5 non-finite rows. These are the same 5 users that SM148 could not fit on the GPU. Rule (b) therefore fails for Big on a strict reading, but Big fails rule (a) anyway (D is -0.012%), so nothing pending changes the Big verdict. One small repair was needed: the script crashed when writing its JSON (a numpy boolean was not serializable), fixed with a `default=` handler on that one line of `sm160_report.py`; no number changed.

## Report-only lines (T24)

| Size | D at w = 0.5 | Probability average (its fitted w) | Oracle w on the evaluation users, D | Two-head instant head alone against Instant-Only | Ensemble against the two-head instant head |
|---|---|---|---|---|---|
| Big | +0.001% [-0.004, +0.007] | -0.012% (0.79) | 0.75: -0.012% | +0.102% [+0.090, +0.116] | -0.113% [-0.125, -0.103] |
| Small | +0.011% [+0.004, +0.017] | -0.021% (0.79) | 0.77: -0.022% | +0.235% [+0.217, +0.255] | -0.256% [-0.275, -0.238] |
| Tiny | -0.053% [-0.061, -0.044] | -0.066% (0.69) | 0.66: -0.067% | +0.184% [+0.165, +0.208] | -0.251% [-0.272, -0.233] |

The fitted weights are within 0.05 of the oracle weights, so the 100-user fit is a good estimate. The two-head model's own instant head is worse than the Instant-Only model for every size (by 0.10% to 0.24%), so the ensemble gain comes from combining two different errors, not from the two-head model being better. A plain 50/50 average is a loss for Small, neutral for Big and only helps Tiny, so the weight matters.

Aggregate % against the Instant-Only model alone as w goes from 0.0 to 1.0 in steps of 0.1 (T24, on the 732 evaluation users, which is what the script computes; w = 1.0 is today's deploy and is 0 by construction):

- Big: +0.102, +0.074, +0.049, +0.029, +0.013, +0.001, -0.007, -0.011, -0.012, -0.008, 0.
- Small: +0.235, +0.173, +0.119, +0.074, +0.038, +0.011, -0.009, -0.019, -0.021, -0.015, 0.
- Tiny: +0.184, +0.113, +0.054, +0.007, -0.029, -0.053, -0.065, -0.066, -0.056, -0.034, 0.

The training-user fit itself is in `gpu/wfit-T24.json`.

## REF (information only)

| Size | Fitted w | D (CI) | Oracle w, D |
|---|---|---|---|
| Big | 0.85 | -0.008% [-0.009, -0.006] | 0.79: -0.008% |
| Small | 0.82 | -0.014% [-0.016, -0.012] | 0.81: -0.014% |
| Tiny | 0.73 | -0.048% [-0.052, -0.043] | 0.69: -0.048% |

At REF the gains are a little smaller for every size, and Tiny would stay just below the 0.05% bar (-0.048%). The Big REF rows have the same 5 out-of-memory users.

## Caveats

- The Big weight (0.80) was fitted on training users that the fork's Big models had already seen, so it may lean toward whichever Big model fits seen users better. The oracle weight on the evaluation users (0.75) is close, and the Big verdict is NO on gain size either way.
- Only the instant head is changed; the curve head is untouched by construction.
- A go means only that the fork's default mode could show the ensemble for Tiny in a later, separate change. Nothing ships from this test.

## Pipeline checks

- Every dump reproduces its row's instant LogLoss: the largest difference is 5.0e-7 (limit 2e-6), at both labels and all sizes. Rule (c) holds.
- Cross-check against SM148 (information): the T24 and REF rows of `gpu/sm160-rows-732.jsonl` against `../sm148/gpu/sm148-rows-732.jsonl` agree on every user and model (no missing pair), and the benchmark-deleted and benchmark-existing fields are identical. The largest difference in any LogLoss field is 1.2e-4 (curve head of Big Instant-Only (SM99), user 3008); on the instant head it is 7.2e-5 (also Big Instant-Only (SM99)) and 3.0e-5 or less for the other five models. This is above the 1e-5 that the smoke check allowed, and looks like GPU run-to-run noise on the largest users; the dump-versus-row check above (5.0e-7) is the one that matters for the decision, and a difference of this size cannot move a D of -0.067%.

## GPU time

About 194 wall minutes in total (sum of the rounded ticket minutes in `gpu/tickets.jsonl`; the smoke tickets round to 0). T24: training users 15, Tiny 24, Small 32, Big 56. REF: training users 7, Tiny 15, Small 18, Big 27. The chain ran from 12:07 to 15:26 on 11-24. The CPU read-out took about 10 minutes of two cores per run.
