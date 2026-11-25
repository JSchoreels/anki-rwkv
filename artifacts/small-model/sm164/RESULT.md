# Release-candidate re-check of the final state formats (SM164): result

Status: **decided, all three sizes GO** (read-out 11-25, after "SM164 GPU DONE" at 02:40). Full tables: `sm164-result.md` and `sm164-result.json`.

## Question and rules

The final state formats are int6 factors on the low-rank states (F6, `w8+lr2.16.4q6x8`) for Big and Small, and the plain low-rank format (T24, `w8+lr2.16.4q8x8`) for the six-layer Tiny. The question: do the six release-candidate models stay inside the user's budgets on the 732 set (the standard read-out) and on the 1,500-user pool (the decisive check)? The statistic is the % change of the benchmark LogLoss (review-weighted, paired user bootstrap, 2,000 draws, 95% CI), judged by the CI upper bound per head.

- Big two-head (SM91s2) and Big Instant-Only (SM99): final format vs their own int8, upper bound at most +0.15%.
- Small: final format vs the matching Big model at int8, total upper bound at most +0.5%. Six-layer Tiny: at most +1.0%.
- On the 1,500 pool the users with id >= 5000 are training users of the Big models, so a direct Small or Tiny vs Big comparison there mostly measures Big's in-sample advantage. The Small / Tiny 1,500 rule is therefore split into its two valid parts and the two upper bounds added (conservative): upper(own int8 vs Big int8, 732 set) + upper(final format vs own int8, 1,500 set).
- A size is GO if every cell passes, NO if one fails, HOLD if only the 1,500 rule fails on a Small / Tiny cell.

## Verdict per size

| size | final format | verdict |
|---|---|---|
| Big | F6 | GO |
| Small | F6 | GO |
| Tiny (six-layer) | T24 | GO |

## Per model and head

| model | head | 732 rule (ceiling) | 1,500 rule upper (ceiling) | margin to ceiling |
|---|---|---|---|---|
| Big two-head (SM91s2) | curve | +0.043% [+0.032, +0.056] (+0.15%) | +0.055% (+0.15%) | 0.095% |
| Big two-head (SM91s2) | instant | +0.069% [+0.063, +0.076] | +0.077% | 0.073% |
| Big Instant-Only (SM99) | instant | +0.074% [+0.067, +0.081] | +0.084% | 0.066% |
| Small two-head (SM118) | curve | +0.246% [+0.088, +0.409] (+0.5%) | +0.412% = +0.374 + 0.038 (+0.5%) | 0.088% |
| Small two-head (SM118) | instant | +0.211% [+0.185, +0.238] | +0.243% | 0.257% |
| Small Instant-Only (SM113) | instant | +0.081% [+0.056, +0.106] | +0.116% | 0.384% |
| six-layer Tiny two-head (SM130) | curve | +0.722% [+0.548, +0.899] (+1.0%) | +0.908% = +0.861 + 0.047 (+1.0%) | 0.092% |
| six-layer Tiny two-head (SM130) | instant | +0.746% [+0.693, +0.802] | +0.806% | 0.194% |
| six-layer Tiny Instant-Only (SM130) | instant | +0.668% [+0.619, +0.717] | +0.720% | 0.280% |

Own final format vs own int8 on the 1,500 set (the second part of the rule, n about 1,472 to 1,484 users): Big +0.048% / +0.073% / +0.078%; Small +0.033% / +0.062% / +0.069%; six-layer Tiny +0.042% / +0.073% / +0.089% (order as in the table). The tightest cell per size is the curve head of its two-head model: Big +0.055% vs +0.15% (Big Instant-Only is +0.084%); Small two-head (SM118) curve +0.412% vs +0.5%; six-layer Tiny two-head (SM130) curve +0.908% vs +1.0%. The format adds only 0.03% to 0.09% on the 1,500 pool on top of what the 732 set already showed.

## Report-only lines (never a verdict)

Direct comparison with Big at int8 (final format), 1,500 set / ids >= 5000 alone, and own int8 vs Big int8 on the ids >= 5000 (Big's in-sample advantage):

| model | head | direct, 1,500 | direct, ids >= 5000 | own int8 vs Big int8, ids >= 5000 |
|---|---|---|---|---|
| Small two-head (SM118) | curve | +1.005% [+0.835, +1.198] | +1.634% [+1.374, +1.963] | +1.600% [+1.341, +1.924] |
| Small two-head (SM118) | instant | +0.354% [+0.327, +0.383] | +0.471% [+0.433, +0.511] | +0.406% [+0.370, +0.444] |
| Small Instant-Only (SM113) | instant | +0.235% [+0.211, +0.261] | +0.362% [+0.330, +0.398] | +0.292% [+0.262, +0.327] |
| six-layer Tiny two-head (SM130) | curve | +1.479% [+1.296, +1.692] | +2.106% [+1.823, +2.454] | +2.061% [+1.780, +2.408] |
| six-layer Tiny two-head (SM130) | instant | +0.875% [+0.826, +0.924] | +0.979% [+0.909, +1.055] | +0.904% [+0.836, +0.978] |
| six-layer Tiny Instant-Only (SM130) | instant | +0.800% [+0.758, +0.844] | +0.908% [+0.846, +0.980] | +0.817% [+0.758, +0.886] |

Almost all of the direct gap on the ids >= 5000 is Big's in-sample advantage (last column); the state format itself costs 0.03% to 0.09% (see above). Final format vs own int8 on the 732 set and on the ids >= 5000 alone agree to within 0.01% (full table in `sm164-result.md`).

## Warnings (copied, not rerun)

- Long cards (128+ answers) and per-card |dR| for Small and Tiny F6 are from `../sm152/RESULT.md`: one flagged cell, the curve head of six-layer Tiny two-head (SM130) on long cards with int6 factors, +0.205% [-0.129, +1.852] vs own int8 (upper bound above the +1.0% tolerance; T24 alone is +0.002% [-0.085, +0.371]); note the six-layer Tiny final format here is T24, not F6. All other F6 cells are unflagged, and at most 0.107% of cards have a review with |dR| above 0.10.
- T24 (the six-layer Tiny format) long cards and per-card are from `../sm148/RESULT.md`: six-layer Tiny two-head (SM130) +0.002% curve / -0.044% instant, six-layer Tiny Instant-Only (SM130) +0.009%; per-card share 0.028% to 0.078%.
- Big T24 long cards (all 50 users) are from SM155 (`C:/Users/jonat/workspace/wt/sm155/artifacts/small-model/sm155/RESULT.md`): Big two-head (SM91s2) curve -0.016% [-0.092, +0.086], instant +0.026% [-0.084, +0.110]; Big Instant-Only (SM99) +0.006% [-0.105, +0.061].
- **Big F6 has no long-card read-out** (stated, not run).

## Row sources, users, GPU time

- Row sources (`row_sources` in the json): SM148 (REF and T24, 732 set: 718 users Big, 723 Small and Tiny), SM152 (F6, 732 set, same counts), SM160 (REF and T24 on 100 users with id >= 5000; 99 for Big Instant-Only), this run (`gpu/sm164-rows-new.jsonl`): Big F6 754 users, Big REF 654, Small F6 761, Small REF 661, Tiny T24 and REF about 662. Users per comparison: 722 on the 732 set; 1,472 to 1,484 on the 1,500 set.
- Smoke ticket: 45 cells equal the SM152 and SM160 rows exactly (worst difference 0).
- GPU minutes (`gpu/tickets.jsonl`, first pass plus short retry passes): smoke 1; Tiny T24 + REF 31 + 4; Small F6 35 + 3; Small REF 19 + 3; Big F6 54 + 10; Big REF 22 + 6. About 3 GPU hours in total, 23:34 to 02:41.
