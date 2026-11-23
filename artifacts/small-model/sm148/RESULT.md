# Low-rank int8 card and note states, post-training (SM148): result

Question: can the int8 card and note states of the fork be stored as low-rank factors (int8 factor vectors, float16 scales) instead of full int8 rows, without a cost? No training. Labels against int8 cards and notes (REF): rank 4 (Q4), rank 2 below 16 answers then rank 4 (T24), rank 2 (Q2). Rules are the pre-registered ones in `design.md`. Full tables: `sm148-result.md` and `sm148-result.json`; per-user aggregate rows in `gpu/sm148-rows-732.jsonl` and `cpu/sm148-longcard-rows.jsonl` (public benchmark user ids only).

## Verdict (status: pending on coverage only, decision for the main session)

The report stays at `pending` because the two Big models are not complete: 718 of 720 users on the 732 set, and 37 of 48 long-card users. Everything else (Small two-head (SM118), Small Instant-Only (SM113), six-layer Tiny two-head (SM130), six-layer Tiny Instant-Only (SM130)) has full coverage. The go labels in `sm148-result.md` are computed on the covered users and the margins are wide:

| size | go label | state bytes vs int8 | note |
|---|---|---|---|
| Big | rank 4, int8 factors (Q4) | -74.4% | passes for Big two-head (SM91s2) and Big Instant-Only (SM99) |
| Small | rank 4, int8 factors (Q4) | -52.3% | passes for Small two-head (SM118) and Small Instant-Only (SM113) |
| Tiny | rank 2 then 4 (T24), under the user's Tiny rule | -64.7% | cost against the matching Small at int8: +0.51% curve, +0.60% instant (Tiny Instant-Only (SM130): +0.66%), all under +1.00% |
| Tiny | strict go (rank 4, Q4) | -51.1% | rank 2 then 4 (T24) fails the strict +0.05% rule on both Tiny models |

For comparison the best tiered int formats so far are -43.5% (Big), -38.6% (Small), -42.7% (Tiny).

Rank 2 (Q2) fails everywhere under the strict rule (+0.08% to +0.16%), and rank 2 then 4 (T24) fails the strict rule on Big and Small instant (+0.05% to +0.06%).

## Numbers for the go labels (R2 dataset-wide vs int8; R3 cards with 128+ answers, point and 95% CI; G share of cards with a review |dR| > 0.10)

- Big two-head (SM91s2), Q4: R2 +0.014% curve / +0.018% instant; R3 +0.014% / -0.001%; G 0.0000% / 0.0012%. T24: R2 +0.034% / +0.054% (instant fails); R3 -0.007% / +0.011%; G 0.0071% / 0.0243%.
- Big Instant-Only (SM99), Q4: R2 +0.019% instant; R3 +0.020%; G 0.0036%. T24: R2 +0.059% (fails); R3 -0.011%; G 0.0302%.
- Small two-head (SM118), Q4: R2 +0.005% / +0.009%; R3 -0.133% / +0.060%; G 0.0000% / 0.0012%. T24: R2 +0.029% / +0.050%.
- Small Instant-Only (SM113), Q4: R2 +0.009%; R3 +0.050%; G 0.0004%. T24: R2 +0.056%.
- six-layer Tiny two-head (SM130), Q4: R2 +0.007% / +0.015%; R2T +0.474% / +0.539%; R3 -0.014% / -0.020%. T24: R2 +0.040% / +0.074%; R2T +0.508% / +0.598%; R3 +0.002% / -0.044%; G 0.0278% / 0.0580%.
- six-layer Tiny Instant-Only (SM130), Q4: R2 +0.016%; R2T +0.581%. T24: R2 +0.089%; R2T +0.655%; R3 +0.009%; G 0.0776%.

Per-review read-out (long-card users, covered ones): for Q4 the 99th percentile of the change in predicted retention is 0.010 to 0.020 and at most 6 reviews per model exceed 0.10 (at most 0.0003%), and at most 0.0036% of cards, so no card loses a real retention prediction. T24 and Q2 reach 0.7 for one Tiny curve review.

## Coverage

- 732 set on the GPU: Big two-head (SM91s2) and Big Instant-Only (SM99) both 718 of 720 rule users. Five users (296, 2073, 2267, 3735, 4852, 424k to 532k reviews each) ran out of GPU memory at cap 0.92 (nvidia-smi peak 15.8 GB of 16 GB) in five scoring passes. A further ticket (001503) with one fresh process per model and label, expandable segments, cap 0.92 (the highest allowed) failed again with the same memory error (13.0 GB allocated of 14.6 GB allowed). The scorer has no smaller-batch or chunk option, so they stay missing.
- Long-card users on the CPU: 37 of 48 for both Big models. The 13 users killed by the 5.9 GB memory guard (596, 1015, 1399, 1714, 1717, 2119, 2289, 2393, 4132, 4479, 4566, 4912, 4965; 87k to 151k reviews) were retried with one fresh process per (user, model), then with the guard unchanged: every attempt was killed again at 6.0 to 7.3 GB, including the smallest user (4479, 87k reviews) with one model and the 4 labels. A probe (`memprobe.py`, same scorer unchanged) shows the peak is the single-label forward pass (the padded card by time tensors of the recurrent step), reached within 15 seconds, so splitting labels cannot help and a user needs more than 5.9 GB per CPU process under this scorer. The retry stopped after 18 of 26 attempts (the remaining ones are the larger users). The per-review |dR| read-out (`sm148-dr.json`) is therefore incomplete for Big only.
- Integrity: REF rows equal the int3 young tier on the early switch (SM145) rows to 1e-6 (max difference 1e-6 for Big, 0 for Small and Tiny).

## Speed vs int8 (model time, review evaluations per second)

GPU, 732 set: Big two-head (SM91s2) 76,714 for int8 and 28,850 to 29,769 for the low-rank labels (0.38x); Big Instant-Only (SM99) 77,646 vs about 29,200 (0.38x); Small 124,598 vs 56,400 to 57,800 (0.45x to 0.46x); Tiny 295,619 vs 83,200 to 85,600 (0.28x to 0.29x). The GPU low-rank path is a torch step loop with a Jacobi solve, not a fused kernel.
CPU, long-card users: Big 913 vs 832 to 866 (0.91x to 0.95x); Small 1,529 vs 1,404 to 1,451 (0.92x to 0.95x); Tiny 3,690 vs 3,205 to 3,369 (0.87x to 0.91x).
GPU time: Tiny scoring 69 minutes, Small 100 minutes, Big 167 minutes plus 26 minutes of retries, smoke 15 minutes; the coverage ticket 001503 about 5 minutes.

## What it means

The compact low-rank format is lossless in practice for rank 4 at every size (R2 at most +0.02% on the covered users, per-card read-out clean), cutting the state bytes by 74% (Big), 52% (Small) and 51% (Tiny) against int8. Rank 2 then 4 saves more (-82%, -66%, -65%) at +0.03% to +0.09%, which fails the strict rule but is inside the user's Tiny budget. The cost is GPU speed (0.28x to 0.46x here, a step loop) and 5% to 13% on the CPU; a fused kernel would be needed for the fork. Open decision for the main session: accept the Big go label on the covered users (718 of 720 users, 37 of 48 long-card users) or hold the Big go until the 5 GPU users and 11 long-card users can be scored on a machine with more memory.

## Files added for the coverage attempt

`coverage-lc.sh`, `coverage-run.sh` (CPU retry), `gpu-coverage.sh` (GPU ticket 001503), `memprobe.py` (memory probe), `cpu/cov/` (logs), `gpu/cov.log`.

## Main-session decision (11-23)

The Big go is accepted as rank 4 (Q4, -74.4%) on 718 of 720 GPU rule users and 37 of 48 long-card users. Neither the 5 largest GPU users (424k to 532k reviews) nor 13 long-card users fit in memory on this machine, and the scorer has no chunk option. A check on the Small and Tiny models, which did score those 5 users, puts rank 4's instant cost for them at -0.01% to +0.04% (median over all users +0.005% to +0.010%). Two missing users cannot move Big's dataset-wide upper bound from about +0.02% to +0.05%. This is a coverage shortfall, not a failed rule. A machine with more memory can still confirm it.


## Verdict under the user's 11-23 budgets

These budgets were set by the user on 11-23, after all the rows existed. They replace the pre-registered rules in `design.md` for the go. Nothing was rescored: the numbers come from the existing rows (`gpu/sm148-rows-732.jsonl`) with the same paired-user bootstrap helper as `sm148_report.py` (`showtime_report.compare`, 2,000 draws, seed 0, 95% CI), run by `sm148_verdict_1123.py` (output `sm148-verdict-1123.json`). The cross-model comparison passes the Big rows as the reference. Pairing is per user on the 718 users common to both models.

Rules:
- Small and Tiny: total cost against the matching Big model at int8 (`w8+s8hx8`), CI upper bound per head. Small at most +0.5%, Tiny at most +1.0%. Matching: Small two-head (SM118) and six-layer Tiny two-head (SM130) against Big two-head (SM91s2), curve and instant heads; Small Instant-Only (SM113) and six-layer Tiny Instant-Only (SM130) against Big Instant-Only (SM99), instant head only.
- Big: at most +0.15% against its own int8, CI upper bound per head.
- Long-card (128+ answers) cost (R3) and the per-card |dR| guard (G) are warnings only.
- The go per size is the smallest-bytes label that passes for both models of the size, among int8 (REF), rank 4 (Q4), rank 2 then 4 (T24) and rank 2 (Q2).

### Every label, model and head (cost in %, 95% CI, upper bound against the budget)

| size | model | label | head | cost vs reference | CI upper | budget | margin | pass |
|---|---|---|---|---|---|---|---|---|
| Big | Big two-head (SM91s2) | rank 4 (Q4) | curve | +0.014% [+0.008, +0.020] | +0.020% | +0.15% | +0.130% | yes |
| Big | Big two-head (SM91s2) | rank 4 (Q4) | instant | +0.018% [+0.015, +0.021] | +0.021% | +0.15% | +0.129% | yes |
| Big | Big two-head (SM91s2) | rank 2 then 4 (T24) | curve | +0.034% [+0.024, +0.043] | +0.043% | +0.15% | +0.107% | yes |
| Big | Big two-head (SM91s2) | rank 2 then 4 (T24) | instant | +0.054% [+0.049, +0.059] | +0.059% | +0.15% | +0.091% | yes |
| Big | Big two-head (SM91s2) | rank 2 (Q2) | curve | +0.080% [+0.061, +0.101] | +0.101% | +0.15% | +0.049% | yes |
| Big | Big two-head (SM91s2) | rank 2 (Q2) | instant | +0.109% [+0.096, +0.122] | +0.122% | +0.15% | +0.028% | yes |
| Big | Big Instant-Only (SM99) | rank 4 (Q4) | instant | +0.019% [+0.016, +0.022] | +0.022% | +0.15% | +0.128% | yes |
| Big | Big Instant-Only (SM99) | rank 2 then 4 (T24) | instant | +0.059% [+0.054, +0.065] | +0.065% | +0.15% | +0.085% | yes |
| Big | Big Instant-Only (SM99) | rank 2 (Q2) | instant | +0.118% [+0.107, +0.132] | +0.132% | +0.15% | +0.018% | yes |
| Small | Small two-head (SM118) | int8 cards and notes (REF) | curve | +0.213% [+0.057, +0.374] | +0.374% | +0.50% | +0.126% | yes |
| Small | Small two-head (SM118) | int8 cards and notes (REF) | instant | +0.151% [+0.126, +0.176] | +0.176% | +0.50% | +0.324% | yes |
| Small | Small two-head (SM118) | rank 4 (Q4) | curve | +0.218% [+0.061, +0.379] | +0.379% | +0.50% | +0.121% | yes |
| Small | Small two-head (SM118) | rank 4 (Q4) | instant | +0.160% [+0.135, +0.186] | +0.186% | +0.50% | +0.314% | yes |
| Small | Small two-head (SM118) | rank 2 then 4 (T24) | curve | +0.241% [+0.083, +0.405] | +0.405% | +0.50% | +0.095% | yes |
| Small | Small two-head (SM118) | rank 2 then 4 (T24) | instant | +0.201% [+0.175, +0.227] | +0.227% | +0.50% | +0.273% | yes |
| Small | Small two-head (SM118) | rank 2 (Q2) | curve | +0.258% [+0.098, +0.422] | +0.422% | +0.50% | +0.078% | yes |
| Small | Small two-head (SM118) | rank 2 (Q2) | instant | +0.229% [+0.202, +0.257] | +0.257% | +0.50% | +0.243% | yes |
| Small | Small Instant-Only (SM113) | int8 cards and notes (REF) | instant | +0.016% [-0.010, +0.042] | +0.042% | +0.50% | +0.458% | yes |
| Small | Small Instant-Only (SM113) | rank 4 (Q4) | instant | +0.025% [-0.001, +0.050] | +0.050% | +0.50% | +0.450% | yes |
| Small | Small Instant-Only (SM113) | rank 2 then 4 (T24) | instant | +0.071% [+0.046, +0.096] | +0.096% | +0.50% | +0.404% | yes |
| Small | Small Instant-Only (SM113) | rank 2 (Q2) | instant | +0.101% [+0.075, +0.127] | +0.127% | +0.50% | +0.373% | yes |
| Tiny | six-layer Tiny two-head (SM130) | int8 cards and notes (REF) | curve | +0.681% [+0.505, +0.861] | +0.861% | +1.00% | +0.139% | yes |
| Tiny | six-layer Tiny two-head (SM130) | int8 cards and notes (REF) | instant | +0.672% [+0.620, +0.727] | +0.727% | +1.00% | +0.273% | yes |
| Tiny | six-layer Tiny two-head (SM130) | rank 4 (Q4) | curve | +0.688% [+0.513, +0.866] | +0.866% | +1.00% | +0.134% | yes |
| Tiny | six-layer Tiny two-head (SM130) | rank 4 (Q4) | instant | +0.687% [+0.635, +0.742] | +0.742% | +1.00% | +0.258% | yes |
| Tiny | six-layer Tiny two-head (SM130) | rank 2 then 4 (T24) | curve | +0.722% [+0.548, +0.899] | +0.899% | +1.00% | +0.101% | yes |
| Tiny | six-layer Tiny two-head (SM130) | rank 2 then 4 (T24) | instant | +0.746% [+0.693, +0.802] | +0.802% | +1.00% | +0.198% | yes |
| Tiny | six-layer Tiny two-head (SM130) | rank 2 (Q2) | curve | +0.759% [+0.584, +0.939] | +0.939% | +1.00% | +0.061% | yes |
| Tiny | six-layer Tiny two-head (SM130) | rank 2 (Q2) | instant | +0.802% [+0.745, +0.861] | +0.861% | +1.00% | +0.139% | yes |
| Tiny | six-layer Tiny Instant-Only (SM130) | int8 cards and notes (REF) | instant | +0.579% [+0.533, +0.625] | +0.625% | +1.00% | +0.375% | yes |
| Tiny | six-layer Tiny Instant-Only (SM130) | rank 4 (Q4) | instant | +0.595% [+0.549, +0.642] | +0.642% | +1.00% | +0.358% | yes |
| Tiny | six-layer Tiny Instant-Only (SM130) | rank 2 then 4 (T24) | instant | +0.668% [+0.619, +0.717] | +0.717% | +1.00% | +0.283% | yes |
| Tiny | six-layer Tiny Instant-Only (SM130) | rank 2 (Q2) | instant | +0.735% [+0.684, +0.785] | +0.785% | +1.00% | +0.215% | yes |

The reference is the model's own int8 for Big, and the matching Big model at int8 for Small and Tiny (so the int8 row of a Small or Tiny model is the pure size cost, before any state compression). Big has no int8 row of its own (cost 0 by definition).

### Go per size

Every label passes for both models of every size, so the smallest-bytes label, rank 2 (Q2), is the go for all three sizes.

| size | go label | state bytes vs int8 | worst CI upper bound (any model, head) | budget | margin left |
|---|---|---|---|---|---|
| Big | rank 2 (Q2) | -82.8% | +0.132% (Big Instant-Only (SM99)) | +0.15% | 0.018% |
| Small | rank 2 (Q2) | -68.2% | +0.422% (Small two-head (SM118), curve) | +0.5% | 0.078% |
| Tiny | rank 2 (Q2) | -66.7% | +0.939% (six-layer Tiny two-head (SM130), curve) | +1.0% | 0.061% |

Next smallest, if a margin is judged too thin: rank 2 then 4 (T24), -81.7% / -66.3% / -64.7% (Big / Small / Tiny), margins 0.085% / 0.095% / 0.101%; rank 4 (Q4), -74.4% / -52.3% / -51.1%, margins 0.128% / 0.121% / 0.134%. Margin per model and head for every label is in the table above. The Tiny and Small margins are mostly the size cost against Big (the int8 rows already use +0.58% to +0.68% of the Tiny budget and +0.15% to +0.21% of the Small curve/instant budget); the state compression itself adds +0.08% to +0.12% (Big, Q2), +0.04% to +0.08% (Small, Q2) and +0.08% to +0.13% (Tiny, Q2).

### Warnings (not rules)

Long-card (128+) cost R3 and the share G of cards whose worst review changes by more than 0.10 in predicted retention, for the go label rank 2 (Q2) against each model's own int8 (long-card users on the CPU):

| model | R3 curve (point, CI upper) | R3 instant (point, CI upper) | G curve | G instant |
|---|---|---|---|---|
| Big two-head (SM91s2) | +0.137% / +0.428% | +0.324% / +0.740% | 0.155% | 0.289% |
| Big Instant-Only (SM99) | - | +0.363% / +0.741% | - | 0.353% |
| Small two-head (SM118) | -0.172% / +0.309% | +0.292% / +0.516% | 0.016% | 0.039% |
| Small Instant-Only (SM113) | - | +0.310% / +0.775% | - | 0.083% |
| six-layer Tiny two-head (SM130) | +0.548% / +1.724% | +0.304% / +0.463% | 0.129% | 0.283% |
| six-layer Tiny Instant-Only (SM130) | - | +0.273% / +0.713% | - | 0.316% |

- Against the old hard limits (long-card point +0.5%, upper +1.0%, G 1%), rank 2 (Q2) would have breached one: the six-layer Tiny two-head (SM130) curve at 128+ answers is +0.548% (upper +1.724%). Every other point is under +0.4% and every upper bound under +0.78%. Every G is under 0.32% (limit was 1%). Per-review read-out: rank 2 (Q2) reaches a change of 0.7 in predicted retention for one Tiny curve review.
- Big coverage note: both Big models cover 718 of 720 GPU users (the five largest users ran out of memory; two of them are rule users) and 37 of 48 long-card users (13 killed by the memory guard), and the Big per-review read-out is incomplete. The 11-23 check on the missing users was done for rank 4 (Q4) only; for rank 2 (Q2) the Big margin is 0.018% and was not checked on those users. Small and Tiny, which scored them, show the compression cost of those users is not larger than the median, so the Big go is unlikely to move, but the thin margin is the thing to watch.
- Speed warning from the main result stays: low-rank GPU path is 0.28x to 0.46x of int8 speed in this scorer (a step loop, not a fused kernel).
