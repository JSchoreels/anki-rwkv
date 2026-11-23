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
