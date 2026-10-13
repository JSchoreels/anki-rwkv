# Final R01 preparation: 0069–0070

The inherited campaign is 23 candidates from 0048 through 0070, with 21 already complete at entry. Accepted 0042 and the original baseline/cumulative loss ceiling 0.001 remain unchanged. No 0071 or broader validation is authorized.

## F34f explicit backward

The candidate retains F34e exact fused forward, but replaces backward graph reconstruction with explicit PyTorch derivative operations, sum-to-size reductions, deterministic gather backward and final branch addition. This removes Python/autograd graph construction while retaining original operation boundaries.

The first 16-case synthetic check failed only weight gradients in four B=T=1 cases: reducing singleton dimensions converted signed zero. The repaired path skips singleton reduction dimensions, as the original sum-to-size does. Both attempts are retained under `f34f-synthetic` and `f34f-synthetic-repaired`; the repaired check passed all 16 FP32/BF16 shapes/edge cases. `f34f-real/result.json` passes all five real loss/every-gradient batches. All relevant source edits finished before these compatible checks; no training source edits occurred during 0069 smoke/full or replicas.

0069 smoke matches every update and result against accepted 0042. Full training is 45.460 s / 168,390 reviews/s; this historical comparison does not decide promotion. Prelaunch CPU 5.8%, no other Python GPU process. Frozen contemporaneous look 2/4/6 remains mandatory.

## F34g fused rounded partial derivatives

A follow-on candidate can fuse the three elementwise derivative outputs, including shift gather, while preserving dtype rounding at subtraction/complement/product boundaries and leaving all original PyTorch reductions and deterministic scatter/add operations intact. This is a new mechanism relative to explicit backward. Throughput is unmeasured and numerical prerequisites remain pending.


F34g uses one kernel for shift gather and three derivative elementwise outputs. It rounds complement and shifted difference to the input dtype before the three separately rounded multiplications, and retains original sum-to-size reductions, deterministic scatter and final addition. `f34g-synthetic/result.json` passes all 16 FP32/BF16 boundary/signed-zero cases; `f34g-real/result.json` passes five loss/every-gradient batches. Source edits finished before both checks and all 0070 training launches. 0070's smoke and full final parameters/held-out inputs/four losses are exact. Full training 45.496 s / 168,257 reviews/s; prelaunch CPU 6.1%, no other Python GPU process. Initial report hit the already documented Windows Errno22 opening the run log. Explicit report-only resume succeeded; no training/profile was rerun and both console attempts remain saved.

0069 completed at two frozen blocks: mean -0.075311%, lower -1.480277%, reject. All four reference/candidate replicates exact. 0070 timing remains pending until the frozen gate resolves. Both candidates compare independently with accepted 0042; neither inherits retained recurrence alternatives or rejected bundles.

0070 resolved at six frozen blocks: mean +0.137395%, lower -0.677921%, retained unproven option. Campaign 23/23 complete, zero authorized candidates remain. Implementation commit `a9ed12390b5d49edc17367726cdf3bb9ab64aa5a` was created after experiments; original origins and source archives remain authoritative. The audit confirms all eighteen full-history cases exact, zero allocator retries/OOMs, 3157 MHz median SM and 12,188 MiB peak reserved. No 0071 or broader validation was prepared or run.
