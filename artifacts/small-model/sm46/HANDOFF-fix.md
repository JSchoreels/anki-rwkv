# SM46 GPU check failure (2026-10-20 21:28): diagnosis and fix

Status: the OOM is fixed. The GPU/CPU mismatch comes from near-ties, so it is not a bug. The runbook was **not** relaunched; relaunching needs the user's go.
If it is relaunched unchanged, the GPU check will STOP again: all 3 users miss 1e-5 after the fix (see below).

## Diagnosis

1. **OOM:** `torch.linalg.eigh` on CUDA uses `cusolverDnXsyevBatched`, which needs ~0.6 MB of workspace per 16 x 16 float64 matrix (measured: 602 MB per 1,000).
   - Timing user 170 (30k reviews) had ~19,700 matrices in one step, so eigh asked for 11.3 GiB.
   - This happened after the check users had already been compared: user 1712 had failed with Δ instant -1.4e-4 before the OOM.
2. **Mismatch: near-ties, amplified through the recurrence.** The rank-6 codec has 53 WKV stages per head and 8 shift stages, so it makes far more nearest-codeword decisions than SM44's codecs, and its deep stages have tiny residuals and gaps.
   - The first divergence was found by recording picks and codec in/out on both devices (scripts in wt/sm32c `artifacts/small-model/sm44/diag/`).
   - User 1712: the first flip is in the card time-mixer shift, position 0, stage 6 of 8. The inputs differ by 6e-8 (1 ulp). The float64 squared-distance gap is +2.9e-11 on the CPU input and -4.8e-11 on the GPU input (relative 4.7e-6), a tie that even float64 cannot settle.
   - User 4676: the first flip is at the first card WKV step (rank-1 states), component 0, stage 13 of 15. The input differs by 5e-7 relative and the gap is 7.6e-9 (relative 9e-6). The output state changes by 3% and propagates from there.
   - Files: `diag/first-wkv-divergence-*.json`.
3. SM46 shares `state_vq.factorize` with SM44, so it also had SM44's rank-component bug: components of lower-rank states are float noise. With SM46's 10-bit full-range norm grid those terms were tiny, but their direction was arbitrary. The SM44 fix is applied here too.

## Fix (branch of wt/sm46, pushed)

- `a4598176`: chunked CUDA eigh (1,024 matrices per call, ~0.6 GB, bitwise the same values; checked on 5,000 Grams with chunks 2,048 .. 1) plus the SM44 rank fix (components j >= 1 with sigma_j <= 1e-4 sigma_0 are invalid).
  - Effect of the rank fix on the CPU reference (users 1712 / 4676 / 2388): curve +1.3e-5 / +3.6e-5 / -9e-6 and instant -1.2e-5 / +6e-7 / +7e-6.
- The next commit has the re-run CPU check (`check-sm46-cpu.json`, ALL OK, bitwise) and the new references.

## Checks done

- **GPU, direct, 21:44, before the break** (`diag/check-sm46-gpu-postfix-staleref.json`; its comparisons used the old references, so only the timing counts): the OOM is gone.
  - Timing user 170 ran in 9.5 s.
  - Rate: 4,026 reviews/s (w8+s8: 56.5k).
- **Post-fix GPU vs post-fix CPU** (per-user benchmark LogLoss Δ curve / instant):
  - 1712: +6e-7 / -2.2e-4
  - 4676: -2.4e-5 / +1.2e-5
  - 2388: +1.7e-5 / -6.5e-6
  - All exceed the 1e-5 tolerance. Pre-fix, 2 of 3 passed, by luck.
- CPU check re-run: ALL OK.

## Needs the user's decision

The same options as SM44 (wt/sm32c `artifacts/small-model/sm44/HANDOFF-fix.md`):
- (a) a separate tolerance rule for discrete codecs (aggregate on many users, wider per-user band); SM46 would need about 3e-4 per user;
- (b) float64 codeword distances, which reduce flips but do not fix them (1712's first flip survives float64);
- (d) CPU-only scoring.

Expected per-user noise from flips at this budget is ~1e-5 to 2e-4 of benchmark LogLoss, of either sign.

## Relaunch (only with the user's go; the runbook runs its GPU check first)

```
nohup bash C:/Users/jonat/workspace/wt/sm46/artifacts/small-model/sm46/runbook-sm46.sh \
  > C:/Users/jonat/workspace/wt/sm46/artifacts/small-model/sm46/runbook-sm46.log 2>&1 & disown
```
- The runbook waits for the SM44 runbook log to show `done` or `STOP`. The current SM44 log already shows STOP, so either relaunch SM44 first (its relaunch overwrites the log) or set `WAIT_SM44=0`.
- GPU check: ~1 min plus timing, queued with an expected time of 25 min. It will STOP unless the check rule changes.
- The failed run's log is kept in `runbook-sm46-run1-failed.log`.
