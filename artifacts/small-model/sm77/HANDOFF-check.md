# HANDOFF: GPU check of tiny architecture additions (SM77), failures of 10-31 01:03

Run of 10-31 01:03 (`gpu/check-gpu-sm77.json`, `gpu/check.out`): K_* all pass, **P** and **L** failed. Both were
checks that were too strict, not bugs in the arm. Fixed in `scripts/sm77_warm.py` and `scripts/small_model_wkv.py`.

## P_gpu_trainer_zero_init_equals_w64 (old gradients 7.8% apart in bf16)

Cause: bf16 rounding of a reordered backward graph, not a different gradient path.
- At zero the arm's forward is exact in any dtype (`k * (1 + 0)`, `a + 0 * (1 - a)`, `+ 0`), hence loss and outputs
  bitwise equal (they were, on all 4 updates). The backward differs: `key_pair` has two `normalize` branches (removal
  and insertion key), so autograd adds their contributions to `k` / `a` in another order. Kernels: same row-tile exact
  kernels; the dS0 backward equals the stock state backward bitwise at S0 = 0 (K_* `zero_state_bitwise_equal_sm21_path`);
  the exact shift / index kernels are sort-free (no atomics); CPU reruns are bitwise deterministic.
- In bf16 each partial result is rounded to 8 bits, so the reordering shows per tensor on small reduction-heavy
  tensors. CPU bf16 twin of the GPU check (same users / batches / options, 10-31, `RWKV_NO_JIT=1`):
  - arm vs w64, bf16: max 8.3% of a tensor's max (GPU: 7.8%), worst `k_scale_linear.bias` (4 entries), layer norms,
    `a_lora` B; 47 of 1,416 (update, tensor) pairs above the old 2e-2 bar.
  - w64's own bf16 gradient vs its float32 gradient: up to 130% per tensor, 2.5-5.7% whole-model L2.
  - arm vs w64, whole-model L2: 0.36-0.80% (0.09-0.25 of w64's own bf16 error); arm's bf16 error / w64's: 0.99-1.01.
  - float32: arm vs w64 at most 3.2e-6 of a tensor's max, whole-model L2 1.3e-7..2.5e-7; losses bitwise equal.
- Fix (check): `gpu_trainer_pair` now also runs both models in float32 on the GPU (same kernels, TF32 off) and w64
  twice in bf16. Pass = bf16 loss / outputs bitwise (strict, kept), float32 loss / outputs bitwise (strict, new),
  float32 old gradients within 1e-4 of each tensor's max (the exactness proof; CPU 3.2e-6), and on the whole-model
  bf16 gradient: arm-vs-w64 gap <= w64's own bf16 error and arm's bf16 error <= 1.25 x w64's, plus the new parameters
  get gradients. Bitwise / per-tensor bf16 numbers, the worst tensor and w64's rerun determinism are reported only.

## L_scorer_long_kernel_from_state (bits 8 errors 4.1e-4 / 1.6e-3)

Cause: the 8-bit state is re-rounded every step, so the kernel and the float32 reference (another summation order)
flip a code by one step now and then; the output is not continuous in 1e-7 noise. Evidence (CPU, 10-31, the test's
own inputs): the float32 reference vs a float64 run of the same recurrence differs by 8.6e-4 (K16) / 5.1e-4 (K32)
from S0 and by 5.7e-4 / 1.2e-3 from zero (the stock self-test's inputs), in 0.04-0.07% of the outputs; bits 0 agree
to 1e-7. The scorer's own CUDA gate (`small_model_heldout.py`) already holds only bits 0 at 1e-4 for this reason.
`zero_state_equals_stock_bitwise` (bits 0 and 8) and bits 0 at ~1e-7 show the S0 load is right.
- Fix (check): bits 0 keep 1e-4 and the S0 = 0 bitwise checks stay; bits 8 pass if max error <= one quantisation step
  on every element of a state row (`/bound` = largest row step x max |r|_1; ~0.03 / 0.06 here) and <= 1% of the
  outputs are off by more than 1e-4 (`/frac_over_1e-4`). `small_model_wkv.reference` takes an optional `steps` list
  (behaviour unchanged without it).

## Status

- CPU run of the new code (`gpu_trainer_pair` with device cpu, without `RWKV_DELETED_CARDS`; `self_test_state` on
  the CPU reference): P pass: bf16 / float32 loss and outputs bitwise, float32 old gradients 2.7e-6, bf16 L2 gap
  0.39-1.08% vs w64's own bf16 error 2.3-8.2% (arm's 2.3-8.0%), per-tensor bf16 6.6% (reported), w64 rerun
  bitwise, 60 new tensors with gradient; L keys present (bits 8 bound 0.032 / 0.065).
- GPU re-check (check only, no smoke, no `gpu-check-ok` marker): queue ticket 000779, enqueued 10-31 01:20 by
  `gpu/recheck-gpu.sh` (hidden, detached; wrapper pid 70544), waiting behind SM67's 540-min cool-down ticket
  000778 (started 01:04, ends ~10:05); runs ~2 min in the gap before SM67's next ticket. Output
  `gpu/check-gpu-sm77-recheck.json`, `gpu/recheck.out` (last line `[recheck] exit N`). A Claude app restart kills
  the wrapper (then the ticket is dropped; the launcher's ticket 0 runs the same check anyway). Read it with:
  `tail -3 artifacts/small-model/sm77/gpu/recheck.out; grep -o '"pass": [a-z]*' artifacts/small-model/sm77/gpu/check-gpu-sm77-recheck.json`
  If P fails only on `fp32_old_gradients_max_relative_diff` slightly above 1e-4 (GPU float32 ordering), look at
  `fp32_worst_tensor` before loosening; any bitwise loss / output failure is a real difference.

## Relaunch

Safe once SM67 has finished (ideally after the re-check above reads ALL OK): the launcher reruns ticket 0 (`gpu/gpu-check-ok` is absent), skips done preparation
steps, and uses the committed check. Command (PowerShell, hidden), as in `HANDOFF-sm77.md`:

```
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"C:\Program Files\Git\usr\bin\bash.exe" -c "export PATH=/usr/bin:/mingw64/bin:$PATH; bash C:/Users/jonat/workspace/wt/sm77/artifacts/small-model/sm77/launch-sm77.sh >> C:/Users/jonat/workspace/wt/sm77/artifacts/small-model/sm77/gpu/launch-sm77.nohup 2>&1"'; ProcessStartupInformation=$si}
```
