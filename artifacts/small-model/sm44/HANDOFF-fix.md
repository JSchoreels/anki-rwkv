# SM44 GPU check failure (2026-10-20 21:28): diagnosis and fix

Status: one real bug found and fixed. One expected failure is left, caused by near-ties. The runbook was **not** relaunched; relaunching needs the user's go.
If it is relaunched unchanged, the GPU check will STOP again on sm42 vq378x412 user 4676 (see "Needs the user's decision").

## Diagnosis

How the first divergence was found: every codeword pick was recorded on both devices. The CPU torch loop is bitwise the CPU scorer. The GPU side ran both the torch loop and the fused kernel, the kernel with `record=True` picks. The two pick logs were then aligned call by call.
Scripts: `diag/vqdiag.py` (record), `diag/vqcmp.py` / `diag/vqcmpk.py` (compare), `diag/pcmp.py` (metric compare).

1. **vq770x828: a real bug in the rank-2 note codec. It hit the CPU reference and the GPU path alike.**
   - Every entity's first stored WKV state is rank 1, so its second singular value is float32 noise (sigma_1 / sigma_0 ~ 1e-8). `factorize` still marked that component valid and coded it.
   - Its direction is an arbitrary null-space vector. LAPACK and cuSOLVER return different ones: median max-deviation 0.44 on random rank-1 states.
   - The 6-bit component-1 norm grid clamps its size up to the grid floor. That gives sigma 6e-3 instead of ~6e-8, about 1e5 times the true size, so every new note got an arbitrary, device-dependent rank-1 term.
   - First divergence for user 1712: rvq call 71, the first note step, component 2. All 628 head-matrices differ, with input max |dx| up to 1.4.
   - Details: `diag/first-divergence-prefix-vq770x828-1712.json`.
2. **vq378x412 user 4676: near-tie. The rank-1 path has no bug.**
   - The fused kernel's WKV picks equal the CPU picks until the first flip.
   - The first flip is in the note channel-mixer shift codec (`shift2/c` position 0, stage 0), on 2 of 5,160 vectors.
   - The inputs differ by 1.5e-7, which is 1 ulp. The squared-distance gap between the best two codewords is 1.1e-7 (relative 7.6e-7), which is the resolution of the float32 `|c|^2 - 2 x.c` distance.
   - From there the recurrence amplifies the flip: Δ instant 2.8e-5, max |dp| 0.017.
   - The GPU torch loop has the same first flip, then other later flips: Δ instant 5e-7, so it passes by luck.
   - Near-tie census before the first flip: of 83,236 coded vectors, 4 have a stage gap < 1e-6 relative, 18 < 1e-5 and 209 < 1e-4.
   - Details: `diag/first-divergence-vq378x412-4676.json`.
3. After the fix, user 2388 / vq770x828 reaches Δ instant -9.96e-6, just inside 1e-5. Its first flip is again a shift near-tie: card time-mixer shift, gap 9e-8, inputs 1 ulp apart.

## Fix (branch sm44, pushed)

- `8f929a35`: `state_vq.factorize` marks components j >= 1 invalid when sigma_j <= 1e-4 x sigma_0 (`RANK_REL_TOL`, env `STATE_VQ_RANK_REL_TOL`; 0 gives the old behaviour).
  - Rank-1 recipes are unaffected: every card codec, the 412-bit note and the fused kernel. Their CPU references are bitwise the old ones.
- `e2dceabd`: CUDA eigh runs in chunks of 1,024 matrices. cuSOLVER needs ~0.6 MB of workspace per matrix, which caused the SM46 OOM; vq770x828 on large users could hit it too. Values are bitwise the same.
  - The same commit has the re-run CPU check and the new vq770x828 references.
- **Score change caused by the fix (CPU reference, vq770x828):** on users 1712 / 2388 / 4676, curve +4.2e-4 / +9e-5 / +2e-5 and instant -4e-5 / -6.5e-5 / -6.2e-5.
  - Earlier CPU-scored rank-2 rows (SM32: vq770x828, vq1506x828, vqR2*) used the old behaviour, so they are not directly comparable with new SM44 rows.

## Checks done

- CPU check re-run (`check-sm44-cpu.json`): ALL OK, bitwise on every case.
- Direct GPU check, 20 s at 21:43 before the break, with `--output` outside the runbook path (`check-sm44-gpu-postfix-direct.json`):
  - vq770x828 passes on all 3 users. User 1712 is now at the base float level (max |dp| 2e-6).
  - sm32c vq378x412 passes.
  - sm42 vq378x412 still fails on user 4676 (the near-tie above, unchanged by the fix).
  - Reviews/s are unchanged: 35.8k (vq378x412), 10.4k (vq770x828).
- Pre-fix GPU report: `check-sm44-gpu-pre-fix.json`; failed run log: `runbook-sm44-run1-failed.log`.

## Needs the user's decision

Bitwise CPU = GPU is impossible for discrete codecs. The base float state already differs by ~2e-6, and a 1-ulp input change flips any pick whose gap is below ~1e-7. Options:
- (a) A separate tolerance rule for discrete-codec paths. For example, per-user Δ ≤ 1e-4 plus an aggregate check on many users (mean Δ ≈ 0, no sign bias), with the float-state base gated at the normal rule.
- (b) Float64 nearest-codeword distances on both paths. This would have resolved the 4676 flip, because the float64 gap stays positive on both inputs. It reduces flips but does not remove them; SM46 user 1712's first flip has a float64 gap of +2.9e-11 / -4.8e-11 across the two inputs. It changes the codec and needs `.cu` kernel work.
- (c) Deterministic tie-breaking alone does not help, because the inputs themselves differ.
- (d) Score these recipes on the CPU only (~720 reviews/s).

## Relaunch (only with the user's go; the runbook runs its GPU check first)

The GPU check will STOP again on user 4676 unless the check rule is changed per (a).
```
nohup bash C:/Users/jonat/workspace/wt/sm32c/artifacts/small-model/sm44/runbook-sm44.sh \
  > C:/Users/jonat/workspace/wt/sm32c/artifacts/small-model/sm44/runbook-sm44.log 2>&1 & disown
```
GPU check ~1 min, queued with an expected time of 10 min. SM43 has ended, so the wait passes at once. The failed `check-sm44-gpu.json` is not "passed", so the check reruns.
