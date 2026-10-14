# QAT int4 per-channel fine-tune, run 1 (ticket 000311, 2026-10-14)

Aggregate only. No checkpoints or per-user data are committed.

## Run
- Command: `jobs/qat.sh 4 0 w4pc-run1 2400`. Init: the fork's 2.76M `RWKV_trained_on_5000_10000.pth`. 4-bit per-channel fake quantization on 265 tensors, bf16 child / fp32 master, lr 1e-4 with a 30-step warmup, 128-review windows, 32,768 reviews per step.
- Exit 0. 24 optimizer steps in 2,410 s of training (2,527 s wall). 648,441 reviews (529,241 curve labels) from a pool of 3,969 training users. 269 reviews/s, 0 OOM-skipped batches, peak reserved 23,160 MiB.
- The first 18 steps are full 32k-review batches. Steps 19-24 are short (2.5k-6.5k reviews) tail batches from the pool, so they are noisier.

## Loss curve (training loss on the current batch, different users each step)
- Step 1: 0.982. Step 24: 0.941. Min 0.832 (step 22), max 1.176 (step 23, 2,560 reviews).
- Mean of steps 1-5: 0.984. Mean of steps 14-18 and 22: 0.952.
- Batch-to-batch spread is about +/-0.07, larger than any drift. The effective learning rate stayed at or below 2.1e-5 (warmup plus decay over the budget), so 24 steps is too few to see a trend. **The curve does not show whether QAT helps.**
- Grad norm 0.15-0.24 throughout, no divergence.

## Comparison
- `summary.json` holds no evaluation numbers, only throughput counters. The training loss is not comparable with the held-out LogLoss in `report.md` (different users, a mix of curve and immediate losses, a different window regime).
- Post-training int4 per-channel on the collection, from `report.md`: +0.02% (curve) / +0.42% (instant) vs the fork fp32. The held-out table has +0.14% / +1.33% for w4-pc.
- **There is no QAT-vs-PTQ number yet, so whether QAT beats PTQ is not established.**
- The fp32 fine-tune-only control (ticket 000291) was cancelled before it ran (`qat/fp32-control1-console.log`). Without it, any QAT delta would mix quantization-aware training with the effect of fine-tuning on 128-review windows.

## What is needed to judge it
1. Collection score (CPU, about 5 min per variant, supported now):
   `.venv/Scripts/python.exe scripts/compress_score_collection.py --checkpoint artifacts/compression/qat/w4pc-run1/final.pth --tag qat-w4pc-run1 --variants w4-pc,w4-pc+s8,fp32`, then `scripts/compress_report.py`. The checkpoint holds latent fp32 weights, so the w4-pc variant applies the fake quantization.
2. Held-out cohort B: `scripts/compress_heldout_b.py` has **no `--weights` option** (it always loads `cq.FORK_WEIGHTS`, lines 71 and 114). Add a `--checkpoint` argument, then run `w4-pc` and `w4-pc+s8` on the same users as the PTQ rows (users 0-999 of `b-users.json`) so the paired delta is vs PTQ w4-pc and vs fp32.
3. Fp32 control: re-queue `bash <abs>/artifacts/compression/jobs/qat.sh 0 0 fp32-control1 2400` and score it with `--variants fp32`. It is only worth the GPU time if step 1 shows w4-pc QAT beating PTQ.
4. Likely a longer run: with 24 steps and lr <= 2e-5 the weights moved very little. A second QAT run with a larger budget (or `--lr` 3e-4) is the real test. That is a decision for the main session.
