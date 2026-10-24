# Small models on the user's own collection (CPU, aggregate only)

`score_collection_small.py` wraps `scripts/small_model_heldout.py` on the collection converted on 2026-10-14
(anki-rwkv `artifacts/collection-eval/2026-10-14`, read-only): full history as one sequence, the 173,690 benchmark test
reviews of its label DB. Fork fp32 reproduces the reference exactly (0.4417 / 0.4366), so the wrapper matches the
2026-10-14 evaluation. The collection's review types come from the user's fork, not SM-2 (806 learning rows, 34,458
type-4 rows). Only aggregate metrics are kept (no `--dump`).

| model (w8+s8 unless noted) | curve LogLoss | Δ vs fork w8+s8 | instant LogLoss | Δ vs fork w8+s8 | RMSE(bins) curve / instant |
|---|---|---|---|---|---|
| fork 2.76M, fp32 | 0.4417 | +0.00% | 0.4366 | −0.02% | 0.0223 / 0.0194 |
| fork 2.76M, w8+s8 | 0.4417 | — | 0.4367 | — | 0.0224 / 0.0200 |
| fork, history review types relabelled as no learning steps | 0.4414 | −0.06% | 0.4369 | +0.04% | 0.0237 / 0.0214 |
| fork, review-type input zeroed | 0.4538 | +2.74% | 0.4412 | +1.03% | 0.0539 / 0.0292 |
| student w64 331.5k, 16k-trained epoch 9, decay at 16k, scored at 32k (`ctx16384c-dec9-L32k`) | 0.4424 | **+0.18%** | 0.4376 | **+0.21%** | 0.0274 / 0.0208 |
| student w64, 32k epoch 10 + 32k quarter decay (`dec10`) | 0.4423 | +0.14% | 0.4375 | +0.18% | 0.0286 / 0.0207 |
| student w64, carried state v2 + one-epoch 32k decay (`sm40v2`, 10-19) | 0.4419 | **+0.06%** | 0.4373 | **+0.14%** | 0.0256 / 0.0209 |
| Muon version: student w64, Muon optimizer + one-epoch 32k decay (`sm36`, 10-22) | 0.4412 | **−0.10%** | 0.4369 | **+0.06%** | 0.0250 / 0.0205 |
| 8k model: Muon version trained on 8k users (`sm55`, 10-22) | 0.4414 | **−0.05%** | 0.4370 | **+0.08%** | 0.0259 / 0.0205 |
| full retrain: all 8,000 users in every epoch, 10 epochs + one-epoch 32k Muon decay (`sm56`, 10-24) | 0.4415 | **−0.03%** | 0.4371 | **+0.10%** | 0.0252 / 0.0207 |

Even on this non-SM-2 collection the fork reads its review types (zeroed: +2.74% / +1.03%); a consistent relabelling
costs it nothing here. For comparison (docs/ACCURACY_LOG.md): the w32 full run at epoch 10 was +2.68% / +3.64% on this collection.
