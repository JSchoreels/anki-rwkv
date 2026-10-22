# SM58 disagreement test: small model vs big model (UM+ and slope)

Compares the Muon best small model (SM36) with the big model (the fork's RWKV, `RWKV_trained_on_5000_10000.pth`) in the
Search Stats Extended conventions, as reproduced in `anki-rwkv/artifacts/collection-eval/2026-10-14/graphs/make_graphs.py`
(definitions copied unchanged into `sm58_umplus.py`; see that README for the add-on source references):
41 buckets `round(20*d)` with `d = p_small - p_big`; UM+ = `sqrt(sum_b n_b err_b^2 / N)` over all buckets; slope = count-weighted
regression of bucket error on bucket disagreement over buckets with >= 200 reviews. Slot mapping: big model in the RWKV slot
(seagreen), small model in the FSRS slot (steelblue). Panels: all benchmark test reviews, and the add-on's eligibility
(not the first review, >= 1 day since the previous review: `elapsed_seconds >= 86400` in the revlog).

**The two slopes always differ by exactly 1** in this construction (the error difference per bucket is the disagreement itself:
`slope_small = slope_big + 1`). So a tie is +0.5 for the small model and -0.5 for the big one; the closer the small model's slope is
to 0 and the big one's to -1, the more the small model is the one that is right where they disagree.

Data: benchmark users = the 732 pool minus the long users 1534, 1601, 4625 (729 users; users that run out of memory at the 0.8 cap
are listed in `summary.json` under `users_skipped`). Per-review predictions are `small_model_heldout.py --dump` files of the deployed
variant `w8+s8` (8-bit weights, int8 state), full-history protocol; the big model's 464-user dumps are reused from
`wt/small-model/artifacts/small-model/sm24-dump` (label `fork/w8+s8`, same checkpoint, verified against its benchmark rows), the other
265 users are dumped here. Own collection: float32 CPU full history (`sm58_own_preds.py`), same prepared batch as
`score_collection.py`, big model from `preds-model-B-full-s0.parquet`; aggregate output only (the per-review predictions stay in the
uncommitted `collection-cache/`). Benchmark and collection therefore use different numeric variants (w8+s8 vs float32).
User bootstrap: users resampled with replacement, 2,000 resamples, seed 0 (benchmark only). LogLoss = pooled review-weighted.

Files: `launch-sm58.sh` (detached launcher), `sm58_plan.py`, `sm58_own_preds.py`, `sm58_umplus.py`, `quiet_wait.py`,
`dryrun-launch-sm58.log`; outputs `summary.{json,md}`, `graphs.html`, `um-plus-*.png`.
