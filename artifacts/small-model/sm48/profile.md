# SM48: where held-out scoring time goes (profile, 2026-10-20)

Scope: `scripts/small_model_heldout.py`, full-history protocol, GPU path for expertium models (SM36, w64: d=64, 4 heads x 16). Machine: RTX 5080 (84 SMs, 16 GB), 16 logical CPUs. GPU numbers here come from existing scorer rows/logs only; no GPU was used for this profile. CPU timings are CPU work that the GPU run also does on the CPU (prediction collection, benchmark metrics).

## 1. Wall-time split from SM45's 732-pool pass (existing rows)

Source: `wt/sm45/artifacts/small-model/sm45/heldout-sm45-pools.jsonl` + `launch-sm45.log` (2 variants per user, w8+s4c8n and w8+s4; 4 prep workers; script `profile_rows.py` here). Per user: wall = difference of the scorer's cumulative elapsed between progress lines; model = sum of the 2 rows' `model_seconds` (forward + `collect_predictions`); prep = `prepare_seconds` (DataLoader workers, overlapped).

| Reviews per user | Users | Reviews | Wall s | Model s | Prep s (workers) | Other s | Model share | review-evals/s (wall) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| < 10k | 131 | 916,939 | 64 | 32 | 43 | 32 | 0.51 | 28,654 |
| 10k-50k | 343 | 8,545,448 | 382 | 186 | 285 | 196 | 0.49 | 44,741 |
| 50k-150k | 179 | 15,255,383 | 674 | 333 | 491 | 341 | 0.49 | 45,268 |
| 150k-400k | 62 | 14,405,442 | 633 | 317 | 468 | 316 | 0.50 | 45,515 |
| > 400k | 11 | 5,355,547 | 225 | 150 | 163 | 75 | 0.67 | 23,802 |
| all | 726 | 44,478,759 | 1,978 | 1,018 | 1,450 | 960 | 0.51 | 44,973 |

- Only half of the wall time is the model call. "Other" (wall - model) is ~10.8 µs per review-evaluation, flat across sizes, so it is per-review CPU work in the main loop, not per-user overhead.
- Prep is overlapped: 1,450 worker-seconds over 4 workers is ~360 s of a 1,978 s loop, so the loop does not wait for data.
- Per review and variant the scorer handles ~1.8 labelled positions (`collect_predictions`) and ~1.4 benchmark entries (0.7 per head, `metrics` -> upstream `get_stats`), each in a Python loop.

## 2. The "other" half: benchmark metrics in Python (CPU, measured)

- cProfile, CPU scorer, 3 smallest pool users x 2 variants (`cpu-profile-3users.prof`): `metrics` = 1.89 s, of which 1.29 s is the one-off sklearn/pandas import and 0.57 s `get_stats` (58,060 entries: a scalar `np.clip` per entry, 290k list appends, 116k `.tolist()` for an unused `raw` dict, sklearn input validation). `collect_predictions` 0.03 s; `load_state_dict` 12 ms per call.
- Without profiler, real labels of 7 pool users from 3,685 to 261,415 reviews (495,545 reviews), synthetic predictions (`check-fast-metrics.json`, `scripts/check_heldout_fast_metrics.py`): **dict path 8.70 µs per review and variant** (collect ~0.7 + metrics ~8.0) vs **numpy path 0.85 µs**, bitwise equal outputs on all 7 users.
- So ~8 of the ~11.4 µs per review-evaluation outside the forward (other 10.8 + collect inside model_seconds) is this Python loop. The rest is per-user/variant overhead (load_state_dict, label read, JSON row write, DataLoader hand-off, H2D copy).

## 3. The model half: serial long-sequence WKV (code reading; GPU not profiled)

- Per user the batch is one sample; each stream's entities are length-bucketed splits (`sub_gather_lens`). Example user 4912 (120,123 reviews, 229,384 time steps incl. query rows): card splits (8,436 x 19) ... (10 x 375); deck (14 x 423) ... (1 x 57,254); preset (1 x 227,596); user (1 x 229,384).
- Layers per stream (w64 = w80 recipe): card 2, deck 4, note 1, preset 3, user 3. Streams run one after another (interleaved by layer, still sequential).
- `small_model_wkv.wkv_long` launches grid (B*H,) of K threads: for the preset and user streams B = 1, so **4 blocks of 16 threads** walk T ~ 1.9 x reviews steps serially, 3 layers each; the deck stream adds up to 4 more layers over its longest deck. That is ~13-19 dependent WKV steps per review, on 4 of 84 SMs.
- Measured `model_seconds` is ~10-13 µs per review per variant from 15k to 570k reviews (flat), i.e. throughput-bound by these chains rather than by launch overhead; at ~0.6-0.8 µs per serial step the chains alone explain it. Track S05 (2026-10-16) reached the same diagnosis ("batch-1-recurrence bound") but padded users into one batch, which made long users slower (0.78x).
- With +s8/+s4 state rounding every step the recurrence cannot be chunk-parallelised over T (no exact parallel scan), so the exact lever is concurrency across independent chains.

## 4. Ranked opportunities (expected gain x effort)

1. **Numpy metrics and prediction collection** (implemented: `scripts/heldout_fast_metrics.py`). Exact (bitwise on CPU). Removes ~7.9 of ~22 µs per review-evaluation: projected ~1.5x on the whole loop (pending the timing ticket). Effort small.
2. **Concurrent users on CUDA streams, `--lanes N`** (implemented in `small_model_heldout.py`). Each lane is a thread with its own model copies and CUDA stream; per-user kernels and inputs are unchanged, so rows should be bitwise equal (the ticket checks). Overlaps the serial chains of N users (the GPU is mostly idle during them) and the CPU work of one lane with the GPU work of the others. GPU memory: ~25 KB per review at peak (SM45: 501k reviews fit under the 0.8 cap, 531k did not), so lanes share a budget of in-flight reviews (`--lane-review-budget`, default 400k; a larger user runs alone). +vq variants run one at a time (process-global `state_vq.SCORE`). Projected 1.5-3x on top of item 1, depending on GIL and memory-allocator contention (pending).
3. Variants of one user as extra batch rows (same weights scheme, different state bits): needs per-row bits in the WKV kernel and the token-shift rounding; same concurrency effect as lanes for multi-variant grids, more invasive. Not started.
4. Fused rank-r codec kernel for SM46-type recipes (the torch step loop, ~0.95M codec steps x 3-8 ms on the 732 pool): large gain for that recipe only (hours -> minutes), large effort and a new exactness check. Already listed as SM46-vq; not started here.
5. Per-step latency of `wkv_long` (independent accumulators for the 16-term dot products, `rcp` instead of 16 divisions in the rounding): changes summation/rounding order, so not exact; only under the evaluation tolerance with a reference re-check. Not started.
6. Small leftovers: `load_state_dict` per variant (12 ms), `.item()` syncs (5 per forward), label DB `torch.load` per user. < 2% together.
