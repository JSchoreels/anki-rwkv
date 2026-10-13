# F69 chunked (parallel-scan) WKV7 prototype (r01-structural, 2026-10-13)

**Recommendation: park F69 (do not spend a counted slot, do not write a fused kernel yet).** The chunked algebra is correct and as accurate as the exact kernel against a float64 truth, but it is not bitwise, so it can only enter through the P-B multi-seed gate, and the eager PyTorch version is slower than the native kernel on every real shape. Whether a fused kernel could win is an open projection (see the end); the evidence does not yet justify building it. Not wired into training.

## What was built (new files only)

- [rwkv/model/chunked_wkv.py](../../../rwkv/model/chunked_wkv.py): chunked forward in plain PyTorch. Per chunk of C steps one unit-lower-triangular solve (`solve_triangular`) replaces C sequential steps; chunk boundaries follow `S_end = S0 P + U` (N = T/C sequential K×K products). Skipped steps use masked carried parameters while the output still uses the candidate state, as the kernels do. Backward is the autograd derivative of this forward (no hand-written backward). Factorised `exp(-L)` terms are stable while the cumulative `-log w` over a chunk stays below about 30 in float32.
- [scripts/check_r01_chunked_wkv.py](../../../scripts/check_r01_chunked_wkv.py) and [scripts/bench_r01_chunked_wkv.py](../../../scripts/bench_r01_chunked_wkv.py).

## Exactness and accuracy (not bitwise)

- CPU float64 identity against the sequential recurrence, all outputs and all six gradients, chunk 8/16/32, skip none/random/all: max abs error **5.3e-15** ([check-cpu-1](check-cpu-1/result.json)).
- **Real batches** ([check-4-real](check-4-real/result.json)): five real Expertium batches (users from the 100-user cohort, 152 recurrence calls, 51 distinct shapes), each call replayed with its real inputs and its real output gradient, captured at the NVRTC launch. A fresh checkpoint zero-initialises output projections, so 218 zero parameters were perturbed (seeded, N(0, 0.02)) to make gradients non-zero; the first attempts with zero gradients ([check-3-real](check-3-real/result.json)) are kept and are not evidence. Real decay is in [0.891, 1), cumulative `-log w` per 16-step chunk at most 1.77. Maximum over all shapes, relative to each tensor's absmax, for chunk 32 (chunk 16 is similar):

| tensor | chunked vs exact (BF16 ABI) | elements unequal | exact vs fp64 truth | chunked fp32 vs fp64 truth |
|---|---:|---:|---:|---:|
| out | 4.1e-3 | 0.054% | 3.4e-3 | 5.3e-7 |
| dr | 1.9e-3 | 0.036% | 3.1e-3 | 4.4e-7 |
| dk | 2.3e-3 | 0.028% | 3.6e-3 | 3.6e-7 |
| dv | 2.5e-3 | 0.025% | 3.5e-3 | 3.6e-7 |
| dw | 1.6e-6 | 85% | 5.9e-7 | 6.4e-7 |
| da | 2.3e-3 | 0.031% | 3.7e-3 | 5.5e-7 |
| dkd | 3.1e-3 | 0.039% | 3.7e-3 | 7.1e-7 |

  So the differences from the exact kernel are one-BF16-ulp flips on about 0.03 to 0.05% of elements (truth is only available for the first batch; truth fields are maxima over the replays that had one). In the compute precision the chunked path matches the float64 truth to about 5e-7, the same scale as the exact kernel's own error (dw 5.9e-7). It is not bitwise, so a candidate would need P-B (or the unchanged single-seed 0.1% gate, where F22 saw +0.84% to +1.48% single-seed shifts for a different reordering); this session does not argue either.
- **Synthetic CUDA grid** ([check-1](check-1/result.json); BF16 payload, `check_gpu_row_tiles.synthetic` inputs): `out`, `dr`, `dk`, `dv`, `da`, `dkd` behave as above, but **dw fails when the cumulative decay is extreme** (synthetic `w` reaches 1e-9; cumulative `-log w` 34 to 88 per chunk): the exp(-L) factorisation loses all precision (relative error 1e-2 up to 1e20). Real decay never gets near this (maximum 1.77), but any other decay parametrisation, or a model that learns smaller `w`, needs a clamp or the pairwise formulation. Records of the failure are kept in the result file.

## Speed (microbenchmark, eager PyTorch, no training; [bench-1](bench-1/result.json), [bench-2-chunk64](bench-2-chunk64/result.json))

Nine real E33 split shapes, synthetic inputs with random skips, 5 warm-up and 30 repeats, CUDA-event device time (median), forward + backward:

| shape (B×T) | native ms | chunk 16 | chunk 32 |
|---|---:|---:|---:|
| 742×256 | 7.45 | 21.5 | 12.0 |
| 869×256 | 9.49 | 25.4 | 14.0 |
| 512×256 | 5.60 | 13.2 | 7.6 |
| 36177×2 | 2.99 | 41.0 | 65.0 |
| 7764×4 | 0.95 | 7.2 | 12.2 |
| 3359×10 | 1.07 | 3.1 | 4.6 |
| 318×26 | 0.37 | 2.2 | 2.0 |
| 695×68 | 2.00 | 4.7 | 3.8 |
| 4240×12 | 1.96 | 4.4 | 6.4 |
| sum | **31.9** | 122.6 | 127.7 |

Chunk 64 is slower than 32 (T=256 shapes, 35.6 vs 33.9 ms for chunk 32, against native 23.3 ms for the three). The prototype is 1.6 times slower than native at T=256 and 3 to 20 times slower on short histories (chunking cannot help T≤4; the host launch cost of about 2 to 3 ms per call alone is 15 times the native host cost of 0.15 ms). Peak allocation and spreads are in the result file.

## Is a fused kernel plausible? (projection, not measured)

Only the three T=256 shapes (62% of the nine-shape native time, 22.5 of 31.9 ms) are candidates. The native kernel runs about 1 TFLOP/s equivalent, latency-bound by 256 sequential steps with warp shuffles; the chunk algebra turns the work into dense 32×32 and chunk×chunk products that could use tensor cores, so a fused kernel might be 2 to 4 times faster than the eager prototype's 12.0 ms. Even a 3 times faster-than-native kernel would save at most about 15 ms of the roughly 230 ms GPU time per large update (about 6%, a projection) and only while GPU-bound, in a regime where the GPU is already about 64% active. That best case sits within the replicated-block timing noise and still needs P-B. The numbers do not support writing the kernel before P-B is approved; revisit if it is.
