# SM44: compressed card/note state on the GPU scorer (~400 and ~800 bits)

Full-history held-out protocol, w8 weights. The compressed rows (`+vq`) were scored on the GPU (SM44: fused kernel with the Gram-squaring factorization in float64 for rank-1 streams, a per-step torch loop for the rank-2 note of vq770x828); the CPU-first check and the GPU check ticket compare that path with the CPU scorer (`check-sm44-cpu.json`, `check-sm44-gpu.json`). Paired, review-weighted benchmark LogLoss; [ ] = 95% user bootstrap (2,000 resamples, seed 0, 2.5/97.5 percentiles). Fork = the 2.76M model, int8 state.

## Pool 732: 722 users (30,126,145 curve test reviews)

**curve** (fork w8+s8 0.29943)

| Label | Card / note bits | LogLoss | vs same model w8+s8 | vs SM42 w8+s8 | vs fork w8+s8 |
|---|---|---:|---:|---:|---:|
| `s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes/w8+s8` (SM42 (best model), int8 state) | int8 | 0.30155 | ref | ref | +0.71% [+0.50, +0.95] |
| `s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes/w8+vq378x412` (SM42, codebooks fitted after training (k-means), ~400 bits) | 378 / 412 | 0.30372 | +0.72% [+0.65, +0.79] | +0.72% [+0.65, +0.79] | +1.43% [+1.21, +1.68] |
| `s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes/w8+vq770x828` (SM42, k-means codebooks, ~800 bits) | 770 / 828 | 0.30265 | +0.37% [+0.32, +0.42] | +0.37% [+0.32, +0.42] | +1.08% [+0.87, +1.31] |
| `s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes-vqlearn/w8+s8` (SM32c (compression trained in, learned codebooks), int8 state) | int8 | 0.30518 | ref | +1.20% [+1.09, +1.31] | +1.92% [+1.69, +2.19] |
| `s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes-vqlearn/w8+vq378x412` (SM32c, its learned codebooks, ~400 bits) | 378 / 412 | 0.30348 | -0.56% [-0.63, -0.48] | +0.64% [+0.57, +0.70] | +1.35% [+1.14, +1.60] |

**instant** (fork w8+s8 0.27584)

| Label | Card / note bits | LogLoss | vs same model w8+s8 | vs SM42 w8+s8 | vs fork w8+s8 |
|---|---|---:|---:|---:|---:|
| `s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes/w8+s8` (SM42 (best model), int8 state) | int8 | 0.27777 | ref | ref | +0.70% [+0.65, +0.75] |
| `s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes/w8+vq378x412` (SM42, codebooks fitted after training (k-means), ~400 bits) | 378 / 412 | 0.28104 | +1.18% [+1.10, +1.25] | +1.18% [+1.10, +1.25] | +1.88% [+1.78, +1.99] |
| `s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes/w8+vq770x828` (SM42, k-means codebooks, ~800 bits) | 770 / 828 | 0.27964 | +0.67% [+0.63, +0.73] | +0.67% [+0.63, +0.73] | +1.38% [+1.30, +1.46] |
| `s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes-vqlearn/w8+s8` (SM32c (compression trained in, learned codebooks), int8 state) | int8 | 0.28195 | ref | +1.50% [+1.41, +1.59] | +2.21% [+2.09, +2.33] |
| `s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes-vqlearn/w8+vq378x412` (SM32c, its learned codebooks, ~400 bits) | 378 / 412 | 0.28057 | -0.49% [-0.54, -0.43] | +1.01% [+0.95, +1.07] | +1.71% [+1.62, +1.81] |

## Pool 464: 458 users (19,833,530 curve test reviews)

**curve** (fork w8+s8 0.29584)

| Label | Card / note bits | LogLoss | vs same model w8+s8 | vs SM42 w8+s8 | vs fork w8+s8 |
|---|---|---:|---:|---:|---:|
| `s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes/w8+s8` (SM42 (best model), int8 state) | int8 | 0.29798 | ref | ref | +0.72% [+0.49, +1.02] |
| `s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes/w8+vq378x412` (SM42, codebooks fitted after training (k-means), ~400 bits) | 378 / 412 | 0.30016 | +0.73% [+0.65, +0.82] | +0.73% [+0.65, +0.82] | +1.46% [+1.20, +1.79] |
| `s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes/w8+vq770x828` (SM42, k-means codebooks, ~800 bits) | 770 / 828 | 0.29910 | +0.38% [+0.32, +0.43] | +0.38% [+0.32, +0.43] | +1.10% [+0.86, +1.41] |
| `s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes-vqlearn/w8+s8` (SM32c (compression trained in, learned codebooks), int8 state) | int8 | 0.30145 | ref | +1.16% [+1.01, +1.30] | +1.90% [+1.63, +2.19] |
| `s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes-vqlearn/w8+vq378x412` (SM32c, its learned codebooks, ~400 bits) | 378 / 412 | 0.29984 | -0.53% [-0.62, -0.44] | +0.62% [+0.53, +0.71] | +1.35% [+1.10, +1.67] |

**instant** (fork w8+s8 0.27223)

| Label | Card / note bits | LogLoss | vs same model w8+s8 | vs SM42 w8+s8 | vs fork w8+s8 |
|---|---|---:|---:|---:|---:|
| `s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes/w8+s8` (SM42 (best model), int8 state) | int8 | 0.27417 | ref | ref | +0.71% [+0.64, +0.79] |
| `s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes/w8+vq378x412` (SM42, codebooks fitted after training (k-means), ~400 bits) | 378 / 412 | 0.27743 | +1.19% [+1.10, +1.29] | +1.19% [+1.10, +1.29] | +1.91% [+1.78, +2.05] |
| `s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes/w8+vq770x828` (SM42, k-means codebooks, ~800 bits) | 770 / 828 | 0.27604 | +0.68% [+0.62, +0.75] | +0.68% [+0.62, +0.75] | +1.40% [+1.29, +1.51] |
| `s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes-vqlearn/w8+s8` (SM32c (compression trained in, learned codebooks), int8 state) | int8 | 0.27830 | ref | +1.51% [+1.40, +1.62] | +2.23% [+2.07, +2.40] |
| `s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes-vqlearn/w8+vq378x412` (SM32c, its learned codebooks, ~400 bits) | 378 / 412 | 0.27691 | -0.50% [-0.56, -0.44] | +1.00% [+0.93, +1.08] | +1.72% [+1.59, +1.84] |

## Checks and speed

- SM42 w8+s8 re-scored by SM44 vs its earlier rows (wt/sm42): 723 users, max |dLogLoss| curve 0.00e+00, instant 0.00e+00.
- Rows per label: {'s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes/w8+s8': 729, 's-w64kd-gate-ctx32768c-dec10-1ep32k-lanes/w8+vq378x412': 722, 's-w64kd-gate-ctx32768c-dec10-1ep32k-lanes/w8+vq770x828': 722, 's-w64kd-gate-ctx32768c-dec10-1ep32k-lanes-vqlearn/w8+s8': 723, 's-w64kd-gate-ctx32768c-dec10-1ep32k-lanes-vqlearn/w8+vq378x412': 722}; fork rows 731.
- GPU scorer model time (review-evals per second of model time, all SM44 rows of the label): `s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes/w8+s8` 92,374/s (723 users); `s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes/w8+vq378x412` 46,828/s (722 users); `s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes/w8+vq770x828` 14,473/s (722 users); `s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes-vqlearn/w8+s8` 91,520/s (723 users); `s-w64kd-gate-ctx32768c-dec10-1ep32k-lanes-vqlearn/w8+vq378x412` 46,596/s (722 users)
- GPU check timing (near-median users): {'w8+s8': 55124.0, 'w8+vq378x412': 36494.3, 'w8+vq770x828': 10590.4} reviews/s.
