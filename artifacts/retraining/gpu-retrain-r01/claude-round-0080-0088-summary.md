# Claude R01 round: iterations 0080–0088

The accepted preliminary expertium reference is **0083: 187,535 reviews/s**, +2.92% versus the round’s starting accepted reference and +525.01% versus the fixed original baseline. Implementation commit `7fced6223659f08303718bd8e14055f3f4c956a4` was made after the experiments.

Authorization: Go for 9 iterations of R01 !. The selected iteration budget is complete; another round requires a new human request.

## Full-history results

Training seconds and raw review presentations/s include user transitions; complete process seconds also include setup, discarded primer, checkpoint saving and evaluation. Gains below use each candidate’s own recorded previous accepted reference, which can differ between rows. One full timing per candidate does not establish repeatability; broader validation remains pending.

| Iteration | Bundle | Training seconds | Reviews/s | Gain vs prior accepted | Complete process seconds | Queue wait s | Summed worker preparation s | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [0079](iteration-0079/summary.md) | E76-F86-F65c | 42.013 | 182,206 | — | 86.216 | 0.26 | 108.8 | Starting accepted reference |
| [0080](iteration-0080/summary.md) | E79-F65d | 38.207 | 200,356 | +9.96% | 82.439 | 0.51 | 108.1 | retain main improvement; preliminary only |
| [0081](iteration-0081/summary.md) | E80-F65e | 38.292 | 199,913 | -0.22% | 83.841 | 0.74 | 107.4 | retain previous accepted run; speed target missed |
| [0082](iteration-0082/summary.md) | E80-F33b | 38.860 | 196,987 | -1.68% | 87.695 | 0.60 | 108.4 | retain previous accepted run; speed target missed |
| [0083](iteration-0083/summary.md) | E80-F33b-F57-F90 | 40.819 | 187,535 | -6.40% | 81.552 | 0.32 | 107.5 | retain main improvement; preliminary only |
| [0084](iteration-0084/summary.md) | E83-F90b | 65.290 | 117,246 | -37.48% | 108.768 | 0.82 | 105.3 | retain previous accepted run; speed target missed |
| [0085](iteration-0085/summary.md) | E83-F86b | 39.522 | 193,691 | +3.28% | 82.389 | 0.62 | 105.2 | retain previous accepted run; speed target missed |
| [0086](iteration-0086/summary.md) | E83-F33c | 40.675 | 188,198 | +0.35% | 81.491 | 0.35 | 105.7 | retain previous accepted run; speed target missed |
| [0087](iteration-0087/summary.md) | E83-F77a | 37.736 | 202,858 | +8.17% | 73.644 | 1.08 | 107.4 | retain previous accepted run; speed target missed |
| [0088](iteration-0088/summary.md) | E83-F77a-F77b | 40.089 | 190,948 | +1.82% | 82.008 | 0.56 | 107.0 | retain previous accepted run; speed target missed |

**0080, E79-F65d.** F65d moves the exact shift-gather (F26, two per layer instance) and padded row-select (F27/F50, one per split) autograd from Python Functions to the native dispatcher, removing Python backward nodes while launching the same kernels. E79 aliases the accepted 0079 options for Windows path length. Recorded decision: retain main improvement; preliminary only; exact full recorded parity: True. Training throughput changed +9.96% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.51 versus 0.26; summed preparation s: 108.07 versus 108.76; validation s: 26.69 versus 27.05; primer s: 11.89 versus 12.02; worker CPU %: 55.35 versus 50.52; GPU activity %: 74.53 versus 66.38. These observations include machine-state variation and do not independently attribute a component’s effect.

**0081, E80-F65e.** F65e replaces three unfused pointwise chains per layer instance (time-mixer bonus products r*bonus*k and gate g*(out+bonus), channel-mixer relu squared) with native single-launch ops whose backward is native too, using the exact F34 kernels whose Python wrappers previously cost more than they saved. E80 aliases the accepted 0080 options. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -0.22% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.74 versus 0.51; summed preparation s: 107.42 versus 108.07; validation s: 28.36 versus 26.69; primer s: 11.51 versus 11.89; worker CPU %: 53.64 versus 55.35; GPU activity %: 74.66 versus 74.53. These observations include machine-state variation and do not independently attribute a component’s effect.

**0082, E80-F33b.** F33b keeps two queued updates' loss/norm reads outstanding instead of one, so the host can enqueue the next update while a large GPU-bound update is still running rather than blocking on the previous one. E80 aliases the accepted 0080 options. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -1.68% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.60 versus 0.51; summed preparation s: 108.42 versus 108.07; validation s: 28.18 versus 26.69; primer s: 15.43 versus 11.89; worker CPU %: 54.40 versus 55.35; GPU activity %: 74.33 versus 74.53. These observations include machine-state variation and do not independently attribute a component’s effect.

**0083, E80-F33b-F57-F90.** Stacks three small exact changes on accepted 0080 (E80): F33b two outstanding loss/norm reads (0082 stack candidate, +0.55%), F57 RSS sampling and progress JSON every eight users instead of every user, and F90 native WKV checkpoint tile 16 (fewer registers per recompute tile). E80 aliases the accepted 0080 options. Recorded decision: retain main improvement; preliminary only; exact full recorded parity: True. Training throughput changed -6.40% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.32 versus 0.51; summed preparation s: 107.53 versus 108.07; validation s: 27.35 versus 26.69; primer s: 8.30 versus 11.89; worker CPU %: 51.29 versus 55.35; GPU activity %: 73.00 versus 74.53. These observations include machine-state variation and do not independently attribute a component’s effect.

**0084, E83-F90b.** Native exact WKV checkpoint tile 8 instead of 16 on the accepted 0083 stack: the backward recomputes from four times the tile-32 checkpoints with smaller per-thread chunk arrays. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed -37.48% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.82 versus 0.32; summed preparation s: 105.34 versus 107.53; validation s: 26.74 versus 27.35; primer s: 11.13 versus 8.30; worker CPU %: 31.27 versus 51.29; GPU activity %: 88.23 versus 73.00. These observations include machine-state variation and do not independently attribute a component’s effect.

**0085, E83-F86b.** Dispatching main process at HIGH_PRIORITY_CLASS instead of above-normal (workers stay above-normal) on the accepted 0083 stack. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed +3.28% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.62 versus 0.32; summed preparation s: 105.23 versus 107.53; validation s: 28.46 versus 27.35; primer s: 9.35 versus 8.30; worker CPU %: 52.71 versus 51.29; GPU activity %: 71.15 versus 73.00. These observations include machine-state variation and do not independently attribute a component’s effect.

**0086, E83-F33c.** Loss/norm/finite reads deferred by three queued updates instead of two on the accepted 0083 stack. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed +0.35% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.35 versus 0.32; summed preparation s: 105.69 versus 107.53; validation s: 27.57 versus 27.35; primer s: 8.17 versus 8.30; worker CPU %: 50.31 versus 51.29; GPU activity %: 71.25 versus 73.00. These observations include machine-state variation and do not independently attribute a component’s effect.

**0087, E83-F77a.** Flush the WDDM software command queue with a stream query after the forward and after the backward on the accepted 0083 stack. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed +8.17% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 1.08 versus 0.32; summed preparation s: 107.39 versus 107.53; validation s: 27.07 versus 27.35; primer s: 3.78 versus 8.30; worker CPU %: 54.91 versus 51.29; GPU activity %: 70.33 versus 73.00. These observations include machine-state variation and do not independently attribute a component’s effect.

**0088, E83-F77a-F77b.** Stack of stack candidate F77a (stream query after forward and backward) with F77b, an identity autograd node that flushes the WDDM queue at each of the five hierarchy-stream boundaries in forward and backward. Recorded decision: retain previous accepted run; speed target missed; exact full recorded parity: True. Training throughput changed +1.82% against its previous accepted reference.

Exposure: 80 training users, 20 held-out users; 7,654,975 training reviews, 868,370 evaluation reviews; 167 updates; 6764613 Curve labels, 6764613 Instant labels. Seed 2025, compute bfloat16, master precision FP32, epochs 1, batch chunks 512, chunk reviews 128. Cohort hash `e1019d37900693bd5cb4f778e10dfc9545cccb1680b73867dd524f206ff6ffc4`. Exposure and schedule differences remain recorded in each comparison; these values must not be inferred from another case.

Resource observations, candidate versus its reference: queue wait s: 0.56 versus 0.32; summed preparation s: 107.02 versus 107.53; validation s: 27.84 versus 27.35; primer s: 9.15 versus 8.30; worker CPU %: 51.34 versus 51.29; GPU activity %: 68.77 versus 73.00. These observations include machine-state variation and do not independently attribute a component’s effect.

## Separate held-out binary losses

| Iteration | Metric | Loss | Delta vs prior accepted | Delta vs original baseline |
|---|---|---:|---:|---:|
| 0080 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0080 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0080 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0080 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0081 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0081 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0081 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0081 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0082 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0082 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0082 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0082 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0083 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0083 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0083 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0083 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0084 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0084 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0084 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0084 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0085 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0085 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0085 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0085 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0086 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0086 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0086 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0086 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0087 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0087 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0087 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0087 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |
| 0088 | Curve label weighted | 0.3874726840790628 | +0.00% | -3.25% |
| 0088 | Curve user averaged | 0.3703879721721576 | +0.00% | -4.48% |
| 0088 | Instant label weighted | 0.3984285960019747 | +0.00% | -0.76% |
| 0088 | Instant user averaged | 0.3825924635443113 | +0.00% | -0.99% |

## Checks and provenance

Full recorded parity is exact in 9/9 cases. The recorded acceptance gates and decisions are reused; this report does not change them or substitute parity for broader quality validation.

- 0080: experiment `a58fd9223ba8646ac947223bed9cab70851adb97`, origin `a2484e487212c8e5e4e50f03093b9724d27ed7d4`, dirty at process start: True; [source archive](../../sources/experiment-sources/49e3553559ff6a0ac1e82c8ccc71c6256d0b74aafb30edf806ad43da22c28ed2/source.zip), [protocol](iteration-0080/protocol.json), [comparison](iteration-0080/comparison.json).
- 0080 pre-run background: 8.0% total CPU; 0 other Python GPU processes detected.
- 0081: experiment `e96f138072e27b3bf133006d9b9c6a056e5f9626`, origin `56a564481d19838d48cf91028ce58967389466d9`, dirty at process start: True; [source archive](../../sources/experiment-sources/eedf7e9a23fb6e432709d9df58cc4f5f942a534bd05293d80a70eee534eb27fc/source.zip), [protocol](iteration-0081/protocol.json), [comparison](iteration-0081/comparison.json).
- 0081 pre-run background: 7.2% total CPU; 0 other Python GPU processes detected.
- 0082: experiment `ab18ebcbbf5012a1e6d65be6449584c722da3f73`, origin `9722e9f6b054251d72f23ac091e23066b8bba615`, dirty at process start: True; [source archive](../../sources/experiment-sources/f01e82b1418e05cd664c19130b6ed295143b2680b0179cef5f6773a988e7ab11/source.zip), [protocol](iteration-0082/protocol.json), [comparison](iteration-0082/comparison.json).
- 0082 pre-run background: 7.6% total CPU; 0 other Python GPU processes detected.
- 0083: experiment `125d0f864e1df4c0c526f25a6b080c6472512b80`, origin `0ace6a89b656ae4d5a2054904ea4c0e23d0f3d6f`, dirty at process start: True; [source archive](../../sources/experiment-sources/53dbdf56a59a017896f9bbe09a8652585f6c3ab8e024cfa6dc874db6a7eff929/source.zip), [protocol](iteration-0083/protocol.json), [comparison](iteration-0083/comparison.json).
- 0083 pre-run background: 6.7% total CPU; 0 other Python GPU processes detected.
- 0084: experiment `85fdb0ae3c65a46e759830f145bfdf8da9fa609c`, origin `26b6e3638a17b81c9adcd309221fbe95c71d841c`, dirty at process start: True; [source archive](../../sources/experiment-sources/c31a10d18dda44f7ef29c222acfdb15985e14b9299560fbc6a848f96ca6e5646/source.zip), [protocol](iteration-0084/protocol.json), [comparison](iteration-0084/comparison.json).
- 0084 pre-run background: 7.9% total CPU; 0 other Python GPU processes detected.
- 0085: experiment `da54bb00efb78c11aedf9d01f2fc56bbd60ddf56`, origin `26b6e3638a17b81c9adcd309221fbe95c71d841c`, dirty at process start: True; [source archive](../../sources/experiment-sources/c40a81958988ffaeb57037a9139eb9e28f749d21053aa70eb1081a82354270b6/source.zip), [protocol](iteration-0085/protocol.json), [comparison](iteration-0085/comparison.json).
- 0085 pre-run background: 7.2% total CPU; 0 other Python GPU processes detected.
- 0086: experiment `cb9eebaab4c9b639ba09892a91a85179be5da8aa`, origin `26b6e3638a17b81c9adcd309221fbe95c71d841c`, dirty at process start: True; [source archive](../../sources/experiment-sources/00779ec0c7f76f500d6d94371117bffa48b1e4d81a065b20e0bcd9f61b2501da/source.zip), [protocol](iteration-0086/protocol.json), [comparison](iteration-0086/comparison.json).
- 0086 pre-run background: 7.8% total CPU; 0 other Python GPU processes detected.
- 0087: experiment `fa72498bf8585aad0743222c428cd0023ba473a7`, origin `a471a27541216fbf80bea999b1d4e6ebebd33927`, dirty at process start: True; [source archive](../../sources/experiment-sources/8e2b85c3730ec8d456693369b685930bdef699bf38605af4500278628bd04870/source.zip), [protocol](iteration-0087/protocol.json), [comparison](iteration-0087/comparison.json).
- 0087 pre-run background: 6.7% total CPU; 0 other Python GPU processes detected.
- 0088: experiment `e0408ff0b7563211d8f4a60f863e144256e04c45`, origin `7fced6223659f08303718bd8e14055f3f4c956a4`, dirty at process start: True; [source archive](../../sources/experiment-sources/241d5f0bcb1e4b83f693ab021d2d37a1ffd6a4076b36607f35d4b38578331a45/source.zip), [protocol](iteration-0088/protocol.json), [comparison](iteration-0088/comparison.json).
- 0088 pre-run background: 8.2% total CPU; 0 other Python GPU processes detected.

## Interpretation and caveats

- The "Decision" column records the single-run report; the replicated-block rule (per-update gains, Pocock looks 2/4/6) governs. Block outcomes: 0084 F90b reject (−1.03%, lower −2.42%), 0085 F86b reject (−0.34%), 0086 F33c reject (−0.24%), 0087 F77a stack candidate (+0.52%, lower −0.30% at six), 0088 F77a+F77b stack candidate (+0.27%, lower −0.55% at six). 0083 remains the accepted reference.
- 0083's single full run (187,535 reviews/s) was taken in a slow machine state. Its 18 replicate control runs in 0084–0088 averaged 196,374 reviews/s (median 196,948; range 181,182–205,361), which is the better estimate of its level.
- F90b tile 8 is exact but its full runs were far slower (117,246 and 143,151 reviews/s) than its per-update block effect (−1.03%) suggests: peak reserved memory rose from 12,496 to 13,588 MiB on the 16 GB card, the regime where earlier runs collapsed. The per-update metric does not capture that whole-run cost.
- Step-1 diagnostic: the eager (RWKV_NO_JIT) model was not faster than TorchScript on 0083 options (224.5 vs 222.3 ms/update; forward+loss host stage 30.2 vs 25.2 ms), so F91 was not implemented.

## Pending next steps

- F90c: native WKV with 2 rows per thread at tile 16 (exact template, untested with tile 16)
- Stack F77a with F65e native glue
- Main-thread module walks per update (train()/zero_grad ~2.2 ms) as an exact F72 slice
