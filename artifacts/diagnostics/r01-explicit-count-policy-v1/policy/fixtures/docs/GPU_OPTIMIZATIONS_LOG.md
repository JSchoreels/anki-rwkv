# GPU retraining run log

<!-- r01-log-format:start -->
One row records each training attempt, including failed attempts and functional smoke runs. Each value and each percentage delta has its own column. Losses are held-out binary logloss; User is the unweighted user mean and Weighted is the label-weighted mean. Percentages use `100 × (run / reference − 1)`, rounded to two decimal places; positive speed deltas are faster and positive loss deltas are worse.

Previous means the recorded previous accepted run of the same model. Baseline is the fixed original no-prefetch baseline for that model. Repeated controls do not replace the original baseline, and rejected or fastmode-only candidates do not become the next main reference. Missing values and unavailable comparisons are `—`.

The fixed nested cohorts and 20% held-out split are recorded in [GPU_OPTIMIZATIONS.md](GPU_OPTIMIZATIONS.md#current-loop-fixed-cohorts-and-iteration-0002-protocol). Training users reached are actual counts; incomplete runs retain their planned counts in Notes. Training reviews count raw review presentations. Training seconds include loads, preparation, worker startup/shutdown, transfers, updates and logs; setup/primer, checkpoint save and evaluation are separate. Incomplete rows show partial elapsed training time and no final throughput. Full-history runs remain preliminary screens; smoke runs are infrastructure checks. Process-start provenance and source snapshots are preserved in the linked raw results and iteration records.
<!-- r01-log-format:end -->

<!-- r01-0002:start -->
## R01 iteration 0002

[Iteration record](../artifacts/retraining/gpu-retrain-r01/iteration-0002/protocol.json)

| Training start (UTC) | Trial commit | Model | Improvement IDs | What was tried | Training users reached | Training reviews | Training seconds | Reviews/s | Speed Δ vs previous | Speed Δ vs baseline | Instant User | Instant User Δ vs previous | Instant User Δ vs baseline | Instant Weighted | Instant Weighted Δ vs previous | Instant Weighted Δ vs baseline | Curve User | Curve User Δ vs previous | Curve User Δ vs baseline | Curve Weighted | Curve Weighted Δ vs previous | Curve Weighted Δ vs baseline | Raw result | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-01-01T00:00:00+00:00 | `synthetic-only` | Alex (default) | Baseline | Baseline with prefetch across users disabled. | 80 | 960 | 9.600 | 100 | — | — | 0.500000000 | — | — | 0.500000000 | — | — | 0.500000000 | — | — | 0.500000000 | — | — | [result](../artifacts/retraining/gpu-retrain-r01/iteration-0002/baseline/control/result.json) | Original baseline. |
| 2026-01-01T00:00:00+00:00 | `synthetic-only` | Expertium (light) | Baseline | Baseline with prefetch across users disabled. | 80 | 960 | 9.600 | 100 | — | — | 0.500000000 | — | — | 0.500000000 | — | — | 0.500000000 | — | — | 0.500000000 | — | — | [result](../artifacts/retraining/gpu-retrain-r01/iteration-0002/expertium/control/result.json) | Original baseline. |
| 2026-01-01T01:00:00+00:00 | `synthetic-only` | Alex (default) | F01 | Prepare the next user while the current user trains, preserving ordered updates and review exposure. | 80 | 960 | 8.000 | 120 | +20.00% | +20.00% | 0.500000000 | +0.00% | +0.00% | 0.500000000 | +0.00% | +0.00% | 0.500000000 | +0.00% | +0.00% | 0.500000000 | +0.00% | +0.00% | [result](../artifacts/retraining/gpu-retrain-r01/iteration-0002/baseline/F01/result.json) | — |
| 2026-01-01T01:00:00+00:00 | `synthetic-only` | Expertium (light) | F01 | Prepare the next user while the current user trains, preserving ordered updates and review exposure. | 80 | 960 | 8.000 | 120 | +20.00% | +20.00% | 0.500000000 | +0.00% | +0.00% | 0.500000000 | +0.00% | +0.00% | 0.500000000 | +0.00% | +0.00% | 0.500000000 | +0.00% | +0.00% | [result](../artifacts/retraining/gpu-retrain-r01/iteration-0002/expertium/F01/result.json) | — |

<!-- r01-0002:end -->

<!-- r01-0003:start -->
## R01 iteration 0003

[Iteration record](../artifacts/retraining/gpu-retrain-r01/iteration-0003/summary.md)

| Training start (UTC) | Trial commit | Model | Improvement IDs | What was tried | Training users reached | Training reviews | Training seconds | Reviews/s | Speed Δ vs previous | Speed Δ vs baseline | Instant User | Instant User Δ vs previous | Instant User Δ vs baseline | Instant Weighted | Instant Weighted Δ vs previous | Instant Weighted Δ vs baseline | Curve User | Curve User Δ vs previous | Curve User Δ vs baseline | Curve Weighted | Curve Weighted Δ vs previous | Curve Weighted Δ vs baseline | Raw result | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-01-02T00:00:00+00:00 | `synthetic-only` | Alex (default) | F01, F08 | Prepare the next user while the current user trains, preserving ordered updates and review exposure. Fuse time-shift gathering and eight interpolation operations on the GPU. | 80 | 960 | 6.667 | 144 | +20.00% | +44.00% | 0.500000000 | +0.00% | +0.00% | 0.500000000 | +0.00% | +0.00% | 0.500000000 | +0.00% | +0.00% | 0.500000000 | +0.00% | +0.00% | [result](../artifacts/retraining/gpu-retrain-r01/iteration-0003/baseline/F01-F08/result.json) | — |
| 2026-01-02T00:00:00+00:00 | `synthetic-only` | Expertium (light) | F01, F08 | Prepare the next user while the current user trains, preserving ordered updates and review exposure. Fuse time-shift gathering and eight interpolation operations on the GPU. | 80 | 960 | 6.667 | 144 | +20.00% | +44.00% | 0.500000000 | +0.00% | +0.00% | 0.500000000 | +0.00% | +0.00% | 0.500000000 | +0.00% | +0.00% | 0.500000000 | +0.00% | +0.00% | [result](../artifacts/retraining/gpu-retrain-r01/iteration-0003/expertium/F01-F08/result.json) | — |

<!-- r01-0003:end -->

<!-- r01-fastmode:start -->
## R01 iteration fastmode

[Iteration record](../artifacts/retraining/gpu-retrain-r01/iteration-fastmode/summary.md)

| Training start (UTC) | Trial commit | Model | Improvement IDs | What was tried | Training users reached | Training reviews | Training seconds | Reviews/s | Speed Δ vs previous | Speed Δ vs baseline | Instant User | Instant User Δ vs previous | Instant User Δ vs baseline | Instant Weighted | Instant Weighted Δ vs previous | Instant Weighted Δ vs baseline | Curve User | Curve User Δ vs previous | Curve User Δ vs baseline | Curve Weighted | Curve Weighted Δ vs previous | Curve Weighted Δ vs baseline | Raw result | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-01-02T00:00:00+00:00 | `synthetic-only` | Alex (default) | F01, F08 | Prepare the next user while the current user trains, preserving ordered updates and review exposure. Fuse time-shift gathering and eight interpolation operations on the GPU. | 80 | 960 | 6.400 | 150 | +25.00% | +50.00% | 0.502500000 | +0.50% | +0.50% | 0.502500000 | +0.50% | +0.50% | 0.502500000 | +0.50% | +0.50% | 0.502500000 | +0.50% | +0.50% | [result](../artifacts/retraining/gpu-retrain-r01/iteration-fastmode/baseline/F01-F08/result.json) | — |

<!-- r01-fastmode:end -->

<!-- r01-retune:start -->
## R01 iteration retune

[Iteration record](../artifacts/retraining/gpu-retrain-r01/iteration-retune/summary.md)

| Training start (UTC) | Trial commit | Model | Improvement IDs | What was tried | Training users reached | Training reviews | Training seconds | Reviews/s | Speed Δ vs previous | Speed Δ vs baseline | Instant User | Instant User Δ vs previous | Instant User Δ vs baseline | Instant Weighted | Instant Weighted Δ vs previous | Instant Weighted Δ vs baseline | Curve User | Curve User Δ vs previous | Curve User Δ vs baseline | Curve Weighted | Curve Weighted Δ vs previous | Curve Weighted Δ vs baseline | Raw result | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-01-02T00:00:00+00:00 | `synthetic-only` | Alex (default) | F01, F08 | Prepare the next user while the current user trains, preserving ordered updates and review exposure. Fuse time-shift gathering and eight interpolation operations on the GPU. | 80 | 960 | 8.000 | 120 | +0.00% | +20.00% | 0.500000000 | +0.00% | +0.00% | 0.500000000 | +0.00% | +0.00% | 0.500000000 | +0.00% | +0.00% | 0.500000000 | +0.00% | +0.00% | [result](../artifacts/retraining/gpu-retrain-r01/iteration-retune/baseline/F01-F08/result.json) | — |

<!-- r01-retune:end -->
