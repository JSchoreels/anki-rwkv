# GPU optimization benchmarks at fixed batch sizes

RTX 5080 16 GB, Windows, PyTorch 2.9.1+cu128. Measured on 2026-10-11.
Default: **128 chunks / 4 CPU workers**. Expertium iter36: **256 chunks / 6 CPU workers**. Each chunk contains 128 raw reviews. Batch sizes and worker counts stay fixed across each model’s variants.

Each scenario starts from fresh weights and AdamW state with seed 2025, BF16 computation and FP32 master weights. Inputs are the same user-17 history window (reviews 4097–12288), repeated. Six warmup updates precede 64 measured updates.
Complete throughput includes CPU preparation waits, CPU-to-GPU transfer, forward/backward, gradient clipping and AdamW. File loads, process startup, compilation and graph capture are excluded. This is a throughput benchmark, not a held-out model-quality comparison.

## Results

| Experiment | Default reviews/s | Expertium reviews/s |
|---|---:|---:|
| Control before | 41,957 | 70,617 |
| Control after | 42,787 | 64,865 |
| Vectorized CPU preparation | 41,952 | 78,375 |
| GPU encodings (unpacked) | 39,066 | 76,926 |
| GPU encodings (packed) | 42,203 | 85,478 |
| RWKV tile 16 | 43,149 | 65,421 |
| RWKV tile 8 | 41,444 | 65,547 |
| Fused casts/copies | 43,552 | 64,886 |
| Direct CUDA pointer launches | 43,214 | 65,569 |
| Fused AdamW | 42,774 | 65,140 |
| CUDA Graphs | 45,909 | 71,955 |
| Combined | 47,728 | 83,211 |
| GPU packed + graphs | failed / not run | 87,153 |

## Interpretation

- baseline: best **47,728 reviews/s**, 12.6% above the mean of its two controls (42,372 reviews/s), using `combined`.
- expertium: best **87,153 reviews/s**, 28.7% above the mean of its two controls (67,741 reviews/s), using `combined-gpu-packed`.

The controls expose run-to-run variation. Small differences near that range should not be treated as established speedups. Expertium is constrained by CPU preparation; its transfer-plus-training rate can improve without a comparable rise in complete throughput.

CUDA Graphs capture forward, loss and backward together. The first capture attempt failed because backward joined the legacy stream. Recreating parameter leaves on the same non-default stream as capture resolved it; deterministic output and gradient checks pass for both models. This matches the kind of stream dependency discussed in [PyTorch issue 164302](https://github.com/pytorch/pytorch/issues/164302).

Graph timing applies to one repeated shape. Full histories need a bounded graph cache or shape buckets before this can be used throughout production training. Finite loss and gradient checks execute after replay, before optimizer updates. Dropout is enabled during throughput runs; parity fixtures disable dropout and perturb initially-zero parameters to exercise their gradients.

Smaller RWKV checkpoint tiles reduce per-thread local storage (32: 256 bytes; 16: 128 bytes; 8: 64 bytes) but increase saved checkpoints. Review chunk and optimizer batch sizes do not change. GPU feature preparation preserves CPU RNG draw order and moves lookup/calendar arithmetic to CUDA; the packed variant consolidates metadata in workers before IPC.

Graph-private pools retain memory after temporary tensors are released. Use **reserved memory** rather than the much smaller live-tensor allocation counter when comparing graph memory footprints.

## Validation and saved artifacts

- Vectorized CPU preparation matches original inputs exactly across nine windows from users 1, 17 and 92, including partial chunks. GPU feature variants match within 1e-6 absolute / 1e-5 relative tolerance; indexes and labels match exactly.
- Float32/BF16 recurrence outputs and six input gradients match the CPU reference at 256 timesteps, for one/four heads and checkpoint tiles 8/16/32.
- Direct casts and gradient copies preserve values exactly. CUDA Graph output/gradient parity checks are stored with graph benchmark results.
- All successful scenarios performed 64 measured optimizer updates with finite final losses and gradient norms. Review rates were independently recomputed from timings.
- Power monitoring reports no cap changes or monitoring errors. The configured 400 W cap was left unchanged.
- The completed 100-user training checkpoint remains separate under `checkpoints/gpu-trial/`.

[Full comparison JSON](../artifacts/benchmarks/perf-optimizations/comparison.json), [parity results](../artifacts/benchmarks/perf-optimizations/verification.json), and per-scenario console/power logs are saved under `artifacts/benchmarks/perf-optimizations/`.

## Held-out log-loss screening

`scripts/check_gpu_logloss.py` adds a separate paired quality screen for every optimization and both models. It trains on user 17 reviews 4097–12288 and evaluates on later reviews 12289–20480. Next-review targets are confined to their split. Each variant starts from identical fresh weights and optimizer state for seeds 2025, 2026 and 2027; batch sizes stay fixed. Training repeats the 8192-review window for 6 warmup and 32 further optimizer updates with dropout enabled.

Evaluation uses the original prepared features, dropout disabled, the updated FP32 master weights and standard WKV tile32 for every variant. It reports curve and instant binary log loss separately. Initial-parameter hashes, validation-input hashes and label counts must match each paired control.

Initial controls and repeated controls after a discarded JIT primer are both retained. This exposes changes from cold profiling and run variation; the per-variant deltas use the repeated control when available. A positive delta means worse loss. The 0.001 flag is a screening threshold, not a confidence interval or a guarantee of unchanged accuracy.

This short one-user screen does not establish quality preservation for complete 100-user training. Numerical parity and finite training losses alone also do not establish that. See the [quality report](../artifacts/benchmarks/perf-optimizations/logloss/summary.md) and [paired results](../artifacts/benchmarks/perf-optimizations/logloss/comparison.json) before selecting a faster configuration.

All **75 runs completed**, with matching paired initial weights, validation inputs and label counts. Against the repeated controls, the previously fastest configurations have these mean deltas across three seeds:

| Fast configuration | Curve log-loss delta | Instant log-loss delta |
|---|---:|---:|
| Default: graphs + copies | +0.001366 | +0.000845 |
| Expertium: packed GPU features + graphs | -0.008127 | -0.007432 |

The default combination's worst seed increased curve loss by 0.008303 and instant loss by 0.019155. The Expertium combination improved both losses in all three tested seeds. Repeated controls themselves changed by up to 0.024974 curve / 0.021929 instant loss, so these short trials do not establish statistical improvement or non-inferiority. The default graph combination is flagged by the screen; its throughput gain should not be described as demonstrated loss-neutral.

BF16 input checks found **762 changed values in unpacked GPU preparation** and **zero in packed GPU preparation**, across 338,928 values from nine real-history windows. Enumerating every integer remainder for all seven calendar periods also found zero packed differences after BF16 casting, across 81,310 sine/cosine values. Packed preparation preserves the original CPU frequency rounding before GPU multiplication. Unpacked preparation does not preserve BF16 inputs exactly and remains an experimental variant. These input checks are separate from predictive-quality validation.

## Reproduction

Run the quality screen with `just check-gpu-logloss`; use `python scripts/check_gpu_logloss.py --in-process --resume` to continue saved runs with the same seeds and update count.

```powershell
.\.venv\Scripts\python.exe scripts/check_gpu_optimizations.py
.\.venv\Scripts\python.exe scripts/run_gpu_optimizations.py
.\.venv\Scripts\python.exe scripts/benchmark_gpu_optimizations.py --model baseline --variant combined --overrides '{"CUDA_GRAPH":true,"FAST_COPIES":true}'
.\.venv\Scripts\python.exe scripts/benchmark_gpu_optimizations.py --model expertium --variant combined --overrides '{"CUDA_GRAPH":true,"PREPARATION":"cpu"}'
.\.venv\Scripts\python.exe scripts/benchmark_gpu_optimizations.py --model expertium --variant combined-gpu-packed --overrides '{"CUDA_GRAPH":true,"PREPARATION":"gpu-packed"}'
```

Use the NVRTC environment from [the GPU setup guide](GPU_TRAINING.md), including its workspace cache/temp paths. `just benchmark-gpu-optimizations` runs the isolated variants and selected combinations. Benchmark scripts leave shutdown under user control.

<!-- optimization-source:start -->
## Experiment source revisions

Commit IDs below come from saved per-run metadata. Missing historical revisions stay unrecorded. Dirty runs also require the linked source snapshot; implementation commits are recorded separately in the change ledger.

| Suite | Runs | Cases | Experiment commit | Working tree | Source snapshot |
|---|---:|---|---|---|---|
| Throughput | 26 | baseline/cast-fusion, baseline/combined, baseline/cuda-graph, baseline/fused-adamw, baseline/preparation-cpu, baseline/preparation-gpu, baseline/preparation-gpu-packed, baseline/raw-launch, baseline/reference, baseline/reference-repeat, baseline/wkv-tile16, baseline/wkv-tile8, expertium/cast-fusion, expertium/combined, expertium/combined-gpu-packed, expertium/cuda-graph, expertium/fused-adamw, expertium/preparation-cpu, expertium/preparation-gpu, expertium/preparation-gpu-packed, expertium/raw-launch, expertium/reference, expertium/reference-repeat, expertium/wkv-tile16, expertium/wkv-tile8 | — (not recorded) | — | — |
| Log loss | 75 | baseline/cast-fusion, baseline/combined, baseline/cuda-graph, baseline/fused-adamw, baseline/preparation-cpu, baseline/preparation-gpu, baseline/preparation-gpu-packed, baseline/raw-launch, baseline/reference, baseline/reference-repeat, baseline/wkv-tile16, baseline/wkv-tile8, expertium/cast-fusion, expertium/combined, expertium/combined-gpu-packed, expertium/cuda-graph, expertium/fused-adamw, expertium/preparation-cpu, expertium/preparation-gpu, expertium/preparation-gpu-packed, expertium/raw-launch, expertium/reference, expertium/reference-repeat, expertium/wkv-tile16, expertium/wkv-tile8 | — (not recorded) | — | — |

<!-- optimization-source:end -->

<!-- optimization-quality:start -->
## Held-out log loss and training time

Curve and Instant binary log loss are reported separately (lower is better). Elapsed measured training time and raw reviews/s describe the same runs as the validation scores. The composite training loss is not log loss.

These one-user results are a preliminary regression screen. Marking a change or bundle as validated requires a separately requested multi-user comparison; the validation gate and agreed minimum cohort size are recorded below.

Suite status: **complete**. Seeds: 2025, 2026, 2027. Each run performs 6 training warmup updates plus **32 measured updates**.

User 17: train on reviews **4097–12288** and validate on later reviews **12289–20480**, with next-review targets confined to their split. The 8,192-review training window is repeated. Validation uses common original features, dropout disabled, FP32 master weights and WKV tile32. This is a bounded quality regression screen.

Times include CPU preparation waits, transfer, forward/backward, copies, gradient clipping and AdamW. They exclude file loads, process/model startup, compilation, discarded JIT primers, graph capture, training warmup and validation. Values are arithmetic means over the listed paired seeds; training seconds cover only the measured updates, while final losses include learning during warmup.

### Default

BF16 computation, FP32 master weights; **128 chunks / 4 workers**. Each run measures **524,288 raw review presentations** after **98,304 warmup presentations**. These are repeated exposures, not unique reviews.

| Experiment | Paired seeds / status | Curve LL | Instant LL | Δ Curve | Δ Instant | Measured s | Reviews/s |
|---|---|---:|---:|---:|---:|---:|---:|
| `reference` | 3/3 | 0.559129 | 0.529757 | -0.002111 | -0.002191 | 11.15 | 47,099 |
| `reference-repeat` | 3/3 | 0.561240 | 0.531949 | +0.000000 | +0.000000 | 10.47 | 50,094 |
| `preparation-cpu` | 3/3 | 0.561240 | 0.531949 | -0.000000 | -0.000000 | 10.78 | 48,672 |
| `preparation-gpu` | 3/3 | 0.555888 | 0.520946 | -0.005353 | -0.011002 | 11.72 | 44,781 |
| `preparation-gpu-packed` | 3/3 | 0.561240 | 0.531949 | -0.000000 | -0.000000 | 10.59 | 49,517 |
| `wkv-tile16` | 3/3 | 0.561240 | 0.531949 | -0.000000 | -0.000000 | 10.60 | 49,459 |
| `wkv-tile8` | 3/3 | 0.561240 | 0.531949 | -0.000000 | -0.000000 | 10.95 | 47,868 |
| `cast-fusion` | 3/3 | 0.561240 | 0.531949 | -0.000000 | -0.000000 | 10.45 | 50,184 |
| `raw-launch` | 3/3 | 0.561383 | 0.532186 | +0.000143 | +0.000237 | 10.53 | 49,800 |
| `fused-adamw` | 3/3 | 0.553764 | 0.522158 | -0.007476 | -0.009791 | 11.05 | 47,512 |
| `cuda-graph` | 3/3 | 0.561039 | 0.532808 | -0.000201 | +0.000860 | 10.56 | 50,138 |
| `combined` | 3/3 | 0.562606 | 0.532794 | +0.001366 | +0.000845 | 9.97 | 52,757 |

Held-out labels per run: **5,097 Curve / 5,902 Instant**.
Controls used for loss deltas: 2025: `reference-repeat`, 2026: `reference-repeat`, 2027: `reference-repeat`. Prefer the repeated control after JIT priming; deltas against the first control remain preliminary until its repeat is available.

### Expertium iter36

BF16 computation, FP32 master weights; **256 chunks / 6 workers**. Each run measures **1,048,576 raw review presentations** after **196,608 warmup presentations**. These are repeated exposures, not unique reviews.

| Experiment | Paired seeds / status | Curve LL | Instant LL | Δ Curve | Δ Instant | Measured s | Reviews/s |
|---|---|---:|---:|---:|---:|---:|---:|
| `reference` | 3/3 | 0.500176 | 0.525537 | -0.008559 | -0.007844 | 13.70 | 78,231 |
| `reference-repeat` | 3/3 | 0.508735 | 0.533381 | +0.000000 | +0.000000 | 13.89 | 75,662 |
| `preparation-cpu` | 3/3 | 0.500262 | 0.526286 | -0.008473 | -0.007095 | 12.95 | 81,967 |
| `preparation-gpu` | 3/3 | 0.499770 | 0.526002 | -0.008965 | -0.007380 | 13.11 | 80,195 |
| `preparation-gpu-packed` | 3/3 | 0.511666 | 0.535942 | +0.002931 | +0.002560 | 12.70 | 83,174 |
| `wkv-tile16` | 3/3 | 0.501119 | 0.526567 | -0.007616 | -0.006814 | 15.50 | 68,254 |
| `wkv-tile8` | 3/3 | 0.511096 | 0.535399 | +0.002361 | +0.002018 | 15.53 | 68,000 |
| `cast-fusion` | 3/3 | 0.499083 | 0.524838 | -0.009652 | -0.008543 | 14.65 | 71,781 |
| `raw-launch` | 3/3 | 0.508261 | 0.533182 | -0.000474 | -0.000199 | 15.28 | 68,872 |
| `fused-adamw` | 3/3 | 0.498476 | 0.524659 | -0.010259 | -0.008722 | 14.32 | 73,313 |
| `cuda-graph` | 3/3 | 0.503689 | 0.529043 | -0.005047 | -0.004338 | 14.20 | 74,088 |
| `combined` | 3/3 | 0.499714 | 0.525777 | -0.009021 | -0.007604 | 11.35 | 92,436 |
| `combined-gpu-packed` | 3/3 | 0.500608 | 0.525949 | -0.008127 | -0.007432 | 10.59 | 99,099 |

Held-out labels per run: **5,097 Curve / 5,902 Instant**.
Controls used for loss deltas: 2025: `reference-repeat`, 2026: `reference-repeat`, 2027: `reference-repeat`. Prefer the repeated control after JIT priming; deltas against the first control remain preliminary until its repeat is available.

Positive loss deltas mean worse validation quality than the same model's paired control. Default and Expertium have different batch sizes and twice as many review presentations per update for Expertium; their absolute scores do not establish an architecture ranking at matched training exposure. Equal-time quality and time to a target score remain future measurements (F11).

Repeated controls, when present, are shown alongside the first control. If controls disagree, treat the corresponding quality deltas as inconclusive until setup/RNG effects are resolved; this table records observations rather than establishing unchanged quality.

[Raw paired loss/time results](../artifacts/benchmarks/perf-optimizations/logloss/comparison.json), [per-seed regression summary](../artifacts/benchmarks/perf-optimizations/logloss/summary.md). Regenerate this section with `python scripts/report_training_tradeoffs.py` as runs finish. Missing and failed measurements remain visible.

<!-- optimization-quality:end -->

<!-- optimization-tracks:start -->
## Pending optimization tracks

This is the persistent idea list. Keep each track here before and after testing,
including failed and inconclusive attempts. `—` means unmeasured. Fill results
from matched runs with held-out Curve/Instant log loss, measured training time,
and complete-update raw reviews/s; record the protocol and artifact links.
Priorities below are investigation priorities, not measured speedup estimates.
Parenthesized percentages beside first-batch measurements show their relative
change from the first matched control for that model. Weighted and user-averaged
losses each use their own control denominator. The [absolute-delta table](../artifacts/benchmarks/gpu-batch1-500/summary-relative.md)
also shows training time, throughput and cold-time percentage changes.

| ID | Track | Priority / effort | Status | Curve / Instant log loss | Measured s / reviews/s | Next experiment |
|---|---|---|---|---|---|---|
| F01 | Prefetch across users | High / medium | Default: pending repeated control | Default weighted 0.435782 (+0.017%) / 0.435904 (+3.423%); user 0.436797 (+0.642%) / 0.441539 (+5.027%); Expertium — (pending) | Default 694.26 (-45.64%) s / 38,466 (+83.97%) reviews/s; Expertium — | Load and prepare the next user while training the current user; preserve update order, review coverage, seeds, and resume positions. Measure full-history time including user transitions. |
| F02 | Cache deterministic preprocessing | High / medium | Default: pending repeated control | Default weighted 0.394084 (-9.553%) / 0.364665 (-13.479%); user 0.381996 (-11.985%) / 0.348074 (-17.205%); Expertium — (pending) | Default 985.45 (-22.84%) s / 27,100 (+29.61%) reviews/s; cold 2,480.37 (+94.20%) s; Expertium — | Cache compact features, labels and permutations with source/data invalidation; regenerate stochastic ID/calendar encodings. A bounded prepared-input cache exists in `scripts/compare.py`; extend and measure the full-history path. |
| F03 | Reduce CPU-worker communication | High / medium | Default: pending repeated control | Default weighted 0.432101 (-0.828%) / 0.435618 (+3.355%); user 0.432939 (-0.247%) / 0.440982 (+4.895%); Expertium — (pending) | Default 1,277.19 (-0.00%) s / 20,909 (+0.00%) reviews/s; Expertium — | Use shared arrays or worker-local user data and pass offsets instead of Pandas slices; pack returned tensors. Measure serialization, queue waits, and RAM use. |
| F04 | Overlap transfers and training | Medium / medium | Default: pending repeated control | Default weighted 0.441723 (+1.381%) / 0.419468 (-0.476%); user 0.440374 (+1.466%) / 0.416412 (-0.949%); Expertium — (pending) | Default 1,003.56 (-21.43%) s / 26,611 (+27.27%) reviews/s; Expertium — | Add a background pinned-buffer stage, nonblocking transfers, a separate CUDA copy stream, and two bounded GPU buffers. Measure actual overlap and reserved VRAM. |
| F05 | CUDA Graphs for changing history shapes | High / high | Implemented; approved 500-user comparison pending | Default — (pending); Expertium — (pending) | Default —; Expertium — | The active suite tests exact shapes. Hierarchical bucket preparation now has CPU prediction/gradient checks and separate graph-stage glue. Verify CUDA capture/replay/fallback after the suite; measure padding, hits, capture costs, full-history time and reserved VRAM before comparing the bucketed variant. |
| F06 | Reduce synchronization and logging overhead | Medium / low–medium | Implemented; approved 500-user comparison pending | Default — (pending); Expertium — (pending) | Default —; Expertium — | Compute label counts on CPU, consolidate metric transfers and buffer log writes; retain finite-value checks before updates. Profile `.item()` calls and device-dependent branches. |
| F07 | Reduce padding | Medium / medium | Implemented; approved 500-user comparison pending | Default — (pending); Expertium — (pending) | Default —; Expertium — | Tune the existing greedy sequence splits and bucket lengths; preserve sequence contents/order. Compare padded work against extra kernel launches and graph variants. |
| F08 | Fuse operations around the recurrence | Medium / high | Implemented; approved 500-user comparison pending | Default — (pending); Expertium — (pending) | Default —; Expertium — | The active trial fuses eight shift interpolations. Check the separate GroupNorm/bonus/gate and masked-loss kernels on CUDA after the suite, then measure full updates. Hierarchy gather/cat fusion remains pending. |
| F09 | Redesign the RWKV CUDA kernel | Exploratory / high | Row-tiled forward/backward prototype; CPU compilation passed; GPU parity/timing pending | — (pending) | — (pending) | Check recurrence outputs and all six input gradients for 1/2/4 rows per thread, then profile register/local-memory tradeoffs and full updates. Held-out comparison requires a separately approved batch. |
| F10 | Selective FP8 linear layers | Exploratory / high | Projection prototype; CPU fallback/quantization checks passed; CUDA backend probe pending | — (pending) | — (pending) | Probe installed Torch 2.9.1 `_scaled_mm` on RTX 5080; test eligible projections with BF16 backward/FP32 master. Measure quantization overhead, fallback counts and held-out quality against a matched Python control. |
| F11 | Reach target accuracy sooner | Exploratory / medium–high | Measured runner implemented; synthetic training/checkpoint/target-stop checks passed; GPU and broader comparisons pending | — (pending) | — (pending) | Run GPU functional checks after the first batch; approve a matched checkpoint comparison. Select a manifest-verified pretrained checkpoint and predeclare real target scores/cadence. Compare actual checkpoints at equal work and complete elapsed time, with missing points explicit. |
| F13 | Separate Curve and Instant specialist models | Exploratory / medium–high | Joint/specialist runner implemented; CPU gradient and synthetic training checks passed; GPU/broader comparisons pending | — (pending) | — (pending) | Compare Expertium joint, Curve-only, Instant-only and adjusted-weight joint training with matched splits/seeds/exposure. Report supported-task binary loss, full-update time/reviews/s and reserved VRAM; account for summed specialist costs. The prepared broader plan still needs explicit run approval. |
| F14 | Packed GPU feature encodings on full histories | High / medium | Integrated in requested B01; real-input/gradient checks passed; 500-user bundle comparison pending | — (pending full-history comparison) | — (pending full-history comparison) | The isolated B01 staging path includes deferred metadata in worker IPC, pinning, transfers and GPU materialization. Compare the frozen bundle against both fresh controls per model; standalone F14 validation remains pending. Include complete histories, preparation, transfers and captures. |

F14 is distinct from F03: F03 packs already prepared CPU tensors for worker
communication, while F14 defers feature lookup/calendar arithmetic to the GPU.
The earlier Expertium `combined-gpu-packed` screen measured **99,099 reviews/s**
with packed GPU features and CUDA Graphs on a repeated user-17 window (three
seeds, 32 measured updates each); its paired losses and timing remain in the
table above and [raw results](../artifacts/benchmarks/perf-optimizations/logloss/comparison.json).
That result does not measure full-history F14 or the proposed F14 + F05 bundle.
Neither is included in the original F01–F08 suite. Full-history quality and speed
remain pending. The user subsequently requested prioritizing B01, including this
feature path as a candidate; record its exact protocol under the existing
validation gate before the comparison.

### Authorized order: preserve training behavior first

On 2026-10-11 the user explicitly requested that the chat **Benchmark first 8 GPU
optimizations** shift to prioritizing B01 before exploring other directions. That
request was sent to the owning chat, within the agreed 500-user, both-model
framework. Exact bundle membership and its recorded comparison protocol remain
pending. The owning chat placed a case-boundary STOP latch at
2026-10-11T14:15:41Z while Default/F04 was evaluating. F04 completed; the
controller then stopped before F05, preserving all five completed cases, raw
logs, checkpoints, source archives and the original protocol. The launcher's
raw exit status is `failed`/1 because the intentional stop returns 1; the suite
records `stopped`, not a failed F04. See the
[transition request](../artifacts/benchmarks/gpu-batch1-500/B01-priority-transition-request.json).
The [B01 draft](../artifacts/benchmarks/gpu-baseline-b01/protocol-draft.json) copied the exact
500-user cohort, reserved 100 users, temporal ranges, review budgets, seeds,
batching, optimizer and precision. Its membership and execution mode remain
null in that preserved initial draft. The final membership and compatible
execution mode are now fixed in the
[frozen protocol](../artifacts/benchmarks/gpu-baseline-b01/protocol.json); live comparison
status is recorded below. Remaining F01–F08 cases stay pending under the revised
priority. The earlier freeze and functional attempts remain visible.

1. Resolve control repeatability and the existing F01/F02/F03 quality differences.
   Preserve paired inputs, initial weights, update order and seeds; check complete
   gradients and dropout/RNG behavior as well as forward predictions. Intended
   mathematical equivalence does not establish identical training trajectories.
2. Prioritize F14, F01, F03, F04 and F06, plus the exact-shape F05 path, as
   candidates with no intended change to the model, labels or update schedule.
   F14 has exact BF16 input evidence on the recorded fixtures, not every history.
   F05 needs dropout-enabled replay checks; its bucketed extension changes shapes
   and RNG consumption and is not assumed equivalent. Consider F02 separately
   when cache reuse can justify its construction cost; its measured Default
   empty-cache time was slower than the first control.
3. Combine only candidates that pass the appropriate input/gradient checks,
   then assess the fixed B01 bundle by a matched quality/speed comparison.
   Fix the final membership before the comparison and measure it for both architectures against their own
   controls. Preserve review exposure, batching, precision, optimizer, seeds and
   holdout protocol. Measure complete-history time, graph captures, preparation
   and transfers; assess weighted and user-averaged Curve/Instant losses with
   the established user-level uncertainty and control-stability requirements.
4. Use the accepted execution baseline for later comparisons. F07 padding,
   bucketed F05, F08 fusion and F09 kernel changes need additional numerical/RNG
   scrutiny. F10 precision, F11 architecture/schedule/exposure and F13 task changes
   explicitly explore different training behavior and follow separately.

| Bundle ID | Description | Status | Full-history Curve / Instant log loss | Complete training s / reviews/s | Next step |
|---|---|---|---|---|---|
| B01 | Combine preparation/execution improvements without an intended training-behavior change | Original comparison inconclusive; deterministic retry stopped mid-case by mistaken STOP marker | — (pending) | — (pending) | Preserve the partial attempt; hold launches for artifact migration, then run fresh controls under a frozen protocol. Details are in iteration0001. |

### B01 frozen comparison and component evidence

The requested comparison has **six fresh isolated cases**: Default/control,
Default/control-repeat, Default/B01, then the same three for Expertium. Establish
each model's repeated-control stability before running its bundle. Cohort,
reserved users, seed, review budgets, 128-review chunks, model batching/workers,
AdamW/LR/warmup/clip, BF16 child/FP32 master and original temporal evaluation
remain as recorded in the first-batch protocol. Exact options and runtime
settings are in [protocol.json](../artifacts/benchmarks/gpu-baseline-b01/protocol.json).
Implementation remains **uncommitted**; process-start commit/dirty/source archives
will identify each actual comparison, separately from the protocol's freeze.

B01 includes **F14 + F01 + F03 + F04 + F05 exact shapes + F06**. It uses one graph
slot with second-use admission and eager fallback, releasing unmatched Default
graphs before eager work; Expertium retains a graph until new capture admission
or eviction. It uses no buckets or F02 cache. The independent
[staging helper](../scripts/gpu_b01.py) carries **all deferred feature metadata**
through packed IPC, pinning and transfers, then materializes features on the copy
stream before its readiness event. Both controls use the same helper with all
switches off; the existing training/optimizer/evaluation loop is reused by the
[isolated runner](../scripts/run_gpu_b01.py).

**TorchScript executor optimization is disabled equally for all six cases**;
deterministic algorithms remain off. This is a recorded execution-mode change
from the historical first-batch controls, which are retained as reference evidence
and are not matched B01 controls. In the two frozen final-control/input cases,
ordinary and deterministic optimized-executor runs both changed loss and full
gradients during the first calls, then stabilized. With executor optimization
disabled, all four repetitions were exact in both modes. This implicates executor
optimization/warmup in these fixtures; it does not establish the historical
F01–F04 divergence's complete cause or whole-training repeatability.

The [CPU staging check](../artifacts/benchmarks/gpu-baseline-b01/checks-cpu-v3/result.json)
passed six real batch locations (users 157/189/237, both model batch spans),
including four spawned workers, original/deferred seeded RNG draws, all packed
metadata sharing one IPC storage, exact BF16 expansion, and a two-user prefetch
boundary/order fixture. Initial diagnostic failures remain: an incorrect
parent-RNG preservation assertion ([v1](../artifacts/benchmarks/gpu-baseline-b01/checks-cpu-v1/result.json))
and Torch serialization rejecting heterogeneous views of one storage
([v2](../artifacts/benchmarks/gpu-baseline-b01/checks-cpu-v2/result.json)). The passing check
compares seeded draw states and clones only the serialized diagnostic tensors;
production IPC still shares one storage.

CUDA checks passed for
[Default](../artifacts/benchmarks/gpu-baseline-b01/checks-cuda-baseline-v3/result.json) and
[Expertium](../artifacts/benchmarks/gpu-baseline-b01/checks-cuda-expertium-v2/result.json):
original versus synchronous/asynchronous packed input, full parameter gradients,
loss/predictions and dropout RNG state were exact on all three real locations per
model, as was F06's loss path. Default uses its frozen final first-control weights;
Expertium uses fresh seed 2025 weights. Neither diagnostic performs optimizer
updates or measures held-out quality/speed. Exact graph admission/capture/replay
and shape fallback also matched, at **16,384 raw reviews for Default** and
**32,768 for Expertium**, including changed features, restored original inputs
and a different dropout seed. A
[retained failed diagnostic](../artifacts/benchmarks/gpu-baseline-b01/checks-cuda-baseline-v2/result.json)
exceeded the fixed memory limit when an eager reference ran beside a retained
full-size graph. The corrected diagnostic computes CPU reference results before
capture; the production Default miss policy still releases unmatched graphs.

Each matched case records one-second per-core and per-process/worker CPU, RAM and
available headroom, I/O, allocator VRAM, and available GPU compute/memory activity
samples, alongside preparation/transfer waits and capture statistics. Separate
startup/primer, training and save/evaluation phases; log sampled overhead and
checkpoint-save time. Windows paging counters can be unavailable; retain other
samples and mark those counters null. NVIDIA memory activity percent is not
achieved bytes/s bandwidth. Training time includes all full-history preparation,
materialization, transfers, updates, captures and logging; startup/primer,
validation, saves and full subprocess wall time stay explicit.

Acceptance keeps the existing **absolute 0.001** paired-user upper bound for all
four binary losses against **both** fresh controls, 10,000 resamples,
seed 20251012 and the conservative existing Bonferroni divisor 64. Both controls
must agree within 0.001 and match inputs/exposure/source; complete cold training
must be faster than both. Uncertainty remains conditional on one training seed.
The later R01 relative 1% budget does not relax this B01 gate. The
[synthetic report checks](../artifacts/benchmarks/gpu-baseline-b01/report-checks-v1/result.json)
passed 11 decision/percentage-rendering checks; these are not benchmark results.
The [initial runner smoke](../artifacts/benchmarks/gpu-baseline-b01/runner-smoke-v1/baseline/control/result.json)
failed before training because a direct invocation omitted the launcher's workspace
CuPy/temp paths. Later smokes use the recorded launcher environment. The preserved
[first freeze](../artifacts/benchmarks/gpu-baseline-b01/protocol-freeze-v1-before-telemetry-fix.json)
predates the telemetry fix that retains CPU/RAM samples when paging counters fail.
The [functional runner audit](../artifacts/benchmarks/gpu-baseline-b01/runner-smoke-audit.json)
passed both architectures: control and B01 had identical initial/trained weights,
validation inputs, all four scores and exposure after two updates on two users.
The revised telemetry retained CPU/RAM/process/VRAM/GPU samples in three checked
cases, with paging explicitly unavailable. These short cases do not establish
500-user quality or resource saturation. All requested full comparisons use
the final frozen implementation, including the corrected telemetry.

The full comparison launched at **2026-10-11T15:01:31Z**. Its launcher, controller and Default/control training child were verified alive, with ordered training updates and resource samples advancing. Actual first-case provenance records commit `fdabf0a248246c494a3f4b480cc764253b7f0cb7`, dirty, with [source archive](../artifacts/sources/experiment-sources/ad6d2cbc22a57776b19c384ba7cd1f3e12dfaee536cdd5f483aa2d4c40960fc8/source.zip). The implementation fingerprint is `87a6ae7f7d13400cc7380d07327d28da486d034ac992a2dd9e3d3a805d8bde01`; each later case preserves its own process-start provenance. See the live [full-history report](../artifacts/benchmarks/gpu-baseline-b01/summary.md) and [raw suite](../artifacts/benchmarks/gpu-baseline-b01/suite.json). No B01 quality/speed result has completed yet. The first Default control has completed all 500 users: 1,881 updates, 26,705,280 training reviews, **1,054.62 s / 25,322 reviews/s**. Curve losses are **0.392797 weighted / 0.380768 user-average**; Instant losses are **0.366719 weighted / 0.351112 user-average**. Its fresh repeated control subsequently completed and failed the stability gate. These measurements use the frozen B01 executor mode; the historical first-batch control is not their matched reference.

B01 stopped after its two completed Default controls disagreed. The bundle
and Expertium cases remain unrun. See the [final comparison report](../artifacts/benchmarks/gpu-baseline-b01/summary.md)
and [iteration0001](../artifacts/retraining/gpu-retrain-r01/iterations/iteration-0001.md)
for preserved repeatability diagnostics and the next prerequisite action.

The [separate deterministic retry protocol](../artifacts/benchmarks/gpu-baseline-b01-deterministic-v3/protocol.json) retains the original 500-user cohort, reserved test users, seed, budgets, bundle, batching, precision and absolute quality gates. Its changed execution configuration and prerequisite evidence are recorded in [iteration0001](../artifacts/retraining/gpu-retrain-r01/iterations/iteration-0001.md). Implementation remains uncommitted; this retry's own process-start provenance is required.

### R01: retraining optimization loop

Reduce complete retraining time with a fixed **<1% relative logloss regression
budget**. The full procedure is in [GPU_RETRAIN_LOOP.md](GPU_RETRAIN_LOOP.md).
Detailed plans, measurements and decisions belong in each iteration's own file.

| ID | Strategy | Status | Curve / Instant log loss | Complete retraining s / reviews/s | Next step |
|---|---|---|---|---|---|
| R01 | Minimize time to acceptable quality | Loop requested; progress is in the iteration index | — (pending) | — (pending) | Accept B01, freeze the quality reference, then profile and screen a bounded shortlist. |

#### Iteration index

| Iteration | Record | Status / outcome |
|---|---|---|
| 0001 | [Iteration 0001](../artifacts/retraining/gpu-retrain-r01/iterations/iteration-0001.md) | Deterministic retry stopped mid-case; artifact migration hold; reference and R01 candidates pending |

F13 tests whether the two tasks compete for shared recurrent capacity or benefit
from learning together. Curve-only training retains the corrected/raw Curve losses
and Curve regularization; Instant-only training retains the four-rating
cross-entropy objective, while evaluation reports Instant binary recall log loss.
Removing an output head or loss does not by itself reduce the backbone's recurrent
state, and training both specialists adds work, so speed and memory benefits remain
unmeasured. Deployment can select one specialist at a time; Instant with a Curve
due-date fallback would require both models and separate states. Apply the existing
explicit-request validation gate before launching a broader comparison.

Profile CPU/GPU activity across real user transitions before selecting the next
track. Existing larger-batch searches, fused AdamW and checkpoint-tile results
remain in the measured table above. The proposed tracks do not inherit those
measurements.

For each new run retain raw review presentations, measured optimizer updates,
unique training reviews, paired seeds and validation label counts. Log loss is
better when lower; reviews/s is better when higher. A rate on repeated data does
not measure unique-data loading or complete-epoch speed. Compare each architecture
against its own control when its batch size or training exposure differs.

### Commit traceability

Keep **experiment commits** and **implementation commits** separate. Each new
experiment records the full Git HEAD at process start, dirty/clean status, and
a content-addressed source archive under `artifacts/sources/experiment-sources/`. The
archive includes tracked and untracked code/configuration in `rwkv/`, `expertium/`,
`scripts/`, `configs/`, plus dependency manifests, `justfile` and `upstream.json`; datasets,
checkpoints and generated results are outside that snapshot. A dirty commit ID
identifies the base revision; the archive identifies the local source actually
captured. Restart the process after source changes. Resumed training retains a
separate source record for each session, including its starting update count.

After committing a change, add its full commit ID to the ledger below and link
the relevant track/bundle and raw experiments. Preserve the original experiment
revision even when an implementation is committed later. Old experiments without
captured provenance remain **not recorded**; the report never substitutes the
current HEAD. Pending ideas keep `—` until implementation/measurement exists.

| Change / bundle | Related tracks / evidence | Implementation commit | Experiment commit | Status |
|---|---|---|---|---|
| Expertium comparison and RTX 5080 CUDA setup | [Comparison documentation](COMPARISON.md); verified from Git history | `020e2fac1e62fe8e69c2d8daa36a81c881919a95` | — (not recorded in existing artifacts) | Committed setup |
| Fixed-batch GPU optimization implementations | [Throughput runs](../artifacts/benchmarks/perf-optimizations/comparison.json), [quality screens](../artifacts/benchmarks/perf-optimizations/logloss/comparison.json) | — (uncommitted) | — (not recorded in existing artifacts) | Measured/screened; pending broader validation |
| Persistent optimization list, loss/time reporting and source traceability | `AGENT.md`, `scripts/report_training_tradeoffs.py`, `scripts/experiment_provenance.py`; [CPU source-capture check](../artifacts/diagnostics/check-temp/experiment-provenance-check.json) | — (uncommitted) | CPU source-capture check: `020e2fac1e62fe8e69c2d8daa36a81c881919a95` (dirty; archive linked in artifact); training: — | Implemented; CPU verification only |

### Validation gate and agreed user count

An optimization or a combined set of changes can be called **validated** only
after a matched multi-user quality and speed comparison on **at least 500 distinct
evaluation users**, as agreed by the user. Numerical parity, a throughput run, or the one-user
log-loss screen above is preliminary evidence. `complete` in a run artifact means
that the run finished; it does not mean that the changes passed this gate.

**The human user must explicitly request broader validation before it is
launched.** Asking for implementation, an idea list, or a recommendation does
not authorize this phase. Previously authorized runs keep their existing scope.
Multiple changes can be validated as a bundle against one control; record the
bundle and do not attribute its validation to untested individual components.

The agreed sampling population is **user IDs 1–5000**. The saved metadata count for
revision `75299740cff05894ef42d7ad990666691efdd2da` is **357,460,182 raw reviews**;
the mean is 71,492 reviews/user and the median is 30,832.5. The largest 10% of
users contribute 47.5% of the reviews. See the
[per-user count artifact](../artifacts/diagnostics/check-temp/first-5000-review-counts.json).

| Evaluation users | Suggested use | Approximate reviews across full histories |
|---:|---|---:|
| 100 | Preliminary regression screen; insufficient for validation | 7.1 million |
| **500** | **User-agreed minimum for validation of an optimization bundle** | **35.7 million** |
| 1,000+ | Numerical/precision, architecture or training-strategy changes; inconclusive smaller samples | 71.5 million at 1,000 |
| 5,000 | Full population-scale confirmation when explicitly requested | 357.5 million |

The 500-user minimum is agreed; launching the validation run still requires an
explicit request. Larger cohorts remain recommendations, and these counts do not
guarantee statistical power. Approximate review counts use the
population mean; actual selected histories and scored held-out labels will differ.
Evaluation users are counted separately from training users; 500 does not imply
training on only 500 users or training on all of their validation labels.

Use a fixed sample across IDs 1–5000, stratified by history size rather than simply
taking the first 500 IDs. Include long-history cases for memory/performance checks.
For unseen-user generalization, keep evaluation users out of gradient training.
If testing later reviews of known users instead, use a chronological holdout with
cross-boundary targets masked and label the result as temporal validation.
Preserve a separate final test cohort instead of repeatedly tuning against it.
[Grouped/time-aware evaluation guidance](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).

Before launching an approved comparison, record the exact bundle/control, cohort,
training and evaluation review budgets, seeds, and allowed log-loss regression.
Use paired seeds and the same evaluation inputs. Report Curve and Instant log
loss separately, both label-weighted and averaged per user, together with paired
user-level uncertainty, full-history elapsed time, and reviews/s. A paired
user-bootstrap confidence interval can quantify the uncertainty; increase the
cohort when it cannot resolve the agreed loss tolerance. Tiny review-level error
bars alone do not establish user-level generalization.
<!-- optimization-tracks:end -->

<!-- gpu-batch1:start -->
## Authorized first batch: F01–F08, 500 users each

The user explicitly authorized this batch on 2026-10-11. Each track is isolated
against fresh controls for Default and Expertium; no component inherits another
track's validation. Implementation commits: **uncommitted**. Status: **pending
completion of the approved comparisons; live progress below**.

The fixed cohort contains **500 distinct users**, 50 per history-size decile from
IDs 1–5000 (sampling seed 20251011). Train on the first 80% of each complete history,
rounded down to 128 reviews, and score the remaining reviews. Mask next-review
targets crossing the split. This is temporal validation of known users. Reserve
100 disjoint users for a final test and leave them unscored. Exact IDs, ranges,
revision and protocol are in [protocol.json](../artifacts/benchmarks/gpu-batch1-500/protocol.json).
The data path is pinned to `data/anki-revlogs-10k-original-20261011` at revision
`75299740cff05894ef42d7ad990666691efdd2da`; replacing the active dataset pointer cannot change
this batch. The first short runner check stopped before training when that pointer
was replaced with the timestamped export; [its failure is retained](../artifacts/diagnostics/gpu-batch1-smoke/baseline/control/result.json).

Training seed **2025**, one full selected-history epoch: **26,705,280
raw training reviews** and **6,716,892 held-out raw reviews**
per case. BF16 computation/FP32 master, 128-review chunks; Default 128 chunks/4
workers, Expertium 256 chunks/6 workers. LR 7e-4 with five-update warmup, AdamW,
clip 0.5. Within each model, update order and exposure are identical across cases.
Evaluation uses original features, dropout off, FP32 master and WKV tile32.

Acceptance requires both binary loss regressions' paired-user bootstrap upper
bounds ≤0.001 for label-weighted and user-averaged scores (10,000 resamples;
Bonferroni across 64 model/track/metric comparisons). Both controls must agree within
0.001. Complete training must be faster than both controls; F02's cold decision
includes cache creation. A warm-cache result is reported separately. User-level
uncertainty is conditional on the single training seed; this does not measure
training-seed uncertainty. Inconclusive or failed trials stay visible.

Training wall time includes user loads, worker startup/preparation, transfers,
optimizer updates, graph capture and log writes. Startup, discarded JIT primer,
cache construction, validation and complete child-process wall time are separate.
F05 uses one exact-shape graph slot and eager fallback, releasing the graph on a
shape miss for Default to leave room for eager work; F08 initially fuses the
eight time-shift/interpolation inputs, with an analytical Torch backward.

[Raw results and progress](../artifacts/benchmarks/gpu-batch1-500/suite.json),
[paired quality/time report](../artifacts/benchmarks/gpu-batch1-500/summary.md).
<!-- gpu-batch1:end -->

### First-batch implementation checks

The [focused GPU checks](../artifacts/diagnostics/check-temp/gpu-batch1-check.json) passed for
both models: packed/cached/worker-local CPU inputs and pinned transfer values are
exact; standalone fused-shift outputs and both input gradients had zero maximum
absolute difference in the tested FP32/BF16 fixture. Model F06/F07 gradients use
BF16 tolerances (absolute 0.002, relative 0.04), not a claim of bitwise equality.
The [initial stricter F06 check failed](../artifacts/diagnostics/check-temp/gpu-batch1-initial-gradient-check.json)
at a maximum absolute difference of 0.00003052; that failed check remains visible.
Graph capture, exact-shape hits and eager fallback also passed focused checks.
These checks retain their own source revision/archive; later runner changes do
not inherit that historical revision. The [two-user runner smoke](../artifacts/diagnostics/gpu-batch1-smoke-v3/summary.md)
exercises all candidate paths; it does not satisfy the 500-user validation gate.


<!-- gpu-batch1-progress:start -->
### First-batch progress

Suite: **stopped**; completed cases: **5/20**.

| Track | Default | Expertium |
|---|---|---|
| F01 | pending repeated control | pending |
| F02 | pending repeated control | pending |
| F03 | pending repeated control | pending |
| F04 | pending repeated control | pending |
| F05 | pending | pending |
| F06 | pending | pending |
| F07 | pending | pending |
| F08 | pending | pending |

[Curve/Instant losses, timing, uncertainty and experiment revisions](../artifacts/benchmarks/gpu-batch1-500/summary.md).
Measurements and pending cases are retained in the raw suite; implementation remains uncommitted.
<!-- gpu-batch1-progress:end -->

### Remaining-track prototypes prepared during the first batch

F09–F11/F13 use separate files under `scripts/`; they do not alter the implementation
loaded by the active F01–F08 suite. All changes remain **uncommitted**. No concurrent
GPU kernels or additional multi-user validation runs were launched for these tracks.

F09's [historical trace analysis](../artifacts/benchmarks/gpu-future/kernel-profile.json) measured
recurrence backward at **50.91%** and forward at **6.30%** of summed kernel time in
that trace, with 1,024-thread blocks, respectively 64/40 registers per thread and
17,152/0 shared-memory bytes. These are profiler kernel durations, not complete
training time. The historical experiment revision is **not recorded** in the trace;
the analysis captures its own revision separately. This evidence informed
[`rwkv_row_tiles.cu`](../scripts/kernels/rwkv_row_tiles.cu): 1/2/4 rows per thread,
matching checkpoint layout, float32 state/reductions and the existing six-gradient
equations. The [CPU-only NVRTC/PTXAS check](../artifacts/benchmarks/gpu-future/row-tiles-compile/result.json)
compiled ten FP32/BF16 configurations for `sm_120`. For BF16 tile8, backward uses
64/96/124 registers per thread at 1/2/4 rows, and 64/128/256-byte stack frames;
all three use 17,152 bytes shared memory and report no register spills. Local
checkpoint arrays still consume stack storage. Compilation does not prove parity,
occupancy, throughput or quality. Runtime checks remain pending.

F10's [`gpu_fp8.py`](../scripts/gpu_fp8.py) selects aligned recurrent/channel
projections, uses per-tensor E4M3 forward GEMMs with the locally verified Torch
2.9.1 API, and uses original BF16 operands for straight-through backward. It leaves
normalization, recurrence and output/loss arithmetic in their original precision.
Small shapes and FP32 evaluation take the ordinary linear path, with call counts
recorded; CUDA backend errors do not silently fall back. Eligible dynamic token
counts are zero-padded to multiples of 16 before quantization and cropped after
the matrix product; original and padded token counts are reported separately.
CPU checks establish unchanged quantization scale/values for the original rows;
the unaligned-token CUDA forward/backward check remains pending. FP8 support for RTX
Blackwell hardware is described by [NVIDIA's CUTLASS documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md),
but installed backend support and model quality are still unmeasured. The isolated
Python rewrite needs its own Python control so interpreter overhead is matched.

F11/F13 share [`gpu_future_models.py`](../scripts/gpu_future_models.py). Width
variants retain 32-dimensional heads and all streams. One-layer baseline ratios
are defined in this isolated prototype; the [initial CPU check failure](../artifacts/benchmarks/gpu-future/cpu-check/result.json)
records the original initialization's division by zero. Pretrained initialization
requires a fingerprint, prior dataset/review exposure, user lists and provenance,
and rejects overlap with reserved users. [`gpu_training_targets.py`](../scripts/gpu_training_targets.py)
provides constant/cosine/WSD schedules and predeclared four-score target stopping.
Equal-work/time comparisons select actual measured checkpoints and expose gaps;
they do not interpolate accuracy or treat missing checkpoints as measurements.

Curve-only models remove Instant heads and retain corrected/raw Curve losses and
Curve regularization. Instant-only models remove Curve heads and retain four-rating
cross-entropy; future evaluation reports Instant **binary** recall log loss. Joint
controls accept explicit task weights. Neither specialist reduces the recurrent
backbone, and measuring both models requires accounting for both training costs.

The [successful CPU checks](../artifacts/benchmarks/gpu-future/cpu-check-v2/result.json) use small
synthetic fixtures for both model families. They cover Python-versus-scripted
prediction/gradient parity, single-layer initialization/forward, specialist head
removal, joint gradient decomposition, FP8 CPU fallback/finite quantization,
schedules, target confirmation, observed budget limits, and pretraining leakage
rejection. They establish functional behavior only. GPU parity, full-update timing,
and 500-user quality for these remaining tracks are pending.

Both the offline compiler check and successful CPU check recorded Git base
`020e2fac1e62fe8e69c2d8daa36a81c881919a95` (**dirty**), with separate source archives
in their result JSONs. Their snapshots identify the exact uncommitted source they
tested; later changes do not inherit those checks' revision.

### Measured runner for the remaining tracks

[`train_gpu_future.py`](../scripts/train_gpu_future.py) now integrates the separate
F09–F11/F13 prototypes into an opt-in runner with fresh AdamW, BF16 children/FP32
master weights, the original parameter groups, finite-loss/gradient checks and
ordered review coverage. Source guards cover the imported repository code and the
new helpers; each process preserves its own source archive. It refuses a new GPU
workload while the first-batch worker is marked live. Proposed plans do not grant
run permission.

The [remaining-batch plan](../artifacts/benchmarks/gpu-future/proposed-remaining-batch-v2/proposed-plan.json)
contains **25 proposed cases**, including architecture-specific repeated controls.
It reuses the fixed 500-user temporal cohort and reserved final-test IDs as a
selection only. **Authorization remains pending**: the existing approval covers
F01–F08. F09 compares tile8 row layouts against tile8 controls with identical raw
pointer launches. The other candidates share Python controls and four exact update
checkpoints, including the last update; checkpoint validation/save costs are
included in elapsed-time comparisons. F09 records compilation separately and
evaluates after restoring the common original recurrence. Pretraining and target
stopping are integrated, but those planned cases still need a verified checkpoint
manifest and real predeclared targets.

Each checkpoint records weighted/user-averaged binary Curve/Instant scores,
per-user values and the number of users that have received gradient updates.
Supported tasks must have labels for every selected evaluation user. Unsupported
specialist scores remain `null`, not zero. Curve evaluation uses the original
logits loss, retaining penalties for confident wrong predictions; Instant keeps
the original binary recall probability definition. A regression fixture verifies
that wrong logits ±25 retain binary log loss **25**, rather than being hidden by
an added probability clamp. Real-data evaluation fingerprints all prepared tensors
and hierarchy lengths per user.

Training seconds include data loads, worker startup/shutdown, preparation,
transfers, updates and logs. Validation is separate; complete elapsed seconds
include model/backend setup and checkpoint validation/saves. F11 budget reporting
uses actual checkpoints for raw-review, training-time and complete-time limits,
with gaps or missing points explicit. Early stopping records reduced exposure and
the first observed target time. It does not claim full selected-history coverage
when it stops early.

The [synthetic integration audit](../artifacts/benchmarks/gpu-future/integration-audit.json)
checks eight recorded runs covering joint training, both specialists, width/depth
changes, cosine/WSD schedules and one/two-observation target confirmation. These
runs performed optimizer updates, changed supported head weights and saved finite
FP32 checkpoints. The audit verifies source-archive fingerprints, task-head removal,
supported/unsupported metric reporting and observed checkpoint budget limits.
Synthetic scores/timings do **not** fill the broader quality/speed columns.

The first runner attempt's [metadata/setup failure](../artifacts/benchmarks/gpu-future/integration-expertium-joint/failure.json)
and a [Windows concurrent-archive setup failure](../artifacts/benchmarks/gpu-future/integration-final-joint/setup-failure.json)
remain visible. The archive failure occurred before training; a serial rerun passed.
Shared source-capture code was left unchanged while the approved GPU suite runs.
The [latest helper CPU checks](../artifacts/benchmarks/gpu-future/cpu-check-v5/result.json) and
integration runs each retain their own Git/source provenance. GPU backend execution
and broader predictive-quality/time comparisons remain pending.

[`report_gpu_future.py`](../scripts/report_gpu_future.py) retains all 25 proposed
cases, per-user bootstrap comparisons against both matched controls, observed
checkpoint budgets and summed specialist costs. The [pending report](../artifacts/benchmarks/gpu-future/remaining-report-v2/summary.md)
contains no broader-run measurements or inferred approval. Twelve
[reporting regression checks](../artifacts/benchmarks/gpu-future/report-check-v1.json) passed:
synthetic/unapproved/undersized runs cannot be accepted; input and aggregate
mismatches are rejected; both controls must meet the quality bound; unstable
controls, unknown pretraining costs and reduced exposure remain explicit. These
are fabricated logic fixtures plus one actual archive check, not validation.

### F01 preliminary quality discrepancy

Default F01 completed 500 users with 1,881 updates and 26,705,280 raw training
reviews. Training took **694.26 s** versus the first control's **1,277.20 s**.
Weighted Curve/Instant log loss changed by **+0.000074 / +0.014429**;
user-averaged losses changed by **+0.002785 / +0.021135**. The preliminary quality
criterion fails against that control; the repeated control is still pending.
F01 is not accepted or validated from this speed result.

A [saved-log audit](../artifacts/benchmarks/gpu-batch1-500/baseline/F01/saved-log-audit.json)
found matching initial weights, validation input fingerprints, consumed source,
and all logged batch IDs, offsets, seeds, review/label counts and LR values.
The first logged numerical difference occurs at update 132: objective values
2.094234705 versus 2.094232321. Later differences grow during training. This
read-only audit does not establish training-tensor parity or identify the cause.
After the active suite exits, run a focused GPU reproducibility/input check if
the repeated control does not resolve the discrepancy; preserve the original
500-user result and do not silently replace it.

### F05 hierarchical buckets beyond the active exact-shape variant

[`gpu_shape_buckets.py`](../scripts/gpu_shape_buckets.py) adds an isolated bucket
prototype. Each intermediate stream uses fixed length buckets and sequence
capacities rounded to a configured quantum; indices are remapped across the full hierarchy. Sequences
retain their internal order. New padding is skipped, carries no labels, and
does not feed real positions. A hierarchy that gathers padded output into a real
position falls back to the original batch. The final sample/label layout and
input stride are padded consistently.

The revised default lengths are **1/2/4/8/16/32/64/128/256**, and sequence counts
round to **multiples of 8** (one dummy sequence in an empty bin). The **1.5 maximum
work ratio** remains in place for input, recurrent and label positions. Oversized lengths or excess padding
keep the original batch. Separate counters report admitted/fallback batches and
actual padded positions. The `BucketedGraphStage` connects admitted buckets to
the existing one-slot, second-use graph cache and keeps policy rejections eager.
It resets captures when training/dropout mode changes. Callers must release their
returned model/statistics references after each optimizer step so eviction can
free the graph pool. This graph-stage integration has not executed on CUDA yet.

The [latest CPU check](../artifacts/benchmarks/gpu-future/shape-buckets-cpu-v4.json) uses a synthetic
five-stream fixture with mixed split lengths and skips. It preserves label/review
identities, verifies that two different source shapes produce the same bucket
tensors, rejects a real dependency on padding, and checks predictions, losses and
all parameter gradients for both model families. Maximum gradient difference was
**3.73e-8** in FP32 with dropout disabled. The deliberately loose fixture work
limit is 20; its 6.96 recurrent-work and 8.0 input/label ratios would be rejected
by the default 1.5 policy. This toy fixture is no evidence of production admission
rate or speed. The [earlier power-of-two count check](../artifacts/benchmarks/gpu-future/shape-buckets-cpu-v3.json)
and its source snapshot remain preserved; pass `count_quantum=None` to select
that older count policy.

The [real-input admission comparison](../artifacts/benchmarks/gpu-future/bucket-real-admission-v4.json)
uses the two fingerprinted saved training inputs from users 189 and 237. The
earlier lengths starting at 8 and power-of-two counts rejected both, requiring
2.481× / 2.144× recurrent work. Adding short lengths alone still exceeded the
1.5 cap. With the revised eight-sequence quantum, both inputs are admitted:
recurrent positions rise **184,560 → 202,648 (+9.80%)** and
**35,264 → 43,492 (+23.33%)**. Input/label work is unchanged for user 189 and
**+20%** for user 237. The cap remains 1.5. Quantum4 has lower padding on these
inputs; its graph hit rate and speed, like quantum8's, remain unmeasured.

Mapped labels, review IDs and input features are exact. Two integer marker
recurrences preserve every **real** review/query through the full gather,
time-shift and skip hierarchy. This is a structural simulation, not RWKV model
or gradient parity on the real batches. The [first structural comparison failure](../artifacts/benchmarks/gpu-future/bucket-real-admission-v3.json)
is retained: it also compared unused padded outputs that gather row zero. The
corrected check verifies that those positions have no labels and negative review
IDs, compares all real positions, and records the **230** changed unused padding
markers for user 237. Real dependencies on padding remain forbidden. This
two-input result does not establish whole-cohort admission or capture/replay speed.

Dropout shapes and RNG consumption change when padding changes. GPU forward/
backward parity, capture/replay/fallback, real-history admission/hit rate, reserved
memory and a matched 500-user bucketed comparison remain pending. The live F05
case still uses its preserved exact-shape source and cannot validate this later
bucket implementation. All new code remains uncommitted.

The pending CUDA functional command is
`scripts/check_gpu_shape_buckets.py --cuda --output artifacts/benchmarks/gpu-future/shape-buckets-cuda-v1.json`.
It checks FP32/BF16 eager buckets, second-use capture, replay, dropout-mode reset
and explicit eager fallback on small five-stream models, and refuses to launch
while the first-batch worker is marked live. It has been syntax-checked; no CUDA
result exists yet.

### F03 completed communication variant

Default F03 completed the same **1,881 updates / 26,705,280 raw training reviews**
and all **500** held-out users. Its [result](../artifacts/benchmarks/gpu-batch1-500/baseline/F03/result.json)
records **1,277.1913 s** and **20,909.3811 raw reviews/s**: a **0.0119 s** time
decrease and about **+0.0009%** throughput against the first control. This is
essentially unchanged elapsed time. Weighted Curve/Instant log loss is
**0.432101 / 0.435618** (**−0.828% / +3.355%**), and user-averaged loss is
**0.432939 / 0.440982** (**−0.247% / +4.895%**). Instant quality fails the declared
criterion against the first control; acceptance and repeated-control stability
remain pending.

The [saved-log audit](../artifacts/benchmarks/gpu-batch1-500/baseline/F03/saved-log-audit.json)
found matching initial weights, held-out input fingerprints, consumed source and
metadata for all 1,881 updates. There are **1,805** differing logged numerical
updates, beginning at **step 77**, user 157, offset 0: objectives **1.509987235 /
1.509986877**, gradient norms **0.662101448 / 0.662107468**. This does not establish
full training-input parity or identify the cause. The original result remains
preserved while the approved suite continues.

### F02 completed quality and cache cost

Default F02 finished constructing **1,881 batches**, with **zero existing-cache
hits**, in **1,494.92 s**. The recorded cache uses **11,686,639,259 bytes**.
These are cache-only measurements from its [completed result](../artifacts/benchmarks/gpu-batch1-500/baseline/F02/result.json).
Because construction alone exceeds
the first control's 1,277.20 s training time, Default F02 cannot satisfy the
declared cold-training speed criterion against that control. Training subsequently
completed all 1,881 updates: **985.45 s** with the constructed cache, **2,480.37 s**
including construction, and **27,100 raw reviews/s**. Relative to the first control,
warm training time changed by **−22.84%**, throughput by **+29.61%**, and setup-inclusive
time by **+94.20%**.

Held-out evaluation completed all **500 users** with the same label counts as
the first control. Weighted Curve/Instant log loss is **0.394084 / 0.364665**
(**−9.553% / −13.479%**); user-averaged loss is **0.381996 / 0.348074**
(**−11.985% / −17.205%**). A [saved-log audit](../artifacts/benchmarks/gpu-batch1-500/baseline/F02/saved-log-audit.json)
found matching metadata for all 1,881 updates, initial weights, held-out inputs
and consumed source files. It found 1,782 differing numerical updates, beginning
with the tiny gradient-norm difference at step 100 described below. The lower
loss is an observed comparison against the first control; its cause is unresolved
and the repeated control is pending. F02 remains unvalidated and fails the
declared cold-training speed requirement against the first control.

Both architectures' F02 cases use the same suite cache directory. Cache keys
include exact ranges and source/data identity; some ranges can be reused when the
Expertium batch size differs. Inspect its recorded hit/build counts before
interpreting its setup-inclusive time: inherited entries do not establish a
standalone empty-cache measurement. Preserve measured warm/setup costs, and keep
independent cold-cache claims pending when earlier construction cost was reused.

### F08 post-recurrence fusion beyond the active shift variant

[`gpu_fused_post.py`](../scripts/gpu_fused_post.py) and its
[CUDA source](../scripts/kernels/rwkv_fused_post.cu) add an opt-in forward/backward
kernel for GroupNorm with 32 features per head, the recurrence bonus reduction
and output gating. An isolated Python model installer preserves state-dict names
and leaves output projection/dropout outside the fused region. No source loaded
by the active suite was edited; its F08 result will concern shift interpolation
only. The new implementation remains **uncommitted**.

The [CPU checks and offline compilation](../artifacts/benchmarks/gpu-future/fused-post-cpu-v4/result.json)
passed. Six synthetic backward checks cover FP32/BF16 with one, two and four heads.
CPU coefficient/chain-rule calculations use native CPU saved statistics; FP32
maximum gradient difference was **2.384e-7**, and all tested BF16 gradients matched
exactly. The CPU forward path is the original eager expression, so these checks
do **not** execute the CUDA kernel. Both small, two-layer, five-stream model
families retained identical FP32 loss, predictions and all parameter gradients
after installing the CPU fallback. No CUDA context was initialized.

FP32 and BF16 forward/backward kernels compiled through NVRTC and PTXAS for
**SM120**, with **no stack or spill bytes**. Forward/backward register counts were
**26/38 (FP32)** and **25/36 (BF16)**. This establishes compilation, not launch,
numerical parity or speed. The [first attempt](../artifacts/benchmarks/gpu-future/fused-post-cpu-v1/result.json)
retains its CuPy cache-permission failure; subsequent attempts put the compiler
cache inside workspace artifacts. Earlier source snapshots remain preserved.

The CUDA prototype preserves BF16 pointwise rounding boundaries and input-dtype
normalization moments/epsilon, following the
[Torch 2.9.1 CUDA GroupNorm source](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/cuda/group_norm_kernel.cu).
Its warp reductions differ from PyTorch's Welford moments, and parameter-gradient
atomic reductions can be nondeterministic. CPU checks do not resolve either CUDA
difference. After verifying the first batch has exited, the pending command is:

```
.venv/Scripts/python.exe scripts/check_gpu_fused_post.py --cuda --output artifacts/benchmarks/gpu-future/fused-post-cuda-v1
```

It checks all eight kernel input gradients and both FP32/BF16 model integrations.
The [active-suite gate check](../artifacts/benchmarks/gpu-future/fused-post-cuda-live-gate-v1/result.json)
refused this command before importing Torch or starting functional checks while
the approved worker was live; it is not a CUDA test result.
Capture/replay, full-update speed/VRAM and a matched broader quality comparison
remain pending; no speed gain is projected. Loss-reduction fusion is addressed
separately below; hierarchy gather/cat fusion remains unimplemented.

### F08 masked loss-reduction fusion beyond the active shift variant

[`gpu_fused_loss.py`](../scripts/gpu_fused_loss.py) and its
[CUDA source](../scripts/kernels/rwkv_masked_means.cu) combine seven FP32
per-position loss arrays into the original **nine masked averages** and four
counts. CUDA uses partial and final reductions with a fixed order and no global
atomics, followed by one backward kernel for all seven loss arrays. The isolated
Python loss-method installer preserves loss creation, objective weights,
equalization reports, label counts, binary Curve/Instant predictions and finite
checks. BCE, rating cross-entropy and regularizer creation remain outside this
region. This is an uncommitted prototype, independent of the active F08 variant.

The [CPU functional checks and offline compilation](../artifacts/benchmarks/gpu-future/fused-loss-cpu-v1/result.json)
passed with **Torch 2.9.1+cu128**. Fifteen reduction fixtures cover single-position
and partial-block shapes, transposed inputs, mixed labels, no Curve labels, no
Instant labels, no equalization labels and no labels. Every tested mean and all
seven input gradients matched the original eager reductions exactly. No-label
means and gradients were zero, with the original denominator epsilon retained.
This exercises the CPU custom backward, not the CUDA reduction kernels.

Both small two-layer, five-stream model families retained every statistics field
and all parameter gradients exactly in FP32. The checks cover loss fusion alone
and its composition with the post-recurrence CPU fallback above, including
equalized Curve/Instant labels and an absent label. Dropout was disabled. No CUDA
context was initialized; these results do not establish CUDA parity, 500-user
quality or training speed.

All three kernels compiled through NVRTC and PTXAS for **SM120**, with **no stack
or spill bytes**. Partial/final/backward register counts were **18/22/24**, and the
two forward reduction kernels use **32 bytes of shared memory** per block. The
[live-suite gate check](../artifacts/benchmarks/gpu-future/fused-loss-cuda-live-gate-v1/result.json)
refused CUDA mode before importing Torch while the approved worker was active.
After verifying that the first batch has exited, use:

```
.venv/Scripts/python.exe scripts/check_gpu_fused_loss.py --cuda --output artifacts/benchmarks/gpu-future/fused-loss-cuda-v1
```

The pending CUDA check compares all reduction outputs/input gradients and every
model statistics field/parameter gradient, in FP32 and BF16, with loss fusion
alone and combined with post-recurrence fusion. Actual capture/replay, full-update
timing/VRAM and matched broader quality comparisons remain pending. The changed
reduction order still needs numerical checks even though it uses no atomics.

### Focused F01/F02 reproducibility diagnostic

[`check_gpu_reproducibility.py`](../scripts/check_gpu_reproducibility.py) now
captures the two first differing Default log locations: user 189, raw offset
32,768 (F02 step 100), and user 237, offset 16,384 (F01 step 132). The
[CPU check](../artifacts/benchmarks/gpu-future/reproducibility-cpu-v2/result.json) found identical
prepared-tensor fingerprints for serial source loading, loader-thread loading
and deterministic sample-cache input at both locations. Encodings were regenerated
with the original per-update seeds. Saved diagnostic input files also passed a
fingerprint check after weights-only loading. The check covers **two input batches**
with serial preparation; it does not establish whole-run or four-worker parity.

F02's first logged difference is in the gradient norm, **2.899392605 versus
2.899392366**, while that update's objective is identical. The
[initial diagnostic serialization failure](../artifacts/benchmarks/gpu-future/reproducibility-cpu-v1/result.json)
is retained: some greedy split lengths were NumPy scalar integers. The successful
rerun stores those lengths as built-in integers, preserving their values and
weights-only loading.

The installed Torch 2.9.1 [determinism documentation](../.venv/Lib/site-packages/torch/__init__.py)
lists CUDA gather/index-select backward among operations that can use deterministic
algorithms. The completed diagnostics repeat frozen final-control weights and
inputs without optimizer updates, recording full gradients/loss/norm and isolated
gather backward. They do not replay the original 100/132-update training prefix or
establish 500-user quality.

The [live-suite launch-gate check](../artifacts/benchmarks/gpu-future/reproducibility-cuda-live-gate-v1/result.json)
refused CUDA execution while the approved batch was active. After its deliberate
case-boundary stop and actual process/lock exit were verified, separate fresh
[ordinary](../artifacts/benchmarks/gpu-future/reproducibility-cuda-ordinary-v1/result.json) and
[deterministic](../artifacts/benchmarks/gpu-future/reproducibility-cuda-deterministic-v1/result.json)
runs completed. Both showed early full-model loss/gradient differences, up to
0.000244140625 maximum absolute BF16 gradient difference, while isolated FP32/BF16
gather backward was bitwise repeatable. A longer
[eight-repeat check](../artifacts/benchmarks/gpu-future/reproducibility-cuda-deterministic-v2/result.json)
found three distinct gradients for user 189 and two for user 237, with a single
gradient fingerprint from the third call onward for each. This does not identify
gather as the cause.

Disabling TorchScript executor optimization in separate fresh processes gave
exact loss/full-gradient/norm repetitions at both locations with
[deterministic algorithms](../artifacts/benchmarks/gpu-future/reproducibility-cuda-jit-unoptimized-v1/result.json)
and [ordinary algorithms](../artifacts/benchmarks/gpu-future/reproducibility-cuda-jit-unoptimized-ordinary-v1/result.json).
The actual executor flag is recorded in these results. This is evidence for an
executor optimization/warmup effect in the tested fixtures; attribution of the
historical multi-update trajectories and repeated-control stability still require
the matched B01 comparisons. B01 applies the same disabled-optimization mode to
both controls and the bundle, preserving the historical results separately.

<!-- gpu-batch1-relative:start -->
### First-batch absolute and percentage changes

Each percentage is **100 × (candidate − matched control) / matched control**.
The reference is the first control for the same model. Positive log-loss
deltas mean worse quality; negative time deltas and positive throughput deltas
mean faster training. Throughput and elapsed-time percentages differ because
throughput is the reciprocal of time at equal review exposure. Cold time includes
F02 cache construction. Missing comparisons remain pending.

| Model / track | Δ Curve LL weighted / user (%) | Δ Instant LL weighted / user (%) | Δ train s (%) | Δ reviews/s (%) | Δ cold train s (%) | Decision |
|---|---:|---:|---:|---:|---:|---|
| baseline/F01 | +0.000074 (+0.017%) / +0.002785 (+0.642%) | +0.014429 (+3.423%) / +0.021135 (+5.027%) | -582.95 (-45.64%) | +17,557 (+83.97%) | -582.95 (-45.64%) | pending repeated control |
| baseline/F02 | -0.041624 (-9.553%) / -0.052015 (-11.985%) | -0.056811 (-13.479%) / -0.072329 (-17.205%) | -291.75 (-22.84%) | +6,190 (+29.61%) | +1,203.17 (+94.20%) | pending repeated control |
| baseline/F03 | -0.003607 (-0.828%) / -0.001072 (-0.247%) | +0.014142 (+3.355%) / +0.020579 (+4.895%) | -0.01 (-0.00%) | +0 (+0.00%) | -0.01 (-0.00%) | pending repeated control |
| baseline/F04 | +0.006015 (+1.381%) / +0.006362 (+1.466%) | -0.002008 (-0.476%) / -0.003991 (-0.949%) | -273.64 (-21.43%) | +5,701 (+27.27%) | -273.64 (-21.43%) | pending repeated control |
| baseline/F05 | — | — | — | — | — | pending |
| baseline/F06 | — | — | — | — | — | pending |
| baseline/F07 | — | — | — | — | — | pending |
| baseline/F08 | — | — | — | — | — | pending |
| expertium/F01 | — | — | — | — | — | pending |
| expertium/F02 | — | — | — | — | — | pending |
| expertium/F03 | — | — | — | — | — | pending |
| expertium/F04 | — | — | — | — | — | pending |
| expertium/F05 | — | — | — | — | — | pending |
| expertium/F06 | — | — | — | — | — | pending |
| expertium/F07 | — | — | — | — | — | pending |
| expertium/F08 | — | — | — | — | — | pending |

[Live table with percentage changes](../artifacts/benchmarks/gpu-batch1-500/summary-relative.md).
[Original scores, adjusted uncertainty and experiment sources](../artifacts/benchmarks/gpu-batch1-500/summary.md).
<!-- gpu-batch1-relative:end -->

<!-- gpu-b01-progress:start -->
### B01 comparison progress

Suite: **failed**; complete cases: **0/6**.

- baseline: pending repeated controls.
- expertium: pending repeated controls.

[Four held-out losses, absolute/percentage deltas against both controls, uncertainty, full-history timing and revisions](../artifacts/benchmarks/gpu-baseline-b01-deterministic-v3/summary.md).
<!-- gpu-b01-progress:end -->

#### Iteration 0004 predeclared screen

Retain Alex F01 with four workers and Expertium F01+F16 with eight workers. On the same 80 training/20 unseen evaluation users, full histories, seed 2025, one epoch and unchanged BF16/FP32 optimizer, batching, update order and schedule, compare fresh incumbents with **F01+F08** for Alex and **F01+F16+F14** for Expertium. F08 fuses the shift/gather and eight interpolation operations to reduce GPU launches and intermediate work; F14 prepares compact metadata and expands dense encodings on GPU to reduce saturated CPU preparation. Feature/kernel numerical differences are possible, so retain separate opt-in fastmode if speed improves with worse scores, even when within the preliminary quality limits. The frozen [plan](../artifacts/retraining/gpu-retrain-r01/iteration-0004/plan.json) defines exact case options; fresh profiles confirm selection before candidate training.

Training exposure remains 7,654,975 reviews and evaluation 868,370 reviews, with 507 Alex and 276 Expertium updates. Compare all four separate losses against fresh incumbent and previous accepted iteration at strict <1%, preferably <2% versus the preserved original baseline; paired-user bootstrap 10,000, seed 20261012, alpha 0.05, family 24. Throughput target is ≥20% versus own fresh incumbent. The complete 100-user resource profile uses a predeclared 60-second minimum observation window because the retained Expertium path now completes in 108 seconds; this changes a telemetry sufficiency threshold, not exposure or timing scope. Budget: two fresh controls, one selected candidate per architecture, paired two-user functional smokes and bounded input checks. Reread the loop before each run; retain both prior no-20%-gain streaks of one. No broader validation is authorized. F08 and F14 remain pending measured results.

#### Iteration 0003 predeclared screen

The user requested one new iteration on 2026-10-11. Reuse the exact 100-user cohort, 80 training/20 unseen evaluation users, full **7,654,975 training / 868,370 held-out reviews**, seed 2025, one epoch, BF16/FP32, deterministic execution, batching and learning-rate schedule from iteration 0002. Run fresh **F01 incumbent controls** for both architectures and one resource-selected candidate per architecture. The shortlist is F01+F06 (defer internal loss/count synchronizations, consolidate scalar transfers and buffer logs while retaining finite-loss/gradient checks) and F01+F16 (8 preparation workers/16 bounded preparations). Two-user functional smokes are infrastructure checks. The previous profile suggests F06 for Alex and F16 for Expertium; confirm the bottlenecks with fresh full-history control profiles before selection.

Compare training reviews/s with each fresh matched incumbent, the previous iteration's F01 run, and its original no-F01 baseline, retaining all three reference provenances. Aim for ≥20% gain over the matched incumbent. Require all four separate weighted/user-average Curve/Instant relative-loss upper bounds below 1% versus both fresh incumbent and previous F01; preferably below 2% versus the original baseline. Conditional paired-user bootstrap: 10,000 resamples, seed 20261012, alpha 0.05, family size24 (2 architectures × 4 metrics × 3 references). Exact logged updates, weights and scores are preferred. Frozen [protocol](../artifacts/retraining/gpu-retrain-r01/iteration-0003/protocol.json) preserves reference hashes, cohort and source identity before experiments. Initially one new iteration was authorized; the [later goal update](../artifacts/retraining/gpu-retrain-r01/iteration-0003/authorization-update-001.json) requests five completed current-loop iterations total, including 0002 and 0003, and rereading the definition between runs. Broader validation remains pending. Both architectures entered with a zero no-20%-gain streak and leave with one; unsuccessful attempts and pending ideas remain recorded.

Outcome: [iteration 0003](../artifacts/retraining/gpu-retrain-r01/iteration-0003/summary.md) contains four full-history cases and four functional smokes. Both fresh incumbents exactly repeat iteration 0002, and both candidates retain exact final weights, every logged update, held-out inputs, label/review coverage and all four scores. Alex F06 loses 3.59% throughput and adds 4.04% process wall; Expertium F16 gains 19.82% throughput and reduces process wall by 12.74% versus its fresh incumbent. One timing per configuration does not establish repeatability; 19.82% remains below the strict 20% target. The revised log uses separate value/delta columns and two-decimal percentages. Raw source snapshots and both versions of the loop definition remain preserved.


## R01 tracks before round 0031–0033

Superseded detailed track notes are preserved below. Current outcomes and diagnostic attempts are in the [0031–0033 round summary](../artifacts/retraining/gpu-retrain-r01/codex-round-0031-0033-summary.md).

| F35 | Parallel exact row-selection backward and further recurrence tuning | Medium / medium | Load-pipelining part accepted preliminary in 0030 inside the E27-F35-F38 bundle (option `prefetch_rows`, BF16 only); splitting long destinations and 8-row recurrence still pending; broader validation pending | All four scores, every update/gradient norm, final weights and held-out inputs identical to 0027 | Bundle with F38: 53.422 s / 143,293 reviews/s (+2.28% vs 0027) | [Iteration 0030](../artifacts/retraining/gpu-retrain-r01/iteration-0030/summary.md). F27's BF16 row-selection backward now loads the next 64 source indices while adding the current 64 values, keeping the sequential per-destination sum in source order (the clamped padding row has up to 47,595 sources). Real-index cases: BF16 summed 40.1 to 15.6 ms, worst 3.91 to 0.87 ms, all bitwise equal ([check](../artifacts/diagnostics/r01-f35-exact-check-v2/result.json), [synthetic](../artifacts/diagnostics/r01-f35-exact-check-v2-synthetic/result.json)); FP32 measured slower (worst 2.3 to 3.1 ms) so it keeps the sequential kernel. Next: the same pipelining for long shift-backward destinations; 8 rows/thread recurrence (80.6 vs 85.4 ms backward on captured shapes) still unmeasured in a full run. |
| F37 | Loss without host synchronization | High / low | Tested in 0026 (E24-F37): exact parity, speed target missed (-1.75%); retained opt-in experimental option `sync_free_loss` (Expertium, needs F33) | All four scores, every update/gradient norm, final weights and held-out inputs identical to 0024 | 56.749 s / 134,891 reviews/s (-1.75% vs 0024) | [Iteration 0026](../artifacts/retraining/gpu-retrain-r01/iteration-0026/summary.md). Removes the two NaN checks after the forward and four unused label-count `.item()` calls in Expertium `_get_loss`; any NaN they catch also makes `average_loss` non-finite, which F33 still rejects one update later. Fixed-window profiles: forward-and-loss host time 46.4 to 30.9 ms, but the host then blocks on a full command buffer (14 to 57 ms); four consecutive profiles drifted 232 to 336 ms per update regardless of option ([profiles](../artifacts/diagnostics/r01-dispatch-profile-v1/)). Retest on top of F39 or F38 only with a stated reason. |
| F42 | Cheaper worker batch preparation | Medium / medium | Proposed; pending | — (pending) | — (pending) | Six workers sum 187–236 s of preparation per 55 s run (worker-pool CPU 58–72%), competing with the host-bound dispatch thread; the same work varied 187 to 236 s between 0027 and 0028 with the machine state. Profile one worker's `prepare` hot spots (NumPy sample construction, packing, pickling) and reduce CPU per batch with byte-identical outputs. |
