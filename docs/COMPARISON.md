# First-20-user CPU comparison

Measured locally on 2026-10-11 with PyTorch 2.9.1, float32, and one CPU thread. Both architectures trained successfully from fresh weights across user IDs 1–20. The smaller architecture uses **14.33× fewer parameters**, was **1.60× faster during measured optimization**, and used **12× less float32 tensor state per card**. Its final validation losses were lower in this short run; this does not establish superiority after full training.

## Results

| Measurement | Baseline | Expertium iter36 |
| --- | ---: | ---: |
| Trainable parameters | 2,762,884 | 192,800 |
| Float32 parameter memory | 10.540 MiB | 0.735 MiB |
| Saved model checkpoint | 10.730 MiB | 0.907 MiB |
| Optimization time | 20.77 s | 12.98 s |
| Raw review events / optimization second | 118.65 | 189.90 |
| Float32 tensor state per card | 51.00 KiB | 4.25 KiB |
| Float32 tensor state per note | 34.00 KiB | 12.75 KiB |
| Validation Curve log loss before training | 0.683394 | 0.673196 |
| Validation Curve log loss after training | 0.649553 | 0.636103 |
| Validation Instant log loss before training | 0.498306 | 0.498306 |
| Validation Instant log loss after training | 0.497008 | 0.496406 |
| Validation Instant rating cross-entropy after training | 1.279654 | 1.275367 |

Lower log loss and cross-entropy are better. Curve is the corrected next-review recall prediction; Instant log loss is binary recall log loss from `1 - P(Again)`, distinct from the four-rating cross-entropy used in training. Validation means are weighted by the number of applicable labels: **1,052 Curve targets** and **1,509 Instant targets**, identical for both models. Curve and Instant are scored independently.

The final small-minus-baseline difference is **-0.013450 Curve** and **-0.000601 Instant**. The Curve initialization already favored the small model (64 uniformly weighted basis curves versus 128), so the final Curve gap is not entirely a difference in learning. Both models improved their held-out Curve and Instant losses relative to their own initial weights.

## Protocol

- Train on reviews **1–128** of each of users **1–20**; validate on reviews **129–256** of those same users.
- Prepare features once and reuse exactly the same in-memory batches, ID/calendar encodings, query rows, and labels for both models. Batch fingerprint: `9bb628ba8f0eebf4bdbf42ed594c5e71c85616bb82b4af4c97929ff4993c2085`.
- Mask next-review labels outside the corresponding split. Query labels belong to their current review inside that split. Validation targets never serve as training targets.
- Use 32-review chunks, one selected-data epoch, seed 2025, float32, one CPU thread, the common AdamW parameter groups, gradient clipping at 0.5, and five-step linear warmup to `7e-4`. Both models use their default TorchScript/reference recurrence; state quantization and low-rank QAT are disabled.
- Both perform **77 optimizer steps**, with all 20 users contributing. Three 32-review chunks have no supervised labels after split filtering and are skipped: **2,464 raw review events** contribute to updates out of 2,560 selected events. Query rows are additional model inputs.
- Reset recurrent context between chunks, consistently with the existing bounded CPU trial. Validation likewise uses independent chunks; it does not replay the complete training prefix into recurrent state.
- Warm the forward path through initial validation before measuring optimization. Measure forward/loss, backward, gradient clipping, and AdamW updates; exclude feature preparation, validation, model construction, serialization, and logging from the reported training times. Common data preparation took **38.38 s**. The baseline runs first, the smaller model second; timings are one observation each.

This comparison reruns the baseline instead of reusing the original smoke-trial result. The original trial permitted next-review labels beyond its selected training prefix and had no held-out validation. Its checkpoint and outputs remain in their original directories.

## Exact model and provenance

The baseline is the unchanged 2,762,884-parameter model pinned in [`../upstream.json`](../upstream.json). The smaller model is Expertium's historical **iter36 fp32** architecture from [`39666c69e3fae73f60ff5c35cd6d175f9cf5cbe7`](https://github.com/Expertium/rwkv-anki-autoresearch/tree/39666c69e3fae73f60ff5c35cd6d175f9cf5cbe7), with width 32, one head, card/deck/note/preset/user layer counts `[1,4,3,3,3]`, channel-mixer factor 1.0, LoRA ranks 16/16/8/16, and 64 Curve bases/interpolation points. [`../expertium/upstream.json`](../expertium/upstream.json) records original and adapted hashes. Only model/architecture import namespaces changed; common data preparation and optimizer code are reused. No pretrained checkpoint, distillation, codebook, or state compression is used.

State figures count the actual float32 WKV matrices and both token-shift tensors returned by recurrent inference for one entity. They exclude Python containers, IDs, allocator overhead, model weights, and activations. This is uncompressed state, so the separate nine-byte research result in the README does not apply. Parameter memory counts tensors; checkpoint sizes also include serialization metadata.

## Verification and reproduction

Run `just compare-cpu` from the repository root. Configuration is [`../configs/compare-cpu.toml`](../configs/compare-cpu.toml), using the shared [`../scripts/compare.py`](../scripts/compare.py) runner. `just check` verifies source/config parsing, provenance, model counts, strict recurrent loading, and recurrent inference for both architectures. The comparison rejects invalid losses/gradients, checks that every selected user contributed updates, verifies shared step/event/validation-label counts, reloads model checkpoints, and exercises three recurrent inference steps.

Generated outputs remain Git-ignored:

- `artifacts/comparison-20-users/comparison.json`: complete configuration, label counts, timing, state sizes, and before/after losses.
- `artifacts/comparison-20-users/baseline_metrics.jsonl` and `expertium_iter36_metrics.jsonl`: step losses, gradient norms, and covered review ranges.
- `checkpoints/comparison-20-users/baseline.pth` and `expertium_iter36.pth`: model state dictionaries, with matching `_optim.pth` optimizer state dictionaries.

Each invocation starts from fresh weights and replaces these comparison outputs. This is a short, single-seed temporal comparison on early reviews of 20 known users, without full-history context, benchmark-label alignment, hyperparameter tuning, or a new-user test set. It confirms that both models train and offers local measurements; it does not reproduce Expertium's published full-training results, estimate statistical significance, or measure deployed inference speed.

## Relation to the completed GPU training track

The [RTX 5080 training report](GPU_TRAINING.md) records the completed 100-user default-model training and the separate default-versus-Expertium **training throughput** comparison. Those GPU measurements include forward passes, backpropagation, and AdamW weight updates. They measured 41,240 reviews/s for the default model, 59,710 for Expertium at identical settings, and 76,002 for tuned Expertium. They are not inference-speed measurements or accuracy results for this 20-user protocol.

Only the default model completed the full 100-user GPU pass. Expertium completed the training-speed scenarios; its full-history training and a matched full-history accuracy study remain separate work. CPU recurrent inference checks verify that saved training checkpoints load correctly.

## Separate bounded CUDA accuracy protocol

The historical CPU numbers above remain the original observation. The shared runner now supports CPU float32 and CUDA float32/BF16, grouped independent chunks, one-time device transfer, synchronized GPU timing, and an optional automatically invalidated prepared-input cache. The metadata loader now reads existing per-user cards/decks partitions directly; missing partitions retain the original filtered-root schema behavior. Feature engineering and both model architectures are unchanged.

Use `just setup`, `just build-cuda`, `just check-cuda`, then `just compare-cuda` on the NVIDIA desktop. [`../configs/compare-cuda.toml`](../configs/compare-cuda.toml) selects eight 32-review chunks per group, BF16 computation with float32 master weights, and separate output/cache/checkpoint directories. Both models share the same GPU inputs, supervision, precision, and update schedule. All 80 selected training chunks fit into 10 labelled groups, so this changes the optimizer trajectory relative to the historical 77-step CPU trial. CUDA timing includes master/model copies and gradient transfers; feature preparation, the initial device transfer, validation, and serialization are separate. Complete training-loop time includes logging and the finite-value checks as well. This bounded CUDA accuracy protocol has not been run on the desktop. The completed GPU performance measurements use separate scripts and data selections, as described in the GPU report.

Prepared-input caches contain only tensors and primitive metadata and are loaded with `weights_only=True`. Cache keys include the data selection, chunk grouping, encoding seed, source hashes, PyTorch version, resolved dataset path, and the selected Parquet files' sizes and modification timestamps. Dataset edits that deliberately preserve both size and timestamp require deleting the cache. Each model still starts from fresh weights; only preparation is cached.

Verification of the shared runner on 2026-10-11 reproduced the original CPU batch fingerprint and both models' before/after validation losses exactly, with 77 updates and 2,464 contributing review events each. Direct metadata partition reads reduced preparation from the historical 38.38 seconds to 3.70 seconds in that rerun. The eight-chunk preparation check retained 1,052 Curve and 1,509 Instant validation targets, built 10 train/10 validation groups across all 20 users, and round-tripped the prepared cache with identical features and labels. Preparation including cache serialization took 3.49 seconds; a cache hit including file/source signature checks and loading took 0.033 seconds locally. These timings are local observations, not RTX 5080 measurements.

The eight-chunk CPU smoke run also completed 10 updates per model over all 2,560 selected review events, with all 20 users contributing, finite losses/gradients, and successful recurrent checkpoint loading. After only 10 updates, Curve log loss was 0.635660 for the baseline and 0.664108 for iter36; Instant log loss was 0.500385 and 0.498575 respectively. The changed Curve ordering reinforces that batch size and update count alter this short comparison; a speed measurement is not an accuracy guarantee. This smoke run verifies grouped training and caching, not BF16 CUDA performance.
