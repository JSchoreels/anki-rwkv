# R01 Expertium update path: where host and GPU time go

Read this instead of re-reading the model and training code when picking R01 candidates. State as of accepted 0033 (E33 = E30 + F37 + F42): 170,718 reviews/s, +19.14% versus 0030; E30 retains six workers, prefetched row backward and shared orders/stream launches. Eight rows per recurrence thread were exact but slower in 0032; accepted layout stays four. Re-verify consequential code lines if their files changed after the 0031–0033 implementation commit.

## Machine

- AMD Ryzen 7 9850X3D: 8 physical cores / 16 logical (SMT siblings are logical 2k, 2k+1; Windows CPU sets rank core 6 then 10 highest), RTX 5080, Windows 11. Background desktop load is typically 7–17% total CPU.
- Run-to-run spread exceeds the 2% exact gate. Fixed-window dispatch profiles (`scripts/profile_r01_dispatch.py`) drifted 232→336 ms per update over four consecutive runs of the same path; do not use them to decide effects under about 10%. Use them for counts (launches, syncs, dispatches), not wall time.

## Model structure (`expertium/architecture.py`, `expertium/model/`)

- `SrsRWKV.forward_batch` runs five RWKV7 streams in sequence (card 1 layer, deck 4, note 3, preset 3, global 3 = 14 layers), d_model 32 (one head width). Each stream runs once per split (`sub_gather` lengths), so a large update has about 31 layer instances; each instance is a time mixer (about 60 small ops: LayerNorm, shift gather, 8-way lerp, 4 linears, LoRAs, sigmoid/softplus/exp, normalize, WKV recurrence, GroupNorm, bonus) and a channel mixer.
- All modules are TorchScript; R01 disables graph-executor optimization, so graph edits do not change fusion or numerics. Module flags set from Python must be real `bool` values (a TorchScript attribute rejects strings or other types).
- Data flow per update: `bulk_to_device` (`scripts/gpu_r01_update.py`) copies each batch into fresh device storage and rebuilds typed views; nothing writes those tensors afterwards. `forward_batch` (`expertium/model/srs_model.py`) makes one `time_shift_select.view(-1, sub_len)` per split and passes it to every layer (time and channel mixer) of that stream; row selection uses `clamp(split_gather, min=0)`, so padding maps to row 0 (the long F27/F35 destination).
- Per large update (0022–0024 profiles): about 12,600 kernel launches, 40,000 ATen dispatches, 4,100 `fill_`, 1,300 `mm`; host stages backward ≈130–175 ms, forward and loss ≈31–52 ms, clipping/child-to-master/optimizer ≈10 ms each. Device time is dominated by the recurrence backward (≈84 ms) and forward (≈13 ms).
- Mean update is 45,838 reviews (1,274–65,536); launch count is nearly size-independent, so small updates are host-bound and large ones GPU-bound.

## Synchronization points

- F33 (`deferred_reads`) reads loss/norm/finite flags one update late.
- Expertium `_get_loss` has two NaN checks and four label-count `.item()` calls; F37 (`sync_free_loss`) removes them (exact; -1.75% on E24 in 0026, accepted +6.07% on E30 in 0031; enabled in E33).
- `profile_r01_dispatch.py` honours `deferred_reads` (and the older name `deferred_checks`); profiles made before this fix ran without deferred reads unless `--extra-options '{"deferred_checks": true}'` was given.

- `scripts/check_r01_exact_options.py`: E27 vs E30 options 5/5 batches bitwise equal; `{}` vs `{"nondeterministic": true}` was also 5/5 equal on these batches (one run, so not a proof that deterministic mode is free); `--self-test` (one flipped weight bit) fails 0/5 as it should.

## Custom exact kernels (all bitwise-checked against PyTorch 2.9.1 / NVRTC 13.4 / sm_120)

- F26/F27 `rwkv/model/exact_index.py`: shift-gather and row-select backward. F38 (`shared_shift_orders`) caches each split's sort order and starts across its layers; F35 (`prefetch_rows`, BF16 only) pipelines index/value loads in the row-select backward, whose clamped padding row has up to 47,595 sequentially summed sources. FP32 is slower with that prefetching.
- F31 `scripts/gpu_row_tiles.py` + `scripts/kernels/rwkv_row_tiles_exact.cu`: accepted 4-row recurrence; 8 rows are now opt-in (F35, 108/108 BF16 synthetic and 5/5 accepted-bundle batches exact, but -2.05% throughput in 0032).
- F32 `rwkv/model/exact_group_norm.py`: GroupNorm weight/bias backward.
- Custom kernels launch through `nvrtc.raw_launch`; with F38 `stream_launches` it passes a cached CuPy `ExternalStream` (4.8 µs per launch) instead of per-call `Device`/`ExternalStream` contexts (16.3 µs). `scripts/check_gpu_exact_index.py --f38 --f35` rechecks all of these against deterministic PyTorch on real batches.

## CPU side

- Preparation workers (spawned pool, ordered prefetch, depth = 2 × workers) build identical seeded batches regardless of count. Six workers (0027) summed 187 s of preparation with 1.04 s total queue wait; eight summed 229 s.
- Pinning the main thread to its own core (F36 `host_placement`) was slower (-7.22%). Five workers (F40, 0029) starved the queue (6.7 s waits): six was the minimum before F42; E33 worker retuning is pending.
- Identical worker work varied 187 s (0027), 236 s (0028), 234 s (0029), 208 s (0030) with machine state at similar 9–10% background load; compare summed `preparation_seconds` and validation seconds across runs to spot a slow machine state before reading a speed delta.
- Python GC inside the training loop costs about 0.17 s per run (`training_gc` in `result.json`); not worth optimizing. Profiler windows trigger their own full collections (4.16 M tracked objects, ~200 ms each), so do not measure GC there.

## Round 0031–0033 preparation and diagnostic findings

- F42 (`columnar_samples`, requires `fast_samples`) extracts unchanged arrays once per batch and uses the original pandas Series arithmetic and groupby-first for segment-local features. All 2,560 checked sample chunks and five packed/RNG edge windows were byte-identical; the full 0033 run also matches every update and final score. See the [round summary](../artifacts/retraining/gpu-retrain-r01/codex-round-0031-0033-summary.md).
- E33 six workers summed 110.3 s of preparation, waited 0.59 s, used 42.4% worker-pool CPU and reached 69.1% GPU activity, versus E31 232.3 s / 2.11 s / 79.5% / 54.2%. Five workers may now have headroom; the earlier five-worker failure was before F42. This retune is pending, not measured.
- The option exactness helper now honors reference `main_threads`, uses accepted fast samples and batched encodings for common input preparation, and prints per-batch progress. Changed worker options still require separate batch-byte checks. Restricted Windows execution can add substantial system-call overhead: one interrupted helper spent 241.875 system CPU seconds over 282.5 wall seconds; the identical final CUDA check passed outside the sandbox. Keep this diagnostic overhead separate from training measurements and use a workspace TMP/TEMP directory for fresh NVRTC compilation.
