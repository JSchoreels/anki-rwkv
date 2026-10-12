# R01 Expertium update path: where host and GPU time go

Read this instead of re-reading the model and training code when picking R01 candidates. State as of the accepted 0030 reference (E30: E27 + F35 + F38, 143,293 reviews/s; E27 = E24 + six preparation workers). Re-verify any line you rely on if the files named below changed since the 0028–0030 round commit.

## Machine

- AMD Ryzen 7 9850X3D: 8 physical cores / 16 logical (SMT siblings are logical 2k, 2k+1; Windows CPU sets rank core 6 then 10 highest), RTX 5080, Windows 11. Background desktop load is typically 7–17% total CPU.
- Run-to-run spread exceeds the 2% exact gate. Fixed-window dispatch profiles (`scripts/profile_r01_dispatch.py`) drifted 232→336 ms per update over four consecutive runs of the same path; do not use them to decide effects under about 10%. Use them for counts (launches, syncs, dispatches), not wall time.

## Model structure (`expertium/architecture.py`, `expertium/model/`)

- `SrsRWKV.forward_batch` runs five RWKV7 streams in sequence (card 1 layer, deck 4, note 3, preset 3, global 3 = 14 layers), d_model 32 (one head width). Each stream runs once per split (`sub_gather` lengths), so a large update has about 31 layer instances; each instance is a time mixer (about 60 small ops: LayerNorm, shift gather, 8-way lerp, 4 linears, LoRAs, sigmoid/softplus/exp, normalize, WKV recurrence, GroupNorm, bonus) and a channel mixer.
- All modules are TorchScript; R01 disables graph-executor optimization, so graph edits do not change fusion or numerics.
- Per large update (0022–0024 profiles): about 12,600 kernel launches, 40,000 ATen dispatches, 4,100 `fill_`, 1,300 `mm`; host stages backward ≈130–175 ms, forward and loss ≈31–52 ms, clipping/child-to-master/optimizer ≈10 ms each. Device time is dominated by the recurrence backward (≈84 ms) and forward (≈13 ms).
- Mean update is 45,838 reviews (1,274–65,536); launch count is nearly size-independent, so small updates are host-bound and large ones GPU-bound.

## Synchronization points

- F33 (`deferred_reads`) reads loss/norm/finite flags one update late.
- Expertium `_get_loss` has two NaN checks and four label-count `.item()` calls; F37 (`sync_free_loss`) removes them (exact, -1.75% in 0026, opt-in).
- `profile_r01_dispatch.py` honours `deferred_reads` (and the older name `deferred_checks`); profiles made before this fix ran without deferred reads unless `--extra-options '{"deferred_checks": true}'` was given.

## Custom exact kernels (all bitwise-checked against PyTorch 2.9.1 / NVRTC 13.4 / sm_120)

- F26/F27 `rwkv/model/exact_index.py`: shift-gather and row-select backward. F38 (`shared_shift_orders`) caches each split's sort order and starts across its layers; F35 (`prefetch_rows`, BF16 only) pipelines index/value loads in the row-select backward, whose clamped padding row has up to 47,595 sequentially summed sources. FP32 is slower with that prefetching.
- F31 `scripts/gpu_row_tiles.py` + `scripts/kernels/rwkv_row_tiles_exact.cu`: 4-row recurrence.
- F32 `rwkv/model/exact_group_norm.py`: GroupNorm weight/bias backward.
- Custom kernels launch through `nvrtc.raw_launch`; with F38 `stream_launches` it passes a cached CuPy `ExternalStream` (4.8 µs per launch) instead of per-call `Device`/`ExternalStream` contexts (16.3 µs). `scripts/check_gpu_exact_index.py --f38 --f35` rechecks all of these against deterministic PyTorch on real batches.

## CPU side

- Preparation workers (spawned pool, ordered prefetch, depth = 2 × workers) build identical seeded batches regardless of count. Six workers (0027) summed 187 s of preparation with 1.04 s total queue wait; eight summed 229 s.
- Pinning the main thread to its own core (F36 `host_placement`) was slower (-7.22%). Five workers (F40, 0029) starved the queue (6.7 s waits): six is the minimum.
- Identical worker work varied 187 s (0027), 236 s (0028), 234 s (0029), 208 s (0030) with machine state at similar 9–10% background load; compare summed `preparation_seconds` and validation seconds across runs to spot a slow machine state before reading a speed delta.
- Python GC inside the training loop costs about 0.17 s per run (`training_gc` in `result.json`); not worth optimizing. Profiler windows trigger their own full collections (4.16 M tracked objects, ~200 ms each), so do not measure GC there.
