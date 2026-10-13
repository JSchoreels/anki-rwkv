# GPU training results and run guide

Measured on 2026-10-11 on Windows with an **RTX 5080, 16 GB VRAM, and PyTorch 2.9.1+cu128**. GPU training, the default model's full 100-user trial, and the default-versus-Expertium throughput comparison are complete.

All reported speed scenarios perform **actual training**: forward passes, loss computation, backpropagation, gradient clipping, and AdamW weight updates. The full-history run saves the trained default-model weights. Recurrent inference is an additional checkpoint compatibility check, separate from training throughput.

## Main result: reviews per second

Expertium iter36 was **45% faster with identical batch and worker settings**, and **84% faster after its own tuning**. The current smaller-model bottleneck is CPU preparation.

| Model | Chunks / workers | Precision | Complete-update reviews/s | Transfer + training reviews/s | Peak Torch allocation |
| --- | --- | --- | ---: | ---: | ---: |
| Default | 128 / 4 | BF16 | **41,240** | 41,897 | 9.00 GiB |
| Expertium iter36 | 128 / 4 | BF16 | **59,710** | 93,099 | 2.08 GiB |
| Expertium iter36 | 128 / 4 | float32 | 57,996 | 91,111 | 3.51 GiB |
| Expertium iter36 | 256 / 6 | BF16 | **76,002** | 121,233 | 4.16 GiB |
| Expertium iter36 | 384 / 6 | BF16 | 73,012 | 146,405 | 5.98 GiB |
| Expertium iter36 | 512 / 6 | BF16 | 69,184 | 158,041 | 7.77 GiB |

The default architecture has **2,762,884 parameters**; Expertium's pinned historical iter36 has **192,800 parameters**. Its width is 32 with one head, card/deck/note/preset/user layers `[1,4,3,3,3]`, and 64 Curve bases/interpolation points. Both use float32 master weights and AdamW state. No pretrained checkpoint, distillation, quantization, or state compression is used. Source revisions and hashes are recorded in [`../upstream.json`](../upstream.json) and [`../expertium/upstream.json`](../expertium/upstream.json).

### What the rates measure

Each scenario starts from fresh weights with seed 2025 and its own optimizer state. Both models use the same preparation and labels from **user 17, reviews 4097–12288**, repeated to supply enough batches. Each independent chunk contains **128 raw reviews**, plus generated query rows; recurrent context resets between chunks. The benchmark uses six warmup updates and 64 measured updates, the NVRTC backend, and a 0.72 Torch memory budget.

**Complete-update reviews/s** divides raw review presentations by the total measured update time. It includes preparation queue waits, CPU-to-GPU transfer, forward/backward, master/model and gradient copies, clipping, AdamW, and synchronization. **Transfer + training reviews/s** excludes preparation waits. Both exclude source file loading and cold setup. Repeated review presentations are counted each time; these rates do not imply that 76,002 unique reviews are loaded from disk each second.

At matched settings, Expertium's transfer/training rate is **2.22×** the default's, but complete-update throughput improves by 45% because preparation cannot keep pace. Doubling its batch from 256 to 512 makes GPU work faster per review while reducing complete-update throughput. The midpoint at 384 also underperforms 256. Thus **256 chunks and six workers** is the best measured Expertium throughput setting for this window; it has not been validated by a full 100-user Expertium run.

These are steady-window speed measurements. They do not establish a full-epoch runtime or an accuracy ranking. Larger batches also reduce optimizer updates per epoch and change the optimization trajectory.

## Completed full-history default run

Only the **default model** completed a full pass over users 1–100 in this GPU track.

| Check | Result |
| --- | --- |
| Dataset revision | `75299740cff05894ef42d7ad990666691efdd2da` |
| Users | 100, complete review histories |
| Raw reviews selected and used in updates | **5,226,500** |
| Optimizer updates | **1,029** |
| Skipped unlabelled chunks | **0** |
| Review coverage | Continuous per user, without repeats or gaps |
| Losses and gradient norms | All finite |
| Final checkpoint | Float32 master weights; strict CPU recurrent load and ten inference steps passed |
| Fixed training-chunk loss | 1.8314 → 1.4989 |

The fixed chunk is training data, so its improvement is not held-out accuracy. Users 1–100 are normally reserved for benchmark validation; this trial trains on them and cannot be used as a benchmark validation result.

The run resumed through reference/NVRTC, precision, batching, prefetch, and memory-setting changes. Its recorded cumulative runtime is therefore not a fresh run with the final settings and is not used for the model speed comparison. The 1,029 updates likewise reflect those changing settings.

### Utilization, power, and memory

The final 128-chunk segment peaked at **9.76 GiB of live Torch allocations**. One-second NVIDIA samples in that segment showed **7% median / 98% maximum utilization**, **84.5 W average / 150.1 W maximum board power**, and **13,360 MiB maximum device memory use**. Device memory includes allocations outside live Torch tensors.

Utilization across real users is bursty: each new user requires loading/preparation, and the last batch of a user can be smaller. The earlier repeated-window batch search reached **94–97% utilization during steady updates**. That observation does not describe the whole 100-user run. High utilization alone also does not guarantee high throughput when Windows pages GPU memory.

The runner left the **400 W cap unchanged**. This card reports a hardware cap range of **250–400 W**, so a 200 W hardware cap is unavailable through this interface. The configured `POWER_LIMIT_WATTS = 0` means monitor only; manage power separately with MSI Afterburner.

## Default settings and why they were selected

The completed 2026-10-11 run used the settings below. The current [`train-gpu-trial.toml`](../configs/train-gpu-trial.toml) permits a 0.95 Torch GPU allocation budget following the later R01 update; the historical measurements retain their original 0.72 budget.

```toml
USER_START = 1
USER_END = 100
REVIEWS_PER_USER = 0
EPOCHS = 1
CHUNK_REVIEWS = 128
BATCH_CHUNKS = 128
DTYPE = "bfloat16"
PREFETCH_WORKERS = 4
PREFETCH_BATCHES = 8
GPU_MEMORY_FRACTION = 0.72
WKV_BACKEND = "nvrtc"
POWER_LIMIT_WATTS = 0
```

A full batch contains **16,384 raw reviews** in independent contexts. Preparation uses group indices rather than copying each group into a new DataFrame. Four worker processes prepare ordered batches while the GPU trains, with a bounded queue of eight. The Torch memory budget and release of unused allocations between users prevent cache growth from pushing Windows into GPU-memory paging.

### Historical default-model batch search

This search used the checkpoint at step 720, with the same repeated history window, six warmup updates, and 64 measured updates per scenario. It is separate from the fresh-model comparison above.

| Chunks | Precision | CPU workers | Complete-update reviews/s | Peak Torch allocation |
| ---: | --- | ---: | ---: | ---: |
| 8, original group-copy preparation | float32 | 0 | 2,101 | 1.04 GiB |
| 8, group-index preparation | float32 | 0 | 4,884 | 1.08 GiB |
| 8, ordered prefetch | float32 | 2 | 6,965 | 1.08 GiB |
| 32 | float32 | 4 | 23,632 | 3.92 GiB |
| 64 | float32 | 4 | 33,099 | 7.81 GiB |
| 128 | BF16 | 4 | 43,010 | 9.00 GiB |
| 144 | BF16 | 4 | **44,687** | 10.13 GiB |
| 152 | BF16 | 4 | 40,278 | 10.87 GiB |
| 160 | BF16 | 4 | 27,914 | 11.28 GiB |

Doubling followed by midpoint refinement found a repeated-window peak near 144 chunks, about **21×** the original measured pipeline. However, a real user-38 batch at 144 exceeded the allocation budget before its optimizer update. The saved position resumed at **128 chunks**, and the remaining histories completed with verified coverage. **128 is the full-history default**; 144 is only the historical window peak. BF16 was slower at small batches but saved enough memory to support the larger batches that improved throughput.

## Training without large LMDBs

Building hundreds of gigabytes of LMDBs is **not required for this GPU trial**. That requirement belongs to the original [`rwkv/train_rwkv.py`](../rwkv/train_rwkv.py) workflow, which loads precomputed training/validation batches from LMDB. The current preprocessing configurations set 205 GB and 195 GB train/test map limits; map limits are capacities rather than guaranteed disk usage.

Our [`train_gpu_trial.py`](../scripts/train_gpu_trial.py) instead calls `get_rwkv_data` to read each user's raw Parquet history and metadata. [`trial_batches.py`](../scripts/trial_batches.py) applies the shared feature/sample preparation in RAM, and its bounded worker queue supplies batches to GPU backpropagation and AdamW. LMDB is a storage/preprocessing choice, not a prerequisite of the model's training computation.

The completed 100-user run used this direct path and did not build the full train/test/label LMDBs. Its selected dataset download was approximately **107 MiB**; that figure covers only users 1–100 and associated metadata/files, not the complete 10,000-user dataset. Keeping preparation online avoids the large precomputed databases but makes CPU preparation a throughput constraint, especially for Expertium.

The original full benchmark preprocessing/evaluation workflow remains available separately. This trial does not reproduce its benchmark-label alignment or held-out evaluation protocol.

## Windows GPU setup

Run from the repository root in PowerShell with Python 3.12 and `uv` installed:

```powershell
uv sync --locked --extra nvrtc
.\.venv\Scripts\python.exe scripts/check.py
```

The tested backend runtime-compiles the original sequential CUDA kernels through CuPy and PyTorch's bundled NVRTC, with the optional CUDA 12.8 runtime headers. BF16 additionally needs CCCL headers containing `nv/target`. This machine used headers from its installed CUDA 13.4 toolkit; CUDA 12.8 supplies the matching runtime and NVRTC. Set `CUDA_PATH` or `CUDA_HOME` if toolkit headers are not found automatically. A host C++ compiler is not required for this path.

NVRTC supports head dimension 32 and **1–383 timesteps**, sufficient for these 128-review chunks including query rows. For longer sequences, use `WKV_BACKEND = "cuda"` after building the native extension with a compatible CUDA toolkit and host compiler. `"reference"` selects the slower PyTorch recurrence on CUDA. NVRTC requires an environment without the native `RWKV_CUDA` extension; if that extension is installed, select the `cuda` backend instead.

### Download the selected users

Authenticate with a Hugging Face account approved for the gated dataset. Keep tokens out of source files and chat. Skip downloading if the selected dataset partitions already exist.

```powershell
$env:HF_HOME = Join-Path $PWD 'artifacts/cache/huggingface'
uv run --locked --extra nvrtc --with huggingface-hub hf auth login
uv run --locked --extra nvrtc --with huggingface-hub python scripts/download_trial_data.py --user-start 1 --user-end 100
```

The downloader resolves the dataset revision, uses that revision throughout the download, and records it in `data/anki-revlogs-10k/download_manifest.json`. A later download may resolve a newer revision; compare the manifest with the measured revision above before interpreting review counts. User 92 in the measured revision has no `cards` partition; the existing preprocessing fills missing hierarchy IDs after its left joins.

As of 2026-10-11, the full active local dataset is the timestamp variant,
`open-spaced-repetition/anki-revlogs-10k-id`, at revision
`93a7e8242690022d75943f67943901f81c556b0e`. The local and NAS originals are
preserved as sibling folders named `anki-revlogs-10k-original-20261011`.
The measured runs above used the original dataset. To reproduce their data
source, set `DATA_PATH` to that original local backup or use a separate output
folder for an original-dataset download. The downloader's default repository
remains the original; it rejects an output manifest naming a different
repository before downloading. Select the timestamp variant explicitly:

```powershell
uv run --locked --extra nvrtc --with huggingface-hub python scripts/download_trial_data.py --repository open-spaced-repetition/anki-revlogs-10k-id --user-start 1 --user-end 100 --output data/anki-revlogs-10k-id-selected
```

[Full replacement record](../artifacts/data/dataset-inspection/anki-revlogs-10k-id/replacement.json)
and [source file inventory](../artifacts/data/dataset-inspection/anki-revlogs-10k-id/remote_files.json)
pin the installed variant. Its file integrity and five-user CPU loading smoke
checks do not constitute a model quality or speed comparison.

### Verify and launch

```powershell
# Run the synthetic CUDA and checkpoint checks in the foreground.
.\scripts\run_gpu_trial.ps1 -Check -Worker
# Exact preparation parity, worker ordering, both precisions, and interrupted resume.
.\.venv\Scripts\python.exe scripts/check_gpu_perf.py
# Launch the full-history default-model job in a hidden process.
.\scripts\run_gpu_trial.ps1
```

The launcher writes PID/status to `artifacts/benchmarks/gpu-trial/process.json` and appends to `console.log`. A fresh run refuses to overwrite an existing training checkpoint. This workspace already has the completed checkpoint; preserve it and choose separate `OUTPUT_DIR` and `CHECKPOINT_DIR` values for another run. The GPU trial trainer currently selects only the default model; the throughput comparison script supports both architectures.

To stop an active job cleanly, create `artifacts/benchmarks/gpu-trial/STOP`. It checkpoints between batches. Remove that file and use `.\scripts\run_gpu_trial.ps1 -Resume` to continue from the next untrained review position with saved RNG state. Backend, power, batch size, precision, worker/queue settings, and memory budget may change on resume; transitions and recovered errors are recorded. Data selection, architecture, and optimizer settings must match. Resuming a completed checkpoint does not start a new epoch.

The PowerShell launcher fixes the default output location and sets workspace-local CuPy cache and temporary paths. For custom output paths, invoke the trainer directly with `--config` and, when needed, `--resume`; it runs in the foreground. `POWER_LIMIT_WATTS = 0` requires no administrator rights. A positive supported limit temporarily applies a cap and restores the startup setting on exit, requiring administrator rights on Windows.

### Reproduce throughput measurements

```powershell
# Fresh default-versus-Expertium comparison and Expertium batch search.
.\.venv\Scripts\python.exe scripts/compare_gpu_throughput.py
# A single default-model scenario using a copy of the existing checkpoint.
.\.venv\Scripts\python.exe scripts/benchmark_gpu_trial.py --prefetch-only --workers 4 --batch-chunks 64 --dtype float32
```

The equivalent recipes are `just compare-gpu-throughput` and `just benchmark-gpu` in a compatible shell. These scripts leave training checkpoints intact. The comparison replaces its report under `artifacts/benchmarks/perf-comparison/`; copy prior reports before rerunning if needed. The scenario benchmark writes under `artifacts/benchmarks/perf-trial/`. Its `--profile` option exports a CUDA trace; the trace's kernel busy fraction is a trace-window measure, distinct from NVIDIA's device-utilization samples.

## Verification and saved evidence

Preparation checks matched legacy sample tensors and seeded batches exactly, including user 92's missing metadata. Serial and worker-prefetched batches matched in order. Float32/BF16 CUDA recurrence outputs and all six input gradients matched the CPU reference for both one-head and four-head shapes. A synthetic interrupted run resumed at the exact next review, matched uninterrupted master weights, and did not repeat completed work. CPU checks cover both architecture sizes and mixed-precision training with float32 master weights.

The completed-run audit independently counted Parquet reviews, checked every metric range for contiguous coverage, verified finite values and float32 checkpoint tensors, and loaded the default checkpoint into CPU recurrent inference. All six throughput scenarios completed their measured updates with finite final loss and gradient norm.

Generated data, metrics, and checkpoints are Git-ignored and remain local:

| Evidence | Location |
| --- | --- |
| Fresh matched comparison and tuning | `artifacts/benchmarks/perf-comparison/comparison.json` |
| Historical default-model search and final default decision | `artifacts/benchmarks/perf-trial/selected-pipeline.json` |
| Completed-run coverage/checkpoint audit | `artifacts/benchmarks/perf-trial/completed-verification.json` |
| Training progress, configuration transitions, recovered OOM | `artifacts/benchmarks/gpu-trial/summary.json`, `metrics.jsonl`, `console.log` |
| Power and utilization samples | `artifacts/benchmarks/gpu-trial/gpu_power.jsonl`, `power_summary.json` |
| Final default weights and optimizer | `checkpoints/gpu-trial/rwkv_gpu_trial.pth`, `rwkv_gpu_trial_optim.pth` |
| Atomic resumable checkpoint | `checkpoints/gpu-trial/latest.pth` |

## Scope at close

This GPU performance track is complete: CUDA training works, the default model's 100-user histories are covered, the batches were tuned, and model throughput was compared. No training or benchmarking job remains running.

Further work would be a separate track: reduce CPU preparation and per-user startup overhead, train Expertium over all 100 users, or compare accuracy with matched training and disjoint evaluation. The earlier [20-user CPU comparison](COMPARISON.md) remains a small temporal-validation experiment, separate from these GPU throughput measurements.

## Full multi-epoch runs (8k/2k)

`scripts/run_full_training.py` runs the accepted R01 trainer (`scripts/gpu_r01_training.py`, current accepted track) over a fixed user split for several epochs, one GPU process per epoch:

- `prepare --run artifacts/full-run/<name>` draws a user-disjoint split stratified by history-length decile: 8,000 training, 1,500 validation (with a fixed 300-user quick-validation subset) and a sealed 500-user final test that contains the preserved 100-user final-test cohort. It also builds the upstream benchmark label database (`rwkv/find_equalize_test_reviews.py`, FSRS-6 features, short-term reviews, seconds) for the held-out users.
- `run --run <dir>` trains the remaining epochs. Each epoch starts from the previous epoch's master weights and AdamW state (`epoch-NN/<model>/<track>/{master,optimizer}.pth`), reshuffles the training users, and continues the update count, so the 5-update warmup happens once. A failed or stopped epoch is kept as `epoch-NN-attempt-K` and rerun from the previous checkpoint; a rerun reproduces the same weights bitwise.
- Non-final epochs score the quick-validation subset; the final epoch scores the full validation set and the final test. The final test is reported separately and must not drive decisions.
- Metrics: the R01 curve/instant held-out losses plus benchmark LogLoss, RMSE(bins) and AUC on the benchmark test reviews, computed with upstream `rwkv/get_result.get_stats` (`scripts/full_run_metrics.py`), review-weighted and user-averaged.
- `pages --run <dir> --push-every 1800` writes `pages/data/train-8k.json` for `pages/train.html` and commits and pushes it every 30 minutes.

These hooks are opt-in through the per-epoch protocol's `full_run` block; R01 iteration protocols do not set it, so their training and evaluation are unchanged.
