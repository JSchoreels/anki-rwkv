# anki-rwkv

Training and compression research for the RWKV spaced-repetition model used in Anki. This repository contains the RWKV model, preprocessing, CUDA training kernel, trainer, evaluation, and recurrent inference code from [srs-benchmark](https://github.com/open-spaced-repetition/srs-benchmark), with the small supporting subset needed to run that pipeline independently.

The starting point is the original 2,762,884-parameter architecture. Expertium's smaller architectures and compressed states are documented below as research references; they have not been integrated into this baseline.

## Sources and provenance

The imported baseline is pinned to srs-benchmark commit [`3effecc37c9584e8323ee9c09c9476df84d87589`](https://github.com/open-spaced-repetition/srs-benchmark/tree/3effecc37c9584e8323ee9c09c9476df84d87589). This revision precedes upstream's runtime tensor-shape annotation changes and uses conventional PyTorch imports. [`upstream.json`](upstream.json) records the source commit, original file hashes, and adapted helpers. The RWKV algorithms and CUDA sources are unchanged.

Local adaptations are limited to FSRS feature selection for benchmark-label alignment, extraction of three required utility functions, packaging, verification, and project-specific configuration files. The broader FSRS/LSTM/DASH benchmark suite is excluded.

Upstream code licensing and checkpoint redistribution terms are discussed in [srs-benchmark issue #353](https://github.com/open-spaced-repetition/srs-benchmark/issues/353). Imported code retains upstream notices and is not assigned a new license here. Dataset terms are available on the dataset's own Hugging Face page.

## How RWKV is trained

RWKV learns shared weights from many users' review histories. The published benchmark describes training on approximately 5,000 users, evaluating on the other half, then swapping halves. The published checkpoint ranges are 5000–10000 and 101–4999, with users 1–100 reserved for validation. The first range is inclusive and contains 5,001 users. Normal Anki use replays a collection's history to update recurrent state; it does not optimize weights separately for each user.

The baseline has five RWKV-7 stacks for card, deck, note, preset, and user scopes, each using a model width of 128 and four heads. Inputs include intervals, grades, review duration, study workload, sibling relationships, hierarchy, and calendar context. It uses review metadata, rather than card text, images, or audio. See [`rwkv/architecture.py`](rwkv/architecture.py), [`rwkv/data_processing.py`](rwkv/data_processing.py), and the [benchmark feature description](https://github.com/open-spaced-repetition/srs-benchmark#features-note).

Curve and Instant are trained jointly in one network:

- **Instant** predicts the four answer ratings with cross-entropy. Recall probability is `1 - P(Again)`.
- **Curve** predicts recall at the actual subsequent review delay with binary cross-entropy. It combines 128 exponential basis curves with a learned correction. The objective trains both the underlying curve and its corrected prediction, plus regularization terms.

The trainer uses backpropagation and AdamW, gradient clipping at 0.5, bfloat16 computation with float32 master weights, and a warmup–stable–decay learning-rate schedule. The documented upstream run used approximately 10 warmup/stable epochs and two decay epochs, with a peak learning rate of `7e-4` and 20,000 warmup steps. See [`rwkv/train_rwkv.py`](rwkv/train_rwkv.py), [`rwkv/model/srs_model.py`](rwkv/model/srs_model.py), and the [upstream training guide](https://github.com/open-spaced-repetition/srs-benchmark/blob/main/rwkv/README.md).

These settings are a documented starting point, not a guarantee of reproducing the published scores. The upstream sample's million-epoch setting is replaced by explicit 10-epoch and two-epoch project configurations. W&B logging defaults to off.

## Dataset

Use [open-spaced-repetition/anki-revlogs-10k on Hugging Face](https://huggingface.co/datasets/open-spaced-repetition/anki-revlogs-10k). The benchmark describes approximately 727 million reviews from 10,000 Anki users.

The dataset is referenced here only. Obtain it separately and place its directory layout under `data/anki-revlogs-10k/`, or change `DATA_PATH` in the preprocessing, label, and inference configurations. Preserve its partitioned `revlogs/user_id=...` directories and accompanying tables. Data, LMDB databases, checkpoints, and generated evaluation results are Git-ignored.

Full preprocessing needs roughly 400 GB of storage, plus the dataset and checkpoints. Begin with a smaller user range if needed; update label, preprocessing, training, validation, and evaluation ranges consistently, and reduce the LMDB map sizes. Upstream training needs at least five batch groups because of its prefetch queue.

## Setup and verification

Use Python 3.12, [uv](https://docs.astral.sh/uv/), and [just](https://github.com/casey/just). Training and parallel evaluation require an NVIDIA CUDA GPU and a compatible installed CUDA toolkit. The dependency configuration selects PyTorch 2.9.1 with CUDA 12.6 on Linux/Windows. macOS can run installation checks and recurrent CPU inference; it cannot run this CUDA trainer.

```sh
git clone https://github.com/JSchoreels/anki-rwkv.git
cd anki-rwkv
just setup
just check
```

`just check` parses source and TOML, checks imported-file hashes, imports the pipeline, constructs the baseline model and AdamW optimizer, and exercises three recurrent CPU inference steps. It uses synthetic features and needs no dataset or pretrained checkpoint. It does not establish training accuracy or parity with the benchmark.

On the CUDA training machine, compile the native extension after installing the CUDA toolkit:

```sh
just build-cuda
```

The extension is built only when requested, so normal installation does not require a CUDA compiler. PyTorch, the CUDA toolkit, and the host C++ compiler must be compatible. The inherited CUDA kernel requires a head dimension of 32.

## Preprocess, train, and evaluate

Run commands from this repository's root. First obtain the dataset separately and adjust configuration paths and resource settings for the training machine.

```sh
just labels
just preprocess-train
just preprocess-test
just train-ws
```

The label database aligns evaluation reviews and RMSE bins with the benchmark. Training data is split into bounded chunks; test preprocessing retains longer histories. These create `artifacts/label_filter_db`, `artifacts/train_db`, and `artifacts/test_db`.

The warmup/stable trainer writes model and optimizer checkpoints to `checkpoints/ws/`. Before decay, copy the final matching pair to the names expected by `configs/train-decay.toml`. For example, replace `12345` below with the final step number printed by your run:

```sh
mkdir -p checkpoints/resume
cp checkpoints/ws/rwkv_ws_12345.pth checkpoints/resume/warmup.pth
cp checkpoints/ws/rwkv_ws_optim_12345.pth checkpoints/resume/warmup_optim.pth
just train-decay
```

The trainer saves model and optimizer filenames in different orders, so both copies matter. Decay resumes weights and optimizer state and starts its own two-epoch learning-rate schedule with `STEP_OFFSET = 1`.

Set `MODEL_PATH` in `configs/evaluate.toml` to the final decay checkpoint, then run:

```sh
just evaluate
```

Evaluation reports Log Loss, AUC, and RMSE (bins) for both prediction heads. Lower Log Loss and RMSE are better; higher AUC is better. For the second training half, use training users 101–4999 and evaluation users 5000–10000, retaining validation users 1–100. Keep train and evaluation users disjoint.

For a single-user recurrent run, set `MODEL_PATH` and `USER` in `configs/inference.toml`, then run `just infer`. This uses CPU float32 and requires the dataset and label database. The inherited runner contains an ID/time-encoding TODO, so this example is not a deployment-parity guarantee.

## Expertium's model and state compression work

Two repositories cover distinct parts of the memory problem:

| Repository | Focus | Useful reading |
| --- | --- | --- |
| [rwkv-anki-autoresearch](https://github.com/Expertium/rwkv-anki-autoresearch) | Smaller models, training experiments, Rust/CPU inference, and quantization | [Experiment history](https://github.com/Expertium/rwkv-anki-autoresearch/blob/main/optimization/HISTORY.md), [research record](https://github.com/Expertium/rwkv-anki-autoresearch/blob/main/optimization/research_5k_verbose.md) |
| [rwkv-state-quant](https://github.com/Expertium/rwkv-state-quant) | Compressing the recurrent state retained for cards and notes | [How state compression works](https://github.com/Expertium/rwkv-state-quant/blob/main/how_state_compression_works.md), [research log](https://github.com/Expertium/rwkv-state-quant/blob/main/research_log.md) |

The autoresearch history records an architecture milestone of **2,762,884 → 192,800 parameters**, approximately 14.3× fewer. Changes include reducing width, shrinking prediction heads, and moving capacity from expensive per-card modules into shared deck modules. This is a historical milestone, not a claim about the latest checkpoint or identical evaluation conditions across experiments.

The state-compression explanation reports **2.25 KiB → 9 bytes per card**, a 256× reduction for its smaller architecture, with reported Log Loss increases of **+0.0011 Instant / +0.0002 Curve**. Its second-seed test reports +0.0011 / +0.0004. These compare that experiment's compressed and uncompressed states; they do not measure this repository's baseline or the current Anki fork.

The methods include low-rank state approximations, scalar quantization, shared codebooks for vector/product quantization, and quantization-aware training. Low-rank storage retains the most useful matrix components; codebooks replace recurring vector patterns with compact indices; quantization-aware training teaches the model to tolerate repeated compression during state updates. The full explanation documents the additional encoding refinements that reach nine bytes.

Model-weight size and recurrent-state size are separate measurements. For large collections, the memory accumulated per card and note can dominate. Compression experiments should report both, along with prediction accuracy, inference speed, dataset split, seed, and compression applied after each recurrent update.

Further background: [RWKV-Lite](https://arxiv.org/abs/2412.10856) and [RWKVQuant](https://arxiv.org/abs/2505.03803). These papers concern other RWKV model sizes and tasks; their results are not measurements of this Anki model.

## Repository layout

```text
rwkv/           Upstream model, training, preprocessing, evaluation, and CUDA kernel
features/       Minimal FSRS features needed to align benchmark labels
configs/        Dataset paths, preprocessing, two training phases, and evaluation
scripts/check.py
                Dataset-free import, provenance, optimizer, and recurrent smoke check
upstream.json   Exact imported revision and source hashes
pyproject.toml  Training dependencies and optional native build metadata
uv.lock         Resolved dependency versions
justfile        Setup, verification, build, preprocessing, training, and evaluation
```
