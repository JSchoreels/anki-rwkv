default:
    just --list

setup:
    uv sync --locked

check:
    uv run --locked python scripts/check.py

# Exact review containment, uninterrupted matching blocks, and an offline D3 graph.
review-relations *args:
    uv run --no-sync python scripts/review_relations.py {{args}}

check-review-relations:
    uv run --no-sync python scripts/check_review_relations.py

benchmark-review-relations *args:
    uv run --no-sync python scripts/benchmark_review_relations.py {{args}}

publish-review-relations *args:
    uv run --no-sync python scripts/publish_review_relations.py {{args}}

build-cuda:
    uv pip install "setuptools>=68" wheel
    RWKV_BUILD_CUDA=1 uv pip install --no-build-isolation --no-deps -e .

labels:
    mkdir -p artifacts
    uv run --locked python -m rwkv.find_equalize_test_reviews --config configs/labels.toml

preprocess-train:
    uv run --locked python -m rwkv.data_processing --config configs/preprocess-train.toml

preprocess-test:
    uv run --locked python -m rwkv.data_processing --config configs/preprocess-test.toml

train-ws:
    uv run --locked python -m rwkv.train_rwkv --config configs/train-ws.toml

train-decay:
    uv run --locked python -m rwkv.train_rwkv --config configs/train-decay.toml

evaluate:
    uv run --locked python -m rwkv.get_result --config configs/evaluate.toml

infer:
    uv run --locked python -m rwkv.run_as_rnn --config configs/inference.toml

# Float32 CPU trial using the upstream reference recurrence and loss.
train-cpu-trial:
    uv run --locked python scripts/train_cpu_trial.py --config configs/train-cpu-trial.toml

# Matched baseline/Expertium trials and temporal validation.
compare-cpu:
    uv run --locked python scripts/compare.py --config configs/compare-cpu.toml

# Run setup and build-cuda first; avoid re-syncing away the native extension.
check-cuda:
    uv run --no-sync python scripts/check_cuda.py

compare-cuda:
    uv run --no-sync python scripts/compare.py --config configs/compare-cuda.toml

# Exact preparation parity, ordered prefetch, BF16 kernels, and interrupted resume.
check-gpu-perf:
    uv run --locked --extra nvrtc python scripts/check_gpu_perf.py

# End-to-end throughput on a common real-history window; keeps training checkpoints intact.
benchmark-gpu:
    uv run --locked --extra nvrtc python scripts/benchmark_gpu_trial.py

# Identical-input throughput comparison, followed by independent Expertium tuning.
compare-gpu-throughput:
    uv run --locked --extra nvrtc python scripts/compare_gpu_throughput.py

# Fixed-batch preparation, recurrence tiles, launch/copy fusion, graphs, and AdamW.
benchmark-gpu-optimizations:
    uv run --locked --extra nvrtc python scripts/run_gpu_optimizations.py

check-gpu-optimizations:
    uv run --locked --extra nvrtc python scripts/check_gpu_optimizations.py

# Paired curve/instant log-loss screen on a chronological user-17 holdout.
check-gpu-logloss:
    uv run --locked --extra nvrtc python scripts/check_gpu_logloss.py --in-process

# Refresh paired held-out log loss, measured seconds and reviews/s; retain pending ideas.
report-gpu-tradeoffs:
    uv run --no-sync python scripts/report_training_tradeoffs.py

# Full-history GPU trial with checkpoint resume support.
train-gpu-trial:
    uv run --locked --extra nvrtc python scripts/train_gpu_trial.py --config configs/train-gpu-trial.toml
