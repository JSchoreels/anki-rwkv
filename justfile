default:
    just --list

setup:
    uv sync --locked

check:
    uv run --locked python scripts/check.py

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
