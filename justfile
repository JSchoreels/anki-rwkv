default:
    just --list

setup:
    uv sync --locked

check:
    uv run --locked python scripts/check.py

build-cuda:
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
