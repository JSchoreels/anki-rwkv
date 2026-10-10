"""Check source/config validity and exercise recurrent inference without a dataset."""

import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch

    torch.set_num_threads(1)
    print("Checking source, configs, provenance, and pipeline imports...", flush=True)
    for path in [*ROOT.glob("*.py"), *ROOT.glob("rwkv/**/*.py"), *ROOT.glob("features/**/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("expertium/**/*.py")]:
        ast.parse(path.read_text(), filename=str(path))
    for path in ROOT.glob("configs/*.toml"):
        tomllib.loads(path.read_text())
    provenance = json.loads((ROOT / "upstream.json").read_text())
    for name, expected in provenance["files"].items():
        if name not in provenance["adapted"]:
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    smaller_provenance = json.loads((ROOT / "expertium/upstream.json").read_text())
    for name, record in smaller_provenance["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == record["adapted_sha256"], name
    for name in ["rwkv.train_rwkv", "rwkv.data_processing", "rwkv.get_result", "rwkv.run_as_rnn", "features"]:
        importlib.import_module(name)

    from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
    from rwkv.model.srs_model import SrsRWKV
    from rwkv.model.srs_model_rnn import SrsRWKVRnn
    from rwkv.train_rwkv import get_optimizer
    from argparse import Namespace

    print("Constructing the baseline model and exercising CPU recurrent inference...", flush=True)
    parallel = SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG)
    recurrent = SrsRWKVRnn(DEFAULT_ANKI_RWKV_CONFIG).eval()
    recurrent.load_state_dict(parallel.state_dict())
    parameters = sum(p.numel() for p in parallel.parameters())
    assert parameters == 2_762_884, parameters
    optimizer = get_optimizer(Namespace(PEAK_LR=7e-4), parallel)
    assert len(optimizer.param_groups) == 5
    states = [None] * 5
    with torch.inference_mode():
        for _ in range(3):
            ahead, weights, ratings, *states = recurrent.review(torch.zeros(1, 92), *states)
            assert torch.isfinite(ahead).all()
            assert torch.isfinite(ratings).all()
            assert torch.allclose(weights.sum(dim=-1), torch.ones(1))
            recall = 1 - ratings.softmax(dim=-1)[:, 0]
            curve = recurrent.forgetting_curve(weights, torch.tensor([[86400.0]]))
            assert ((recall > 0) & (recall < 1)).all()
            assert ((curve > 0) & (curve < 1)).all()
    from expertium.architecture import DEFAULT_ANKI_RWKV_CONFIG as small_config
    from expertium.model.srs_model import SrsRWKV as SmallModel
    from expertium.model.srs_model_rnn import SrsRWKVRnn as SmallRNN

    small_model = SmallModel(small_config)
    assert sum(p.numel() for p in small_model.parameters()) == 192_800
    small_recurrent = SmallRNN(small_config).eval()
    small_recurrent.load_state_dict(small_model.state_dict(), strict=True)
    states = [None] * 5
    with torch.inference_mode():
        for _ in range(3):
            ahead, weights, ratings, *states = small_recurrent.review(torch.zeros(1, 92), *states)
            assert torch.isfinite(ahead).all() and torch.isfinite(ratings).all()
            assert torch.allclose(weights.sum(dim=-1), torch.ones(1))
    from scripts.check_cuda import check_training
    from rwkv.data_processing import read_user_metadata
    import pandas as pd
    from tempfile import TemporaryDirectory

    print("Checking metadata partitions and shared float32/BF16 training paths...", flush=True)
    with TemporaryDirectory() as temporary:
        table = Path(temporary) / "cards"
        for user, card in [(1, 123), (2, 456)]:
            partition = table / f"user_id={user}"
            partition.mkdir(parents=True)
            pd.DataFrame({"card_id": [card], "note_id": [card + 1]}).to_parquet(partition / "data.parquet", index=False)
        for user in [1, 2, 3]:
            expected = pd.read_parquet(table, filters=[("user_id", "=", user)]).drop(columns="user_id")
            actual = read_user_metadata(table, user).drop(columns="user_id")
            pd.testing.assert_frame_equal(actual, expected)
        from scripts.compare import cache_key
        config = tomllib.loads((ROOT / "configs/compare-cuda.toml").read_text())
        config.update(DATA_PATH=temporary, USER_START=1, USER_END=2)
        key = cache_key(config)
        assert cache_key({**config, "SEED": config["SEED"] + 1}) != key
        assert cache_key({**config, "BATCH_CHUNKS": 1}) != key
    for dtype in [torch.float32, torch.bfloat16]:
        check_training(torch.device("cpu"), dtype)
    print("Expertium provenance, 192,800 parameters, strict recurrent load, and 3 CPU steps passed.")
    print(f"Source, TOML, provenance, imports, AdamW setup, and 3 CPU recurrent steps passed ({parameters:,} parameters).")
    print(f"CUDA available: {torch.cuda.is_available()}; CPU reference training is available; accelerated training requires the CUDA extension.")


if __name__ == "__main__":
    main()
