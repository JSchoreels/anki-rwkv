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
    for path in [*ROOT.glob("*.py"), *ROOT.glob("rwkv/**/*.py"), *ROOT.glob("features/**/*.py"), *ROOT.glob("scripts/*.py")]:
        ast.parse(path.read_text(), filename=str(path))
    for path in ROOT.glob("configs/*.toml"):
        tomllib.loads(path.read_text())
    provenance = json.loads((ROOT / "upstream.json").read_text())
    for name, expected in provenance["files"].items():
        if name not in provenance["adapted"]:
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
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
    print(f"Source, TOML, provenance, imports, AdamW setup, and 3 CPU recurrent steps passed ({parameters:,} parameters).")
    print(f"CUDA available: {torch.cuda.is_available()}; CPU reference training is available; accelerated training requires the CUDA extension.")


if __name__ == "__main__":
    main()
