"""End-to-end vectors for the low-rank card / note state (SM151) on the Big models, from benchmark users.

Adapted from the fork's `qt/tools/rwkv_compact_state_big_vectors.py`: the same patched fork model (card and note state
rounded after every answer, the first row of a stream keeps its own shift input, as the fork does), but the inputs are
real benchmark users (show-time inputs `e2s-query`, deleted cards `neighbour`, built exactly as the scorer's
`LeakBatches` / `sm31_inputs.samples` with seed 2025 + user) and the formats include the low-rank int8 states
experiment (SM148) codec on the fork-model path (`state_lowrank.install_fork` + `set_fork`):

- `fp32`: float32 state;
- `compact_int8`: `compact_cards_int8` (`w8+s8hx8` state formats; the fork's Big weights are float32);
- `t24`: `w8+lr2.16.4q8x8` (rank 2 below 16 answers, rank 4 from 16, int8 factors, compact int8 shifts);
- `q2`: `w8+lr2q8x8` (rank 2 always).

Per row: features, ids, skip, elapsed, label (label_y, has_label, is_query) and, per format, [p_imm, raw curve at the
row's elapsed time (the fork's curve), curve with the ahead residual]. Outputs stay under `wt/sm151/work/` (never
committed: they hold benchmark rows); only aggregates go into RESULT.md.

Run from the training repository worktree that has the SM148 codec (`wt/sm148`), on the CPU:

    PYTORCH_JIT=0 RWKV_NO_JIT=1 CUDA_VISIBLE_DEVICES= python -X utf8 <this file> <tag> <weights.pth> <out.json> USER...
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ["PYTORCH_JIT"] = "0"
os.environ.setdefault("RWKV_NO_JIT", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.pop("RWKV_DELETED_CARDS", None)   # the transform is applied explicitly below, as the scorer does
sys.path.insert(0, os.getcwd())

from scripts import compress_quant as cq  # noqa: E402

_STORED_ONLY = """

def _round_shift_stored(x: torch.Tensor, select: torch.Tensor, bits: int) -> torch.Tensor:
    rounded = _round_shift(x, bits)
    own = select == torch.arange(select.shape[1], device=select.device).unsqueeze(0)
    return torch.where(own.unsqueeze(-1), x, rounded)
"""


def install_state_quant_model():
    import importlib.util
    import tempfile

    import rwkv.model  # noqa: F401

    name = "rwkv.model.rwkv_model"
    assert name not in sys.modules, "rwkv.model.rwkv_model is already imported"
    path = cq.ROOT / "rwkv" / "model" / "rwkv_model.py"
    source = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    for old, new in cq._PATCHES:
        assert source.count(old) == 1, old[:60]
        if "def _quantize_rows" in new:
            new += _STORED_ONLY
        new = new.replace(
            "_round_shift(x_shift_BTC, self.shift_bits)",
            "_round_shift_stored(x_shift_BTC, time_shift_select_BT, self.shift_bits)",
        )
        source = source.replace(old, new)
    assert source.count("_round_shift_stored(") == 3, "patch anchors changed"
    source += "\nSTATE_QUANT_PATCHED = True\n"
    generated = Path(tempfile.mkdtemp()) / "rwkv_model_state_quant_stored_shifts.py"
    generated.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, generated)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


from scripts import compress_wkv_cuda  # noqa: E402

compress_wkv_cuda.define()
install_state_quant_model()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rwkv import data_processing as dp  # noqa: E402
from rwkv.data_processing import add_segment_features  # noqa: E402
from rwkv.prepare_batch import prepare  # noqa: E402
from scripts import compress_common as cc  # noqa: E402
from scripts import sm31_inputs, sm71_leak, state_lowrank  # noqa: E402
from scripts.small_model_wkv import F16_SCALE  # noqa: E402

COMPACT8 = 8 | F16_SCALE


def codec(spec):
    c = state_lowrank.parse_sm148(spec)
    c.rows_bits = c.shift_bits = COMPACT8
    return c


FORMATS = [("fp32", None), ("compact_int8", None), ("t24", "lr2.16.4q8x8"), ("q2", "lr2q8x8")]


def main() -> None:
    tag, weights, out_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    users = [int(user) for user in sys.argv[4:]]
    torch.set_num_threads(4)
    model = cq.load_model(state={k: v.float() for k, v in
                                 torch.load(weights, map_location="cpu", weights_only=True).items()})
    for module in model.rwkv_modules:
        for block in module.blocks:
            block.time_mixer.use_reference_wkv = True
    state_lowrank.install_fork(model)
    data = cc.data_path()
    variant = "e2s-query"
    out = []
    for user in users:
        started = time.perf_counter()
        frame = sm31_inputs.load_frame(data, user, variant)
        frame, _ = sm71_leak.neutralise(frame, "neighbour", data, user)
        frame = sm31_inputs.apply_frame(frame, variant)
        with sm31_inputs.query_patch(variant):
            sample = cc.make_samples(frame, user, len(frame))
            assert len(sample) == 1
            rows = dp.add_queries(add_segment_features(frame.copy()).reset_index(drop=True), [])
        batch = prepare(sample, seed=2025 + user)
        features = batch.start.cpu().numpy()
        labels = batch.labels.cpu().numpy()[0]
        T = len(rows)
        assert features.shape == (T, 92), features.shape
        values = {}
        for label, spec in FORMATS:
            if spec is None:
                state_lowrank.set_fork(None)
                bits = (0, 0) if label == "fp32" else (COMPACT8, COMPACT8)
                cq.set_state_streams(model, bits, bits)
            else:
                state_lowrank.set_fork(codec(spec))
                cq.set_state_streams(model, (COMPACT8, COMPACT8), (COMPACT8, COMPACT8))
            t0 = time.perf_counter()
            with torch.inference_mode():
                stats = model.get_loss(batch)
                elapsed = batch.labels[0, :, 0].float()
                raw = model.forgetting_curve(stats.w, elapsed.unsqueeze(-1).unsqueeze(0)).float().numpy().reshape(-1)
            p_imm = stats.p_imm.float().numpy().reshape(-1)
            p_curve = stats.p_curve.float().numpy().reshape(-1)
            values[label] = np.stack([p_imm, raw, p_curve], 1)
            print(f"  user {user} {label}: {time.perf_counter() - t0:.1f} s, ahead {stats.ahead_n}, "
                  f"curve {stats.ahead_avg.item():.6f} raw {stats.ahead_raw_avg.item():.6f}", flush=True)
        state_lowrank.set_fork(None)
        cq.set_state_streams(model, (0, 0), (0, 0))
        out_rows = []
        for i in range(T):
            out_rows.append({
                "features": features[i].astype(np.float32).tolist(),
                "ids": [int(rows[s].iloc[i]) for s in ("card_id", "deck_id", "note_id", "preset_id", "user_id")],
                "skip": bool(rows["skip"].iloc[i]),
                "elapsed": float(labels[i, 0]),
                "label": [float(labels[i, 2]), float(labels[i, 4]), float(labels[i, 6])],
                **{label: [float(x) for x in values[label][i]] for label, _ in FORMATS},
            })
        print(f"user {user}: {T} rows in {time.perf_counter() - started:.0f} s", flush=True)
        out.append({"name": f"{tag}_user_{user}", "user": user, "rows": out_rows})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out_path}: {sum(len(u['rows']) for u in out)} rows", flush=True)


if __name__ == "__main__":
    main()
