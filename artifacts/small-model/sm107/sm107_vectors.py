"""The Big size on the show-time big model (SM107): big-model (2.76M, fork architecture) test vectors from the
training code, in the fork's vector JSON format (the legacy-vectors.json.gz layout of rslib/src/rwkv/legacy_tests.rs):
per user the 92 model inputs per row, entity ids, skip flag, label elapsed time, each row's own interval inputs
(elapsed seconds, elapsed days) and the PyTorch outputs (p_imm, 4 button probabilities, p_curve at the row's elapsed
time, p_curve at the sweep times). The curve is the raw forgetting curve: the fork ignores the big model's "ahead"
logit residual. Same method as the big model (1 leak-free epoch) vectors of 2026-11-10, plus --inputs e2s-query.

inputs normal: the normal inputs. inputs e2s-query: show-time inputs (scripts/sm31_inputs.py), the inputs the
show-time big model (SM91s2) was trained on: every instant query row and every curve query time uses the interval
from the card's previous answer to the moment the card is shown; update rows keep the answer-to-answer interval.

usage (cwd = an anki-rwkv worktree with scripts/sm31_inputs.py, e.g. wt/sm91s2; RWKV_DELETED_CARDS=neighbour,
CUDA_VISIBLE_DEVICES= for CPU):
    python sm107_vectors.py <state dict .pth> <out.json[.gz]> <limit|0> <normal|e2s-query> user...
"""
import contextlib
import gzip
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("RWKV_NO_JIT", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.getcwd())
from scripts import compress_common as cc  # noqa: E402
from scripts import compress_quant as cq  # noqa: E402
from scripts import sm31_inputs  # noqa: E402
from rwkv import data_processing as dp  # noqa: E402
from rwkv.data_processing import add_segment_features, get_rwkv_data  # noqa: E402
from rwkv.prepare_batch import prepare  # noqa: E402

weights, out_path, limit, inputs = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
users = [int(u) for u in sys.argv[5:]]
assert inputs in ("normal", "e2s-query"), inputs
assert os.environ.get("RWKV_DELETED_CARDS") == "neighbour", "run with RWKV_DELETED_CARDS=neighbour"
torch.set_num_threads(4)
state = torch.load(weights, map_location="cpu", weights_only=True)
model = cq.load_model(state=state)


def raw_curves(batch, times):
    _, out_w, _, _ = model.forward_batch(batch.start, batch.sub_gather, batch.sub_gather_lens,
                                         batch.time_shift_selects, batch.skips, batch.num_data)
    return [model.forgetting_curve(out_w, t.unsqueeze(-1)).float().numpy().reshape(-1) for t in times]


SWEEP = [60.0, 3600.0, 86400.0, 7 * 86400.0, 30 * 86400.0, 365 * 86400.0]
data = cc.data_path()
out = []
for user in users:
    frame = get_rwkv_data(data, user)
    patch = contextlib.nullcontext()
    if inputs == "e2s-query":
        frame = sm31_inputs.apply_frame(frame, "e2s-query")   # curve query times at show time
        patch = sm31_inputs.query_patch("e2s-query")         # instant query rows at show time
    with patch:
        sample = cc.make_samples(frame, user, len(frame))
        assert len(sample) == 1
        batch = prepare(sample, seed=2025 + user)
        rows = dp.add_queries(add_segment_features(frame.copy()).reset_index(drop=True), [])
    features = batch.start.cpu().numpy()
    labels = batch.labels.cpu().numpy()[0]
    T = len(rows)
    assert features.shape == (T, 92), features.shape
    with torch.inference_mode():
        stats = model.get_loss(batch)
        p_imm = stats.p_imm.float().numpy().reshape(-1)
        p_rating = stats.p_imm_all.float().numpy().reshape(T, 4)
        elapsed = batch.labels[0, :, 0].float()
        curves = raw_curves(batch, [elapsed] + [torch.full_like(elapsed, s) for s in SWEEP])
        p_curve = curves[0]
        sweep = np.stack(curves[1:], axis=1)
    count = T if limit == 0 else min(limit, T)
    out_rows = []
    for i in range(count):
        values = [float(p_imm[i]), *map(float, p_rating[i]), float(p_curve[i]), *map(float, sweep[i])]
        out_rows.append({
            "features": features[i].astype(np.float32).tolist(),
            "ids": [int(rows[s].iloc[i]) for s in ("card_id", "deck_id", "note_id", "preset_id", "user_id")],
            "skip": bool(rows["skip"].iloc[i]),
            "elapsed": float(labels[i, 0]),
            "interval": [float(rows["elapsed_seconds"].iloc[i]), float(rows["elapsed_days"].iloc[i])],
            "fp32": values,
            "int8": [],
        })
    print(f"user {user} ({inputs}): {T} rows, {int(rows['skip'].sum())} queries", flush=True)
    out.append({"name": f"user_{user}", "sweep": SWEEP, "rows": out_rows})
blob = json.dumps(out, separators=(",", ":")).encode()
out_path.write_bytes(gzip.compress(blob, mtime=0) if out_path.suffix == ".gz" else blob)
print(f"[sm107-vectors] {out_path}: {sum(len(u['rows']) for u in out)} rows", flush=True)
