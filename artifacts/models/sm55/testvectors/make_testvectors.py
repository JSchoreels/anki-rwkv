"""Provenance only: how the golden test vectors were made. Needs the training repo (torch, pandas, the
anki-revlogs-10k data); a developer who only uses the package does not run this.

    cd <repo worktree that exported the package>      # e.g. <workspace>/wt/small-model
    CUDA_VISIBLE_DEVICES= python -X utf8 <this file> <package dir> <out dir> 4676 1745

Per user it builds the full-history sample exactly like the held-out scorer (scripts/compress_common.full_history_batch,
ID-encoding seed 2025 + user), saves every model input per row (92 features, entity ids, skip flag, query time) and
the outputs of the parallel NumPy reference for fp32 and int8 weights, after checking them against PyTorch (<= 1e-5).
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("RWKV_NO_JIT", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, os.getcwd())
package, out_dir, users = Path(sys.argv[1]), Path(sys.argv[2]), [int(u) for u in sys.argv[3:]]
sys.path.insert(0, str(package / "reference"))

import torch  # noqa: E402
from scripts import compress_common as cc  # noqa: E402
from scripts import compress_quant as cq  # noqa: E402
from scripts.small_model_export import build_torch, load_checkpoint  # noqa: E402
from rwkv.data_processing import add_queries, add_segment_features, get_rwkv_data  # noqa: E402
from rwkv.prepare_batch import prepare  # noqa: E402
from small_model_numpy import NumpyModel  # noqa: E402
from weights_io import read_weights  # noqa: E402

torch.set_num_threads(2)
spec = json.loads((package / "architecture.json").read_text())
checkpoint = load_checkpoint(package / "master.pth")
variants = {"fp32": (checkpoint, package / "weights.f32.bin"),
            "int8": (cq.quantize_state_dict(checkpoint, cq.linear_weight_names(checkpoint), 8, None),
                     package / "weights.int8.bin")}
SWEEP = np.array([60.0, 3600.0, 86400.0, 7 * 86400.0, 30 * 86400.0, 365 * 86400.0])
STREAMS = ["card_id", "deck_id", "note_id", "preset_id", "user_id"]
out_dir.mkdir(parents=True, exist_ok=True)


def arrays(batch):
    to = lambda t: t.cpu().numpy()  # noqa: E731
    return {"start": to(batch.start), "sub_gather": [[to(g) for g in s] for s in batch.sub_gather],
            "sub_gather_lens": [list(s) for s in batch.sub_gather_lens],
            "time_shift_selects": [[to(g) for g in s] for s in batch.time_shift_selects],
            "skips": [[to(g) for g in s] for s in batch.skips], "num_data": int(batch.num_data),
            "labels": to(batch.labels)}


data = cc.data_path()
for user in users:
    frame = get_rwkv_data(data, user)
    sample = cc.make_samples(frame, user, len(frame))
    assert len(sample) == 1
    batch = prepare(sample, seed=2025 + user)
    # the same row table create_sample builds: segment features, then the query rows
    rows = add_queries(add_segment_features(frame.copy()).reset_index(drop=True), [])
    T = len(rows)
    a = arrays(batch)
    assert a["start"].shape == (T, 92) and a["labels"].shape == (1, T, 7)
    labels = a["labels"][0]
    out = {
        "user": np.int64(user), "n_rows": np.int64(T),
        "features": a["start"].astype(np.float32),
        "skip": rows["skip"].to_numpy().astype(np.uint8),
        "is_query": rows["is_query"].to_numpy().astype(np.uint8),
        "has_label": labels[:, 4].astype(np.uint8),
        "label_elapsed_seconds": labels[:, 0].astype(np.float32),
        "label_rating": labels[:, 3].astype(np.int8),
        "review_th": rows["review_th"].to_numpy().astype(np.int64),
        "sweep_seconds": SWEEP.astype(np.float32),
    }
    for name in STREAMS:
        out["id_" + name] = rows[name].to_numpy().astype(np.int64)
    # raw columns the 92 features are computed from (zeroed for query rows, as in the model input)
    for name in ("elapsed_seconds", "elapsed_days", "day_offset", "day_offset_first", "rating", "duration", "state"):
        out["raw_" + name] = rows[name].to_numpy().astype(np.float64)
    # parallel-form batch arrays for small_model_numpy.NumpyModel
    out["par_start"] = a["start"]
    out["par_num_streams"] = np.int64(len(a["sub_gather"]))
    for i, splits in enumerate(a["sub_gather"]):
        out[f"par_lens_{i}"] = np.array(a["sub_gather_lens"][i], dtype=np.int64)
        for j in range(len(splits)):
            out[f"par_gather_{i}_{j}"] = splits[j].astype(np.int64)
            out[f"par_select_{i}_{j}"] = a["time_shift_selects"][i][j].astype(np.int64)
            out[f"par_skip_{i}_{j}"] = a["skips"][i][j].astype(np.uint8)
    out["par_labels"] = a["labels"]

    for name, (torch_state, path) in variants.items():
        ref = NumpyModel(read_weights(path), spec)
        res = ref.predict(**arrays(batch))
        model = build_torch("w64", torch_state)
        with torch.inference_mode():
            stats = model.get_loss(batch)
        has = stats.has_label.bool().numpy()
        for key, theirs in (("p_curve", stats.p_curve), ("p_imm", stats.p_imm), ("p_rating", stats.p_imm_all)):
            diff = np.abs(res[key].astype(np.float64) - theirs.float().numpy().astype(np.float64))
            print(f"user {user} {name} {key}: max |numpy - torch| {diff[has].max():.2e} (labelled), {diff.max():.2e} (all)")
            assert diff[has].max() < 1e-5, key
        out[f"{name}_p_curve"] = res["p_curve"][0].astype(np.float32)
        out[f"{name}_p_imm"] = res["p_imm"][0].astype(np.float32)
        out[f"{name}_p_rating"] = res["p_rating"][0].astype(np.float32)
        sweep = []
        for t in SWEEP:
            lab = a["labels"].copy()
            lab[..., 0] = t
            sweep.append(ref.predict(**{**arrays(batch), "labels": lab})["p_curve"][0])
        out[f"{name}_p_curve_sweep"] = np.stack(sweep, axis=1).astype(np.float32)
    path = out_dir / f"user_{user}.npz"
    np.savez_compressed(path, **out)
    print(f"[testvectors] {path}: {T} rows, {path.stat().st_size:,} bytes", flush=True)
