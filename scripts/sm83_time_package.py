#!/usr/bin/env python
"""Answer-time head release (SM83): package the answer-time head on richer frozen features (SM82c) for the fork.

The head ships as a separate optional file next to the released small model (the leak-free small model, cool-down
from epoch 25 (SM72), package `artifacts/models/sm72e25`), so the model's own weights file stays byte-identical (the
fork's state caches are keyed by its SHA-256) and every grade output is unchanged by construction.

Package folder (default `artifacts/models/sm72e25-time`):
  time_head.f32.bin   ARWKVWEIGHTS1 container (scripts/small_model_export.write_f32): the head's 4 tensors
                      time_mlps.0.hidden.weight [128, 474], .hidden.bias [128], .out.weight [64, 128], .out.bias [64]
                      + time_head.format_version [1] = 1.0.
  time_head.int8.bin  ARWKVWEIGHTS8PC1 copy (2-D weights int8 per output row, the scorer's w8-pc).
  time_head.json      the spec: input layout, explicit features, bins, edges, validity rule, outputs.
  head.pth            the head-only state_dict (time_mlps.0.*), cut from the SM82c master.
  reference/answer_time_numpy.py   NumPy reference: the 26 explicit features, the head, median, P(fast).
  testvectors/user_<u>.npz         the released package's vectors for users 4676 and 1745 (unchanged arrays) + per
                      instant row: time_row, time_review_th, time_explicit [n, 26], fp32_time_frozen [500, 448] (first rows),
                      fp32_time_logits / int8_time_logits [n, 64] (PyTorch, the decision's frozen-feature path), and
                      the user's revlog columns rev_card_id, rev_rating, rev_duration, rev_elapsed_days.
  testvectors/check_time_vectors.py  NumPy checker of the vectors.
  parity.json, manifest.json, ANSWER_TIME.md (manual section).

usage (CPU; PY = the main checkout's venv python, run from this worktree):
  $PY -X utf8 scripts/sm83_time_package.py package [--out DIR]
"""
from argparse import ArgumentParser
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("RWKV_NO_JIT", "1")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RELEASED = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/models/sm72e25")
MASTER = Path("C:/Users/jonat/workspace/wt/sm82c/artifacts/small-model/sm82c-head-released/master.pth")
DECISION = Path("C:/Users/jonat/workspace/wt/sm82c/artifacts/small-model/sm82c/sm82c-decision.json")
USERS = (4676, 1745)
HEAD = "time_mlps.0."
FORMAT_VERSION = 1
FROZEN_ROWS = 500   # the frozen inputs are stored for the first time rows only (the NumPy checker uses them)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def spec():
    from scripts.expertium_options import TIME_BIN_EDGES_MS, TIME_CAP_MS
    from scripts.sm82c_head import EXPLICIT_NAMES, INPUTS, HIDDEN, BINS, SNAPS, D
    return {
        "format": "anki-rwkv answer-time head", "format_version": FORMAT_VERSION,
        "model": "sm72e25 (the leak-free small model, cool-down from epoch 25): the head reads its frozen trunk",
        "weights": {"f32": "time_head.f32.bin", "int8": "time_head.int8.bin",
                    "tensors": {HEAD + "hidden.weight": [HIDDEN, INPUTS], HEAD + "hidden.bias": [HIDDEN],
                                HEAD + "out.weight": [4 * BINS, HIDDEN], HEAD + "out.bias": [4 * BINS],
                                "time_head.format_version": [1]}},
        "head": "logits = out(relu(hidden(inputs))); inputs = 474 float32 values at the query (instant) row",
        "inputs": {
            "frozen": {"count": SNAPS * D + 128, "layout": [
                "layer_norm(x0) (64): the residual after the input encoder (features2card, after its SiLU)",
                "layer_norm(x1) (64): after layer round 0 (steps (0,0) (1,0) (2,0) (3,0) (4,0))",
                "layer_norm(x2) (64): after layer round 1 (steps (0,1) (1,1) (3,1) (4,1))",
                "layer_norm(x3) (64): after layer round 2 (steps (1,2) (3,2) (4,2))",
                "layer_norm(x4) (64): after layer round 3 (step (1,3)) = the final residual before prehead_norm",
                "head_p (128): relu(head_p.0(prehead_norm(x4))), the instant head's hidden layer"],
                "layer_norm": "no affine parameters, eps 1e-5, biased variance (torch.nn.functional.layer_norm)"},
            "explicit": {"count": len(EXPLICIT_NAMES), "names": EXPLICIT_NAMES,
                         "source": "the user's reviews before the query (card id, grade 1-4, duration ms) and the "
                                   "query's own card id and elapsed days; reference: reference/answer_time_numpy.py "
                                   "explicit_features (= scripts/sm82c_head.py explicit_features)",
                         "ln_time": "ln(max(duration_ms, 1) / 1000) in float64; only durations > 0 are recorded",
                         "centre": "ln times are centred on ln 8 s",
                         "median": "exact running median (two heaps; even count = mean of the two middle values) "
                                   "per grade 1-4 of recorded times, fallback the median over all grades, then ln 8 s"},
        },
        "output": {"bins": BINS, "layout": "logits [4 grades (Again, Hard, Good, Easy) x 16 bins], softmax per grade",
                   "bin_edges_ms": [float(e) for e in TIME_BIN_EDGES_MS],
                   "bins_meaning": "bin 0 < 500 ms, bin k = [edge k-1, edge k), bin 15 >= 60 s (the time cap)",
                   "cap_ms": TIME_CAP_MS,
                   "median": "ln seconds, linear in ln time inside the crossing bin (bin 0 from ln 0.25 s, bin 15 = "
                             "60 s); reference median_log_s",
                   "fast_threshold_ms": float(TIME_BIN_EDGES_MS[6]),
                   "fast": "P(answer under 3.891 s | grade) = sum of bins 0..6"},
        "valid": "query (instant) rows of reviews on a later day than the card's previous review (elapsed days >= 1); "
                 "first reviews and same-day reviews are not predicted (their interval input holds their own answer "
                 "time in training)",
    }


def build_vectors(user, master_state, released_dir, out_dir):
    from scripts import small_model_heldout as smh  # noqa: F401  first: installs the state quantizer before the model
    import torch
    from scripts import compress_common as cc
    from scripts import compress_quant as cq
    from scripts.sm82c_head import _features_cpu, explicit_table, revlog, INPUTS, HIDDEN, BINS
    from expertium.model.variant_model import TimeMLP
    data = cc.data_path()
    old = dict(np.load(released_dir / "testvectors" / f"user_{user}.npz", allow_pickle=False))
    rev = revlog(data, user)
    explicit = explicit_table(rev)
    out = dict(old)
    variants = {"fp32": master_state,
                "int8": cq.quantize_state_dict(master_state, cq.linear_weight_names(master_state), 8, None)}
    report = {}
    for name, state in variants.items():
        th, feats, batch, o, qi = _features_cpu(state, user, data)
        start = batch.start.cpu().numpy()
        same_inputs = start.shape == old["features"].shape and np.array_equal(start, old["features"])
        if not same_inputs:
            raise SystemExit(f"user {user}: rebuilt inputs differ from the released vectors")
        head = TimeMLP(INPUTS, HIDDEN, 4 * BINS)
        head.load_state_dict({k[len(HEAD):]: v.float() for k, v in state.items() if k.startswith(HEAD)})
        x = torch.cat([feats.float(), torch.as_tensor(explicit[th.numpy() - 1])], -1)
        with torch.inference_mode():
            logits = head(x).numpy().astype(np.float32)
        rows = qi.numpy().astype(np.int64)
        if not np.array_equal(old["review_th"][rows], th.numpy()):
            raise SystemExit(f"user {user}: instant rows do not map to the vector rows")
        if not (old["is_query"][rows].all() and old["has_label"][rows].all()):
            raise SystemExit(f"user {user}: a time row is not a labelled query row")
        if name == "fp32":
            out["time_row"] = rows
            out["time_review_th"] = th.numpy().astype(np.int64)
            out["time_explicit"] = explicit[th.numpy() - 1].astype(np.float32)
            out["fp32_time_frozen"] = feats.numpy()[:FROZEN_ROWS].astype(np.float32)   # a prefix: keeps the npz small
        out[f"{name}_time_logits"] = logits
        # the released grade outputs: the head path recomputes the stock forward, check it against the vectors
        report[name] = {"rows": int(len(rows)), "inputs_equal_released_vectors": bool(same_inputs)}
    n = int(out["time_review_th"].max())
    for col, src in (("rev_card_id", "card_id"), ("rev_rating", "rating"), ("rev_duration", "duration"),
                     ("rev_elapsed_days", "elapsed_days")):
        out[col] = rev[src].to_numpy()[:n].astype(np.int64)
    path = out_dir / "testvectors" / f"user_{user}.npz"
    np.savez_compressed(path, **out)
    report["file"] = path.name
    report["bytes"] = path.stat().st_size
    print(f"[sm83] vectors {path}: {report}", flush=True)
    return report


def package(args):
    import torch
    from scripts.small_model_export import write_f32, write_int8, read_weights, int8_names
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    (out / "reference").mkdir(parents=True)
    (out / "testvectors").mkdir()
    started = time.time()
    sd = torch.load(MASTER, map_location="cpu", weights_only=True)
    sd = {k: v.float() for k, v in sd.items()}
    released = torch.load(RELEASED / "master.pth", map_location="cpu", weights_only=True)
    if "state_dict" in released and isinstance(released["state_dict"], dict):
        released = released["state_dict"]
    trunk = {k: v for k, v in sd.items() if not k.startswith(HEAD)}
    if trunk.keys() != released.keys() or not all(torch.equal(trunk[k], released[k].float()) for k in trunk):
        raise SystemExit("the head master's trunk is not the released master bit for bit")
    # the released weights file is exactly the trunk: nothing of the grade path changes
    shipped = read_weights(RELEASED / "weights.f32.bin")
    if shipped.keys() != trunk.keys() or not all(np.array_equal(shipped[k], trunk[k].numpy()) for k in trunk):
        raise SystemExit("the released weights.f32.bin is not the head master's trunk")
    head = {k: v for k, v in sd.items() if k.startswith(HEAD)}
    torch.save(head, out / "head.pth")
    tensors = {k: v.numpy() for k, v in head.items()}
    tensors["time_head.format_version"] = np.array([float(FORMAT_VERSION)], np.float32)
    write_f32(out / "time_head.f32.bin", tensors)
    write_int8(out / "time_head.int8.bin", tensors)
    back = read_weights(out / "time_head.f32.bin")
    assert back.keys() == tensors.keys() and all(np.array_equal(back[k], tensors[k]) for k in tensors)
    (out / "time_head.json").write_text(json.dumps(spec(), indent=1) + "\n")
    for f in ("answer_time_numpy.py",):
        shutil.copy(ROOT / "artifacts" / "small-model" / "sm83" / "reference" / f, out / "reference" / f)
    shutil.copy(ROOT / "artifacts" / "small-model" / "sm83" / "reference" / "check_time_vectors.py",
                out / "testvectors" / "check_time_vectors.py")
    vectors = {str(u): build_vectors(u, sd, RELEASED, out) for u in USERS}
    # the NumPy checker on the written package (both weight formats, explicit features, medians)
    import subprocess
    res = subprocess.run([sys.executable, "-X", "utf8", str(out / "testvectors" / "check_time_vectors.py"), str(out)],
                         capture_output=True, text=True)
    print(res.stdout, res.stderr, flush=True)
    if res.returncode != 0:
        raise SystemExit("check_time_vectors failed")
    check = json.loads(res.stdout.strip().splitlines()[-1])
    parity = {"vectors": vectors, "numpy_check": check}
    (out / "parity.json").write_text(json.dumps(parity, indent=1) + "\n")
    files = {}
    for p in sorted(out.rglob("*")):
        if p.is_file():
            files[p.relative_to(out).as_posix()] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    decision = json.loads(DECISION.read_text()) if DECISION.exists() else None
    manifest = {
        "label": "sm72e25-time", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": {"package": "sm72e25", "weights.f32.bin sha256": sha256(RELEASED / "weights.f32.bin"),
                  "master.pth sha256": sha256(RELEASED / "master.pth")},
        "head_master": {"path": "<workspace>/wt/sm82c/artifacts/small-model/sm82c-head-released/master.pth",
                        "sha256": sha256(MASTER)},
        "parameters": {"head": int(sum(v.numel() for v in head.values())),
                       "int8_tensors": int8_names(tensors)},
        "training": "one pass over 80,988,171 labelled rows (1,991 users, first 400,000 reviews each), 9,887 AdamW "
                    "steps of 8,192 rows, lr 0.002 cosine, heads only on the frozen released model",
        "evaluation": (decision or {}).get("summary") if decision else None,
        "parity": parity, "files": files,
        "provenance": {"commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                                cwd=ROOT).stdout.strip()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"[sm83] package {out} in {time.time() - started:.0f} s", flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("package")
    p.add_argument("--out", default=str(ROOT / "artifacts" / "models" / "sm72e25-time"))
    args = parser.parse_args()
    package(args)


if __name__ == "__main__":
    main()
