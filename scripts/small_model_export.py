"""Small-model track: package a trained Expertium checkpoint for use outside PyTorch (docs/SMALL_MODEL.md,
"From checkpoint to Anki").

Files written by `export` (and `package`):
  weights.f32.bin     every state_dict tensor, float32, in the fork's ARWKVWEIGHTS1 container (the exact layout
                      rslib/src/rwkv/mod.rs WeightMap::load parses: magic, u32 count, then per tensor u16 name
                      length, UTF-8 name, u8 rank, u32 dims, little-endian f32 values). Tensor names are our
                      state_dict names (PyTorch layout, Linear weights [out, in]).
  weights.int8.bin    ARWKVWEIGHTS8PC1: same header per tensor plus a u8 kind after the name. kind 0 = f32 values,
                      kind 1 = int8 per-output-channel (rows): out f32 scales, then out x in int8 codes;
                      value = code * scale. Kind 1 is every 2-D `.weight` (scripts/compress_quant.py "w8-pc", the
                      scorer's w8 variant: symmetric abs-max / 127, round-half-even); everything else stays f32.
  architecture.json   the full forward specification (streams, depths, heads, LoRA ranks, stripped mixers, layer
                      schedule, curve head, constants, input layout) that small_model_numpy.py executes.
`verify` runs the PyTorch model and the NumPy reference (scripts/small_model_numpy.py) on the same full-history
batches of held-out users on CPU and reports max |delta p| for the curve, instant and 4-rating outputs.
`manifest` writes manifest.json: sha256, parameter count, architecture, training config (experiment.json /
decay.json, the epoch result.json, phases.jsonl), held-out evaluation summary (artifacts/small-model/heldout.jsonl).
`package` = copy master.pth + export + verify (fp32 and int8) + manifest into a new folder (refuses an existing one).

usage (CPU only; PY = the main checkout's venv python, run from the small-model worktree):
  $PY -X utf8 scripts/small_model_export.py package --checkpoint <master.pth> --arch w64 --label scale-w64kd-dec \
      --out C:/Users/jonat/workspace/anki-rwkv/artifacts/models/scale-w64kd-dec [--users 3 --max-reviews 3000]
  $PY -X utf8 scripts/small_model_export.py export   --checkpoint <master.pth> --arch iter36 --out <dir>
  $PY -X utf8 scripts/small_model_export.py verify   --dir <dir> --checkpoint <master.pth> [--users 3]
  $PY -X utf8 scripts/small_model_export.py manifest --dir <dir> --checkpoint <master.pth> --arch w64 --label <tag>
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path

os.environ.setdefault("RWKV_NO_JIT", "1")  # eager PyTorch modules (CPU reference recurrence)

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

F32_MAGIC = b"ARWKVWEIGHTS1"
I8_MAGIC = b"ARWKVWEIGHTS8PC1"
SPEC_VERSION = 1


# ---------------------------------------------------------------- containers

def _header(f, name):
    raw = name.encode("utf-8")
    f.write(struct.pack("<H", len(raw)))
    f.write(raw)


def write_f32(path, tensors):
    with open(path, "wb") as f:
        f.write(F32_MAGIC)
        f.write(struct.pack("<I", len(tensors)))
        for name, array in tensors.items():
            array = np.ascontiguousarray(array, dtype="<f4")
            _header(f, name)
            f.write(struct.pack("<B", array.ndim))
            f.write(struct.pack(f"<{array.ndim}I", *array.shape))
            f.write(array.tobytes())


def quantize_rows_int8(weight):
    """compress_quant.fake_quant(bits=8, group=None) in float32: returns (codes int8 [out, in], scales f32 [out])."""
    weight = np.asarray(weight, dtype=np.float32)
    scale = (np.abs(weight).max(axis=1) / np.float32(127.0)).astype(np.float32)
    scale = np.where(scale > 0, scale, np.float32(1.0)).astype(np.float32)
    codes = np.clip(np.round(weight / scale[:, None]), -127, 127).astype(np.int8)
    return codes, scale


def int8_names(tensors):
    return [n for n, a in tensors.items() if n.endswith(".weight") and np.asarray(a).ndim == 2]


def write_int8(path, tensors):
    quantized = set(int8_names(tensors))
    with open(path, "wb") as f:
        f.write(I8_MAGIC)
        f.write(struct.pack("<I", len(tensors)))
        for name, array in tensors.items():
            array = np.ascontiguousarray(array, dtype="<f4")
            _header(f, name)
            kind = 1 if name in quantized else 0
            f.write(struct.pack("<BB", kind, array.ndim))
            f.write(struct.pack(f"<{array.ndim}I", *array.shape))
            if kind:
                codes, scale = quantize_rows_int8(array)
                f.write(scale.astype("<f4").tobytes())
                f.write(codes.tobytes())
            else:
                f.write(array.tobytes())


def read_weights(path):
    """Either container -> {name: float32 array} (int8 tensors dequantized as code * scale in float32)."""
    data = Path(path).read_bytes()
    pos = 0

    def take(n):
        nonlocal pos
        out = data[pos:pos + n]
        if len(out) != n:
            raise ValueError(f"{path}: truncated")
        pos += n
        return out

    if data.startswith(I8_MAGIC):
        take(len(I8_MAGIC))
        int8 = True
    elif data.startswith(F32_MAGIC):
        take(len(F32_MAGIC))
        int8 = False
    else:
        raise ValueError(f"{path}: unknown magic")
    count, = struct.unpack("<I", take(4))
    tensors = {}
    for _ in range(count):
        n, = struct.unpack("<H", take(2))
        name = take(n).decode("utf-8")
        kind = struct.unpack("<B", take(1))[0] if int8 else 0
        rank, = struct.unpack("<B", take(1))
        shape = struct.unpack(f"<{rank}I", take(4 * rank)) if rank else ()
        size = int(np.prod(shape)) if shape else 1
        if kind == 1:
            scale = np.frombuffer(take(4 * shape[0]), dtype="<f4")
            codes = np.frombuffer(take(size), dtype=np.int8).reshape(shape)
            tensors[name] = (codes.astype(np.float32) * scale[:, None]).astype(np.float32)
        else:
            tensors[name] = np.frombuffer(take(4 * size), dtype="<f4").reshape(shape).astype(np.float32)
    if pos != len(data):
        raise ValueError(f"{path}: {len(data) - pos} trailing bytes")
    return tensors


# ---------------------------------------------------------------- architecture

def architecture(arch):
    """architecture.json content for an expertium.variants preset name (iter36 = the R01 width-32 default)."""
    from expertium.variants import variant
    from expertium.model.variant_model import interleave_schedule, layer_steps
    v = variant(arch)
    if v.rep_flags:
        raise NotImplementedError("rep_flags (SM30) is not in the export format yet")
    base = v.base
    d = base.d_model
    streams = []
    for name, c in base.modules:
        stripped = sorted(int(e.partition(":")[2]) for e in v.strip_cmix if e.partition(":")[0] == name)
        streams.append({"name": name, "n_layers": c.n_layers, "n_heads": c.n_heads, "head_size": c.d_model // c.n_heads,
                        "channel_mixer_dim": int(c.channel_mixer_factor * c.d_model), "decay_lora": c.decay_lora,
                        "a_lora": c.a_lora, "v0_mix_amt_lora": c.v0_mix_amt_lora, "gate_lora": c.gate_lora,
                        "channel_mixer_stripped_layers": stripped,
                        "value_lora_stripped_layer0": bool(v.strip_l0_vlora)})
    depths = [c.n_layers for _, c in base.modules]
    steps = layer_steps(interleave_schedule(depths)) if v.interleave else \
        [(i, layer) for i, depth in enumerate(depths) for layer in range(depth)]
    return {
        "format": "anki-rwkv expertium export", "spec_version": SPEC_VERSION, "architecture": v.name,
        "d_model": d, "card_features": 92, "features_fc_dim": base.features_fc_mult * d,
        "head_fc_dim": base.head_fc_mult * d,
        "curve_head": "power_mixture" if v.gru_head else "softmax_mixture",
        "num_curves": v.gru_head or base.num_curves, "num_points": base.num_points,
        "ahead_residual": not v.no_ahead_residual, "zero_features": [int(i) for i in v.zero_features],
        "layer_schedule": "interleaved" if v.interleave else "sequential",
        "layer_steps": [list(s) for s in steps],
        "streams": streams,
        "constants": {"layer_norm_eps": 1e-5, "group_norm_eps": 64e-5, "normalize_eps": 1e-12,
                      "softplus_threshold": 20.0, "prob_clamp": 1e-5,
                      "s_point_spread": 18.5, "s_max": 22.0, "s_min": 0.1, "curve_base": 0.9,
                      "point_spread": 18.5, "max_e": 21.0, "point_min": 0.5,
                      "power_clamp": 25.0, "power_s_eps": 1e-7, "min_elapsed_seconds": 1.0},
        "inputs": {"streams_order": [s["name"] for s in streams],
                   "features": "92 card features of rwkv.prepare_batch / rwkv.data_processing (the same input layout "
                               "and normalization the fork's 2.76M model uses: rslib/src/rwkv/mod.rs CARD_FEATURES)",
                   "label_elapsed_seconds": "curve query time in seconds (labels[..., 0])"},
        "outputs": {"p_curve": "recall probability at the queried elapsed time (ahead / curve head)",
                    "p_imm": "1 - P(again) of the 4-rating head (instant head), clamped to [1e-5, 1 - 1e-5]",
                    "p_rating": "softmax over again/hard/good/easy"},
    }


def load_checkpoint(path):
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    return {k: v.float() for k, v in state.items()}


def build_torch(arch, state):
    """The model class the held-out scorer uses (small_model_heldout.build_expertium), CPU reference recurrence."""
    if arch == "iter36":
        from expertium.architecture import DEFAULT_ANKI_RWKV_CONFIG
        from expertium.model.srs_model import SrsRWKV
        model = SrsRWKV(DEFAULT_ANKI_RWKV_CONFIG).float()
    else:
        from expertium.variants import build
        model = build(arch).float()
    model.load_state_dict(state, strict=True)
    return model.eval()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(checkpoint, arch, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    state = load_checkpoint(checkpoint)
    build_torch(arch, state)  # strict load: the checkpoint matches the declared architecture
    tensors = {k: v.numpy() for k, v in state.items()}
    write_f32(out / "weights.f32.bin", tensors)
    write_int8(out / "weights.int8.bin", tensors)
    spec = architecture(arch)
    spec["parameters"] = int(sum(a.size for a in tensors.values()))
    spec["int8_tensors"] = int8_names(tensors)
    (out / "architecture.json").write_text(json.dumps(spec, indent=1))
    # Round-trip checks: the f32 file is bit-exact, the int8 file dequantizes to compress_quant's w8-pc weights.
    back = read_weights(out / "weights.f32.bin")
    assert back.keys() == tensors.keys() and all(np.array_equal(back[k], tensors[k]) for k in tensors)
    from scripts import compress_quant as cq
    reference = cq.quantize_state_dict(state, cq.linear_weight_names(state), 8, None)
    back8 = read_weights(out / "weights.int8.bin")
    mismatch = [k for k in tensors if not np.array_equal(back8[k], reference[k].numpy())]
    if mismatch:
        raise SystemExit(f"int8 container differs from compress_quant w8-pc on {mismatch[:5]}")
    sizes = {f: (out / f).stat().st_size for f in ("weights.f32.bin", "weights.int8.bin", "architecture.json")}
    print(f"[export] {arch}: {spec['parameters']:,} params -> {sizes}", flush=True)
    return sizes


# ---------------------------------------------------------------- verification

def pick_users(count, max_reviews, start=0):
    from scripts import compress_common as cc
    from scripts.small_model_heldout import heldout_users
    counts = cc.run_info()["review_counts"]
    users = [u for u in heldout_users(counts) if counts[str(u)] <= max_reviews]
    return users[start:start + count], counts


def batch_arrays(batch):
    to = lambda t: t.cpu().numpy()  # noqa: E731
    return {"start": to(batch.start), "sub_gather": [[to(g) for g in s] for s in batch.sub_gather],
            "sub_gather_lens": [list(s) for s in batch.sub_gather_lens],
            "time_shift_selects": [[to(g) for g in s] for s in batch.time_shift_selects],
            "skips": [[to(g) for g in s] for s in batch.skips], "num_data": int(batch.num_data),
            "labels": to(batch.labels)}


def verify(directory, checkpoint, users=3, max_reviews=3000, start=0, dtype="float32"):
    import torch
    from scripts import compress_common as cc
    from scripts import compress_quant as cq
    from scripts.small_model_numpy import NumpyModel
    torch.set_num_threads(2)
    directory = Path(directory)
    spec = json.loads((directory / "architecture.json").read_text())
    arch = "iter36" if spec["architecture"] == "iter36" else spec["architecture"]
    state = load_checkpoint(checkpoint)
    variants = {"fp32": (state, directory / "weights.f32.bin"),
                "w8": (cq.quantize_state_dict(state, cq.linear_weight_names(state), 8, None),
                       directory / "weights.int8.bin")}
    chosen, counts = pick_users(users, max_reviews, start)
    if not chosen:
        raise SystemExit(f"no held-out user with <= {max_reviews} reviews (the smallest has about 3,700)")
    data = cc.data_path()
    report = {"checkpoint": str(checkpoint), "users": chosen, "dtype": dtype, "variants": {}}
    batches = {}
    for user in chosen:
        batches[user] = cc.full_history_batch(data, user)
    for name, (torch_state, path) in variants.items():
        model = build_torch(arch, torch_state)
        reference = NumpyModel(read_weights(path), spec, dtype=dtype)
        worst = {"p_curve": 0.0, "p_imm": 0.0, "p_rating": 0.0}
        per_user = []
        for user in chosen:
            batch = batches[user]
            t0 = time.perf_counter()
            with torch.inference_mode():
                stats = model.get_loss(batch)
            t1 = time.perf_counter()
            out = reference.predict(**batch_arrays(batch))
            t2 = time.perf_counter()
            labelled = stats.has_label.bool().numpy()
            row = {"user": user, "reviews": counts[str(user)], "labelled": int(labelled.sum()),
                   "torch_seconds": round(t1 - t0, 2), "numpy_seconds": round(t2 - t1, 2)}
            for key, theirs in (("p_curve", stats.p_curve), ("p_imm", stats.p_imm), ("p_rating", stats.p_imm_all)):
                theirs = theirs.float().numpy()
                diff = np.abs(out[key].astype(np.float64) - theirs.astype(np.float64))
                row[key] = float(diff[labelled].max()) if labelled.any() else 0.0
                row[key + "_all_positions"] = float(diff.max())
                worst[key] = max(worst[key], row[key])
            per_user.append(row)
            print(f"[verify] {name} user {user} ({row['reviews']} reviews): max |dp| curve {row['p_curve']:.2e} "
                  f"instant {row['p_imm']:.2e} rating {row['p_rating']:.2e} | torch {row['torch_seconds']}s "
                  f"numpy {row['numpy_seconds']}s", flush=True)
        report["variants"][name] = {"max_abs_dp_labelled": worst, "per_user": per_user,
                                    "pass_1e-5": max(worst.values()) < 1e-5}
    (directory / "parity.json").write_text(json.dumps(report, indent=1))
    return report


# ---------------------------------------------------------------- manifest

def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def _compact(value, limit=64):
    """Drop bulky per-user payloads (lists / maps longer than `limit`) from copied training records."""
    if isinstance(value, dict):
        if len(value) > limit:
            return f"<{len(value)} entries omitted>"
        return {k: _compact(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return f"<{len(value)} items omitted>" if len(value) > limit else [_compact(v, limit) for v in value]
    return value


def training_record(checkpoint):
    """experiment.json / decay.json and per-epoch records found above the checkpoint (accuracy_loop or decay layout)."""
    checkpoint = Path(checkpoint).resolve()
    record = {"checkpoint_folder_result": _json(checkpoint.parent / "result.json")}
    for parent in checkpoint.parents:
        if (parent / "experiment.json").exists():
            record["experiment_dir"] = str(parent)
            record["experiment"] = _json(parent / "experiment.json")
            phases = parent / "phases.jsonl"
            if phases.exists():
                record["phases"] = [json.loads(line) for line in phases.read_text().splitlines() if line.strip()]
            break
        if (parent / "decay.json").exists():
            record["decay_dir"] = str(parent)
            record["decay"] = _json(parent / "decay.json")
            protocol = _json(parent / "protocol.json")
            if protocol:
                record["decay_protocol_full_run"] = protocol.get("full_run")
            break
        if parent == ROOT or parent.parent == parent:
            break
    return _compact(record)


def eval_summary(label, heldout, reference="fork/w8+s8"):
    """Review-weighted benchmark LogLoss / RMSE(bins) of every `label/<variant>` row, alone and paired with the
    reference on shared users (delta % vs reference)."""
    from scripts.full_run_metrics import aggregate
    path = Path(heldout)
    if not path.exists():
        return {"heldout": str(path), "missing": True}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if "benchmark" in r]
    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], {})[row["user_id"]] = row
    out = {"heldout": str(path), "reference": reference, "labels": {}}
    ref = by_label.get(reference, {})
    for name, per_user in sorted(by_label.items()):
        if name.split("/", 1)[0] != label:
            continue
        entry = {"checkpoint": next(iter(per_user.values())).get("checkpoint"), "alone": aggregate(per_user.values())}
        shared = sorted(set(per_user) & set(ref))
        if shared:
            mine, theirs = aggregate([per_user[u] for u in shared]), aggregate([ref[u] for u in shared])
            delta = {}
            for head in ("ahead", "instant"):
                for metric in ("LogLoss", "RMSE(bins)"):
                    try:
                        a, b = mine[head][metric]["weighted"], theirs[head][metric]["weighted"]
                        delta[f"{head} {metric}"] = {"value": a, "reference": b, "delta_pct": 100 * (a - b) / b}
                    except (KeyError, TypeError, ZeroDivisionError):
                        pass
            entry["vs_reference"] = {"shared_users": len(shared), "delta": delta}
        out["labels"][name] = entry
    report = ROOT / "artifacts" / "small-model" / "heldout-report.json"
    if report.exists():
        data = _json(report) or {}
        out["heldout_report_rows"] = [r for r in data.get("rows", []) if str(r.get("label", "")).startswith(label + "/")]
    return out


def manifest(directory, checkpoint, arch, label, heldout):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = load_checkpoint(checkpoint)
    files = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    from scripts.experiment_provenance import process_provenance
    data = {
        "label": label, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": {"path": str(Path(checkpoint).resolve()), "bytes": Path(checkpoint).stat().st_size,
                       "sha256": sha256(checkpoint)},
        "parameters": int(sum(t.numel() for t in state.values())),
        "architecture": architecture(arch),
        "training": training_record(checkpoint),
        "evaluation": eval_summary(label, heldout),
        "parity": _json(directory / "parity.json"),
        "files": files,
        "provenance": process_provenance(),
    }
    (directory / "manifest.json").write_text(json.dumps(data, indent=1, default=str))
    print(f"[manifest] {directory / 'manifest.json'}: {data['parameters']:,} params, sha256 "
          f"{data['checkpoint']['sha256'][:16]}..., {len(data['evaluation'].get('labels', {}))} evaluated labels",
          flush=True)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "verify", "manifest", "package"):
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", required=True)
        if name in ("export", "package"):
            p.add_argument("--out", required=True)
        if name in ("verify", "manifest"):
            p.add_argument("--dir", required=True)
        if name in ("export", "manifest", "package"):
            p.add_argument("--arch", required=True, help="iter36 (w32), w64, w64-sequential, w80, ...")
        if name in ("manifest", "package"):
            p.add_argument("--label", required=True, help="held-out tag, e.g. scale-w64kd-dec")
            p.add_argument("--heldout", default=str(ROOT / "artifacts" / "small-model" / "heldout.jsonl"))
        if name in ("verify", "package"):
            p.add_argument("--users", type=int, default=3)
            p.add_argument("--max-reviews", type=int, default=6000)
            p.add_argument("--start", type=int, default=0)
            p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    args = parser.parse_args()
    if args.command == "export":
        export(args.checkpoint, args.arch, args.out)
    elif args.command == "verify":
        report = verify(args.dir, args.checkpoint, args.users, args.max_reviews, args.start, args.dtype)
        return 0 if all(v["pass_1e-5"] for v in report["variants"].values()) else 1
    elif args.command == "manifest":
        manifest(args.dir, args.checkpoint, args.arch, args.label, args.heldout)
    else:
        out = Path(args.out)
        if out.exists():
            raise SystemExit(f"{out} exists; packages are never overwritten (pick a new --out)")
        out.mkdir(parents=True)
        shutil.copy2(args.checkpoint, out / "master.pth")
        export(args.checkpoint, args.arch, out)
        report = verify(out, args.checkpoint, args.users, args.max_reviews, args.start, args.dtype)
        manifest(out, args.checkpoint, args.arch, args.label, args.heldout)
        ok = all(v["pass_1e-5"] for v in report["variants"].values())
        print(f"[package] {out}: parity {'PASS' if ok else 'FAIL'} "
              f"{ {k: v['max_abs_dp_labelled'] for k, v in report['variants'].items()} }", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
