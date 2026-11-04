#!/usr/bin/env python
"""Show-time retrain of the release (SM92): package the answer-time head retrained on the frozen SM92 model (the
answer-time head on richer frozen features (SM82c) recipe, artifacts/small-model/sm92/head/) as `artifacts/models/
sm92-time`, with the packaging of the answer-time head release (SM83) (scripts/sm83_time_package.py of the sm83
worktree, read only: its spec, containers, NumPy reference and checker), and two changes:

  - the model is the SM92 package `artifacts/models/sm92` (its master.pth = the head master's trunk bit for bit, its
    weights.f32.bin = that trunk);
  - the vectors are the SM92 package's vectors (show-time query rows, `--inputs e2s-query`): the frozen features and the
    head logits are computed on the same show-time batch (the rows the fork feeds the native trunk), checked equal to the
    package vectors' 92 inputs before anything is written.

usage (CPU): $PY -X utf8 sm92_time_package.py [--out DIR]
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

os.environ.setdefault("RWKV_NO_JIT", "1")
os.environ.pop("RWKV_DELETED_CARDS", None)   # the leak-free transform is applied explicitly (sm71_leak.neutralise)
import numpy as np  # noqa: E402

SM83 = Path("C:/Users/jonat/workspace/wt/sm83")
sys.path.insert(0, str(SM83))
from scripts import sm83_time_package as s83  # noqa: E402

HERE = Path(__file__).resolve().parent
WT = HERE.parents[3]                                   # the sm92 worktree
SM = WT / "artifacts" / "small-model"
MODEL = WT / "artifacts" / "models" / "sm92"
MASTER = SM / "sm92-head" / "master.pth"
DECISION = SM / "sm92" / "head" / "sm92-head-decision.json"
USERS = (4676, 1745)
HEAD = s83.HEAD
FROZEN_ROWS = s83.FROZEN_ROWS


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sm31_inputs():
    """scripts/sm31_inputs.py of the sm92 worktree (the module the SM92 vectors and scores used; the sm83 worktree
    predates it). It imports only rwkv.data_processing, identical in both worktrees."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("sm92_sm31_inputs", WT / "scripts" / "sm31_inputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def show_time_features(state, user, data):
    """sm82c_head._features_cpu on the show-time batch: the leak-free frame, then the scorer's e2s-query frame and
    query-row changes (scripts/sm31_inputs.py), the same rows the package vectors hold."""
    import torch
    from rwkv.data_processing import get_rwkv_data
    from rwkv.prepare_batch import prepare
    from scripts import compress_common as cc
    from scripts import sm71_leak
    from scripts.sm82c_head import Frozen, build_frozen
    sm31_inputs = load_sm31_inputs()
    frame = get_rwkv_data(data, user)
    frame, _ = sm71_leak.neutralise(frame, "neighbour", data, user)
    frame = sm31_inputs.apply_frame(frame, "e2s-query")
    with sm31_inputs.query_patch("e2s-query"):
        batch = prepare(cc.make_samples(frame, user, len(frame)), seed=2025 + user)
    fz = Frozen(build_frozen(state, torch.device("cpu")))
    with torch.inference_mode():
        qi, th, feats, o = fz.forward(batch)
    order = torch.argsort(th)
    return th[order], feats[order], batch, o, qi[order]


def build_vectors(user, master_state, model_dir, out_dir):
    from scripts import small_model_heldout as smh  # noqa: F401  first: installs the state quantizer before the model
    import torch
    from scripts import compress_common as cc
    from scripts import compress_quant as cq
    from scripts.sm82c_head import explicit_table, revlog, INPUTS, HIDDEN, BINS
    from expertium.model.variant_model import TimeMLP
    data = cc.data_path()
    old = dict(np.load(model_dir / "testvectors" / f"user_{user}.npz", allow_pickle=False))
    rev = revlog(data, user)
    explicit = explicit_table(rev)
    out = dict(old)
    variants = {"fp32": master_state,
                "int8": cq.quantize_state_dict(master_state, cq.linear_weight_names(master_state), 8, None)}
    report = {}
    for name, state in variants.items():
        th, feats, batch, o, qi = show_time_features(state, user, data)
        start = batch.start.cpu().numpy()
        same_inputs = start.shape == old["features"].shape and np.array_equal(start, old["features"])
        if not same_inputs:
            raise SystemExit(f"user {user}: rebuilt show-time inputs differ from the SM92 package vectors")
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
            out["fp32_time_frozen"] = feats.numpy()[:FROZEN_ROWS].astype(np.float32)
        out[f"{name}_time_logits"] = logits
        report[name] = {"rows": int(len(rows)), "inputs_equal_package_vectors": bool(same_inputs)}
    n = int(out["time_review_th"].max())
    for col, src in (("rev_card_id", "card_id"), ("rev_rating", "rating"), ("rev_duration", "duration"),
                     ("rev_elapsed_days", "elapsed_days")):
        out[col] = rev[src].to_numpy()[:n].astype(np.int64)
    path = out_dir / "testvectors" / f"user_{user}.npz"
    np.savez_compressed(path, **out)
    report["file"] = path.name
    report["bytes"] = path.stat().st_size
    print(f"[sm92-time] vectors {path}: {report}", flush=True)
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
    released = torch.load(MODEL / "master.pth", map_location="cpu", weights_only=True)
    if "state_dict" in released and isinstance(released["state_dict"], dict):
        released = released["state_dict"]
    trunk = {k: v for k, v in sd.items() if not k.startswith(HEAD)}
    if trunk.keys() != released.keys() or not all(torch.equal(trunk[k], released[k].float()) for k in trunk):
        raise SystemExit("the head master's trunk is not the SM92 master bit for bit")
    shipped = read_weights(MODEL / "weights.f32.bin")
    if shipped.keys() != trunk.keys() or not all(np.array_equal(shipped[k], trunk[k].numpy()) for k in trunk):
        raise SystemExit("the SM92 weights.f32.bin is not the head master's trunk")
    head = {k: v for k, v in sd.items() if k.startswith(HEAD)}
    torch.save(head, out / "head.pth")
    tensors = {k: v.numpy() for k, v in head.items()}
    tensors["time_head.format_version"] = np.array([float(s83.FORMAT_VERSION)], np.float32)
    write_f32(out / "time_head.f32.bin", tensors)
    write_int8(out / "time_head.int8.bin", tensors)
    back = read_weights(out / "time_head.f32.bin")
    assert back.keys() == tensors.keys() and all(np.array_equal(back[k], tensors[k]) for k in tensors)
    spec = s83.spec()
    spec["model"] = ("sm92 (the show-time retrain of the release): the head reads its frozen trunk; queries use "
                     "show-time intervals (the package's INTEGRATION.md section 6.10)")
    (out / "time_head.json").write_text(json.dumps(spec, indent=1) + "\n")
    shutil.copy(SM83 / "artifacts/small-model/sm83/reference/answer_time_numpy.py", out / "reference" / "answer_time_numpy.py")
    shutil.copy(SM83 / "artifacts/small-model/sm83/reference/check_time_vectors.py",
                out / "testvectors" / "check_time_vectors.py")
    vectors = {str(u): build_vectors(u, sd, MODEL, out) for u in USERS}
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
    decision = json.loads(DECISION.read_text(encoding="utf-8")) if DECISION.exists() else {}
    gate = decision.get("sm92_gate")
    tj = (decision.get("training") or {})
    manifest = {
        "label": "sm92-time", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": {"package": "sm92", "weights.f32.bin sha256": sha256(MODEL / "weights.f32.bin"),
                  "master.pth sha256": sha256(MODEL / "master.pth")},
        "head_master": {"path": "<workspace>/wt/sm92/artifacts/small-model/sm92-head/master.pth", "sha256": sha256(MASTER)},
        "parameters": {"head": int(sum(v.numel() for v in head.values())), "int8_tensors": int8_names(tensors)},
        "training": (f"the answer-time head on richer frozen features (SM82c) recipe on the frozen SM92 model: one pass "
                     f"over {tj.get('rows')} labelled rows ({tj.get('users')} users, first 400,000 reviews each), "
                     f"{tj.get('step')} AdamW steps of {tj.get('batch')} rows, lr {tj.get('lr')} cosine, heads only"),
        "evaluation": {"gate": gate, "time": (decision.get("time") or {}).get("predictors"),
                       "vs": {k: v for k, v in ((decision.get("time") or {}).get("vs") or {}).items()
                              if k in ("card_last", "user_median", "ablation", "sm82")},
                       "grade_unchanged": decision.get("grade_unchanged"), "verdict": decision.get("verdict")},
        "parity": parity, "files": files,
        "provenance": {"sm83_packaging_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                                               text=True, cwd=SM83).stdout.strip(),
                       "sm92_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                                     cwd=WT).stdout.strip()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"[sm92-time] package {out} in {time.time() - started:.0f} s", flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(WT / "artifacts" / "models" / "sm92-time"))
    package(parser.parse_args())


if __name__ == "__main__":
    main()
