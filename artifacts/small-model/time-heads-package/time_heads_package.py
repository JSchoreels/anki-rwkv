#!/usr/bin/env python
"""Answer-time heads of the RWKV family for the Anki fork (time heads for the family (SM117) heads, format 2).

One command per model:

  $PY -X utf8 artifacts/small-model/time-heads-package/time_heads_package.py <key> --fork <fork worktree> \
      [--head-dir DIR] [--master M] [--vectors DIR] [--data SYNTHETIC_DIR]

key: sm92, sm106 (Small / Tiny, 474 inputs), sm99, sm91s2 (Big, 1,306 inputs); with --head-dir / --master also a key the
SM117 table does not know yet (sm113, sm115: the SM119 heads, see the report for the commands).

Steps:
 1. trunk check: the head directory's master.pth (the frozen trunk written after head training) equals the given master
    and the fork's shipped weights file tensor for tensor, bit for bit;
 2. head files into the fork: time_head.f32.bin (ARWKVWEIGHTS1), time_head.int8.bin (ARWKVWEIGHTS8PC1, 2-D weights int8
    per output row), time_head.json; format_version 2 = show-time inputs, same-day reviews included (valid for every
    review that is not the card's first);
 3. synthetic parity fixture (users 900001 and 900002, no dataset data): the show-time batch (SM117 show_time_batch on
    the synthetic data), checked equal to the family package's vectors; per labelled instant row with a time label
    (SM117 rule: duration > 0, not the card's first review) the frozen inputs, the 26 explicit features and the PyTorch
    head logits, f32 (f32 trunk + f32 head) and int8 (Small / Tiny: the int8 trunk = weights.int8.bin values + the
    int8 head = time_head.int8.bin values; Big: the f32 trunk + the int8 head, the fork has no int8 Big weights);
    converted with the fork's qt/tools/rwkv_sm40_vectors.py --limit 500 --intervals into
    rslib/src/rwkv/testdata/<fixture>-time-vectors.json.gz.
"""
from argparse import ArgumentParser
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("RWKV_NO_JIT", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
WT = Path("C:/Users/jonat/workspace/wt")
HF = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/hf-family/repo-new")
USERS = (900001, 900002)
FORMAT_VERSION = 2
PREFIX = "time_mlps.0."

TARGETS = {
    # key: (plain name, kind, arch, fork weights file, head file stem, fixture name, family package)
    "sm92": ("the show-time retrain (SM92)", "expertium", "w64", "sm92/weights.f32.bin", "sm92/time_head", "sm92",
             "rwkv-small-two-head"),
    "sm106": ("the Tiny two-head model (SM106)", "expertium", "w64-c1", "sm106/weights.f32.bin", "sm106/time_head",
              "sm106", "rwkv-tiny-two-head"),
    "sm99": ("the show-time instant-only big model (SM99)", "fork", "", "sm99/weights.f32.bin", "sm99/time_head",
             "sm99", "rwkv-big-instant-only"),
    "sm91s2": ("the show-time big model (SM91s2)", "fork", "", "RWKV_trained_on_5000_10000.bin",
               "RWKV_trained_on_5000_10000.time_head", "legacy", "rwkv-big-two-head"),
    "sm113": ("the longer Small Instant-Only model (SM113)", "expertium", "w64", "sm113/weights.f32.bin",
              "sm113/time_head", "sm113", None),
    "sm115": ("the longer cool-down for Tiny (SM115)", "expertium", "w64-c1", "sm115/weights.f32.bin",
              "sm115/time_head", "sm115", None),
}

SMALL_LAYOUT = [
    "layer_norm(x0) (64): the residual after the input encoder (features2card, after its SiLU)",
    "layer_norm(x1) (64): after layer round 0 (steps (0,0) (1,0) (2,0) (3,0) (4,0))",
    "layer_norm(x2) (64): after layer round 1 (steps (0,1) (1,1) (3,1) (4,1))",
    "layer_norm(x3) (64): after layer round 2 (steps (1,2) (3,2) (4,2))",
    "layer_norm(x4) (64): after layer round 3 (step (1,3)) = the final residual before prehead_norm",
    "head_p (128): relu(head_p.0(prehead_norm(x4))), the instant head's hidden layer",
]
TINY_LAYOUT = [
    "layer_norm(x0) (64): the residual after the input encoder (features2card, after its SiLU)",
    "layer_norm(x1) (64): after layer round 0 (steps (0,0) (1,0) (2,0) (3,0) (4,0))",
    "layer_norm(x2) (64): after layer round 1 (steps (1,1) (3,1) (4,1)); the one-card-layer model has no card step here",
    "layer_norm(x3) (64): after layer round 2 (steps (1,2) (3,2) (4,2))",
    "layer_norm(x4) (64): after layer round 3 (step (1,3)) = the final residual before prehead_norm",
    "head_p (128): relu(head_p.0(prehead_norm(x4))), the instant head's hidden layer",
]
BIG_LAYOUT = [
    "layer_norm(x0) (128): the residual after the input encoder (features2card)",
    "layer_norm(x1) (128): after the card stream (layers 0-2)",
    "layer_norm(x2) (128): after the deck stream (layers 0-3)",
    "layer_norm(x3) (128): after the note stream (layers 0-1)",
    "layer_norm(x4) (128): after the preset stream (layers 0-2)",
    "layer_norm(x5) (128): after the user stream (layers 0-3) = the final residual before prehead_norm",
    "head_p (512): relu(head_p.0(prehead_norm(x5))), the instant head's hidden layer",
]


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def spec(key, inputs, explicit_names, edges, cap_ms):
    name, kind, arch = TARGETS[key][:3]
    big = kind == "fork"
    layout = BIG_LAYOUT if big else (TINY_LAYOUT if arch == "w64-c1" else SMALL_LAYOUT)
    frozen = inputs - len(explicit_names)
    stem = Path(TARGETS[key][4]).name
    return {
        "format": "anki-rwkv answer-time head", "format_version": FORMAT_VERSION,
        "model": f"{name}: the head reads its frozen trunk (time heads for the family (SM117) recipe); queries use "
                 "show-time intervals (the package's INTEGRATION.md section 6.10)",
        "answer_time": "the answer time is the whole time on the card, front and back together: from the moment the "
                       "question is shown to the moment a grade button is pressed, capped at the preset's maximum "
                       "answer seconds (the revlog time). The moment 'Show answer' is pressed is not stored, so "
                       "'recall within S seconds' means recalled (Hard, Good or Easy) and answered within S s, "
                       "reading the back included.",
        "weights": {"f32": f"{stem}.f32.bin", "int8": f"{stem}.int8.bin",
                    "tensors": {PREFIX + "hidden.weight": [128, inputs], PREFIX + "hidden.bias": [128],
                                PREFIX + "out.weight": [64, 128], PREFIX + "out.bias": [64],
                                "time_head.format_version": [1]}},
        "head": f"logits = out(relu(hidden(inputs))); inputs = {inputs} float32 values at the query (instant) row",
        "inputs": {
            "intervals": "show-time: the instant query row's interval runs from the card's previous answer to the "
                         "moment the card is shown, floor(max(elapsed_seconds - duration / 1000, 0))",
            "frozen": {"count": frozen, "layout": layout,
                       "unit": ("a snapshot after each stream of the sequential schedule (card, deck, note, preset, "
                                "user)") if big else "a snapshot after each layer round of the interleaved schedule",
                       "layer_norm": "no affine parameters, eps 1e-5, biased variance (torch.nn.functional.layer_norm)"},
            "explicit": {"count": len(explicit_names), "names": list(explicit_names),
                         "source": "the user's reviews before the query (card id, grade 1-4, duration ms) and the "
                                   "query's own card id and elapsed days",
                         "ln_time": "ln(max(duration_ms, 1) / 1000) in float64; only durations > 0 are recorded",
                         "centre": "ln times are centred on ln 8 s",
                         "median": "exact running median (two heaps; even count = mean of the two middle values) "
                                   "per grade 1-4 of recorded times, fallback the median over all grades, then ln 8 s"},
        },
        "output": {"bins": 16, "layout": "logits [4 grades (Again, Hard, Good, Easy) x 16 bins], softmax per grade",
                   "bin_edges_ms": [float(e) for e in edges],
                   "bins_meaning": "bin 0 < 500 ms, bin k = [edge k-1, edge k), bin 15 >= 60 s (the time cap)",
                   "cap_ms": int(cap_ms),
                   "median": "ln seconds, linear in ln time inside the crossing bin (bin 0 from ln 0.25 s, bin 15 = "
                             "60 s)",
                   "fast_threshold_ms": float(edges[6]),
                   "fast": "P(answer under 3.891 s | grade) = sum of bins 0..6"},
        "valid": "the query (instant) row of every review that is not the card's first (elapsed days >= 0, same-day "
                 "reviews included: on show-time inputs the instant row does not hold the review's own answer time); "
                 "first reviews are not predicted",
    }


def state_equal(a, b):
    return set(a) == set(b) and all(np.array_equal(np.asarray(a[k], np.float32), np.asarray(b[k], np.float32))
                                    for k in a)


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("key", choices=sorted(TARGETS))
    ap.add_argument("--fork", default=str(WT / "fork-time-heads"))
    ap.add_argument("--head-dir", help="default: wt/sm117/artifacts/small-model/sm117-<key>-head")
    ap.add_argument("--master", help="the released master.pth (default: SM117's verified master of the key)")
    ap.add_argument("--vectors", help="package testvectors folder to check the batch against (default: the family "
                                      "package's, if any)")
    ap.add_argument("--data", default=os.environ.get("RWKV_TESTVECTOR_DATA"), help="synthetic dataset dir")
    ap.add_argument("--out", default=None, help="scratch folder (default: <this folder>/work/<key>)")
    args = ap.parse_args()
    if not args.data:
        raise SystemExit("--data (the synthetic users folder) is required")
    import torch
    torch.set_num_threads(4)
    from scripts import sm117_head as sh
    from scripts.sm117_timelib import EXPLICIT_NAMES, TIME_BIN_EDGES_MS, TIME_CAP_MS, explicit_table, revlog, time_labels
    from scripts.small_model_export import write_f32, write_int8, read_weights
    from scripts import compress_quant as cq

    key = args.key
    name, kind, arch, weights_rel, head_stem, fixture, package = TARGETS[key]
    fork = Path(args.fork)
    infer = fork / "qt" / "aqt" / "rwkv_inference"
    head_dir = Path(args.head_dir) if args.head_dir else WT / f"sm117/artifacts/small-model/sm117-{key}-head"
    out = Path(args.out) if args.out else HERE / "work" / key
    out.mkdir(parents=True, exist_ok=True)
    report = {"key": key, "model": name}

    # 1. trunk check
    if args.master:
        master_path = Path(args.master)
    else:
        master_path = sh.verified_master(key)["master"]
    report["master_sha256"] = sha256(master_path)
    trunk = torch.load(head_dir / "master.pth", map_location="cpu", weights_only=True)
    released = torch.load(master_path, map_location="cpu", weights_only=True)
    if isinstance(released, dict) and "state_dict" in released and isinstance(released["state_dict"], dict):
        released = released["state_dict"]
    trunk = {k: v.float() for k, v in trunk.items()}
    if not state_equal({k: v.numpy() for k, v in trunk.items()}, {k: v.float().numpy() for k, v in released.items()}):
        raise SystemExit(f"{key}: the head's trunk is not the released master")
    shipped = read_weights(infer / weights_rel)
    trunk_np = {k: v.numpy() for k, v in trunk.items()}
    if state_equal(shipped, trunk_np):
        report["trunk_equals_fork"] = f"{weights_rel}: {len(shipped)} tensors, names and values bit for bit"
    else:
        common = set(shipped) & set(trunk_np)
        diff = sorted(set(shipped) ^ set(trunk_np))
        bad = [k for k in common if not np.array_equal(shipped[k], trunk_np[k])]
        raise SystemExit(f"{key}: fork weights differ from the trunk: {len(diff)} names differ ({diff[:6]}), "
                         f"{len(bad)} values differ ({bad[:6]})")
    print(f"[time-heads] {key}: {report['trunk_equals_fork']}", flush=True)

    # 2. head files
    head_sd = torch.load(head_dir / "head.pth", map_location="cpu", weights_only=True)
    tensors = {PREFIX + k: v.float().numpy() for k, v in head_sd.items()}
    assert set(tensors) == {PREFIX + s for s in ("hidden.weight", "hidden.bias", "out.weight", "out.bias")}, tensors.keys()
    inputs = tensors[PREFIX + "hidden.weight"].shape[1]
    want_inputs = 1306 if kind == "fork" else 474
    if inputs != want_inputs or tensors[PREFIX + "out.weight"].shape != (64, 128):
        raise SystemExit(f"{key}: head shapes {[(k, v.shape) for k, v in tensors.items()]}")
    tensors["time_head.format_version"] = np.array([float(FORMAT_VERSION)], np.float32)
    f32_path = infer / f"{head_stem}.f32.bin"
    i8_path = infer / f"{head_stem}.int8.bin"
    write_f32(f32_path, tensors)
    write_int8(i8_path, tensors)
    back = read_weights(f32_path)
    assert state_equal(back, tensors)
    (infer / f"{head_stem}.json").write_text(
        json.dumps(spec(key, inputs, EXPLICIT_NAMES, TIME_BIN_EDGES_MS, TIME_CAP_MS), indent=1) + "\n", encoding="utf-8")
    report["head_shapes"] = {k: list(v.shape) for k, v in tensors.items()}
    report["head_files"] = {p.name: p.stat().st_size for p in (f32_path, i8_path, infer / f"{head_stem}.json")}
    print(f"[time-heads] {key}: head files {report['head_files']}", flush=True)

    def mlp(weights):
        head = sh.make_time_mlp(inputs, 128, 64)
        head.load_state_dict({k[len(PREFIX):]: torch.as_tensor(v) for k, v in weights.items() if k.startswith(PREFIX)})
        return head.eval()

    head_f32 = mlp(tensors)
    head_i8 = mlp(read_weights(i8_path))
    # the int8 trunk: weights.int8.bin's own values (Small / Tiny); Big has no int8 weights in the fork
    if kind == "fork":
        trunk_i8 = None
    else:
        i8 = read_weights(infer / weights_rel.replace("weights.f32.bin", "weights.int8.bin"))
        q = cq.quantize_state_dict(trunk, cq.linear_weight_names(trunk), 8, None)
        if not state_equal(i8, {k: v.numpy() for k, v in q.items()}):
            mx = max(float(np.abs(i8[k] - q[k].numpy()).max()) for k in i8)
            print(f"[time-heads] {key}: weights.int8.bin vs quantize_state_dict max diff {mx:.3g}; using the file's values")
        trunk_i8 = {k: torch.as_tensor(v) for k, v in i8.items()}
    models = {"fp32": sh.build_model(key if key in sh.MODELS else _alias(key), trunk, "cpu")}
    if trunk_i8 is not None:
        models["int8"] = sh.build_model(key if key in sh.MODELS else _alias(key), trunk_i8, "cpu")
    frozen = {n: sh.Frozen(m, kind) for n, m in models.items()}
    for f in frozen.values():
        assert f.inputs == inputs, (f.inputs, inputs)

    # 3. fixture
    vec_dir = Path(args.vectors) if args.vectors else (HF / package / "testvectors" if package else None)
    pkg = out / "package"
    (pkg / "testvectors").mkdir(parents=True, exist_ok=True)
    for old in (pkg / "testvectors").glob("*.npz"):
        old.unlink()
    report["users"] = {}
    selfcheck = 0.0
    for user in USERS:
        gb, _ = sh.show_time_batch(Path(args.data), user)
        start = gb.start.cpu().numpy()
        base = dict(np.load(vec_dir / f"user_{user}.npz", allow_pickle=False)) if vec_dir else None
        if base is not None and not (base["features"].shape == start.shape and np.array_equal(base["features"], start)):
            raise SystemExit(f"{key} user {user}: the show-time batch differs from {vec_dir}")
        rev = revlog(Path(args.data), user)
        table = explicit_table(rev)
        lower, _, _ = time_labels(rev["duration"].to_numpy(), rev["elapsed_days"].to_numpy(), same_day=True)
        res = {}
        with torch.inference_mode():
            for n, f in frozen.items():
                qi, th, feats, o = f.forward(gb)
                stock = f.stock(gb)
                for a, b in zip(o, stock):
                    if torch.is_tensor(a) and not torch.equal(a, b):
                        raise SystemExit(f"{key} {n}: capture outputs differ from the stock forward")
                res[n] = (qi.numpy(), th.numpy(), feats.float(), o)
        qi, th, feats_f32, o = res["fp32"]
        keep = lower[th - 1] >= 0
        qi, th = qi[keep], th[keep]
        order = np.argsort(qi)
        qi, th = qi[order], th[order]
        explicit = table[th - 1].astype(np.float32)
        x32 = torch.cat([feats_f32[torch.as_tensor(keep)][torch.as_tensor(order)], torch.as_tensor(explicit)], -1)
        with torch.inference_mode():
            l32 = head_f32(x32).numpy().astype(np.float32)
            if "int8" in res:
                q2, t2, f8, _ = res["int8"]
                assert np.array_equal(q2[keep][order], qi)
                x8 = torch.cat([f8[torch.as_tensor(keep)][torch.as_tensor(order)], torch.as_tensor(explicit)], -1)
            else:
                x8 = x32
            l8 = head_i8(x8).numpy().astype(np.float32)
        # NumPy recomputation from the written containers
        for x, lg, path in ((x32.numpy(), l32, f32_path), (x8.numpy(), l8, i8_path)):
            w = read_weights(path)
            h = np.maximum(x.astype(np.float64) @ w[PREFIX + "hidden.weight"].T.astype(np.float64)
                           + w[PREFIX + "hidden.bias"], 0)
            ref = h @ w[PREFIX + "out.weight"].T.astype(np.float64) + w[PREFIX + "out.bias"]
            selfcheck = max(selfcheck, float(np.abs(ref - lg).max()))
        if base is None:
            raise SystemExit(f"{key}: no package vectors to build the fixture rows from (pass --vectors)")
        if not (base["is_query"][qi].all() and base["has_label"][qi].all()):
            raise SystemExit(f"{key} user {user}: a time row is not a labelled query row")
        if not np.array_equal(base["review_th"][qi], th):
            raise SystemExit(f"{key} user {user}: time rows do not map to the vector rows")
        # P(rating) at the time rows equals the vectors' (the frozen pass is the stock forward)
        n = int(th.max())
        out_npz = dict(base)
        out_npz.update({
            "time_row": qi.astype(np.int64), "time_review_th": th.astype(np.int64), "time_explicit": explicit,
            "fp32_time_logits": l32, "int8_time_logits": l8,
            "rev_card_id": rev["card_id"].to_numpy()[:n].astype(np.int64),
            "rev_rating": rev["rating"].to_numpy()[:n].astype(np.int64),
            "rev_duration": rev["duration"].to_numpy()[:n].astype(np.int64),
            "rev_elapsed_days": rev["elapsed_days"].to_numpy()[:n].astype(np.int64),
            "time_int8_meaning": np.array("int8 trunk + int8 head" if "int8" in res else
                                          "f32 trunk + int8 head (no int8 Big weights in the fork)"),
        })
        np.savez_compressed(pkg / "testvectors" / f"user_{user}.npz", **out_npz)
        el = rev["elapsed_days"].to_numpy()[th - 1]
        first500 = qi < 500
        report["users"][str(user)] = {
            "time_rows": int(len(qi)), "same_day": int((el == 0).sum()), "later_day": int((el >= 1).sum()),
            "first_500_rows": {"time_rows": int(first500.sum()), "same_day": int(((el == 0) & first500).sum()),
                               "later_day": int(((el >= 1) & first500).sum())}}
    report["numpy_logit_selfcheck_max_abs"] = selfcheck
    target = fork / "rslib" / "src" / "rwkv" / "testdata" / f"{fixture}-time-vectors.json.gz"
    subprocess.run([sys.executable, "-X", "utf8", str(fork / "qt" / "tools" / "rwkv_sm40_vectors.py"), str(pkg),
                    str(target), "--limit", "500", "--intervals"], check=True)
    report["fixture"] = {"path": str(target), "bytes": target.stat().st_size, "sha256": sha256(target)}
    report["int8_meaning"] = "int8 trunk + int8 head" if trunk_i8 is not None else \
        "f32 trunk + int8 head (the fork has no int8 Big weights)"
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1), flush=True)


def _alias(key):
    """A model the SM117 table does not list: the scorer kind of its architecture."""
    from scripts import sm117_head as sh
    name, kind, arch = TARGETS[key][:3]
    sh.MODELS[key] = {"name": name, "kind": kind, "arch": arch}
    return key


if __name__ == "__main__":
    main()
