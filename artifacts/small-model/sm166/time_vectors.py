#!/usr/bin/env python
"""Weekend family release (SM166): answer-time head test vectors for one package (CPU only, synthetic users only).

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python -X utf8 time_vectors.py <key> \
        --package <package folder: master.pth, weights.*.bin, testvectors/user_9000xx.npz> \
        --head <folder with the fork main head files time_head.f32.bin / time_head.int8.bin> \
        --fixture <fork main rslib/src/rwkv/testdata/<fixture>-time-vectors.json.gz> \
        --data <synthetic users folder (scripts/synthetic_testvector_users.py)> --out <folder> [--code <tree>]

key: sm91s2, sm99, sm118, sm113, sm130th, sm130io.

What it does (the recipe of the time heads for the family (SM117) packaging step, time_heads_package.py step 3, without
writing anything into the fork):
 1. the frozen trunk = the package's master.pth; it must equal the package's weights.f32.bin tensor for tensor (bit for
    bit);
 2. the show-time batch of each synthetic user (900001, 900002) must equal the package's own test vectors' features;
 3. per labelled instant row with a time label (duration > 0, not the card's first review) the frozen inputs (layer-
    normalised residual snapshots + the instant head's hidden layer), the 26 explicit features and the head logits:
    fp32 = f32 trunk + time_head.f32.bin; int8 = the int8 trunk (weights.int8.bin values) + time_head.int8.bin values
    (Big: f32 trunk + int8 head, as the fork runs it);
 4. cross-check: the logits of the rows the fork's own fixture holds must equal the fixture's (fork main), so the
    package vectors and the fork's parity test describe the same head;
 5. writes <out>/time_user_<u>.npz (frozen inputs for the first 500 time rows, logits for all of them, the revlog
    columns the explicit features are computed from) and <out>/time-vectors-report.json.
The model code is imported read-only from --code (default: the worktree that holds scripts/sm117_head.py).
"""
from argparse import ArgumentParser
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("RWKV_NO_JIT", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

USERS = (900001, 900002)
FROZEN_ROWS = 500
PREFIX = "time_mlps.0."
KIND = {"sm91s2": "fork", "sm99": "fork", "sm118": "expertium", "sm113": "expertium", "sm130th": "expertium",
        "sm130io": "expertium"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def same(a, b):
    return set(a) == set(b) and all(np.array_equal(np.asarray(a[k], np.float32), np.asarray(b[k], np.float32)) for k in a)


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("key", choices=sorted(KIND))
    ap.add_argument("--package", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--code", default="C:/Users/jonat/workspace/wt/fork-rc-export")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.code)))
    import torch
    torch.set_num_threads(4)
    from scripts import sm117_head as sh
    from scripts.sm117_timelib import explicit_table, revlog, time_labels
    from scripts.small_model_export import read_weights

    key, kind = args.key, KIND[args.key]
    pkg, head_dir, out = Path(args.package), Path(args.head), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"key": key, "kind": kind, "package": str(pkg), "users": {}}

    # 1. trunk = master.pth = weights.f32.bin
    trunk = torch.load(pkg / "master.pth", map_location="cpu", weights_only=True)
    if isinstance(trunk, dict) and isinstance(trunk.get("state_dict"), dict):
        trunk = trunk["state_dict"]
    trunk = {k: v.float() for k, v in trunk.items()}
    shipped = read_weights(pkg / "weights.f32.bin")
    if not same(shipped, {k: v.numpy() for k, v in trunk.items()}):
        raise SystemExit(f"{key}: master.pth differs from weights.f32.bin")
    report["trunk"] = f"master.pth == weights.f32.bin, {len(shipped)} tensors bit for bit"

    head32 = read_weights(head_dir / "time_head.f32.bin")
    head8 = read_weights(head_dir / "time_head.int8.bin")
    version = float(head32["time_head.format_version"][0])
    inputs = head32[PREFIX + "hidden.weight"].shape[1]
    report["head"] = {"format_version": version, "inputs": int(inputs),
                      "f32_sha256": sha256(head_dir / "time_head.f32.bin"),
                      "int8_sha256": sha256(head_dir / "time_head.int8.bin")}

    def mlp(w):
        m = sh.make_time_mlp(inputs, 128, 64)
        m.load_state_dict({k[len(PREFIX):]: torch.as_tensor(v) for k, v in w.items() if k.startswith(PREFIX)})
        return m.eval()

    mlp32, mlp8 = mlp(head32), mlp(head8)
    models = {"fp32": sh.build_model(key, trunk, "cpu")}
    if kind != "fork":
        i8 = read_weights(pkg / "weights.int8.bin")
        models["int8"] = sh.build_model(key, {k: torch.as_tensor(v) for k, v in i8.items()}, "cpu")
    frozen = {n: sh.Frozen(m, kind) for n, m in models.items()}
    for f in frozen.values():
        if f.inputs != inputs:
            raise SystemExit(f"{key}: the model gives {f.inputs} head inputs, the head file wants {inputs}")

    fixture = {u["name"]: u for u in json.load(gzip.open(args.fixture, "rt", encoding="utf-8"))}
    worst_fixture = 0.0
    for user in USERS:
        gb, _ = sh.show_time_batch(Path(args.data), user)
        base = np.load(pkg / "testvectors" / f"user_{user}.npz", allow_pickle=False)
        start = gb.start.cpu().numpy()
        if not (base["features"].shape == start.shape and np.array_equal(base["features"], start)):
            raise SystemExit(f"{key} user {user}: the show-time batch differs from the package's test vectors")
        rev = revlog(Path(args.data), user)
        table = explicit_table(rev)
        lower, _, _ = time_labels(rev["duration"].to_numpy(), rev["elapsed_days"].to_numpy(), same_day=True)
        res = {}
        with torch.inference_mode():
            for n, f in frozen.items():
                qi, th, feats, _ = f.forward(gb)
                res[n] = (qi.numpy(), th.numpy(), feats.float())
                f.clear()
        qi, th, f32 = res["fp32"]
        keep = lower[th - 1] >= 0
        order = np.argsort(qi[keep])
        qi, th = qi[keep][order], th[keep][order]
        explicit = table[th - 1].astype(np.float32)
        sel = torch.as_tensor(np.nonzero(keep)[0][order])
        x32 = torch.cat([f32.index_select(0, sel), torch.as_tensor(explicit)], -1)
        if "int8" in res:
            q2, _, f8 = res["int8"]
            if not np.array_equal(q2[keep][order], qi):
                raise SystemExit(f"{key}: int8 rows differ")
            x8 = torch.cat([f8.index_select(0, sel), torch.as_tensor(explicit)], -1)
        else:
            x8 = x32
        with torch.inference_mode():
            l32 = mlp32(x32).numpy().astype(np.float32)
            l8 = mlp8(x8).numpy().astype(np.float32)
        if not (base["is_query"][qi].all() and base["has_label"][qi].all()
                and np.array_equal(base["review_th"][qi], th)):
            raise SystemExit(f"{key} user {user}: time rows are not labelled query rows of the package vectors")
        # 4. the fork fixture's rows
        fx = fixture[f"user_{user}.npz"]["time"]
        rows = np.asarray(fx["rows"], np.int64)
        pos = np.searchsorted(qi, rows)
        if not (pos < len(qi)).all() or not np.array_equal(qi[pos], rows):
            raise SystemExit(f"{key} user {user}: the fork fixture's time rows are not a subset of these rows")
        d32 = float(np.abs(l32[pos] - np.asarray(fx["fp32"], np.float32)).max())
        d8 = float(np.abs(l8[pos] - np.asarray(fx["int8"], np.float32)).max())
        worst_fixture = max(worst_fixture, d32, d8)
        n = int(th.max())
        k = min(FROZEN_ROWS, len(qi))
        npz = {
            "user": np.int64(user),
            "source": np.array("synthetic user (no dataset data): scripts/synthetic_testvector_users.py"),
            "time_row": qi.astype(np.int64), "time_review_th": th.astype(np.int64), "time_explicit": explicit,
            "fp32_time_frozen": x32[:k, :inputs - explicit.shape[1]].numpy().astype(np.float32),
            "fp32_time_logits": l32, "int8_time_logits": l8,
            "rev_card_id": rev["card_id"].to_numpy()[:n].astype(np.int64),
            "rev_rating": rev["rating"].to_numpy()[:n].astype(np.int64),
            "rev_duration": rev["duration"].to_numpy()[:n].astype(np.int64),
            "rev_elapsed_days": rev["elapsed_days"].to_numpy()[:n].astype(np.int64),
            "int8_meaning": np.array("int8 trunk (weights.int8.bin) + int8 head (time_head.int8.bin)" if "int8" in res
                                     else "f32 trunk + int8 head (time_head.int8.bin), as the fork runs Big"),
        }
        if "int8" in res:
            npz["int8_time_frozen"] = x8[:k, :inputs - explicit.shape[1]].numpy().astype(np.float32)
        np.savez_compressed(out / f"time_user_{user}.npz", **npz)
        report["users"][str(user)] = {"time_rows": int(len(qi)), "frozen_rows": k, "fixture_rows": int(len(rows)),
                                      "fixture_max_abs_logit_diff_fp32": d32, "fixture_max_abs_logit_diff_int8": d8}
        print(f"[sm166-time] {key} user {user}: {len(qi)} time rows, fixture rows {len(rows)} "
              f"max |d| fp32 {d32:.2e} int8 {d8:.2e}", flush=True)
    report["fixture_max_abs_logit_diff"] = worst_fixture
    report["fixture"] = {"path": str(args.fixture), "sha256": sha256(args.fixture)}
    ok = worst_fixture <= 1e-5
    report["result"] = "PASS" if ok else "FAIL"
    (out / "time-vectors-report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"[sm166-time] {key}: RESULT {report['result']} (fork fixture max |d| {worst_fixture:.2e})", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
