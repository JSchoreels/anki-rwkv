"""SM32: corpus collection, codebook fitting and a quick tuning screen for the student's VQ state (scripts/state_vq.py).

Codebooks come from TRAINING users of the student only (the run's protocol.json rows = gate-tier training users that
the student was trained on), never from evaluation users. CPU only.

  collect : run the w8 student over the corpus users (full history) and keep a random sample of the STORED card/note
            states (WKV matrices after each review, token-shift vectors as the next review reads them), about
            --per-user samples per (stream, layer) per user.
  fit     : k-means (Lloyd, torch) codebooks: WKV joint (u, v) residual stages per rank component, shift
            product-quantization stages per (role, m, position), log2 norm ranges per bit width.
  tune    : model-side curve / instant log loss (scripts.full_run_metrics.evaluate_with_predictions values; the label
            DB has no training users) of w8 vs w8+vq recipes on a few other training users, for bit allocation.

usage: state_vq_fit.py collect --users 50 --shard 0/2 ; state_vq_fit.py fit ; state_vq_fit.py tune --recipes a,b
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RWKV_NO_JIT", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)

from scripts import small_model_heldout as smh  # noqa: E402  (installs the state-quant fork model first)
from scripts import compress_common as cc  # noqa: E402
from scripts import compress_quant as cq  # noqa: E402
from scripts import state_vq  # noqa: E402

STUDENT = Path(r"C:\Users\jonat\workspace\wt\small-model\artifacts\small-model"
               r"\scale-w64kd-gate-cached-ctx16384-cont-decay-L32k\expertium\E89-F65e-F77a\master.pth")
PROTOCOL = STUDENT.parents[2] / "protocol.json"
OUT = ROOT / "artifacts" / "small-model" / "state-vq"


def training_users(count, offset, counts, lo=3000, hi=60000):
    """`count` training users with lo..hi reviews, spread over log(history length); `offset` picks a disjoint set."""
    rows = json.loads(PROTOCOL.read_text())["rows"]
    users = sorted({r["user_id"] for r in rows if lo <= counts[str(r["user_id"])] <= hi},
                   key=lambda u: (counts[str(u)], u))
    step = len(users) / (2 * count)
    picks = [users[int((2 * i + offset) * step)] for i in range(count)]
    return picks


def load_student():
    base = {k: v.float() for k, v in torch.load(STUDENT, map_location="cpu", weights_only=True).items()}
    model = smh.build_expertium("w64", base)
    layout = state_vq.install(model)
    smh.set_expertium_state_bits(model, 0)
    bits, group = cq.WEIGHT_SCHEMES["w8-pc"]
    w8 = cq.quantize_state_dict(base, cq.linear_weight_names(base), bits, group)
    model.load_state_dict(w8)
    return model.eval(), layout


def collect(args):
    from scripts.full_run_metrics import evaluate_with_predictions
    run = cc.run_info()
    counts = run["review_counts"]
    users = training_users(args.users, 0, counts)
    index, total = (int(x) for x in args.shard.split("/"))
    users = users[index::total]
    model, layout = load_student()
    data = cc.data_path(run)
    folder = OUT / "corpus"
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1000 + index)
    for user in users:
        path = folder / f"{user}.npz"
        if path.exists():
            continue
        p = min(1.0, args.per_user / counts[str(user)])
        samples = {}

        def rec_wkv(s, cur, rows, t):
            keep = rng.random(s.shape[0]) < p
            if keep.any():
                samples.setdefault(f"wkv/{cur[0]}/{cur[1]}", []).append(s[torch.from_numpy(keep)].numpy().copy())
            return s

        def rec_shift(x, cur):
            keep = rng.random(x.shape[0]) < p
            if keep.any():
                samples.setdefault(f"shift/{cur[0]}/{cur[1]}/{cur[2]}", []).append(
                    x[torch.from_numpy(keep)].numpy().copy())
            return x

        state_vq.Hooks.wkv = {(s, l): rec_wkv for s, info in layout.items() for l in range(info["wkv_layers"])}
        state_vq.Hooks.shift = {}
        state_vq.Hooks.record_shift = rec_shift
        t0 = time.perf_counter()
        batch = cc.full_history_batch(data, user)
        with torch.inference_mode():
            values, _ = evaluate_with_predictions(model, [{"batch": batch}])
        np.savez_compressed(path, **{k: np.concatenate(v) for k, v in samples.items()})
        print(f"[collect] user {user} ({counts[str(user)]} reviews) {time.perf_counter() - t0:.0f}s "
              f"{ {k: sum(len(a) for a in v) for k, v in samples.items()} } curve {values['curve_log_loss']:.4f}",
              flush=True)
    state_vq.Hooks.wkv, state_vq.Hooks.shift, state_vq.Hooks.record_shift = {}, {}, None


def kmeans(X, k, iters, seed):
    """Lloyd k-means on X [N, D] (float32 torch), random data init, empty clusters reseeded from far points."""
    g = torch.Generator().manual_seed(seed)
    N = X.shape[0]
    C = X[torch.randperm(N, generator=g)[:k]].clone()
    for it in range(iters):
        idx = state_vq.nearest(X, C)
        sums = torch.zeros_like(C).index_add_(0, idx, X)
        cnt = torch.bincount(idx, minlength=k).float()
        empty = cnt == 0
        C = torch.where(empty[:, None], C, sums / cnt.clamp(min=1)[:, None])
        if empty.any():
            err = ((X - C[idx]) ** 2).sum(-1)
            far = torch.topk(err, int(empty.sum())).indices
            C[empty] = X[far]
    idx = state_vq.nearest(X, C)
    return C, float(((X - C[idx]) ** 2).sum(-1).mean())


def fit_rvq(X, stages, bits, iters, seed, name):
    books, R = [], X.clone()
    for s in range(stages):
        C, err = kmeans(R, 2 ** bits, iters, seed + s)
        R = R - C[state_vq.nearest(R, C)]
        books.append(C)
        print(f"[fit] {name} stage {s + 1}/{stages}: mean squared residual {float((R ** 2).sum(-1).mean()):.5f} "
              f"(of {float((X ** 2).sum(-1).mean()):.4f})", flush=True)
    return books


def fit(args):
    files = sorted((OUT / "corpus").glob("*.npz"))
    wkv, shift = [], {"t": [], "c": []}
    for f in files:
        z = np.load(f)
        for key in z.files:
            parts = key.split("/")
            if parts[0] == "wkv":
                wkv.append(z[key].reshape(-1, state_vq.K, state_vq.K))
            else:
                shift[parts[3]].append(z[key])
    A = torch.from_numpy(np.concatenate(wkv)).float()
    rng = torch.Generator().manual_seed(0)
    if A.shape[0] > args.max_points:
        A = A[torch.randperm(A.shape[0], generator=rng)[:args.max_points]]
    print(f"[fit] {len(files)} users, WKV heads {A.shape[0]:,}, shift t {sum(map(len, shift['t'])):,} "
          f"c {sum(map(len, shift['c'])):,}", flush=True)
    books = state_vq.Codebooks()
    u, v, sigma, valid = state_vq.factorize(A, 2)
    energy = (A.double() ** 2).sum((1, 2))
    for r in (1, 2):
        kept = (sigma[:, :r] ** 2 * valid[:, :r]).sum(1)
        print(f"[fit] rank-{r} truncation keeps {float(kept.sum() / energy.sum()):.4%} of the WKV energy", flush=True)
    for j, stages in enumerate((args.wkv_stages, args.wkv_stages_2)):
        m = valid[:, j]
        X = torch.cat([u[m, j], v[m, j]], dim=-1).float()
        books.wkv.append(fit_rvq(X, stages, 10, args.iters, 100 * j, f"wkv comp {j}"))
        ln = torch.log2(torch.sqrt(sigma[m, j])).numpy()
        for b in args.norm_bits:
            books.wkv_norm[(j, b)] = state_vq.fit_norm_range(ln, b, sigma[m, j].numpy())
        print(f"[fit] wkv comp {j} log2 sqrt(sigma) quantiles 1/50/99%: "
              f"{np.quantile(ln, [0.01, 0.5, 0.99]).round(3)}; ranges {[(b, books.wkv_norm[(j, b)]) for b in args.norm_bits]}",
              flush=True)
    for role in ("t", "c"):
        X = torch.from_numpy(np.concatenate(shift[role])).float()
        if X.shape[0] > args.max_points:
            X = X[torch.randperm(X.shape[0], generator=rng)[:args.max_points]]
        n = X.norm(dim=-1)
        X = X[n > 1e-20]
        n = n[n > 1e-20]
        unit = X / n[:, None]
        for b in args.norm_bits:
            books.shift_norm[(role, b)] = state_vq.fit_norm_range(torch.log2(n).numpy(), b)
        print(f"[fit] shift {role} log2 norm quantiles 1/50/99%: {np.quantile(torch.log2(n).numpy(), [0.01, 0.5, 0.99]).round(3)}",
              flush=True)
        for m in args.shift_m:
            sub = X.shape[1] // m
            books.shift[(role, m)] = [fit_rvq(unit[:, p * sub:(p + 1) * sub].contiguous(), args.shift_stages, 12,
                                              args.iters, 1000 + 10 * p, f"shift {role} m{m} pos {p}")
                                      for p in range(m)]
    books.save(OUT / args.out)
    print(f"[fit] wrote {OUT / args.out}", flush=True)


def refit_norms(args):
    """Recompute only the norm ranges of an existing codebook file from the corpus (no k-means)."""
    books = state_vq.Codebooks.load(OUT / args.codebooks)
    files = sorted((OUT / "corpus").glob("*.npz"))
    wkv, shift = [], {"t": [], "c": []}
    for f in files:
        z = np.load(f)
        for key in z.files:
            if key.startswith("wkv"):
                wkv.append(z[key].reshape(-1, state_vq.K, state_vq.K))
            else:
                shift[key.split("/")[3]].append(z[key])
    A = torch.from_numpy(np.concatenate(wkv)).float()
    A = A[torch.randperm(A.shape[0], generator=torch.Generator().manual_seed(0))[:args.max_points]]
    _, _, sigma, valid = state_vq.factorize(A, 2)
    for j in range(2):
        m = valid[:, j]
        ln = torch.log2(torch.sqrt(sigma[m, j])).numpy()
        for b in args.norm_bits:
            books.wkv_norm[(j, b)] = state_vq.fit_norm_range(ln, b, sigma[m, j].numpy())
        print(f"[norms] wkv comp {j}: {[(b, tuple(round(x, 3) for x in books.wkv_norm[(j, b)])) for b in args.norm_bits]}")
    for role in ("t", "c"):
        n = np.linalg.norm(np.concatenate(shift[role]), axis=1)
        n = n[n > 1e-20]
        for b in args.norm_bits:
            books.shift_norm[(role, b)] = state_vq.fit_norm_range(np.log2(n), b)
        print(f"[norms] shift {role}: {[(b, tuple(round(x, 3) for x in books.shift_norm[(role, b)])) for b in args.norm_bits]}")
    books.save(OUT / args.codebooks)


def tune(args):
    from scripts.full_run_metrics import evaluate_with_predictions
    run = cc.run_info()
    counts = run["review_counts"]
    corpus = {int(f.stem) for f in (OUT / "corpus").glob("*.npz")}
    users = [u for u in training_users(args.users + 2, 1, counts, hi=args.max_reviews) if u not in corpus][:args.users]
    model, layout = load_student()
    books = state_vq.Codebooks.load(OUT / args.codebooks)
    data = cc.data_path(run)
    out = OUT / args.out
    for user in users:
        batch = cc.full_history_batch(data, user)
        for name in ["w8"] + args.recipes.split(","):
            t0 = time.perf_counter()
            if name == "w8":
                state_vq.configure(layout, None, 0)
            elif name == "w8+s8":
                state_vq.configure(layout, None, 8)
            else:
                recs = state_vq.recipe(name)
                state_vq.configure(layout, {s: state_vq.StateCodec(r, books) for s, r in zip(layout, recs)})
            with torch.inference_mode():
                values, _ = evaluate_with_predictions(model, [{"batch": batch}])
            row = {"user_id": user, "label": name, "reviews": counts[str(user)],
                   "seconds": round(time.perf_counter() - t0, 1), **values}
            if name.startswith("vq"):
                row["state_vq"] = state_vq.describe(name, layout)
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[tune] {user} {name} curve {values['curve_log_loss']:.5f} instant {values['instant_log_loss']:.5f} "
                  f"{row['seconds']}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--users", type=int, default=50)
    c.add_argument("--shard", default="0/1")
    c.add_argument("--per-user", type=int, default=600)
    f = sub.add_parser("fit")
    f.add_argument("--max-points", type=int, default=150_000)
    f.add_argument("--iters", type=int, default=20)
    f.add_argument("--wkv-stages", type=int, default=8)
    f.add_argument("--wkv-stages-2", type=int, default=4)
    f.add_argument("--shift-m", type=int, nargs="+", default=[2, 4])
    f.add_argument("--shift-stages", type=int, default=2)
    f.add_argument("--norm-bits", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    f.add_argument("--out", default="codebooks.npz")
    n = sub.add_parser("refit-norms")
    n.add_argument("--codebooks", default="codebooks.npz")
    n.add_argument("--max-points", type=int, default=150_000)
    n.add_argument("--norm-bits", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    t = sub.add_parser("tune")
    t.add_argument("--users", type=int, default=8)
    t.add_argument("--max-reviews", type=int, default=30000)
    t.add_argument("--recipes", required=True)
    t.add_argument("--codebooks", default="codebooks.npz")
    t.add_argument("--out", default="tune.jsonl")
    args = parser.parse_args()
    {"collect": collect, "fit": fit, "refit-norms": refit_norms, "tune": tune}[args.cmd](args)


if __name__ == "__main__":
    main()
