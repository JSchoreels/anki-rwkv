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
SM46: --student <master.pth> (its run's protocol.json gives the training users) and --out-dir <folder> select another
model and output folder (defaults: the SM32 student, artifacts/small-model/state-vq); --exclude-pool 732 drops any
evaluation-pool user from the corpus / tune users; fit takes rank-R stage lists (--wkv-stages 24 16 10 6) and per-m
shift stage counts (--shift-m 4 8 --shift-stages 4 6).
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

torch.set_num_threads(int(os.environ.get("VQ_FIT_THREADS", "2")))  # SM46: override for fit

from scripts import small_model_heldout as smh  # noqa: E402  (installs the state-quant fork model first)
from scripts import compress_common as cc  # noqa: E402
from scripts import compress_quant as cq  # noqa: E402
from scripts import state_vq  # noqa: E402

STUDENT = Path(r"C:\Users\jonat\workspace\wt\small-model\artifacts\small-model"
               r"\scale-w64kd-gate-cached-ctx16384-cont-decay-L32k\expertium\E89-F65e-F77a\master.pth")
PROTOCOL = STUDENT.parents[2] / "protocol.json"
OUT = ROOT / "artifacts" / "small-model" / "state-vq"


EXCLUDE = set()   # SM46 --exclude-pool: evaluation users never used for codebooks or tuning


def training_users(count, offset, counts, lo=3000, hi=60000):
    """`count` training users with lo..hi reviews, spread over log(history length); `offset` picks a disjoint set."""
    rows = json.loads(PROTOCOL.read_text())["rows"]
    users = sorted({r["user_id"] for r in rows if lo <= counts[str(r["user_id"])] <= hi and r["user_id"] not in EXCLUDE},
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


def corpus_files(holdout_every=0, held=False):
    """Corpus files in sorted order; holdout_every k > 0 (SM46): every k-th file (index % k == k - 1) is held out of
    the fit (held=True returns those instead)."""
    files = sorted((OUT / "corpus").glob("*.npz"))
    if not holdout_every:
        return [] if held else files
    return [f for i, f in enumerate(files) if (i % holdout_every == holdout_every - 1) == held]


def fit(args):
    files = corpus_files(args.holdout_every)
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
    wkv_stages = list(args.wkv_stages) if len(args.wkv_stages) > 1 else [args.wkv_stages[0], args.wkv_stages_2]
    R = len(wkv_stages)
    u, v, sigma, valid = state_vq.factorize(A, max(R, 2))
    energy = (A.double() ** 2).sum((1, 2))
    for r in range(1, max(R, 2) + 1):
        kept = (sigma[:, :r] ** 2 * valid[:, :r]).sum(1)
        print(f"[fit] rank-{r} truncation keeps {float(kept.sum() / energy.sum()):.4%} of the WKV energy", flush=True)
    for j, stages in enumerate(wkv_stages if "wkv" in args.parts else []):
        m = valid[:, j]
        X = torch.cat([u[m, j], v[m, j]], dim=-1).float()
        books.wkv.append(fit_rvq(X, stages, 10, args.iters, 100 * j, f"wkv comp {j}"))
        ln = torch.log2(torch.sqrt(sigma[m, j])).numpy()
        for b in args.norm_bits:
            books.wkv_norm[(j, b)] = state_vq.fit_norm_range(ln, b, sigma[m, j].numpy())
        print(f"[fit] wkv comp {j} log2 sqrt(sigma) quantiles 1/50/99%: "
              f"{np.quantile(ln, [0.01, 0.5, 0.99]).round(3)}; ranges {[(b, books.wkv_norm[(j, b)]) for b in args.norm_bits]}",
              flush=True)
    for role in (("t", "c") if "shift" in args.parts else ()):
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
        shift_stages = dict(zip(args.shift_m, args.shift_stages if len(args.shift_stages) > 1
                                else args.shift_stages * len(args.shift_m)))
        for m in args.shift_m:
            sub = X.shape[1] // m
            books.shift[(role, m)] = [fit_rvq(unit[:, p * sub:(p + 1) * sub].contiguous(), shift_stages[m], 12,
                                              args.iters, 1000 + 10 * p, f"shift {role} m{m} pos {p}")
                                      for p in range(m)]
    books.save(OUT / args.out)
    print(f"[fit] wrote {OUT / args.out}", flush=True)


def merge(args):
    """SM46: one codebook file from a WKV-only and a shift-only fit (fit --parts)."""
    wkv = state_vq.Codebooks.load(OUT / args.wkv)
    shift = state_vq.Codebooks.load(OUT / args.shift)
    books = state_vq.Codebooks(wkv=wkv.wkv, shift=shift.shift, wkv_norm=wkv.wkv_norm, shift_norm=shift.shift_norm)
    books.save(OUT / args.out)
    print(f"[merge] {OUT / args.out}: wkv stages {[len(s) for s in books.wkv]}, shift "
          f"{ {k: len(v[0]) for k, v in books.shift.items()} }, sha256 {state_vq.file_sha256(OUT / args.out)}", flush=True)


def refit_norms(args):
    """Recompute only the norm ranges of an existing codebook file from the corpus (no k-means)."""
    books = state_vq.Codebooks.load(OUT / args.codebooks)
    files = corpus_files(args.holdout_every)
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
    R = max(2, len(books.wkv))
    _, _, sigma, valid = state_vq.factorize(A, R)
    for j in range(R):
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
    index, total = (int(x) for x in args.shard.split("/"))   # SM46: users split over processes
    users = users[index::total]
    done = set()
    if (OUT / args.out).exists():   # SM46: resumable
        done = {(r["user_id"], r["label"]) for r in map(json.loads, (OUT / args.out).read_text().splitlines())}
    model, layout = load_student()
    books = state_vq.Codebooks.load(OUT / args.codebooks)
    data = cc.data_path(run)
    out = OUT / args.out
    for user in users:
        batch = cc.full_history_batch(data, user)
        for name in ["w8"] + args.recipes.split(","):
            if (user, name) in done:
                continue
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
    f.add_argument("--wkv-stages", type=int, nargs="+", default=[8], help="stages per rank component (one value: "
                   "component 0, with --wkv-stages-2 for component 1)")
    f.add_argument("--wkv-stages-2", type=int, default=4)
    f.add_argument("--shift-m", type=int, nargs="+", default=[2, 4])
    f.add_argument("--shift-stages", type=int, nargs="+", default=[2], help="one value, or one per --shift-m")
    f.add_argument("--norm-bits", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    f.add_argument("--out", default="codebooks.npz")
    f.add_argument("--parts", default="wkv,shift", help="SM46: fit only these parts (wkv / shift), for two "
                   "processes in parallel; merge the files with state_vq_fit.py merge")
    f.add_argument("--holdout-every", type=int, default=0, help="SM46: leave every k-th corpus user out of the fit")
    n = sub.add_parser("refit-norms")
    n.add_argument("--codebooks", default="codebooks.npz")
    n.add_argument("--max-points", type=int, default=150_000)
    n.add_argument("--norm-bits", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    n.add_argument("--holdout-every", type=int, default=0, help="SM46: as fit --holdout-every")
    mg = sub.add_parser("merge")
    mg.add_argument("--wkv", required=True)
    mg.add_argument("--shift", required=True)
    mg.add_argument("--out", required=True)
    t = sub.add_parser("tune")
    t.add_argument("--users", type=int, default=8)
    t.add_argument("--max-reviews", type=int, default=30000)
    t.add_argument("--recipes", required=True)
    t.add_argument("--codebooks", default="codebooks.npz")
    t.add_argument("--out", default="tune.jsonl")
    t.add_argument("--shard", default="0/1")
    parser.add_argument("--student", default="", help="SM46: model checkpoint (default: the SM32 student)")
    parser.add_argument("--out-dir", default="", help="SM46: corpus / codebook / tune folder (default: state-vq)")
    parser.add_argument("--exclude-pool", type=int, default=0, help="SM46: drop this evaluation pool's users")
    args = parser.parse_args()
    global STUDENT, PROTOCOL, OUT
    if args.student:
        STUDENT = Path(args.student)
        PROTOCOL = STUDENT.parents[2] / "protocol.json"
    if args.out_dir:
        OUT = Path(args.out_dir) if Path(args.out_dir).is_absolute() else ROOT / args.out_dir
    if args.exclude_pool:
        from scripts.eval_pools import pool_users
        EXCLUDE.update(pool_users(args.exclude_pool))
    print(f"[state_vq_fit] student {STUDENT}, out {OUT}, excluded users {len(EXCLUDE)}", flush=True)
    {"collect": collect, "fit": fit, "refit-norms": refit_norms, "merge": merge, "tune": tune}[args.cmd](args)


if __name__ == "__main__":
    main()
