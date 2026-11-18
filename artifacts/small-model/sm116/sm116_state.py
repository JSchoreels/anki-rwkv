"""Training-free state check (SM116): card / note state directions of the show-time retrain (SM92) and the loss cost of
keeping 70% of them, with no training (design: design.md in this folder).

Everything runs through the stock CPU scorer, scripts/small_model_heldout.py (w8+s8, show-time inputs `e2s-query`,
leak-free flags), whose CPU path sends every STORED card / note state through scripts/state_vq.Hooks:
  * WKV state: per review of a card / note the answer's output uses the float update; the stored state (what the next
    review of that entity starts from) is Hooks.wkv[(stream, layer)](nxt) -> today the int8 row rounding (+s8);
  * token-shift vectors: the vector a review reads from the entity's previous review is Hooks.shift[(stream, layer,
    role)](x) (identity today: w8+s8 keeps them float).
This file wraps those hooks; the scorer itself is unchanged.

  users      write users-directions.json (100 users of the 1,500 validation pool outside the 732 set, i.e. validation users
             with id >= 5000, unseen by the show-time retrain (SM92); seed 116)
  collect    SM116_STATS=<npz> + scorer args: also accumulate n, sum x, sum x x^T (float64) of every stored WKV state
             (per stream, layer, head: 256 = 16 x 16 values) and every read token-shift vector (per stream, layer,
             role: 64 values); the scorer runs unchanged otherwise
  basis      merge stats files -> per block centered PCA (eigh, float64): mean, eigenvalues, eigenvectors;
             writes work/basis.npz and directions.json (explained-variance curves)
  score      SM116_BASIS=<npz> SM116_KEEP=<fraction> + scorer args: the stored state is projected onto the block's
             first round(KEEP x D) directions before the stock rounding: WKV stored = s8(mu + P P^T (nxt - mu)),
             shift read = mu + P P^T (x - mu). KEEP 1.0 = the proof (P P^T = I in float64)

usage: python sm116_state.py users
       SM116_STATS=work/stats-a.npz python sm116_state.py collect <small_model_heldout.py args>
       python sm116_state.py basis work/stats-*.npz
       SM116_BASIS=work/basis.npz SM116_KEEP=0.7 python sm116_state.py score <small_model_heldout.py args>
"""
import atexit
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
WORK = HERE / "work"
N_DIRECTION_USERS = 100
SEED = 116
MAX_REVIEWS = 200_000   # direction users: <= 200k reviews each (CPU time and the 6 GB per-process RAM limit)
STREAM_NAMES = {0: "card", 2: "note"}


def block_name(key):
    if len(key) == 3 and isinstance(key[2], int):
        stream, layer, head = key
        return f"{STREAM_NAMES[stream]}.L{layer}.wkv.h{head}"
    stream, layer, role = key
    return f"{STREAM_NAMES[stream]}.L{layer}.shift.{role}"


def cmd_users(_argv):
    from scripts.eval_pools import pool_users
    from scripts import compress_common as cc
    counts = cc.run_info()["review_counts"]
    s72 = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s")
    eval_users = set(json.loads((s72 / "eval-users-732-fit.json").read_text())["users"])
    eval_users |= set(json.loads((s72 / "users-732-long.json").read_text())["users"])
    eval_users |= set(pool_users(732))
    candidates = sorted(u for u in set(pool_users(1500)) - eval_users if counts[str(u)] <= MAX_REVIEWS)
    users = sorted(random.Random(SEED).sample(candidates, N_DIRECTION_USERS))
    assert not set(users) & eval_users
    doc = {"note": "The training-free state check (SM116): the 100 users whose stored card / note states give the state "
                   "directions. Drawn with random.Random(116).sample from the 1,500 validation pool minus the 732 set and its "
                   f"3 long users (validation users with id >= 5000, unseen by the show-time retrain (SM92)), <= {MAX_REVIEWS} reviews each; disjoint from every scored user. Public benchmark user ids only.",
           "users_count": len(users), "candidates": len(candidates), "reviews": sum(counts[str(u)] for u in users),
           "users": users}
    WORK.mkdir(parents=True, exist_ok=True)
    (HERE / "users-directions.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{len(users)} users of {len(candidates)} candidates, {doc['reviews']:,} reviews")


# ----------------------------------------------------------------------------------------------- hooks

def _wrap_configure(on_wkv, on_shift):
    """Patch scripts.state_vq.configure: after the stock configure, every card / note WKV hook becomes
    stock(on_wkv(nxt)) and every shift hook stock(on_shift(x)) (stock shift = identity when absent)."""
    from scripts import state_vq
    original = state_vq.configure

    def configure(layout, codecs=None, int_bits=0, shift_bits=0):
        original(layout, codecs, int_bits, shift_bits)
        if codecs is not None:
            raise SystemExit("SM116 wraps the int / float state path only (no +vq)")
        H = state_vq.Hooks
        for stream, info in layout.items():
            for layer in range(info["wkv_layers"]):
                stock = H.wkv.get((stream, layer))
                H.wkv[(stream, layer)] = (lambda s, cur, rows, t, st=stock, k=(stream, layer):
                                          (lambda y: y if st is None else st(y, cur, rows, t))(on_wkv(s, k)))
            for layer, role in info["shift_vectors"]:
                stock = H.shift.get((stream, layer, role))
                H.shift[(stream, layer, role)] = (lambda x, cur, st=stock, k=(stream, layer, role):
                                                  (lambda y: y if st is None else st(y, cur))(on_shift(x, k)))
    state_vq.configure = configure


def _run_scorer(argv):
    from scripts import small_model_heldout
    sys.argv = ["small_model_heldout.py"] + argv
    if "--device" not in argv or argv[argv.index("--device") + 1] != "cpu":
        raise SystemExit("SM116 runs on the CPU scorer path only (--device cpu)")
    small_model_heldout.main()


def cmd_collect(argv):
    import torch
    out = Path(os.environ["SM116_STATS"])
    stats = {}

    def add(key, x):
        x = x.detach().double()
        n, s, ss = stats.get(key, (0, None, None))
        s1, ss1 = x.sum(0), x.T @ x
        stats[key] = (n + x.shape[0], s1 if s is None else s + s1, ss1 if ss is None else ss + ss1)

    def on_wkv(S, key):
        for h in range(S.shape[1]):
            add((*key, h), S[:, h].reshape(S.shape[0], -1))
        return S

    def on_shift(x, key):
        add(key, x)
        return x

    def save():
        if not stats:
            return
        keys = sorted(stats, key=block_name)
        np.savez(out, names=np.array([block_name(k) for k in keys]),
                 n=np.array([stats[k][0] for k in keys], dtype=np.int64),
                 **{f"sum_{block_name(k)}": stats[k][1].numpy() for k in keys},
                 **{f"ss_{block_name(k)}": stats[k][2].numpy() for k in keys})
        print(f"[sm116] stats of {len(keys)} blocks -> {out}", flush=True)

    atexit.register(save)
    torch.set_grad_enabled(False)
    _wrap_configure(on_wkv, on_shift)
    _run_scorer(argv)


def cmd_basis(paths):
    total = {}
    for p in paths:
        z = np.load(p)
        for i, name in enumerate(z["names"]):
            name = str(name)
            n, s, ss = total.get(name, (0, 0.0, 0.0))
            total[name] = (n + int(z["n"][i]), s + z[f"sum_{name}"], ss + z[f"ss_{name}"])
    arrays, report = {}, {}
    for name in sorted(total):
        n, s, ss = total[name]
        mu = s / n
        cov = ss / n - np.outer(mu, mu)
        cov = (cov + cov.T) / 2
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals, evecs = np.clip(evals[order], 0, None), evecs[:, order]
        arrays[f"mu_{name}"], arrays[f"evals_{name}"], arrays[f"evecs_{name}"] = mu, evals, evecs
        frac = np.cumsum(evals) / evals.sum()
        D = len(evals)
        keep70 = int(round(0.7 * D))
        report[name] = {"dims": D, "vectors": n, "total_variance": float(evals.sum()),
                        "mean_sq_norm": float(mu @ mu), "second_moment": float(np.trace(ss) / n),
                        "dims_for": {str(q): int(np.searchsorted(frac, q) + 1) for q in (0.5, 0.9, 0.95, 0.99, 0.999)},
                        "kept_at_70pct": keep70, "variance_kept_at_70pct": float(frac[keep70 - 1]),
                        "curve": [round(float(f), 6) for f in frac]}
    np.savez(WORK / "basis.npz", names=np.array(sorted(total)), **arrays)
    (HERE / "directions.json").write_text(json.dumps(report, indent=1) + "\n")
    for name, r in report.items():
        print(f"{name:24s} n={r['vectors']:>9,} dims {r['dims']:3d}  50/90/95/99/99.9%: "
              + "/".join(str(r['dims_for'][q]) for q in ('0.5', '0.9', '0.95', '0.99', '0.999'))
              + f"  kept {r['kept_at_70pct']} -> {100 * r['variance_kept_at_70pct']:.3f}% of variance")


def projector(basis, name, keep):
    import torch
    mu = torch.from_numpy(basis[f"mu_{name}"])
    V = torch.from_numpy(basis[f"evecs_{name}"])
    k = int(round(keep * V.shape[0]))
    P = V[:, :k]
    return mu, P @ P.T, k


def cmd_score(argv):
    import torch
    keep = float(os.environ["SM116_KEEP"])
    basis = np.load(os.environ["SM116_BASIS"])
    names = set(map(str, basis["names"]))
    tags = [a.split("=", 1)[1].split("/")[0] for a in argv if a.startswith("--only=")] + \
           [argv[i + 1].split("/")[0] for i, a in enumerate(argv) if a == "--only"]
    if not tags or not all(t.startswith("sm116") for t in tags):
        raise SystemExit("score: every --only label must use a tag starting with sm116 (the projected model)")
    proj = {}

    def get(key):
        name = block_name(key)
        if name not in proj:
            if name not in names:
                raise SystemExit(f"no basis for block {name}")
            proj[name] = projector(basis, name, keep)
            print(f"[sm116] {name}: keep {proj[name][2]} of {proj[name][1].shape[0]} dims", flush=True)
        return proj[name]

    def project(x, key):
        mu, Q, _ = get(key)
        xd = x.double()
        return (mu + (xd - mu) @ Q).to(x.dtype)

    def on_wkv(S, key):
        N, H, K, _ = S.shape
        out = torch.empty_like(S)
        for h in range(H):
            out[:, h] = project(S[:, h].reshape(N, K * K), (*key, h)).reshape(N, K, K)
        return out

    def on_shift(x, key):
        return project(x, key)

    _wrap_configure(on_wkv, on_shift)
    _run_scorer(argv)


if __name__ == "__main__":
    command, rest = sys.argv[1], sys.argv[2:]
    {"users": cmd_users, "collect": cmd_collect, "basis": cmd_basis, "score": cmd_score}[command](rest)
