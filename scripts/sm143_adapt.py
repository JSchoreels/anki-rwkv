"""Per-user adaptation (SM143): does adapting an instant-only model to each user from that user's own earlier reviews
improve the later reviews' LogLoss and within-user AUC? Design: artifacts/small-model/sm143/design.md (pre-registered).

Per user (show-time inputs e2s-query, leak-free --deleted-cards neighbour, fp32, the scorer's batch and seed):
  1. one full-history forward of the base model (the scorer's path, scripts/small_model_heldout.py), with a hook that
     keeps the trunk output x (the input of prehead_norm) of every row; the "none" row is the scorer's row bit for bit
     (instant_log_loss and friends + "benchmark" from heldout_fast_metrics.metrics_arrays).
  2. rolling time split = the benchmark's own: the 5 TimeSeriesSplit test folds of the published FSRS-7 protocol,
     read from the scheduling-time vs review-time accuracy (SM81) per-user files (review_th, fold, y, p_fsrs7_fit: FSRS-7
     fitted per user on the reviews before each fold). For fold k: fit set = the model's labelled instant query rows with
     review_th < start_k (the fold's first review_th; revlog rows are in time order, day_offset non-decreasing), the
     last 10% of them (>= 30) a causal validation slice, the rest the training part; evaluation = fold k's FSRS-7 test
     reviews that are also benchmark (label-DB) reviews with an instant prediction (SM81's common-row rule).
  3. variants, each refit per fold from the base weights, checkpoint chosen on the validation slice (step 0 = base is
     always a candidate):
       a  Platt control: p' = sigmoid(exp(s) * logit(p) + b), 2 parameters (monotone: cannot change AUC)
       b  user vector: u (d_model values) added to the normalised pre-head features, 64 parameters
       c  instant head: prehead_norm + head_p + p_linear fine-tuned with L2 to the base values (8,964 parameters, Small)
       d  full fine-tune: every weight, Adam on the training part's 128-review windows (the training protocol), L2 to
          the base weights; each checkpoint scored by a full-history forward (expensive: a subset of users)
  4. per user: review-weighted LogLoss sums, AUC and ECE per variant + FSRS-7 on the evaluation reviews, per fold and
     pooled; per-review arrays to --save DIR/<user>.npz (not committed).

Checks recorded per user: head_identity_max_abs (the head recomputed on the kept x with the base weights vs the
forward's p_imm: 0 expected), causal (every fit row's review_th < start_k and day_offset <= the fold start's; the
(d) windows likewise), y_equal (FSRS-7 y == the model's label on every evaluation review), d_identity_max_abs (d's
scoring forward at step 0 vs the base forward). --steps-scale 0 runs every variant with 0 steps (the CPU proof:
every variant's evaluation predictions must equal the base ones bit for bit).

usage: sm143_adapt.py --model TAG=expertium:ARCH:PATH --users-file F --fsrs DIR --out rows.jsonl --save DIR
       [--variants a,b,c] [--d-users-file F2] [--device cpu] [--steps-scale 1] [--time-limit S]
       sm143_adapt.py selftest     (the causality assertion trips on a leaking fit set)
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RWKV_NO_JIT", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUPY_CACHE_DIR", str(Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/cupy-cache")))

import numpy as np  # noqa: E402

# Pre-registered hyperparameters (design.md "Variants"); --steps-scale multiplies every step count (0 = proof)
HP = {
    "a": {"lr": 1e-2, "steps": 100, "every": 10, "l2": 1e-3},
    "b": {"lr": 1e-2, "steps": 100, "every": 10, "l2": 1e-3},
    "c": {"lr": 1e-3, "steps": 100, "every": 10, "l2": 1e-3},
    "d": {"lr": 1e-4, "steps": 30, "every": 15, "l2": 1e-3, "windows": 64, "window": 128},
}
MIN_FIT = 200          # labelled instant query rows before the fold; fewer -> no adaptation (base kept, "skipped")
VAL_SHARE, MIN_VAL = 0.10, 30
ECE_BINS = 10


class CausalityError(AssertionError):
    pass


def assert_causal(fit_ths, start_th, day_offset, what):
    """No fit row at or after the evaluation block's first review (revlog rows are in time order: review_th is the
    row number, day_offset non-decreasing, asserted by get_rwkv_data)."""
    fit_ths = np.asarray(fit_ths, dtype=np.int64)
    if len(fit_ths) == 0:
        return {"n": 0}
    if fit_ths.max() >= start_th:
        raise CausalityError(f"{what}: fit review_th {int(fit_ths.max())} >= fold start {start_th}")
    if day_offset[fit_ths - 1].max() > day_offset[start_th - 1]:
        raise CausalityError(f"{what}: fit day_offset {int(day_offset[fit_ths - 1].max())} > fold start day "
                             f"{int(day_offset[start_th - 1])}")
    return {"n": int(len(fit_ths)), "max_th": int(fit_ths.max()), "start_th": int(start_th)}


def log_loss_sum(y, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1 - 1e-15)   # sklearn's clipping
    y = np.asarray(y, dtype=np.float64)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).sum())


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if y.min() == y.max():
        return None
    return float(roc_auc_score(y, np.asarray(p, dtype=np.float64)))


def ece(y, p):
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    b = np.minimum((p * ECE_BINS).astype(int), ECE_BINS - 1)
    sy, sp = np.bincount(b, y, ECE_BINS), np.bincount(b, p, ECE_BINS)
    return float(np.abs(sy - sp).sum() / max(len(p), 1))


def summary(y, p):
    return {"n": int(len(y)), "ll_sum": log_loss_sum(y, p), "auc": auc(y, p), "ece": ece(y, p)}


def selftest():
    day = np.array([0, 0, 1, 1, 2, 3])
    assert assert_causal([1, 2, 3], 4, day, "ok")["max_th"] == 3
    for ths, start in (([1, 4], 4), ([1, 5], 4)):
        try:
            assert_causal(ths, start, day, "leak")
        except CausalityError:
            continue
        raise SystemExit("selftest FAILED: a leaking fit set passed")
    try:   # same review_th order but a later day (a corrupted order) also trips
        assert_causal([3], 2, np.array([0, 0, 5]), "day")
    except CausalityError:
        print("sm143 selftest OK")
        return
    raise SystemExit("selftest FAILED: day check")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="TAG=expertium:ARCH:PATH")
    ap.add_argument("--users-file", required=True)
    ap.add_argument("--fsrs", required=True, help="SM81 per-user FSRS files (<user>.npz)")
    ap.add_argument("--variants", default="a,b,c", help="subset of a,b,c,d")
    ap.add_argument("--d-users-file", default="", help="run variant d only for these users (others: a,b,c)")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save", default="", help="per-user evaluation arrays DIR/<user>.npz")
    ap.add_argument("--steps-scale", type=float, default=1.0)
    ap.add_argument("--time-limit", type=float, default=1e9)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from scripts import small_model_heldout as H
    from scripts import compress_common as cc
    from scripts import compress_quant as cq
    from scripts import state_vq, sm31_inputs
    from scripts.compress_metrics import read_labels
    from scripts.heldout_fast_metrics import collect_arrays, metrics_arrays
    from scripts.experiment_provenance import process_provenance
    from expertium.model import rwkv_model as erm
    import lmdb

    cpu = args.device == "cpu"
    device = torch.device(args.device)
    variants = [v for v in args.variants.split(",") if v]
    assert set(variants) <= set(HP), variants
    if cpu:
        torch.set_num_threads(args.threads)
    else:
        from scripts import compress_wkv_cuda, small_model_wkv
        compress_wkv_cuda.register_cuda()
        errors = {**{f"fork/bits{b}": e for b, e in compress_wkv_cuda.self_test().items()}, **small_model_wkv.self_test()}
        if max(e for key, e in errors.items() if key.endswith("0")) > 1e-4:
            raise SystemExit("CUDA recurrence disagrees with the reference")
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("HELDOUT_GPU_MEMORY_FRACTION", "0.8")))
        if "d" in variants:
            from rwkv.model import nvrtc
            nvrtc.register()
    from expertium.model.rwkv_ops import RWKV7_WKV as TRAIN_WKV

    tag, _, rest = args.model.partition("=")
    kind, arch, path = rest.split(":", 2)
    assert kind == "expertium", "adaptation needs the expertium head layout"
    label = f"{tag}/fp32@e2s-query"
    base_state = {k: v.float() for k, v in torch.load(path, map_location="cpu", weights_only=True).items()}
    model = H.build_expertium(arch, base_state)
    layout = state_vq.install(model) if cpu else state_vq.install_scoring(model)
    model = model.to(device)
    names = cq.linear_weight_names(base_state)
    state, _, n_params = (cq.quantize_state_dict(base_state, names, *cq.WEIGHT_SCHEMES["fp32"]),
                          None, sum(t.numel() for t in base_state.values()))
    model.load_state_dict(state)
    H.set_expertium_state_bits(model, 0)    # the scorer's fp32 configuration (score_user, int state_bits 0)
    if cpu:
        state_vq.configure(layout, None, 0)
    else:
        state_vq.set_scoring(layout, None)
    base_params = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for p in model.parameters():   # only variant d turns gradients on (and off again)
        p.requires_grad_(False)
    cap = {}
    model.prehead_norm.register_forward_hook(lambda mod, inp, out: cap.__setitem__("x", inp[0]))

    run = cc.run_info()
    counts = run["review_counts"]
    data = cc.data_path(run)
    wanted = json.loads(Path(args.users_file).read_text())["users"]
    users = [u for u in H.heldout_users(counts, 732) if u in set(wanted)]
    d_users = set(json.loads(Path(args.d_users_file).read_text())["users"]) if args.d_users_file else None
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "error" not in r and r.get("label") == label:
                done.add((r["user_id"], tuple(r["variants"])))
    save = Path(args.save) if args.save else None
    if save:
        save.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(cc.RUN / "label-db"), readonly=True, lock=False, map_size=10_000_000_000)
    prov = process_provenance()

    def sync():
        if not cpu:
            torch.cuda.synchronize()

    def head_prob(x, s=None, b=None, u=None, parts=None):
        """p_binary of the instant head on x exactly as VariantSrsRWKV (eval: dropout off). parts = (norm, head_p,
        p_linear) modules (variant c) or None (the model's own)."""
        norm, hp, pl = parts or (model.prehead_norm, model.head_p, model.p_linear)
        xn = model.prehead_dropout(norm(x))
        if u is not None:
            xn = xn + u
        probs = torch.softmax(pl(hp(xn).float()), dim=-1)
        p = torch.clamp(1.0 - probs.unbind(dim=-1)[0], min=1e-5, max=1.0 - 1e-5)
        if s is not None:
            z = torch.log(p / (1 - p))
            p = torch.sigmoid(torch.exp(s) * z + b)
        return p

    def forward(batch):
        """The scorer's evaluate_with_arrays on one batch, also returning stats and the kept x."""
        with torch.inference_mode():
            stats = model.get_loss(batch)
            if stats is None or not torch.isfinite(stats.average_loss):
                raise RuntimeError("Non-finite evaluation loss")
            mask = stats.is_query.bool() & stats.has_label.bool()
            losses = F.binary_cross_entropy(stats.p_imm.clamp(1e-5, 1 - 1e-5), batch.labels[:, :, 2], reduction="none")
            values = {"curve_labels": stats.ahead_n, "instant_labels": stats.imm_n,
                      "curve_log_loss": stats.ahead_avg.item(), "curve_raw_log_loss": stats.ahead_raw_avg.item(),
                      "instant_log_loss": losses[mask].sum().item() / stats.imm_n,
                      "instant_rating_cross_entropy": stats.imm_avg.item()}
            predictions = collect_arrays(stats)
        x = cap.pop("x").clone()
        return stats, values, predictions, x

    def predictions_with(stats, p_full):
        class S:
            pass
        s = S()
        for f in ("has_label", "is_query", "label_review_th", "label_rating", "p_curve"):
            setattr(s, f, getattr(stats, f))
        s.p_imm = p_full
        return collect_arrays(s)

    def adam_fit(params, loss_fn, val_fn, hp, steps, extra=None):
        """Full-batch Adam; validation every hp['every'] steps; returns (best step, val losses, best params)."""
        opt = torch.optim.Adam(params, lr=hp["lr"])
        best = (val_fn(), 0, [p.detach().clone() for p in params])
        losses = [round(best[0], 6)]
        for step in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            loss_fn().backward()
            opt.step()
            if step % hp["every"] == 0 or step == steps:
                v = val_fn()
                losses.append(round(v, 6))
                if v < best[0]:
                    best = (v, step, [p.detach().clone() for p in params])
        return best[1], losses, best[2]

    def score_full(state_dict):
        """Full-history scoring forward with these weights (scorer WKV path), p_imm [B, T]."""
        model.load_state_dict(state_dict)
        with torch.inference_mode():
            st = model.get_loss(gpu_batch)
            if st is None or not torch.isfinite(st.average_loss):
                raise RuntimeError("Non-finite evaluation loss (d)")
            p = st.p_imm.clone()
        cap.pop("x", None)
        return p

    def windows_batch(frame, user, fold, step, rng, samples):
        from rwkv.prepare_batch import prepare
        n = HP["d"]["windows"]
        pick = samples if len(samples) <= n else [samples[i] for i in sorted(rng.choice(len(samples), n, replace=False))]
        return prepare(pick, seed=143_000_000 + user * 100 + fold * 10 + step).to(device)

    started = time.perf_counter()
    print(f"[sm143] {label}: {len(users)} users, variants {variants}, d users {len(d_users) if d_users else 'all'}, "
          f"steps x{args.steps_scale}", flush=True)
    for index, user in enumerate(users):
        user_variants = [v for v in variants if v != "d" or d_users is None or user in d_users]
        if (user, tuple(user_variants)) in done:
            continue
        if time.perf_counter() - started > args.time_limit:
            print(f"[sm143] time limit after {index} users", flush=True)
            break
        t_user = time.perf_counter()
        try:
            uid, batch, error, prep = H.LeakBatches([user], data, counts, "neighbour", 0, "e2s-query")[0]
            if batch is None:
                raise RuntimeError(error)
            with env.begin() as txn:
                labels = read_labels(txn, user)
            model.load_state_dict(base_params)
            gpu_batch = batch.to(device)
            t0 = time.perf_counter()
            stats, values, predictions, x = forward(gpu_batch)
            sync()
            forward_seconds = time.perf_counter() - t0
            row = {"user_id": user, "label": label, "tag": tag, "arch": arch, "checkpoint": path, "parameters": n_params,
                   "reviews": counts[str(user)], "variants": user_variants, "inputs": "e2s-query",
                   "deleted_cards": "neighbour", **values, "benchmark": metrics_arrays(user, labels, predictions),
                   "forward_seconds": round(forward_seconds, 3), "prepare_seconds": round(prep, 2)}
            p_base = stats.p_imm
            with torch.inference_mode():
                p_head = head_prob(x)
            row["head_identity_max_abs"] = float((p_head - p_base).abs().max().item())
            # labelled instant query rows (positions in the flat [B*T] layout)
            mask = (stats.is_query.bool() & stats.has_label.bool()).reshape(-1)
            pos = torch.nonzero(mask).reshape(-1)
            q_th = stats.label_review_th.reshape(-1)[pos].cpu().numpy().astype(np.int64)
            q_y = gpu_batch.labels[:, :, 2].reshape(-1)[pos]
            x_flat = x.reshape(-1, x.shape[-1])
            x_q = x_flat[pos].clone()
            pb_q = p_base.reshape(-1)[pos].clone()
            q_y = q_y.clone()
            # FSRS-7 folds (SM81) and the evaluation rows
            f = np.load(Path(args.fsrs) / f"{user}.npz")
            ths_lab = np.asarray(labels[0], dtype=np.int64) if labels else np.zeros(0, dtype=np.int64)
            has_i, _ = predictions.lookup("instant", f["review_th"])
            has_r, y_r = predictions.lookup("rating", f["review_th"])
            keep = np.isin(f["review_th"], ths_lab) & has_i & has_r
            ev_th, ev_fold = f["review_th"][keep].astype(np.int64), f["fold"][keep].astype(np.int64)
            ev_y = np.clip(y_r[keep], 0, 1)
            row["y_equal"] = bool(np.array_equal(ev_y.astype(np.int64), f["y"][keep].astype(np.int64)))
            row["eval_common"] = {"fsrs_test": int(len(f["review_th"])), "common": int(keep.sum())}
            if not row["y_equal"]:
                raise RuntimeError("FSRS-7 y differs from the model's labels on common reviews")
            frame_day = None
            import pandas as pd
            frame_day = pd.read_parquet(data / "revlogs" / f"user_id={user}", columns=["day_offset"])["day_offset"].to_numpy()
            assert np.all(np.diff(frame_day) >= 0), "day_offset not monotonic"
            _, pb_ev = predictions.lookup("instant", ev_th)
            preds = {"none": pb_ev.copy(), "fsrs7": f["p_fsrs7_fit"][keep].astype(np.float64)}
            for v in user_variants:
                preds[v] = pb_ev.copy()
            folds, causal = [], []
            frame_d, d_samples_all = None, None
            for k in range(5):
                in_fold = ev_fold == k
                fold_ths = f["review_th"][f["fold"] == k]
                if len(fold_ths) == 0:
                    continue
                start = int(fold_ths.min())
                fit = np.flatnonzero(q_th < start)
                causal.append(assert_causal(q_th[fit], start, frame_day, f"user {user} fold {k}"))
                fit = fit[np.argsort(q_th[fit], kind="stable")]
                info = {"fold": k, "start_th": start, "n_fit": int(len(fit)), "n_eval": int(in_fold.sum()), "variants": {}}
                folds.append(info)
                if len(fit) < MIN_FIT or not in_fold.any():
                    info["skipped"] = f"fit rows {len(fit)} < {MIN_FIT}" if len(fit) < MIN_FIT else "no evaluation rows"
                    continue
                n_val = max(MIN_VAL, int(round(VAL_SHARE * len(fit))))
                tr, va = torch.as_tensor(fit[:-n_val], device=device), torch.as_tensor(fit[-n_val:], device=device)
                info["n_val"], info["val_start_th"] = int(n_val), int(q_th[fit[-n_val]])
                for v in user_variants:
                    hp = HP[v]
                    steps = int(round(hp["steps"] * args.steps_scale))
                    t1 = time.perf_counter()
                    if v in ("a", "b", "c"):
                        xt, yt, xv, yv = x_q[tr], q_y[tr], x_q[va], q_y[va]
                        pbt, pbv = pb_q[tr], pb_q[va]
                        if v == "a":
                            s = torch.zeros((), device=device, requires_grad=True)
                            b = torch.zeros((), device=device, requires_grad=True)
                            params = [s, b]

                            def plat(p, s=s, b=b):
                                return torch.sigmoid(torch.exp(s) * torch.log(p / (1 - p)) + b)
                            loss_fn = lambda: F.binary_cross_entropy(plat(pbt), yt) + hp["l2"] * (s * s + b * b)  # noqa: E731

                            def val_fn():
                                with torch.no_grad():
                                    return float(F.binary_cross_entropy(plat(pbv), yv))
                        elif v == "b":
                            u = torch.zeros(x.shape[-1], device=device, requires_grad=True)
                            params = [u]
                            loss_fn = lambda: F.binary_cross_entropy(head_prob(xt, u=u), yt) + hp["l2"] * (u * u).sum()  # noqa: E731

                            def val_fn():
                                with torch.no_grad():
                                    return float(F.binary_cross_entropy(head_prob(xv, u=u), yv))
                        else:
                            parts = tuple(copy.deepcopy(m) for m in (model.prehead_norm, model.head_p, model.p_linear))
                            params = [p for m in parts for p in m.parameters()]
                            for p in params:
                                p.requires_grad_(True)
                            init = [p.detach().clone() for p in params]
                            loss_fn = lambda: (F.binary_cross_entropy(head_prob(xt, parts=parts), yt)  # noqa: E731
                                               + hp["l2"] * sum(((p - q) ** 2).sum() for p, q in zip(params, init)))

                            def val_fn():
                                with torch.no_grad():
                                    return float(F.binary_cross_entropy(head_prob(xv, parts=parts), yv))
                        with torch.enable_grad():
                            step, vlosses, best = adam_fit(params, loss_fn, val_fn, hp, steps)
                        if step > 0:
                            with torch.no_grad():
                                if v == "a":
                                    p_full = head_prob(x, s=best[0], b=best[1])
                                elif v == "b":
                                    p_full = head_prob(x, u=best[0])
                                else:
                                    for p, q in zip(params, best):
                                        p.data.copy_(q)
                                    p_full = head_prob(x, parts=parts)
                            _, pk = predictions_with(stats, p_full).lookup("instant", ev_th[in_fold])
                            preds[v][in_fold] = pk
                        sync()
                        info["variants"][v] = {"step": step, "val": vlosses, "seconds": round(time.perf_counter() - t1, 3)}
                    else:   # d: full fine-tune on the training part's 128-review windows
                        if frame_d is None:
                            from scripts import sm71_leak
                            frame_d = sm31_inputs.load_frame(data, user, "e2s-query")
                            frame_d, _ = sm71_leak.neutralise(frame_d, "neighbour", data, user)
                            frame_d = sm31_inputs.apply_frame(frame_d, "e2s-query")
                        cut = int(q_th[fit[-n_val]])   # first validation review: windows end before it
                        part = frame_d[frame_d["review_th"].to_numpy() < cut]
                        causal.append(assert_causal(part["review_th"].to_numpy(), start, frame_day,
                                                    f"user {user} fold {k} d windows"))
                        with sm31_inputs.query_patch("e2s-query"):
                            samples = cc.make_samples(part, user, hp["window"])
                        rng = np.random.default_rng([143, user, k])
                        model.load_state_dict(base_params)
                        params = [p for p in model.parameters()]
                        init = [p.detach().clone() for p in params]
                        opt = torch.optim.Adam(params, lr=hp["lr"])
                        va_np = fit[-n_val:]
                        yv_np = q_y[va].detach().cpu().numpy().astype(np.float64)
                        pv0 = pb_q[va].detach().cpu().numpy().astype(np.float64)
                        cands = [(log_loss_sum(yv_np, pv0) / len(yv_np), 0, None)]
                        vlosses = [round(cands[0][0], 6)]
                        train_seconds = 0.0
                        for step in range(1, steps + 1):
                            ts = time.perf_counter()
                            wb = windows_batch(frame_d, user, k, step, rng, samples)
                            if not cpu:
                                erm.RWKV7_WKV = TRAIN_WKV
                            try:
                                for p in params:
                                    p.requires_grad_(True)
                                with torch.enable_grad():
                                    st = model.get_loss(wb)
                                    xw = cap.pop("x")
                                    mw = st.is_query.bool() & st.has_label.bool()
                                    pw = head_prob(xw)
                                    loss = F.binary_cross_entropy(pw[mw], wb.labels[:, :, 2][mw])
                                    loss = loss + hp["l2"] * sum(((p - q) ** 2).sum() for p, q in zip(params, init))
                                    opt.zero_grad(set_to_none=True)
                                    loss.backward()
                                    opt.step()
                            finally:
                                if not cpu:
                                    erm.RWKV7_WKV = H.LongWKV
                                for p in params:
                                    p.requires_grad_(False)
                            sync()
                            train_seconds += time.perf_counter() - ts
                            if step % hp["every"] == 0 or step == steps:
                                snap = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
                                p_full = score_full(snap)
                                pv = p_full.reshape(-1)[pos[va]].cpu().numpy().astype(np.float64)
                                vl = log_loss_sum(yv_np, pv) / len(yv_np)
                                vlosses.append(round(vl, 6))
                                cands.append((vl, step, p_full))
                                model.load_state_dict(snap)
                        if k == 0 or steps == 0:   # the d scoring path at step 0 = the base forward
                            p0 = score_full(base_params)
                            row["d_identity_max_abs"] = max(row.get("d_identity_max_abs", 0.0),
                                                            float((p0 - p_base).abs().max().item()))
                        best = min(cands, key=lambda c: (c[0], c[1]))
                        if best[1] > 0:
                            _, pk = predictions_with(stats, best[2]).lookup("instant", ev_th[in_fold])
                            preds[v][in_fold] = pk
                        model.load_state_dict(base_params)
                        del opt
                        sync()
                        info["variants"][v] = {"step": best[1], "val": vlosses, "windows": len(samples),
                                               "train_seconds": round(train_seconds, 3),
                                               "seconds": round(time.perf_counter() - t1, 3)}
            row["causal"] = causal
            row["folds"] = folds
            row["metrics"] = {name: summary(ev_y, p) for name, p in preds.items()}
            row["fold_metrics"] = {str(k): {name: summary(ev_y[ev_fold == k], p[ev_fold == k]) for name, p in preds.items()}
                                   for k in range(5) if (ev_fold == k).any()}
            row["zero_step_equal"] = {v: bool(np.array_equal(preds[v], preds["none"])) for v in user_variants}
            row["seconds"] = round(time.perf_counter() - t_user, 2)
            if not cpu:
                row["gpu_max_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
            else:
                import psutil
                row["peak_wset_mib"] = round(getattr(psutil.Process().memory_info(), "peak_wset", 0) / 2 ** 20, 1)
            if save:
                np.savez_compressed(save / f"{user}.npz", review_th=ev_th.astype(np.int32), fold=ev_fold.astype(np.int8),
                                    y=ev_y.astype(np.int8), **{f"p_{n}": p.astype(np.float32) for n, p in preds.items()})
            del gpu_batch, x, x_q, stats
        except CausalityError:
            raise
        except (torch.OutOfMemoryError, RuntimeError) as err:
            if not cpu:
                torch.cuda.empty_cache()
            row = {"user_id": user, "label": label, "error": str(err)[:300], "variants": user_variants}
            model.load_state_dict(base_params)
            if not cpu:
                erm.RWKV7_WKV = H.LongWKV
        row["provenance_commit"] = prov["git_commit"]
        row["provenance_dirty"] = prov["git_dirty"]
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        el = time.perf_counter() - started
        print(f"[sm143] {index + 1}/{len(users)} user {user} ({counts[str(user)]} reviews) "
              f"{'ERROR ' + row['error'][:80] if 'error' in row else 'ok'} {row.get('seconds', 0)}s | {el:.0f}s", flush=True)
    env.close()


if __name__ == "__main__":
    main()
