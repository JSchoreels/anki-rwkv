#!/usr/bin/env python
"""Frozen-model grade side head (SM87): a small side head on a FROZEN copy of the released small model (the leak-free
small model (SM72s) 25 epochs + the 1-epoch 32k Muon cool-down) that corrects the released instant (grade) logits.
The trunk, the curve and the released instant head are untouched: corrected instant logits = released instant logits +
side head(inputs), the side head's output Linear zero-initialised, so it starts exactly at the released prediction and
can only add a correction. Nothing of the released model changes (its master.pth is read only; the side head is a
separate file).

Inputs at each instant (query) row, concatenated (494 values):
  frozen trunk features (448), as the answer-time head on richer frozen features (SM82c): the canonical residual
    stream (d 64) after the input encoder and after each of the 4 layer rounds, layer-normalised (5 x 64), and the
    instant head's hidden layer (head_p, 128) (scripts/sm82c_head.py Frozen / Capture).
  the released instant prediction (5): its 4 log-probabilities / 5 and logit(P(pass)) / 5.
  explicit causal grade features (41, `explicit_features`), from the reviews BEFORE the predicted one plus its own card
    id, elapsed days and elapsed seconds (the frame's columns the released model's interval input is built from, so the
    side head sees the same elapsed definition, incl. the same-day answer-time content of elapsed seconds (the
    duration-leak item (SM31)), as the released instant head): the card's last 3 grades (one-hot + "none") and ln answer
    times (+ "none"), ln(1 + the card's review count), ln(1 + its lapses = earlier Again grades), ln(1 + the elapsed
    days of the card's previous review), ln(1 + elapsed days), ln(1 + elapsed seconds), a same-day flag, the user's
    pass rate over the last 50 / 500 / all reviews (smoothed), ln(1 + the user's review index), the user's grade mix over
    the last 500 reviews, the user's running median ln time per grade 1-4 and over all grades, and an exponential
    average of the user's ln times (the last three from scripts/sm82c_head.explicit_features, CPU proof C4b there).
Side head: Linear 494 -> 128, ReLU, Linear 128 -> 4 (zero init), 63,876 values. Ablation without the model states
(same pass): the released prediction + the explicit features, Linear 46 -> 128 -> 4 (6,532 values).
Loss: the scorer's instant metric, the binary LogLoss of P(pass) = clamp(1 - softmax(corrected)[Again], 1e-5).

Training: the SM82c plan (1,991 users of the SM82 quarter-epoch cool-down, none in the 732 pool, each history's first
400k reviews) through the frozen model (fp32, leak-free inputs, no grad, every trunk parameter requires_grad False),
every labelled instant row (all non-first reviews of a card), one pass through a 1M-row shuffle buffer, minibatches of
8,192 rows, AdamW lr 2e-3 (2% warm-up, cosine to 0), no weight decay, the two side heads only.

Scoring (`score`): the scorer's own path (scripts/small_model_heldout.py: w8+s8 weights and state, leak-free inputs,
the same model construction and metrics code), one forward per user giving the released rows (bitwise the scorer's)
and, from the same captured features, the corrected and the ablation rows (the instant probabilities replaced, the curve
untouched). On --apply-check users a second forward applies the correction inside the model (a p_linear forward hook)
and checks the curve outputs stay bitwise the released ones.

  plan        --source P --out P2                    the SM82c plan with this script's labelled-row counts.
  train       --released M --plan P --out DIR        the side head training (GPU, or CPU for the proof); resumable.
  score       --released M --heads H --out R ...     the scorer (released + corrected + ablation rows).
  check       --released M --plan P --out report     the CPU proof.
  decide      --rows R ...                           the rule -> artifacts/small-model/sm87/sm87-decision.{md,json}.
"""
from argparse import ArgumentParser
import json
import math
import os
import subprocess
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sm82b_head import atomic_json, sha256, data_path, lr_at, first, GRADE_FIELDS, U732, SOLD, SLBL   # noqa: E402

HERE = ROOT / "artifacts/small-model/sm87"
NAME = "frozen-model grade side head (SM87)"
FROZEN_DIM = 448                # SM82c: 5 x 64 layer-normalised residual snapshots + head_p (128)
RELEASED_DIM = 5                # the released instant prediction: 4 log-probabilities / 5, logit(P(pass)) / 5
EXPLICIT_DIM = 41
INPUTS = FROZEN_DIM + RELEASED_DIM + EXPLICIT_DIM
ABL_INPUTS = RELEASED_DIM + EXPLICIT_DIM
HIDDEN = 128
LN8 = math.log(8.0)
VARIANT = "w8+s8"
TAGS = {"released": "released", "model": "sm87", "ablation": "sm87-ablation"}
EXPLICIT_NAMES = ([f"card_prev{k}_grade_{g}" for k in (1, 2, 3) for g in (1, 2, 3, 4)] + [f"card_prev{k}_none" for k in (1, 2, 3)]
                  + [f"card_prev{k}_ln_t" for k in (1, 2, 3)] + [f"card_prev{k}_t_none" for k in (1, 2, 3)]
                  + ["card_reviews_ln1p", "card_lapses_ln1p", "card_prev_elapsed_days_ln1p", "elapsed_days_ln1p",
                     "elapsed_seconds_ln1p", "same_day", "user_pass_50", "user_pass_500", "user_pass_all", "user_index_ln1p"]
                  + [f"user_mix500_g{g}" for g in (1, 2, 3, 4)] + [f"user_median_ln_t_g{g}" for g in (1, 2, 3, 4)]
                  + ["user_median_ln_t_all", "user_ema_ln_t"])
assert len(EXPLICIT_NAMES) == EXPLICIT_DIM


# ------------------------------------------------------------------ explicit causal features
def _window_sum(cs, i, w):
    """sum of the w values before position i from the exclusive cumulative sum cs (len n + 1), and the count."""
    lo = np.maximum(i - w, 0)
    return cs[i] - cs[lo], i - lo


def explicit_features(card, rating, duration, elapsed_days, elapsed_seconds):
    """[n, 41] float32: row i = the features for predicting review i (review_th i + 1) from reviews 0..i-1 and review
    i's card id, elapsed days and elapsed seconds only (CPU proof C4b perturbs review j and checks rows 0..j)."""
    from scripts.sm82c_head import explicit_features as sm82c_features
    card = np.asarray(card, np.int64); rating = np.asarray(rating, np.int64)
    duration = np.asarray(duration, np.int64)
    ed = np.asarray(elapsed_days, np.float64); es = np.asarray(elapsed_seconds, np.float64)
    n = len(card)
    out = np.zeros((n, EXPLICIT_DIM), np.float64)
    if n == 0:
        return out.astype(np.float32)
    idx = np.arange(n)
    # the card's earlier reviews: k-th previous review of the same card (stable sort keeps the time order)
    order = np.argsort(card, kind="stable")
    sc = card[order]
    start = np.ones(n, bool); start[1:] = sc[1:] != sc[:-1]
    gstart = np.maximum.accumulate(np.where(start, np.arange(n), 0))
    pos = np.arange(n) - gstart                         # earlier reviews of the card (sorted order)
    prev = np.full((3, n), -1, np.int64)
    for k in (1, 2, 3):
        ok = pos >= k
        p = np.full(n, -1, np.int64)
        p[ok] = order[np.arange(n)[ok] - k]
        prev[k - 1, order] = p
    lt = np.where(duration > 0, np.log(np.maximum(duration, 1) / 1000.0) - LN8, 0.0)
    for k in range(3):
        p = prev[k]
        has = p >= 0
        g = np.where(has, rating[np.maximum(p, 0)], 0)
        for gg in (1, 2, 3, 4):
            out[:, 4 * k + gg - 1] = (has & (g == gg)).astype(np.float64)
        out[:, 12 + k] = (~has).astype(np.float64)
        th = has & (duration[np.maximum(p, 0)] > 0)
        out[:, 15 + k] = np.where(th, lt[np.maximum(p, 0)], 0.0)
        out[:, 18 + k] = (~th).astype(np.float64)
    cnt = np.empty(n, np.int64); cnt[order] = pos
    again = (rating == 1).astype(np.int64)
    cs_sorted = np.concatenate([[0], np.cumsum(again[order])])
    lap_sorted = cs_sorted[np.arange(n)] - cs_sorted[gstart]   # Again grades among the card's earlier reviews
    lapses = np.empty(n, np.int64); lapses[order] = lap_sorted
    out[:, 21] = np.log1p(cnt) / 5.0
    out[:, 22] = np.log1p(lapses) / 3.0
    p1 = prev[0]
    out[:, 23] = np.where(p1 >= 0, np.log1p(np.maximum(ed[np.maximum(p1, 0)], 0)), 0.0) / 3.0
    out[:, 24] = np.log1p(np.maximum(ed, 0)) / 3.0
    out[:, 25] = np.log1p(np.maximum(es, 0)) / 10.0
    out[:, 26] = (ed == 0).astype(np.float64)
    # the user's earlier reviews
    passed = (rating > 1).astype(np.float64)
    cs = np.concatenate([[0.0], np.cumsum(passed)])
    for col, w in ((27, 50), (28, 500), (29, n + 1)):
        s, c = _window_sum(cs, idx, w)
        out[:, col] = (s + 2 * 0.85) / (c + 2)
    out[:, 30] = np.log1p(idx) / 10.0
    for gg in (1, 2, 3, 4):
        csg = np.concatenate([[0.0], np.cumsum((rating == gg).astype(np.float64))])
        s, c = _window_sum(csg, idx, 500)
        out[:, 30 + gg] = (s + 0.25) / (c + 1)
    t82 = sm82c_features(card, rating, duration, np.asarray(elapsed_days, np.int64))
    out[:, 35:39] = t82[:, 11:15]       # user's running median ln time per grade (centred on ln 8 s)
    out[:, 39] = t82[:, 19]             # over all grades
    out[:, 40] = t82[:, 22]             # exponential average (0.1) of the user's ln times
    return out.astype(np.float32)


def frame_table(frame):
    return explicit_features(frame["card_id"].to_numpy(), frame["rating"].to_numpy(), frame["duration"].to_numpy(),
                             frame["elapsed_days"].to_numpy(), frame["elapsed_seconds"].to_numpy())


def leakfree_frame(data, user, cap=None):
    from rwkv.data_processing import get_rwkv_data
    from scripts import sm71_leak
    frame = get_rwkv_data(data, user)
    frame, _ = sm71_leak.neutralise(frame, "neighbour", data, user)
    full = len(frame)
    return (frame if cap is None else frame.iloc[:cap]), full


# ------------------------------------------------------------------ heads
def released_inputs(logits):
    """[n, 5] from the released instant logits [n, 4] (float32): log-probabilities / 5, logit(P(pass)) / 5."""
    import torch
    lp = torch.log_softmax(logits.float(), -1)
    p = torch.clamp(1.0 - torch.softmax(logits.float(), -1)[:, 0], 1e-5, 1 - 1e-5)
    return torch.cat([lp / 5.0, (torch.log(p) - torch.log1p(-p)).unsqueeze(-1) / 5.0], -1)


def make_heads(seed, device):
    import torch
    from expertium.model.variant_model import TimeMLP   # Linear + ReLU, zero-initialised output Linear
    torch.manual_seed(seed)
    return TimeMLP(INPUTS, HIDDEN, 4).to(device), TimeMLP(ABL_INPUTS, HIDDEN, 4).to(device)


def pass_prob(logits):
    """the model's instant P(pass) (variant_model._get_loss_variant): clamp(1 - softmax[Again], 1e-5, 1 - 1e-5)."""
    import torch
    return torch.clamp(1.0 - torch.softmax(logits, dim=-1)[..., 0], min=1e-5, max=1.0 - 1e-5)


def bce_rows(logits, y):
    """per-row binary LogLoss of P(pass) (the scorer's instant_log_loss: F.binary_cross_entropy of the clamped p)."""
    import torch.nn.functional as F
    return F.binary_cross_entropy(pass_prob(logits.float()).clamp(1e-5, 1 - 1e-5), y.float(), reduction="none")


def build_frozen(released, device):
    from scripts.sm82c_head import Frozen
    from scripts import small_model_heldout as smh
    model = smh.build_expertium("w64", {k: v.float() for k, v in released.items()}).to(device)
    smh.set_expertium_state_bits(model, 0)
    for p in model.parameters():
        p.requires_grad_(False)
    return Frozen(model)


def row_inputs(frozen_out, table, labels_q=None):
    """(x [n, 494], released logits [n, 4], th) of the instant rows from Frozen.forward's output and the table."""
    import torch
    qi, th, feats, o = frozen_out
    logits = o[3].reshape(-1, 4).index_select(0, qi).float()
    ex = torch.as_tensor(table, device=feats.device)[th - 1]
    return torch.cat([feats, released_inputs(logits), ex], -1), logits, th


# ------------------------------------------------------------------ plan
def plan(args):
    """The SM82c plan (same users, order, cap) with `labelled` = the labelled instant rows within the cap: every
    review of the prefix except each card's first (CPU proof C5 checks it against the model's rows)."""
    import pandas as pd
    src = json.loads(Path(args.source).read_text())
    data = data_path()
    order, total = [], 0
    for r in src["order"]:
        cards = pd.read_parquet(Path(data) / "revlogs" / f"user_id={r['user_id']}", columns=["card_id"])["card_id"].to_numpy()
        cards = cards[:src["cap"]]
        lab = int(len(cards) - len(np.unique(cards)))
        order.append({"user_id": r["user_id"], "reviews": r["reviews"], "labelled": lab})
        total += lab
    out = dict(src, order=order, labelled_rows=total, source_plan=str(args.source), source_plan_sha256=sha256(args.source),
               labelled="instant rows: every review of the first `cap` except each card's first")
    atomic_json(args.out, out)
    print(f"[sm87-plan] {len(order):,} users, {total:,} labelled instant rows -> {args.out}")
    return 0


# ------------------------------------------------------------------ training
class TrainBatches:
    """The leak-free frame cut to the first `cap` reviews, its batch (the SM82b / SM82c training inputs) and the
    explicit table of the same prefix."""

    def __init__(self, users, data, cap):
        self.users, self.data, self.cap = users, data, cap

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        started = time.perf_counter()
        try:
            from rwkv.prepare_batch import prepare
            from scripts import compress_common as cc
            frame, full = leakfree_frame(self.data, user, self.cap)
            batch = prepare(cc.make_samples(frame, user, len(frame)), seed=2025 + user)
            extra = {"full": full, "explicit": frame_table(frame), "rating": frame["rating"].to_numpy(np.int64)}
        except Exception as error:  # recorded, not fatal
            return user, None, None, repr(error), time.perf_counter() - started
        return user, batch, extra, None, time.perf_counter() - started


def vars_config(args):
    return {k: getattr(args, k) for k in ("batch", "lr", "warmup", "buffer_rows", "cap", "seed", "max_users", "max_user_reviews",
                                          "smoke_longest")}


def train(args):
    from scripts import small_model_heldout as smh  # noqa: F401  first: sets RWKV_NO_JIT before the model module loads
    import torch
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("HELDOUT_GPU_MEMORY_FRACTION", "0.8")))
    else:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    released_path = Path(args.released)
    released = torch.load(released_path, map_location="cpu", weights_only=True)
    frozen = build_frozen(released, device)
    model = frozen.model
    main, ablation = make_heads(args.seed, device)
    trainable_model = [n for n, p in model.named_parameters() if p.requires_grad]
    params = list(main.parameters()) + list(ablation.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    model_params = {id(p) for p in model.parameters()}
    in_optimizer_model = sum(id(q) in model_params for g in optimizer.param_groups for q in g["params"])
    plan_rec = json.loads(Path(args.plan).read_text())
    args.cap = int(plan_rec["cap"])
    order = plan_rec["order"]
    if args.max_users:
        small = [r for r in order if r["reviews"] <= args.max_user_reviews][:args.max_users]
        order = small + ([max(order, key=lambda r: (r["reviews"], r["user_id"]))] if args.smoke_longest else [])
    total_rows = sum(r["labelled"] for r in order)
    total_steps = -(-total_rows // args.batch)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    resume = out / "resume.pt"
    gen = torch.Generator().manual_seed(args.seed)
    step, done, passes = 0, 0, 0
    buf = []
    stats = {"users": 0, "skipped": [], "rows": 0, "rows_planned_mismatch": [], "label_mismatch": 0, "duplicate_review_th": 0,
             "reviews_forward": 0, "loss_sum": 0.0, "loss_sum_ablation": 0.0, "loss_sum_released": 0.0, "loss_rows": 0,
             "first_loss": None, "first_loss_ablation": None, "first_loss_released": None, "first_corrected_equals_released": None,
             "pass_losses": [], "pass_losses_ablation": [], "pass_losses_released": [], "nonfinite_features": 0}
    if resume.exists():
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        if ck["plan_sha256"] != sha256(args.plan) or ck["config"] != vars_config(args):
            raise SystemExit(f"{resume} belongs to another plan or configuration")
        main.load_state_dict(ck["main"]); ablation.load_state_dict(ck["ablation"])
        optimizer.load_state_dict(ck["optimizer"])
        step, done, passes, stats = ck["step"], ck["done"], ck["passes"], ck["stats"]
        gen.set_state(ck["generator"])
        if ck["leftover"] is not None:
            buf = [tuple(t.to(device) for t in ck["leftover"])]
        print(f"[sm87-train] resumed at user {done}/{len(order)}, step {step}/{total_steps}", flush=True)
    log = {"sum": 0.0, "abl": 0.0, "rel": 0.0, "rows": 0, "last": step}

    def run_pass(final):
        nonlocal step, buf, passes
        if not buf:
            return
        x = torch.cat([b[0] for b in buf]); lg = torch.cat([b[1] for b in buf]); y = torch.cat([b[2] for b in buf])
        perm = torch.randperm(len(x), generator=gen).to(device)
        full = len(x) // args.batch
        stop = len(x) if final else full * args.batch
        ps, pa, pr, pn = 0.0, 0.0, 0.0, 0
        for s in range(0, stop, args.batch):
            idx = perm[s:min(s + args.batch, stop)]
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, total_steps, args.lr, args.warmup)
            xb, lb, yb = x[idx], lg[idx], y[idx]
            corrected = lb + main(xb)
            loss_m = bce_rows(corrected, yb).mean()
            loss_a = bce_rows(lb + ablation(xb[:, FROZEN_DIM:]), yb).mean()
            optimizer.zero_grad(set_to_none=True)
            (loss_m + loss_a).backward()   # disjoint parameters: each head gets its own loss's gradient
            optimizer.step()
            with torch.no_grad():
                loss_r = float(bce_rows(lb, yb).mean())
            vm, va = float(loss_m.detach()), float(loss_a.detach())
            if stats["first_loss"] is None:
                stats["first_loss"], stats["first_loss_ablation"], stats["first_loss_released"] = vm, va, loss_r
                stats["first_corrected_equals_released"] = bool(torch.equal(corrected.detach(), lb))
                print(f"[sm87-train] step 1: instant LogLoss {vm:.5f} / ablation {va:.5f} / released {loss_r:.5f} over "
                      f"{len(idx):,} rows (zero output layers: corrected = released bitwise "
                      f"{stats['first_corrected_equals_released']})", flush=True)
            step += 1
            n = len(idx)
            log["sum"] += vm * n; log["abl"] += va * n; log["rel"] += loss_r * n; log["rows"] += n
            ps += vm * n; pa += va * n; pr += loss_r * n; pn += n
            if step - log["last"] >= args.log_every:
                el = time.perf_counter() - started
                r = max(log["rows"], 1)
                print(f"[sm87-train] step {step}/{total_steps}: instant LogLoss {log['sum'] / r:.5f} / ablation "
                      f"{log['abl'] / r:.5f} / released {log['rel'] / r:.5f} (last {step - log['last']} steps), lr "
                      f"{optimizer.param_groups[0]['lr']:.2e}, users {done}/{len(order)}, {stats['reviews_forward']:,} reviews, "
                      f"{el / 60:.1f} min this run", flush=True)
                log.update(sum=0.0, abl=0.0, rel=0.0, rows=0, last=step)
        stats["loss_sum"] += ps; stats["loss_sum_ablation"] += pa; stats["loss_sum_released"] += pr; stats["loss_rows"] += pn
        stats["pass_losses"].append(round(ps / max(pn, 1), 6)); stats["pass_losses_ablation"].append(round(pa / max(pn, 1), 6))
        stats["pass_losses_released"].append(round(pr / max(pn, 1), 6))
        rest = perm[stop:]
        buf = [(x[rest], lg[rest], y[rest])] if len(rest) else []
        passes += 1

    def save_resume():
        left = tuple(t.cpu() for t in buf[0]) if buf else None
        tmp = resume.with_suffix(".tmp")
        torch.save({"plan_sha256": sha256(args.plan), "config": vars_config(args), "main": main.state_dict(),
                    "ablation": ablation.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "done": done,
                    "passes": passes, "stats": stats, "generator": gen.get_state(), "leftover": left}, tmp)
        os.replace(tmp, resume)

    started = time.perf_counter()
    users = [r["user_id"] for r in order]
    planned = {r["user_id"]: r["labelled"] for r in order}
    loader = torch.utils.data.DataLoader(TrainBatches(users[done:], data_path(), args.cap), batch_size=1,
                                         num_workers=args.workers, collate_fn=first,
                                         prefetch_factor=1 if args.workers else None)
    print(f"[sm87-train] {len(order)} users ({len(order) - done} to go), {total_rows:,} labelled instant rows planned, "
          f"{total_steps:,} steps of {args.batch:,}; model parameters trainable {trainable_model}, in the optimizer "
          f"{in_optimizer_model}; side head {sum(p.numel() for p in main.parameters()):,} + ablation "
          f"{sum(p.numel() for p in ablation.parameters()):,} values; device {device}", flush=True)
    for user, batch, extra, error, prep in loader:
        if args.stop_after_passes and passes >= args.stop_after_passes:
            break
        if batch is None:
            stats["skipped"].append([user, error[:200]])
            done += 1
            continue
        try:
            gb = batch.to(device)
            with torch.no_grad():
                fo = frozen.forward(gb)
                x, logits, th = row_inputs(fo, extra["explicit"])
                qi = fo[0]
                y = gb.labels[0][qi, 2].float()
                rating = torch.as_tensor(extra["rating"], device=device)[th - 1]
                stats["label_mismatch"] += int(((rating > 1).float() != y).sum())
                stats["duplicate_review_th"] += int(len(th) - len(torch.unique(th)))
                stats["nonfinite_features"] += int((~torch.isfinite(x)).any(-1).sum())
            rows = (x, logits, y)
            del gb, fo
        except torch.OutOfMemoryError as oom:
            frozen.hp.clear(); frozen.capture.snaps = []
            torch.cuda.empty_cache()
            stats["skipped"].append([user, f"OOM: {oom}"[:200]])
            done += 1
            continue
        if len(th) != planned[user]:
            stats["rows_planned_mismatch"].append([user, int(len(th)), planned[user]])
        stats["users"] += 1
        stats["rows"] += int(len(th))
        stats["reviews_forward"] += min(extra["full"], args.cap)
        buf.append(rows)
        done += 1
        if sum(len(b[0]) for b in buf) >= args.buffer_rows:
            run_pass(final=False)
            save_resume()
    finished = done >= len(order)
    if finished:
        run_pass(final=True)
    save_resume()
    trunk_same = all(torch.equal(v.detach().cpu(), released[k].float()) for k, v in model.state_dict().items())
    r = max(stats["loss_rows"], 1)
    result = {"status": "complete" if finished else "stopped", "users_done": done, "users": len(order), "step": step,
              "steps_planned": total_steps, "rows_planned": total_rows, "passes": passes, "batch": args.batch,
              "lr": args.lr, "warmup": args.warmup, "buffer_rows": args.buffer_rows, "cap": args.cap, "seed": args.seed,
              "optimizer": "AdamW (betas 0.9/0.999, eps 1e-8, no weight decay), the two side heads only",
              "trainable_model_parameters": trainable_model, "model_parameters_in_optimizer": in_optimizer_model,
              "head_values": sum(p.numel() for p in main.parameters()),
              "ablation_values": sum(p.numel() for p in ablation.parameters()),
              "inputs": {"frozen": FROZEN_DIM, "released": RELEASED_DIM, "explicit": EXPLICIT_DIM, "explicit_names": EXPLICIT_NAMES},
              "trunk_unchanged_in_memory": trunk_same,
              "mean_logloss": round(stats["loss_sum"] / r, 6), "mean_logloss_ablation": round(stats["loss_sum_ablation"] / r, 6),
              "mean_logloss_released": round(stats["loss_sum_released"] / r, 6),
              "seconds_this_run": round(time.perf_counter() - started, 1), "device": str(device),
              "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None,
              "released_master": str(released_path), "released_master_sha256": sha256(released_path),
              **{k: v for k, v in stats.items() if k not in ("loss_sum", "loss_sum_ablation", "loss_sum_released", "loss_rows")}}
    if finished:
        torch.save({k: v.detach().cpu().float().clone() for k, v in main.state_dict().items()}, out / "side-head.pth")
        torch.save({k: v.detach().cpu().float().clone() for k, v in ablation.state_dict().items()}, out / "ablation-head.pth")
        result.update(heads=str(out), side_head_sha256=sha256(out / "side-head.pth"),
                      ablation_sha256=sha256(out / "ablation-head.pth"))
    atomic_json(out / "train.json", result)
    print(f"[sm87-train] {result['status']}: {done}/{len(order)} users, {step:,} steps, {stats['rows']:,} rows, instant "
          f"LogLoss {result['mean_logloss']:.5f} / ablation {result['mean_logloss_ablation']:.5f} / released "
          f"{result['mean_logloss_released']:.5f}, skipped {len(stats['skipped'])}, trunk unchanged {trunk_same}, "
          f"{result['seconds_this_run'] / 60:.1f} min -> {out}", flush=True)
    ok = trunk_same and stats["nonfinite_features"] == 0 and in_optimizer_model == 0 and stats["label_mismatch"] == 0
    return 0 if ok else 1


# ------------------------------------------------------------------ scoring
class ScoreBatches:
    """smh.LeakBatches (neighbour, seed offset 0) + the explicit table of the same frame."""

    def __init__(self, users, data):
        self.users, self.data = users, data

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        started = time.perf_counter()
        try:
            from rwkv.prepare_batch import prepare
            from scripts import compress_common as cc
            frame, _ = leakfree_frame(self.data, user)
            batch = prepare(cc.make_samples(frame, user, len(frame)), seed=2025 + user)
            table = frame_table(frame)
        except Exception as error:  # recorded, not fatal
            return user, None, repr(error), time.perf_counter() - started, None
        return user, batch, None, time.perf_counter() - started, table


def build_scoring(released, variant, device):
    """The scorer's model (scripts/small_model_heldout.py make_model + score_user for an int state variant):
    w64 eager, CPU Hooks / CUDA scoring slots, quantized weights, state bits. -> (Frozen, quantized weights)."""
    from scripts import small_model_heldout as smh
    from scripts import compress_quant as cq
    from scripts import state_vq
    from scripts.sm82c_head import Frozen
    base = {k: v.float() for k, v in released.items()}
    model = smh.build_expertium("w64", base)
    cpu = device.type == "cpu"
    layout = state_vq.install(model) if cpu else state_vq.install_scoring(model)
    model = model.to(device)
    scheme, bits = smh.parse_variant(variant)
    wbits, group = cq.WEIGHT_SCHEMES[scheme]
    names = cq.linear_weight_names(base)
    model.load_state_dict(cq.quantize_state_dict(base, names, wbits, group))
    if cpu:
        smh.set_expertium_state_bits(model, 0)
        state_vq.configure(layout, None, bits)
    else:
        state_vq.set_scoring(layout, None)
        smh.set_expertium_state_bits(model, bits)
    for p in model.parameters():
        p.requires_grad_(False)
    return Frozen(model), {"parameters": sum(t.numel() for t in base.values()),
                           "weight_bytes": cq.weight_bytes(base, names, wbits, group)["total"], "state_bits": bits}


def stats_metrics(stats, batch):
    """scripts/heldout_fast_metrics.evaluate_with_arrays after model.get_loss, for given stats."""
    import torch.nn.functional as F
    from scripts.heldout_fast_metrics import collect_arrays
    mask = stats.is_query.bool() & stats.has_label.bool()
    losses = F.binary_cross_entropy(stats.p_imm.clamp(1e-5, 1 - 1e-5), batch.labels[:, :, 2], reduction="none")
    totals = {"curve_labels": stats.ahead_n, "instant_labels": stats.imm_n,
              "curve_log_loss": stats.ahead_avg.item() * stats.ahead_n,
              "curve_raw_log_loss": stats.ahead_raw_avg.item() * stats.ahead_n,
              "instant_log_loss": losses[mask].sum().item(),
              "instant_rating_cross_entropy": stats.imm_avg.item() * stats.imm_n}
    assert totals["curve_labels"] > 0 and totals["instant_labels"] > 0
    for key in ["curve_log_loss", "curve_raw_log_loss"]:
        totals[key] /= totals["curve_labels"]
    for key in ["instant_log_loss", "instant_rating_cross_entropy"]:
        totals[key] /= totals["instant_labels"]
    return totals, collect_arrays(stats)


class Scorer:
    """One user: the released forward (model.get_loss with the feature capture and a p_linear output tap), then the
    corrected / ablation instant probabilities from the same features."""

    def __init__(self, frozen, heads):
        self.frozen, self.heads = frozen, heads   # heads: {"model": (main, ablation-or-None)...}
        self.tap = {}
        self.apply = None
        model = frozen.model
        model.p_linear.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.tap["logits"] = output
        if self.apply is not None:   # apply-check: the correction inside the model
            flat = output.reshape(-1, 4).clone()
            flat.index_copy_(0, self.apply[0], flat.index_select(0, self.apply[0]) + self.apply[1])
            return flat.view(output.shape)
        return None

    def forward(self, gb, apply=None):
        """(stats, qi, th, feats, full logits [B, T, 4])."""
        import torch
        from scripts.sm82c_head import capturing
        fz = self.frozen
        lab = gb.labels[0]
        qi = torch.nonzero((lab[:, 6] == 1) & (lab[:, 4] == 1)).squeeze(-1)
        fz.capture.rows = qi
        self.apply = apply
        try:
            with capturing(fz.capture):
                stats = fz.model.get_loss(gb)
        finally:
            self.apply = None
        if len(fz.capture.snaps) != 5:
            raise RuntimeError(f"captured {len(fz.capture.snaps)} snapshots, expected 5")
        hp = fz.hp.pop("x").reshape(-1, 128).index_select(0, qi).float()
        parts = [torch.nn.functional.layer_norm(s.float(), (64,)) for s in fz.capture.snaps] + [hp]
        fz.capture.snaps = []
        th = gb.label_review_th[0].long()[qi]
        logits = self.tap.pop("logits")
        return stats, qi, th, torch.cat(parts, -1), logits

    @staticmethod
    def replaced(stats, logits_full, qi, corr):
        """stats with the instant outputs recomputed from released logits + corr at the instant rows (the model's ops)."""
        flat = logits_full.reshape(-1, 4).clone()
        flat.index_copy_(0, qi, flat.index_select(0, qi) + corr)
        z = flat.view(logits_full.shape)
        probs = __import__("torch").softmax(z, dim=-1)
        p_binary = __import__("torch").clamp(1.0 - probs[..., 0], min=1e-5, max=1.0 - 1e-5)
        return stats._replace(p_imm=p_binary.detach(), p_imm_all=probs.detach())


def bench_rows(user, labels, totals_pred, data):
    from scripts.heldout_fast_metrics import metrics_arrays
    from scripts import sm71_leak
    totals, predictions = totals_pred
    rec = {**totals, "benchmark": metrics_arrays(user, labels, predictions)}
    gone = sm71_leak.deleted_review_ths(data, user)
    ths = np.asarray(labels[0], dtype=np.int64)
    is_gone = np.isin(ths, gone)
    for key, keep in (("benchmark_existing", ~is_gone), ("benchmark_deleted", is_gone)):
        sub = ([t for t, k in zip(labels[0], keep) if k], [b for b, k in zip(labels[1], keep) if k])
        rec[key] = metrics_arrays(user, sub, predictions)
    return rec


def load_side_heads(folder, device):
    import torch
    main, abl = make_heads(0, device)
    main.load_state_dict(torch.load(Path(folder) / "side-head.pth", map_location="cpu", weights_only=True))
    abl.load_state_dict(torch.load(Path(folder) / "ablation-head.pth", map_location="cpu", weights_only=True))
    return main.eval(), abl.eval()


def score(args):
    from scripts import small_model_heldout as smh  # noqa: F401  first: sets RWKV_NO_JIT
    import lmdb
    import torch
    from scripts import compress_common as cc
    from scripts.compress_metrics import read_labels
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("HELDOUT_GPU_MEMORY_FRACTION", "0.8")))
        from scripts import compress_wkv_cuda, small_model_wkv   # as small_model_heldout.main
        compress_wkv_cuda.register_cuda()
        errors = {**{f"fork/bits{b}": e for b, e in compress_wkv_cuda.self_test().items()}, **small_model_wkv.self_test()}
        if max(e for key, e in errors.items() if key.endswith("0")) > 1e-4:
            raise SystemExit("CUDA recurrence disagrees with the reference")
    else:
        torch.set_num_threads(2)   # as small_model_heldout.main on CPU (float32 reductions depend on the thread count)
    released = torch.load(args.released, map_location="cpu", weights_only=True)
    frozen, size = build_scoring(released, args.variant, device)
    if args.zero_heads:
        main, abl = make_heads(0, device)
        main.eval(); abl.eval()
    else:
        main, abl = load_side_heads(args.heads, device)
    scorer = Scorer(frozen, None)
    users = json.loads(Path(args.users_file).read_text())["users"]
    apply_users = set(json.loads(Path(args.apply_check_users).read_text())["users"]) if args.apply_check_users else set()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    labels_all = [f"{t}/{args.variant}" for t in TAGS.values()]
    done = {}
    if out.exists():
        for r in map(json.loads, out.read_text().splitlines()):
            if "error" not in r:
                done.setdefault(r["user_id"], set()).add(r["label"])
    todo = [u for u in users if not set(labels_all) <= done.get(u, set())]
    data = cc.data_path()
    counts = cc.run_info()["review_counts"]
    loader = torch.utils.data.DataLoader(ScoreBatches(todo, data), batch_size=1, num_workers=args.workers,
                                         collate_fn=first, prefetch_factor=1 if args.workers else None)
    env = lmdb.open(str(cc.RUN / "label-db"), readonly=True, lock=False, map_size=10_000_000_000)
    started = time.perf_counter()
    print(f"[sm87-score] {len(users)} users, {len(todo)} to score, variant {args.variant}, heads "
          f"{'zero' if args.zero_heads else args.heads}, device {device}", flush=True)
    feats_dump = {}
    with out.open("a", encoding="utf-8") as f:
        for k, (user, batch, error, prep, table) in enumerate(loader):
            with env.begin() as txn:
                labels = read_labels(txn, user)
            if batch is None or labels is None:
                f.write(json.dumps({"user_id": user, "error": error or "no labels"}) + "\n"); f.flush()
                continue
            t0 = time.perf_counter()
            try:
                gb = batch.to(device)
                with torch.inference_mode():
                    stats, qi, th, feats, logits_full = scorer.forward(gb)
                    if stats is None:
                        raise RuntimeError("non-finite evaluation loss")
                    lq = logits_full.reshape(-1, 4).index_select(0, qi).float()
                    x = torch.cat([feats, released_inputs(lq), torch.as_tensor(table, device=device)[th - 1]], -1)
                    nonfinite = int((~torch.isfinite(x)).any(-1).sum())
                    corr = {"model": main(x), "ablation": abl(x[:, FROZEN_DIM:])}
                    sets = {"released": stats_metrics(stats, gb)}
                    for name, c in corr.items():
                        sets[name] = stats_metrics(Scorer.replaced(stats, logits_full, qi, c.to(logits_full.dtype)), gb)
                    corr_abs = {name: float(c.abs().mean()) for name, c in corr.items()}
                    apply = None
                    if user in apply_users:   # the correction applied inside the model: curve bitwise, instant = replaced
                        rep = Scorer.replaced(stats, logits_full, qi, corr["model"].to(logits_full.dtype))
                        s2, _, _, _, _ = scorer.forward(gb, apply=(qi, corr["model"].to(logits_full.dtype)))
                        apply = {"p_curve_bitwise": bool(torch.equal(s2.p_curve, stats.p_curve)),
                                 "ahead_avg_bitwise": bool(torch.equal(s2.ahead_avg, stats.ahead_avg)),
                                 "w_loss_bitwise": bool(torch.equal(s2.w_loss_avg, stats.w_loss_avg)),
                                 "p_imm_equals_replaced": bool(torch.equal(s2.p_imm, rep.p_imm)),
                                 "p_imm_changed": bool(not torch.equal(s2.p_imm, stats.p_imm))}
                    if args.features_out:
                        feats_dump[user] = (th.cpu().numpy(), x.cpu().numpy(), lq.cpu().numpy())
                del gb
            except torch.OutOfMemoryError as oom:
                frozen.hp.clear(); frozen.capture.snaps = []; scorer.tap.clear()
                torch.cuda.empty_cache()
                f.write(json.dumps({"user_id": user, "error": f"OOM: {oom}"[:300]}) + "\n"); f.flush()
                continue
            secs = round(time.perf_counter() - t0, 3)
            for name, tp in sets.items():
                rec = {"user_id": user, "label": f"{TAGS[name]}/{args.variant}", "tag": TAGS[name], "variant": args.variant,
                       "kind": "expertium", "arch": "w64", "checkpoint": str(args.released),
                       "heads": None if name == "released" else ("zero" if args.zero_heads else str(args.heads)),
                       **size, "reviews": counts[str(user)], "prepare_seconds": round(prep, 2), "model_seconds": secs,
                       **bench_rows(user, labels, tp, data), "deleted_cards": "neighbour", "instant_rows": int(len(qi)),
                       "nonfinite_inputs": nonfinite}
                if name != "released":
                    rec["correction_abs_mean"] = corr_abs[name]
                if name == "model" and apply is not None:
                    rec["apply_check"] = apply
                f.write(json.dumps(rec) + "\n")
            f.flush()
            if (k + 1) % 50 == 0:
                print(f"[sm87-score] {k + 1}/{len(todo)} users, {time.perf_counter() - started:.0f} s", flush=True)
    env.close()
    if args.features_out:
        np.savez_compressed(args.features_out, **{f"{u}_{n}": a for u, v in feats_dump.items() for n, a in zip(("th", "x", "logits"), v)})
    print(f"[sm87-score] done {len(todo)} users in {time.perf_counter() - started:.0f} s -> {out}", flush=True)
    return 0


def load_rows(path, label):
    return {r["user_id"]: r for r in map(json.loads, Path(path).read_text().splitlines())
            if r.get("label") == label and "error" not in r} if Path(path).exists() else {}


def rows_compare(rows_path, users, variant=VARIANT, existing=None, existing_label=None):
    """Per user: the corrected / ablation rows' curve fields = the released rows'; with `existing`, the released rows'
    every grade field = the recorded scorer rows'."""
    rel, new, abl = (load_rows(rows_path, f"{TAGS[k]}/{variant}") for k in ("released", "model", "ablation"))
    old = load_rows(existing, existing_label) if existing else {}
    curve_fields = ("curve_labels", "curve_log_loss", "curve_raw_log_loss")
    per, ok_curve, ok_existing, missing = [], True, True, []
    for u in users:
        a, b, c = rel.get(u), new.get(u), abl.get(u)
        if a is None or b is None or c is None:
            missing.append(u)
            continue
        cur = all(a[f] == x[f] for f in curve_fields for x in (b, c)) and all(
            a[s].get("ahead") == x[s].get("ahead") for s in ("benchmark", "benchmark_existing", "benchmark_deleted") for x in (b, c))
        rec = {"user_id": u, "curve_equal": bool(cur)}
        ok_curve &= cur
        if existing:
            e = old.get(u)
            eq = bool(e) and all(e.get(f) == a.get(f) for f in GRADE_FIELDS)
            rec["released_equals_recorded"] = eq
            ok_existing &= eq
        if "apply_check" in b:
            rec["apply_check"] = b["apply_check"]
        per.append(rec)
    return {"users": len(users), "missing": missing, "curve_equal": bool(ok_curve and not missing),
            "released_equals_recorded": bool(ok_existing and not missing) if existing else None,
            "unequal_recorded": [r["user_id"] for r in per if r.get("released_equals_recorded") is False],
            "apply_checks": [r for r in per if "apply_check" in r], "per_user": per}


def compare_cmd(args):
    users = json.loads(Path(args.users_file).read_text())["users"]
    out = rows_compare(args.rows, users, existing=args.existing or None, existing_label=args.existing_label or None)
    ap = out["apply_checks"]
    ap_ok = bool(ap) and all(all(v for k, v in r["apply_check"].items()) for r in ap)
    out["apply_ok"] = ap_ok if args.need_apply else None
    ok = out["curve_equal"] and (out["released_equals_recorded"] is not False) and (ap_ok or not args.need_apply)
    out["pass"] = bool(ok)
    atomic_json(args.out, out)
    print(f"[sm87-compare] {len(users)} users: curve equal {out['curve_equal']}; released = recorded "
          f"{out['released_equals_recorded']}; apply checks {len(ap)} ok {ap_ok}; missing {len(out['missing'])}: "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ------------------------------------------------------------------ CPU proof
def check(args):
    """CPU proof, before any GPU ticket:
    C1 frozen: a CPU training run on 3 small training users: no model parameter trainable or in the optimizer (only
       the two side heads are), the frozen model's tensors unchanged in memory, the released master file untouched
       (sha256 before = after); at the first step the corrected logits are bitwise the released ones (zero output
       layers) and the first loss = the released LogLoss of the same rows; both heads moved.
    C2 resume: the same run stopped after one buffer pass and resumed gives bitwise identical heads.
    C3 zero correction = released, scorer path = the scorer: on 2 evaluation users (CPU, w8+s8), this scorer with
       zero-init side heads writes corrected and ablation rows whose every grade field equals the released rows', and
       the released rows' every grade field equals scripts/small_model_heldout.py's rows (same users, same variant);
       with the trained CPU heads the curve fields stay equal and the apply-check (correction inside the model) keeps
       the curve outputs bitwise and gives the replaced instant outputs bitwise.
    C4 no leak: (a) model features: SM82's T3 perturbation (one review's answer time) changes none of the 448 frozen
       features nor the released logits of any instant row predicting that review or an earlier one (6 picks);
       (b) explicit features: changing review j's grade, duration and elapsed seconds changes no feature row 0..j
       (20 picks x 3 users) and does change later rows; the table of a prefix = the full table's first rows.
    C5 labels: rows per user = the plan's, every target = the frame's grade > 1, no duplicate review.
    C6 loss = metric: the training loss per row = sklearn's log_loss of the same probabilities.
    C8 train path = score path: the 494 inputs of an evaluation user from the training path (fp32) equal the score
       path's at variant fp32 (CPU, both at the CPU scorer's 2 threads), and the released logits too."""
    from scripts import small_model_heldout as smh  # noqa: F401  first: sets RWKV_NO_JIT
    import tempfile
    import torch
    from scripts import compress_common as cc
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    started = time.perf_counter()
    report = {"checks": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    fails = []
    released_path = Path(args.released).resolve()
    sha_before = sha256(released_path)
    released = torch.load(released_path, map_location="cpu", weights_only=True)
    py = [sys.executable, "-X", "utf8", str(Path(__file__).resolve())]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    env.pop("RWKV_DELETED_CARDS", None)
    data = cc.data_path()
    counts = cc.run_info()["review_counts"]
    (HERE / "work").mkdir(parents=True, exist_ok=True)

    def mark(name, ok, **rec):
        report["checks"][name] = {"pass": bool(ok), **rec}
        if not ok:
            fails.append(name.split("_")[0])
        print(f"[sm87-check] {name}: {'PASS' if ok else 'FAIL'} ({time.perf_counter() - started:.0f} s)", flush=True)

    def run(cmd):
        r = subprocess.run(py + cmd, capture_output=True, text=True, env=env, cwd=str(ROOT))
        return r.returncode, r.stdout[-3000:] + r.stderr[-3000:]

    with tempfile.TemporaryDirectory(dir=str(HERE / "work")) as tmp:
        tmp = Path(tmp)
        plan_rec = json.loads(Path(args.plan).read_text())
        small = [r for r in plan_rec["order"] if 400 <= r["reviews"] <= 3000][:3]
        mini_plan = tmp / "plan.json"
        mini_plan.write_text(json.dumps(dict(plan_rec, order=small)))
        common = ["--released", str(released_path), "--plan", str(mini_plan), "--device", "cpu", "--batch", "128",
                  "--buffer-rows", "1500", "--workers", "0", "--log-every", "5"]
        rc_a, log_a = run(["train", "--out", str(tmp / "a")] + common)
        rc_b1, log_b1 = run(["train", "--out", str(tmp / "b"), "--stop-after-passes", "1"] + common)
        rc_b2, log_b2 = run(["train", "--out", str(tmp / "b")] + common)
        ta = json.loads((tmp / "a/train.json").read_text()) if (tmp / "a/train.json").exists() else {}
        tb = json.loads((tmp / "b/train.json").read_text()) if (tmp / "b/train.json").exists() else {}
        if rc_a != 0 or not ta.get("heads"):
            mark("C1_frozen", False, log=log_a)
            report.update(passed=False, fails=fails, seconds=round(time.perf_counter() - started, 1))
            atomic_json(args.out, report)
            print(f"[sm87-check] FAIL {','.join(fails)} ({args.out})")
            return 1
        ha = torch.load(tmp / "a/side-head.pth", weights_only=True); aa = torch.load(tmp / "a/ablation-head.pth", weights_only=True)
        moved = [float(ha["out.weight"].abs().max()), float(aa["out.weight"].abs().max())]
        f1 = (min(moved) > 0 and ta["trunk_unchanged_in_memory"] and ta["trainable_model_parameters"] == []
              and ta["model_parameters_in_optimizer"] == 0 and ta["step"] >= 5 and ta["first_corrected_equals_released"]
              and abs(ta["first_loss"] - ta["first_loss_released"]) < 1e-7 and abs(ta["first_loss_ablation"] - ta["first_loss_released"]) < 1e-7
              and sha256(released_path) == sha_before and ta["head_values"] == 63_876 and ta["ablation_values"] == 6_532)
        mark("C1_frozen", f1, users=[r["user_id"] for r in small], steps=ta["step"], rows=ta["rows"], out_layer_abs_max=moved,
             trainable_model_parameters=ta["trainable_model_parameters"],
             model_parameters_in_optimizer=ta["model_parameters_in_optimizer"], head_values=ta["head_values"],
             ablation_values=ta["ablation_values"], trunk_unchanged_in_memory=ta["trunk_unchanged_in_memory"],
             first_corrected_equals_released=ta["first_corrected_equals_released"],
             first_loss=[ta["first_loss"], ta["first_loss_ablation"], ta["first_loss_released"]],
             pass_losses=[ta["pass_losses"], ta["pass_losses_ablation"], ta["pass_losses_released"]],
             released_master_sha256_unchanged=sha256(released_path) == sha_before)
        f5 = (ta["rows_planned_mismatch"] == [] and ta["label_mismatch"] == 0 and ta["duplicate_review_th"] == 0
              and not ta["skipped"] and ta["rows"] == sum(r["labelled"] for r in small) and ta["nonfinite_features"] == 0)
        mark("C5_labels", f5, rows=ta["rows"], planned=sum(r["labelled"] for r in small), mismatch=ta["rows_planned_mismatch"],
             label_mismatch=ta["label_mismatch"], duplicate_review_th=ta["duplicate_review_th"], skipped=ta["skipped"],
             nonfinite_features=ta["nonfinite_features"])
        if rc_b1 == 0 and rc_b2 == 0 and tb.get("heads"):
            hb = torch.load(tmp / "b/side-head.pth", weights_only=True); ab = torch.load(tmp / "b/ablation-head.pth", weights_only=True)
            f2 = (all(torch.equal(ha[k], hb[k]) for k in ha) and all(torch.equal(aa[k], ab[k]) for k in aa)
                  and tb["step"] == ta["step"] and tb["passes"] == ta["passes"] and ta["passes"] >= 2)
        else:
            f2 = False
        mark("C2_resume_bitwise", f2, steps=[ta.get("step"), tb.get("step")], passes=[ta.get("passes"), tb.get("passes")],
             log=None if f2 else (log_b1 + log_b2)[-2000:])
        # C3 scorer path
        pool = json.loads(U732.read_text())["users"]
        eval_small = sorted((u for u in pool if 1500 <= counts[str(u)] <= 4000), key=lambda u: counts[str(u)])[:2]
        ufile = tmp / "users.json"
        ufile.write_text(json.dumps({"users": eval_small}))
        zrows, trows = tmp / "zero.jsonl", tmp / "trained.jsonl"
        sc = ["--released", str(released_path), "--users-file", str(ufile), "--device", "cpu", "--workers", "0"]
        rc_z, log_z = run(["score", "--zero-heads", "--out", str(zrows)] + sc)
        rc_t, log_t = run(["score", "--heads", str(tmp / "a"), "--out", str(trows), "--apply-check-users", str(ufile)] + sc)
        orows = tmp / "official.jsonl"
        r = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "scripts/small_model_heldout.py"), "--pool", "732",
                            "--users-file", str(ufile), "--model", f"rel=expertium:w64:{released_path}", "--only", "rel/w8+s8",
                            "--device", "cpu", "--workers", "0", "--split-existing", "--deleted-cards", "neighbour",
                            "--out", str(orows)], capture_output=True, text=True, env=env, cwd=str(ROOT))
        zero_eq, official_eq, trained = [], [], {}
        if rc_z == 0 and r.returncode == 0:
            rel = load_rows(zrows, f"released/{VARIANT}")
            off = load_rows(orows, f"rel/{VARIANT}")
            for u in eval_small:
                for k in ("model", "ablation"):
                    z = load_rows(zrows, f"{TAGS[k]}/{VARIANT}").get(u)
                    zero_eq.append(bool(z) and u in rel and all(z.get(f) == rel[u].get(f) for f in GRADE_FIELDS))
                official_eq.append(u in rel and u in off and all(rel[u].get(f) == off[u].get(f) for f in GRADE_FIELDS))
        if rc_t == 0:
            trained = rows_compare(trows, eval_small)
            instant_changed = [load_rows(trows, f"sm87/{VARIANT}")[u]["benchmark"]["instant"]["LogLoss"]
                               != load_rows(trows, f"released/{VARIANT}")[u]["benchmark"]["instant"]["LogLoss"] for u in eval_small]
            trained["instant_changed"] = instant_changed
        ap = trained.get("apply_checks", [])
        f3 = (len(zero_eq) == 4 and all(zero_eq) and len(official_eq) == 2 and all(official_eq) and trained.get("curve_equal")
              and len(ap) == 2 and all(all(x["apply_check"].values()) for x in ap))
        mark("C3_zero_correction_and_scorer", f3, users=eval_small, zero_heads_rows_equal_released=zero_eq,
             released_rows_equal_scorer=official_eq, trained_heads=trained,
             logs=None if f3 else {"zero": log_z[-1500:], "trained": log_t[-1500:], "official": (r.stdout + r.stderr)[-1500:]})
        # C8 train path = score path (fp32)
        from rwkv.prepare_batch import prepare
        u8 = eval_small[0]
        fz = build_frozen(released, torch.device("cpu"))
        frame, _ = leakfree_frame(data, u8)
        batch = prepare(cc.make_samples(frame, u8, len(frame)), seed=2025 + u8)
        threads = torch.get_num_threads()
        torch.set_num_threads(2)   # the CPU scorer's thread count (float32 reductions depend on it)
        with torch.inference_mode():
            xa, la, tha = row_inputs(fz.forward(batch), frame_table(frame))
        torch.set_num_threads(threads)
        o = torch.argsort(tha)
        xa, la, tha = xa[o], la[o], tha[o]
        fo = tmp / "feats.npz"
        u8file = tmp / "u8.json"
        u8file.write_text(json.dumps({"users": [u8]}))
        rc8, log8 = run(["score", "--zero-heads", "--variant", "fp32", "--out", str(tmp / "fp32.jsonl"), "--features-out", str(fo),
                         "--released", str(released_path), "--users-file", str(u8file), "--device", "cpu", "--workers", "0"])
        f8, rec8 = False, {}
        if rc8 == 0 and fo.exists():
            with np.load(fo) as d:
                thb, xb, lb = d[f"{u8}_th"], d[f"{u8}_x"], d[f"{u8}_logits"]
            ob = np.argsort(thb, kind="stable")
            rec8 = {"rows": int(len(thb)), "th_equal": bool(np.array_equal(tha.numpy(), thb[ob])),
                    "inputs_equal": bool(np.array_equal(xa.numpy(), xb[ob])), "logits_equal": bool(np.array_equal(la.numpy(), lb[ob])),
                    "inputs_max_abs_gap": float(np.abs(xa.numpy() - xb[ob]).max()) if len(thb) == len(tha) else None}
            f8 = rec8["th_equal"] and rec8["inputs_equal"] and rec8["logits_equal"]
        mark("C8_train_path_equals_score_path", f8, user=u8, **rec8, log=None if f8 else log8[-2000:])
        # C4a model features: SM82's T3 perturbation
        from rwkv.data_processing import scale_duration
        user = small[1]["user_id"]
        frame0, _ = leakfree_frame(data, user)

        def feats_of(fr):
            b = prepare(cc.make_samples(fr, user, len(fr)), seed=2025 + user)
            with torch.inference_mode():
                x, lg, th = row_inputs(fz.forward(b), frame_table(fr))
            oo = torch.argsort(th)
            return th[oo], x[oo, :FROZEN_DIM + RELEASED_DIM], lg[oo]
        th0, f0, l0 = feats_of(frame0)
        rng = np.random.default_rng(0)
        cands = [j for j in range(len(frame0)) if frame0["elapsed_days"].iat[j] >= 0 and frame0["duration"].iat[j] > 0]
        picks = sorted(rng.choice(cands, size=6, replace=False).tolist())
        leak = []
        for j in picks:
            fr = frame0.copy()
            new_d = int(fr["duration"].iat[j]) * 3 + 1000
            fr.iloc[j, fr.columns.get_loc("duration")] = new_d
            fr.iloc[j, fr.columns.get_loc("scaled_duration")] = scale_duration(np.float64(new_d))
            th1, f1_, l1 = feats_of(fr)
            upto = th0 <= j + 1
            later = th0 > j + 1
            leak.append({"review_th": j + 1, "rows_up_to_it": int(upto.sum()),
                         "up_to_its_review_bitwise": bool(torch.equal(th0, th1) and torch.equal(f0[upto], f1_[upto]) and torch.equal(l0[upto], l1[upto])),
                         "later_changed": bool(not torch.equal(f0[later], f1_[later])) if bool(later.any()) else None})
        f4a = all(x["up_to_its_review_bitwise"] for x in leak) and all(x["later_changed"] is not False for x in leak)
        mark("C4a_no_leak_model_features", f4a, user=user, picks=leak)
        # C4b explicit features
        eb = []
        for u in [small[0]["user_id"], small[2]["user_id"], eval_small[0]]:
            fr, _ = leakfree_frame(data, u)
            full = frame_table(fr)
            r2 = np.random.default_rng(1)
            js = sorted(r2.choice(len(fr) - 1, size=min(20, len(fr) - 1), replace=False).tolist())
            ok_upto, changed = True, 0
            for j in js:
                pr = fr.copy()
                pr.iloc[j, pr.columns.get_loc("duration")] = int(fr["duration"].iat[j]) * 3 + 1000
                pr.iloc[j, pr.columns.get_loc("rating")] = int(fr["rating"].iat[j]) % 4 + 1
                pr.iloc[j, pr.columns.get_loc("elapsed_seconds")] = int(fr["elapsed_seconds"].iat[j]) + 7777
                t2 = frame_table(pr)
                ok_upto &= bool(np.array_equal(full[:j], t2[:j]))
                # row j itself sees review j's own elapsed (known when the card is shown) but not its grade or time
                rest = [c for c in range(EXPLICIT_DIM) if EXPLICIT_NAMES[c] != "elapsed_seconds_ln1p"]
                ok_upto &= bool(np.array_equal(full[j, rest], t2[j, rest]))
                changed += int(not np.array_equal(full[j + 1:], t2[j + 1:]))
            cuts = sorted(r2.choice(np.arange(1, len(fr)), size=5, replace=False).tolist())
            prefix_ok = all(np.array_equal(frame_table(fr.iloc[:k]), full[:k]) for k in cuts)
            rec = {"user": u, "reviews": len(fr), "perturbed": len(js), "rows_up_to_j_unchanged": ok_upto,
                   "later_rows_changed": changed, "prefix_equal": prefix_ok, "finite": bool(np.isfinite(full).all())}
            rec["pass"] = bool(ok_upto and changed > 0 and prefix_ok and rec["finite"])
            eb.append(rec)
        mark("C4b_no_leak_explicit_features", all(r["pass"] for r in eb), users=eb, names=EXPLICIT_NAMES)
    # C6 training loss == the scorer's metric (sklearn log_loss)
    from sklearn.metrics import log_loss
    g = torch.Generator().manual_seed(0)
    n = 4096
    logits = torch.randn(n, 4, generator=g) * 3
    y = (torch.rand(n, generator=g) < 0.85).float()
    ours = float(bce_rows(logits, y).double().mean())
    ref = log_loss(y.numpy(), pass_prob(logits).double().numpy(), labels=[0, 1])
    mark("C6_loss_equals_metric", abs(ours - ref) < 1e-6, rows=n, ours=ours, sklearn=ref)
    report.update(passed=not fails, fails=fails, seconds=round(time.perf_counter() - started, 1))
    atomic_json(args.out, report)
    print(f"[sm87-check] {'PASS' if not fails else 'FAIL ' + ','.join(fails)} ({report['seconds']} s; {args.out})")
    return 0 if not fails else 1


# ------------------------------------------------------------------ decision
GO_POINTS, STOP_POINTS = 0.15, 0.05


def decide(args):
    from scripts import sm77_rule
    users732 = set(json.loads(U732.read_text())["users"])
    big = sm77_rule.load(*sm77_rule.BIG, users732)
    pool = sorted(users732 & set(big))
    rows = {k: load_rows(args.rows, f"{TAGS[k]}/{VARIANT}") for k in TAGS}
    missing = [u for u in pool if any(u not in rows[k] for k in rows)]
    if missing:
        print(f"[sm87-rule] rows missing for {len(missing)} users ({missing[:5]})")
        return 2
    from scripts.sm71_report import load as _load
    loaded = {k: {u: v for u, v in _load(args.rows, f"{TAGS[k]}/{VARIANT}", set(pool)).items() if u in set(pool)} for k in rows}
    rec = rows_compare(args.rows, pool, existing=str(SOLD), existing_label=f"{SLBL}/w8+s8")
    gc = json.loads(Path(args.check).read_text()) if args.check and Path(args.check).exists() else {}
    cmp = {"model": sm77_rule.compare(loaded["model"], loaded["released"], big, pool),
           "ablation": sm77_rule.compare(loaded["ablation"], loaded["released"], big, pool),
           "model_vs_ablation": sm77_rule.compare(loaded["model"], loaded["ablation"], big, pool)}
    inst = cmp["model"]["instant"]
    gain = -inst["points"]
    if gain >= GO_POINTS and inst["ci"][1] < 0:
        verdict = "go"
    elif gain < STOP_POINTS:
        verdict = "stop"
    else:
        verdict = "grey"
    ok_curve = rec["curve_equal"] and cmp["model"]["curve"]["delta"] == 0
    tj = json.loads(Path(args.train_json).read_text())
    res = {"verdict": verdict if ok_curve else "fail-curve", "users": len(pool), "compare": cmp,
           "instant_points_gain": round(gain, 3), "curve_unchanged": bool(ok_curve),
           "released_equals_recorded_scores": rec["released_equals_recorded"], "released_unequal_users": rec["unequal_recorded"],
           "apply_checks": rec["apply_checks"], "apply_check_file": args.check, "apply_check_pass": gc.get("pass"),
           "rule": {"go": f"instant improves by >= {GO_POINTS} points of the big model (1 leak-free epoch) LogLoss, 95% CI of the "
                          "paired user bootstrap (2,000, seed 0) clear of zero", "stop": f"gain < {STOP_POINTS} points",
                    "grey": "in between: report", "curve": "unchanged by construction; checked: corrected rows' curve fields = "
                    "the released rows' on every user, and the correction applied inside the model keeps the curve outputs bitwise"},
           "training": {k: tj.get(k) for k in ("users", "users_done", "rows", "step", "passes", "batch", "lr", "warmup", "buffer_rows",
                                               "cap", "reviews_forward", "skipped", "optimizer", "head_values", "ablation_values",
                                               "inputs", "mean_logloss", "mean_logloss_ablation", "mean_logloss_released",
                                               "first_loss", "first_loss_released", "first_corrected_equals_released",
                                               "seconds_this_run", "peak_gpu_gib", "released_master", "released_master_sha256",
                                               "side_head_sha256", "ablation_sha256")},
           "rows": str(args.rows)}
    HERE.mkdir(parents=True, exist_ok=True)
    atomic_json(HERE / "sm87-decision.json", res)
    m, a, mv = cmp["model"], cmp["ablation"], cmp["model_vs_ablation"]
    md = ["# Frozen-model grade side head (SM87): decision", "",
          "A side head (MLP 494 -> 128 -> 4, zero-init output) adds a correction to the released instant (grade) logits of a "
          "frozen copy of the released small model (the leak-free small model (SM72s) 25 epochs + the 1-epoch 32k Muon "
          "cool-down); the trunk, the curve and the released instant head are untouched. Inputs at each instant row: the "
          "trunk's residual stream after the input encoder and after each of the 4 layer rounds (5 x 64, layer-normalised), the "
          "instant head's hidden layer (128), the released instant prediction (5), and 41 explicit causal grade features (the "
          "card's last 3 grades and answer times, review count, lapses, previous interval; elapsed days and seconds as the "
          "released model sees them; the user's pass rate over the last 50 / 500 / all reviews, grade mix over the last 500, "
          "running median time per grade). Ablation in the same pass: without the model states (released prediction + explicit "
          f"features, MLP 46 -> 128 -> 4). Loss: the instant binary LogLoss. Training: the {tj.get('users_done'):,} users of the "
          f"SM82 quarter-epoch cool-down (none in the 732 set), each history's first {tj.get('cap'):,} reviews, "
          f"{tj.get('rows') or 0:,} instant rows, one pass, {tj.get('step') or 0:,} AdamW steps of {tj.get('batch'):,} rows "
          f"(lr {tj.get('lr')}, cosine), side heads only; {(tj.get('seconds_this_run') or 0) / 60:.1f} min (last run segment). "
          f"Training-set instant LogLoss: side head {tj.get('mean_logloss')}, ablation {tj.get('mean_logloss_ablation')}, "
          f"released {tj.get('mean_logloss_released')}.", "",
          f"Scoring: the scorer's own path (w8+s8, leak-free inputs) on {len(pool)} users of the 732 set; one forward per user "
          "gives the released rows and, from the same features, the corrected and the ablation rows. Released rows equal to the "
          f"recorded released scores (every grade field): {rec['released_equals_recorded']}"
          + (f" (unequal: {rec['unequal_recorded'][:10]})" if rec["unequal_recorded"] else "") + ".", "",
          "| model | curve LogLoss | vs the big model (1 leak-free epoch) | instant LogLoss | vs the big model (1 leak-free epoch) |",
          "|---|---|---|---|---|",
          f"| released small model | {m['curve']['reference']:.5f} | {m['curve']['reference_vs_big_pct']:+.3f}% | "
          f"{m['instant']['reference']:.5f} | {m['instant']['reference_vs_big_pct']:+.3f}% |",
          f"| + grade side head (SM87) | {m['curve']['arm']:.5f} | {m['curve']['arm_vs_big_pct']:+.3f}% | {m['instant']['arm']:.5f} | "
          f"{m['instant']['arm_vs_big_pct']:+.3f}% |",
          f"| + ablation without the model states | {a['curve']['arm']:.5f} | {a['curve']['arm_vs_big_pct']:+.3f}% | "
          f"{a['instant']['arm']:.5f} | {a['instant']['arm_vs_big_pct']:+.3f}% |",
          f"| big model (1 leak-free epoch) | {m['curve']['big']:.5f} | | {m['instant']['big']:.5f} | |", "",
          "| comparison (instant) | points [95% CI] | LogLoss delta | users better |", "|---|---|---|---|"]
    for label, x in (("side head vs released", m), ("ablation vs released", a), ("side head vs ablation: what the model states add", mv)):
        i = x["instant"]
        md.append(f"| {label} | {i['points']:+.3f} [{i['ci'][0]:+.3f}, {i['ci'][1]:+.3f}] | {i['delta']:+.6f} | {i['users_better']} / {x['users']} |")
    md += ["", "Points = LogLoss change as % of the big model's (1 leak-free epoch) LogLoss, review-weighted; negative = better.",
           "", f"Curve: unchanged ({ok_curve}): corrected and ablation rows' curve fields = the released rows' on all {len(pool)} "
           f"users; the correction applied inside the model (apply-check, {len(rec['apply_checks'])} users) keeps the curve "
           f"outputs bitwise: {all(all(r['apply_check'].values()) for r in rec['apply_checks']) if rec['apply_checks'] else None}.", "",
           f"Size: side head {tj.get('head_values'):,} values (ablation {tj.get('ablation_values'):,}); the explicit features need "
           "per card the last 3 grades and times, review and lapse counts and the previous interval, and per user windowed pass "
           "and grade counts plus a running median time per grade at serving time.", "",
           f"Verdict: **{res['verdict']}** (instant gain {gain:+.3f} points, CI [{-inst['ci'][1]:+.3f}, {-inst['ci'][0]:+.3f}]; "
           f"go >= {GO_POINTS} with CI clear of zero, stop < {STOP_POINTS})."]
    (HERE / "sm87-decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[sm87-rule] {len(pool)} users: instant {m['instant']['reference']:.5f} -> {m['instant']['arm']:.5f} "
          f"({m['instant']['points']:+.3f} points, CI {m['instant']['ci']}); ablation {a['instant']['points']:+.3f}; model "
          f"states add {mv['instant']['points']:+.3f}; curve unchanged {ok_curve}; released = recorded "
          f"{rec['released_equals_recorded']}: {res['verdict'].upper()}")
    return 0


def main():
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("train")
    p.add_argument("--released", required=True, help="the released small model's master.pth (read only)")
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--warmup", type=float, default=0.02)
    p.add_argument("--buffer-rows", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--max-users", type=int, default=0)
    p.add_argument("--max-user-reviews", type=int, default=20_000)
    p.add_argument("--smoke-longest", action="store_true")
    p.add_argument("--stop-after-passes", type=int, default=0)
    p = sub.add_parser("score")
    p.add_argument("--released", required=True)
    p.add_argument("--heads", default="", help="folder with side-head.pth and ablation-head.pth")
    p.add_argument("--zero-heads", action="store_true", help="check: fresh zero-output heads")
    p.add_argument("--variant", default=VARIANT)
    p.add_argument("--users-file", required=True)
    p.add_argument("--apply-check-users", default="")
    p.add_argument("--features-out", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=3)
    p = sub.add_parser("compare")
    p.add_argument("--rows", required=True)
    p.add_argument("--users-file", required=True)
    p.add_argument("--existing", default="")
    p.add_argument("--existing-label", default="")
    p.add_argument("--need-apply", action="store_true")
    p.add_argument("--out", required=True)
    p = sub.add_parser("check")
    p.add_argument("--released", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("decide")
    p.add_argument("--rows", required=True)
    p.add_argument("--check", default="")
    p.add_argument("--train-json", required=True)
    args = parser.parse_args()
    return {"plan": plan, "train": train, "score": score, "compare": compare_cmd, "check": check,
            "decide": decide}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
