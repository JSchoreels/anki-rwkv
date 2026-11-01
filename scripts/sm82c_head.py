#!/usr/bin/env python
"""Answer-time head on richer frozen features (SM82c): an answer-time head trained on a FROZEN copy of the released
small model (the leak-free small model (SM72s) 25 epochs + the 1-epoch 32k Muon cool-down), as the frozen-model
answer-time head (SM82b), but with richer inputs and a small MLP, so no grade prediction can change.

Inputs at each instant (query) row, concatenated (474 values):
  frozen trunk features (448): the canonical residual stream (d = 64) at the row after the input encoder
    (features2card), after each of the 4 interleaved layer rounds (round 0: layer 0 of all 5 streams card / deck / note
    / preset / user, ..., round 3: deck layer 3; the last one is the final residual that feeds every head), each
    layer-normalised without affine parameters (5 x 64), and the instant head's 128-wide hidden layer (head_p output,
    the SM82b input). Captured by a CARRY_HOOK stand-in that runs the stock non-exact interleaved path (same gathers,
    blocks and scatters; CPU proof C3 shows every model output is bitwise the stock forward's) and keeps x at the
    instant rows after every round.
  explicit causal time features (26, `explicit_features`), from the revlog rows BEFORE the predicted review only (plus
    that review's own card id and elapsed days, known when the card is shown): the card's last 3 recorded ln answer
    times with "none" flags, ln(1 + the card's earlier review count), the card's last grade (one-hot), the user's
    running median ln time per grade 1-4 exactly as the user-median baseline computes it (fallback: the median over all
    grades, then ln 8 s) with ln(1 + count) per grade, the median over all grades, the user's previous recorded ln time
    (any card) with a flag, an exponential average (0.1) of the user's recorded ln times, ln(1 + elapsed days) with a
    new-card flag, ln(1 + the user's review index). ln times are centred on ln 8 s.
Head (preset w64-sm82c, time_mlps.0): Linear 474 -> 128, ReLU, Linear 128 -> 64 = 4 grades x 16 log-time bins (zero
init: the first loss is ln 16), 69,056 values. Ablation (explicit features only, same pass, not exported): Linear 26 ->
128, ReLU, Linear 128 -> 64 (11,712 values). Loss: SM82's censored cross-entropy of the true grade's bins.

Training: the frozen-model answer-time head (SM82b) setup: the 1,991 users of the SM82 quarter-epoch cool-down (none
in the 732 pool), each history's first 400k reviews through the frozen model on the time scorer's path (fp32, leak-free
inputs, torch.no_grad, every trunk parameter requires_grad False), one pass in SM82b's seeded user order through a
1M-row shuffle buffer (SM82b: 2M; the rows are 474 wide, not 128), minibatches of 8,192 rows, AdamW lr 2e-3 (2%
warm-up, cosine to 0), no weight decay, head only. The written master.pth = the released master's tensors + the head.

  train       --released M --plan P --out DIR       the head training (GPU, or CPU for the proof); resumable.
  score       --master M --ablation A --out R ...   the time scorer (both heads, one pass).
  check       --plan P --out report.json            the CPU proof.
  gradecompare ...                                  the grade check (scripts/sm82b_head.py grade_compare).
  decide      ...                                   the rule -> artifacts/small-model/sm82c/sm82c-decision.{md,json}.
"""
from argparse import ArgumentParser
from contextlib import contextmanager
import hashlib
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

from scripts.expertium_options import time_labels   # noqa: E402  the SM82 label definition (one copy)
from scripts.sm82b_head import (atomic_json, sha256, data_path, time_loss_rows, lr_at, first, grade_compare,   # noqa: E402
                                _paired, _time_arrays, U732, SOLD, SLBL, SM82)

HERE = ROOT / "artifacts/small-model/sm82c"
ARCH = "w64-sm82c"
BINS = 16
HEAD = "time_mlps.0."
D = 64
SNAPS = 5                       # after features2card, after rounds 0..3 (the last = the final residual)
FROZEN_DIM = SNAPS * D + 128    # + head_p
EXPLICIT_DIM = 26
INPUTS = FROZEN_DIM + EXPLICIT_DIM
HIDDEN = 128
LN8 = math.log(8.0)
SM82B = Path("C:/Users/jonat/workspace/wt/sm82b/artifacts/small-model/sm82b")   # read only (SM82b time rows)
EXPLICIT_NAMES = (["card_last_ln_t_1", "card_last_ln_t_2", "card_last_ln_t_3", "card_none_1", "card_none_2", "card_none_3",
                   "card_reviews_ln1p", "card_last_grade_1", "card_last_grade_2", "card_last_grade_3", "card_last_grade_4"]
                  + [f"user_median_ln_t_g{g}" for g in range(1, 5)] + [f"user_count_ln1p_g{g}" for g in range(1, 5)]
                  + ["user_median_ln_t_all", "user_prev_ln_t", "user_prev_none", "user_ema_ln_t", "elapsed_days_ln1p",
                     "new_card", "user_index_ln1p"])
assert len(EXPLICIT_NAMES) == EXPLICIT_DIM


# ------------------------------------------------------------------ explicit causal features
def explicit_features(card, rating, duration, elapsed):
    """[n, 26] float32: row i = the features for predicting review i (review_th i + 1) from reviews 0..i-1 and review
    i's card id and elapsed days only (CPU proof C4b perturbs review j and checks rows 0..j are unchanged)."""
    from scripts.sm82_time import RunningMedian
    card = np.asarray(card, np.int64); rating = np.asarray(rating, np.int64)
    duration = np.asarray(duration, np.int64); elapsed = np.asarray(elapsed, np.int64)
    n = len(duration)
    out = np.zeros((n, EXPLICIT_DIM), np.float64)
    logd = np.log(np.maximum(duration, 1) / 1000.0)
    per = {g: RunningMedian() for g in (1, 2, 3, 4)}
    every = RunningMedian()
    counts = [0, 0, 0, 0, 0]
    cards = {}
    prev_user, ema = None, LN8
    for i in range(n):
        c = int(card[i])
        st = cards.get(c)
        if st is None:
            st = [[], 0, 0]
            cards[c] = st
        last3, cnt, last_grade = st
        row = out[i]
        for k in range(3):
            if k < len(last3):
                row[k] = last3[k] - LN8
            else:
                row[3 + k] = 1.0
        row[6] = math.log1p(cnt) / 5.0
        if 1 <= last_grade <= 4:
            row[6 + last_grade] = 1.0
        em = every.median()
        fb = em if em is not None else LN8
        for g in (1, 2, 3, 4):
            m = per[g].median()
            row[10 + g] = (m if m is not None else fb) - LN8
            row[14 + g] = math.log1p(counts[g]) / 10.0
        row[19] = fb - LN8
        if prev_user is None:
            row[21] = 1.0
        else:
            row[20] = prev_user - LN8
        row[22] = ema - LN8
        e = int(elapsed[i])
        row[23] = math.log1p(max(e, 0)) / 3.0
        row[24] = 1.0 if e < 0 else 0.0
        row[25] = math.log1p(i) / 10.0
        # then review i enters the history (the baseline's push rule, scripts/sm82_time.py baseline_points)
        r = int(rating[i])
        st[1] = cnt + 1
        st[2] = r
        if duration[i] > 0:
            x = float(logd[i])
            per.get(r, every).push(x)
            every.push(x)
            if r in per:
                counts[r] += 1
            last3.insert(0, x)
            del last3[3:]
            prev_user = x
            ema = 0.9 * ema + 0.1 * x
    return out.astype(np.float32)


def revlog(data, user, cap=None):
    import pandas as pd
    rev = pd.read_parquet(Path(data) / "revlogs" / f"user_id={user}", columns=["card_id", "rating", "duration", "elapsed_days"])
    return rev if cap is None else rev.iloc[:cap]


def explicit_table(rev):
    return explicit_features(rev["card_id"].to_numpy(), rev["rating"].to_numpy(), rev["duration"].to_numpy(),
                             rev["elapsed_days"].to_numpy())


# ------------------------------------------------------------------ frozen trunk features
class Capture:
    """Stand-in for variant_model.CARRY_HOOK: VariantSrsRWKV._interleaved_streams' non-exact path verbatim (the path
    the scorer and the frozen-model head (SM82b) run: exact_index False), keeping x at `rows` after features2card and
    after every layer round."""

    def __init__(self, model):
        self.model, self.rows, self.snaps = model, None, []

    def owns(self, model):
        return model is self.model

    def streams(self, model, x, batch_sub_gather, batch_sub_gather_lens, batch_time_shift_selects, batch_skips):
        import torch
        m = self.model
        if m.exact_index:
            raise RuntimeError("Capture copies the non-exact path only")
        self.snaps = [x.index_select(0, self.rows)]
        cur = torch.arange(x.size(0), dtype=torch.long, device=x.device)
        gathers, positions, targets = [], [], []
        for splits in batch_sub_gather:
            stream_gathers, stream_positions, stream_targets, parts = [], [], [], []
            for split in splits:
                index = split.long()
                canonical = torch.where(index >= 0, cur.index_select(0, index.clamp(min=0)), torch.full_like(index, -1))
                keep = torch.nonzero(canonical >= 0).squeeze(-1)
                stream_gathers.append(canonical)
                stream_positions.append(keep)
                stream_targets.append(canonical.index_select(0, keep))
                parts.append(canonical)
            gathers.append(stream_gathers)
            positions.append(stream_positions)
            targets.append(stream_targets)
            cur = torch.cat(parts)
        fused_gather, fused_sizes, fused_positions, fused_targets = [], [], [], []
        for i in range(len(gathers)):
            sizes = [canonical.numel() for canonical in gathers[i]]
            offsets = [0]
            for size in sizes[:-1]:
                offsets.append(offsets[-1] + size)
            fused_gather.append(torch.cat(gathers[i]).clamp(min=0))
            fused_sizes.append(sizes)
            fused_positions.append(torch.cat([keep + offset for keep, offset in zip(positions[i], offsets)]))
            fused_targets.append(torch.cat(targets[i]))
        v0s = [[None] * len(splits) for splits in batch_sub_gather]
        steps = m.layer_steps
        for k, (i, layer) in enumerate(steps):
            block = getattr(getattr(m.rwkv_modules, str(i)).blocks, str(layer))
            inputs = x.index_select(0, fused_gather[i]).split(fused_sizes[i])
            outputs = []
            for s, x_split in enumerate(inputs):
                length = batch_sub_gather_lens[i][s]
                x_in = x_split.view(-1, length, m.d_model)
                v0_in = torch.empty_like(x_in) if layer == 0 else v0s[i][s]
                x_out, v0s[i][s] = block.forward(x_in, v0_in, batch_time_shift_selects[i][s].view(-1, length),
                                                 batch_skips[i][s].view(-1, length))
                outputs.append(x_out.reshape(-1, m.d_model))
            flat = torch.cat(outputs)
            x = x.index_copy(0, fused_targets[i], flat.index_select(0, fused_positions[i]))
            if k + 1 == len(steps) or steps[k + 1][1] != layer:   # end of a round (front-loaded: layer == round)
                self.snaps.append(x.index_select(0, self.rows))
        return x


@contextmanager
def capturing(capture):
    from expertium.model import variant_model
    previous = variant_model.CARRY_HOOK[0]
    variant_model.CARRY_HOOK[0] = capture
    try:
        yield capture
    finally:
        variant_model.CARRY_HOOK[0] = previous


class Frozen:
    """The frozen released model with the feature capture: forward(gb) -> (instant rows, their review_th, [n, 448]
    frozen features, the model outputs)."""

    def __init__(self, model):
        self.model = model
        self.capture = Capture(model)
        self.hp = {}
        model.head_p.register_forward_hook(lambda module, inputs, output: self.hp.__setitem__("x", output))
        if len(model.layer_steps) and sorted({layer for _, layer in model.layer_steps}) != list(range(SNAPS - 1)):
            raise RuntimeError(f"expected {SNAPS - 1} layer rounds, got {model.layer_steps}")

    def forward(self, gb):
        import torch
        lab = gb.labels[0]
        qi = torch.nonzero((lab[:, 6] == 1) & (lab[:, 4] == 1)).squeeze(-1)
        self.capture.rows = qi
        with capturing(self.capture):
            o = self.model.forward_batch_variant(gb.start, gb.sub_gather, gb.sub_gather_lens, gb.time_shift_selects,
                                                 gb.skips, gb.num_data)
        if len(self.capture.snaps) != SNAPS:
            raise RuntimeError(f"captured {len(self.capture.snaps)} snapshots, expected {SNAPS}")
        hp = self.hp.pop("x").reshape(-1, 128).index_select(0, qi).float()
        parts = [torch.nn.functional.layer_norm(s.float(), (D,)) for s in self.capture.snaps] + [hp]
        self.capture.snaps = []
        th = gb.label_review_th[0].long()[qi]
        return qi, th, torch.cat(parts, -1), o


def build_frozen(state, device):
    from scripts import small_model_heldout as smh   # sets RWKV_NO_JIT; the scorer's long WKV
    model = smh.build_expertium(ARCH, state).to(device)
    smh.set_expertium_state_bits(model, 0)   # fp32 state, as the time scorer
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def released_plus_head(released, head_state=None):
    """The released tensors + time_mlps.0 (given, or a fresh head: zero output layer)."""
    import torch
    from expertium.model.variant_model import TimeMLP
    state = {k: v.float() for k, v in released.items()}
    if head_state is None:
        head_state = TimeMLP(INPUTS, HIDDEN, 4 * BINS).state_dict()
    state.update({HEAD + k: v.float() for k, v in head_state.items()})
    return state


# ------------------------------------------------------------------ training
class TrainBatches:
    """SM82b's PrefixBatches (the time scorer's inputs cut to the first `cap` reviews, the revlog's grade and time
    labels) + the explicit feature table of the same prefix."""

    def __init__(self, users, data, cap):
        self.users, self.data, self.cap = users, data, cap

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        started = time.perf_counter()
        try:
            from rwkv.data_processing import get_rwkv_data
            from rwkv.prepare_batch import prepare
            from scripts import compress_common as cc
            from scripts import sm71_leak
            frame = get_rwkv_data(self.data, user)
            frame, _ = sm71_leak.neutralise(frame, "neighbour", self.data, user)
            full = len(frame)
            frame = frame.iloc[:self.cap]
            batch = prepare(cc.make_samples(frame, user, len(frame)), seed=2025 + user)
            rev = revlog(self.data, user)
            lower, cens, _ = time_labels(rev["duration"].to_numpy(), rev["elapsed_days"].to_numpy())
            labels = {"rating": np.array(rev["rating"].to_numpy(np.int64)), "lower": np.array(lower), "cens": np.array(cens),
                      "full": full, "explicit": explicit_table(rev.iloc[:self.cap])}
        except Exception as error:  # recorded, not fatal
            return user, None, None, repr(error), time.perf_counter() - started
        return user, batch, labels, None, time.perf_counter() - started


def make_heads(seed, device):
    import torch
    from expertium.model.variant_model import TimeMLP
    torch.manual_seed(seed)
    main = TimeMLP(INPUTS, HIDDEN, 4 * BINS)
    ablation = TimeMLP(EXPLICIT_DIM, HIDDEN, 4 * BINS)
    return main.to(device), ablation.to(device)


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
    model = build_frozen(released_plus_head(released), device)
    frozen = Frozen(model)
    main, ablation = make_heads(args.seed, device)
    for p in list(main.parameters()) + list(ablation.parameters()):
        p.requires_grad_(True)
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
    stats = {"users": 0, "skipped": [], "rows": 0, "rows_planned_mismatch": [], "duration_input_nonzero": 0,
             "grade_out_of_range": 0, "duplicate_review_th": 0, "reviews_forward": 0, "loss_sum": 0.0, "loss_sum_ablation": 0.0,
             "loss_rows": 0, "first_loss": None, "first_loss_ablation": None, "pass_losses": [], "pass_losses_ablation": [],
             "nonfinite_features": 0}
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
        print(f"[sm82c-train] resumed at user {done}/{len(order)}, step {step}/{total_steps}", flush=True)
    log = {"sum": 0.0, "abl": 0.0, "rows": 0, "last": step}

    def run_pass(final):
        nonlocal step, buf, passes
        if not buf:
            return
        x = torch.cat([b[0] for b in buf]); g = torch.cat([b[1] for b in buf])
        lo = torch.cat([b[2] for b in buf]); ce = torch.cat([b[3] for b in buf])
        perm = torch.randperm(len(x), generator=gen).to(device)
        full = len(x) // args.batch
        stop = len(x) if final else full * args.batch
        ps, pa, pr = 0.0, 0.0, 0
        for s in range(0, stop, args.batch):
            idx = perm[s:min(s + args.batch, stop)]
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, total_steps, args.lr, args.warmup)
            xb = x[idx]
            loss_m = time_loss_rows(main(xb), g[idx], lo[idx], ce[idx]).mean()
            loss_a = time_loss_rows(ablation(xb[:, FROZEN_DIM:]), g[idx], lo[idx], ce[idx]).mean()
            optimizer.zero_grad(set_to_none=True)
            (loss_m + loss_a).backward()   # disjoint parameters: each head gets its own loss's gradient
            optimizer.step()
            vm, va = float(loss_m.detach()), float(loss_a.detach())
            if stats["first_loss"] is None:
                stats["first_loss"], stats["first_loss_ablation"] = vm, va
                print(f"[sm82c-train] step 1: time loss {vm:.4f} / ablation {va:.4f} nats over {len(idx):,} rows "
                      f"(zero output layers: ln 16 = 2.7726)", flush=True)
            step += 1
            n = len(idx)
            log["sum"] += vm * n; log["abl"] += va * n; log["rows"] += n
            ps += vm * n; pa += va * n; pr += n
            if step - log["last"] >= args.log_every:
                el = time.perf_counter() - started
                print(f"[sm82c-train] step {step}/{total_steps}: time loss {log['sum'] / max(log['rows'], 1):.4f} / ablation "
                      f"{log['abl'] / max(log['rows'], 1):.4f} nats (last {step - log['last']} steps), lr "
                      f"{optimizer.param_groups[0]['lr']:.2e}, users {done}/{len(order)}, {stats['reviews_forward']:,} reviews, "
                      f"{el / 60:.1f} min this run", flush=True)
                log.update(sum=0.0, abl=0.0, rows=0, last=step)
        stats["loss_sum"] += ps; stats["loss_sum_ablation"] += pa; stats["loss_rows"] += pr
        stats["pass_losses"].append(round(ps / max(pr, 1), 5)); stats["pass_losses_ablation"].append(round(pa / max(pr, 1), 5))
        rest = perm[stop:]
        buf = [(x[rest], g[rest], lo[rest], ce[rest])] if len(rest) else []
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
    print(f"[sm82c-train] {len(order)} users ({len(order) - done} to go), {total_rows:,} labelled rows planned, "
          f"{total_steps:,} steps of {args.batch:,}; model parameters trainable {trainable_model}, in the optimizer "
          f"{in_optimizer_model}; head {sum(p.numel() for p in main.parameters()):,} + ablation "
          f"{sum(p.numel() for p in ablation.parameters()):,} values; device {device}", flush=True)
    for user, batch, labels, error, prep in loader:
        if args.stop_after_passes and passes >= args.stop_after_passes:
            break
        if batch is None:
            stats["skipped"].append([user, error[:200]])
            done += 1
            continue
        try:
            gb = batch.to(device)
            with torch.no_grad():
                qi, th, feats, _ = frozen.forward(gb)
            lower = torch.as_tensor(labels["lower"], device=device)
            cens = torch.as_tensor(labels["cens"], device=device)
            rating = torch.as_tensor(labels["rating"], device=device)
            ex = torch.as_tensor(labels["explicit"], device=device)
            keep = (th >= 1) & (th <= len(labels["explicit"]))
            keep &= lower[(th - 1).clamp(min=0)] >= 0
            qi, th, feats = qi[keep], th[keep], feats[keep]
            stats["duplicate_review_th"] += int(len(th) - len(torch.unique(th)))
            stats["duration_input_nonzero"] += int((gb.start.reshape(-1, gb.start.shape[-1])[qi, 8] != 0).sum())
            grade = rating[th - 1] - 1
            stats["grade_out_of_range"] += int(((grade < 0) | (grade > 3)).sum())
            x = torch.cat([feats, ex[th - 1]], -1)
            stats["nonfinite_features"] += int((~torch.isfinite(x)).any(-1).sum())
            rows = (x, grade.clamp(0, 3), lower[th - 1], cens[th - 1])
            del gb, feats
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
        stats["reviews_forward"] += min(labels["full"], args.cap)
        buf.append(rows)
        done += 1
        if sum(len(b[0]) for b in buf) >= args.buffer_rows:
            run_pass(final=False)
            save_resume()
    finished = done >= len(order)
    if finished:
        run_pass(final=True)
    save_resume()
    trunk_same = all(torch.equal(v.detach().cpu(), released[k].float()) for k, v in model.state_dict().items() if k in released)
    model_head_zero = all(not bool(v.any()) for k, v in model.state_dict().items() if k.startswith(HEAD + "out."))
    result = {"status": "complete" if finished else "stopped", "users_done": done, "users": len(order), "step": step,
              "steps_planned": total_steps, "rows_planned": total_rows, "passes": passes, "batch": args.batch,
              "lr": args.lr, "warmup": args.warmup, "buffer_rows": args.buffer_rows, "cap": args.cap, "seed": args.seed,
              "optimizer": "AdamW (betas 0.9/0.999, eps 1e-8, no weight decay), the two heads only",
              "trainable_model_parameters": trainable_model, "model_parameters_in_optimizer": in_optimizer_model,
              "head_values": sum(p.numel() for p in main.parameters()),
              "ablation_values": sum(p.numel() for p in ablation.parameters()),
              "inputs": {"frozen": FROZEN_DIM, "explicit": EXPLICIT_DIM, "explicit_names": EXPLICIT_NAMES},
              "trunk_unchanged_in_memory": trunk_same, "model_copy_of_head_untouched": model_head_zero,
              "mean_time_loss": round(stats["loss_sum"] / max(stats["loss_rows"], 1), 5),
              "mean_time_loss_ablation": round(stats["loss_sum_ablation"] / max(stats["loss_rows"], 1), 5),
              "seconds_this_run": round(time.perf_counter() - started, 1), "device": str(device),
              "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None,
              "released_master": str(released_path), "released_master_sha256": sha256(released_path),
              **{k: v for k, v in stats.items() if k not in ("loss_sum", "loss_sum_ablation", "loss_rows")}}
    if finished:
        out_sd = {k: v.clone() for k, v in released.items()}
        out_sd.update({HEAD + k: v.detach().cpu().float().clone() for k, v in main.state_dict().items()})
        torch.save(out_sd, out / "master.pth")
        torch.save({k: v.detach().cpu().float().clone() for k, v in ablation.state_dict().items()}, out / "ablation-head.pth")
        result.update(master=str(out / "master.pth"), master_sha256=sha256(out / "master.pth"),
                      ablation=str(out / "ablation-head.pth"))
    atomic_json(out / "train.json", result)
    print(f"[sm82c-train] {result['status']}: {done}/{len(order)} users, {step:,} steps, {stats['rows']:,} rows, mean "
          f"time loss {result['mean_time_loss']:.4f} / ablation {result['mean_time_loss_ablation']:.4f}, skipped "
          f"{len(stats['skipped'])}, trunk unchanged {trunk_same}, {result['seconds_this_run'] / 60:.1f} min -> {out}", flush=True)
    ok = trunk_same and stats["duration_input_nonzero"] == 0 and stats["nonfinite_features"] == 0 and in_optimizer_model == 0
    return 0 if ok else 1


def vars_config(args):
    return {k: getattr(args, k) for k in ("batch", "lr", "warmup", "buffer_rows", "cap", "seed", "max_users", "max_user_reviews",
                                          "smoke_longest")}


# ------------------------------------------------------------------ time scoring
class ScoreBatches:
    """smh.LeakBatches (the grade scorer's and SM82's time scorer's inputs) + the explicit table of the whole revlog."""

    def __init__(self, users, data, counts):
        from scripts import small_model_heldout as smh
        self.inner = smh.LeakBatches(users, data, counts, "neighbour", 0)
        self.data = data

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, index):
        user, batch, error, prep = self.inner[index]
        table = None
        if batch is not None:
            try:
                table = explicit_table(revlog(self.data, user))
            except Exception as e:  # recorded, not fatal
                batch, error = None, repr(e)
        return user, batch, error, prep, table


def load_heads(master, ablation_path, device):
    import torch
    from expertium.model.variant_model import TimeMLP
    sd = torch.load(master, map_location="cpu", weights_only=True)
    main = TimeMLP(INPUTS, HIDDEN, 4 * BINS)
    main.load_state_dict({k[len(HEAD):]: v.float() for k, v in sd.items() if k.startswith(HEAD)})
    abl = TimeMLP(EXPLICIT_DIM, HIDDEN, 4 * BINS)
    abl.load_state_dict(torch.load(ablation_path, map_location="cpu", weights_only=True))
    released = {k: v for k, v in sd.items() if not k.startswith(HEAD)}
    return sd, released, main.to(device).eval(), abl.to(device).eval()


def score(args):
    """Both heads' time distributions at every time-scored test review (fp32, full history, leak-free inputs), in the
    record format of scripts/sm82_time.py score (so SM82b's rule code reads them)."""
    from scripts import small_model_heldout as smh  # noqa: F401  first: sets RWKV_NO_JIT
    import torch
    from scripts import compress_common as cc
    from scripts.sm82_time import label_db, test_reviews, user_sums, CAP_MS
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("HELDOUT_GPU_MEMORY_FRACTION", "0.8")))
    else:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    sd, _, main, abl = load_heads(args.master, args.ablation, device)
    frozen = Frozen(build_frozen({k: v.float() for k, v in sd.items()}, device))
    users = json.loads(Path(args.users_file).read_text())["users"]
    outs = {"model": Path(args.out), "ablation": Path(args.out_ablation)}
    done = None
    for p in outs.values():
        p.parent.mkdir(parents=True, exist_ok=True)
        got = {json.loads(line)["user_id"] for line in p.read_text().splitlines() if "error" not in line} if p.exists() else set()
        done = got if done is None else done & got
    base = {json.loads(line)["user_id"]: json.loads(line) for line in Path(args.baselines).read_text().splitlines()}
    todo = [u for u in users if u not in done]
    data = cc.data_path()
    loader = torch.utils.data.DataLoader(ScoreBatches(todo, data, cc.run_info()["review_counts"]), batch_size=1,
                                         num_workers=args.workers, collate_fn=first, prefetch_factor=1 if args.workers else None)
    env = label_db()
    started = time.perf_counter()
    print(f"[sm82c-score] {len(users)} users, {len(todo)} to score, {args.master}", flush=True)
    files = {k: p.open("a", encoding="utf-8") for k, p in outs.items()}
    try:
        for k, (user, batch, error, prep, table) in enumerate(loader):
            if batch is None:
                for f in files.values():
                    f.write(json.dumps({"user_id": user, "error": error}) + "\n"); f.flush()
                continue
            with env.begin() as txn:
                ths, frame, lower, cens, cap = test_reviews(data, user, txn)
            try:
                gb = batch.to(device)
                with torch.inference_mode():
                    qi, q_th, feats, o = frozen.forward(gb)
                    p_logits = o[3][0].float()
                    order = torch.argsort(q_th)
                    q_th, qi, feats = q_th[order], qi[order], feats[order]
                    want = torch.as_tensor(ths, device=device)
                    pos = torch.searchsorted(q_th, want).clamp(max=len(q_th) - 1)
                    found = q_th[pos] == want
                    if not bool(found.all()):
                        raise RuntimeError(f"{int((~found).sum())} test reviews without an instant row")
                    x = torch.cat([feats[pos], torch.as_tensor(table, device=device)[want - 1]], -1)
                    grade = torch.as_tensor(frame["rating"].to_numpy()[ths - 1] - 1, device=device)
                    pg = torch.softmax(p_logits[qi[pos]], -1)
                    probs = {}
                    for name, logits in (("model", main(x)), ("ablation", abl(x[:, FROZEN_DIM:]))):
                        pt = torch.softmax(logits.float().view(-1, 4, BINS), -1)
                        probs[name] = (pt[torch.arange(len(pos), device=device), grade].double().cpu().numpy(),
                                       (pg.unsqueeze(-1) * pt).sum(1).double().cpu().numpy())
                    nonfinite = int((~torch.isfinite(x)).any(-1).sum())
                del gb, o
            except torch.OutOfMemoryError as oom:
                frozen.hp.clear(); frozen.capture.snaps = []
                torch.cuda.empty_cache()
                for f in files.values():
                    f.write(json.dumps({"user_id": user, "error": f"OOM: {oom}"[:300]}) + "\n"); f.flush()
                continue
            d = frame["duration"].to_numpy(np.int64)
            t = {"grade": grade.cpu().numpy().astype(np.int8), "lower": lower[ths - 1].astype(np.int8),
                 "cens": cens[ths - 1].astype(np.int8), "logd": np.log(np.maximum(d[ths - 1], 1) / 1000.0),
                 "uncensored": (d[ths - 1] < CAP_MS) & (cens[ths - 1] == 0)}
            sha = hashlib.sha256(np.asarray(ths, np.int64).tobytes()).hexdigest()
            b = base.get(user, {})
            for name, (cond, marg) in probs.items():
                under = cond[:, :7].sum(1)
                rec = {"user_id": user, "reviews": int(len(ths)), "review_th_sha256": sha,
                       "baseline_same_reviews": b.get("review_th_sha256") == sha, "checkpoint": str(args.master),
                       "head": name, "arch": ARCH, "precision": "fp32", "deleted_cards": "neighbour",
                       "prepare_seconds": round(prep, 2), "nonfinite_inputs": nonfinite,
                       "model": user_sums(cond, t), "model_marginal": user_sums(marg, t),
                       "under_5_5s": [[int((t["grade"] == g).sum()), float(under[t["grade"] == g].sum()),
                                       int(((t["grade"] == g) & (t["lower"] <= 6) & (t["cens"] == 0)).sum())] for g in range(4)]}
                files[name].write(json.dumps(rec) + "\n"); files[name].flush()
            if (k + 1) % 50 == 0:
                print(f"[sm82c-score] {k + 1}/{len(todo)} users, {time.perf_counter() - started:.0f} s", flush=True)
    finally:
        for f in files.values():
            f.close()
        env.close()
    print(f"[sm82c-score] done {len(todo)} users in {time.perf_counter() - started:.0f} s -> {args.out}", flush=True)
    return 0


# ------------------------------------------------------------------ CPU proof
def _features_cpu(model_state, user, data, cap=None, frame_edit=None, score_path=False):
    """CPU: (review_th sorted, [n, 448] frozen features, start rows, labels, instant rows) of one user."""
    import torch
    from rwkv.data_processing import get_rwkv_data
    from rwkv.prepare_batch import prepare
    from scripts import compress_common as cc
    from scripts import sm71_leak
    if score_path:
        from scripts import small_model_heldout as smh
        _, batch, error, _ = smh.LeakBatches([user], data, cc.run_info()["review_counts"], "neighbour", 0)[0]
        if batch is None:
            raise RuntimeError(error)
    else:
        frame = get_rwkv_data(data, user)
        frame, _ = sm71_leak.neutralise(frame, "neighbour", data, user)
        if cap:
            frame = frame.iloc[:cap]
        if frame_edit is not None:
            frame = frame_edit(frame)
        batch = prepare(cc.make_samples(frame, user, len(frame)), seed=2025 + user)
    fz = model_state if isinstance(model_state, Frozen) else Frozen(build_frozen(model_state, torch.device("cpu")))
    with torch.inference_mode():
        qi, th, feats, o = fz.forward(batch)
    order = torch.argsort(th)
    return th[order], feats[order], batch, o, qi[order]


def check(args):
    """CPU proof, before any GPU ticket:
    C1 frozen: a CPU training run on 3 small training users (several AdamW updates): no model parameter is trainable
       or in the optimizer (only the two heads are); the written master's every non-head tensor is bitwise the released
       master's (keys = released + time_mlps.0.*, dtypes, values); the trunk in memory is unchanged; both heads moved off
       their start; both first losses are ln 16 (zero output layers).
    C2 resume: the same run stopped after one buffer pass and resumed gives bitwise identical heads.
    C3 grade outputs: on full histories of 2 evaluation users, every grade output (ahead logits, w, w_log_p, instant
       logits, s, d) of the w64-sm82c model with the trained head is bitwise the released w64 model's (fp32 state and
       int8 state), and the capture path (the CARRY_HOOK stand-in) gives bitwise the stock forward's outputs.
    C4 no leak: (a) model features: changing one review's answer time (SM82's T3 perturbation) changes none of the 448
       frozen features of any instant row predicting that review or an earlier one; only later input rows change and
       only the duration column; training rows carry a zero duration input. (b) explicit features: changing review j's
       duration and grade changes no feature row 0..j; features of a prefix = the first rows of the whole history's;
       the per-grade running median equals the user-median baseline's point (sm82_time.baseline_points) at every review
       of grade 1-4, the card's last time equals the card-last baseline's wherever the card has a recorded time.
    C5 labels: rows per user = the plan's, grades in range, no duplicate review.
    C6 loss = metric: the training loss per row equals the scorer's censored LogLoss (sm82_time.censored_ll).
    C7 scorer and export: scripts/small_model_heldout.py (CPU, w8+s8, leak-free) scores the released model (w64) and
       the exported master (w64-sm82c): every grade field and the per-review dump arrays are equal; this script's time
       scorer scores both heads on 2 users (finite, same reviews as the baselines).
    C8 train path = score path: the frozen features of an evaluation user from the training batch builder equal the
       scorer's (LeakBatches) bitwise, and the explicit table of a prefix equals the full one's first rows."""
    from scripts import small_model_heldout as smh  # noqa: F401  first: sets RWKV_NO_JIT
    import tempfile
    import torch
    from scripts import sm82_time as t82
    from scripts import compress_common as cc
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    started = time.perf_counter()
    report = {"checks": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    fails = []
    released_path = Path(args.released).resolve()
    released = torch.load(released_path, map_location="cpu", weights_only=True)
    py = [sys.executable, "-X", "utf8", str(Path(__file__).resolve())]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    env.pop("RWKV_DELETED_CARDS", None)
    data = cc.data_path()
    (HERE / "work").mkdir(parents=True, exist_ok=True)

    def mark(name, ok, **rec):
        report["checks"][name] = {"pass": bool(ok), **rec}
        if not ok:
            fails.append(name.split("_")[0])
        print(f"[sm82c-check] {name}: {'PASS' if ok else 'FAIL'} ({time.perf_counter() - started:.0f} s)", flush=True)

    with tempfile.TemporaryDirectory(dir=str(HERE / "work")) as tmp:
        tmp = Path(tmp)
        plan_rec = json.loads(Path(args.plan).read_text())
        small = [r for r in plan_rec["order"] if 400 <= r["reviews"] <= 3000][:3]
        mini_plan = tmp / "plan.json"
        mini_plan.write_text(json.dumps(dict(plan_rec, order=small)))
        common = ["--released", str(released_path), "--plan", str(mini_plan), "--device", "cpu", "--batch", "128",
                  "--buffer-rows", "600", "--workers", "0", "--log-every", "5"]

        def run(cmd):
            r = subprocess.run(py + cmd, capture_output=True, text=True, env=env, cwd=str(ROOT))
            return r.returncode, r.stdout[-3000:] + r.stderr[-3000:]
        rc_a, log_a = run(["train", "--out", str(tmp / "a")] + common)
        rc_b1, log_b1 = run(["train", "--out", str(tmp / "b"), "--stop-after-passes", "1"] + common)
        rc_b2, log_b2 = run(["train", "--out", str(tmp / "b")] + common)
        ta = json.loads((tmp / "a/train.json").read_text()) if (tmp / "a/train.json").exists() else {}
        tb = json.loads((tmp / "b/train.json").read_text()) if (tmp / "b/train.json").exists() else {}
        if rc_a != 0 or not ta.get("master"):
            mark("C1_frozen", False, log=log_a)
            report.update(passed=False, fails=fails, seconds=round(time.perf_counter() - started, 1))
            atomic_json(args.out, report)
            print(f"[sm82c-check] FAIL {','.join(fails)} ({args.out})")
            return 1
        new = torch.load(ta["master"], map_location="cpu", weights_only=True)
        abl = torch.load(ta["ablation"], map_location="cpu", weights_only=True)
        head_keys = sorted(HEAD + k for k in ("hidden.weight", "hidden.bias", "out.weight", "out.bias"))
        same_keys = sorted(set(new) - set(released)) == head_keys and not (set(released) - set(new))
        bitwise = all(torch.equal(new[k], released[k]) and new[k].dtype == released[k].dtype for k in released)
        moved = [float(new[HEAD + "out.weight"].abs().max()), float(new[HEAD + "out.bias"].abs().max()),
                 float(abl["out.weight"].abs().max())]
        f1 = (same_keys and bitwise and min(moved) > 0 and ta["trunk_unchanged_in_memory"] and ta["trainable_model_parameters"] == []
              and ta["model_parameters_in_optimizer"] == 0 and ta["step"] >= 5 and abs(ta["first_loss"] - math.log(BINS)) < 1e-5
              and abs(ta["first_loss_ablation"] - math.log(BINS)) < 1e-5 and ta["head_values"] == 69_056)
        mark("C1_frozen", f1, users=[r["user_id"] for r in small], steps=ta["step"], rows=ta["rows"],
             keys_released_plus_head=same_keys, non_head_tensors_bitwise=bitwise, out_layer_abs_max=moved,
             trainable_model_parameters=ta["trainable_model_parameters"],
             model_parameters_in_optimizer=ta["model_parameters_in_optimizer"], head_values=ta["head_values"],
             ablation_values=ta["ablation_values"], trunk_unchanged_in_memory=ta["trunk_unchanged_in_memory"],
             first_loss=[ta["first_loss"], ta["first_loss_ablation"]], pass_losses=ta["pass_losses"],
             pass_losses_ablation=ta["pass_losses_ablation"])
        f5 = (ta["rows_planned_mismatch"] == [] and ta["grade_out_of_range"] == 0 and ta["duplicate_review_th"] == 0
              and not ta["skipped"] and ta["rows"] == sum(r["labelled"] for r in small) and ta["nonfinite_features"] == 0)
        mark("C5_labels", f5, rows=ta["rows"], planned=sum(r["labelled"] for r in small), mismatch=ta["rows_planned_mismatch"],
             grade_out_of_range=ta["grade_out_of_range"], duplicate_review_th=ta["duplicate_review_th"], skipped=ta["skipped"],
             nonfinite_features=ta["nonfinite_features"])
        if rc_b1 == 0 and rc_b2 == 0 and tb.get("master"):
            hb = torch.load(tb["master"], map_location="cpu", weights_only=True)
            ab = torch.load(tb["ablation"], map_location="cpu", weights_only=True)
            f2 = (all(torch.equal(new[k], hb[k]) for k in head_keys) and all(torch.equal(abl[k], ab[k]) for k in abl)
                  and tb["step"] == ta["step"] and tb["passes"] == ta["passes"])
        else:
            f2 = False
        mark("C2_resume_bitwise", f2, steps=[ta.get("step"), tb.get("step")], log=None if f2 else (log_b1 + log_b2)[-2000:])
        # C3 grade outputs bitwise (released w64 vs exported w64-sm82c; stock forward vs the capture path)
        from scripts import small_model_heldout as smh
        pool = json.loads(U732.read_text())["users"]
        counts = cc.run_info()["review_counts"]
        eval_small = sorted((u for u in pool if 1500 <= counts[str(u)] <= 4000), key=lambda u: counts[str(u)])[:2]
        names = ["ahead_logits", "w", "w_log_p", "p_logits", "s_raw", "d_raw", "t_logits"]
        probes = []
        rel_model = smh.build_expertium("w64", {k: v.float() for k, v in released.items()})
        new_model = smh.build_expertium(ARCH, {k: v.float() for k, v in new.items()})
        fz_new = Frozen(new_model)
        for user in eval_small:
            _, batch, error, _ = smh.LeakBatches([user], data, counts, "neighbour", 0)[0]
            for bits in (0, 8):
                smh.set_expertium_state_bits(rel_model, bits); smh.set_expertium_state_bits(new_model, bits)
                with torch.inference_mode():
                    a = rel_model.forward_batch_variant(batch.start, batch.sub_gather, batch.sub_gather_lens,
                                                        batch.time_shift_selects, batch.skips, batch.num_data)
                    b = new_model.forward_batch_variant(batch.start, batch.sub_gather, batch.sub_gather_lens,
                                                        batch.time_shift_selects, batch.skips, batch.num_data)
                    _, _, feats, c = fz_new.forward(batch)
                probes.append({"user": user, "bits": bits, "rows": int(a[3].shape[1]),
                               "released_vs_new": {n: bool(torch.equal(a[i], b[i])) for i, n in enumerate(names[:6])},
                               "stock_vs_capture_path": {n: bool(torch.equal(b[i], c[i])) for i, n in enumerate(names)},
                               "features_finite": bool(torch.isfinite(feats).all()), "feature_shape": list(feats.shape)})
        smh.set_expertium_state_bits(new_model, 0)
        f3 = all(all(p["released_vs_new"].values()) and all(p["stock_vs_capture_path"].values()) and p["features_finite"]
                 for p in probes)
        mark("C3_grade_outputs_bitwise", f3, probes=probes)
        # C4a model features: SM82's T3 perturbation on a training user's prefix (CPU, the training path)
        from rwkv.data_processing import scale_duration
        user = small[1]["user_id"]
        th0, f0, b0, _, qi0 = _features_cpu(fz_new, user, data)
        frame0 = sm71_leak_frame(data, user)
        rng = np.random.default_rng(0)
        cands = [j for j in range(len(frame0)) if frame0["elapsed_days"].iat[j] >= 0 and frame0["duration"].iat[j] > 0]
        picks = sorted(rng.choice(cands, size=6, replace=False).tolist())
        leak = []
        for j in picks:
            def edit(frame, j=j):
                frame = frame.copy()
                new_d = int(frame["duration"].iat[j]) * 3 + 1000
                frame.iloc[j, frame.columns.get_loc("duration")] = new_d
                frame.iloc[j, frame.columns.get_loc("scaled_duration")] = scale_duration(np.float64(new_d))
                return frame
            th1, f1_, b1, _, qi1 = _features_cpu(fz_new, user, data, frame_edit=edit)
            review_th = j + 1
            upto = th0 <= review_th
            same = bool(torch.equal(th0, th1) and torch.equal(f0[upto], f1_[upto]))
            later = th0 > review_th
            changed_later = bool(not torch.equal(f0[later], f1_[later])) if bool(later.any()) else None
            s0, s1 = b0.start.reshape(-1, b0.start.shape[-1]), b1.start.reshape(-1, b1.start.shape[-1])
            diff = s0 != s1
            rows_changed = torch.nonzero(diff.any(-1)).squeeze(-1).tolist()
            cols_changed = sorted(set(torch.nonzero(diff)[:, 1].tolist()))
            qrow = int(qi0[th0 == review_th][0]) if bool((th0 == review_th).any()) else None
            leak.append({"review_th": review_th, "query_row": qrow, "features_up_to_its_review_bitwise_equal": same,
                         "later_features_changed": changed_later, "input_rows_changed": rows_changed,
                         "input_columns_changed": cols_changed,
                         "changed_rows_after_query": qrow is not None and all(r > qrow for r in rows_changed),
                         "query_row_duration_input_zero": qrow is not None and bool(float(s0[qrow, 8]) == 0.0)})
        f4a = (all(r["features_up_to_its_review_bitwise_equal"] and r["query_row_duration_input_zero"] and r["changed_rows_after_query"]
                   and r["input_columns_changed"] == [8] and len(r["input_rows_changed"]) == 1 for r in leak)
               and all(r["later_features_changed"] is not False for r in leak) and ta["duration_input_nonzero"] == 0)
        mark("C4a_no_leak_model_features", f4a, user=user, picks=leak,
             training_rows_with_nonzero_duration_input=ta["duration_input_nonzero"])
        # C4b explicit features
        eb = []
        for u in [small[0]["user_id"], small[2]["user_id"], eval_small[0]]:
            rev = revlog(data, u)
            full = explicit_table(rev)
            rec = {"user": u, "reviews": len(rev)}
            r2 = np.random.default_rng(1)
            js = sorted(r2.choice(len(rev) - 1, size=min(20, len(rev) - 1), replace=False).tolist())
            ok_upto, changed = True, 0
            for j in js:
                pr = rev.copy()
                pr.iloc[j, pr.columns.get_loc("duration")] = int(rev["duration"].iat[j]) * 3 + 1000
                pr.iloc[j, pr.columns.get_loc("rating")] = int(rev["rating"].iat[j]) % 4 + 1
                t2 = explicit_table(pr)
                ok_upto &= bool(np.array_equal(full[:j + 1], t2[:j + 1]))
                changed += int(not np.array_equal(full[j + 1:], t2[j + 1:]))
            cuts = sorted(r2.choice(np.arange(1, len(rev)), size=5, replace=False).tolist())
            prefix_ok = all(np.array_equal(explicit_table(rev.iloc[:k]), full[:k]) for k in cuts)
            last, med = t82.baseline_points(rev)
            g = rev["rating"].to_numpy(np.int64)
            m = (g >= 1) & (g <= 4)
            mine = full[np.flatnonzero(m), 10 + g[m]].astype(np.float64) + LN8
            med_gap = float(np.abs(mine - med[m]).max()) if m.any() else 0.0
            have = full[:, 3] == 0
            card_gap = float(np.abs(full[have, 0].astype(np.float64) + LN8 - last[have]).max()) if have.any() else 0.0
            # where the card has no recorded time the card-last baseline falls back to the median (so: equal to med)
            nocard_ok = bool(np.allclose(last[~have], med[~have])) if (~have).any() else True
            rec.update(perturbed=len(js), rows_up_to_j_unchanged=ok_upto, later_rows_changed=changed, prefix_equal=prefix_ok,
                       median_vs_baseline_max_gap=med_gap, card_last_vs_baseline_max_gap=card_gap, no_card_time_falls_back=nocard_ok,
                       finite=bool(np.isfinite(full).all()))
            rec["pass"] = bool(ok_upto and changed > 0 and prefix_ok and med_gap < 1e-5 and card_gap < 1e-5 and nocard_ok and rec["finite"])
            eb.append(rec)
        mark("C4b_no_leak_explicit_features", all(r["pass"] for r in eb), users=eb, names=EXPLICIT_NAMES)
        # C8 train path = score path (an evaluation user, full history)
        tha, fa, _, _, _ = _features_cpu(fz_new, eval_small[0], data)
        thb, fb_, _, _, _ = _features_cpu(fz_new, eval_small[0], data, score_path=True)
        f8 = bool(torch.equal(tha, thb) and torch.equal(fa, fb_))
        mark("C8_train_path_equals_score_path", f8, user=eval_small[0], rows=int(len(tha)))
        # C7 scorer and export
        ufile = tmp / "users.json"
        ufile.write_text(json.dumps({"users": eval_small}))
        rows, dump = tmp / "grade.jsonl", tmp / "dump"
        cmd = [sys.executable, "-X", "utf8", str(ROOT / "scripts/small_model_heldout.py"), "--pool", "732", "--users-file", str(ufile),
               "--model", f"rel=expertium:w64:{released_path}", "--model", f"new=expertium:{ARCH}:{ta['master']}",
               "--only", "rel/w8+s8", "--only", "new/w8+s8", "--device", "cpu", "--workers", "0", "--split-existing",
               "--deleted-cards", "neighbour", "--dump", str(dump), "--out", str(rows)]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(ROOT))
        cmp = grade_compare(rows, dump, "rel", "new", eval_small, existing=None) if r.returncode == 0 and rows.exists() else {"pass": False}
        trows, arows = tmp / "time.jsonl", tmp / "time-ablation.jsonl"
        r2 = subprocess.run(py + ["score", "--master", ta["master"], "--ablation", ta["ablation"],
                                  "--baselines", str(SM82 / "time-baselines.jsonl"), "--users-file", str(ufile), "--out", str(trows),
                                  "--out-ablation", str(arows), "--device", "cpu", "--workers", "0"],
                            capture_output=True, text=True, env=env, cwd=str(ROOT))
        tr = [json.loads(x) for x in trows.read_text().splitlines()] if trows.exists() else []
        ar = [json.loads(x) for x in arows.read_text().splitlines()] if arows.exists() else []
        time_ok = r2.returncode == 0 and len(tr) == 2 and len(ar) == 2 and all(
            "error" not in x and x["baseline_same_reviews"] and math.isfinite(x["model"]["ll"]) and x["nonfinite_inputs"] == 0
            for x in tr + ar)
        mark("C7_scorer_export", cmp["pass"] and time_ok, grade=cmp, scorer_exit=r.returncode,
             scorer_log=None if r.returncode == 0 else (r.stdout + r.stderr)[-2000:], time_scorer_exit=r2.returncode,
             time_rows=[{"user_id": x.get("user_id"), "head": x.get("head"), "reviews": x.get("reviews"),
                         "ll_per_review": x["model"]["ll"] / max(x["model"]["n"], 1)} for x in tr + ar if "model" in x],
             time_log=None if time_ok else (r2.stdout + r2.stderr)[-2000:])
    # C6 training loss == the scorer's metric
    g = torch.Generator().manual_seed(0)
    n = 4096
    logits = torch.randn(n, 4 * BINS, generator=g, dtype=torch.float64) * 1.5
    grade = torch.randint(0, 4, (n,), generator=g)
    lower = torch.randint(0, BINS, (n,), generator=g)
    cens = (torch.rand(n, generator=g) < 0.2).long()
    ours = time_loss_rows(logits, grade, lower, cens).double().numpy()
    prob = torch.softmax(logits.view(n, 4, BINS)[torch.arange(n), grade], -1).numpy()
    gap = float(np.abs(ours - t82.censored_ll(prob, lower.numpy(), cens.numpy())).max())
    mark("C6_loss_equals_metric", gap < 1e-5 and float(prob.min()) > 1e-7, rows=n, max_abs_gap=gap)
    report.update(passed=not fails, fails=fails, seconds=round(time.perf_counter() - started, 1))
    atomic_json(args.out, report)
    print(f"[sm82c-check] {'PASS' if not fails else 'FAIL ' + ','.join(fails)} ({report['seconds']} s; {args.out})")
    return 0 if not fails else 1


def sm71_leak_frame(data, user):
    from rwkv.data_processing import get_rwkv_data
    from scripts import sm71_leak
    frame = get_rwkv_data(data, user)
    frame, _ = sm71_leak.neutralise(frame, "neighbour", data, user)
    return frame


# ------------------------------------------------------------------ grade check
def gradecompare(args):
    users = json.loads(Path(args.users_file).read_text())["users"]
    out = grade_compare(args.rows, args.dump, args.rel_tag, args.new_tag, users,
                        existing=args.existing or None, existing_label=args.existing_label or None)
    atomic_json(args.out, out)
    print(f"[sm82c-grade] {len(users)} users: released vs the exported model with the head bitwise {out['pass']}; vs the "
          f"recorded released scores {out.get('existing_equal')}")
    return 0 if out["pass"] else 1


# ------------------------------------------------------------------ decision
def decide(args):
    from scripts import sm77_rule
    users732 = set(json.loads(U732.read_text())["users"])
    big = sm77_rule.load(*sm77_rule.BIG, users732)
    pool = sorted(users732 & set(big))
    load_rows = lambda p: {r["user_id"]: r for r in map(json.loads, Path(p).read_text().splitlines()) if "error" not in r} if Path(p).exists() else {}
    rows = {"model": load_rows(args.time_rows), "ablation": load_rows(args.ablation_rows),
            "sm82": load_rows(args.sm82_time_rows), "sm82b": load_rows(args.sm82b_time_rows)}
    base = {r["user_id"]: r for r in map(json.loads, Path(args.baselines).read_text().splitlines())}
    tmissing = [u for u in pool if any(u not in v for v in rows.values())]
    if tmissing:
        print(f"[sm82c-rule] time rows missing for {len(tmissing)} users ({tmissing[:5]})")
        return 2
    tusers = [u for u in pool if rows["model"][u]["reviews"] > 0]
    mis = [u for u in tusers if not rows["model"][u]["baseline_same_reviews"]
           or any(rows[k][u]["review_th_sha256"] != rows["model"][u]["review_th_sha256"] for k in rows)]
    if mis:
        print(f"[sm82c-rule] REFUSE: {len(mis)} users' time-scored reviews differ from the baselines' or SM82's / SM82b's ({mis[:5]})")
        return 1
    idx = np.random.default_rng(0).integers(0, len(tusers), size=(2000, len(tusers)))
    arr = {k: _time_arrays(rows[k], base, tusers) for k in rows}
    n, nu, ll, ae = arr["model"]
    tr = {"users": len(tusers), "reviews": int(n.sum()), "uncensored": int(nu.sum()), "predictors": {}, "vs": {}}
    for k in rows:
        _, _, l, a = arr[k]
        tr["predictors"][k] = {"time_logloss": round(float(l["model"].sum() / n.sum()), 5),
                               "time_logloss_marginal": round(float(l["model_marginal"].sum() / n.sum()), 5),
                               "mae_ln_s": round(float(a["model"].sum() / nu.sum()), 5),
                               "mae_ln_s_marginal": round(float(a["model_marginal"].sum() / nu.sum()), 5)}
    for b in ("card_last", "user_median"):
        tr["predictors"][b] = {"time_logloss": round(float(ll[b].sum() / n.sum()), 5), "mae_ln_s": round(float(ae[b].sum() / nu.sum()), 5)}
    beats = True
    for b in ("card_last", "user_median"):
        row = _paired(ll["model"], ll[b], ae["model"], ae[b], n, nu, idx)
        beats &= all(row[m]["delta"] < 0 and row[m]["ci"][1] < 0 for m in row)
        tr["vs"][b] = row
        tr["vs"][f"ablation_vs_{b}"] = _paired(arr["ablation"][2]["model"], ll[b], arr["ablation"][3]["model"], ae[b], n, nu, idx)
    for k in ("ablation", "sm82", "sm82b"):
        tr["vs"][k] = _paired(ll["model"], arr[k][2]["model"], ae["model"], arr[k][3]["model"], n, nu, idx)
    by_grade = {}
    for g in range(4):
        key = f"g{g + 1}"
        c = sum(rows["model"][u]["model"][key][0] for u in tusers)
        rec = {"reviews": c}
        for k in rows:
            rec[k] = round(sum(rows[k][u]["model"][key][1] for u in tusers) / max(c, 1), 4)
            under = [sum(rows[k][u]["under_5_5s"][g][i] for u in tusers) for i in range(3)]
            rec[f"{k}_p_under_5_5s"] = round(under[1] / max(under[0], 1), 4)
            if k == "model":
                rec["actual_p_under_5_5s"] = round(under[2] / max(under[0], 1), 4)
        for b in ("card_last", "user_median"):
            rec[b] = round(sum(base[u][b][key][1] for u in tusers) / max(c, 1), 4)
        by_grade[str(g + 1)] = rec
    tr["by_grade_time_logloss"] = by_grade
    gc = json.loads(Path(args.grade_check).read_text())
    grade_same = bool(gc["pass"])
    rel = sm77_rule.load(SOLD, f"{SLBL}/w8+s8", set(big))
    rg = sm77_rule.compare(rel, rel, big, [u for u in pool if u in rel])
    tj = json.loads(Path(args.train_json).read_text())
    res = {"grade": {"bitwise_equal_released": grade_same, "check_users": gc["users"], "fields": gc["fields"],
                     "existing_equal": gc.get("existing_equal"), "per_user": gc["per_user"],
                     "released_722": {h: {"logloss": rg[h]["arm"], "vs_big_pct": rg[h]["arm_vs_big_pct"]} for h in ("curve", "instant")},
                     "grade_check_file": str(args.grade_check)},
           "time": tr, "training": {k: tj.get(k) for k in ("users", "users_done", "rows", "step", "passes", "batch", "lr", "warmup",
                                                           "buffer_rows", "cap", "reviews_forward", "skipped", "optimizer",
                                                           "head_values", "ablation_values", "inputs", "mean_time_loss",
                                                           "mean_time_loss_ablation", "first_loss", "first_loss_ablation",
                                                           "seconds_this_run", "peak_gpu_gib", "master", "master_sha256",
                                                           "released_master", "released_master_sha256")},
           "time_beats_baselines": bool(beats), "grade_unchanged": grade_same}
    res["verdict"] = "pass" if grade_same and beats else ("fail-grade" if not grade_same else "fail-time")
    res["rule"] = {"grade": "grade unchanged: released model vs the exported model with the head, scorer rows (every grade field) "
                            "and per-review dump arrays bitwise equal on the check users; the non-head weights are the released "
                            "master's bitwise (CPU proof C1)",
                   "time": "head beats both baselines on the censored time LogLoss and on the mean |ln error| (uncensored), 95% CI "
                           "of the review-weighted difference clear of zero (paired user bootstrap 2,000, seed 0)",
                   "time_reviews": "benchmark test reviews of the 722 users with a recorded time, on a later day than the card's "
                                   "previous review; conditional on the true grade (as response-time prediction (SM82))"}
    res.update(time_rows=str(args.time_rows), ablation_rows=str(args.ablation_rows), sm82_time_rows=str(args.sm82_time_rows),
               sm82b_time_rows=str(args.sm82b_time_rows), baselines=str(args.baselines))
    p = gc["per_user"][0] if gc["per_user"] else {}
    res["size"] = {"parameters": p.get("parameters"), "weight_bytes": p.get("weight_bytes")}
    HERE.mkdir(parents=True, exist_ok=True)
    atomic_json(HERE / "sm82c-decision.json", res)
    P = tr["predictors"]
    md = ["# Answer-time head on richer frozen features (SM82c): decision", "",
          "An answer-time head (MLP 474 -> 128 -> 4 grades x 16 log-time bins, zero-init output, censored cross-entropy of the "
          "true grade's bins) trained alone on a frozen copy of the released small model (the leak-free small model (SM72s) 25 "
          "epochs + the 1-epoch 32k Muon cool-down): every other weight is the released master's, bit for bit. Inputs: the "
          "trunk's residual stream at the instant row after the input encoder and after each of the 4 layer rounds (5 x 64, "
          "layer-normalised), the instant head's hidden layer (128), and 26 explicit causal time features (the card's last 3 "
          "answer times, the user's running median per grade, the user's previous time and running average, elapsed days, "
          "review counts). Ablation in the same pass: the explicit features alone (MLP 26 -> 128 -> 64). Training: the "
          f"frozen-model answer-time head (SM82b) setup, the {tj.get('users_done'):,} users of the SM82 quarter-epoch cool-down, "
          f"each history's first {tj.get('cap'):,} reviews, {tj.get('rows') or 0:,} labelled rows, one pass, "
          f"{tj.get('step') or 0:,} AdamW steps of {tj.get('batch'):,} rows (lr {tj.get('lr')}, cosine), heads only; "
          f"{(tj.get('seconds_this_run') or 0) / 60:.1f} min (last run segment).", "",
          f"Grade (unchanged): the scorer (w8+s8, leak-free) on {len(gc['users'])} users of the 732 set, the released model vs "
          "the exported model with the head, in the same ticket: every grade field of the rows and the per-review dump arrays "
          f"compared exactly. Recorded released scores ({SOLD.name}) compared too.", "",
          "| user | reviews | dump rows | rows and dumps bitwise equal | equal to the recorded released scores |", "|---|---|---|---|---|"]
    for r in gc["per_user"]:
        md.append(f"| {r['user_id']} | {r.get('reviews') or 0:,} | {r.get('dump_rows') or 0:,} | "
                  f"{r.get('fields_equal') and r.get('dump_arrays_equal')} | {r.get('existing_fields_equal')} |")
    md += ["", "| head | released model, 722 users (LogLoss) | vs the big model (1 leak-free epoch) | change |", "|---|---|---|---|"]
    for h in ("curve", "instant"):
        x = res["grade"]["released_722"][h]
        md.append(f"| {h} | {x['logloss']:.5f} | {x['vs_big_pct']:+.2f}% | {'none (bitwise)' if grade_same else 'NOT EQUAL'} |")
    md += ["", f"Time ({tr['users']} users, {tr['reviews']:,} test reviews with a recorded time on a later day than the card's "
           f"previous review, {tr['uncensored']:,} under the cap): censored 16-bin time LogLoss given the true grade, and the mean "
           "|ln error| of the predicted median (uncensored reviews). Baselines as in SM82 (same rows).", "",
           "| predictor | time LogLoss | mean abs ln error | grade mixed: LogLoss | grade mixed: abs ln error |", "|---|---|---|---|---|"]
    for k, label in (("model", "richer frozen features head (SM82c)"), ("ablation", "ablation: explicit features only"),
                     ("sm82b", "frozen-model answer-time head (SM82b)"), ("sm82", "jointly trained head, response-time prediction (SM82)")):
        md.append(f"| {label} | {P[k]['time_logloss']:.4f} | {P[k]['mae_ln_s']:.4f} | {P[k]['time_logloss_marginal']:.4f} | "
                  f"{P[k]['mae_ln_s_marginal']:.4f} |")
    for k, label in (("card_last", "baseline: the card's last time"), ("user_median", "baseline: the user's running median per grade")):
        md.append(f"| {label} | {P[k]['time_logloss']:.4f} | {P[k]['mae_ln_s']:.4f} | | |")
    md += ["", "| comparison | LogLoss delta [95% CI] | relative | abs ln error delta [95% CI] | relative |", "|---|---|---|---|---|"]
    for k, label in (("card_last", "head vs the card's last time"), ("user_median", "head vs the user's running median per grade"),
                     ("ablation_vs_card_last", "ablation vs the card's last time"),
                     ("ablation_vs_user_median", "ablation vs the user's running median per grade"),
                     ("ablation", "head vs the ablation: what the model features add"),
                     ("sm82b", "head vs the frozen-model answer-time head (SM82b)"),
                     ("sm82", "head vs the jointly trained head (SM82): the remaining cost of freezing")):
        x, y = tr["vs"][k]["time_logloss"], tr["vs"][k]["mae_ln_s"]
        md.append(f"| {label} | {x['delta']:+.4f} [{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}] | {x['relative_pct']:+.1f}% | "
                  f"{y['delta']:+.4f} [{y['ci'][0]:+.4f}, {y['ci'][1]:+.4f}] | {y['relative_pct']:+.1f}% |")
    md += ["", "| grade | reviews | head LogLoss | ablation | SM82b | SM82 | card's last time | user's median per grade | "
           "P(under 5.5 s) predicted / actual | ablation / SM82b / SM82 predicted |", "|---|---|---|---|---|---|---|---|---|---|"]
    for g, x in by_grade.items():
        md.append(f"| {g} | {x['reviews']:,} | {x['model']:.4f} | {x['ablation']:.4f} | {x['sm82b']:.4f} | {x['sm82']:.4f} | "
                  f"{x['card_last']:.4f} | {x['user_median']:.4f} | {x['model_p_under_5_5s']:.3f} / {x['actual_p_under_5_5s']:.3f} | "
                  f"{x['ablation_p_under_5_5s']:.3f} / {x['sm82b_p_under_5_5s']:.3f} / {x['sm82_p_under_5_5s']:.3f} |")
    s = res["size"]
    if s["parameters"] and s["parameters"][0]:
        md += ["", f"Size: {s['parameters'][0]:,} -> {s['parameters'][1]:,} parameters (+{s['parameters'][1] - s['parameters'][0]:,}); "
               f"scorer w8 weight bytes {s['weight_bytes'][0]:,} -> {s['weight_bytes'][1]:,} ({s['weight_bytes'][1] - s['weight_bytes'][0]:+,} B); "
               "state unchanged. The 26 explicit features need a per-user running median per grade (4 small heaps or a "
               "histogram), the card's last 3 times, and two user scalars at serving time."]
    md += ["", f"Verdict: **{res['verdict']}** (grade unchanged bitwise: {grade_same}; head beats both baselines: {bool(beats)}; "
           f"vs SM82: LogLoss {tr['vs']['sm82']['time_logloss']['delta']:+.4f}, |ln err| {tr['vs']['sm82']['mae_ln_s']['delta']:+.4f}; "
           f"vs SM82b: LogLoss {tr['vs']['sm82b']['time_logloss']['delta']:+.4f}, |ln err| {tr['vs']['sm82b']['mae_ln_s']['delta']:+.4f})."]
    (HERE / "sm82c-decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[sm82c-rule] {tr['users']} users: grade unchanged {grade_same}; time LogLoss {P['model']['time_logloss']:.4f} (ablation "
          f"{P['ablation']['time_logloss']:.4f}) vs card's last {P['card_last']['time_logloss']:.4f} / user median "
          f"{P['user_median']['time_logloss']:.4f} / SM82b {P['sm82b']['time_logloss']:.4f} / SM82 {P['sm82']['time_logloss']:.4f}, "
          f"|ln err| {P['model']['mae_ln_s']:.4f} (ablation {P['ablation']['mae_ln_s']:.4f}) vs {P['card_last']['mae_ln_s']:.4f} / "
          f"{P['user_median']['mae_ln_s']:.4f} / {P['sm82b']['mae_ln_s']:.4f} / {P['sm82']['mae_ln_s']:.4f}: {res['verdict'].upper()}")
    return 0


def main():
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
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
    p.add_argument("--max-users", type=int, default=0, help="smoke: only the first N plan users with <= --max-user-reviews")
    p.add_argument("--max-user-reviews", type=int, default=20_000)
    p.add_argument("--smoke-longest", action="store_true", help="smoke: add the plan's longest user (cut at the cap)")
    p.add_argument("--stop-after-passes", type=int, default=0, help="check only: stop after N buffer passes")
    p = sub.add_parser("score")
    p.add_argument("--master", required=True)
    p.add_argument("--ablation", required=True)
    p.add_argument("--baselines", default=str(SM82 / "time-baselines.jsonl"))
    p.add_argument("--users-file", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--out-ablation", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=3)
    p = sub.add_parser("check")
    p.add_argument("--released", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("gradecompare")
    p.add_argument("--rows", required=True)
    p.add_argument("--dump", required=True)
    p.add_argument("--rel-tag", required=True)
    p.add_argument("--new-tag", required=True)
    p.add_argument("--users-file", required=True)
    p.add_argument("--existing", default="")
    p.add_argument("--existing-label", default="")
    p.add_argument("--out", required=True)
    p = sub.add_parser("decide")
    p.add_argument("--time-rows", required=True)
    p.add_argument("--ablation-rows", required=True)
    p.add_argument("--sm82-time-rows", default=str(SM82 / "gpu/sm82-time.jsonl"))
    p.add_argument("--sm82b-time-rows", default=str(SM82B / "gpu/sm82b-time.jsonl"))
    p.add_argument("--baselines", default=str(SM82 / "time-baselines.jsonl"))
    p.add_argument("--grade-check", required=True)
    p.add_argument("--train-json", required=True)
    args = parser.parse_args()
    return {"train": train, "score": score, "check": check, "gradecompare": gradecompare, "decide": decide}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
