#!/usr/bin/env python
"""Time heads for the family (SM117): the answer-time head recipe of the answer-time head on richer frozen features
(SM82c) applied to all six models of the RWKV family release (rwkv-family-2026-11-13), trained and scored on show-time
inputs (scripts/sm31_inputs.py e2s-query, what the fork feeds the models), and a new read-out: how reliable is
"P(recall within S seconds)".

Recipe (scripts/sm82c_head.py of branch sm82c at 7f5f4150, copied into scripts/sm117_timelib.py and this file): an MLP
(inputs -> 128 -> 4 grades x 16 log-time bins, zero-init output layer, censored cross-entropy of the true grade's bins)
trained ALONE on a frozen copy of the model (no model parameter requires grad or is in the optimizer). Inputs at each
instant (query) row: the trunk's residual stream after the input encoder and after each schedule unit, each layer-
normalised without affine parameters, the instant head's hidden layer (head_p output), and the 26 explicit causal time
features. Same plan users / order / prefix cap (1,991 users, first 400k reviews), AdamW lr 2e-3 (2% warm-up, cosine),
batch 8,192, seed 0, the explicit-only ablation in the same pass. Per model (artifacts/small-model/sm117/design.md):

  model                                          kind       schedule     snapshots   head_p   inputs
  Tiny two-head / Tiny Instant-Only (w64-c1)     expertium  interleaved  5 x 64      128      474
  Small two-head / Small Instant-Only (w64)      expertium  interleaved  5 x 64      128      474
  Big two-head / Big instant-only (fork)         fork       sequential   6 x 128     512      1306

SM117 changes to the recipe: show-time inputs everywhere (training and scoring), same-day reviews included (the label
rule of scripts/sm117_timelib.py time_labels same_day=True; CPU proof C4a), the head lives outside the model (the
trunk file written after training is the frozen model's own state_dict, checked bitwise against the release), and the
Big models get the sequential-schedule capture (ForkCapture). The shuffle buffer is 1M rows (Big: 500k, rows are 1306
wide).

  plan        --out DIR                              plan.json + priors.json (CPU): the SM82c plan users with the
                                                     SM117 label counts, and the pooled priors of the R-under-S baselines
  baselines   --out F                                the SM82 time baselines on the SM117 rows of the 723 users (CPU)
  train       --model KEY --plan P --out DIR         head training (GPU, or CPU for the proof); resumable
  score       --model KEY --head-dir DIR --out R     time + R-under-S scoring of the 723 users (one record per user)
  check       --out report.json                      the CPU proof
  gradecompare ...                                   the grade check (released vs the trunk written after training)
  decide      --model KEY                            the per-model gate -> artifacts/small-model/sm117/<key>-gate.json
  report                                             all six -> artifacts/small-model/sm117/sm117-result.{md,json}
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

from scripts.sm117_timelib import (BINS, BIN_EDGES_MS, CAP_MS, LOG_EDGES, EXPLICIT_DIM, EXPLICIT_NAMES, LN8,  # noqa: E402
                                   atomic_json, sha256, first, time_loss_rows, lr_at, make_time_mlp, time_labels,
                                   time_user_cap, explicit_table, revlog, label_db, test_reviews, median_log_s,
                                   censored_ll, baseline_points, residual_cdf, baseline_prob, user_sums, _paired,
                                   grade_compare, GRID)

HERE = ROOT / "artifacts/small-model/sm117"
HF = Path("C:/Users/jonat/workspace/anki-rwkv/artifacts/hf-family/repo-new")                     # read only
S72 = Path("C:/Users/jonat/workspace/wt/sm72s/artifacts/small-model/sm72s")                      # read only
U732 = S72 / "eval-users-732-fit.json"
S82C_PLAN = Path("C:/Users/jonat/workspace/wt/sm82c/artifacts/small-model/sm82c/plan.json")       # read only
WT = Path("C:/Users/jonat/workspace/wt")
INPUTS = "e2s-query"
VARIANT = f"w8+s8@{INPUTS}"
HIDDEN = 128
S_GRID = (3, 5, 8, 10, 15, 20, 30)          # seconds, pre-registered (design.md)
PRIOR_M = 10.0                              # pseudo-count of the running baselines (a) and (b)
HIST_BINS, HIST_LO, HIST_HI = 1000, -8.0, 8.0   # fine logit bins for ECE / AUC / reliability
ECE_BINS = 20

MODELS = {
    "sm106": {"name": "the Tiny two-head model (SM106)", "short": "Tiny two-head", "kind": "expertium", "arch": "w64-c1",
              "package": "rwkv-tiny-two-head", "two_head": True,
              "existing": WT / "sm106/artifacts/small-model/sm106/gpu/sm106-rows.jsonl"},
    "sm98": {"name": "the Small Instant-Only model (SM98)", "short": "Small Instant-Only", "kind": "expertium", "arch": "w64",
             "package": "rwkv-small-instant-only", "two_head": False,
             "existing": WT / "sm98/artifacts/small-model/sm98/gpu/sm98-rows.jsonl"},
    "sm101": {"name": "the Tiny Instant-Only model (SM101)", "short": "Tiny Instant-Only", "kind": "expertium", "arch": "w64-c1",
              "package": "rwkv-tiny-instant-only", "two_head": False,
              "existing": WT / "sm101/artifacts/small-model/sm101/gpu/sm101-rows.jsonl"},
    "sm99": {"name": "the show-time instant-only big model (SM99)", "short": "Big instant-only", "kind": "fork", "arch": "",
             "package": "rwkv-big-instant-only", "two_head": False,
             "existing": WT / "sm99/artifacts/small-model/sm99/gpu/sm99-rows.jsonl"},
    "sm91s2": {"name": "the show-time big model (SM91s2)", "short": "Big two-head", "kind": "fork", "arch": "",
               "package": "rwkv-big-two-head", "two_head": True,
               "existing": WT / "sm91s2/artifacts/small-model/sm91s2/gpu/sm91s2-inputs.jsonl"},
    "sm92": {"name": "the show-time retrain (SM92)", "short": "Small two-head", "kind": "expertium", "arch": "w64",
             "package": "rwkv-small-two-head", "two_head": True,
             "master": WT / "sm92/artifacts/small-model/sm92-e26-dec-1ep32k-muon/expertium/E89-F65e-F77a/master.pth",
             "shipped_head": WT / "sm92/artifacts/small-model/sm92-head/master.pth",
             "existing": WT / "sm92/artifacts/small-model/sm92/gpu/sm92-inputs.jsonl"},
}
ORDER = ("sm106", "sm98", "sm101", "sm99", "sm91s2", "sm92")


def model_info(key):
    m = dict(MODELS[key])
    man = json.loads((HF / m["package"] / "manifest.json").read_text(encoding="utf-8"))
    m["label"] = man["label"]
    m["sha256"] = man["checkpoint"]["sha256"]
    m["master"] = Path(m.get("master") or HF / m["package"] / "master.pth")
    m["existing_label"] = f"{man['label']}/{VARIANT}"
    return m


def verified_master(key):
    m = model_info(key)
    got = sha256(m["master"])
    if got != m["sha256"]:
        raise SystemExit(f"{key}: {m['master']} sha256 {got} != the package manifest's {m['sha256']}")
    return m


def spec(key, path):
    m = MODELS[key]
    return f"fork::{path}" if m["kind"] == "fork" else f"expertium:{m['arch']}:{path}"


# ------------------------------------------------------------------ frozen model + feature capture
def build_model(key, state, device):
    """The scorer's model of `key` (fp32 weights, fp32 state), every parameter frozen."""
    from scripts import small_model_heldout as smh   # first: RWKV_NO_JIT + the fork state-quant patch
    import torch
    m = MODELS[key]
    state = {k: v.float() for k, v in state.items()}
    if m["kind"] == "fork":
        from scripts import compress_quant as cq
        if torch.device(device).type == "cuda":   # the generated fork module calls compress::wkv_q: CUDA impl, as the scorer does
            from scripts import compress_wkv_cuda
            compress_wkv_cuda.register_cuda()
            err = compress_wkv_cuda.self_test()[0]
            if err > 1e-4:
                raise SystemExit(f"CUDA wkv_q disagrees with the float32 reference (max abs error {err})")
        model = cq.load_model(state=state, reference_wkv=True)
        for module in model.rwkv_modules:
            for block in module.blocks:
                block.time_mixer.use_reference_wkv = True
        cq.set_state_bits(model, 0)
    else:
        model = smh.build_expertium(m["arch"], state)
        smh.set_expertium_state_bits(model, 0)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    del torch
    return model


class Capture:
    """sm82c_head.py Capture (verbatim): stand-in for variant_model.CARRY_HOOK, VariantSrsRWKV._interleaved_streams'
    non-exact path, keeping x at `rows` after features2card and after every layer round."""

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


def fork_forward_capture(model, gb, rows):
    """SM117: rwkv.model.srs_model.SrsRWKV.forward_batch verbatim (sequential streams card, deck, note, preset, user),
    keeping x at the canonical rows `rows` after features2card and after each stream. Each stream's output layout is
    the concatenation of its split gathers (into the previous stream's layout); `cur` tracks the canonical row of every
    position, and the snapshot after stream i takes, for each canonical row, its position in that layout. The final
    layout must be canonical (the labels' order); checked."""
    import torch
    x = model.features2card(gb.start)
    n = x.size(0)
    snaps = [x.index_select(0, rows)]
    cur = torch.arange(n, dtype=torch.long, device=x.device)
    assert len(gb.sub_gather) == len(model.rwkv_modules)
    for i, submodule in enumerate(model.rwkv_modules):
        y, parts = [], []
        for split_gather, sub_len, time_shift_select, skip in zip(gb.sub_gather[i], gb.sub_gather_lens[i],
                                                                  gb.time_shift_selects[i], gb.skips[i]):
            module_in = torch.index_select(x, dim=0, index=torch.clamp(split_gather, min=0)).view(-1, sub_len, model.d_model)
            time_shift_select_BT = time_shift_select.view(-1, sub_len)
            skip_BT = skip.view(-1, sub_len)
            assert module_in.size(0) == time_shift_select_BT.size(0) and module_in.size(0) == skip_BT.size(0)
            module_out = submodule(module_in, time_shift_select_BT=time_shift_select_BT, skip_BT=skip_BT)
            y.append(module_out.view(-1, model.d_model))
            index = split_gather.long()
            parts.append(torch.where(index >= 0, cur.index_select(0, index.clamp(min=0)), torch.full_like(index, -1)))
        x = torch.cat(y)
        cur = torch.cat(parts)
        valid = torch.nonzero(cur >= 0).squeeze(-1)
        if valid.numel() != n or not bool((torch.bincount(cur.index_select(0, valid), minlength=n) == 1).all()):
            raise RuntimeError(f"stream {i}: the layout does not hold every canonical row exactly once")
        inv = torch.full((n,), -1, dtype=torch.long, device=x.device)
        inv[cur.index_select(0, valid)] = valid
        snaps.append(x.index_select(0, inv.index_select(0, rows)))
    if x.size(0) != n or not torch.equal(cur, torch.arange(n, dtype=torch.long, device=x.device)):
        raise RuntimeError("the final layout is not the canonical row order")
    x = x.view(gb.num_data, -1, model.d_model)
    out = model.head_and_out(x)
    # the instant head's hidden layer as head_and_out computes it (the scripted fork model takes no hooks): the same
    # modules on the same full tensor (prehead_dropout is the identity in eval mode)
    hp = model.head_p(model.prehead_dropout(model.prehead_norm(x)))
    return out, snaps, hp


class Frozen:
    """The frozen model with the feature capture: forward(gb) -> (instant rows, their review_th, [n, frozen_dim]
    features, the model outputs)."""

    def __init__(self, model, kind):
        self.model, self.kind = model, kind
        self.hp = {}
        if kind != "fork":   # the fork model is a ScriptModule (no hooks): fork_forward_capture returns head_p's output
            model.head_p.register_forward_hook(lambda module, inputs, output: self.hp.__setitem__("x", output))
        self.d = int(model.d_model)
        self.hp_dim = int(getattr(model.head_p, "0").out_features)
        if kind == "fork":
            self.snaps_n = len(model.rwkv_modules) + 1
        else:
            rounds = sorted({layer for _, layer in model.layer_steps})
            if rounds != list(range(len(rounds))):
                raise RuntimeError(f"unexpected layer rounds {model.layer_steps}")
            self.snaps_n = len(rounds) + 1
            self.capture = Capture(model)
        self.frozen_dim = self.snaps_n * self.d + self.hp_dim
        self.inputs = self.frozen_dim + EXPLICIT_DIM

    def forward(self, gb):
        import torch
        lab = gb.labels[0]
        qi = torch.nonzero((lab[:, 6] == 1) & (lab[:, 4] == 1)).squeeze(-1)
        if self.kind == "fork":
            o, snaps, hp_all = fork_forward_capture(self.model, gb, qi)
            self.hp["x"] = hp_all
        else:
            self.capture.rows = qi
            with capturing(self.capture):
                o = self.model.forward_batch_variant(gb.start, gb.sub_gather, gb.sub_gather_lens, gb.time_shift_selects,
                                                     gb.skips, gb.num_data)
            snaps, self.capture.snaps = self.capture.snaps, []
        if len(snaps) != self.snaps_n:
            raise RuntimeError(f"captured {len(snaps)} snapshots, expected {self.snaps_n}")
        hp = self.hp.pop("x").reshape(-1, self.hp_dim).index_select(0, qi).float()
        parts = [torch.nn.functional.layer_norm(s.float(), (self.d,)) for s in snaps] + [hp]
        th = gb.label_review_th[0].long()[qi]
        return qi, th, torch.cat(parts, -1), o

    def stock(self, gb):
        if self.kind == "fork":
            return self.model.forward_batch(gb.start, gb.sub_gather, gb.sub_gather_lens, gb.time_shift_selects, gb.skips,
                                            gb.num_data)
        return self.model.forward_batch_variant(gb.start, gb.sub_gather, gb.sub_gather_lens, gb.time_shift_selects,
                                                gb.skips, gb.num_data)

    def clear(self):
        self.hp.clear()
        if self.kind != "fork":
            self.capture.snaps = []


# ------------------------------------------------------------------ show-time batches
@contextmanager
def revlog_edit(edit):
    """CPU proof only: pandas.read_parquet of a revlogs/ path returns edit(frame)."""
    import pandas as pd
    original = pd.read_parquet

    def patched(path, *a, **k):
        frame = original(path, *a, **k)
        return edit(frame) if "/revlogs/" in str(path).replace("\\", "/") and edit is not None else frame

    pd.read_parquet = patched
    try:
        yield
    finally:
        pd.read_parquet = original


def show_time_batch(data, user, cap=None, inputs=INPUTS):
    """The scorer's show-time batch (smh.LeakBatches(..., inputs="e2s-query") = sm31_inputs.samples + prepare with
    seed 2025 + user), optionally cut to the first `cap` reviews (after the frame transforms, before the samples).
    inputs="normal" gives SM82c's batch (the C4a contrast)."""
    from rwkv.prepare_batch import prepare
    from scripts import compress_common as cc
    from scripts import sm31_inputs
    from scripts import sm71_leak
    frame = sm31_inputs.load_frame(data, user, inputs)
    frame, _ = sm71_leak.neutralise(frame, "neighbour", data, user)
    frame = sm31_inputs.apply_frame(frame, inputs)
    full = len(frame)
    if cap:
        frame = frame.iloc[:cap]
    with sm31_inputs.query_patch(inputs):
        samples = cc.make_samples(frame, user, len(frame))
    return prepare(samples, seed=2025 + user), full


# ------------------------------------------------------------------ plan + priors (CPU)
def _plan_user(args):
    data, user, cap = args
    import pandas as pd
    f = pd.read_parquet(Path(data) / "revlogs" / f"user_id={user}", columns=["rating", "duration", "elapsed_days"])
    d, e, r = f["duration"].to_numpy(np.int64), f["elapsed_days"].to_numpy(np.int64), f["rating"].to_numpy(np.int64)
    lower, _, ucap = time_labels(d, e, same_day=True)
    lab = lower >= 0
    cnt = {"labelled": int(lab[:cap].sum()), "same_day": int((lab & (e == 0))[:cap].sum())}
    # priors: pooled over the whole history's labelled reviews (training users only)
    pri = {"n": int(lab.sum()), "n_g": [int((lab & (r == g)).sum()) for g in (1, 2, 3, 4)],
           "under_g": [[int((lab & (r == g) & (d <= 1000 * s)).sum()) for s in S_GRID] for g in (1, 2, 3, 4)],
           "event": [int((lab & (r >= 2) & (d <= 1000 * s)).sum()) for s in S_GRID]}
    return user, len(f), cnt, int(ucap), pri


def plan(args):
    """plan.json = SM82c's plan (same 1,991 users, order, cap 400k; none in the 732 pool) with the labelled-row counts
    of the SM117 rule; priors.json = the pooled rates of the R-under-S baselines from those training users."""
    from multiprocessing import Pool
    from scripts import compress_common as cc
    old = json.loads(S82C_PLAN.read_text())
    order = [r["user_id"] for r in old["order"]]
    if set(order) & set(json.loads(U732.read_text())["users"]):
        raise SystemExit("plan users overlap the 732 pool")
    data = str(cc.data_path())
    with Pool(args.workers) as p:
        info = {u: (n, c, uc, pri) for u, n, c, uc, pri in p.map(_plan_user, [(data, u, old["cap"]) for u in order], chunksize=8)}
    bad = [u for r in old["order"] for u in [r["user_id"]] if info[u][0] != r["reviews"]]
    if bad:
        raise SystemExit(f"revlog length differs from SM82c's plan for {len(bad)} users")
    cap = old["cap"]
    rec = {"source": str(S82C_PLAN), "source_sha256": sha256(S82C_PLAN), "label_rule": "SM117 (same-day reviews kept)",
           "seed": old["seed"], "cap": cap, "users": len(order), "reviews_all": old["reviews_all"],
           "reviews_used": old["reviews_used"], "users_cut": old["users_cut"],
           "labelled_rows": int(sum(info[u][1]["labelled"] for u in order)),
           "same_day_rows": int(sum(info[u][1]["same_day"] for u in order)),
           "labelled_rows_sm82_rule": old["labelled_rows"],
           "order": [{"user_id": u, "reviews": info[u][0], "labelled": info[u][1]["labelled"],
                      "same_day": info[u][1]["same_day"], "user_cap_ms": info[u][2]} for u in order]}
    out = Path(args.out)
    atomic_json(out / "plan.json", rec)
    n = sum(info[u][3]["n"] for u in order)
    ng = [sum(info[u][3]["n_g"][g] for u in order) for g in range(4)]
    under = [[sum(info[u][3]["under_g"][g][k] for u in order) / max(ng[g], 1) for k in range(len(S_GRID))] for g in range(4)]
    event = [sum(info[u][3]["event"][k] for u in order) / max(n, 1) for k in range(len(S_GRID))]
    atomic_json(out / "priors.json", {"users": len(order), "source": "the plan's training users (none in the 732 pool), "
                                      "whole histories, time-labelled reviews (SM117 rule)", "S": list(S_GRID),
                                      "pseudo_count": PRIOR_M, "reviews": n, "reviews_per_grade": ng,
                                      "F0_under_S_given_grade": [[round(x, 6) for x in row] for row in under],
                                      "E0_event": [round(x, 6) for x in event]})
    print(json.dumps({k: v for k, v in rec.items() if k != "order"}))
    print(f"[sm117-plan] priors: event rate {[round(x, 3) for x in event]} at S {list(S_GRID)}")
    return 0


# ------------------------------------------------------------------ SM82 time baselines on the SM117 rows (CPU)
def _baseline_user(args):
    data, user = args
    env = label_db()
    with env.begin() as txn:
        ths, frame, lower, cens, cap = test_reviews(data, user, txn, same_day=True)
    env.close()
    last, med = baseline_points(frame)
    d = frame["duration"].to_numpy(np.int64)
    grade = frame["rating"].to_numpy(np.int64) - 1
    labelled = np.flatnonzero(lower >= 0)
    is_test = np.zeros(len(frame), bool)
    if ths is not None:
        is_test[ths - 1] = True
    fit = labelled[~is_test[labelled] & (d[labelled] < cap)]
    logd = np.log(np.maximum(d, 1) / 1000.0)
    res = {"user": user, "cap_ms": cap}
    for name, mu in (("card_last", last), ("user_median", med)):
        res[f"fit_{name}"] = (grade[fit].astype(np.int8), (logd[fit] - mu[fit]).astype(np.float32))
    t = (ths - 1) if ths is not None else np.zeros(0, np.int64)
    later = frame["elapsed_days"].to_numpy(np.int64)[t] >= 1
    res["test"] = {"review_th": ths if ths is not None else np.zeros(0, np.int64), "grade": grade[t].astype(np.int8),
                   "lower": lower[t].astype(np.int8), "cens": cens[t].astype(np.int8), "logd": logd[t].astype(np.float32),
                   "uncensored": (d[t] < CAP_MS) & (cens[t] == 0), "card_last": last[t].astype(np.float32),
                   "user_median": med[t].astype(np.float32), "later": later}
    return res


def baselines(args):
    """scripts/sm82_time.py baselines on the SM117 rows (same-day reviews kept): per user the card's last time and the
    user's running median per grade, with pooled per-grade residual CDFs fitted on the non-test labelled reviews."""
    from multiprocessing import Pool
    from scripts import compress_common as cc
    data = cc.data_path()
    users = json.loads(U732.read_text())["users"]
    started = time.time()
    with Pool(args.workers) as pool:
        rows = pool.map(_baseline_user, [(data, u) for u in users], chunksize=2)
    cdfs = {}
    for name in ("card_last", "user_median"):
        g = np.concatenate([r[f"fit_{name}"][0] for r in rows]); x = np.concatenate([r[f"fit_{name}"][1] for r in rows])
        cdfs[name] = {k: residual_cdf(x[g == k]) for k in range(4)}
    out = []
    for r in rows:
        t = r["test"]
        rec = {"user_id": r["user"], "cap_ms": r["cap_ms"], "reviews": int(len(t["review_th"])),
               "later_day_reviews": int(t["later"].sum()),
               "review_th_sha256": hashlib.sha256(np.asarray(t["review_th"], np.int64).tobytes()).hexdigest()}
        for name in ("card_last", "user_median"):
            prob = np.zeros((len(t["grade"]), BINS))
            for k in range(4):
                m = t["grade"] == k
                if m.any():
                    prob[m] = baseline_prob(t[name][m].astype(np.float64), cdfs[name][k])
            tt = {k: v for k, v in t.items() if k in ("grade", "lower", "cens", "logd", "uncensored")}
            rec[name] = user_sums(prob, tt) if len(prob) else {"n": 0}
            rec[f"{name}_later"] = user_sums(prob, tt, mask=t["later"]) if len(prob) else {"n": 0}
        out.append(rec)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    np.savez_compressed(path.with_suffix(".cdf.npz"), grid=GRID, **{f"{n}_g{k + 1}": cdfs[n][k] for n in cdfs for k in range(4)})
    n = sum(r["reviews"] for r in out)
    nl = sum(r["later_day_reviews"] for r in out)
    summary = {name: {"ll": sum(r[name]["ll"] for r in out if r["reviews"]) / n,
                      "mae_ln_s": sum(r[name]["ae"] for r in out if r["reviews"]) / sum(r[name]["n_unc"] for r in out if r["reviews"])}
               for name in ("card_last", "user_median")}
    print(f"[sm117-baselines] {len(out)} users, {n:,} test reviews with a time label ({nl:,} on a later day than the card's "
          f"previous review), {time.time() - started:.0f} s: " + json.dumps({k: {m: round(v, 4) for m, v in s.items()} for k, s in summary.items()}))
    return 0


# ------------------------------------------------------------------ training
class TrainBatches:
    """The show-time scorer's batch cut to the first `cap` reviews + the revlog's grade and time labels (SM117 rule)
    + the explicit feature table of the same prefix."""

    def __init__(self, users, data, cap):
        self.users, self.data, self.cap = users, data, cap

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        started = time.perf_counter()
        try:
            batch, full = show_time_batch(self.data, user, self.cap)
            rev = revlog(self.data, user)
            lower, cens, _ = time_labels(rev["duration"].to_numpy(), rev["elapsed_days"].to_numpy(), same_day=True)
            labels = {"rating": np.array(rev["rating"].to_numpy(np.int64)), "lower": np.array(lower), "cens": np.array(cens),
                      "same_day": np.array(rev["elapsed_days"].to_numpy(np.int64) == 0), "full": full,
                      "explicit": explicit_table(rev.iloc[:self.cap])}
        except Exception as error:  # recorded, not fatal
            return user, None, None, repr(error), time.perf_counter() - started
        return user, batch, labels, None, time.perf_counter() - started


def vars_config(args):
    return {k: getattr(args, k) for k in ("model", "batch", "lr", "warmup", "buffer_rows", "cap", "seed", "max_users",
                                          "max_user_reviews", "smoke_longest")}


def train(args):
    from scripts import small_model_heldout as smh  # noqa: F401  first: RWKV_NO_JIT + the fork state-quant patch
    import torch
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("HELDOUT_GPU_MEMORY_FRACTION", "0.8")))
    else:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    info = verified_master(args.model)
    released = torch.load(info["master"], map_location="cpu", weights_only=True)
    model = build_model(args.model, released, device)
    frozen = Frozen(model, MODELS[args.model]["kind"])
    torch.manual_seed(args.seed)
    main = make_time_mlp(frozen.inputs, HIDDEN, 4 * BINS).to(device)
    ablation = make_time_mlp(EXPLICIT_DIM, HIDDEN, 4 * BINS).to(device)
    params = list(main.parameters()) + list(ablation.parameters())
    trainable_model = [n for n, p in model.named_parameters() if p.requires_grad]
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
    stats = {"users": 0, "skipped": [], "rows": 0, "same_day_rows": 0, "rows_planned_mismatch": [], "duration_input_nonzero": 0,
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
            buf = [tuple(ck["leftover"])]   # the buffer lives on the CPU
        print(f"[sm117-train] resumed at user {done}/{len(order)}, step {step}/{total_steps}", flush=True)
    log = {"sum": 0.0, "abl": 0.0, "rows": 0, "last": step}
    fd = frozen.frozen_dim

    def run_pass(final):
        nonlocal step, buf, passes
        if not buf:
            return
        def join(j):   # the CPU pieces copied into one device tensor (no second full copy in RAM or on the GPU)
            out = torch.empty((sum(len(b[j]) for b in buf),) + tuple(buf[0][j].shape[1:]), dtype=buf[0][j].dtype, device=device)
            at = 0
            for b in buf:
                out[at:at + len(b[j])].copy_(b[j]); at += len(b[j])
            return out
        x, g, lo, ce = (join(j) for j in range(4))
        buf = []
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
            loss_a = time_loss_rows(ablation(xb[:, fd:]), g[idx], lo[idx], ce[idx]).mean()
            optimizer.zero_grad(set_to_none=True)
            (loss_m + loss_a).backward()   # disjoint parameters: each head gets its own loss's gradient
            optimizer.step()
            vm, va = float(loss_m.detach()), float(loss_a.detach())
            if stats["first_loss"] is None:
                stats["first_loss"], stats["first_loss_ablation"] = vm, va
                print(f"[sm117-train] step 1: time loss {vm:.4f} / ablation {va:.4f} nats over {len(idx):,} rows "
                      f"(zero output layers: ln 16 = 2.7726)", flush=True)
            step += 1
            n = len(idx)
            log["sum"] += vm * n; log["abl"] += va * n; log["rows"] += n
            ps += vm * n; pa += va * n; pr += n
            if step - log["last"] >= args.log_every:
                el = time.perf_counter() - started
                print(f"[sm117-train] step {step}/{total_steps}: time loss {log['sum'] / max(log['rows'], 1):.4f} / ablation "
                      f"{log['abl'] / max(log['rows'], 1):.4f} nats (last {step - log['last']} steps), lr "
                      f"{optimizer.param_groups[0]['lr']:.2e}, users {done}/{len(order)}, {stats['reviews_forward']:,} reviews, "
                      f"{el / 60:.1f} min this run", flush=True)
                log.update(sum=0.0, abl=0.0, rows=0, last=step)
        stats["loss_sum"] += ps; stats["loss_sum_ablation"] += pa; stats["loss_rows"] += pr
        stats["pass_losses"].append(round(ps / max(pr, 1), 5)); stats["pass_losses_ablation"].append(round(pa / max(pr, 1), 5))
        rest = perm[stop:]
        buf = [tuple(t[rest].cpu() for t in (x, g, lo, ce))] if len(rest) else []
        passes += 1
        del x, g, lo, ce
        if torch.cuda.is_available():
            torch.cuda.empty_cache()   # the pass's copy must not fragment the next users' forward

    def save_resume():
        left = tuple(t.cpu() for t in buf[0]) if buf else None
        tmp = resume.with_suffix(".tmp")
        torch.save({"plan_sha256": sha256(args.plan), "config": vars_config(args), "main": main.state_dict(),
                    "ablation": ablation.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "done": done,
                    "passes": passes, "stats": stats, "generator": gen.get_state(), "leftover": left}, tmp)
        os.replace(tmp, resume)

    from scripts import compress_common as cc
    started = time.perf_counter()
    users = [r["user_id"] for r in order]
    planned = {r["user_id"]: r["labelled"] for r in order}
    loader = torch.utils.data.DataLoader(TrainBatches(users[done:], cc.data_path(), args.cap), batch_size=1,
                                         num_workers=args.workers, collate_fn=first,
                                         prefetch_factor=1 if args.workers else None)
    print(f"[sm117-train] {MODELS[args.model]['name']}: {len(order)} users ({len(order) - done} to go), {total_rows:,} "
          f"labelled rows planned, {total_steps:,} steps of {args.batch:,}; inputs {frozen.inputs} ({frozen.snaps_n} x "
          f"{frozen.d} + head_p {frozen.hp_dim} + {EXPLICIT_DIM}); model parameters trainable {trainable_model}, in the "
          f"optimizer {in_optimizer_model}; head {sum(p.numel() for p in main.parameters()):,} + ablation "
          f"{sum(p.numel() for p in ablation.parameters()):,} values; device {device}", flush=True)
    for user, batch, labels, error, prep in loader:
        if args.stop_after_passes and passes >= args.stop_after_passes:
            break
        if batch is None:
            stats["skipped"].append([user, error[:200]])
            done += 1
            continue
        oom = None
        for attempt in range(2):   # an OOM (also the TorchScript RuntimeError form of the big models) frees the cache and retries once
            try:
                gb = batch.to(device)
                with torch.no_grad():
                    qi, th, feats, _ = frozen.forward(gb)
                del _   # the model output of every row (not needed here)
                oom = None
                break
            except (torch.OutOfMemoryError, RuntimeError) as err:
                if not isinstance(err, torch.OutOfMemoryError) and "out of memory" not in str(err):
                    raise
                oom, gb = err, None
                frozen.clear()
                torch.cuda.empty_cache()
                print(f"[sm117-train] user {user}: out of memory (attempt {attempt + 1} of 2), cache freed", flush=True)
        if oom is not None:
            stats["skipped"].append([user, f"OOM: {oom}"[:200]])
            done += 1
            continue
        try:
            lower = torch.as_tensor(labels["lower"], device=device)
            cens = torch.as_tensor(labels["cens"], device=device)
            rating = torch.as_tensor(labels["rating"], device=device)
            sday = torch.as_tensor(labels["same_day"], device=device)
            ex = torch.as_tensor(labels["explicit"], device=device)
            keep = (th >= 1) & (th <= len(labels["explicit"]))
            keep &= lower[(th - 1).clamp(min=0)] >= 0
            qi, th, feats = qi[keep], th[keep], feats[keep]
            stats["duplicate_review_th"] += int(len(th) - len(torch.unique(th)))
            stats["duration_input_nonzero"] += int((gb.start.reshape(-1, gb.start.shape[-1])[qi, 8] != 0).sum())
            stats["same_day_rows"] += int(sday[th - 1].sum())
            grade = rating[th - 1] - 1
            stats["grade_out_of_range"] += int(((grade < 0) | (grade > 3)).sum())
            x = torch.cat([feats, ex[th - 1]], -1)
            stats["nonfinite_features"] += int((~torch.isfinite(x)).any(-1).sum())
            rows = tuple(t.cpu() for t in (x, grade.clamp(0, 3), lower[th - 1], cens[th - 1]))   # buffer on the CPU (big: ~3.4 GB)
            del gb, feats, x, keep, lower, cens, rating, sday, ex   # nothing of this user may stay on the GPU during the
            qi, th, grade = qi.cpu(), th.cpu(), grade.cpu()       # next user's forward (big: x is up to ~2 GB)
        except torch.OutOfMemoryError as oom:
            frozen.clear()
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
    trunk = model.state_dict()
    trunk_same = set(trunk) == set(released) and all(torch.equal(v.detach().cpu(), released[k].float()) for k, v in trunk.items())
    result = {"status": "complete" if finished else "stopped", "model": args.model, "model_name": MODELS[args.model]["name"],
              "users_done": done, "users": len(order), "step": step, "steps_planned": total_steps, "rows_planned": total_rows,
              "passes": passes, "batch": args.batch, "lr": args.lr, "warmup": args.warmup, "buffer_rows": args.buffer_rows,
              "cap": args.cap, "seed": args.seed, "inputs_kind": INPUTS,
              "optimizer": "AdamW (betas 0.9/0.999, eps 1e-8, no weight decay), the two heads only",
              "trainable_model_parameters": trainable_model, "model_parameters_in_optimizer": in_optimizer_model,
              "head_values": sum(p.numel() for p in main.parameters()),
              "ablation_values": sum(p.numel() for p in ablation.parameters()),
              "inputs": {"total": frozen.inputs, "frozen": frozen.frozen_dim, "snapshots": frozen.snaps_n, "d_model": frozen.d,
                         "head_p": frozen.hp_dim, "explicit": EXPLICIT_DIM, "explicit_names": EXPLICIT_NAMES},
              "trunk_unchanged_in_memory": trunk_same,
              "mean_time_loss": round(stats["loss_sum"] / max(stats["loss_rows"], 1), 5),
              "mean_time_loss_ablation": round(stats["loss_sum_ablation"] / max(stats["loss_rows"], 1), 5),
              "seconds_this_run": round(time.perf_counter() - started, 1), "device": str(device),
              "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None,
              "released_master": str(info["master"]), "released_master_sha256": info["sha256"],
              **{k: v for k, v in stats.items() if k not in ("loss_sum", "loss_sum_ablation", "loss_rows")}}
    if finished:
        torch.save({k: v.detach().cpu().clone() for k, v in trunk.items()}, out / "master.pth")   # the frozen trunk
        torch.save({k: v.detach().cpu().float().clone() for k, v in main.state_dict().items()}, out / "head.pth")
        torch.save({k: v.detach().cpu().float().clone() for k, v in ablation.state_dict().items()}, out / "ablation-head.pth")
        result.update(master=str(out / "master.pth"), master_sha256=sha256(out / "master.pth"), head=str(out / "head.pth"),
                      head_sha256=sha256(out / "head.pth"), ablation=str(out / "ablation-head.pth"))
    atomic_json(out / "train.json", result)
    print(f"[sm117-train] {result['status']}: {done}/{len(order)} users, {step:,} steps, {stats['rows']:,} rows "
          f"({stats['same_day_rows']:,} same-day), mean time loss {result['mean_time_loss']:.4f} / ablation "
          f"{result['mean_time_loss_ablation']:.4f}, skipped {len(stats['skipped'])}, trunk unchanged {trunk_same}, "
          f"{result['seconds_this_run'] / 60:.1f} min -> {out}", flush=True)
    ok = trunk_same and stats["duration_input_nonzero"] == 0 and stats["nonfinite_features"] == 0 and in_optimizer_model == 0
    return 0 if ok else 1


# ------------------------------------------------------------------ R-under-S arithmetic
LOG_LO = np.r_[math.log(0.25), LOG_EDGES]          # bin k's lower ln-second edge (bin 0 from 0.25 s, as median_log_s)
LOG_HI = np.r_[LOG_EDGES, np.inf]                  # bin 15 (60 s or more) has no upper edge


def cdf_at(prob, s):
    """P(t <= s seconds) of [..., 16] bin distributions: the bins below the one holding s, plus that bin's share,
    linear in ln time inside it (ln s - lower edge) / (upper - lower edge); s is never an edge (S grid) and < 60 s."""
    b = int(np.searchsorted(np.asarray(BIN_EDGES_MS), 1000.0 * s, side="right"))
    if b >= BINS - 1:
        raise ValueError("s must be below 60 s")
    frac = (math.log(s) - LOG_LO[b]) / (LOG_HI[b] - LOG_LO[b])
    return prob[..., :b].sum(-1) + frac * prob[..., b]


def event_prob(p_grade, p_time, s):
    """P(recalled and t <= s) = sum over Hard, Good, Easy of P(grade) P(t <= s | grade); p_grade [n, 4],
    p_time [n, 4, 16]."""
    return (p_grade[:, 1:] * cdf_at(p_time[:, 1:, :], s)).sum(-1)


def running_baselines(rev, priors):
    """Causal per-review baselines, from the user's earlier time-labelled reviews (SM117 rule) only:
    F[i, g, k] = (# earlier reviews of grade g with t <= S_k + M F0[g][k]) / (# earlier of grade g + M) and
    E[i, k] = (# earlier recalled with t <= S_k + M E0[k]) / (# earlier + M)."""
    d = rev["duration"].to_numpy(np.int64); r = rev["rating"].to_numpy(np.int64); e = rev["elapsed_days"].to_numpy(np.int64)
    lower, _, _ = time_labels(d, e, same_day=True)
    lab = lower >= 0
    m = float(priors["pseudo_count"])
    f0 = np.asarray(priors["F0_under_S_given_grade"], np.float64)
    e0 = np.asarray(priors["E0_event"], np.float64)
    ex = lambda a: np.r_[0, np.cumsum(a)[:-1]].astype(np.float64)   # exclusive running count
    n_all = ex(lab)
    F = np.zeros((len(d), 4, len(S_GRID)))
    E = np.zeros((len(d), len(S_GRID)))
    for g in range(4):
        ng = ex(lab & (r == g + 1))
        for k, s in enumerate(S_GRID):
            F[:, g, k] = (ex(lab & (r == g + 1) & (d <= 1000 * s)) + m * f0[g, k]) / (ng + m)
    for k, s in enumerate(S_GRID):
        E[:, k] = (ex(lab & (r >= 2) & (d <= 1000 * s)) + m * e0[k]) / (n_all + m)
    return F, E


def logit_bin(p):
    z = np.log(p) - np.log1p(-p)
    return np.clip(((z - HIST_LO) / (HIST_HI - HIST_LO) * HIST_BINS).astype(np.int64), 0, HIST_BINS - 1)


def binary_sums(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    return [int(len(p)), float(ll.sum()), float(((p - y) ** 2).sum()), float(p.sum()), float(y.sum())]


def hist_of(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    b = logit_bin(p)
    return (np.bincount(b, minlength=HIST_BINS).astype(np.int32), np.bincount(b, weights=y, minlength=HIST_BINS).astype(np.int32),
            np.bincount(b, weights=p, minlength=HIST_BINS))


def ece_auc_from_hist(n, pos, sp, groups=ECE_BINS):
    """ECE over `groups` equal-mass bins (contiguous fine logit bins, cut where the cumulative count crosses k / groups)
    and AUC (pairs within one fine bin count half) from fine histograms; also the reliability table."""
    n = np.asarray(n, np.float64); pos = np.asarray(pos, np.float64); sp = np.asarray(sp, np.float64)
    tot = n.sum()
    if tot == 0:
        return None
    mid = np.cumsum(n) - n / 2
    gid = np.minimum((mid / tot * groups).astype(np.int64), groups - 1)
    table, ece = [], 0.0
    for k in range(groups):
        m = gid == k
        c = n[m].sum()
        if c == 0:
            continue
        table.append([int(c), round(float(sp[m].sum() / c), 5), round(float(pos[m].sum() / c), 5)])
        ece += abs(sp[m].sum() - pos[m].sum())
    neg = n - pos
    below = np.r_[0.0, np.cumsum(neg)[:-1]]
    P, N = pos.sum(), neg.sum()
    auc = float((pos * (below + 0.5 * neg)).sum() / (P * N)) if P > 0 and N > 0 else float("nan")
    return {"ece": float(ece / tot), "auc": auc, "table": table, "max_gap": max(abs(t[1] - t[2]) for t in table)}


# ------------------------------------------------------------------ scoring
class ScoreBatches:
    """smh.LeakBatches on show-time inputs (the grade scorer's batches) + the revlog and its explicit table."""

    def __init__(self, users, data, counts):
        from scripts import small_model_heldout as smh
        self.inner = smh.LeakBatches(users, data, counts, "neighbour", 0, INPUTS)
        self.data = data

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, index):
        user, batch, error, prep = self.inner[index]
        table = rev = None
        if batch is not None:
            try:
                rev = revlog(self.data, user)
                table = explicit_table(rev)
            except Exception as e:  # recorded, not fatal
                batch, error = None, repr(e)
        return user, batch, error, prep, table, rev


def load_head(path, inputs, prefix=""):
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k[len(prefix):]: v.float() for k, v in sd.items() if k.startswith(prefix)}
    head = make_time_mlp(inputs, HIDDEN, 4 * BINS)
    head.load_state_dict(sd)
    return head.eval()


def score(args):
    """Every time-labelled test review of the users (fp32, full history, show-time inputs, leak-free): per head the
    time sums (SM82 format, all rows and the later-day subset), and per S the R-under-S sums and fine histograms of the
    heads and baselines (a) and (b). One JSON record per user; histograms in <out>.hist/<user>.npz."""
    from scripts import small_model_heldout as smh  # noqa: F401  first
    import torch
    from scripts import compress_common as cc
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("HELDOUT_GPU_MEMORY_FRACTION", "0.8")))
    else:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    info = verified_master(args.model)
    hd = Path(args.head_dir)
    trunk_path = hd / "master.pth"
    trunk = torch.load(trunk_path, map_location="cpu", weights_only=True)
    released = torch.load(info["master"], map_location="cpu", weights_only=True)
    if set(trunk) != set(released) or not all(torch.equal(trunk[k].float(), released[k].float()) for k in released):
        raise SystemExit(f"{trunk_path} is not the released model of {args.model}")
    frozen = Frozen(build_model(args.model, trunk, device), MODELS[args.model]["kind"])
    heads = {"model": load_head(hd / "head.pth", frozen.inputs).to(device),
             "ablation": load_head(hd / "ablation-head.pth", EXPLICIT_DIM).to(device)}
    if args.shipped_head:
        heads["shipped"] = load_head(args.shipped_head, frozen.inputs, prefix="time_mlps.0.").to(device)
    priors = json.loads(Path(args.priors).read_text())
    users = json.loads(Path(args.users_file).read_text())["users"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    hist_dir = out.with_suffix(".hist")
    hist_dir.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["user_id"] for line in out.read_text().splitlines() if '"error"' not in line} if out.exists() else set()
    base = {json.loads(line)["user_id"]: json.loads(line) for line in Path(args.baselines).read_text().splitlines()}
    todo = [u for u in users if u not in done]
    data = cc.data_path()
    loader = torch.utils.data.DataLoader(ScoreBatches(todo, data, cc.run_info()["review_counts"]), batch_size=1,
                                         num_workers=args.workers, collate_fn=first, prefetch_factor=1 if args.workers else None)
    env = label_db()
    started = time.perf_counter()
    preds = list(heads) + ["base_a", "base_b"]
    print(f"[sm117-score] {MODELS[args.model]['name']}: {len(users)} users, {len(todo)} to score, heads {list(heads)}", flush=True)
    f = out.open("a", encoding="utf-8")
    try:
        for k, (user, batch, error, prep, table, rev) in enumerate(loader):
            if batch is None:
                f.write(json.dumps({"user_id": user, "error": error}) + "\n"); f.flush()
                continue
            with env.begin() as txn:
                ths, frame, lower, cens, cap = test_reviews(data, user, txn, same_day=True)
            if ths is None or len(ths) == 0:
                f.write(json.dumps({"user_id": user, "error": "no time-labelled test reviews"}) + "\n"); f.flush()
                continue
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
                    pg = torch.softmax(p_logits[qi[pos]], -1).double().cpu().numpy()
                    pt = {name: torch.softmax(h(x if name != "ablation" else x[:, frozen.frozen_dim:]).float().view(-1, 4, BINS),
                                              -1).double().cpu().numpy() for name, h in heads.items()}
                    nonfinite = int((~torch.isfinite(x)).any(-1).sum())
                del gb, o
            except torch.OutOfMemoryError as oom:
                frozen.clear()
                torch.cuda.empty_cache()
                f.write(json.dumps({"user_id": user, "error": f"OOM: {oom}"[:300]}) + "\n"); f.flush()
                continue
            except RuntimeError as err:   # e.g. a test review without an instant row: recorded, the user is left out
                frozen.clear()            # (a TorchScript OOM of the big models lands here too: recorded, retried by pass 2)
                if "out of memory" in str(err):
                    torch.cuda.empty_cache()
                print(f"[sm117-score] user {user}: {err}", flush=True)
                f.write(json.dumps({"user_id": user, "error": f"{err}"[:300]}) + "\n"); f.flush()
                continue
            d = frame["duration"].to_numpy(np.int64)
            gr = frame["rating"].to_numpy(np.int64)[ths - 1] - 1
            ed = frame["elapsed_days"].to_numpy(np.int64)[ths - 1]
            t = {"grade": gr.astype(np.int8), "lower": lower[ths - 1].astype(np.int8), "cens": cens[ths - 1].astype(np.int8),
                 "logd": np.log(np.maximum(d[ths - 1], 1) / 1000.0), "uncensored": (d[ths - 1] < CAP_MS) & (cens[ths - 1] == 0)}
            later = ed >= 1
            rows_idx = np.arange(len(ths))
            sha = hashlib.sha256(np.asarray(ths, np.int64).tobytes()).hexdigest()
            b = base.get(user, {})
            rec = {"user_id": user, "reviews": int(len(ths)), "later_day_reviews": int(later.sum()), "review_th_sha256": sha,
                   "baseline_same_reviews": b.get("review_th_sha256") == sha, "model": args.model, "trunk": str(trunk_path),
                   "head_dir": str(hd), "precision": "fp32", "inputs": INPUTS, "deleted_cards": "neighbour",
                   "user_cap_ms": int(cap), "prepare_seconds": round(prep, 2), "nonfinite_inputs": nonfinite, "time": {}, "rus": {}}
            for name, p in pt.items():
                cond = p[rows_idx, gr]
                marg = (pg[:, :, None] * p).sum(1)
                rec["time"][name] = {"cond": user_sums(cond, t), "marg": user_sums(marg, t),
                                     "cond_later": user_sums(cond, t, mask=later), "marg_later": user_sums(marg, t, mask=later),
                                     "fast_3_9s": [[int((gr == g).sum()), float(cond[gr == g, :7].sum()),
                                                    int(((gr == g) & (t["lower"] <= 6) & (t["cens"] == 0)).sum())] for g in range(4)]}
            F, E = running_baselines(rev, priors)
            F, E = F[ths - 1], E[ths - 1]
            hists = {pn: [] for pn in preds}
            for kk, s in enumerate(S_GRID):
                valid = np.ones(len(ths), bool) if cap > 1000 * s else np.zeros(len(ths), bool)   # user cap above S
                y = ((gr >= 1) & (d[ths - 1] <= 1000 * s)).astype(np.float64)
                pp = {name: event_prob(pg, p, s) for name, p in pt.items()}
                pp["base_a"] = (pg[:, 1:] * F[:, 1:, kk]).sum(-1)
                pp["base_b"] = E[:, kk]
                for pn in preds:
                    rec["rus"].setdefault(pn, {})[str(s)] = binary_sums(pp[pn][valid], y[valid])
                    hists[pn].append(hist_of(pp[pn][valid], y[valid]))
            np.savez_compressed(hist_dir / f"{user}.npz", preds=np.array(preds), S=np.array(S_GRID),
                                n=np.array([[h[0] for h in hists[pn]] for pn in preds]),
                                pos=np.array([[h[1] for h in hists[pn]] for pn in preds]),
                                sp=np.array([[h[2] for h in hists[pn]] for pn in preds]))
            f.write(json.dumps(rec) + "\n"); f.flush()
            if (k + 1) % 50 == 0:
                print(f"[sm117-score] {k + 1}/{len(todo)} users, {time.perf_counter() - started:.0f} s", flush=True)
    finally:
        f.close()
        env.close()
    print(f"[sm117-score] done {len(todo)} users in {time.perf_counter() - started:.0f} s -> {out}", flush=True)
    return 0


# ------------------------------------------------------------------ grade check
def gradecompare(args):
    info = model_info(args.model)
    users = json.loads(Path(args.users_file).read_text())["users"]
    ex = info["existing"] if Path(info["existing"]).exists() else None
    out = grade_compare(args.rows, args.dump, args.rel_tag, args.new_tag, users, existing=ex,
                        existing_label=info["existing_label"] if ex else None, variant=VARIANT)
    atomic_json(args.out, out)
    print(f"[sm117-grade] {args.model} {len(users)} users: released vs the trunk after head training bitwise {out['pass']}; "
          f"vs the recorded show-time scores {out.get('existing_equal')}")
    return 0 if out["pass"] else 1


# ------------------------------------------------------------------ decision
def pool_users():
    from scripts import sm77_rule
    users732 = set(json.loads(U732.read_text())["users"])
    big = sm77_rule.load(*sm77_rule.BIG, users732)
    return sorted(users732 & set(big))


def load_rows(path):
    p = Path(path)
    return {r["user_id"]: r for r in map(json.loads, p.read_text().splitlines()) if "error" not in r} if p.exists() else {}


def bootstrap_idx(n):
    return np.random.default_rng(0).integers(0, n, size=(2000, n))


def _diff(a, b, w, idx):
    delta = float((a - b).sum() / w.sum())
    boot = ((a - b)[idx]).sum(1) / w[idx].sum(1)
    return {"delta": round(delta, 6), "ci": [round(float(np.percentile(boot, q)), 6) for q in (2.5, 97.5)],
            "relative_pct": round(100 * delta / float(b.sum() / w.sum()), 2)}


def evaluate(key, rows_path, grade_check, train_json, hist_dir=None):
    """Per-model numbers and the pre-registered gate + reliability verdicts (design.md)."""
    pool = pool_users()
    rows = load_rows(rows_path)
    base = {r["user_id"]: r for r in map(json.loads, Path(args_baselines()).read_text().splitlines())}
    missing = [u for u in pool if u not in rows]
    users = [u for u in pool if u in rows and rows[u]["reviews"] > 0]
    mis = [u for u in users if not rows[u]["baseline_same_reviews"]]
    res = {"model": key, "name": MODELS[key]["name"], "pool": len(pool), "missing": missing, "users": len(users),
           "refused": bool(mis), "baseline_mismatch": mis[:20]}
    if mis or not users:
        return res
    idx = bootstrap_idx(len(users))
    heads = [h for h in ("model", "ablation", "shipped") if h in rows[users[0]]["time"]]
    tm = {}
    for sub in ("", "_later"):
        n = np.array([rows[u]["time"]["model"][f"cond{sub}"]["n"] for u in users], float)
        nu = np.array([rows[u]["time"]["model"][f"cond{sub}"]["n_unc"] for u in users], float)
        ll = {h: np.array([rows[u]["time"][h][f"cond{sub}"]["ll"] for u in users]) for h in heads}
        ae = {h: np.array([rows[u]["time"][h][f"cond{sub}"]["ae"] for u in users]) for h in heads}
        llm = {h: np.array([rows[u]["time"][h][f"marg{sub}"]["ll"] for u in users]) for h in heads}
        aem = {h: np.array([rows[u]["time"][h][f"marg{sub}"]["ae"] for u in users]) for h in heads}
        for bn in ("card_last", "user_median"):
            ll[bn] = np.array([base[u][f"{bn}{sub}"]["ll"] for u in users]); ae[bn] = np.array([base[u][f"{bn}{sub}"]["ae"] for u in users])
        rec = {"reviews": int(n.sum()), "uncensored": int(nu.sum()),
               "time_logloss": {k: round(float(v.sum() / n.sum()), 5) for k, v in ll.items()},
               "mae_ln_s": {k: round(float(v.sum() / nu.sum()), 5) for k, v in ae.items()},
               "time_logloss_grade_mixed": {k: round(float(v.sum() / n.sum()), 5) for k, v in llm.items()},
               "mae_ln_s_grade_mixed": {k: round(float(v.sum() / nu.sum()), 5) for k, v in aem.items()}, "vs": {}}
        for bn in ("card_last", "user_median"):
            for h in heads:
                rec["vs"][f"{h}_vs_{bn}"] = _paired(ll[h], ll[bn], ae[h], ae[bn], n, nu, idx)
        if "ablation" in heads:
            rec["vs"]["model_vs_ablation"] = _paired(ll["model"], ll["ablation"], ae["model"], ae["ablation"], n, nu, idx)
        if "shipped" in heads:
            rec["vs"]["model_vs_shipped"] = _paired(ll["model"], ll["shipped"], ae["model"], ae["shipped"], n, nu, idx)
        tm["all" if not sub else "later_day"] = rec
    beats = all(tm["all"]["vs"][f"model_vs_{bn}"][m]["delta"] < 0 and tm["all"]["vs"][f"model_vs_{bn}"][m]["ci"][1] < 0
                for bn in ("card_last", "user_median") for m in ("time_logloss", "mae_ln_s"))
    gc = json.loads(Path(grade_check).read_text()) if grade_check and Path(grade_check).exists() else {"pass": False}
    tj = json.loads(Path(train_json).read_text()) if train_json and Path(train_json).exists() else {}
    res.update(time=tm, grade_check={"pass": bool(gc.get("pass")), "existing_equal": gc.get("existing_equal")},
               train={k: tj.get(k) for k in ("status", "users_done", "step", "rows", "same_day_rows", "skipped", "seconds_this_run",
                                              "peak_gpu_gib", "head_values", "inputs", "mean_time_loss", "trunk_unchanged_in_memory")},
               gate={"grade_unchanged": bool(gc.get("pass")), "beats_both_time_baselines": bool(beats),
                     "pass": bool(gc.get("pass")) and bool(beats)})
    # R-under-S
    preds = [p for p in ("model", "ablation", "shipped", "base_a", "base_b") if p in rows[users[0]]["rus"]]
    hd = Path(hist_dir) if hist_dir else Path(rows_path).with_suffix(".hist")
    H = {}
    for u in users:
        z = np.load(hd / f"{u}.npz")
        names = list(z["preds"])
        for p in preds:
            i = names.index(p)
            if p not in H:
                H[p] = [np.zeros((len(S_GRID), HIST_BINS)) for _ in range(3)]
            for j, arr in enumerate(("n", "pos", "sp")):
                H[p][j] += z[arr][i]
    rus = {}
    reliable_all = True
    for kk, s in enumerate(S_GRID):
        S = str(s)
        arr = {p: np.array([rows[u]["rus"][p][S] for u in users], float) for p in preds}   # [users, 5]
        n = arr["model"][:, 0]
        rec = {"rows": int(n.sum()), "users_with_rows": int((n > 0).sum()), "event_rate": round(float(arr["model"][:, 4].sum() / n.sum()), 5),
               "predictors": {}, "vs": {}}
        for p in preds:
            st = ece_auc_from_hist(H[p][0][kk], H[p][1][kk], H[p][2][kk])
            rec["predictors"][p] = {"logloss": round(float(arr[p][:, 1].sum() / n.sum()), 6),
                                    "brier": round(float(arr[p][:, 2].sum() / n.sum()), 6),
                                    "mean_p": round(float(arr[p][:, 3].sum() / n.sum()), 5),
                                    "ece": round(st["ece"], 5), "auc": round(st["auc"], 5), "max_bin_gap": round(st["max_gap"], 5),
                                    "reliability": st["table"]}
        w = np.maximum(n, 0)
        keep = w > 0
        sub_idx = bootstrap_idx(int(keep.sum()))
        for p in [q for q in preds if q != "model"]:
            rec["vs"][f"model_vs_{p}"] = {m: _diff(arr["model"][keep, c], arr[p][keep, c], w[keep], sub_idx)
                                          for m, c in (("logloss", 1), ("brier", 2))}
        ok = (rec["predictors"]["model"]["ece"] < 0.01
              and all(rec["vs"][f"model_vs_{b}"]["logloss"]["ci"][1] < 0 for b in ("base_a", "base_b")))
        rec["reliable"] = bool(ok)
        reliable_all &= ok
        rus[S] = rec
    res.update(rus=rus, reliable=bool(reliable_all))
    return res


def args_baselines():
    return str(HERE / "time-baselines.jsonl")


def decide(args):
    res = evaluate(args.model, args.rows, args.grade_check, args.train_json)
    atomic_json(HERE / f"{args.model}-gate.json", res)
    if res.get("refused"):
        print(f"[sm117-rule] {args.model}: REFUSE ({len(res['baseline_mismatch'])} users with other reviews than the baselines)")
        return 1
    if "gate" not in res:
        print(f"[sm117-rule] {args.model}: no users")
        return 2
    t = res["time"]["all"]
    s10 = res["rus"]["10"]["predictors"]["model"]
    print(f"[sm117-rule] {args.model}: {res['users']} users ({len(res['missing'])} missing): grade unchanged "
          f"{res['gate']['grade_unchanged']}; time LogLoss {t['time_logloss']['model']:.4f} vs card's last "
          f"{t['time_logloss']['card_last']:.4f} / user median {t['time_logloss']['user_median']:.4f}; R under 10 s LogLoss "
          f"{s10['logloss']:.4f} ECE {s10['ece']:.4f}; reliable at every S {res['reliable']}: "
          f"{'PASS' if res['gate']['pass'] else 'FAIL'}")
    return 0 if res["gate"]["pass"] else 3


def fmt(x, nd=4):
    return "" if x is None else f"{x:.{nd}f}"


def ci(d, nd=4, pct=False):
    return f"{d['delta']:+.{nd}f} [{d['ci'][0]:+.{nd}f}, {d['ci'][1]:+.{nd}f}]" + (f" ({d['relative_pct']:+.1f}%)" if pct else "")


def report(args):
    """All six: artifacts/small-model/sm117/sm117-result.{md,json} from the <key>-gate.json files."""
    res = {k: json.loads((HERE / f"{k}-gate.json").read_text()) for k in ORDER if (HERE / f"{k}-gate.json").exists()}
    L = ["# Time heads for the family (SM117): result", "",
         "An answer-time head (the answer-time head on richer frozen features (SM82c) recipe) trained alone on each frozen "
         "family model, on show-time inputs with same-day reviews included, then scored on the 732 set (722 users), "
         "show-time inputs, leak-free. Pre-registered rules: `design.md`.", "",
         "## Gate per head (grade unchanged bitwise on 5 users; the head beats both answer-time baselines, CI clear of zero)", "",
         "| model | users | grade unchanged | time LogLoss (head / ablation / card's last / user median) | head vs user median | "
         "mean abs ln error (head / user median) | gate |", "|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        if "gate" not in r:
            L.append(f"| {MODELS[k]['name']} | {r.get('users')} | not scored | | | | |"); continue
        t = r["time"]["all"]
        L.append(f"| {r['name']} | {r['users']} | {r['gate']['grade_unchanged']} | {fmt(t['time_logloss']['model'])} / "
                 f"{fmt(t['time_logloss'].get('ablation'))} / {fmt(t['time_logloss']['card_last'])} / {fmt(t['time_logloss']['user_median'])} | "
                 f"{ci(t['vs']['model_vs_user_median']['time_logloss'], pct=True)} | {fmt(t['mae_ln_s']['model'])} / "
                 f"{fmt(t['mae_ln_s']['user_median'])} | {'pass' if r['gate']['pass'] else 'FAIL'} |")
    L += ["", "Same numbers on the later-day subset (the rows of the earlier answer-time heads, SM82 rule), for comparison "
          "with the shipped head of the show-time retrain (SM92) (gated at 1.9580 on normal inputs):", "",
          "| model | rows | head | card's last | user median | shipped head |", "|---|---|---|---|---|---|"]
    for k, r in res.items():
        if "gate" in r:
            t = r["time"]["later_day"]
            L.append(f"| {r['name']} | {t['reviews']:,} | {fmt(t['time_logloss']['model'])} | {fmt(t['time_logloss']['card_last'])} | "
                     f"{fmt(t['time_logloss']['user_median'])} | {fmt(t['time_logloss'].get('shipped'))} |")
    L += ["", "## P(recall within S seconds)", "",
          "p = sum over Hard, Good, Easy of P(grade) x P(t <= S | grade); label: grade >= Hard and recorded time <= S. "
          "Baseline (a): the model's own P(grade) x the user's running share of earlier answers of that grade under S; "
          "baseline (b): the user's running rate of the event. LogLoss deltas are head minus baseline, review-weighted, "
          "95% CI from 2,000 user bootstraps (seed 0).", ""]
    for k, r in res.items():
        if "rus" not in r:
            continue
        L += [f"### {r['name']}", "", "| S | rows | event rate | LogLoss head | (a) | (b) | head vs (a) | head vs (b) | Brier head | "
              "ECE head | ECE (a) | AUC head | AUC (a) | AUC (b) | reliable |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for S, x in r["rus"].items():
            p = x["predictors"]
            L.append(f"| {S} s | {x['rows']:,} | {x['event_rate']:.3f} | {p['model']['logloss']:.4f} | {p['base_a']['logloss']:.4f} | "
                     f"{p['base_b']['logloss']:.4f} | {ci(x['vs']['model_vs_base_a']['logloss'], pct=True)} | "
                     f"{ci(x['vs']['model_vs_base_b']['logloss'], pct=True)} | {p['model']['brier']:.4f} | {p['model']['ece']:.4f} | "
                     f"{p['base_a']['ece']:.4f} | {p['model']['auc']:.4f} | {p['base_a']['auc']:.4f} | {p['base_b']['auc']:.4f} | "
                     f"{'yes' if x['reliable'] else 'no'} |")
        L += ["", f"Reliable at every S (ECE < 0.01 and LogLoss below both baselines, CI clear of zero): **{'yes' if r['reliable'] else 'no'}**.", ""]
    L += ["## Reliability tables (head, 20 equal-mass bins: rows, mean predicted, observed)", "",
          "In `sm117-result.json` (`<model>.rus.<S>.predictors.model.reliability`); the largest bin gap per S is below.", "",
          "| model | " + " | ".join(f"{s} s" for s in S_GRID) + " |", "|---|" + "---|" * len(S_GRID)]
    for k, r in res.items():
        if "rus" in r:
            L.append(f"| {r['name']} | " + " | ".join(f"{r['rus'][str(s)]['predictors']['model']['max_bin_gap']:.3f}" for s in S_GRID) + " |")
    (HERE / "sm117-result.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    atomic_json(HERE / "sm117-result.json", res)
    print(f"[sm117-report] {len(res)} models: gate {[(k, r.get('gate', {}).get('pass')) for k, r in res.items()]}; reliable "
          f"{[(k, r.get('reliable')) for k, r in res.items()]} -> {HERE / 'sm117-result.md'}")
    return 0


# ------------------------------------------------------------------ CPU proof
def check(args):
    from scripts.sm117_check import run_check
    return run_check(args)


def info(args):
    """Shell assignments for the launcher (eval): the model's master, scorer spec, label, name and buffer size."""
    import shlex
    m = verified_master(args.model)
    master = str(m["master"]).replace("\\", "/")
    vals = {"MASTER": master, "SPEC": spec(args.model, master), "KIND": m["kind"], "MNAME": m["name"], "MLABEL": m["label"],
            "BUFFER": "500000" if m["kind"] == "fork" else "1000000",
            "SHIPPED": str(m.get("shipped_head", "")).replace("\\", "/"),
            "SPECPFX": "fork::" if m["kind"] == "fork" else f"expertium:{m['arch']}:"}
    print("\n".join(f"{k}={shlex.quote(v)}" for k, v in vals.items()))
    return 0


def main():
    ap = ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", default=str(HERE))
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("baselines")
    p.add_argument("--out", default=str(HERE / "time-baselines.jsonl"))
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("train")
    p.add_argument("--model", required=True, choices=ORDER)
    p.add_argument("--plan", default=str(HERE / "plan.json"))
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
    p.add_argument("--model", required=True, choices=ORDER)
    p.add_argument("--head-dir", required=True)
    p.add_argument("--shipped-head", default="")
    p.add_argument("--baselines", default=str(HERE / "time-baselines.jsonl"))
    p.add_argument("--priors", default=str(HERE / "priors.json"))
    p.add_argument("--users-file", default=str(U732))
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=3)
    p = sub.add_parser("check")
    p.add_argument("--out", default=str(HERE / "check-cpu-sm117.json"))
    p.add_argument("--models", default="sm106,sm98,sm91s2")
    p = sub.add_parser("gradecompare")
    p.add_argument("--model", required=True, choices=ORDER)
    p.add_argument("--rows", required=True)
    p.add_argument("--dump", required=True)
    p.add_argument("--rel-tag", required=True)
    p.add_argument("--new-tag", required=True)
    p.add_argument("--users-file", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("decide")
    p.add_argument("--model", required=True, choices=ORDER)
    p.add_argument("--rows", required=True)
    p.add_argument("--grade-check", required=True)
    p.add_argument("--train-json", required=True)
    sub.add_parser("report")
    p = sub.add_parser("info")
    p.add_argument("--model", required=True, choices=ORDER)
    args = ap.parse_args()
    return {"plan": plan, "baselines": baselines, "train": train, "score": score, "check": check,
            "gradecompare": gradecompare, "decide": decide, "report": report, "info": info}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
