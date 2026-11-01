"""Curve consistency on real rapid Good pairs (ST01a), stage 1: measure the far-tail jumps of the curve head.

The curve (ahead) head gives, after every answer, a recall curve p(t) over the time since that answer: the big
model p(t) = sigmoid(logit(sum_c w_c 0.9^(t / s_c)) + interp(ahead_logits, t)) (rwkv/model/srs_model.py), the small
model (w64) the 3-term power law sum_i w_i (1 + t / S_i)^(-d_i) with w, S, d per answer
(expertium/model/variant_model.py gru_forgetting_curve). It is trained only at the card's next real review, so the far tail is
never directly supervised. For each pair of consecutive reviews a -> b of the same card with b answered Good
(rating 3) after a gap of 0-7 days, this script evaluates both curves at the same absolute future dates, H days after
b: curve_a(gap + H) and curve_b(H), for H in HORIZONS. Where the card has a real next review c after b (elapsed e),
it also stores curve_b(e) (the model's real prediction) and curve_a(gap + e) (the curve before b's answer, at the same
date) with c's outcome: which curve was closer.

Protocol = the scorer's (scripts/small_model_heldout.py) for one variant, w8+s8, full history as one sequence,
--deleted-cards neighbour (scripts/sm71_leak.py), id/day codes seed 2025 + user. One forward pass per user
(model.forward_batch, the same call get_loss makes); the curve is evaluated with the model's own forgetting_curve and
interp methods. Check per user: the curve LogLoss over every labelled review row recomputed from the head outputs
(field curve_log_loss) against the scorer row (scripts compare it afterwards).

Output: --out JSONL (one row per user: reviews, pairs, curve_log_loss, seconds; resumable) and --pairs DIR/<user>.npz.

usage: st01a_pairs.py --model TAG=KIND:ARCH:PATH --users-file F --out OUT.jsonl --pairs DIR [--device cpu --count 2]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RWKV_NO_JIT", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts import small_model_heldout as smh  # noqa: E402  (installs the state-quant model patches)
from scripts import compress_quant as cq  # noqa: E402
from scripts import compress_common as cc  # noqa: E402

HORIZONS = (7, 30, 90, 365)   # days after b
DAY = 86400.0
MAX_GAP_DAYS = 7


class PairBatches(torch.utils.data.Dataset):
    """Scorer inputs (LeakBatches, neighbour) plus the per-review metadata of the frame."""

    def __init__(self, users, data):
        self.users, self.data = users, data

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        started = time.perf_counter()
        try:
            from rwkv.data_processing import get_rwkv_data
            from rwkv.prepare_batch import prepare
            from scripts import sm71_leak
            frame = get_rwkv_data(self.data, user)
            frame, _ = sm71_leak.neutralise(frame, "neighbour", self.data, user)
            samples = cc.make_samples(frame, user, len(frame))
            assert len(samples) == 1
            sample = samples[0]
            meta = {
                "pos_review_th": sample.review_ths.numpy().astype(np.int64),
                "pos_is_query": sample.global_labels[:, 6].numpy() > 0.5,
                "review_th": frame["review_th"].to_numpy(np.int64),
                "card": frame["card_id"].to_numpy(np.int64),
                "rating": frame["rating"].to_numpy(np.int64),
                "elapsed_days": frame["elapsed_days"].to_numpy(np.float64),
                "elapsed_seconds": frame["elapsed_seconds"].to_numpy(np.float64),
                "i": frame["i"].to_numpy(np.int64),
                "has_label": frame["has_label"].to_numpy(np.int64),
                "label_elapsed_seconds": frame["label_elapsed_seconds"].to_numpy(np.float64),
                "label_y": frame["label_y"].to_numpy(np.float64),
            }
            batch = prepare(samples, seed=2025 + user)
        except Exception as error:  # recorded, not fatal
            return user, None, None, repr(error), time.perf_counter() - started
        return user, batch, meta, None, time.perf_counter() - started


def first(items):
    return items[0]


def make_model(kind, arch, path, device, cpu):
    """The scorer's w8+s8 setup (small_model_heldout.make_model + score_user) for one model."""
    from scripts import state_vq
    base = {k: v.float() for k, v in torch.load(path, map_location="cpu", weights_only=True).items()}
    layout = None
    if kind == "fork":
        model = cq.load_model(state=base, reference_wkv=cpu)
        for module in model.rwkv_modules:
            for block in module.blocks:
                block.time_mixer.use_reference_wkv = True
    else:
        model = smh.build_expertium(arch, base)
        layout = state_vq.install(model) if cpu else state_vq.install_scoring(model)
    model = model.to(device)
    bits, group = cq.WEIGHT_SCHEMES["w8-pc"]
    model.load_state_dict(cq.quantize_state_dict(base, cq.linear_weight_names(base), bits, group))
    if kind == "fork":
        cq.set_state_bits(model, 8)
    elif cpu:
        smh.set_expertium_state_bits(model, 0)
        state_vq.configure(layout, None, 8)
    else:
        state_vq.set_scoring(layout, None)
        smh.set_expertium_state_bits(model, 8)
    return model.eval()


def forward_heads(model, gpu):
    """Per-position curve-head outputs of the one sequence: (w, logits, s_raw, d_raw); s/d None for the stock head."""
    args = (gpu.start, gpu.sub_gather, gpu.sub_gather_lens, gpu.time_shift_selects, gpu.skips, gpu.num_data)
    if hasattr(model, "forward_batch_variant"):   # expertium variants (w64: the 3-term power-law curve head)
        logits, w, _, _, s_raw, d_raw = model.forward_batch_variant(*args)
        if model.gru_on:
            return w[0], logits[0], s_raw[0], d_raw[0]
        return w[0], logits[0], None, None
    logits, w, _, _ = model.forward_batch(*args)
    return w[0], logits[0], None, None


def take(heads, rows):
    return tuple(None if h is None else h.index_select(0, rows) for h in heads)


def curve_at(model, heads, seconds):
    """p(t) of the curves `heads` (rows of w, logits, s, d) at `seconds` (1-D, same rows), as float64 numpy.
    Same arithmetic as the model's _get_loss / _get_loss_variant."""
    w, logits, s_raw, d_raw = heads
    t = seconds.reshape(-1, 1).to(w.dtype)
    if s_raw is not None:
        raw = model.gru_forgetting_curve(w, s_raw, d_raw, t)
    else:
        raw = model.forgetting_curve(w, t)
    logit = torch.log(raw / (1 - raw)) + model.interp(logits, t)
    return torch.sigmoid(logit).double().cpu().numpy()


def user_pairs(model, batch, meta, device, check=False):
    with torch.inference_mode():
        gpu = batch.to(device)
        heads = forward_heads(model, gpu)
        n_pos = len(meta["pos_review_th"])
        review_pos = np.flatnonzero(~meta["pos_is_query"])
        assert len(review_pos) == len(meta["review_th"])
        assert np.array_equal(meta["pos_review_th"][review_pos], meta["review_th"])
        assert heads[0].shape[0] >= n_pos
        rh = take(heads, torch.as_tensor(review_pos, device=device))   # curve after each review, in review order
        out = {}
        # check: curve LogLoss over every labelled review row (the scorer's curve_log_loss)
        lab = np.flatnonzero(meta["has_label"] > 0)
        li = torch.as_tensor(lab, device=device)
        p_lab = curve_at(model, take(rh, li), torch.as_tensor(meta["label_elapsed_seconds"][lab], device=device))
        y_lab = meta["label_y"][lab]
        pc = np.clip(p_lab, 1e-7, 1 - 1e-7)
        out["curve_log_loss"] = float(-(y_lab * np.log(pc) + (1 - y_lab) * np.log(1 - pc)).mean())
        out["curve_labels"] = int(len(lab))
        if check:
            stats = model.get_loss(gpu)
            hl = stats.has_label.reshape(-1).bool() & ~stats.is_query.reshape(-1).bool()
            ref = stats.p_curve.reshape(-1)[hl].double().cpu().numpy()
            out["check_max_abs_p_curve"] = float(np.abs(ref - p_lab).max())
        # pairs a -> b (consecutive reviews of one card), b Good, gap 0..MAX_GAP_DAYS days
        card, rating = meta["card"], meta["rating"]
        order = np.lexsort((meta["review_th"], card))
        same = card[order][1:] == card[order][:-1]
        a_idx, b_idx = order[:-1][same], order[1:][same]
        gap_days = meta["elapsed_days"][b_idx]
        keep = (rating[b_idx] == 3) & (gap_days >= 0) & (gap_days <= MAX_GAP_DAYS)
        a_idx, b_idx = a_idx[keep], b_idx[keep]
        pairs = {"review_th_a": meta["review_th"][a_idx].astype(np.int32),
                 "review_th_b": meta["review_th"][b_idx].astype(np.int32),
                 "rating_a": rating[a_idx].astype(np.int8),
                 "i_a": meta["i"][a_idx].astype(np.int32),
                 "ivl_a": meta["elapsed_days"][a_idx].astype(np.float32),     # interval before a (-1 = new card)
                 "gap_days": meta["elapsed_days"][b_idx].astype(np.float32),
                 "gap_seconds": meta["elapsed_seconds"][b_idx].astype(np.float64)}
        if len(a_idx):
            ta, tb = torch.as_tensor(a_idx, device=device), torch.as_tensor(b_idx, device=device)
            ha, hb = take(rh, ta), take(rh, tb)
            gap = torch.as_tensor(pairs["gap_seconds"], device=device)
            pairs["p_a_at_b"] = curve_at(model, ha, gap).astype(np.float32)
            for h in HORIZONS:
                pairs[f"pa_{h}"] = curve_at(model, ha, gap + h * DAY).astype(np.float32)
                pairs[f"pb_{h}"] = curve_at(model, hb, torch.full_like(gap, h * DAY)).astype(np.float32)
            has_c = meta["has_label"][b_idx] > 0
            e = torch.as_tensor(np.where(has_c, meta["label_elapsed_seconds"][b_idx], 1.0), device=device)
            pairs["has_c"] = has_c
            pairs["e_c_seconds"] = e.double().cpu().numpy()
            pairs["y_c"] = meta["label_y"][b_idx].astype(np.int8)
            pairs["pc_b"] = curve_at(model, hb, e).astype(np.float32)
            pairs["pc_a"] = curve_at(model, ha, gap + e).astype(np.float32)
        out["pairs"] = int(len(a_idx))
        del gpu
    return out, pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="TAG=KIND:ARCH:PATH (KIND fork or expertium)")
    parser.add_argument("--users-file", required=True)
    parser.add_argument("--pool", type=int, default=732)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pairs", required=True, help="folder for <user>.npz")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--max-reviews", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=3000)
    parser.add_argument("--check", action="store_true", help="also run get_loss and compare p_curve (smoke)")
    args = parser.parse_args()
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    cpu = args.device == "cpu"
    if cpu:
        torch.set_num_threads(2)
        args.workers = 0
    else:
        from scripts import compress_wkv_cuda, small_model_wkv
        compress_wkv_cuda.register_cuda()
        errors = {**{f"fork/bits{b}": e for b, e in compress_wkv_cuda.self_test().items()}, **small_model_wkv.self_test()}
        if max(e for key, e in errors.items() if key.endswith("0")) > 1e-4:
            raise SystemExit("CUDA recurrence disagrees with the reference")
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("HELDOUT_GPU_MEMORY_FRACTION", "0.8")))
    device = torch.device(args.device)
    tag, _, rest = args.model.partition("=")
    kind, arch, path = rest.split(":", 2)
    run = cc.run_info()
    counts = run["review_counts"]
    users = smh.heldout_users(counts, args.pool)
    wanted = set(json.loads(Path(args.users_file).read_text())["users"])
    users = [u for u in users if u in wanted]
    if args.max_reviews:
        users = [u for u in users if counts[str(u)] <= args.max_reviews]
    users = users[args.start:args.start + args.count] if args.count else users[args.start:]
    out, folder = Path(args.out), Path(args.pairs)
    out.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {r["user_id"] for r in map(json.loads, out.read_text().splitlines()) if "error" not in r}
    todo = [u for u in users if u not in done]
    model = make_model(kind, arch, path, device, cpu)
    print(f"[st01a] {tag}: {len(users)} users, {len(todo)} to do", flush=True)
    loader = torch.utils.data.DataLoader(PairBatches(todo, cc.data_path(run)), batch_size=1, num_workers=args.workers,
                                         collate_fn=first, prefetch_factor=1 if args.workers else None)
    started = time.perf_counter()
    for index, (user, batch, meta, error, prep) in enumerate(loader):
        if time.perf_counter() - started > args.time_limit:
            print(f"[st01a] time limit reached after {index} users", flush=True)
            break
        row = {"user_id": user, "tag": tag, "kind": kind, "checkpoint": path, "reviews": counts[str(user)],
               "prepare_seconds": round(prep, 2)}
        if batch is None:
            row["error"] = error
        else:
            t0 = time.perf_counter()
            try:
                values, pairs = user_pairs(model, batch, meta, device, args.check)
                np.savez_compressed(folder / f"{user}.npz", **pairs)
                row.update(values)
            except (torch.OutOfMemoryError, RuntimeError) as oom:
                if not isinstance(oom, torch.OutOfMemoryError) and "out of memory" not in str(oom):
                    raise
                torch.cuda.empty_cache()
                row["error"] = f"OOM: {oom}"[:300]
            row["model_seconds"] = round(time.perf_counter() - t0, 3)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[st01a] {index + 1}/{len(todo)} user {user} ({row['reviews']} reviews, {row.get('pairs')} pairs) | "
              f"{time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
